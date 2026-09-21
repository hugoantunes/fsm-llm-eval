"""T-18: census a ``sim run`` from ``manifest.jobs`` and export scored CSVs.

The manifest is the audit universe. Dialogue logs are execution outcome; metric
files are inclusion. Contract false positives stay failed executions. Export
fails closed on unexplained, duplicate, missing, or set-mismatched jobs.
"""

from __future__ import annotations

import csv
import hashlib
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from sim.adjudication import (
    TARGET_FAILURE_KIND,
    TARGET_FAILURE_REASON,
    AdjudicationError,
    contract_false_positives_path,
    load_failed_inclusion_allowlist,
)
from sim.eval import METRICS_CSV, UNSCORED_CSV
from sim.llm import LLM_CALLS_LOG, LlmCallRecord
from sim.schemas import DialogueLog, JobRef, Manifest, dialogue_filename, dialogue_id

FROZEN_GATE_METRICS_SHA256 = (
    "968bb4ad04aca98ba1d23dcccdfd9aa1fa8db547c2c2850483f09dc937d38534"
)
SIDECAR_METRICS_SHA256 = (
    "29fe9712d8e4eaa29676daa795c9663dc5955ef6237bed59a8e6eb6a880d6e2b"
)

PRIMARY_CSV = METRICS_CSV
FROZEN_EXPORT_CSV = "metrics_frozen_gate.csv"
OUTLIERS_CSV = "latency_outliers.csv"

ExecutionStatus = Literal["ok", "failed", "empty", "missing"]
FailureClass = Literal[
    "max_turns_with_incomplete_beat",
    "instrument_contract_false_positive",
    "instrument_true_failure",
]
PrimaryStatus = Literal["included_semantic", "excluded_unscored", "excluded_failure"]
FrozenStatus = Literal["included_frozen", "excluded_unscored", "excluded_failure"]
EventType = Literal["timeout", "prompt_too_long", "retry"]
Population = Literal["semantic", "frozen"]
OutlierStatus = Literal["computed", "insufficient_n"]

LOCKED_UNSCORED_IDS = frozenset(
    {
        dialogue_id("edge_01", "fsm", 2),
        dialogue_id("edge_13", "fsm", 2),
    }
)
_DIALOGUE_ID_IN_TEXT = re.compile(r"([a-z0-9_]+__(?:baseline|fsm)__rep\d{2})")
_MIN_TUKEY_N = 4
_LIVE_FROZEN_ELIGIBLE = 342
_LIVE_SEMANTIC_ELIGIBLE = 350
_LIVE_FROZEN_SCORED = 340
_LIVE_SEMANTIC_SCORED = 348
_LIVE_UNSCORED = 2
_LIVE_CFP = 8
_LIVE_MAX_TURNS = 9
_LIVE_TIF = 1
_LIVE_JOBS = 360


class AuditError(RuntimeError):
    """The audit cannot proceed or export must refuse; the message says why."""


@dataclass(frozen=True)
class JobAudit:
    """One expected job: execution outcome and inclusion, never collapsed."""

    dialogue_id: str
    scenario_id: str
    agent: str
    repetition: int
    execution_status: ExecutionStatus
    failure_class: FailureClass | None = None
    primary_status: PrimaryStatus | None = None
    frozen_status: FrozenStatus | None = None
    unscored_reason: str | None = None


@dataclass(frozen=True)
class LlmCallEvent:
    """One timeout, prompt-budget, or retry event; not an inclusion decision."""

    event_type: EventType
    caller: str
    dialogue_id: str | None
    eventual_primary_status: PrimaryStatus | None
    eventual_frozen_status: FrozenStatus | None
    error: str | None
    attempts: int


@dataclass(frozen=True)
class LatencyGroup:
    """Tukey fences for one agent in one scored population."""

    agent: str
    population: Population
    n: int
    status: OutlierStatus
    q1: float | None
    q3: float | None
    lower: float | None
    upper: float | None
    outlier_ids: list[str]


@dataclass(frozen=True)
class AuditReport:
    """Census of every manifest job plus independent event and outlier reports."""

    jobs: list[JobAudit]
    events: list[LlmCallEvent]
    latency: list[LatencyGroup]
    anomalies: list[str]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``'s bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(run_dir: Path, sidecar_dir: Path) -> AuditReport:
    """Census ``run_dir`` against ``sidecar_dir`` without writing exports."""
    manifest = _load_manifest(run_dir)
    anomalies: list[str] = []
    _check_duplicate_jobs(manifest.jobs, anomalies)
    cfp_ids = _cfp_ids(run_dir, expected_exp_id=manifest.exp_id, anomalies=anomalies)
    expected_ids = {
        dialogue_id(job.scenario_id, job.agent, job.repetition) for job in manifest.jobs
    }
    logs_by_id, extra_logs = _index_logs(run_dir, expected_ids, anomalies)
    frozen_rows = _read_metrics(run_dir / METRICS_CSV, anomalies, label="frozen-gate")
    sidecar_rows = _read_metrics(sidecar_dir / METRICS_CSV, anomalies, label="sidecar")
    unscored = _read_unscored(run_dir / UNSCORED_CSV, anomalies)
    jobs = [
        _classify_job(
            job,
            logs_by_id=logs_by_id,
            frozen_ids=set(frozen_rows),
            sidecar_ids=set(sidecar_rows),
            unscored=unscored,
            cfp_ids=cfp_ids,
            anomalies=anomalies,
        )
        for job in manifest.jobs
    ]
    job_ids = {job.dialogue_id for job in jobs}
    for extra_id in extra_logs:
        anomalies.append(f"extra dialogue log {extra_id} is not a manifest job")
    _check_metric_orphans_and_duplicates(
        frozen_rows, job_ids, anomalies, label="frozen-gate"
    )
    _check_metric_orphans_and_duplicates(
        sidecar_rows, job_ids, anomalies, label="sidecar"
    )
    _check_inclusion_sets(jobs, frozen_rows, sidecar_rows, anomalies)
    if any(job.primary_status is None or job.frozen_status is None for job in jobs):
        leftovers = [
            job.dialogue_id
            for job in jobs
            if job.primary_status is None or job.frozen_status is None
        ]
        anomalies.append("residual unclassified job: " + ", ".join(leftovers))
    by_id = {job.dialogue_id: job for job in jobs}
    events = _llm_call_events(run_dir, by_id)
    latency = [
        *_latency_groups(sidecar_rows, by_id, "semantic"),
        *_latency_groups(frozen_rows, by_id, "frozen"),
    ]
    return AuditReport(
        jobs=jobs, events=events, latency=latency, anomalies=_unique(anomalies)
    )


def audited_export_dir(run_dir: Path) -> Path:
    """Return ``results/<exp_id>`` from the run manifest."""
    return Path("results") / _load_manifest(run_dir).exp_id


def export_audited_metrics(
    run_dir: Path,
    sidecar_dir: Path,
    out_dir: Path | None = None,
    *,
    expected_sidecar_sha256: str = SIDECAR_METRICS_SHA256,
    expected_frozen_sha256: str = FROZEN_GATE_METRICS_SHA256,
) -> AuditReport:
    """Byte-copy the two scored CSVs into ``out_dir`` after the audit gates pass.

    ``out_dir`` defaults to :func:`audited_export_dir` (``results/<exp_id>``).
    """
    report = audit_run(run_dir, sidecar_dir)
    if report.anomalies:
        raise AuditError("; ".join(report.anomalies))
    sidecar_source = sidecar_dir / METRICS_CSV
    frozen_source = run_dir / METRICS_CSV
    _require_full_hash(sidecar_source, expected_sidecar_sha256, label="sidecar")
    _require_full_hash(frozen_source, expected_frozen_sha256, label="frozen-gate")
    if (
        expected_sidecar_sha256 == SIDECAR_METRICS_SHA256
        and expected_frozen_sha256 == FROZEN_GATE_METRICS_SHA256
    ):
        _require_live_exp_final(report)
    destination = audited_export_dir(run_dir) if out_dir is None else out_dir
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_byte_copy(sidecar_source, destination / PRIMARY_CSV)
    _atomic_byte_copy(frozen_source, destination / FROZEN_EXPORT_CSV)
    _write_outliers(destination / OUTLIERS_CSV, report.latency)
    return report


def _load_manifest(run_dir: Path) -> Manifest:
    """Load ``manifest.json`` or refuse."""
    path = run_dir / "manifest.json"
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as failure:
        raise AuditError(f"{path} is not a sim run manifest: {failure}") from failure


def _check_duplicate_jobs(jobs: Sequence[JobRef], anomalies: list[str]) -> None:
    """Record duplicate manifest jobs."""
    counts = Counter(
        dialogue_id(job.scenario_id, job.agent, job.repetition) for job in jobs
    )
    for item, count in counts.items():
        if count > 1:
            anomalies.append(f"duplicate manifest job {item} ({count} times)")


def _cfp_ids(run_dir: Path, *, expected_exp_id: str, anomalies: list[str]) -> set[str]:
    """Return adjudicated CFP ids, or empty when the freeze file is absent."""
    path = contract_false_positives_path(run_dir)
    if not path.is_file():
        return set()
    try:
        allowlist = load_failed_inclusion_allowlist(path)
    except AdjudicationError as failure:
        anomalies.append(str(failure))
        return set()
    if allowlist.exp_id != expected_exp_id:
        anomalies.append(
            f"inclusion artifact exp_id={allowlist.exp_id!r} does not match "
            f"run exp_id={expected_exp_id!r}"
        )
    return set(allowlist.ids)


def _index_logs(
    run_dir: Path, expected_ids: set[str], anomalies: list[str]
) -> tuple[dict[str, list[Path]], set[str]]:
    """Map dialogue ids to log paths; extra files are not in the job list."""
    directory = run_dir / "dialogues"
    by_id: dict[str, list[Path]] = {}
    if not directory.is_dir():
        return by_id, set()
    for path in sorted(directory.glob("*.jsonl")):
        try:
            log = DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
            item = log.dialogue_id
        except (OSError, ValueError):
            item = path.stem
        by_id.setdefault(item, []).append(path)
    for item, paths in by_id.items():
        if len(paths) > 1:
            anomalies.append(
                f"duplicate dialogue log for {item}: "
                + ", ".join(path.name for path in paths)
            )
    extra = {item for item in by_id if item not in expected_ids}
    return by_id, extra


def _classify_job(
    job: JobRef,
    *,
    logs_by_id: dict[str, list[Path]],
    frozen_ids: set[str],
    sidecar_ids: set[str],
    unscored: dict[str, str],
    cfp_ids: set[str],
    anomalies: list[str],
) -> JobAudit:
    """Classify one manifest job: execution first, then inclusion."""
    item = dialogue_id(job.scenario_id, job.agent, job.repetition)
    expected = run_relative_log(job)
    paths = logs_by_id.get(item, [])
    execution, failure_class, log = _execution_of(
        item, paths, expected=expected, cfp_ids=cfp_ids, anomalies=anomalies
    )
    primary, frozen, reason = _inclusion_of(
        item,
        execution_status=execution,
        failure_class=failure_class,
        frozen_ids=frozen_ids,
        sidecar_ids=sidecar_ids,
        unscored=unscored,
        anomalies=anomalies,
    )
    del log
    return JobAudit(
        dialogue_id=item,
        scenario_id=job.scenario_id,
        agent=job.agent,
        repetition=job.repetition,
        execution_status=execution,
        failure_class=failure_class,
        primary_status=primary,
        frozen_status=frozen,
        unscored_reason=reason,
    )


def run_relative_log(job: JobRef) -> str:
    """Return the expected T-14a file name for ``job``."""
    return dialogue_filename(job.scenario_id, job.agent, job.repetition)


def _execution_of(
    item: str,
    paths: Sequence[Path],
    *,
    expected: str,
    cfp_ids: set[str],
    anomalies: list[str],
) -> tuple[ExecutionStatus, FailureClass | None, DialogueLog | None]:
    """Return execution status and named failure class from the log files."""
    if not paths:
        anomalies.append(f"missing expected dialogue {item} ({expected})")
        return "missing", None, None
    path = paths[0]
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        anomalies.append(f"empty dialogue {item}")
        return "empty", None, None
    try:
        log = DialogueLog.model_validate_json(raw)
    except ValueError as failure:
        anomalies.append(f"unreadable dialogue {item}: {failure}")
        return "empty", None, None
    if not log.records:
        anomalies.append(f"empty dialogue {item}")
        return "empty", None, None
    if log.status == "ok":
        return "ok", None, log
    failure_class = _failure_class(log, cfp_ids)
    if failure_class is None:
        anomalies.append(
            f"unknown or unexplained failure class for {item}: "
            f"{log.failure_kind}/{log.failure_reason}"
        )
    return "failed", failure_class, log


def _failure_class(log: DialogueLog, cfp_ids: set[str]) -> FailureClass | None:
    """Name a known failed-execution class, or None when the class is unexplained."""
    if (
        log.failure_kind == "simulation"
        and log.failure_reason == "max_turns_with_incomplete_beat"
    ):
        return "max_turns_with_incomplete_beat"
    if (
        log.failure_kind == TARGET_FAILURE_KIND
        and log.failure_reason == TARGET_FAILURE_REASON
    ):
        if log.dialogue_id in cfp_ids:
            return "instrument_contract_false_positive"
        return "instrument_true_failure"
    return None


def _inclusion_of(
    item: str,
    *,
    execution_status: ExecutionStatus,
    failure_class: FailureClass | None,
    frozen_ids: set[str],
    sidecar_ids: set[str],
    unscored: dict[str, str],
    anomalies: list[str],
) -> tuple[PrimaryStatus | None, FrozenStatus | None, str | None]:
    """Set inclusion from metric files; never rewrite execution."""
    if execution_status in {"missing", "empty"}:
        return None, None, None
    if execution_status == "ok":
        if item in unscored:
            return "excluded_unscored", "excluded_unscored", unscored[item]
        if item in sidecar_ids and item in frozen_ids:
            return "included_semantic", "included_frozen", None
        anomalies.append(f"ok job {item} is not a complete scored pair")
        return None, None, None
    if failure_class == "instrument_contract_false_positive":
        if item not in sidecar_ids:
            anomalies.append(f"CFP {item} is missing from the sidecar metrics")
            return None, "excluded_failure", None
        if item in frozen_ids:
            anomalies.append(f"CFP {item} must not appear in frozen-gate metrics")
        return "included_semantic", "excluded_failure", None
    if failure_class in {
        "max_turns_with_incomplete_beat",
        "instrument_true_failure",
    }:
        if item in sidecar_ids or item in frozen_ids:
            anomalies.append(f"excluded failure {item} has a metric row")
        return "excluded_failure", "excluded_failure", None
    return None, None, None


def _read_metrics(
    path: Path, anomalies: list[str], *, label: str
) -> dict[str, dict[str, str]]:
    """Index a metrics CSV by dialogue id."""
    if not path.is_file():
        anomalies.append(f"{label} {path.name} is missing")
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                item = dialogue_id(
                    row["scenario_id"], row["agent"], int(row["repetition"])
                )
            except (KeyError, TypeError, ValueError) as failure:
                anomalies.append(f"{label} row is not a dialogue id: {failure}")
                continue
            if item in rows:
                anomalies.append(f"duplicate {label} metric dialogue_id {item}")
            rows[item] = row
    return rows


def _read_unscored(path: Path, anomalies: list[str]) -> dict[str, str]:
    """Index ``unscored.csv`` by dialogue id."""
    if not path.is_file():
        return {}
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                item = dialogue_id(
                    row["scenario_id"], row["agent"], int(row["repetition"])
                )
            except (KeyError, TypeError, ValueError) as failure:
                anomalies.append(f"unscored row is not a dialogue id: {failure}")
                continue
            rows[item] = row.get("reason") or ""
    return rows


def _check_metric_orphans_and_duplicates(
    rows: dict[str, dict[str, str]],
    job_ids: set[str],
    anomalies: list[str],
    *,
    label: str,
) -> None:
    """Refuse metric rows that are not manifest jobs."""
    for item in rows:
        if item not in job_ids:
            anomalies.append(f"{label} metric row {item} is not a manifest job")


def _check_inclusion_sets(
    jobs: Sequence[JobAudit],
    frozen_rows: dict[str, dict[str, str]],
    sidecar_rows: dict[str, dict[str, str]],
    anomalies: list[str],
) -> None:
    """Require exact ID-set equality between inclusion flags and CSV rows."""
    primary = {
        job.dialogue_id for job in jobs if job.primary_status == "included_semantic"
    }
    frozen = {job.dialogue_id for job in jobs if job.frozen_status == "included_frozen"}
    if primary != set(sidecar_rows):
        anomalies.append("sidecar id set does not match audited included_semantic ids")
    if frozen != set(frozen_rows):
        anomalies.append(
            "frozen-gate id set does not match audited included_frozen ids"
        )


def _llm_call_events(run_dir: Path, jobs: dict[str, JobAudit]) -> list[LlmCallEvent]:
    """Read ``llm_calls.jsonl`` as events, never as exclusions."""
    path = run_dir / LLM_CALLS_LOG
    if not path.is_file():
        return []
    events: list[LlmCallEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = LlmCallRecord.model_validate_json(stripped)
            event_type = _event_type(record)
            if event_type is None:
                continue
            item = _dialogue_id_from_record(record)
            job = jobs.get(item) if item is not None else None
            events.append(
                LlmCallEvent(
                    event_type=event_type,
                    caller=record.caller,
                    dialogue_id=item,
                    eventual_primary_status=job.primary_status if job else None,
                    eventual_frozen_status=job.frozen_status if job else None,
                    error=record.error,
                    attempts=record.attempts,
                )
            )
    return events


def _event_type(record: LlmCallRecord) -> EventType | None:
    """Classify one LLM-call log line as a reportable event, or skip it."""
    error = (record.error or "").lower()
    if "timed out" in error:
        return "timeout"
    if "over the" in error and "budget" in error:
        return "prompt_too_long"
    if "prompt uses" in error:
        return "prompt_too_long"
    if record.attempts > 1:
        return "retry"
    return None


def _dialogue_id_from_record(record: LlmCallRecord) -> str | None:
    """Return a dialogue id embedded in the call's messages, when present."""
    for message in record.messages:
        match = _DIALOGUE_ID_IN_TEXT.search(message.get("content", ""))
        if match:
            return match.group(1)
    return None


def _latency_groups(
    rows: dict[str, dict[str, str]],
    jobs: dict[str, JobAudit],
    population: Population,
) -> list[LatencyGroup]:
    """Tukey report per agent for one scored population."""
    included = "included_semantic" if population == "semantic" else "included_frozen"
    wanted = {
        job.dialogue_id
        for job in jobs.values()
        if (
            job.primary_status == included
            if population == "semantic"
            else job.frozen_status == included
        )
    }
    by_agent: dict[str, list[tuple[str, float]]] = {}
    for item in wanted:
        row = rows.get(item)
        if row is None:
            continue
        latency = _finite(row.get("llm_latency_s"))
        if latency is None:
            continue
        by_agent.setdefault(jobs[item].agent, []).append((item, latency))
    return [
        _tukey(agent, population, values) for agent, values in sorted(by_agent.items())
    ]


def _tukey(
    agent: str, population: Population, values: Sequence[tuple[str, float]]
) -> LatencyGroup:
    """Inclusive Tukey 1.5 IQR, or ``insufficient_n`` below four finite values."""
    latencies = [latency for _item, latency in values]
    if len(latencies) < _MIN_TUKEY_N:
        return LatencyGroup(
            agent=agent,
            population=population,
            n=len(latencies),
            status="insufficient_n",
            q1=None,
            q3=None,
            lower=None,
            upper=None,
            outlier_ids=[],
        )
    quartiles = statistics.quantiles(latencies, n=4, method="inclusive")
    q1, _q2, q3 = quartiles
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = [item for item, latency in values if latency < lower or latency > upper]
    return LatencyGroup(
        agent=agent,
        population=population,
        n=len(latencies),
        status="computed",
        q1=q1,
        q3=q3,
        lower=lower,
        upper=upper,
        outlier_ids=outliers,
    )


def _finite(value: str | None) -> float | None:
    """Parse a finite float, or None when the cell is empty or non-numeric."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _require_full_hash(path: Path, expected: str, *, label: str) -> None:
    """Refuse unless ``path`` matches the complete 64-character SHA-256."""
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise AuditError(f"{label} expected SHA-256 is not a 64-character hex digest")
    digest = file_sha256(path)
    if digest != expected:
        raise AuditError(f"{label} SHA-256 is {digest}, expected {expected}")


def _atomic_byte_copy(source: Path, dest: Path) -> None:
    """Copy ``source`` to ``dest`` via a temp file after hashing the copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f"{dest.name}.{uuid4().hex}.tmp")
    payload = source.read_bytes()
    temporary.write_bytes(payload)
    if file_sha256(temporary) != file_sha256(source):
        temporary.unlink(missing_ok=True)
        raise AuditError(f"destination hash of {dest.name} does not match source")
    temporary.replace(dest)


def _write_outliers(path: Path, groups: Sequence[LatencyGroup]) -> None:
    """Write the report-only Tukey table; never a metrics column."""
    fieldnames = (
        "population",
        "agent",
        "n",
        "status",
        "q1",
        "q3",
        "lower",
        "upper",
        "outlier_ids",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    "population": group.population,
                    "agent": group.agent,
                    "n": group.n,
                    "status": group.status,
                    "q1": group.q1 if group.q1 is not None else "",
                    "q3": group.q3 if group.q3 is not None else "",
                    "lower": group.lower if group.lower is not None else "",
                    "upper": group.upper if group.upper is not None else "",
                    "outlier_ids": " ".join(group.outlier_ids),
                }
            )


def _require_live_exp_final(report: AuditReport) -> None:
    """Check the frozen T-17 accounting when exporting the real hashed CSVs."""
    n_ok = sum(job.execution_status == "ok" for job in report.jobs)
    n_cfp = sum(
        job.failure_class == "instrument_contract_false_positive" for job in report.jobs
    )
    n_max = sum(
        job.failure_class == "max_turns_with_incomplete_beat" for job in report.jobs
    )
    n_tif = sum(job.failure_class == "instrument_true_failure" for job in report.jobs)
    n_unscored = sum(job.primary_status == "excluded_unscored" for job in report.jobs)
    n_frozen = sum(job.frozen_status == "included_frozen" for job in report.jobs)
    n_semantic = sum(job.primary_status == "included_semantic" for job in report.jobs)
    if len(report.jobs) != _LIVE_JOBS:
        raise AuditError(
            f"live census expected {_LIVE_JOBS} jobs, got {len(report.jobs)}"
        )
    if n_ok != _LIVE_FROZEN_ELIGIBLE:
        raise AuditError(
            f"live frozen-gate eligible expected {_LIVE_FROZEN_ELIGIBLE}, got {n_ok}"
        )
    if n_ok + n_cfp != _LIVE_SEMANTIC_ELIGIBLE:
        raise AuditError(
            "live semantic-primary eligible expected "
            f"{_LIVE_SEMANTIC_ELIGIBLE}, got {n_ok + n_cfp}"
        )
    if n_ok + n_cfp + n_max + n_tif != _LIVE_JOBS:
        raise AuditError("live execution census does not close over 360 jobs")
    if n_unscored != _LIVE_UNSCORED:
        raise AuditError(f"live eligible-but-unscored expected {_LIVE_UNSCORED}")
    if n_cfp != _LIVE_CFP:
        raise AuditError(f"live CFP expected {_LIVE_CFP}, got {n_cfp}")
    if n_max != _LIVE_MAX_TURNS or n_tif != _LIVE_TIF:
        raise AuditError(
            f"live failures expected {_LIVE_MAX_TURNS} max_turns and {_LIVE_TIF} TIF"
        )
    if n_frozen != _LIVE_FROZEN_SCORED or n_semantic != _LIVE_SEMANTIC_SCORED:
        raise AuditError(
            f"live scored expected {_LIVE_FROZEN_SCORED}/{_LIVE_SEMANTIC_SCORED}, "
            f"got {n_frozen}/{n_semantic}"
        )
    missing_locked = LOCKED_UNSCORED_IDS - {
        job.dialogue_id
        for job in report.jobs
        if job.primary_status == "excluded_unscored"
    }
    if missing_locked:
        raise AuditError(f"locked unscored ids missing: {sorted(missing_locked)}")


def _unique(items: Sequence[str]) -> list[str]:
    """Preserve order while dropping duplicate anomaly strings."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
