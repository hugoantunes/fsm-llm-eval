"""Run-local adjudication provenance. Not experimental observations.

Discovery, audit, and freeze artifacts belong under
``runs/<exp_id>/adjudication/``. They do not recode original JSONL files.
Scripts automate discovery, evidence collection, mechanical validation, and
JSON materialization. They do not decide semantic inclusion.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from sim.io import atomic_write
from sim.schemas import DialogueLog, FailureKind, Manifest, dialogue_id

ADJUDICATION_DIRNAME = "adjudication"
INSTRUMENT_FAILURES_NAME = "instrument_failures.json"
AUDIT_JSON_NAME = "audit.json"
AUDIT_MD_NAME = "audit.md"
CONTRACT_FALSE_POSITIVES_NAME = "contract_false_positives.json"
SOURCE_DISCOVERY_RELATIVE = f"{ADJUDICATION_DIRNAME}/{INSTRUMENT_FAILURES_NAME}"
CONTRACT_FALSE_POSITIVES_RELATIVE = Path(
    f"{ADJUDICATION_DIRNAME}/{CONTRACT_FALSE_POSITIVES_NAME}"
)
TARGET_FAILURE_KIND: FailureKind = "instrument"
TARGET_FAILURE_REASON = "invalid_candidate_retry_exhausted"
SCHEMA_VERSION = 1
SIDECAR_ADJUDICATION = "CONTRACT_FALSE_POSITIVE"


class AdjudicationError(RuntimeError):
    """Discovery or freeze cannot proceed; the message says what is missing."""


class FailureClassCensus(BaseModel):
    """Count of one failed runtime class in a run."""

    model_config = ConfigDict(extra="forbid")

    failure_kind: str | None
    failure_reason: str | None
    count: int
    target_adjudication: bool


class InstrumentFailureCandidate(BaseModel):
    """One observation that requires human/code-level adjudication."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    scenario_id: str
    agent: str
    repetition: int
    seed: int
    status: Literal["failed"]
    failure_kind: str
    failure_reason: str
    source: str
    needs_adjudication: Literal[True] = True


class InstrumentFailuresReport(BaseModel):
    """Deterministic census of instrument retry-exhausted failures."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SCHEMA_VERSION
    exp_id: str
    failure_kind: Literal["instrument"] = TARGET_FAILURE_KIND
    failure_reason: Literal["invalid_candidate_retry_exhausted"] = TARGET_FAILURE_REASON
    census: list[FailureClassCensus]
    candidates: list[InstrumentFailureCandidate]


class FailedInclusionAllowlist(BaseModel):
    """Executable inclusion artifact generated from an explicit human selection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SCHEMA_VERSION
    exp_id: str
    failure_kind: FailureKind
    failure_reason: str
    source_discovery: str
    ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ids_are_unique_full_run_ids(self) -> FailedInclusionAllowlist:
        """Fail unless every id is unique, well-formed, and of the target class."""
        if self.failure_kind != TARGET_FAILURE_KIND:
            raise ValueError(
                f"failure_kind must be {TARGET_FAILURE_KIND!r}, not "
                f"{self.failure_kind!r}"
            )
        if self.failure_reason != TARGET_FAILURE_REASON:
            raise ValueError(
                f"failure_reason must be {TARGET_FAILURE_REASON!r}, not "
                f"{self.failure_reason!r}"
            )
        if self.source_discovery != SOURCE_DISCOVERY_RELATIVE:
            raise ValueError(
                f"source_discovery must be {SOURCE_DISCOVERY_RELATIVE!r}, not "
                f"{self.source_discovery!r}"
            )
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("ids must be unique")
        for item in self.ids:
            parse_dialogue_id(item)
        return self


def adjudication_dir(run_dir: Path) -> Path:
    """Return ``<run>/adjudication``."""
    return run_dir / ADJUDICATION_DIRNAME


def instrument_failures_path(run_dir: Path) -> Path:
    """Return the run-local discovery artifact path."""
    return adjudication_dir(run_dir) / INSTRUMENT_FAILURES_NAME


def audit_json_path(run_dir: Path) -> Path:
    """Return the run-local audit JSON path."""
    return adjudication_dir(run_dir) / AUDIT_JSON_NAME


def audit_markdown_path(run_dir: Path) -> Path:
    """Return the run-local audit Markdown path."""
    return adjudication_dir(run_dir) / AUDIT_MD_NAME


def contract_false_positives_path(run_dir: Path) -> Path:
    """Return the run-local executable inclusion artifact path."""
    return adjudication_dir(run_dir) / CONTRACT_FALSE_POSITIVES_NAME


def resolve_include_failed_from(path: Path | None, run_dir: Path) -> Path | None:
    """Resolve ``--include-failed-from``; a bare flag uses the run-local freeze."""
    if path is None:
        return None
    if path == CONTRACT_FALSE_POSITIVES_RELATIVE:
        return contract_false_positives_path(run_dir)
    return path


def canonical_json(payload: BaseModel) -> str:
    """Serialize ``payload`` with stable formatting and no timestamp."""
    return json.dumps(payload.model_dump(mode="json"), indent=2) + "\n"


def parse_dialogue_id(value: str) -> tuple[str, str, int]:
    """Parse a full dialogue-run id, or raise ``ValueError``."""
    try:
        scenario_id, agent, rep = value.rsplit("__", 2)
        if not rep.startswith("rep"):
            raise ValueError
        repetition = int(rep.removeprefix("rep"))
    except ValueError as failure:
        raise ValueError(
            f"{value!r} is not a full dialogue-run id of the form "
            "{scenario}__{agent}__repNN"
        ) from failure
    if agent not in ("baseline", "fsm"):
        raise ValueError(f"{value!r} names unknown agent {agent!r}")
    if dialogue_id(scenario_id, agent, repetition) != value:
        raise ValueError(f"{value!r} does not round-trip as a dialogue id")
    return scenario_id, agent, repetition


def load_failed_inclusion_allowlist(path: Path) -> FailedInclusionAllowlist:
    """Load and validate a generated failed-inclusion artifact."""
    try:
        return FailedInclusionAllowlist.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as failure:
        raise AdjudicationError(
            f"{path} is not a failed-inclusion allowlist: {failure}"
        ) from failure


def load_run_manifest(run_dir: Path) -> Manifest:
    """Load ``manifest.json`` from ``run_dir``."""
    path = run_dir / "manifest.json"
    if not path.exists():
        raise AdjudicationError(
            f"{path} is missing. Discovery and freeze read a sim run directory"
        )
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as failure:
        raise AdjudicationError(
            f"{path} does not match the manifest schema:\n{failure}"
        ) from failure


def iter_dialogue_logs(run_dir: Path) -> list[tuple[Path, DialogueLog]]:
    """Load every dialogue JSONL under ``run_dir/dialogues``, in file-name order."""
    directory = run_dir / "dialogues"
    if not directory.exists():
        raise AdjudicationError(
            f"{directory} is missing. Discovery reads dialogue JSONL, never metrics"
        )
    rows: list[tuple[Path, DialogueLog]] = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            log = DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as failure:
            raise AdjudicationError(
                f"{path} does not match the dialogue schema:\n{failure}"
            ) from failure
        rows.append((path, log))
    return rows


def logs_by_id(rows: Sequence[tuple[Path, DialogueLog]]) -> dict[str, DialogueLog]:
    """Index logs by full run id; refuse duplicates."""
    by_id: dict[str, DialogueLog] = {}
    for _path, log in rows:
        if log.dialogue_id in by_id:
            raise AdjudicationError(
                f"{log.dialogue_id} identifies more than one observation"
            )
        by_id[log.dialogue_id] = log
    return by_id


def discover_instrument_failures(run_dir: Path) -> InstrumentFailuresReport:
    """Census failed classes and list target instrument observations."""
    manifest = load_run_manifest(run_dir)
    rows = iter_dialogue_logs(run_dir)
    failed = [(path, log) for path, log in rows if log.status == "failed"]
    counts: Counter[tuple[str | None, str | None]] = Counter(
        (log.failure_kind, log.failure_reason) for _path, log in failed
    )
    census = [
        FailureClassCensus(
            failure_kind=kind,
            failure_reason=reason,
            count=count,
            target_adjudication=(
                kind == TARGET_FAILURE_KIND and reason == TARGET_FAILURE_REASON
            ),
        )
        for (kind, reason), count in sorted(
            counts.items(), key=lambda item: (item[0][0] or "", item[0][1] or "")
        )
    ]
    candidates = [
        InstrumentFailureCandidate(
            run_id=log.dialogue_id,
            scenario_id=log.scenario_id,
            agent=log.agent,
            repetition=log.repetition,
            seed=log.seed,
            status="failed",
            failure_kind=log.failure_kind or "",
            failure_reason=log.failure_reason or "",
            source=path.relative_to(run_dir).as_posix(),
        )
        for path, log in failed
        if log.failure_kind == TARGET_FAILURE_KIND
        and log.failure_reason == TARGET_FAILURE_REASON
    ]
    candidates.sort(key=lambda item: item.run_id)
    return InstrumentFailuresReport(
        exp_id=manifest.exp_id,
        census=census,
        candidates=candidates,
    )


def write_instrument_failures(
    run_dir: Path, report: InstrumentFailuresReport | None = None
) -> Path:
    """Write ``instrument_failures.json`` under the run's adjudication directory."""
    payload = report if report is not None else discover_instrument_failures(run_dir)
    path = instrument_failures_path(run_dir)
    atomic_write(path, canonical_json(payload))
    return path


def load_instrument_failures(run_dir: Path) -> InstrumentFailuresReport:
    """Load the run-local discovery artifact."""
    path = instrument_failures_path(run_dir)
    if not path.exists():
        raise AdjudicationError(
            f"{path} is missing. Discover instrument failures before freeze: "
            f"just instrument-failures {run_dir}"
        )
    try:
        return InstrumentFailuresReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as failure:
        raise AdjudicationError(
            f"{path} is not an instrument-failures census: {failure}"
        ) from failure


def resolve_included_failed(
    logs: Sequence[DialogueLog], allowlist: FailedInclusionAllowlist
) -> list[DialogueLog]:
    """Return the failed logs named by ``allowlist``, in allowlist order."""
    by_id: dict[str, DialogueLog] = {}
    for log in logs:
        if log.dialogue_id in by_id:
            raise AdjudicationError(
                f"{log.dialogue_id} identifies more than one observation"
            )
        by_id[log.dialogue_id] = log
    resolved: list[DialogueLog] = []
    for item in allowlist.ids:
        log = by_id.get(item)
        if log is None:
            raise AdjudicationError(
                f"{item} is in the failed-inclusion allowlist but missing from the run"
            )
        if log.status != "failed":
            raise AdjudicationError(
                f"{item} is in the failed-inclusion allowlist but has "
                f"status={log.status}"
            )
        if log.failure_kind != allowlist.failure_kind:
            raise AdjudicationError(
                f"{item} has failure_kind={log.failure_kind!r}, allowlist expected "
                f"{allowlist.failure_kind!r}"
            )
        if log.failure_reason != allowlist.failure_reason:
            raise AdjudicationError(
                f"{item} has failure_reason={log.failure_reason!r}, allowlist expected "
                f"{allowlist.failure_reason!r}"
            )
        resolved.append(log)
    return resolved


def freeze_contract_false_positives(
    run_dir: Path,
    ids: Sequence[str],
    *,
    replace: bool = False,
) -> Path:
    """Validate ``ids`` and write the run-local executable inclusion artifact."""
    selected = list(ids)
    errors = _freeze_errors(run_dir, selected)
    if errors:
        raise AdjudicationError("\n".join(errors))
    manifest = load_run_manifest(run_dir)
    artifact = FailedInclusionAllowlist(
        exp_id=manifest.exp_id,
        failure_kind=TARGET_FAILURE_KIND,
        failure_reason=TARGET_FAILURE_REASON,
        source_discovery=SOURCE_DISCOVERY_RELATIVE,
        ids=tuple(sorted(set(selected))),
    )
    path = contract_false_positives_path(run_dir)
    candidate = canonical_json(artifact)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if _json_equivalent(existing, candidate):
            return path
        if not replace:
            raise AdjudicationError(
                f"{path} already encodes a different frozen selection. "
                "Re-running with the same IDs is idempotent; changing the "
                "experimental population requires --replace"
            )
    atomic_write(path, candidate)
    return path


def _freeze_errors(run_dir: Path, ids: Sequence[str]) -> list[str]:
    """Return validation problems for a freeze selection; empty if valid."""
    errors: list[str] = []
    seen: dict[str, int] = {}
    for item in ids:
        seen[item] = seen.get(item, 0) + 1
        try:
            parse_dialogue_id(item)
        except ValueError as failure:
            errors.append(str(failure))
    duplicates = sorted(item for item, count in seen.items() if count > 1)
    if duplicates:
        errors.append("duplicate ids: " + ", ".join(duplicates))
    unique = list(dict.fromkeys(ids))
    well_formed = []
    for item in unique:
        try:
            parse_dialogue_id(item)
        except ValueError:
            continue
        well_formed.append(item)
    rows = iter_dialogue_logs(run_dir)
    by_id = logs_by_id(rows)
    discovery = load_instrument_failures(run_dir)
    discovered = {candidate.run_id for candidate in discovery.candidates}
    for item in well_formed:
        log = by_id.get(item)
        if log is None:
            errors.append(f"{item} is not in the run")
            continue
        if log.status != "failed":
            errors.append(f"{item} has status={log.status}")
        if (
            log.failure_kind != TARGET_FAILURE_KIND
            or log.failure_reason != TARGET_FAILURE_REASON
        ):
            errors.append(
                f"{item} has failure_kind={log.failure_kind!r} "
                f"failure_reason={log.failure_reason!r}; freeze accepts only "
                f"{TARGET_FAILURE_KIND!r} / {TARGET_FAILURE_REASON!r}"
            )
        if item not in discovered:
            errors.append(f"{item} is absent from {SOURCE_DISCOVERY_RELATIVE}")
    return errors


def _json_equivalent(left: str, right: str) -> bool:
    """Whether two JSON documents decode to the same value."""
    try:
        return json.loads(left) == json.loads(right)
    except json.JSONDecodeError:
        return False
