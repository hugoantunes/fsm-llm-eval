"""T-18: census every manifest job; fail closed; export scored CSVs as byte copies."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import pytest

from helpers import (
    DOCS_DIR,
    make_dialogue_log,
    make_llm_call_record,
    make_manifest,
    make_turn_record,
    write_llm_calls,
)
from sim.adjudication import (
    SOURCE_DISCOVERY_RELATIVE,
    contract_false_positives_path,
)
from sim.audit import (
    FROZEN_GATE_METRICS_SHA256,
    SIDECAR_METRICS_SHA256,
    AuditError,
    audit_run,
    audited_export_dir,
    export_audited_metrics,
    file_sha256,
)
from sim.schemas import DialogueLog, JobRef, dialogue_filename, dialogue_id

_LABELER_REASON = (
    "the stage labeler returned 5 label(s) for 4 turn(s). The schema enum "
    "cannot pin the length, so a mismatch is a failed call, not a padded path"
)


def _ok(
    scenario_id: str,
    agent: str,
    repetition: int,
    *,
    latency: float = 1.0,
) -> DialogueLog:
    """One ok log with a single turn."""
    return make_dialogue_log(
        [make_turn_record(llm_latency_s=latency)],
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
    )


def _failed(
    scenario_id: str,
    agent: str,
    repetition: int,
    *,
    kind: str,
    reason: str,
) -> DialogueLog:
    """One failed log that still has a turn on disk."""
    log = make_dialogue_log(
        [make_turn_record()],
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
        status="failed",
        stop_reason=None,
    )
    return log.model_copy(update={"failure_kind": kind, "failure_reason": reason})


def _job(log: DialogueLog) -> JobRef:
    """The manifest job for ``log``."""
    return JobRef(
        scenario_id=log.scenario_id,
        agent=log.agent,
        repetition=log.repetition,
    )


def _write_log(run_dir: Path, log: DialogueLog) -> Path:
    """Write ``log`` under the T-14a file name."""
    path = (
        run_dir
        / "dialogues"
        / dialogue_filename(log.scenario_id, log.agent, log.repetition)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(log.model_dump_json() + "\n", encoding="utf-8")
    return path


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]
) -> Path:
    """Write a metrics or unscored CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _metrics_row(
    log: DialogueLog, *, latency: float | None = None
) -> dict[str, object]:
    """Identity plus latency for one scored row."""
    if latency is None:
        latency = log.records[0].llm_latency_s if log.records else 1.0
    return {
        "scenario_id": log.scenario_id,
        "agent": log.agent,
        "repetition": log.repetition,
        "llm_latency_s": latency,
    }


def _write_cfp(run_dir: Path, ids: Sequence[str], *, exp_id: str = "exp") -> None:
    """Write a valid T-17 inclusion JSON for ``ids``."""
    path = contract_false_positives_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "exp_id": exp_id,
                "failure_kind": "instrument",
                "failure_reason": "invalid_candidate_retry_exhausted",
                "source_discovery": SOURCE_DISCOVERY_RELATIVE,
                "ids": list(ids),
            }
        ),
        encoding="utf-8",
    )


def _layout(
    tmp_path: Path,
    logs: Sequence[DialogueLog],
    *,
    jobs: Sequence[JobRef] | None = None,
    n_ok: int = 99,
    write_logs: bool = True,
    frozen_rows: Sequence[dict[str, object]] | None = None,
    sidecar_rows: Sequence[dict[str, object]] | None = None,
    unscored: Sequence[dict[str, object]] = (),
    cfp_ids: Sequence[str] = (),
    sidecar_extra: Sequence[tuple[str, ...]] | None = None,
) -> tuple[Path, Path]:
    """A run dir and a sibling sidecar dir for one test."""
    run_dir = tmp_path / "exp"
    sidecar_dir = tmp_path / "exp_semantic"
    run_dir.mkdir(parents=True)
    sidecar_dir.mkdir(parents=True)
    job_list = list(jobs) if jobs is not None else [_job(log) for log in logs]
    (run_dir / "manifest.json").write_text(
        make_manifest(jobs=job_list, n_ok=n_ok, n_failed=0).model_dump_json(),
        encoding="utf-8",
    )
    if write_logs:
        for log in logs:
            _write_log(run_dir, log)
    if cfp_ids:
        _write_cfp(run_dir, cfp_ids)
    _write_csv(
        run_dir / "unscored.csv",
        ("scenario_id", "agent", "repetition", "reason"),
        list(unscored),
    )
    frozen = (
        list(frozen_rows)
        if frozen_rows is not None
        else [_metrics_row(log) for log in logs if log.status == "ok"]
    )
    sidecar = list(sidecar_rows) if sidecar_rows is not None else list(frozen)
    if sidecar_extra:
        sidecar.extend(
            {
                "scenario_id": scenario_id,
                "agent": agent,
                "repetition": repetition,
                "llm_latency_s": 1.0,
                "runtime_status": "failed",
                "adjudication": "CONTRACT_FALSE_POSITIVE",
            }
            for scenario_id, agent, repetition in sidecar_extra
        )
    _write_csv(
        run_dir / "metrics.csv",
        ("scenario_id", "agent", "repetition", "llm_latency_s"),
        frozen,
    )
    fields = (
        "scenario_id",
        "agent",
        "repetition",
        "llm_latency_s",
        "runtime_status",
        "adjudication",
    )
    sidecar_written = [
        {
            "runtime_status": row.get("runtime_status", "ok"),
            "adjudication": row.get("adjudication", ""),
            **{
                key: row[key]
                for key in ("scenario_id", "agent", "repetition", "llm_latency_s")
            },
        }
        for row in sidecar
    ]
    _write_csv(sidecar_dir / "metrics.csv", fields, sidecar_written)
    return run_dir, sidecar_dir


def _by_id(report) -> dict[str, object]:
    """Index the census by dialogue id."""
    return {job.dialogue_id: job for job in report.jobs}


def test_census_universe_is_manifest_jobs_not_n_ok(tmp_path: Path) -> None:
    present = _ok("happy_path_01", "baseline", 1)
    other = _ok("happy_path_01", "fsm", 1)
    missing = JobRef(scenario_id="edge_01", agent="baseline", repetition=1)
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [present, other],
        jobs=[_job(present), _job(other), missing],
        n_ok=99,
        frozen_rows=[_metrics_row(present), _metrics_row(other)],
        sidecar_rows=[_metrics_row(present), _metrics_row(other)],
    )

    report = audit_run(run_dir, sidecar_dir)

    assert len(report.jobs) == 3
    jobs = _by_id(report)
    assert jobs[present.dialogue_id].execution_status == "ok"
    assert jobs[other.dialogue_id].execution_status == "ok"
    assert jobs[dialogue_id("edge_01", "baseline", 1)].execution_status == "missing"
    assert "missing" in " ".join(report.anomalies).lower()


def test_census_separates_execution_status_from_inclusion(tmp_path: Path) -> None:
    scored = _ok("happy_path_01", "baseline", 1)
    cfp = _failed(
        "adversarial_01",
        "baseline",
        3,
        kind="instrument",
        reason="invalid_candidate_retry_exhausted",
    )
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [scored, cfp],
        frozen_rows=[_metrics_row(scored)],
        sidecar_rows=[_metrics_row(scored)],
        sidecar_extra=[("adversarial_01", "baseline", 3)],
        cfp_ids=[cfp.dialogue_id],
    )

    jobs = _by_id(audit_run(run_dir, sidecar_dir))

    assert jobs[scored.dialogue_id].execution_status == "ok"
    assert jobs[scored.dialogue_id].primary_status == "included_semantic"
    assert jobs[scored.dialogue_id].frozen_status == "included_frozen"
    assert jobs[cfp.dialogue_id].execution_status == "failed"
    assert jobs[cfp.dialogue_id].failure_class == "instrument_contract_false_positive"
    assert jobs[cfp.dialogue_id].primary_status == "included_semantic"
    assert jobs[cfp.dialogue_id].frozen_status == "excluded_failure"


def test_empty_dialogue_is_an_integrity_anomaly(tmp_path: Path) -> None:
    empty = make_dialogue_log(
        [],
        scenario_id="happy_path_01",
        agent="baseline",
        repetition=1,
        stop_reason=None,
    )
    run_dir, sidecar_dir = _layout(tmp_path, [empty], frozen_rows=[], sidecar_rows=[])

    report = audit_run(run_dir, sidecar_dir)

    job = _by_id(report)[empty.dialogue_id]
    assert job.execution_status == "empty"
    with pytest.raises(AuditError, match="empty"):
        export_audited_metrics(
            run_dir,
            sidecar_dir,
            tmp_path / "results",
            expected_sidecar_sha256=file_sha256(sidecar_dir / "metrics.csv"),
            expected_frozen_sha256=file_sha256(run_dir / "metrics.csv"),
        )


def test_cfp_stays_failed_execution_and_is_included_only_in_semantic(
    tmp_path: Path,
) -> None:
    scored = _ok("happy_path_01", "baseline", 1)
    cfp = _failed(
        "edge_16",
        "fsm",
        2,
        kind="instrument",
        reason="invalid_candidate_retry_exhausted",
    )
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [scored, cfp],
        frozen_rows=[_metrics_row(scored)],
        sidecar_rows=[_metrics_row(scored)],
        sidecar_extra=[("edge_16", "fsm", 2)],
        cfp_ids=[cfp.dialogue_id],
    )
    before = (
        run_dir / "dialogues" / dialogue_filename("edge_16", "fsm", 2)
    ).read_bytes()

    report = audit_run(run_dir, sidecar_dir)
    job = _by_id(report)[cfp.dialogue_id]
    export_audited_metrics(
        run_dir,
        sidecar_dir,
        tmp_path / "results",
        expected_sidecar_sha256=file_sha256(sidecar_dir / "metrics.csv"),
        expected_frozen_sha256=file_sha256(run_dir / "metrics.csv"),
    )

    assert job.execution_status == "failed"
    assert job.failure_class == "instrument_contract_false_positive"
    assert job.primary_status == "included_semantic"
    assert job.frozen_status == "excluded_failure"
    frozen_ids = {
        dialogue_id(row["scenario_id"], row["agent"], int(row["repetition"]))
        for row in csv.DictReader(
            (tmp_path / "results" / "metrics_frozen_gate.csv").open()
        )
    }
    primary_ids = {
        dialogue_id(row["scenario_id"], row["agent"], int(row["repetition"]))
        for row in csv.DictReader((tmp_path / "results" / "metrics.csv").open())
    }
    assert cfp.dialogue_id not in frozen_ids
    assert cfp.dialogue_id in primary_ids
    assert (
        run_dir / "dialogues" / dialogue_filename("edge_16", "fsm", 2)
    ).read_bytes() == before


def test_max_turns_and_true_instrument_failure_are_excluded_not_rescored(
    tmp_path: Path,
) -> None:
    scored = _ok("happy_path_01", "baseline", 1)
    max_turns = _failed(
        "adversarial_07",
        "baseline",
        1,
        kind="simulation",
        reason="max_turns_with_incomplete_beat",
    )
    tif = _failed(
        "adversarial_06",
        "fsm",
        2,
        kind="instrument",
        reason="invalid_candidate_retry_exhausted",
    )
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [scored, max_turns, tif],
        frozen_rows=[_metrics_row(scored)],
        sidecar_rows=[_metrics_row(scored)],
        cfp_ids=[],
    )
    _write_cfp(run_dir, [])

    jobs = _by_id(audit_run(run_dir, sidecar_dir))

    assert jobs[max_turns.dialogue_id].failure_class == "max_turns_with_incomplete_beat"
    assert jobs[max_turns.dialogue_id].primary_status == "excluded_failure"
    assert jobs[max_turns.dialogue_id].frozen_status == "excluded_failure"
    assert jobs[tif.dialogue_id].failure_class == "instrument_true_failure"
    assert jobs[tif.dialogue_id].primary_status == "excluded_failure"
    assert jobs[tif.dialogue_id].frozen_status == "excluded_failure"


def test_unscored_labeler_mismatches_are_excluded_with_reason(tmp_path: Path) -> None:
    scored = _ok("happy_path_01", "baseline", 1)
    unscored = _ok("edge_01", "fsm", 2)
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [scored, unscored],
        frozen_rows=[_metrics_row(scored)],
        sidecar_rows=[_metrics_row(scored)],
        unscored=[
            {
                "scenario_id": "edge_01",
                "agent": "fsm",
                "repetition": 2,
                "reason": _LABELER_REASON,
            }
        ],
    )

    job = _by_id(audit_run(run_dir, sidecar_dir))[unscored.dialogue_id]

    assert job.execution_status == "ok"
    assert job.primary_status == "excluded_unscored"
    assert job.frozen_status == "excluded_unscored"
    assert job.unscored_reason == _LABELER_REASON


def test_llm_call_errors_are_events_not_exclusions(tmp_path: Path) -> None:
    scored = _ok("happy_path_12", "fsm", 2)
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [scored],
        frozen_rows=[_metrics_row(scored)],
        sidecar_rows=[_metrics_row(scored)],
    )
    timed_out = make_llm_call_record(caller="judge_facts", role="judge").model_copy(
        update={
            "error": "judge_facts: judge-model failed after 1 attempt(s): timed out",
            "attempts": 1,
            "messages": [
                {"role": "user", "content": f"transcript {scored.dialogue_id}"}
            ],
        }
    )
    too_long = make_llm_call_record(caller="baseline", role="agent").model_copy(
        update={
            "error": (
                "baseline: prompt uses 7000 of 8192 tokens, over the 80% budget "
                "of 6553: shorten the prompt"
            ),
            "attempts": 1,
            "messages": [{"role": "user", "content": f"prompt {scored.dialogue_id}"}],
        }
    )
    retry = make_llm_call_record(caller="judge_facts", role="judge").model_copy(
        update={
            "attempts": 2,
            "error": None,
            "messages": [{"role": "user", "content": f"retry {scored.dialogue_id}"}],
        }
    )
    write_llm_calls(run_dir, [timed_out, too_long, retry])

    report = audit_run(run_dir, sidecar_dir)
    job = _by_id(report)[scored.dialogue_id]
    types = {event.event_type for event in report.events}

    assert job.primary_status == "included_semantic"
    assert job.frozen_status == "included_frozen"
    assert types == {"timeout", "prompt_too_long", "retry"}
    assert all(event.dialogue_id == scored.dialogue_id for event in report.events)
    assert all(
        event.eventual_primary_status == "included_semantic" for event in report.events
    )


def test_latency_outliers_use_inclusive_quartiles_and_are_not_exported(
    tmp_path: Path,
) -> None:
    latencies = [10.0, 11.0, 12.0, 13.0, 100.0]
    logs = [
        _ok("happy_path_01", "baseline", i, latency=latency)
        for i, latency in enumerate(latencies, start=1)
    ]
    short = [_ok("edge_01", "fsm", i, latency=float(i)) for i in range(1, 4)]
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [*logs, *short],
        frozen_rows=[_metrics_row(log) for log in [*logs, *short]],
        sidecar_rows=[_metrics_row(log) for log in [*logs, *short]],
    )
    q1, _, q3 = statistics.quantiles(latencies, n=4, method="inclusive")
    iqr = q3 - q1
    source = (sidecar_dir / "metrics.csv").read_bytes()

    report = audit_run(run_dir, sidecar_dir)
    export_audited_metrics(
        run_dir,
        sidecar_dir,
        tmp_path / "results",
        expected_sidecar_sha256=file_sha256(sidecar_dir / "metrics.csv"),
        expected_frozen_sha256=file_sha256(run_dir / "metrics.csv"),
    )

    baseline = next(
        group
        for group in report.latency
        if group.agent == "baseline" and group.population == "semantic"
    )
    fsm = next(
        group
        for group in report.latency
        if group.agent == "fsm" and group.population == "semantic"
    )
    assert baseline.status == "computed"
    assert baseline.outlier_ids == [logs[-1].dialogue_id]
    assert baseline.lower == pytest.approx(q1 - 1.5 * iqr)
    assert baseline.upper == pytest.approx(q3 + 1.5 * iqr)
    assert fsm.status == "insufficient_n"
    assert fsm.outlier_ids == []
    assert (tmp_path / "results" / "metrics.csv").read_bytes() == source
    assert (
        "llm_latency_outlier" not in (tmp_path / "results" / "metrics.csv").read_text()
    )


def test_export_refuses_on_integrity_anomalies(tmp_path: Path) -> None:
    a = _ok("happy_path_01", "baseline", 1)
    b = _ok("happy_path_01", "fsm", 1)
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [a, b],
        jobs=[_job(a), _job(a)],
        frozen_rows=[_metrics_row(a), _metrics_row(b)],
        sidecar_rows=[_metrics_row(a), _metrics_row(b)],
    )

    with pytest.raises(AuditError, match="duplicate"):
        export_audited_metrics(
            run_dir,
            sidecar_dir,
            tmp_path / "results",
            expected_sidecar_sha256=file_sha256(sidecar_dir / "metrics.csv"),
            expected_frozen_sha256=file_sha256(run_dir / "metrics.csv"),
        )

    unknown = _failed(
        "happy_path_01",
        "baseline",
        1,
        kind="operational",
        reason="disk_full",
    )
    run_dir, sidecar_dir = _layout(
        tmp_path / "unknown",
        [unknown],
        frozen_rows=[],
        sidecar_rows=[],
    )
    with pytest.raises(AuditError, match=r"unknown|unexplained"):
        export_audited_metrics(
            run_dir,
            sidecar_dir,
            tmp_path / "results-unknown",
            expected_sidecar_sha256=file_sha256(sidecar_dir / "metrics.csv"),
            expected_frozen_sha256=file_sha256(run_dir / "metrics.csv"),
        )

    scored = _ok("happy_path_01", "baseline", 1)
    run_dir, sidecar_dir = _layout(
        tmp_path / "orphan",
        [scored],
        frozen_rows=[
            _metrics_row(scored),
            {
                "scenario_id": "edge_99",
                "agent": "baseline",
                "repetition": 1,
                "llm_latency_s": 1.0,
            },
        ],
        sidecar_rows=[_metrics_row(scored)],
    )
    with pytest.raises(AuditError, match=r"orphan|not a manifest"):
        export_audited_metrics(
            run_dir,
            sidecar_dir,
            tmp_path / "results-orphan",
            expected_sidecar_sha256=file_sha256(sidecar_dir / "metrics.csv"),
            expected_frozen_sha256=file_sha256(run_dir / "metrics.csv"),
        )


def test_export_requires_exact_audited_id_set_match(tmp_path: Path) -> None:
    a = _ok("happy_path_01", "baseline", 1)
    b = _ok("happy_path_01", "fsm", 1)
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [a, b],
        frozen_rows=[_metrics_row(a), _metrics_row(b)],
        sidecar_rows=[_metrics_row(a)],
    )

    with pytest.raises(AuditError, match="id set"):
        export_audited_metrics(
            run_dir,
            sidecar_dir,
            tmp_path / "results",
            expected_sidecar_sha256=file_sha256(sidecar_dir / "metrics.csv"),
            expected_frozen_sha256=file_sha256(run_dir / "metrics.csv"),
        )


def test_export_is_atomic_byte_copy_with_full_sha256_gate(tmp_path: Path) -> None:
    log = _ok("happy_path_01", "baseline", 1)
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [log],
        frozen_rows=[_metrics_row(log)],
        sidecar_rows=[_metrics_row(log)],
    )
    sidecar_hash = file_sha256(sidecar_dir / "metrics.csv")
    frozen_hash = file_sha256(run_dir / "metrics.csv")
    sidecar_bytes = (sidecar_dir / "metrics.csv").read_bytes()
    frozen_bytes = (run_dir / "metrics.csv").read_bytes()
    assert len(sidecar_hash) == 64
    assert len(frozen_hash) == 64

    with pytest.raises(AuditError, match="SHA-256"):
        export_audited_metrics(
            run_dir,
            sidecar_dir,
            tmp_path / "results-bad",
            expected_sidecar_sha256="0" * 64,
            expected_frozen_sha256=frozen_hash,
        )

    export_audited_metrics(
        run_dir,
        sidecar_dir,
        tmp_path / "results",
        expected_sidecar_sha256=sidecar_hash,
        expected_frozen_sha256=frozen_hash,
    )

    primary = tmp_path / "results" / "metrics.csv"
    sensitivity = tmp_path / "results" / "metrics_frozen_gate.csv"
    assert primary.read_bytes() == sidecar_bytes
    assert sensitivity.read_bytes() == frozen_bytes
    assert file_sha256(primary) == sidecar_hash
    assert file_sha256(sensitivity) == frozen_hash
    assert (sidecar_dir / "metrics.csv").read_bytes() == sidecar_bytes
    assert (run_dir / "metrics.csv").read_bytes() == frozen_bytes


def test_omitted_out_dir_writes_under_results_exp_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    log = _ok("happy_path_01", "baseline", 1)
    run_dir, sidecar_dir = _layout(
        tmp_path,
        [log],
        frozen_rows=[_metrics_row(log)],
        sidecar_rows=[_metrics_row(log)],
    )

    assert audited_export_dir(run_dir) == Path("results") / "exp"
    export_audited_metrics(
        run_dir,
        sidecar_dir,
        expected_sidecar_sha256=file_sha256(sidecar_dir / "metrics.csv"),
        expected_frozen_sha256=file_sha256(run_dir / "metrics.csv"),
    )

    dest = tmp_path / "results" / "exp"
    assert (dest / "metrics.csv").read_bytes() == (
        sidecar_dir / "metrics.csv"
    ).read_bytes()
    assert (dest / "metrics_frozen_gate.csv").read_bytes() == (
        run_dir / "metrics.csv"
    ).read_bytes()
    assert not (tmp_path / "results" / "metrics.csv").exists()


def test_pinned_export_hashes_match_execution_doc() -> None:
    doc = (DOCS_DIR / "execution.md").read_text(encoding="utf-8")

    assert FROZEN_GATE_METRICS_SHA256 in doc
    assert SIDECAR_METRICS_SHA256 in doc
    assert len(FROZEN_GATE_METRICS_SHA256) == 64
    assert len(SIDECAR_METRICS_SHA256) == 64
    assert FROZEN_GATE_METRICS_SHA256 != SIDECAR_METRICS_SHA256


def test_docs_audit_names_every_exclusion_class() -> None:
    doc = (DOCS_DIR / "audit.md").read_text(encoding="utf-8")

    assert "eligible-but-unscored" in doc
    assert "instrument_contract_false_positive" in doc
    assert "instrument_true_failure" in doc
    assert "max_turns_with_incomplete_beat" in doc
    assert "excluded_unscored" in doc
    assert "edge_01" in doc
    assert "edge_13" in doc
    assert "342" in doc
    assert "350" in doc
    assert "Preserve judge provenance" in doc
    assert FROZEN_GATE_METRICS_SHA256 in doc
    assert SIDECAR_METRICS_SHA256 in doc
    assert "insufficient_n" in doc
