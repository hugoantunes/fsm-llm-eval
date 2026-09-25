"""Sidecar eligibility for T-17 contract-false-positive failed logs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import (
    CONFIG,
    CannedEvalLlm,
    RunCanned,
    make_dialogue_log,
    make_turn_record,
)
from sim.adjudication import SOURCE_DISCOVERY_RELATIVE
from sim.config import load_models_config
from sim.eval import (
    EvalError,
    evaluate_run,
    load_failed_inclusion_allowlist,
    sidecar_out_dir,
)
from sim.fsm import FsmSpec
from sim.kb import KnowledgeBase
from sim.schemas import DialogueLog, dialogue_filename


def _instrument_failed(
    *,
    scenario_id: str,
    agent: str,
    repetition: int = 2,
    kind: str = "instrument",
    reason: str = "invalid_candidate_retry_exhausted",
) -> DialogueLog:
    """One failed log with a single turn the canned judge can score."""
    log = make_dialogue_log(
        [make_turn_record(agent_reply="Hello from support.")],
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
        status="failed",
        stop_reason=None,
    )
    return log.model_copy(update={"failure_kind": kind, "failure_reason": reason})


def _write_log(run_dir: Path, log: DialogueLog) -> Path:
    """Write ``log`` under the T-14a file name and return the path."""
    path = (
        run_dir
        / "dialogues"
        / dialogue_filename(log.scenario_id, log.agent, log.repetition)
    )
    path.write_text(log.model_dump_json() + "\n", encoding="utf-8")
    return path


def _write_allowlist(path: Path, ids: list[str]) -> Path:
    """Write a sidecar inclusion JSON for tests."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "exp_id": "exp",
                "failure_kind": "instrument",
                "failure_reason": "invalid_candidate_retry_exhausted",
                "source_discovery": SOURCE_DISCOVERY_RELATIVE,
                "ids": ids,
            }
        ),
        encoding="utf-8",
    )
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Return rows of a CSV written by eval."""
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_eval_still_skips_failed_logs_when_no_allowlist_is_given(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    included = _instrument_failed(scenario_id="happy_path_01", agent="baseline")
    other = _instrument_failed(
        scenario_id="happy_path_01",
        agent="fsm",
        reason="max_turns_with_incomplete_beat",
        kind="simulation",
    )
    _write_log(run_dir, included)
    _write_log(run_dir, other)

    result = evaluate_run(
        run_dir,
        llm=canned_eval_llm,
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
    )
    rows = _read_csv_rows(run_dir / "metrics.csv")
    judged = [call for call in canned_eval_llm.calls if call["caller"] == "judge_facts"]

    assert result.n_scored == 4
    assert result.n_failed == 2
    assert result.n_included_failed == 0
    assert len(rows) == 4
    assert len(judged) == 4
    assert ("happy_path_01", "baseline", 2) not in {
        (row["scenario_id"], row["agent"], int(row["repetition"])) for row in rows
    }


def test_allowlist_includes_only_the_named_failed_logs(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    tmp_path: Path,
) -> None:
    run_canned()
    included = _instrument_failed(scenario_id="happy_path_01", agent="baseline")
    true_instrument = _instrument_failed(scenario_id="adversarial_01", agent="fsm")
    max_turns = _instrument_failed(
        scenario_id="happy_path_01",
        agent="fsm",
        kind="simulation",
        reason="max_turns_with_incomplete_beat",
    )
    included_path = _write_log(run_dir, included)
    _write_log(run_dir, true_instrument)
    _write_log(run_dir, max_turns)
    before = included_path.read_bytes()
    allowlist = _write_allowlist(
        tmp_path / "allowlist.json",
        ["happy_path_01__baseline__rep02"],
    )
    sidecar = tmp_path / "sidecar"
    llm = CannedEvalLlm()

    result = evaluate_run(
        run_dir,
        llm=llm,
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
        include_failed_from=allowlist,
        out_dir=sidecar,
    )

    rows = _read_csv_rows(sidecar / "metrics.csv")
    keys = {(row["scenario_id"], row["agent"], int(row["repetition"])) for row in rows}
    judged = [call for call in llm.calls if call["caller"] == "judge_facts"]
    sidecar_row = next(
        row
        for row in rows
        if (row["scenario_id"], row["agent"], int(row["repetition"]))
        == ("happy_path_01", "baseline", 2)
    )

    assert result.n_scored == 5
    assert result.n_failed == 3
    assert result.n_included_failed == 1
    assert keys == {
        ("happy_path_01", "baseline", 1),
        ("happy_path_01", "fsm", 1),
        ("adversarial_01", "baseline", 1),
        ("adversarial_01", "fsm", 1),
        ("happy_path_01", "baseline", 2),
    }
    assert ("adversarial_01", "fsm", 2) not in keys
    assert ("happy_path_01", "fsm", 2) not in keys
    assert len(judged) == 5
    assert sidecar_row["runtime_status"] == "failed"
    assert sidecar_row["failure_kind"] == "instrument"
    assert sidecar_row["failure_reason"] == "invalid_candidate_retry_exhausted"
    assert sidecar_row["adjudication"] == "CONTRACT_FALSE_POSITIVE"
    assert sidecar_row["adjudication_date"] == ""
    assert sidecar_row["inclusion_source"] == str(allowlist)
    assert json.loads(included_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert included_path.read_bytes() == before
    assert not (run_dir / "metrics.csv").exists()
    inclusion = json.loads((sidecar / "inclusion.json").read_text(encoding="utf-8"))
    assert inclusion["included_ids"] == ["happy_path_01__baseline__rep02"]
    assert inclusion["n_included_failed"] == 1


def test_sidecar_scores_ok_dialogues_with_the_same_metric_values_as_default_eval(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    tmp_path: Path,
) -> None:
    run_canned()
    included = _instrument_failed(scenario_id="happy_path_01", agent="baseline")
    _write_log(run_dir, included)
    evaluate_run(
        run_dir,
        llm=CannedEvalLlm(),
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
    )
    default_rows = {
        (row["scenario_id"], row["agent"], int(row["repetition"])): row
        for row in _read_csv_rows(run_dir / "metrics.csv")
    }

    sidecar = tmp_path / "sidecar"
    evaluate_run(
        run_dir,
        llm=CannedEvalLlm(),
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
        include_failed_from=_write_allowlist(
            tmp_path / "allowlist.json", ["happy_path_01__baseline__rep02"]
        ),
        out_dir=sidecar,
    )
    sidecar_rows = {
        (row["scenario_id"], row["agent"], int(row["repetition"])): row
        for row in _read_csv_rows(sidecar / "metrics.csv")
    }

    for key, row in default_rows.items():
        sidecar_row = sidecar_rows[key]
        for field in (
            "fact_f1",
            "task_completed",
            "claim_support",
            "accuracy_score",
            "n_turns",
            "flow_adherence",
        ):
            assert sidecar_row[field] == row[field]
        assert sidecar_row["runtime_status"] == "ok"
        assert sidecar_row["adjudication"] == ""
        assert sidecar_row["inclusion_source"] == "status_ok"


def test_allowlist_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_allowlist(
        tmp_path / "allowlist.json",
        ["happy_path_01__baseline__rep02", "happy_path_01__baseline__rep02"],
    )

    with pytest.raises(EvalError, match="unique"):
        load_failed_inclusion_allowlist(path)


def test_sidecar_eval_defaults_out_to_a_sibling_of_the_run(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    tmp_path: Path,
) -> None:
    run_canned()
    allowlist = _write_allowlist(tmp_path / "allowlist.json", [])
    expected = sidecar_out_dir(run_dir)

    result = evaluate_run(
        run_dir,
        llm=canned_eval_llm,
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
        include_failed_from=allowlist,
    )

    assert expected != run_dir
    assert (expected / "metrics.csv").exists()
    assert not (run_dir / "metrics.csv").exists()
    assert result.n_included_failed == 0


def test_sidecar_eval_refuses_to_write_into_the_original_run(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    tmp_path: Path,
) -> None:
    run_canned()
    allowlist = _write_allowlist(tmp_path / "allowlist.json", [])

    with pytest.raises(EvalError, match="distinct"):
        evaluate_run(
            run_dir,
            llm=canned_eval_llm,
            kb=real_kb,
            fsm=real_fsm,
            config=load_models_config(CONFIG),
            include_failed_from=allowlist,
            out_dir=run_dir,
        )


def test_allowlist_rejects_a_missing_or_ok_or_wrong_class_log(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    tmp_path: Path,
) -> None:
    run_canned()
    sidecar = tmp_path / "sidecar"
    config = load_models_config(CONFIG)
    missing = _write_allowlist(
        tmp_path / "missing.json", ["happy_path_01__baseline__rep02"]
    )

    with pytest.raises(EvalError, match="happy_path_01__baseline__rep02"):
        evaluate_run(
            run_dir,
            llm=canned_eval_llm,
            kb=real_kb,
            fsm=real_fsm,
            config=config,
            include_failed_from=missing,
            out_dir=sidecar,
        )

    ok_as_failed = _write_allowlist(
        tmp_path / "ok.json", ["happy_path_01__baseline__rep01"]
    )
    with pytest.raises(EvalError, match="status=ok"):
        evaluate_run(
            run_dir,
            llm=canned_eval_llm,
            kb=real_kb,
            fsm=real_fsm,
            config=config,
            include_failed_from=ok_as_failed,
            out_dir=sidecar,
        )

    _write_log(
        run_dir,
        _instrument_failed(
            scenario_id="happy_path_01",
            agent="fsm",
            kind="simulation",
            reason="max_turns_with_incomplete_beat",
        ),
    )
    wrong_class = _write_allowlist(
        tmp_path / "max_turns.json", ["happy_path_01__fsm__rep02"]
    )
    with pytest.raises(EvalError, match="failure_kind='simulation'"):
        evaluate_run(
            run_dir,
            llm=canned_eval_llm,
            kb=real_kb,
            fsm=real_fsm,
            config=config,
            include_failed_from=wrong_class,
            out_dir=sidecar,
        )
