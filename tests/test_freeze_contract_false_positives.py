"""Freeze/materialize contract false positives from explicit full run IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import (
    REPO_ROOT,
    load_script,
    make_dialogue_log,
    make_turn_record,
    write_run,
)
from sim.adjudication import (
    SOURCE_DISCOVERY_RELATIVE,
    AdjudicationError,
    canonical_json,
    contract_false_positives_path,
    freeze_contract_false_positives,
    write_instrument_failures,
)
from sim.eval import load_failed_inclusion_allowlist as eval_load
from sim.schemas import DialogueLog, Manifest

freeze = load_script("scripts/freeze_contract_false_positives.py")


def _failed(
    *,
    scenario_id: str,
    agent: str = "baseline",
    repetition: int = 1,
    kind: str = "instrument",
    reason: str = "invalid_candidate_retry_exhausted",
    status: str = "failed",
) -> DialogueLog:
    """One log with a single turn and the given runtime class."""
    log = make_dialogue_log(
        [make_turn_record(user_message="Where is order NL-20260145?")],
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
        status=status,  # type: ignore[arg-type]
        stop_reason=None if status == "failed" else "goal_reached",
    )
    return log.model_copy(update={"failure_kind": kind, "failure_reason": reason})


def _prepare(tmp_path: Path, *logs: DialogueLog, exp_id: str = "exp") -> Path:
    """Write a run, set ``exp_id``, and discover instrument failures."""
    run_dir = write_run(tmp_path / "run", logs)
    manifest = Manifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    (run_dir / "manifest.json").write_text(
        manifest.model_copy(update={"exp_id": exp_id}).model_dump_json(),
        encoding="utf-8",
    )
    write_instrument_failures(run_dir)
    return run_dir


def test_unknown_id_is_rejected(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path, _failed(scenario_id="edge_16", repetition=2))

    with pytest.raises(AdjudicationError, match="is not in the run"):
        freeze_contract_false_positives(run_dir, ["happy_path_01__baseline__rep01"])


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path, _failed(scenario_id="edge_16", repetition=2))

    with pytest.raises(AdjudicationError, match="duplicate"):
        freeze_contract_false_positives(
            run_dir, ["edge_16__baseline__rep02", "edge_16__baseline__rep02"]
        )


def test_ok_status_is_rejected(tmp_path: Path) -> None:
    ok = _failed(scenario_id="happy_path_01", status="ok", kind="instrument")
    retry = _failed(scenario_id="edge_16", repetition=2)
    run_dir = _prepare(tmp_path, ok, retry)

    with pytest.raises(AdjudicationError, match="status=ok"):
        freeze_contract_false_positives(run_dir, ["happy_path_01__baseline__rep01"])


def test_max_turns_is_rejected(tmp_path: Path) -> None:
    run_dir = _prepare(
        tmp_path,
        _failed(
            scenario_id="happy_path_01",
            agent="fsm",
            kind="simulation",
            reason="max_turns_with_incomplete_beat",
        ),
        _failed(scenario_id="edge_16", repetition=2),
    )

    with pytest.raises(AdjudicationError, match="max_turns_with_incomplete_beat"):
        freeze_contract_false_positives(run_dir, ["happy_path_01__fsm__rep01"])


def test_id_absent_from_discovery_is_rejected(tmp_path: Path) -> None:
    retry = _failed(scenario_id="edge_16", repetition=2)
    extra = _failed(scenario_id="happy_path_01", agent="fsm", repetition=3)
    run_dir = _prepare(tmp_path, retry, extra)
    discovery = json.loads(
        (run_dir / "adjudication" / "instrument_failures.json").read_text(
            encoding="utf-8"
        )
    )
    discovery["candidates"] = [
        row
        for row in discovery["candidates"]
        if row["run_id"] != "happy_path_01__fsm__rep03"
    ]
    (run_dir / "adjudication" / "instrument_failures.json").write_text(
        json.dumps(discovery, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(AdjudicationError, match="absent from"):
        freeze_contract_false_positives(run_dir, ["happy_path_01__fsm__rep03"])
    assert not contract_false_positives_path(run_dir).exists()


def test_output_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    run_dir = _prepare(
        tmp_path,
        _failed(scenario_id="edge_16", repetition=2),
        _failed(scenario_id="happy_path_01", agent="fsm", repetition=1),
        exp_id="exp_final",
    )
    ids = ["happy_path_01__fsm__rep01", "edge_16__baseline__rep02"]

    first = freeze_contract_false_positives(run_dir, ids)
    blob = first.read_bytes()
    freeze_contract_false_positives(run_dir, list(reversed(ids)))

    assert first.read_bytes() == blob
    payload = json.loads(blob.decode())
    assert payload["schema_version"] == 1
    assert payload["exp_id"] == "exp_final"
    assert payload["ids"] == [
        "edge_16__baseline__rep02",
        "happy_path_01__fsm__rep01",
    ]
    assert "timestamp" not in blob.decode()
    assert str(run_dir.resolve()) not in blob.decode()


def test_existing_different_freeze_is_not_silently_overwritten(tmp_path: Path) -> None:
    run_dir = _prepare(
        tmp_path,
        _failed(scenario_id="edge_16", repetition=2),
        _failed(scenario_id="happy_path_01", agent="fsm", repetition=1),
    )
    path = freeze_contract_false_positives(run_dir, ["edge_16__baseline__rep02"])
    before = path.read_bytes()

    with pytest.raises(AdjudicationError, match="--replace"):
        freeze_contract_false_positives(run_dir, ["happy_path_01__fsm__rep01"])

    assert path.read_bytes() == before
    freeze_contract_false_positives(
        run_dir, ["happy_path_01__fsm__rep01"], replace=True
    )
    assert json.loads(path.read_text(encoding="utf-8"))["ids"] == [
        "happy_path_01__fsm__rep01"
    ]


def test_validation_failure_does_not_modify_an_existing_freeze(tmp_path: Path) -> None:
    run_dir = _prepare(
        tmp_path,
        _failed(scenario_id="edge_16", repetition=2),
        _failed(scenario_id="happy_path_01", agent="fsm", repetition=1),
    )
    path = freeze_contract_false_positives(run_dir, ["edge_16__baseline__rep02"])
    before = path.read_bytes()

    with pytest.raises(AdjudicationError):
        freeze_contract_false_positives(run_dir, ["edge_16"])

    assert path.read_bytes() == before


def test_generated_artifact_is_accepted_by_eval_loader(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path, _failed(scenario_id="edge_16", repetition=2))
    path = freeze_contract_false_positives(run_dir, ["edge_16__baseline__rep02"])

    loaded = eval_load(path)
    assert loaded.ids == ("edge_16__baseline__rep02",)
    assert loaded.failure_reason == "invalid_candidate_retry_exhausted"


def test_cli_writes_run_local_json(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path, _failed(scenario_id="edge_16", repetition=2))

    assert freeze.main([str(run_dir), "edge_16__baseline__rep02"]) == 0
    assert contract_false_positives_path(run_dir).exists()


def test_freeze_implementation_does_not_pin_production_ids() -> None:
    source = (REPO_ROOT / "scripts" / "freeze_contract_false_positives.py").read_text(
        encoding="utf-8"
    )

    for item in (
        "adversarial_01__baseline__rep03",
        "adversarial_06__fsm__rep02",
        "edge_16__baseline__rep02",
    ):
        assert item not in source


def test_canonical_json_is_stable() -> None:
    from sim.adjudication import FailedInclusionAllowlist

    left = canonical_json(
        FailedInclusionAllowlist(
            exp_id="exp_final",
            failure_kind="instrument",
            failure_reason="invalid_candidate_retry_exhausted",
            source_discovery=SOURCE_DISCOVERY_RELATIVE,
            ids=("b__fsm__rep01", "a__baseline__rep01"),
        )
    )
    right = canonical_json(
        FailedInclusionAllowlist(
            exp_id="exp_final",
            failure_kind="instrument",
            failure_reason="invalid_candidate_retry_exhausted",
            source_discovery=SOURCE_DISCOVERY_RELATIVE,
            ids=("b__fsm__rep01", "a__baseline__rep01"),
        )
    )
    assert left == right
    assert left.endswith("\n")
