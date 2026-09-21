"""Instrument-failure discovery: census and candidates, never a verdict."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
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
    discover_instrument_failures,
    instrument_failures_path,
    write_instrument_failures,
)
from sim.schemas import DialogueLog, TurnRecord

discover = load_script("scripts/list_instrument_failures.py")

PRODUCTION_IDS = (
    "adversarial_01__baseline__rep03",
    "edge_16__baseline__rep02",
    "edge_16__baseline__rep03",
    "happy_path_09__baseline__rep02",
    "adversarial_19__fsm__rep02",
    "edge_16__fsm__rep02",
    "edge_16__fsm__rep03",
    "happy_path_09__fsm__rep02",
    "adversarial_06__fsm__rep02",
)
VERDICTS = ("CONTRACT_FALSE_POSITIVE", "TRUE_INSTRUMENT_FAILURE")


def _failed(
    *,
    scenario_id: str,
    agent: str = "baseline",
    repetition: int = 1,
    seed: int = 42,
    kind: str = "instrument",
    reason: str = "invalid_candidate_retry_exhausted",
    status: str = "failed",
) -> DialogueLog:
    """One log with a single turn and the given runtime class."""
    records: Sequence[TurnRecord] = [
        make_turn_record(user_message="Where is order NL-20260145?")
    ]
    log = make_dialogue_log(
        records,
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
        seed=seed,
        status=status,  # type: ignore[arg-type]
        stop_reason=None if status == "failed" else "goal_reached",
    )
    return log.model_copy(update={"failure_kind": kind, "failure_reason": reason})


def test_discovers_all_instrument_retry_exhausted(tmp_path: Path) -> None:
    retry_a = _failed(scenario_id="happy_path_01", agent="baseline", repetition=1)
    retry_b = _failed(scenario_id="edge_16", agent="fsm", repetition=2)
    max_turns = _failed(
        scenario_id="happy_path_02",
        agent="fsm",
        repetition=1,
        kind="simulation",
        reason="max_turns_with_incomplete_beat",
    )
    ok = make_dialogue_log(
        [make_turn_record()],
        scenario_id="happy_path_03",
        agent="baseline",
        status="ok",
        stop_reason="goal_reached",
    )
    run_dir = write_run(tmp_path / "run", [retry_a, retry_b, max_turns, ok])

    report = discover_instrument_failures(run_dir)

    assert [item.run_id for item in report.candidates] == [
        "edge_16__fsm__rep02",
        "happy_path_01__baseline__rep01",
    ]
    assert all(item.needs_adjudication is True for item in report.candidates)
    census = {(row.failure_kind, row.failure_reason): row for row in report.census}
    assert census[("instrument", "invalid_candidate_retry_exhausted")].count == 2
    assert census[
        ("instrument", "invalid_candidate_retry_exhausted")
    ].target_adjudication
    assert census[("simulation", "max_turns_with_incomplete_beat")].count == 1
    assert not census[
        ("simulation", "max_turns_with_incomplete_beat")
    ].target_adjudication
    max_ids = {item.run_id for item in report.candidates}
    assert "happy_path_02__fsm__rep01" not in max_ids
    assert "happy_path_03__baseline__rep01" not in max_ids


def test_candidates_use_full_run_ids_and_relative_source(tmp_path: Path) -> None:
    log = _failed(scenario_id="edge_16", agent="baseline", repetition=2, seed=99)
    run_dir = write_run(tmp_path / "run", [log])

    report = discover_instrument_failures(run_dir)

    item = report.candidates[0]
    assert item.run_id == "edge_16__baseline__rep02"
    assert item.scenario_id == "edge_16"
    assert item.agent == "baseline"
    assert item.repetition == 2
    assert item.seed == 99
    assert item.status == "failed"
    assert item.failure_kind == "instrument"
    assert item.failure_reason == "invalid_candidate_retry_exhausted"
    assert item.source == "dialogues/edge_16__baseline__rep02.jsonl"


def test_discovery_is_deterministic(tmp_path: Path) -> None:
    logs = [
        _failed(scenario_id="z_last", agent="fsm", repetition=3),
        _failed(scenario_id="a_first", agent="baseline", repetition=1),
    ]
    run_dir = write_run(tmp_path / "run", logs)

    first = write_instrument_failures(run_dir)
    blob = first.read_bytes()
    write_instrument_failures(run_dir)
    assert first.read_bytes() == blob
    report = json.loads(blob.decode())
    assert [row["run_id"] for row in report["candidates"]] == [
        "a_first__baseline__rep01",
        "z_last__fsm__rep03",
    ]
    assert first == instrument_failures_path(run_dir)


def test_discovery_contains_no_semantic_verdict(tmp_path: Path) -> None:
    log = _failed(scenario_id="happy_path_01")
    run_dir = write_run(tmp_path / "run", [log])

    path = write_instrument_failures(run_dir)
    blob = path.read_text(encoding="utf-8")

    for verdict in VERDICTS:
        assert verdict not in blob
    assert "needs_adjudication" in blob


def test_default_output_is_under_run_adjudication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _failed(scenario_id="happy_path_01")
    run_dir = write_run(tmp_path / "run", [log])

    assert discover.main([str(run_dir)]) == 0

    path = run_dir / "adjudication" / "instrument_failures.json"
    assert path.exists()
    out = capsys.readouterr().out
    assert "instrument / invalid_candidate_retry_exhausted: 1" in out


def test_discovery_does_not_mutate_original_jsonls(tmp_path: Path) -> None:
    log = _failed(scenario_id="happy_path_01")
    run_dir = write_run(tmp_path / "run", [log])
    path = next((run_dir / "dialogues").glob("*.jsonl"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    write_instrument_failures(run_dir)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_discovery_implementation_does_not_pin_production_ids() -> None:
    source = (REPO_ROOT / "scripts" / "list_instrument_failures.py").read_text(
        encoding="utf-8"
    )
    library = (REPO_ROOT / "src" / "sim" / "adjudication.py").read_text(
        encoding="utf-8"
    )

    for item in PRODUCTION_IDS:
        assert item not in source
        assert item not in library
