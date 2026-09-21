"""Failed-dialogue audit: generated evidence, never an inclusion verdict."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from helpers import (
    REPO_ROOT,
    load_script,
    make_dialogue_log,
    make_llm_call_record,
    make_turn_record,
    write_llm_calls,
    write_run,
    write_scenarios,
)
from sim.adjudication import contract_false_positives_path
from sim.schemas import (
    DialogueLog,
    Scenario,
    TurnRecord,
    as_messages,
    transcript_from_records,
)
from sim.user import UserReply

audit = load_script("scripts/audit_failed_dialogues.py")
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
VERDICT_STRINGS = ("CONTRACT_FALSE_POSITIVE", "TRUE_INSTRUMENT_FAILURE")


def _scenario(*script: str, scenario_id: str = "happy_path_80") -> Scenario:
    """A valid order-tracking scenario whose script is the contract under test."""
    return Scenario(
        id=scenario_id,
        category="happy_path",
        intent="order_tracking",
        user_persona=f"A customer in {scenario_id}.",
        user_goal=f"Track the order for {scenario_id}.",
        script=list(script),
        reference_answer="Standard delivery takes 5 business days.",
        required_facts=["F09"],
        expected_final_state="closing",
        success_criterion="The estimate is provided.",
        max_turns=6,
    )


def _failed(
    records: Sequence[TurnRecord],
    scenario: Scenario,
    *,
    kind: str = "instrument",
    reason: str = "invalid_candidate_retry_exhausted",
    agent: str = "baseline",
    repetition: int = 1,
    seed: int = 42,
    invalid_reason: str = "beat 2: incomplete requirements still owed",
    extra_metadata: dict[str, object] | None = None,
) -> DialogueLog:
    """One failed log with the given turns and runtime class."""
    log = make_dialogue_log(
        records,
        scenario_id=scenario.id,
        agent=agent,
        repetition=repetition,
        seed=seed,
        status="failed",
        stop_reason=None,
    )
    metadata: dict[str, object] = {"invalid_reason": invalid_reason}
    if extra_metadata:
        metadata.update(extra_metadata)
    return log.model_copy(
        update={
            "failure_kind": kind,
            "failure_reason": reason,
            "failure_metadata": metadata,
        }
    )


def _turn(
    turn: int,
    message: str,
    *,
    beat: int | None = None,
    start: int | None = None,
    end: int | None = None,
    agent_reply: str = "I can help with that.",
) -> TurnRecord:
    """One committed turn the experimental agent actually received."""
    return make_turn_record(
        turn,
        user_message=message,
        agent_reply=agent_reply,
        user_beat=beat,
        beat_started_at_turn=start,
        beat_completed_at_turn=end,
    )


def _prepare_run(
    tmp_path: Path,
    *logs: DialogueLog,
    scenarios: Sequence[Scenario],
) -> Path:
    """Write a synthetic run whose scenarios live under ``tmp_path``."""
    scenario_dir = write_scenarios(
        tmp_path / "scenarios",
        cases=[item.model_dump(mode="json") for item in scenarios],
    )
    return write_run(tmp_path / "run", logs, scenarios_dir=str(scenario_dir))


def _user_history(records: Sequence[TurnRecord]) -> list[dict[str, str]]:
    """Non-system messages of a simulated-user call after these committed turns."""
    return as_messages("unused", transcript_from_records(records), speaking_as="user")[
        1:
    ]


def _simulated_user_call(
    scenario: Scenario,
    records: Sequence[TurnRecord],
    reply: UserReply,
    *,
    timestamp: str = "2026-09-14T00:00:00+00:00",
    prompt_hash: str = "ab",
) -> object:
    """One persisted simulated-user candidate after ``records``."""
    messages = [
        {
            "role": "system",
            "content": f"{scenario.user_persona}\n{scenario.user_goal}",
        },
        *_user_history(records),
    ]
    return make_llm_call_record(
        caller="simulated_user",
        role="simulated_user",
        prompt_hash=prompt_hash,
    ).model_copy(
        update={
            "timestamp": timestamp,
            "messages": messages,
            "text": reply.model_dump_json(),
        }
    )


def _mechanisms(entry: object) -> set[str]:
    """Mechanism names flagged on one dialogue audit."""
    return {item["mechanism"] for item in entry.possible_mechanisms}


def test_future_beat_content_is_flagged_when_earlier_turns_hold_a_later_requirement(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "Where is order NL-20260145?",
        "My email is irene.pohl@example.com",
        scenario_id="happy_path_80",
    )
    log = _failed(
        [
            _turn(
                1,
                "Where is order NL-20260145? Also irene.pohl@example.com",
                beat=1,
                start=1,
                end=1,
            ),
            _turn(2, "Thanks, that is all.", beat=None),
        ],
        scenario,
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    assert "possible_future_beat_content" in _mechanisms(entry)
    evidence = next(
        item
        for item in entry.possible_mechanisms
        if item["mechanism"] == "possible_future_beat_content"
    )
    assert evidence["evidence"][0]["sent_to_agent"] is True
    assert evidence["evidence"][0]["credited_to_active_beat"] is False


def test_later_beat_content_is_flagged_while_an_earlier_beat_is_still_open(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "Where is order NL-20260145?",
        "My email is pat@example.com",
        scenario_id="happy_path_94",
    )
    log = _failed(
        [_turn(1, "Where is it? Also pat@example.com", beat=None)],
        scenario,
        invalid_reason="beat 1: incomplete requirements still owed",
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    assert "possible_future_beat_content" in _mechanisms(entry)
    flag = next(
        item
        for item in entry.possible_mechanisms
        if item["mechanism"] == "possible_future_beat_content"
    )
    assert any("later beat" in str(row["requirement"]) for row in flag["evidence"])


def test_earlier_question_with_current_pending_question_is_nonsticky(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "Where is order NL-20260145? My email is pat@example.com",
        scenario_id="happy_path_81",
    )
    log = _failed(
        [
            _turn(1, "Where is order NL-20260145?", beat=None),
            _turn(2, "pat@example.com", beat=None),
        ],
        scenario,
        invalid_reason="beat 1: incomplete requirements still owed",
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    assert "possible_nonsticky_question" in _mechanisms(entry)
    flag = next(
        item
        for item in entry.possible_mechanisms
        if item["mechanism"] == "possible_nonsticky_question"
    )
    assert flag["pending_now"] is True
    assert 1 in flag["seen_earlier_on_turns"]


def test_earlier_denial_with_current_pending_denial_is_nonsticky(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "Cancel is not what I want for NL-20260145.",
        scenario_id="happy_path_82",
    )
    log = _failed(
        [
            _turn(1, "I do not want a cancel.", beat=None),
            _turn(2, "The order is NL-20260145.", beat=None),
        ],
        scenario,
        invalid_reason="beat 1: incomplete requirements still owed",
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    assert "possible_nonsticky_denial" in _mechanisms(entry)
    flag = next(
        item
        for item in entry.possible_mechanisms
        if item["mechanism"] == "possible_nonsticky_denial"
    )
    assert flag["pending_now"] is True
    assert 1 in flag["seen_earlier_on_turns"]


def test_exact_string_case_mismatch_is_flagged_without_fuzzy_matching(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "My email is irene.pohl@example.com for NL-20260145.",
        scenario_id="happy_path_83",
    )
    log = _failed(
        [_turn(1, "My email is Irene.pohl@example.com for NL-20260145.", beat=None)],
        scenario,
        invalid_reason="beat 1: incomplete requirements still owed",
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    assert "possible_exact_string_case_mismatch" in _mechanisms(entry)
    flag = next(
        item
        for item in entry.possible_mechanisms
        if item["mechanism"] == "possible_exact_string_case_mismatch"
    )
    match = flag["evidence"][0]
    assert match["exact_match"] is False
    assert match["case_insensitive_match"] is True
    assert "possible_required_content_absent" not in _mechanisms(entry)


def test_genuinely_absent_cumulative_requirement_is_flagged(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "Please cancel in my name for NL-20260145.",
        scenario_id="happy_path_84",
    )
    log = _failed(
        [_turn(1, "Please cancel NL-20260145.", beat=None)],
        scenario,
        invalid_reason="beat 1: incomplete requirements still owed",
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    assert "possible_required_content_absent" in _mechanisms(entry)
    flag = next(
        item
        for item in entry.possible_mechanisms
        if item["mechanism"] == "possible_required_content_absent"
    )
    assert any("name" in str(item) for item in flag["evidence"])
    assert "possible_future_beat_content" not in _mechanisms(entry)


def test_recovered_invalid_retry_candidate_is_reported_with_inferred_seed(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "Where is order NL-20260145?",
        scenario_id="happy_path_85",
    )
    records = [
        _turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1),
    ]
    log = _failed(records, scenario, seed=100)
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])
    write_llm_calls(
        run_dir,
        [
            _simulated_user_call(
                scenario,
                records,
                UserReply(
                    message="",
                    answering_agent_question=False,
                    status="goal_reached",
                ),
                timestamp="2026-09-14T00:00:01+00:00",
                prompt_hash="c1",
            ),
        ],
    )

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    recovered = entry.retry_recovery["candidates"]
    assert len(recovered) == 1
    assert recovered[0]["attempt_index"] == 0
    assert recovered[0]["seed"] == 100
    assert recovered[0]["seed_source"] == "inferred: dialogue.seed + attempt"
    assert recovered[0]["status"] == "goal_reached"
    assert recovered[0]["message"] == ""
    assert "possible_empty_stop_candidate" in _mechanisms(entry)


def test_unavailable_retry_candidate_is_reported_without_guessing_text(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "Where is order NL-20260145?",
        scenario_id="happy_path_86",
    )
    log = _failed(
        [_turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1)],
        scenario,
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    recovery = entry.retry_recovery
    assert recovery["status"] == "unavailable"
    assert recovery["candidates"] == []
    assert "message" not in json.dumps(recovery)


def test_max_turns_appears_in_census_but_is_not_a_retry_exhausted_case(
    tmp_path: Path,
) -> None:
    retry_scenario = _scenario(
        "Where is order NL-20260145?",
        scenario_id="happy_path_87",
    )
    max_scenario = _scenario(
        "Where is order NL-20260145?",
        "Thanks and goodbye.",
        scenario_id="happy_path_88",
    )
    retry_log = _failed(
        [_turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1)],
        retry_scenario,
        agent="fsm",
        repetition=1,
    )
    max_log = _failed(
        [
            _turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1),
            _turn(2, "Still waiting.", beat=None),
        ],
        max_scenario,
        kind="simulation",
        reason="max_turns_with_incomplete_beat",
        agent="fsm",
        repetition=2,
        invalid_reason="",
        extra_metadata={},
    )
    run_dir = _prepare_run(
        tmp_path, retry_log, max_log, scenarios=[retry_scenario, max_scenario]
    )

    report = audit.audit_run(run_dir)

    max_entry = report.dialogue_map[max_log.dialogue_id]
    retry_entry = report.dialogue_map[retry_log.dialogue_id]
    assert max_entry.failure_kind == "simulation"
    assert max_entry.failure_reason == "max_turns_with_incomplete_beat"
    assert max_entry.retry_recovery["applicable"] is False
    assert retry_entry.failure_reason == "invalid_candidate_retry_exhausted"
    assert retry_entry.retry_recovery["applicable"] is True


def test_audit_does_not_modify_source_jsonls(tmp_path: Path) -> None:
    scenario = _scenario("Where is order NL-20260145?", scenario_id="happy_path_89")
    log = _failed(
        [_turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1)],
        scenario,
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])
    path = next((run_dir / "dialogues").glob("*.jsonl"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    audit.audit_run(run_dir)
    audit.write_reports(audit.audit_run(run_dir), tmp_path / "audit")

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_audit_default_output_is_under_run_adjudication(tmp_path: Path) -> None:
    scenario = _scenario("Where is order NL-20260145?", scenario_id="happy_path_95")
    log = _failed(
        [_turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1)],
        scenario,
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    assert audit.main(["--run", str(run_dir)]) == 0

    assert (run_dir / "adjudication" / "audit.json").exists()
    assert (run_dir / "adjudication" / "audit.md").exists()
    assert not (run_dir / "adjudication" / "contract_false_positives.json").exists()


def test_audit_does_not_write_or_modify_the_frozen_allowlist(tmp_path: Path) -> None:
    scenario = _scenario("Where is order NL-20260145?", scenario_id="happy_path_90")
    log = _failed(
        [_turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1)],
        scenario,
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])
    frozen = contract_false_positives_path(run_dir)
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}\n", encoding="utf-8")
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    out_dir = audit.default_out_dir(run_dir)

    audit.write_reports(audit.audit_run(run_dir), out_dir)

    written = {path.name for path in out_dir.iterdir()}
    assert {"audit.json", "audit.md"} <= written
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before


def test_audit_output_contains_no_automatic_inclusion_verdict(tmp_path: Path) -> None:
    scenario = _scenario("Where is order NL-20260145?", scenario_id="happy_path_91")
    log = _failed(
        [_turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1)],
        scenario,
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])
    out_dir = tmp_path / "audit"

    report = audit.audit_run(run_dir)
    audit.write_reports(report, out_dir)
    blob = (out_dir / "audit.json").read_text(encoding="utf-8") + (
        out_dir / "audit.md"
    ).read_text(encoding="utf-8")

    for verdict in VERDICT_STRINGS:
        assert verdict not in blob
    assert "Needs adjudication" in (out_dir / "audit.md").read_text(encoding="utf-8")
    assert (
        "Audit flags are evidence-retrieval heuristics, not semantic classifiers."
        in (out_dir / "audit.md").read_text(encoding="utf-8")
    )
    assert report.dialogue_map[log.dialogue_id].review_status == "needs_adjudication"


def test_audit_implementation_does_not_pin_production_dialogue_ids() -> None:
    source = (REPO_ROOT / "scripts" / "audit_failed_dialogues.py").read_text(
        encoding="utf-8"
    )

    for item in PRODUCTION_IDS:
        assert item not in source


def test_ok_dialogues_are_omitted_from_the_failed_census(tmp_path: Path) -> None:
    scenario = _scenario("Where is order NL-20260145?", scenario_id="happy_path_92")
    failed = _failed(
        [_turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1)],
        scenario,
        agent="fsm",
    )
    ok = make_dialogue_log(
        [_turn(1, "Where is order NL-20260145?", beat=1, start=1, end=1)],
        scenario_id=scenario.id,
        agent="baseline",
        status="ok",
        stop_reason="goal_reached",
    )
    run_dir = _prepare_run(tmp_path, failed, ok, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    assert failed.dialogue_id in report.dialogue_map
    assert ok.dialogue_id not in report.dialogue_map


def test_shipment_phrase_that_does_not_fire_the_frozen_detector_is_a_gap(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        "I do not want this cancelled; the order is NL-20260145.",
        scenario_id="happy_path_93",
    )
    log = _failed(
        [_turn(1, "Please cancel NL-20260145 before anything shipped.", beat=None)],
        scenario,
        invalid_reason="beat 1: incomplete requirements still owed",
    )
    run_dir = _prepare_run(tmp_path, log, scenarios=[scenario])

    report = audit.audit_run(run_dir)

    entry = report.dialogue_map[log.dialogue_id]
    assert "possible_denial_detector_gap" in _mechanisms(entry)
