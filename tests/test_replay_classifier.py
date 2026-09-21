"""Tests for scripts/replay_classifier.py."""

from pathlib import Path

import pytest

from helpers import (
    FakeLlm,
    classifier_reply,
    load_labeled_events,
    load_script,
    make_dialogue_log,
    make_turn_record,
    write_run,
)
from sim.fsm import FsmSpec
from sim.kb import KnowledgeBase

replay = load_script("scripts/replay_classifier.py")


def test_replay_pilot_splits_rule_and_llm_and_does_not_call_llm_on_rules(
    tmp_path: Path,
    real_fsm: FsmSpec,
    real_kb: KnowledgeBase,
) -> None:
    log = make_dialogue_log(
        [
            make_turn_record(
                1,
                user_message="Goodbye.",
                agent_reply="Goodbye.",
                state_before="greeting",
                state_after="closing",
                event="farewell",
            ).model_copy(update={"intent": None, "collected": {}}),
            make_turn_record(
                2,
                user_message="I want to know where my package is.",
                agent_reply="I can track that.",
                state_before="intent_classification",
                state_after="data_collection",
                event="intent_classified",
            ).model_copy(
                update={"intent": "order_tracking", "collected": {}},
            ),
            make_turn_record(
                3,
                user_message="What's the weather in Lisbon?",
                agent_reply="That is out of scope.",
                state_before="greeting",
                state_after="out_of_scope",
                event="out_of_scope_request",
            ).model_copy(update={"intent": "order_tracking", "collected": {}}),
        ],
        scenario_id="happy_path_01",
        agent="fsm",
        seed=99,
    )
    source = write_run(
        tmp_path / "source",
        [log],
        scenarios_dir="data/scenarios/examples",
    )
    llm = FakeLlm(
        [
            classifier_reply("intent_classified", "cancellation"),
            classifier_reply("request_received"),
        ]
    )
    target = tmp_path / "sidecar"

    report = replay.replay_pilot(source, target, llm=llm, fsm=real_fsm, kb=real_kb)

    assert report.n_turns == 3
    assert report.n_rule == 1
    assert report.n_llm == 2
    assert report.n_llm_event_agree == 1
    assert report.n_llm_event_change == 1
    assert report.n_intent_classified == 1
    assert report.n_intent_change == 1
    assert llm.calls[0]["seed"] == 99
    prompt = "\n".join(message["content"] for message in llm.calls[0]["messages"])
    assert "Goodbye." in prompt
    assert all(call["role"] == "classifier" for call in llm.calls)
    rows = list(report.turns)
    assert rows[0].via == "rule"
    assert rows[0].event_match is True
    assert rows[1].via == "llm"
    assert rows[1].logged_event == "intent_classified"
    assert rows[1].replayed_intent == "cancellation"
    assert rows[2].via == "llm"
    assert rows[2].event_match is False


def test_replay_pilot_refuses_frozen_pilot_v2_target(
    tmp_path: Path,
    real_fsm: FsmSpec,
    real_kb: KnowledgeBase,
) -> None:
    source = write_run(
        tmp_path / "source",
        [
            make_dialogue_log(
                [
                    make_turn_record(
                        1,
                        user_message="Goodbye.",
                        state_before="greeting",
                        state_after="closing",
                        event="farewell",
                    )
                ],
                scenario_id="happy_path_01",
                agent="fsm",
            )
        ],
        scenarios_dir="data/scenarios/examples",
    )
    frozen = Path("runs/pilot_v2")
    before = (
        (frozen / "manifest.json").read_bytes()
        if (frozen / "manifest.json").exists()
        else None
    )

    with pytest.raises(replay.ReplayError, match="pilot_v2"):
        replay.replay_pilot(source, frozen, llm=FakeLlm([]), fsm=real_fsm, kb=real_kb)

    if before is not None:
        assert (frozen / "manifest.json").read_bytes() == before


def test_replay_fixtures_scores_gold_phrases_with_fake_llm(
    real_fsm: FsmSpec,
    real_kb: KnowledgeBase,
) -> None:
    events = load_labeled_events()
    cases = [row for row in events if row.via == "rule"][:2]
    cases.append(next(row for row in events if row.event == "intent_classified"))
    llm_cases = [row for row in cases if row.via == "llm"]
    llm = FakeLlm([classifier_reply(row.event, row.intent) for row in llm_cases])

    report = replay.score_fixtures(cases, llm=llm, fsm=real_fsm, kb=real_kb)

    assert report.n == len(cases)
    assert report.n_correct == len(cases)
    assert report.accuracy == 1.0
    assert len(llm.calls) == len(llm_cases)
