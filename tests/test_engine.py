"""Tests for the FSM engine of T-08, on the real machine."""

import pytest

from helpers import ORDER_AND_EMAIL, PROMPTS_DIR, FakeLlm, classifier_reply
from sim.engine import NONE, FsmEngine
from sim.events import patterned_keys_required_for_every_intent
from sim.fsm import FsmSpec, Transition
from sim.kb import KnowledgeBase

#: One message carrying both identifying data, so the rules alone can walk the
#: engine several states forward without the classifier being asked anything.
IDENTIFIED = "My order is NL-20260145 and the email is jane@example.com."


@pytest.fixture
def fsm_engine(real_fsm: FsmSpec, real_kb: KnowledgeBase) -> FsmEngine:
    """A fresh engine on the real machine; mutable, so one per test."""
    return FsmEngine(real_fsm, kb=real_kb, prompts_dir=PROMPTS_DIR)


def test_engine_starts_in_the_initial_state(
    fsm_engine: FsmEngine, real_fsm: FsmSpec
) -> None:
    assert fsm_engine.state == real_fsm.initial


def test_every_real_transition_fires_when_its_guard_is_met(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    for transition in real_fsm.transitions:
        engine = FsmEngine(real_fsm, kb=real_kb)
        _arm(engine, transition, real_kb)

        record = engine.apply(transition.event, turn=1)

        assert record.valid, (transition.source, transition.event)
        assert record.source == transition.source
        assert record.dest == transition.dest
        assert record.event == transition.event
        assert record.fired_by == "user"
        assert engine.state == transition.dest
        assert engine.history[-1] == record


def settled_in_intent_classification(
    spec: FsmSpec, kb: KnowledgeBase, intent: str = "order_tracking"
) -> FsmEngine:
    """An engine one classifier answer away from the solution of ``intent``.

    Parked where the request is settled, with the identifying data already
    collected, so the next turn walks the user's own edge and then the one the
    engine fires because ``data_collection`` has nothing left to ask.
    """
    llm = FakeLlm([classifier_reply("intent_classified", intent)])
    engine = FsmEngine(spec, kb=kb, llm=llm, prompts_dir=PROMPTS_DIR)
    engine.park("intent_classification")
    engine.collected.update(ORDER_AND_EMAIL)
    return engine


def test_a_turn_records_every_edge_it_walks_and_who_fired_it(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    engine = settled_in_intent_classification(real_fsm, real_kb)

    walk = engine.step("I want to know where my package is.", turn=1)

    assert [(edge.source, edge.event, edge.dest, edge.fired_by) for edge in walk] == [
        ("intent_classification", "intent_classified", "data_collection", "user"),
        ("data_collection", "data_provided", "solution", "engine"),
    ]
    assert all(edge.turn == 1 and edge.valid for edge in walk)
    assert list(engine.history) == list(walk)


def test_every_walked_edge_is_a_transition_of_the_machine(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    edges = {(edge.source, edge.event, edge.dest) for edge in real_fsm.transitions}
    engine = settled_in_intent_classification(real_fsm, real_kb)

    walk = engine.step("I want to know where my package is.", turn=1)

    assert all((edge.source, edge.event, edge.dest) in edges for edge in walk)
    assert [edge.source for edge in walk[1:]] == [edge.dest for edge in walk[:-1]]


def test_a_request_already_named_still_passes_through_intent_classification(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm([classifier_reply("request_received", "order_tracking")])
    engine = FsmEngine(real_fsm, kb=real_kb, llm=llm, prompts_dir=PROMPTS_DIR)

    engine.step(f"Where is my order? {IDENTIFIED}", turn=1)

    assert engine.intent == "order_tracking"
    assert engine.state == "intent_classification"


def test_identification_completed_in_the_opening_message_does_not_wait(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm([classifier_reply("request_received")])
    engine = FsmEngine(real_fsm, kb=real_kb, llm=llm, prompts_dir=PROMPTS_DIR)

    walk = engine.step(f"Hi, where is my order? {IDENTIFIED}", turn=1)

    assert [(edge.event, edge.fired_by) for edge in walk] == [
        ("request_received", "user"),
        ("order_identified", "engine"),
    ]
    assert engine.state == "intent_classification"


def test_entering_intent_classification_with_a_known_intent_does_not_auto_advance(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm([classifier_reply("request_received", "cancellation")])
    engine = FsmEngine(real_fsm, kb=real_kb, llm=llm, prompts_dir=PROMPTS_DIR)

    walk = engine.step(
        "I want to cancel my order. "
        "My order is NL-20260145 and the email is jane@example.com.",
        turn=1,
    )

    assert [(edge.event, edge.fired_by) for edge in walk] == [
        ("request_received", "user"),
        ("order_identified", "engine"),
    ]
    assert engine.intent == "cancellation"
    assert engine.state == "intent_classification"


def test_a_later_none_cannot_leave_intent_classification_parked_with_a_known_intent(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm(
        [
            classifier_reply("request_received", "cancellation"),
            classifier_reply("none"),
        ]
    )
    engine = FsmEngine(real_fsm, kb=real_kb, llm=llm, prompts_dir=PROMPTS_DIR)

    engine.step(
        "I want to cancel my order. "
        "My order is NL-20260145 and the email is jane@example.com.",
        turn=1,
    )
    walk = engine.step("Yes, that's what I need.", turn=2)

    assert walk[0].event == NONE
    assert walk[0].fired_by == "user"
    assert walk[1].event == "intent_classified"
    assert walk[1].fired_by == "engine"
    assert walk[1].source == "intent_classification"
    assert walk[1].dest == "data_collection"
    assert engine.state == "data_collection"


@pytest.mark.parametrize(
    ("state", "event", "blocked_by_guard"),
    [
        ("greeting", "user_confirmed", False),
        ("identification", "order_identified", True),
        ("greeting", NONE, False),
    ],
    ids=["not_accepted", "guard_fails", "none"],
)
def test_a_rejected_event_stays_and_is_invalid(
    fsm_engine: FsmEngine, state: str, event: str, blocked_by_guard: bool
) -> None:
    fsm_engine.park(state)

    record = fsm_engine.apply(event, turn=1)

    assert record.valid is False
    assert record.source == state
    assert record.dest == state
    assert record.event == event
    assert record.blocked_by_guard is blocked_by_guard
    assert fsm_engine.state == state
    assert fsm_engine.history[-1] == record


def _arm(engine: FsmEngine, transition: Transition, kb: KnowledgeBase) -> None:
    """Park on the edge's source and fill whatever its guard reads."""
    engine.park(transition.source)
    if transition.guard == "order_and_email_present":
        examples = {field.key: field.example for field in kb.user_data_fields}
        for key in patterned_keys_required_for_every_intent(
            kb.user_data_fields, kb.intents()
        ):
            engine.collected[key] = examples[key]
        return
    if transition.guard != "required_data_collected":
        return
    engine.intent = "order_tracking"
    for field in kb.user_data_fields:
        if "order_tracking" in field.required_for:
            engine.collected[field.key] = field.example
