"""Tests for the FSM engine of T-08, on the real machine."""

import pytest

from sim.engine import NONE, FsmEngine
from sim.events import patterned_keys_required_for_every_intent
from sim.fsm import FsmSpec, Transition
from sim.kb import KnowledgeBase


@pytest.fixture
def fsm_engine(real_fsm: FsmSpec, real_kb: KnowledgeBase) -> FsmEngine:
    """A fresh engine on the real machine; mutable, so one per test."""
    return FsmEngine(real_fsm, kb=real_kb)


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
        assert engine.state == transition.dest
        assert engine.history[-1] == record


@pytest.mark.parametrize(
    ("state", "event"),
    [
        ("greeting", "user_confirmed"),
        ("identification", "order_identified"),
        ("greeting", NONE),
    ],
    ids=["not_accepted", "guard_fails", "none"],
)
def test_a_rejected_event_stays_and_is_invalid(
    fsm_engine: FsmEngine, state: str, event: str
) -> None:
    fsm_engine.park(state)

    record = fsm_engine.apply(event, turn=1)

    assert record.valid is False
    assert record.source == state
    assert record.dest == state
    assert record.event == event
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
