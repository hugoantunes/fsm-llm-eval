"""Tests for the hybrid user-event detector of T-08."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from helpers import PROMPTS_DIR, FakeLlm, LabeledEvent, load_labeled_events
from sim.engine import NONE, FsmEngine
from sim.events import CLASSIFIER, Detection, EventError, detect_user_event
from sim.fsm import FsmSpec
from sim.kb import KnowledgeBase, UserDataField
from sim.llm import LlmClient

RULE_EVENTS = [row for row in load_labeled_events() if row.via == "rule"]

TRACKING = "I want to track my order"
INTENT_REPLY = '{"event": "intent_classified", "intent": "order_tracking"}'
_ORDER_AND_EMAIL = {
    "order_number": "NL-20260145",
    "email": "jane@example.com",
}


@dataclass
class Classified:
    """One LLM-fallback step, shared by the schema, prompt and intent tests."""

    engine: FsmEngine
    llm: FakeLlm


def _engine(
    spec: FsmSpec,
    kb: KnowledgeBase,
    llm: FakeLlm | LlmClient | None = None,
    *,
    seed: int | None = None,
) -> FsmEngine:
    """Build an engine that loads prompts from the repo, not from cwd."""
    return FsmEngine(spec, kb=kb, llm=llm, prompts_dir=PROMPTS_DIR, seed=seed)


@pytest.fixture
def classified(real_fsm: FsmSpec, real_kb: KnowledgeBase) -> Classified:
    """Park in intent_classification and classify a message the rules miss."""
    llm = FakeLlm([INTENT_REPLY])
    engine = _engine(real_fsm, real_kb, llm)
    engine.park("intent_classification")
    engine.step(TRACKING, turn=1)
    return Classified(engine=engine, llm=llm)


@pytest.fixture(scope="session")
def labeled_events() -> list[LabeledEvent]:
    """The gold utterances of tests/fixtures/user_events.jsonl."""
    return load_labeled_events()


@pytest.mark.parametrize(
    "case",
    RULE_EVENTS,
    ids=lambda case: f"{case.state}-{case.event}-{_slug(case.text)}",
)
def test_a_rule_fires_without_the_llm(
    case: LabeledEvent, real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm([])
    engine = _engine(real_fsm, real_kb, llm)
    engine.park(case.state)

    record = engine.step(case.text, turn=1)

    assert record.event == case.event
    assert llm.calls == []


def test_llm_fallback_uses_the_current_states_enum_plus_none(
    classified: Classified, real_fsm: FsmSpec
) -> None:
    call = classified.llm.calls[0]
    schema = call["schema"]

    assert call["role"] == "simulator"
    assert call["caller"] == CLASSIFIER
    assert set(schema.model_json_schema()["properties"]["event"]["enum"]) == set(
        real_fsm.events_for("intent_classification")
    ) | {NONE}


def test_classifier_prompt_carries_no_knowledge_base(
    classified: Classified, real_kb: KnowledgeBase
) -> None:
    prompt = "\n".join(
        message["content"] for message in classified.llm.calls[0]["messages"]
    )

    assert all(fact.text not in prompt for fact in real_kb.facts)


def test_intent_classified_stores_the_intent_on_the_engine(
    classified: Classified,
) -> None:
    assert classified.engine.intent == "order_tracking"
    assert classified.engine.history[-1].event == "intent_classified"


def test_intent_classified_without_intent_raises(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm(['{"event": "intent_classified", "intent": null}'])
    engine = _engine(real_fsm, real_kb, llm)
    engine.park("intent_classification")

    with pytest.raises(EventError, match="intent_classification"):
        engine.step(TRACKING, turn=1)

    assert engine.state == "intent_classification"
    assert engine.intent is None


def test_a_farewell_wins_over_slots_already_held(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm([])
    engine = _engine(real_fsm, real_kb, llm)
    engine.park("identification")
    engine.collected.update(_ORDER_AND_EMAIL)

    record = engine.step("Goodbye.", turn=1)

    assert record.event == "farewell"
    assert record.dest == "closing"
    assert llm.calls == []


def test_a_corrected_slot_in_data_collection_fires_without_the_llm(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm([])
    engine = _engine(real_fsm, real_kb, llm)
    engine.park("data_collection")
    engine.intent = "order_tracking"
    engine.collected.update(_ORDER_AND_EMAIL)

    record = engine.step("The order is NL-20260199.", turn=1)

    assert record.event == "data_provided"
    assert engine.collected["order_number"] == "NL-20260199"
    assert llm.calls == []


@pytest.mark.parametrize(
    ("intent", "message", "reply", "expected_slots"),
    [
        (
            "cancellation",
            "The reason is I changed my mind.",
            '{"event": "data_provided", "intent": null, "reason": "I changed my mind"}',
            {"reason": "I changed my mind"},
        ),
        (
            "exchange_return",
            "The blue jacket, too small, I want an exchange.",
            '{"event": "data_provided", "intent": null, "item": "the blue jacket",'
            ' "reason": "too small", "preferred_resolution": "an exchange"}',
            {
                "item": "the blue jacket",
                "reason": "too small",
                "preferred_resolution": "an exchange",
            },
        ),
    ],
    ids=["cancellation", "exchange_return"],
)
def test_patternless_slots_from_the_classifier_leave_data_collection(
    intent: str,
    message: str,
    reply: str,
    expected_slots: dict[str, str],
    real_fsm: FsmSpec,
    real_kb: KnowledgeBase,
) -> None:
    llm = FakeLlm([reply])
    engine = _engine(real_fsm, real_kb, llm)
    engine.park("data_collection")
    engine.intent = intent
    engine.collected.update(_ORDER_AND_EMAIL)

    record = engine.step(message, turn=1)

    assert record.event == "data_provided"
    assert record.dest == "solution"
    assert engine.collected.items() >= expected_slots.items()


@pytest.mark.integration
def test_real_classifier_accuracy_on_labeled_phrases(
    labeled_events: list[LabeledEvent],
    real_fsm: FsmSpec,
    real_kb: KnowledgeBase,
    real_client: LlmClient,
) -> None:
    misses: list[str] = []
    for case in labeled_events:
        engine = _engine(real_fsm, real_kb, real_client, seed=42)
        engine.park(case.state)
        record = engine.step(case.text, turn=1)
        matched = record.event == case.event
        if matched and case.event == "intent_classified":
            matched = engine.intent == case.intent
        if not matched:
            misses.append(
                f"{case.state}: {case.text!r} → {record.event}"
                f"{f'/{engine.intent}' if engine.intent else ''} "
                f"(want {case.event}"
                f"{f'/{case.intent}' if case.intent else ''})"
            )

    accuracy = 1 - len(misses) / len(labeled_events)
    print(f"\nclassifier accuracy: {accuracy:.0%} ({len(misses)} miss)")
    for miss in misses:
        print(miss)
    assert accuracy >= 0.85, misses


def test_engine_loads_the_classifier_prompt_when_cwd_is_not_the_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_fsm: FsmSpec,
    real_kb: KnowledgeBase,
) -> None:
    monkeypatch.chdir(tmp_path)
    llm = FakeLlm([INTENT_REPLY])
    engine = _engine(real_fsm, real_kb, llm)
    engine.park("intent_classification")

    engine.step(TRACKING, turn=1)

    assert engine.intent == "order_tracking"


def test_identification_rule_follows_the_guarded_edge(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    spec = _rename_guarded_edge(
        real_fsm,
        "order_and_email_present",
        source="intake",
        event="customer_identified",
    )
    llm = FakeLlm([])

    detection = _detect(
        "My order is NL-20260145 and the email is jane@example.com.",
        state="intake",
        spec=spec,
        kb=real_kb,
        llm=llm,
    )

    assert detection.event == "customer_identified"
    assert llm.calls == []


def test_data_collection_rule_follows_the_guarded_edge(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    spec = _rename_guarded_edge(
        real_fsm, "required_data_collected", source="gathering", event="slots_updated"
    )
    llm = FakeLlm([])

    detection = _detect(
        "The order is NL-20260145.",
        state="gathering",
        spec=spec,
        kb=real_kb,
        collected={"order_number": "NL-20260100", "email": "jane@example.com"},
        intent="order_tracking",
        llm=llm,
    )

    assert detection.event == "slots_updated"
    assert llm.calls == []


def test_detect_raises_when_a_guarded_edge_is_missing(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    spec = real_fsm.model_copy(
        update={"transitions": [t for t in real_fsm.transitions if t.guard is None]}
    )

    with pytest.raises(EventError, match="order_and_email_present"):
        _detect("Hi.", state="greeting", spec=spec, kb=real_kb)


def test_detect_raises_when_farewell_is_not_in_the_spec(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    spec = real_fsm.model_copy(
        update={
            "transitions": [t for t in real_fsm.transitions if t.event != "farewell"]
        }
    )

    with pytest.raises(EventError, match="farewell"):
        _detect("Hi.", state="greeting", spec=spec, kb=real_kb)


def test_identification_rule_uses_patterned_fields_every_intent_requires(
    real_fsm: FsmSpec, real_kb: KnowledgeBase
) -> None:
    fields = [
        field.model_copy(update={"required_for": ["order_tracking"]})
        if field.key == "email"
        else field
        for field in real_kb.user_data_fields
    ]
    llm = FakeLlm([])

    detection = _detect(
        "My order is NL-20260145.",
        state="identification",
        spec=real_fsm,
        kb=real_kb,
        fields=fields,
        llm=llm,
    )

    assert detection.event == "order_identified"
    assert llm.calls == []


def _detect(
    message: str,
    *,
    state: str,
    spec: FsmSpec,
    kb: KnowledgeBase,
    collected: dict[str, str] | None = None,
    fields: Sequence[UserDataField] | None = None,
    intent: str | None = None,
    llm: FakeLlm | None = None,
) -> Detection:
    return detect_user_event(
        message,
        state=state,
        spec=spec,
        collected={} if collected is None else collected,
        fields=kb.user_data_fields if fields is None else fields,
        intents=kb.intents(),
        intent=intent,
        llm=FakeLlm([]) if llm is None else llm,
        transcript=message,
        prompts_dir=PROMPTS_DIR,
    )


def _rename_guarded_edge(
    spec: FsmSpec, guard: str, *, source: str, event: str
) -> FsmSpec:
    """Copy the spec with the guarded edge's from-state and event renamed."""
    edge = next(item for item in spec.transitions if item.guard == guard)
    states = {**spec.states, source: spec.states[edge.source]}
    transitions = [
        item.model_copy(update={"source": source, "event": event})
        if item.guard == guard
        else item
        for item in spec.transitions
    ]
    return spec.model_copy(update={"states": states, "transitions": transitions})


def _slug(text: str) -> str:
    return re.sub(r"\W+", "-", text.lower()).strip("-")[:24]
