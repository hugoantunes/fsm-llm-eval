"""Tests for the FSM agent (T-10)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from helpers import (
    FSM_DIR,
    PROMPTS_DIR,
    FakeLlm,
    LabeledEvent,
    assert_prompt_carries_no_answer_key,
    classifier_reply,
    load_labeled_events,
    play_user_turns,
)
from sim.agents import FsmAgent
from sim.fsm import FsmSpec
from sim.kb import Fact, KnowledgeBase
from sim.prompts import load_prompt
from sim.schemas import Scenario, Turn, TurnRecord

AGENT_REPLY = "Understood."

MakeFsm = Callable[..., FsmAgent]


@dataclass(frozen=True)
class AgentTurn:
    """One canned turn: the record, the agent system prompt, and the fake."""

    record: TurnRecord
    prompt: str
    llm: FakeLlm


@dataclass(frozen=True)
class Walk:
    """A canned walk through the machine, with every agent call on ``llm``."""

    records: list[TurnRecord]
    llm: FakeLlm


@dataclass(frozen=True)
class ScriptTurn:
    """One step of the canonical tracking walk: gold utterance and how it fires."""

    state: str
    event: str
    intent: str | None = None
    via: str = "llm"


CANONICAL_WALK = (
    ScriptTurn("greeting", "request_received"),
    ScriptTurn("identification", "order_identified", via="rule"),
    ScriptTurn("intent_classification", "intent_classified", intent="order_tracking"),
    ScriptTurn("data_collection", "data_provided"),
    ScriptTurn("solution", "solution_accepted"),
    ScriptTurn("confirmation", "user_confirmed"),
)


@pytest.fixture(scope="session")
def labeled_events() -> list[LabeledEvent]:
    """The gold utterances of ``tests/fixtures/user_events.jsonl``."""
    return load_labeled_events()


@pytest.fixture
def fsm_agent(real_kb: KnowledgeBase, real_fsm: FsmSpec) -> MakeFsm:
    """Build an FSM agent on the real machine, KB and prompt files."""

    def _make(llm: FakeLlm, **kwargs: object) -> FsmAgent:
        return FsmAgent(
            llm,
            kb=real_kb,
            spec=real_fsm,
            fsm_dir=FSM_DIR,
            prompts_dir=PROMPTS_DIR,
            **kwargs,
        )

    return _make


@pytest.fixture
def greeting_none(fsm_agent: MakeFsm, labeled_events: list[LabeledEvent]) -> AgentTurn:
    """One ``Hello.`` turn that stays in greeting, classified as ``none``."""
    hello = labeled_utterance(labeled_events, "greeting", "none")
    llm = FakeLlm([classifier_reply("none"), AGENT_REPLY])
    record = fsm_agent(llm).respond([Turn(speaker="user", text=hello.text)])
    return AgentTurn(record=record, prompt=last_agent_prompt(llm), llm=llm)


@pytest.fixture
def out_of_scope_turn(
    fsm_agent: MakeFsm, labeled_events: list[LabeledEvent]
) -> AgentTurn:
    """One out-of-domain turn from greeting into ``out_of_scope``."""
    weather = labeled_utterance(labeled_events, "greeting", "out_of_scope_request")
    llm = FakeLlm([classifier_reply("out_of_scope_request"), AGENT_REPLY])
    record = fsm_agent(llm).respond([Turn(speaker="user", text=weather.text)])
    return AgentTurn(record=record, prompt=last_agent_prompt(llm), llm=llm)


@pytest.fixture
def closing_walk(fsm_agent: MakeFsm, labeled_events: list[LabeledEvent]) -> Walk:
    """The canonical tracking path from greeting through to closing."""
    replies: list[str] = []
    messages: list[str] = []
    for step in CANONICAL_WALK:
        gold = labeled_utterance(
            labeled_events, step.state, step.event, intent=step.intent
        )
        messages.append(gold.text)
        if step.via == "llm":
            replies.append(classifier_reply(step.event, step.intent))
        replies.append(AGENT_REPLY)
    llm = FakeLlm(replies)
    return Walk(records=play_user_turns(fsm_agent(llm), messages), llm=llm)


def test_a_none_turn_answers_from_greeting(
    greeting_none: AgentTurn,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    example_scenarios: dict[str, Scenario],
) -> None:
    prompt = greeting_none.prompt
    record = greeting_none.record
    greeting = package_text(real_fsm, "greeting")
    preferred = field_label(real_kb, "preferred_resolution")

    assert load_prompt("agent_shared", directory=PROMPTS_DIR).template in prompt
    assert greeting in prompt
    assert all(
        fact_of(real_kb, fact_id).text in prompt
        for fact_id in ("F01", "F02", "F03", "F04")
    )
    assert fact_of(real_kb, "F09").text not in prompt
    assert "# State: identification" not in prompt
    assert preferred not in prompt
    assert_prompt_carries_no_answer_key(prompt, example_scenarios["adversarial_01"])
    assert record.state_before == "greeting"
    assert record.state_after == "greeting"
    assert record.event == "none"
    assert last_agent_call(greeting_none.llm)["caller"] == "fsm"


def test_a_scripted_dialogue_walks_greeting_to_closing(
    closing_walk: Walk, real_kb: KnowledgeBase
) -> None:
    records = closing_walk.records
    prompts = [call["messages"][0]["content"] for call in agent_calls(closing_walk.llm)]
    collection = [record.state_after for record in records].index("data_collection")
    tracking_fields = [
        field
        for field in real_kb.user_data_fields
        if "order_tracking" in field.required_for
    ]

    assert [record.state_after for record in records] == [
        "identification",
        "intent_classification",
        "data_collection",
        "solution",
        "confirmation",
        "closing",
    ]
    assert tracking_fields
    assert all(field.label in prompts[collection] for field in tracking_fields)


def test_out_of_scope_does_not_carry_another_states_facts(
    out_of_scope_turn: AgentTurn,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    prompt = out_of_scope_turn.prompt

    assert out_of_scope_turn.record.state_after == "out_of_scope"
    assert package_text(real_fsm, "out_of_scope") in prompt
    assert all(
        fact_of(real_kb, fact_id).text in prompt for fact_id in ("F01", "F03", "F04")
    )
    assert fact_of(real_kb, "F09").text not in prompt
    assert fact_of(real_kb, "F18").text not in prompt
    assert "Give the outcome in the first sentence" not in prompt


def labeled_utterance(
    events: Sequence[LabeledEvent],
    state: str,
    event: str,
    *,
    intent: str | None = None,
) -> LabeledEvent:
    """Return the first gold line for ``state`` / ``event`` / ``intent``."""
    matches = [
        row
        for row in events
        if row.state == state
        and row.event == event
        and (intent is None or row.intent == intent)
    ]
    assert matches, f"no gold utterance for {state}/{event}/{intent}"
    return matches[0]


def package_text(spec: FsmSpec, state: str) -> str:
    """Read the instruction package of ``state`` from the real FSM directory."""
    return (FSM_DIR / spec.states[state].package).read_text(encoding="utf-8")


def fact_of(kb: KnowledgeBase, fact_id: str) -> Fact:
    """Return the KB fact with ``fact_id``."""
    return next(fact for fact in kb.facts if fact.id == fact_id)


def field_label(kb: KnowledgeBase, key: str) -> str:
    """Return the label of the user-data field ``key``."""
    return next(field.label for field in kb.user_data_fields if field.key == key)


def agent_calls(llm: FakeLlm) -> list[dict[str, Any]]:
    """Return the agent (non-schema) calls recorded on ``llm``, in order."""
    return [call for call in llm.calls if call["schema"] is None]


def last_agent_call(llm: FakeLlm) -> dict[str, Any]:
    """Return the last agent (non-schema) call recorded on ``llm``."""
    calls = agent_calls(llm)
    assert calls, "FakeLlm recorded no agent call"
    return calls[-1]


def last_agent_prompt(llm: FakeLlm) -> str:
    """Return the system prompt of the last agent call on ``llm``."""
    return last_agent_call(llm)["messages"][0]["content"]
