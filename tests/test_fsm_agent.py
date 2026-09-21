"""Tests for the FSM agent (T-10)."""

from collections.abc import Callable
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
    labeled_utterance,
    load_labeled_events,
    play_user_turns,
)
from sim.agents import BaselineAgent, FsmAgent, _fields_to_collect
from sim.fsm import FsmSpec
from sim.kb import KnowledgeBase, render_facts
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
    assert render_facts(real_kb.facts) in prompt
    assert "# State: identification" not in prompt
    assert preferred not in prompt
    assert_prompt_carries_no_answer_key(prompt, example_scenarios["adversarial_01"])
    assert record.state_before == "greeting"
    assert record.state_after == "greeting"
    assert record.event == "none"
    assert last_agent_call(greeting_none.llm)["caller"] == "fsm"


def test_a_scripted_dialogue_walks_greeting_to_closing(closing_walk: Walk) -> None:
    records = closing_walk.records

    assert [record.state_after for record in records] == [
        "identification",
        "intent_classification",
        "solution",
        "confirmation",
        "closing",
    ]


def test_the_turn_that_walks_two_edges_records_both_of_them(
    closing_walk: Walk,
) -> None:
    solved = next(
        record for record in closing_walk.records if record.state_after == "solution"
    )

    assert [
        (edge.source, edge.event, edge.dest, edge.fired_by)
        for edge in solved.transitions
    ] == [
        ("intent_classification", "intent_classified", "data_collection", "user"),
        ("data_collection", "data_provided", "solution", "engine"),
    ]
    assert solved.state_before == "intent_classification"


def test_the_recorded_walk_is_a_contiguous_path_of_the_machine(
    closing_walk: Walk, real_fsm: FsmSpec
) -> None:
    edges = {(edge.source, edge.event, edge.dest) for edge in real_fsm.transitions}
    walked = [edge for record in closing_walk.records for edge in record.transitions]

    assert walked
    assert all((edge.source, edge.event, edge.dest) in edges for edge in walked)
    assert [edge.source for edge in walked[1:]] == [edge.dest for edge in walked[:-1]]
    assert walked[0].source == real_fsm.initial
    assert walked[-1].dest == "closing"


def test_out_of_scope_carries_its_own_package_not_anothers(
    out_of_scope_turn: AgentTurn,
    real_fsm: FsmSpec,
) -> None:
    prompt = out_of_scope_turn.prompt

    assert out_of_scope_turn.record.state_after == "out_of_scope"
    assert package_text(real_fsm, "out_of_scope") in prompt
    assert "Give the outcome in the first sentence" not in prompt


def test_every_state_and_the_baseline_carry_the_same_knowledge_base(
    greeting_none: AgentTurn,
    closing_walk: Walk,
    out_of_scope_turn: AgentTurn,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    kb = render_facts(real_kb.facts)
    prompts = [
        greeting_none.prompt,
        *[call["messages"][0]["content"] for call in agent_calls(closing_walk.llm)],
        out_of_scope_turn.prompt,
    ]
    baseline = BaselineAgent(FakeLlm(), kb=real_kb, prompts_dir=PROMPTS_DIR)

    assert all(kb in prompt for prompt in prompts)
    assert kb in baseline.system_prompt
    assert package_text(real_fsm, "greeting") in greeting_none.prompt
    assert package_text(real_fsm, "out_of_scope") in out_of_scope_turn.prompt
    assert package_text(real_fsm, "greeting") not in out_of_scope_turn.prompt
    assert package_text(real_fsm, "out_of_scope") not in greeting_none.prompt


def test_the_known_order_number_and_email_are_in_the_messages_the_fsm_agent_sends(
    fsm_agent: MakeFsm,
) -> None:
    llm = FakeLlm([classifier_reply("request_received", "cancellation"), AGENT_REPLY])
    history = [
        Turn(
            speaker="user",
            text="I want to cancel order NL-20260145 from jane@example.com.",
        )
    ]

    fsm_agent(llm).respond(history)

    sent = "\n".join(message["content"] for message in last_agent_call(llm)["messages"])
    assert "NL-20260145" in sent
    assert "jane@example.com" in sent


@pytest.mark.parametrize(
    ("intent", "opening"),
    [
        (
            "cancellation",
            "I want to cancel order NL-20260145 from jane@example.com.",
        ),
        (
            "payment_reissue",
            "Please reissue expired slip NL-20260423 for tara.quinn@example.com.",
        ),
    ],
    ids=["cancellation", "payment_reissue"],
)
def test_when_identification_is_auto_skipped_the_next_prompt_keeps_identity_contract(
    intent: str, opening: str, fsm_agent: MakeFsm
) -> None:
    llm = FakeLlm([classifier_reply("request_received", intent), AGENT_REPLY])

    record = fsm_agent(llm).respond([Turn(speaker="user", text=opening)])
    prompt = last_agent_prompt(llm).lower()

    assert record.state_after == "intent_classification"
    assert "read back the exact order number" in prompt
    assert "do not ask the customer to provide, repeat or" in prompt
    assert "confirm either value again unless the customer explicitly corrects one" in (
        prompt
    )


def test_fsm_turn_records_persist_the_engine_intent_and_collected_slots(
    fsm_agent: MakeFsm,
) -> None:
    llm = FakeLlm([classifier_reply("request_received", "cancellation"), AGENT_REPLY])
    record = fsm_agent(llm).respond(
        [
            Turn(
                speaker="user",
                text="I want to cancel order NL-20260145 from jane@example.com.",
            )
        ]
    )

    assert record.intent == "cancellation"
    assert record.collected["order_number"] == "NL-20260145"
    assert record.collected["email"] == "jane@example.com"


def test_data_collection_lists_only_the_fields_still_missing(
    real_kb: KnowledgeBase, real_fsm: FsmSpec
) -> None:
    collect_state = next(
        edge.source
        for edge in real_fsm.transitions
        if edge.guard == "required_data_collected" and not edge.from_any
    )

    fields = _fields_to_collect(
        collect_state,
        "exchange_return",
        real_kb,
        collect_state,
        collected={"order_number": "NL-20260145", "email": "jane@example.com"},
    )

    assert [field.key for field in fields] == ["item", "reason", "preferred_resolution"]


def test_data_collection_keeps_the_fields_still_missing_for_the_active_intent(
    real_kb: KnowledgeBase, real_fsm: FsmSpec
) -> None:
    collect_state = next(
        edge.source
        for edge in real_fsm.transitions
        if edge.guard == "required_data_collected" and not edge.from_any
    )

    fields = _fields_to_collect(
        collect_state,
        "cancellation",
        real_kb,
        collect_state,
        collected={"order_number": "NL-20260145", "email": "jane@example.com"},
    )

    assert [field.key for field in fields] == ["reason"]


def test_data_collection_omits_fields_of_other_intents(
    real_kb: KnowledgeBase, real_fsm: FsmSpec
) -> None:
    collect_state = next(
        edge.source
        for edge in real_fsm.transitions
        if edge.guard == "required_data_collected" and not edge.from_any
    )

    fields = _fields_to_collect(
        collect_state,
        "exchange_return",
        real_kb,
        collect_state,
        collected={
            "order_number": "NL-20260145",
            "email": "jane@example.com",
            "item": "blue jacket",
            "reason": "too small",
            "preferred_resolution": "exchange",
        },
    )

    assert fields == []


def test_data_collection_prompt_lists_exactly_the_still_missing_fields_in_execution(
    fsm_agent: MakeFsm, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm(
        [
            '{"event":"request_received","intent":"exchange_return",'
            '"reason":"too small"}',
            AGENT_REPLY,
            '{"event":"intent_classified","intent":"exchange_return"}',
            AGENT_REPLY,
        ]
    )
    agent = fsm_agent(llm)
    history = [
        Turn(
            speaker="user",
            text=(
                "I need an exchange for order NL-20260145, "
                "jane@example.com, because it is too small."
            ),
        )
    ]
    first = agent.respond(history)
    history.append(Turn(speaker="agent", text=first.agent_reply))
    history.append(Turn(speaker="user", text="Yes, exchange return."))
    second = agent.respond(history)
    prompt = last_agent_prompt(llm)

    assert second.state_after == "data_collection"
    assert field_label(real_kb, "item") in prompt
    assert field_label(real_kb, "preferred_resolution") in prompt
    assert field_label(real_kb, "order_number") not in prompt
    assert field_label(real_kb, "email") not in prompt
    assert field_label(real_kb, "reason") not in prompt


def test_data_collection_prompt_equals_required_minus_collected_in_execution(
    fsm_agent: MakeFsm, real_kb: KnowledgeBase
) -> None:
    llm = FakeLlm(
        [
            classifier_reply("request_received", "cancellation"),
            AGENT_REPLY,
            classifier_reply("none"),
            AGENT_REPLY,
        ]
    )
    agent = fsm_agent(llm)
    history = [
        Turn(
            speaker="user",
            text="I want to cancel order NL-20260145 from jane@example.com.",
        )
    ]

    first = agent.respond(history)
    history.append(Turn(speaker="agent", text=first.agent_reply))
    history.append(Turn(speaker="user", text="Yes, cancel it."))
    second = agent.respond(history)
    prompt = last_agent_prompt(llm)

    assert second.state_after == "data_collection"

    required = {
        field.key
        for field in real_kb.user_data_fields
        if "cancellation" in field.required_for
    }
    expected_missing = required - set(second.collected)
    shown_fields = {
        field.key for field in real_kb.user_data_fields if field.label in prompt
    }

    assert shown_fields == expected_missing
    assert shown_fields == {"reason"}
    assert "item" not in shown_fields
    assert "preferred_resolution" not in shown_fields


def package_text(spec: FsmSpec, state: str) -> str:
    """Read the instruction package of ``state`` from the real FSM directory."""
    return (FSM_DIR / spec.states[state].package).read_text(encoding="utf-8")


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
