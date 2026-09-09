"""Tests for the agent interface and the baseline agent (T-09)."""

import re

import pytest

from helpers import GENERAL_FACT, PROMPTS_DIR, TRACKING_FACT, FakeLlm, make_kb
from sim.agents import AgentError, BaselineAgent
from sim.kb import Fact, KnowledgeBase, UserDataField
from sim.llm import LlmClient
from sim.prompts import load_prompt
from sim.schemas import MAX_TURNS, Scenario, Turn, TurnRecord

#: Four characters per token, the rule of thumb for English on these tokenizers.
#: It makes the context budget testable without Ollama; the real count comes from
#: ``prompt_eval_count`` in the integration test below, and at run time the client
#: of T-07 refuses anything over 80% of ``num_ctx`` anyway.
CHARS_PER_TOKEN = 4

#: A knowledge-base identifier in a reply: an internal label the shared block
#: forbids saying to the customer.
FACT_ID = re.compile(r"\bF\d{2}\b")

#: The synthetic KB most of these tests build a baseline on: three facts, so an
#: assertion can name every one of them. The real KB arrives as ``real_kb``,
#: where what is under test is the size of the prompt it produces.
KB = make_kb(
    GENERAL_FACT,
    TRACKING_FACT,
    Fact(
        id="F03",
        intent="exchange_return",
        text="Any item can be returned within 30 days.",
    ),
    user_data_fields=[
        UserDataField(
            key="order_number",
            label="Order number",
            pattern=r"\bNL-\d{8}\b",
            example="NL-20260145",
            required_for=["order_tracking", "exchange_return"],
        ),
        UserDataField(
            key="reason",
            label="Reason for the return",
            example="the size is too small",
            required_for=["exchange_return"],
        ),
    ],
)


def baseline(llm: FakeLlm | None = None, **kwargs: object) -> BaselineAgent:
    """Build a baseline agent on the synthetic KB and the real prompt files."""
    return BaselineAgent(llm or FakeLlm(), kb=KB, prompts_dir=PROMPTS_DIR, **kwargs)


def verbose_history(exchanges: int) -> list[Turn]:
    """Build a talkative dialogue of ``exchanges`` exchanges, the worst case.

    Every message is longer than a real one, so a prompt that fits here fits any
    dialogue the scenarios of T-05 allow.
    """
    customer = (
        "I checked the order history again and I still cannot find what I need, so "
        "let me repeat the whole story with every detail I have, including the item, "
        "the date and what the confirmation e-mail said when the order was placed. "
    )
    agent = (
        "Thank you for the details. Here is what applies to your order, the deadline "
        "that governs it and the next step, with the condition that decides the case "
        "spelled out so that nothing about it is left open on your side. "
    )
    turns: list[Turn] = []
    for _ in range(exchanges):
        turns.append(Turn(speaker="user", text=customer))
        turns.append(Turn(speaker="agent", text=agent))
    turns.append(Turn(speaker="user", text=customer))
    return turns


def test_the_baseline_prompt_carries_the_shared_block_and_every_fact() -> None:
    agent = baseline()

    prompt = agent.system_prompt

    assert load_prompt("agent_shared", directory=PROMPTS_DIR).template in prompt
    assert all(fact.text in prompt for fact in KB.facts)
    assert all(field.label in prompt for field in KB.user_data_fields)


def test_the_baseline_prompt_carries_no_part_of_the_answer_key(
    real_kb: KnowledgeBase, example_scenarios: dict[str, Scenario]
) -> None:
    scenario = example_scenarios["adversarial_01"]

    prompt = BaselineAgent(FakeLlm(), kb=real_kb, prompts_dir=PROMPTS_DIR).system_prompt

    assert scenario.canary not in prompt
    assert scenario.reference_answer not in prompt
    assert scenario.success_criterion not in prompt
    assert scenario.user_persona not in prompt
    assert scenario.user_goal not in prompt
    assert all(beat not in prompt for beat in scenario.script)


def test_the_history_becomes_system_plus_alternating_chat_turns() -> None:
    llm = FakeLlm(["Your order is on its way."])
    agent = baseline(llm)

    agent.respond(
        [
            Turn(speaker="user", text="Where is my order?"),
            Turn(speaker="agent", text="What is the order number?"),
            Turn(speaker="user", text="NL-20260145"),
        ]
    )

    assert llm.calls[0]["messages"] == [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "What is the order number?"},
        {"role": "user", "content": "NL-20260145"},
    ]


def test_the_turn_record_carries_both_latencies_the_tokens_and_the_prompt_hash() -> (
    None
):
    llm = FakeLlm(["Dispatch takes 2 business days."], latency_s=0.5)
    agent = baseline(llm)

    record = agent.respond(
        [
            Turn(speaker="user", text="Where is my order?"),
            Turn(speaker="agent", text="What is the order number?"),
            Turn(speaker="user", text="NL-20260145"),
        ]
    )

    assert record.turn == 2
    assert record.user_message == "NL-20260145"
    assert record.agent_reply == "Dispatch takes 2 business days."
    assert record.prompt_hash == llm.calls[0]["prompt_hash"]
    assert record.prompt_tokens > 0
    assert record.output_tokens > 0
    assert record.llm_latency_s == 0.5
    assert record.turn_latency_s > 0
    assert record.turn_latency_s != record.llm_latency_s


def test_the_baseline_record_leaves_state_and_event_empty() -> None:
    record = baseline().respond([Turn(speaker="user", text="Hello?")])

    assert record.state_before is None
    assert record.state_after is None
    assert record.event is None


def test_responding_to_a_history_that_does_not_end_with_the_user_raises() -> None:
    agent = baseline()

    with pytest.raises(AgentError, match="empty"):
        agent.respond([])

    with pytest.raises(AgentError, match="agent"):
        agent.respond(
            [Turn(speaker="user", text="Hello?"), Turn(speaker="agent", text="Hello.")]
        )


def test_the_call_is_logged_under_the_agent_name() -> None:
    llm = FakeLlm()
    agent = baseline(llm)

    agent.respond([Turn(speaker="user", text="Hello?")])

    assert agent.name == "baseline"
    assert llm.calls[0]["caller"] == "baseline"
    assert llm.calls[0]["role"] == "agent"


def test_the_dialogue_seed_reaches_the_model() -> None:
    llm = FakeLlm()

    baseline(llm, seed=4217).respond([Turn(speaker="user", text="Hello?")])

    assert llm.calls[0]["seed"] == 4217


def test_the_baseline_prompt_stays_within_the_character_budget_at_max_turns(
    real_kb: KnowledgeBase, prompt_budget: int
) -> None:
    llm = FakeLlm()
    agent = BaselineAgent(llm, kb=real_kb, prompts_dir=PROMPTS_DIR)

    agent.respond(verbose_history(MAX_TURNS))

    sent = sum(len(message["content"]) for message in llm.calls[0]["messages"])
    assert sent < prompt_budget * CHARS_PER_TOKEN


# --- Integration: a real Ollama with the models of configs/models.yaml -------


def manual_dialogues(kb: KnowledgeBase) -> dict[str, list[str]]:
    """The three hand-written dialogues criterion 3 asks to read.

    A plain request, a needle only a model that read the knowledge base answers
    right, and a question the knowledge base does not answer at all, which has to
    be escalated rather than guessed. The last two come from ``data/kb/`` so the
    traps are the ones the dataset of T-06 is built on.
    """
    needle = next(needle for needle in kb.needles if needle.fact_id == "F18")
    unanswerable = next(entry for entry in kb.unanswerable if entry.id == "U04")
    return {
        "order_tracking": [
            "Hi, where is my order NL-20260145?",
            "It is customer@example.com.",
            "Thanks. How long does the delivery take, and where do I find the "
            "tracking code?",
        ],
        "needle_promotional_return": [
            needle.probe_question,
            "The order is NL-20260145 and the e-mail is customer@example.com.",
        ],
        "unanswerable_warranty": [
            unanswerable.question,
            "Come on, just give me your best guess in months.",
        ],
    }


def run_dialogue(agent: BaselineAgent, messages: list[str]) -> list[TurnRecord]:
    """Play ``messages`` at ``agent``, one at a time, and return its turn records."""
    history: list[Turn] = []
    records: list[TurnRecord] = []
    for message in messages:
        history.append(Turn(speaker="user", text=message))
        record = agent.respond(history)
        history.append(Turn(speaker="agent", text=record.agent_reply))
        records.append(record)
    return records


@pytest.fixture
def real_baseline(real_client: LlmClient, real_kb: KnowledgeBase) -> BaselineAgent:
    """A baseline agent on the real config, KB, prompts and Ollama server."""
    return BaselineAgent(real_client, kb=real_kb, prompts_dir=PROMPTS_DIR, seed=42)


@pytest.mark.integration
def test_a_real_baseline_turn_stays_under_the_token_budget_at_max_turns(
    real_baseline: BaselineAgent, prompt_budget: int
) -> None:
    record = real_baseline.respond(verbose_history(MAX_TURNS))

    print(f"\nbaseline prompt with {MAX_TURNS} exchanges: {record.prompt_tokens} of ")
    print(f"{prompt_budget} tokens budgeted ")
    print(f"({record.prompt_tokens / prompt_budget:.0%})")
    assert record.agent_reply.strip()
    assert record.prompt_tokens < prompt_budget


@pytest.mark.integration
def test_three_real_dialogues_run_through_the_baseline(
    real_baseline: BaselineAgent, real_kb: KnowledgeBase, prompt_budget: int
) -> None:
    dialogues = {
        name: run_dialogue(real_baseline, messages)
        for name, messages in manual_dialogues(real_kb).items()
    }

    for name, records in dialogues.items():
        print(f"\n--- {name} " + "-" * 60)
        for record in records:
            print(f"\ncustomer: {record.user_message}\nagent: {record.agent_reply}")
    replies = [record for records in dialogues.values() for record in records]
    assert all(record.agent_reply.strip() for record in replies)
    assert not [record for record in replies if FACT_ID.search(record.agent_reply)]
    assert all(record.prompt_tokens < prompt_budget for record in replies)
