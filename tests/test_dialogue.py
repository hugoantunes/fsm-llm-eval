"""Tests for the dialogue loop (T-11)."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from helpers import (
    CONFIG,
    KB_DIR,
    PROMPTS_DIR,
    FakeLlm,
    load_example_scenarios,
    user_reply,
)
from sim.agents import BaselineAgent
from sim.config import load_models_config
from sim.dialogue import DialogueResult, run_dialogue
from sim.kb import load_kb
from sim.llm import LlmClient
from sim.user import SimulatedUser

KB = load_kb(KB_DIR)
SCENARIO = load_example_scenarios()["happy_path_01"]


def play(
    user_replies: Sequence[str],
    agent_replies: Sequence[str],
    *,
    max_turns: int = 8,
) -> DialogueResult:
    """Run one dialogue with canned replies on both sides of it.

    The real agent and the real simulated user are used, each with its own fake
    of the client of T-07: the loop is what is under test, not the models.
    """
    user = SimulatedUser(
        FakeLlm(user_replies), scenario=SCENARIO, prompts_dir=PROMPTS_DIR
    )
    agent = BaselineAgent(FakeLlm(agent_replies), kb=KB, prompts_dir=PROMPTS_DIR)
    return run_dialogue(agent, user, max_turns=max_turns)


def test_a_dialogue_alternates_the_user_and_the_agent() -> None:
    result = play(
        [
            user_reply("Hi, where is order NL-20260145?"),
            user_reply("Thanks, that is all.", status="goal_reached"),
        ],
        [
            "What is the e-mail used in the purchase?",
            "It ships within 2 business days.",
        ],
    )

    assert [turn.speaker for turn in result.transcript] == [
        "user",
        "agent",
        "user",
        "agent",
    ]
    assert result.transcript[0].text == "Hi, where is order NL-20260145?"
    assert result.transcript[1].text == "What is the e-mail used in the purchase?"


def test_a_dialogue_ends_on_the_agents_reply_when_the_user_reached_its_goal() -> None:
    result = play(
        [user_reply("Thanks, that is all.", status="goal_reached")],
        ["Glad to help. Support answers between 9:00 and 18:00."],
    )

    assert result.stop_reason == "goal_reached"
    assert result.transcript[-1].speaker == "agent"
    assert (
        result.transcript[-1].text
        == "Glad to help. Support answers between 9:00 and 18:00."
    )


def test_a_dialogue_stops_when_the_user_gave_up() -> None:
    result = play(
        [user_reply("This is going nowhere, I am done.", status="gave_up")],
        ["I am sorry it came to that."],
    )

    assert result.stop_reason == "user_gave_up"
    assert len(result.records) == 1


def test_a_dialogue_stops_without_a_further_message_when_the_agent_closed() -> None:
    result = play(
        [
            user_reply("Hi, where is order NL-20260145?"),
            user_reply("", status="agent_ended"),
        ],
        ["It ships within 2 business days. Goodbye."],
    )

    assert result.stop_reason == "agent_closed"
    assert [turn.speaker for turn in result.transcript] == ["user", "agent"]
    assert all(turn.text for turn in result.transcript)
    assert len(result.records) == 1


def test_a_dialogue_stops_at_the_scenarios_max_turns() -> None:
    budget = SCENARIO.max_turns

    result = play(
        [user_reply("Are you still there?")] * (budget + 1),
        ["Yes, I am here."] * (budget + 1),
        max_turns=budget,
    )

    assert result.stop_reason == "max_turns"
    assert len(result.records) == budget
    assert len(result.transcript) == 2 * budget


def test_a_dialogue_keeps_one_turn_record_per_agent_turn() -> None:
    result = play(
        [
            user_reply("Hi, where is order NL-20260145?"),
            user_reply("Thanks, that is all.", status="goal_reached"),
        ],
        [
            "What is the e-mail used in the purchase?",
            "It ships within 2 business days.",
        ],
    )

    agent_turns = [turn for turn in result.transcript if turn.speaker == "agent"]
    assert len(result.records) == len(agent_turns)
    assert [record.turn for record in result.records] == [1, 2]
    assert [record.user_message for record in result.records] == [
        "Hi, where is order NL-20260145?",
        "Thanks, that is all.",
    ]
    assert [record.agent_reply for record in result.records] == [
        turn.text for turn in agent_turns
    ]


# --- Integration: a real Ollama with the models of configs/models.yaml -------


@pytest.mark.integration
def test_the_example_scenarios_run_through_the_baseline_without_stalling(
    tmp_path: Path,
) -> None:
    client = LlmClient(
        load_models_config(CONFIG),
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm_calls.jsonl",
    )
    seed = 42

    results = {
        scenario.id: run_dialogue(
            BaselineAgent(client, kb=KB, prompts_dir=PROMPTS_DIR, seed=seed),
            SimulatedUser(
                client, scenario=scenario, prompts_dir=PROMPTS_DIR, seed=seed
            ),
            max_turns=scenario.max_turns,
        )
        for scenario in load_example_scenarios().values()
    }

    for scenario_id, result in results.items():
        print(f"\n--- {scenario_id}: stopped on {result.stop_reason} " + "-" * 40)
        for turn in result.transcript:
            print(f"\n{turn.speaker}: {turn.text}")
    for result in results.values():
        assert len(result.records) >= 2
        assert all(record.agent_reply.strip() for record in result.records)
        assert all(turn.text.strip() for turn in result.transcript)
        customer = [turn.text for turn in result.transcript if turn.speaker == "user"]
        assert len(customer) == len(set(customer))
