"""Tests for the dialogue loop (T-11)."""

from collections.abc import Callable, Sequence

import pytest

from helpers import PROMPTS_DIR, FakeLlm, make_turn_record, user_reply
from sim.agents import BaselineAgent
from sim.dialogue import DialogueError, DialogueResult, run_dialogue
from sim.events import EventError
from sim.kb import KnowledgeBase
from sim.llm import LlmClient
from sim.schemas import Scenario, Turn, TurnRecord
from sim.user import SimulatedUser

#: The ``play`` fixture below, as the tests calling it see it: the customer's
#: canned replies, then the agent's.
Play = Callable[[Sequence[str], Sequence[str]], DialogueResult]


@pytest.fixture(scope="session")
def happy_path(example_scenarios: dict[str, Scenario]) -> Scenario:
    """The scenario these tests play; the loop, not the scenario, is under test."""
    return example_scenarios["happy_path_01"]


@pytest.fixture
def play(real_kb: KnowledgeBase, happy_path: Scenario) -> Play:
    """Run one dialogue with canned replies on both sides of it.

    The real agent and the real simulated user are used, each with its own fake
    of the client of T-07, and the turn budget is the scenario's own: the loop is
    what is under test, not the models.
    """

    def _play(
        user_replies: Sequence[str], agent_replies: Sequence[str]
    ) -> DialogueResult:
        user = SimulatedUser(
            FakeLlm(user_replies), scenario=happy_path, prompts_dir=PROMPTS_DIR
        )
        agent = BaselineAgent(
            FakeLlm(agent_replies), kb=real_kb, prompts_dir=PROMPTS_DIR
        )
        return run_dialogue(agent, user, max_turns=happy_path.max_turns)

    return _play


def test_a_dialogue_alternates_the_user_and_the_agent(play: Play) -> None:
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


def test_a_dialogue_ends_on_the_agents_reply_when_the_user_reached_its_goal(
    play: Play,
) -> None:
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


def test_a_dialogue_stops_when_the_user_gave_up(play: Play) -> None:
    result = play(
        [user_reply("This is going nowhere, I am done.", status="gave_up")],
        ["I am sorry it came to that."],
    )

    assert result.stop_reason == "user_gave_up"
    assert len(result.records) == 1


def test_agent_ended_on_an_empty_transcript_raises(play: Play) -> None:
    with pytest.raises(DialogueError, match="never spoke"):
        play([user_reply("", status="agent_ended")], [])


def test_a_dialogue_stops_without_a_further_message_when_the_agent_closed(
    play: Play,
) -> None:
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


def test_a_dialogue_stops_at_the_scenarios_max_turns(
    play: Play, happy_path: Scenario
) -> None:
    budget = happy_path.max_turns

    result = play(
        [user_reply("Are you still there?")] * (budget + 1),
        ["Yes, I am here."] * (budget + 1),
    )

    assert result.stop_reason == "max_turns"
    assert len(result.records) == budget
    assert len(result.transcript) == 2 * budget


def test_a_dialogue_result_does_not_store_the_transcript_twice() -> None:
    assert "transcript" not in DialogueResult.model_fields


def test_a_failed_turn_keeps_the_turns_already_played(happy_path: Scenario) -> None:
    user = SimulatedUser(
        FakeLlm(
            [
                user_reply("Hi, where is order NL-20260145?"),
                user_reply("Thanks, that is all.", status="goal_reached"),
            ]
        ),
        scenario=happy_path,
        prompts_dir=PROMPTS_DIR,
    )

    with pytest.raises(DialogueError) as exc_info:
        run_dialogue(_AgentThatFailsOnTheSecondTurn(), user, max_turns=8)

    assert len(exc_info.value.records) == 1
    assert exc_info.value.records[0].user_message == "Hi, where is order NL-20260145?"
    assert exc_info.value.records[0].agent_reply == "What is the e-mail?"


class _AgentThatFailsOnTheSecondTurn:
    """Speaks once, then raises the classifier contract error."""

    def respond(self, history: Sequence[Turn]) -> TurnRecord:
        n_user = sum(1 for turn in history if turn.speaker == "user")
        if n_user > 1:
            raise EventError("intent_classified with no intent")
        return make_turn_record(
            turn=1,
            user_message=history[-1].text,
            agent_reply="What is the e-mail?",
        )


def test_a_dialogue_keeps_one_turn_record_per_agent_turn(play: Play) -> None:
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
    real_client: LlmClient,
    real_kb: KnowledgeBase,
    example_scenarios: dict[str, Scenario],
) -> None:
    seed = 42

    results = {
        scenario.id: run_dialogue(
            BaselineAgent(real_client, kb=real_kb, prompts_dir=PROMPTS_DIR, seed=seed),
            SimulatedUser(
                real_client, scenario=scenario, prompts_dir=PROMPTS_DIR, seed=seed
            ),
            max_turns=scenario.max_turns,
        )
        for scenario in example_scenarios.values()
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
