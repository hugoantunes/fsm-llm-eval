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
from sim.script import beats_of
from sim.user import SimulatedUser

#: The ``play`` fixture below, as the tests calling it see it: the customer's
#: canned replies, the agent's, and how many beats the brief is cut to.
Play = Callable[..., DialogueResult]


@pytest.fixture(scope="session")
def happy_path(example_scenarios: dict[str, Scenario]) -> Scenario:
    """The scenario these tests play; the loop, not the scenario, is under test."""
    return example_scenarios["happy_path_01"]


@pytest.fixture
def sent(happy_path: Scenario) -> list[str]:
    """Each beat's own words, which is the message its contract accepts.

    A canned customer here is not what the beat contract of T-11 is tested on —
    ``test_script.py`` does that — so these tests send the beat and spend the
    assertions on the loop.
    """
    return list(happy_path.script)


@pytest.fixture
def play(real_kb: KnowledgeBase, happy_path: Scenario) -> Play:
    """Run one dialogue with canned replies on both sides of it.

    The real agent and the real simulated user are used, each with its own fake
    of the client of T-07, and the turn budget is the scenario's own unless
    ``max_turns`` says otherwise: the loop is what is under test, not the models.

    The brief is cut to its last beat unless ``script`` is given, because a
    script is a mandatory ordered plan and a canned dialogue of one or two turns
    cannot deliver the four beats of the real one. The last beat is the default
    because it spells out no order number or e-mail: what a beat requires is
    tested in ``test_script.py``, not here.
    """

    def _play(
        user_replies: Sequence[str],
        agent_replies: Sequence[str],
        *,
        script: Sequence[str] | None = None,
        max_turns: int | None = None,
    ) -> DialogueResult:
        scenario = happy_path.model_copy(
            update={
                "script": list(script or happy_path.script[-1:]),
                "max_turns": max_turns or happy_path.max_turns,
            }
        )
        user = SimulatedUser(
            FakeLlm(user_replies),
            scenario=scenario,
            data_fields=real_kb.user_data_fields,
            prompts_dir=PROMPTS_DIR,
        )
        agent = BaselineAgent(
            FakeLlm(agent_replies), kb=real_kb, prompts_dir=PROMPTS_DIR
        )
        return run_dialogue(agent, user, max_turns=scenario.max_turns)

    return _play


def test_a_dialogue_keeps_one_turn_record_per_agent_turn_with_its_beat(
    play: Play, sent: list[str]
) -> None:
    result = play(
        [
            user_reply(sent[-2]),
            user_reply(sent[-1], status="goal_reached"),
        ],
        [
            "What is the e-mail used in the purchase?",
            "It ships within 2 business days.",
        ],
        script=sent[-2:],
    )

    agent_turns = [turn for turn in result.transcript if turn.speaker == "agent"]
    assert len(result.records) == len(agent_turns)
    assert [record.turn for record in result.records] == [1, 2]
    assert [record.user_message for record in result.records] == [sent[-2], sent[-1]]
    assert [record.agent_reply for record in result.records] == [
        turn.text for turn in agent_turns
    ]
    assert [record.user_beat for record in result.records] == [1, 2]
    assert [record.beat_started_at_turn for record in result.records] == [1, 2]
    assert [record.beat_completed_at_turn for record in result.records] == [1, 2]


def test_a_dialogue_ends_on_the_agents_reply_when_the_user_reached_its_goal(
    play: Play, sent: list[str]
) -> None:
    result = play(
        [user_reply(sent[-1], status="goal_reached")],
        ["Glad to help. Support answers between 9:00 and 18:00."],
    )

    assert result.stop_reason == "goal_reached"
    assert result.goal_reached_seen is True
    assert result.goal_reached_at_turn == 1
    assert result.script_complete_at_goal_reached is True
    assert result.transcript[-1].speaker == "agent"
    assert (
        result.transcript[-1].text
        == "Glad to help. Support answers between 9:00 and 18:00."
    )


def test_a_dialogue_stops_when_the_user_gave_up(play: Play, sent: list[str]) -> None:
    result = play(
        [user_reply(f"{sent[-1]} This is going nowhere.", status="gave_up")],
        ["I am sorry it came to that."],
    )

    assert result.stop_reason == "user_gave_up"
    assert len(result.records) == 1


def test_agent_ended_on_an_empty_transcript_raises() -> None:
    with pytest.raises(DialogueError, match="never spoke"):
        run_dialogue(
            _AgentThatFailsOnTheSecondTurn(),
            _UserThatOnlySaysTheAgentClosed(),
            max_turns=8,
        )


def test_a_dialogue_stops_without_a_further_message_when_the_agent_closed(
    play: Play, sent: list[str]
) -> None:
    result = play(
        [
            user_reply(sent[-1]),
            user_reply("", status="agent_ended"),
        ],
        ["It ships within 2 business days. Goodbye."],
    )

    assert result.stop_reason == "agent_closed"
    assert [turn.speaker for turn in result.transcript] == ["user", "agent"]
    assert all(turn.text for turn in result.transcript)
    assert len(result.records) == 1


def test_a_dialogue_stops_at_the_scenarios_max_turns(
    play: Play, happy_path: Scenario, sent: list[str]
) -> None:
    budget = happy_path.max_turns

    result = play(
        [user_reply(sent[-1]), *[user_reply("Are you still there?")] * budget],
        ["Yes, I am here."] * (budget + 1),
    )

    assert result.stop_reason == "max_turns"
    assert len(result.records) == budget
    assert len(result.transcript) == 2 * budget


def test_goal_reached_is_recorded_without_ending_while_beats_remain(
    play: Play,
) -> None:
    result = play(
        [
            user_reply("alpha request"),
            user_reply("bravo request"),
            user_reply("charlie request", status="goal_reached"),
            user_reply("Goodbye."),
        ],
        ["ok"] * 4,
        script=["alpha request", "bravo request", "charlie request", "Goodbye."],
        max_turns=6,
    )

    assert result.stop_reason == "goal_reached"
    assert result.goal_reached_seen is True
    assert result.goal_reached_at_turn == 3
    assert result.script_complete_at_goal_reached is False
    assert [record.user_beat for record in result.records] == [1, 2, 3, 4]
    assert result.records[3].beat_started_at_turn == 4
    assert result.records[3].beat_completed_at_turn == 4


def test_goal_reached_before_completion_still_fails_on_max_turns(play: Play) -> None:
    result = play(
        [
            user_reply("alpha request"),
            user_reply("bravo request", status="goal_reached"),
            *[user_reply("order NL-20260145")] * 3,
        ],
        ["ok"] * 5,
        script=[
            "alpha request",
            "bravo request",
            "order NL-20260145 and user@example.com",
            "Goodbye.",
        ],
        max_turns=5,
    )

    assert result.stop_reason == "max_turns"
    assert result.goal_reached_seen is True
    assert result.goal_reached_at_turn == 2
    assert result.script_complete_at_goal_reached is False
    assert [record.user_beat for record in result.records] == [1, 2, None, None, None]


def test_a_dialogue_result_does_not_store_the_transcript_twice() -> None:
    assert "transcript" not in DialogueResult.model_fields


def test_a_failed_turn_keeps_the_turns_already_played(
    real_kb: KnowledgeBase, happy_path: Scenario, sent: list[str]
) -> None:
    user = SimulatedUser(
        FakeLlm([user_reply(sent[-1]), user_reply("Anything else?")]),
        scenario=happy_path.model_copy(update={"script": happy_path.script[-1:]}),
        data_fields=real_kb.user_data_fields,
        prompts_dir=PROMPTS_DIR,
    )

    with pytest.raises(DialogueError) as exc_info:
        run_dialogue(_AgentThatFailsOnTheSecondTurn(), user, max_turns=8)

    assert len(exc_info.value.records) == 1
    assert exc_info.value.records[0].user_message == sent[-1]
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


class _UserThatOnlySaysTheAgentClosed:
    """Declares ``agent_ended`` on the first turn, before the agent has spoken.

    Hand-written rather than a :class:`~sim.user.SimulatedUser`, because the real
    one refuses that status while a beat is pending and asks the model again:
    the invariant under test here is the loop's, and it holds for any customer.
    """

    def speak(self, history: Sequence[Turn]) -> object:
        return _SilentTurn()


class _SilentTurn:
    """The empty closing turn the loop must refuse on an empty transcript."""

    message = ""
    status = "agent_ended"
    beat = None


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
                real_client,
                scenario=scenario,
                data_fields=real_kb.user_data_fields,
                prompts_dir=PROMPTS_DIR,
                seed=seed,
            ),
            max_turns=scenario.max_turns,
        )
        for scenario in example_scenarios.values()
    }

    for scenario_id, result in results.items():
        print(f"\n--- {scenario_id}: stopped on {result.stop_reason} " + "-" * 40)
        for turn in result.transcript:
            print(f"\n{turn.speaker}: {turn.text}")
    for scenario_id, result in results.items():
        assert len(result.records) >= 2
        assert all(record.agent_reply.strip() for record in result.records)
        assert all(turn.text.strip() for turn in result.transcript)
        customer = [turn.text for turn in result.transcript if turn.speaker == "user"]
        assert len(customer) == len(set(customer))
        scenario = example_scenarios[scenario_id]
        beats = [
            record.user_beat
            for record in result.records
            if record.user_beat is not None
        ]
        assert beats == sorted(beats)
        assert beats == list(range(1, len(beats) + 1))
        if scenario.canary is not None:
            assert any(
                scenario.canary in record.user_message for record in result.records
            )
        if result.stop_reason in ("goal_reached", "user_gave_up", "agent_closed"):
            assert len(beats) == len(beats_of(scenario, real_kb.user_data_fields))
