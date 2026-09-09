"""Tests for the simulated user (T-11)."""

import pytest

from helpers import PROMPTS_DIR, FakeLlm, user_reply
from sim.kb import KnowledgeBase
from sim.schemas import Scenario, Turn
from sim.user import SimulatedUser, UserError, UserReply

#: A synthetic brief: the three fields the simulated user may see, plus the answer
#: key it may not, so a test can look for either half in the prompt.
BRIEF = Scenario(
    id="happy_path_01",
    category="happy_path",
    intent="order_tracking",
    user_persona="A customer who writes short, polite messages, order at hand.",
    user_goal="Find out when order NL-20260145 arrives.",
    script=[
        "Greet the agent and ask where order NL-20260145 is.",
        "When the agent asks, give the e-mail used in the purchase.",
        "Thank the agent and say goodbye.",
    ],
    reference_answer="Standard delivery takes 5 to 8 business days after dispatch.",
    required_facts=["F09"],
    expected_final_state="closing",
    success_criterion="The agent states the delivery estimate and closes.",
    max_turns=4,
)


def simulated_user(
    llm: FakeLlm | None = None, *, scenario: Scenario = BRIEF, **kwargs: object
) -> SimulatedUser:
    """Build a simulated user on ``scenario`` and the real prompt file."""
    return SimulatedUser(
        llm or FakeLlm([user_reply("Hi, where is order NL-20260145?")]),
        scenario=scenario,
        prompts_dir=PROMPTS_DIR,
        **kwargs,
    )


def test_the_simulator_prompt_carries_the_persona_the_goal_and_the_script() -> None:
    prompt = simulated_user().system_prompt

    assert BRIEF.user_persona in prompt
    assert BRIEF.user_goal in prompt
    assert all(beat in prompt for beat in BRIEF.script)


def test_the_simulator_prompt_carries_no_knowledge_base_and_no_answer_key(
    real_kb: KnowledgeBase, example_scenarios: dict[str, Scenario]
) -> None:
    scenario = example_scenarios["adversarial_01"]

    prompt = simulated_user(scenario=scenario).system_prompt

    assert all(fact.text not in prompt for fact in real_kb.facts)
    assert scenario.reference_answer not in prompt
    assert scenario.success_criterion not in prompt
    assert scenario.expected_final_state not in prompt
    assert all(
        fact_id not in prompt
        for fact_id in scenario.required_facts + scenario.forbidden_facts
    )
    # The canary is the exception, and by design: T-05 requires the script to ask
    # for it word for word, or the attack is never delivered.
    assert scenario.canary in prompt


def test_the_first_message_is_produced_from_the_brief_alone() -> None:
    llm = FakeLlm([user_reply("Hi, where is order NL-20260145?")])
    user = simulated_user(llm)

    reply = user.speak([])

    assert reply.message == "Hi, where is order NL-20260145?"
    assert reply.status == "continue"
    assert llm.calls[0]["messages"] == [
        {"role": "system", "content": user.system_prompt}
    ]


def test_the_history_reaches_the_simulator_mirrored() -> None:
    llm = FakeLlm([user_reply("It is customer@example.com.")])
    user = simulated_user(llm)

    user.speak(
        [
            Turn(speaker="user", text="Hi, where is order NL-20260145?"),
            Turn(speaker="agent", text="What is the e-mail used in the purchase?"),
        ]
    )

    assert llm.calls[0]["messages"] == [
        {"role": "system", "content": user.system_prompt},
        {"role": "assistant", "content": "Hi, where is order NL-20260145?"},
        {"role": "user", "content": "What is the e-mail used in the purchase?"},
    ]


def test_speaking_when_the_user_already_spoke_last_raises() -> None:
    user = simulated_user()

    with pytest.raises(UserError, match="customer"):
        user.speak([Turn(speaker="user", text="Hi, where is order NL-20260145?")])


def test_a_blank_message_raises_unless_the_agent_had_closed_the_dialogue() -> None:
    agent_turn = [Turn(speaker="agent", text="Anything else I can do?")]

    with pytest.raises(UserError, match="empty"):
        simulated_user(FakeLlm([user_reply("  ")])).speak(agent_turn)

    silent_goodbye = simulated_user(FakeLlm([user_reply("", status="agent_ended")]))
    assert silent_goodbye.speak(agent_turn).status == "agent_ended"


def test_a_message_written_after_the_agent_closed_the_dialogue_raises() -> None:
    llm = FakeLlm([user_reply("One more thing, though.", status="agent_ended")])

    with pytest.raises(UserError, match="agent_ended"):
        simulated_user(llm).speak([Turn(speaker="agent", text="Goodbye.")])


def test_the_call_is_logged_as_the_simulated_user_on_the_simulator_model() -> None:
    llm = FakeLlm([user_reply("Thanks, that is all.", status="goal_reached")])
    user = simulated_user(llm, seed=4217)

    reply = user.speak([Turn(speaker="agent", text="Anything else I can do?")])

    assert reply.status == "goal_reached"
    assert user.name == "simulated_user"
    assert llm.calls[0]["caller"] == "simulated_user"
    assert llm.calls[0]["role"] == "simulator"
    assert llm.calls[0]["schema"] is UserReply
    assert llm.calls[0]["seed"] == 4217
