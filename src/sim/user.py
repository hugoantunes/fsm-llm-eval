"""The simulated user that plays a scenario against an agent (T-11).

It sees three fields of the scenario and no more: the persona, the goal and the
script. The knowledge base, the reference answer, the required facts and the
success criterion belong to the evaluation side, and any of them in this prompt
would leak the answer key into the dialogue and from there into the judge.

The simulated user is not evaluated. Its calls are logged under its own name and
its replies are plain user turns, never the :class:`~sim.schemas.TurnRecord` the
metrics of T-13 read.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from sim.llm import Chat
from sim.prompts import DEFAULT_PROMPTS_DIR, load_prompt
from sim.schemas import Scenario, Turn, as_messages, render_script

#: The prompt file of ``data/prompts/``, holding the brief and nothing else.
PROMPT = "simulated_user"

#: Where the customer thinks the dialogue stands after the message it just wrote.
#: Three of the four stopping conditions of T-11 are these; the fourth is the
#: turn budget, which the loop of :mod:`sim.dialogue` owns.
UserStatus = Literal["continue", "goal_reached", "gave_up", "agent_ended"]


class UserError(RuntimeError):
    """The simulated user cannot speak here; the message says why."""


class UserReply(BaseModel):
    """One customer message and where the customer thinks the dialogue stands.

    ``status`` is what stops a dialogue (T-11): three of the four stopping
    conditions are decided here and ``max_turns`` is decided by the loop. It is a
    stopping condition and never a metric — whether the task was completed is the
    judge's call in T-12 — because a simulated user grading its own dialogue
    would be measuring the instrument.
    """

    model_config = ConfigDict(extra="forbid")

    message: str
    status: UserStatus


class SimulatedUser:
    """The customer side of a dialogue, played by the small model of T-03."""

    #: What the LLM log of T-07 records this caller as; the model is the same one
    #: the event classifier of T-08 and the stage labeler of T-13 use.
    name = "simulated_user"

    def __init__(
        self,
        llm: Chat,
        *,
        scenario: Scenario,
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
        seed: int | None = None,
    ) -> None:
        self._llm = llm
        self._seed = seed
        self.system_prompt = load_prompt(PROMPT, directory=prompts_dir).render(
            persona=scenario.user_persona,
            goal=scenario.user_goal,
            script=render_script(scenario.script),
        )

    def speak(self, history: Sequence[Turn]) -> UserReply:
        """Write the customer's next message, given the dialogue so far.

        ``history`` is the transcript as the agent recorded it, which
        :func:`~sim.schemas.as_messages` mirrors for this side of the
        conversation. The client of T-07 validates the answer against
        :class:`UserReply` before handing it back, so what comes out is the object.

        Raises:
            UserError: when the history already ends with the customer's turn. The
                customer speaks first and then one message per agent reply, so
                anything else is the loop having lost the agent's turn, and a
                second customer message would paper over it. Also when the model
                returns a blank message the dialogue is still expecting.
        """
        _check_the_agent_spoke_last(history)
        answer = self._llm.chat(
            as_messages(self.system_prompt, history, speaking_as="user"),
            role="simulator",
            caller=self.name,
            schema=UserReply,
            seed=self._seed,
        )
        reply = cast(UserReply, answer.parsed)
        _check_the_message_matches_the_status(reply)
        return reply


def _check_the_message_matches_the_status(reply: UserReply) -> None:
    """Fail unless the reply says something exactly when its status implies it.

    ``agent_ended`` is the silent one and the only silent one: there the agent
    has closed the conversation and the customer has nothing left to answer, so
    the loop sends no further turn. Every other status has to carry a message,
    or the transcript would gain an empty customer turn and the agent would be
    answering silence. A message under ``agent_ended`` is the same fault seen
    from the other side: the loop cannot deliver it, and dropping it would lose
    a customer turn the model did write.
    """
    spoke = bool(reply.message.strip())
    if spoke == (reply.status != "agent_ended"):
        return
    raise UserError(
        f"the simulated user returned {reply.message!r} with status "
        f"{reply.status!r}. 'agent_ended' means the agent closed the dialogue and "
        f"the customer has nothing to add, so it is the one status that comes "
        f"empty, and the only one that may: fix the prompt of "
        f"data/prompts/{PROMPT}.md if the model keeps mixing the two"
    )


def _check_the_agent_spoke_last(history: Sequence[Turn]) -> None:
    """Fail unless the customer is the one whose turn it is."""
    if history and history[-1].speaker == "user":
        raise UserError(
            "the last turn of the history is the customer's, so the simulated user "
            "would be answering itself. Append the agent's reply first; two "
            "customer turns in a row mean the loop lost one"
        )
