"""Play one dialogue between the simulated user and an agent (T-11).

The loop alternates the two sides and stops on one of four conditions: the
customer got what it came for, the customer gave up, the agent closed the
conversation, or the turn budget of the scenario ran out. It is the same loop for
both agents, and it holds no answer key: the brief lives in the simulated user
and the knowledge base in the agent.

The runner of T-14a wraps this with the CLI, the JSONL per dialogue, the
manifest, ``--resume`` and the thread pool. None of that belongs here.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from sim.agents import Agent
from sim.schemas import Turn, TurnRecord
from sim.user import SimulatedUser

#: Why a dialogue ended. T-13 reads it as part of efficiency, and the pilot of
#: T-15 compares its distribution between the agents: a stopping rule that fired
#: earlier for one of them would truncate that agent's dialogues and flatter it.
StopReason = Literal["goal_reached", "user_gave_up", "agent_closed", "max_turns"]


class DialogueError(RuntimeError):
    """The dialogue cannot continue; the message says why."""


class DialogueResult(BaseModel):
    """One dialogue as it happened, and why it ended.

    ``records`` is what the metrics of T-13 measure; ``transcript`` flattens
    them into the turns the judge of T-12 reads. The simulated user's own
    replies are turns of that transcript and nothing more: it is not evaluated.
    """

    model_config = ConfigDict(extra="forbid")

    records: list[TurnRecord]
    stop_reason: StopReason

    @property
    def transcript(self) -> list[Turn]:
        """The user and agent turns, flattened from ``records``."""
        return [
            turn
            for record in self.records
            for turn in (
                Turn(speaker="user", text=record.user_message),
                Turn(speaker="agent", text=record.agent_reply),
            )
        ]


def run_dialogue(
    agent: Agent, user: SimulatedUser, *, max_turns: int
) -> DialogueResult:
    """Let ``user`` and ``agent`` talk until one of the four stops fires.

    The agent always answers the customer's last message, so a dialogue ends on
    the agent's turn: it is what lets the conversation reach ``closing``, which
    the flow adherence of T-13 asks about.

    Args:
        agent: the agent under test, baseline or FSM.
        user: the simulated user, holding the scenario's brief.
        max_turns: the scenario's turn budget, counted in agent turns.

    Raises:
        DialogueError: when the user declares the agent closed the conversation
            before the agent has spoken. ``agent_closed`` on an empty transcript
            would flatter that agent in the stop-reason distribution of T-15.
    """
    transcript: list[Turn] = []
    records: list[TurnRecord] = []
    stop: StopReason | None = None
    while stop is None:
        reply = user.speak(transcript)
        if reply.status == "agent_ended":
            if not transcript:
                raise DialogueError(
                    "the simulated user declared agent_ended on an empty "
                    "transcript. The agent never spoke, so it cannot have closed "
                    "the dialogue; agent_closed on an empty transcript would "
                    "flatter that agent in the stop-reason distribution of T-15"
                )
            stop = "agent_closed"
            break
        transcript.append(Turn(speaker="user", text=reply.message))
        record = agent.respond(transcript)
        records.append(record)
        transcript.append(Turn(speaker="agent", text=record.agent_reply))
        if reply.status != "continue":
            stop = "goal_reached" if reply.status == "goal_reached" else "user_gave_up"
        elif len(records) >= max_turns:
            stop = "max_turns"
    return DialogueResult(records=records, stop_reason=stop)
