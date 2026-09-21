"""Play one dialogue between the simulated user and an agent (T-11).

The loop alternates the two sides and stops on one of four conditions: the
customer got what it came for, the customer gave up, the agent closed the
conversation, or the turn budget of the scenario ran out. It is the same loop for
both agents, and it holds no answer key: the brief lives in the simulated user
and the knowledge base in the agent.

The first three stops are the customer's and are guarded where the script lives,
in :mod:`sim.user`: a dialogue may not end while a mandatory beat is still owed.
The loop records which beat each customer message delivered on the turn it
belongs to, and stops on ``max_turns`` whatever the script still holds.

The runner of T-14a wraps this with the CLI, the JSONL per dialogue, the
manifest, ``--resume`` and the thread pool. None of that belongs here.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from sim.agents import Agent, AgentError
from sim.events import EventError
from sim.fsm import FsmError
from sim.llm import LlmError
from sim.schemas import StopReason, Turn, TurnRecord, transcript_from_records
from sim.user import SimulatedUser, UserError

_OPERATIONAL = (LlmError, AgentError, UserError, EventError, FsmError)


class DialogueError(RuntimeError):
    """The dialogue cannot continue; the message says why.

    ``records`` are the completed agent turns. A turn that died before the
    agent answered is not stored as a fake reply.
    """

    def __init__(self, message: str, *, records: Sequence[TurnRecord] = ()) -> None:
        super().__init__(message)
        self.records = list(records)


class DialogueResult(BaseModel):
    """One dialogue as it happened, and why it ended.

    ``records`` is what the metrics of T-13 measure; ``transcript`` flattens
    them into the turns the judge of T-12 reads. The simulated user's own
    replies are turns of that transcript and nothing more: it is not evaluated.
    """

    model_config = ConfigDict(extra="forbid")

    records: list[TurnRecord]
    stop_reason: StopReason
    goal_reached_seen: bool = False
    goal_reached_at_turn: int | None = None
    script_complete_at_goal_reached: bool | None = None

    @property
    def transcript(self) -> list[Turn]:
        """The user and agent turns, flattened from ``records``."""
        return transcript_from_records(self.records)


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
            before the agent has spoken, or when an operational failure stops a
            later turn. ``records`` on the error are the turns already played.
    """
    transcript: list[Turn] = []
    records: list[TurnRecord] = []
    stop: StopReason | None = None
    goal_reached_at_turn: int | None = None
    script_complete_at_goal_reached: bool | None = None
    try:
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
            record = agent.respond(transcript).model_copy(
                update={
                    "user_beat": reply.beat,
                    "beat_started_at_turn": reply.beat_started_at_turn,
                    "beat_completed_at_turn": reply.beat_completed_at_turn,
                }
            )
            records.append(record)
            transcript.append(Turn(speaker="agent", text=record.agent_reply))
            script_complete = user.progress.complete
            if reply.status == "goal_reached":
                if goal_reached_at_turn is None:
                    goal_reached_at_turn = len(records)
                    script_complete_at_goal_reached = script_complete
                if script_complete:
                    stop = "goal_reached"
            elif reply.status == "gave_up" and script_complete:
                stop = "user_gave_up"
            if stop is None and goal_reached_at_turn is not None and script_complete:
                stop = "goal_reached"
            elif stop is None and len(records) >= max_turns:
                stop = "max_turns"
    except DialogueError:
        raise
    except _OPERATIONAL as failure:
        raise DialogueError(str(failure), records=records) from failure
    return DialogueResult(
        records=records,
        stop_reason=stop,
        goal_reached_seen=goal_reached_at_turn is not None,
        goal_reached_at_turn=goal_reached_at_turn,
        script_complete_at_goal_reached=script_complete_at_goal_reached,
    )
