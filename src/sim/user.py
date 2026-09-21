"""The simulated user that plays a scenario against an agent (T-11).

It sees three fields of the scenario and no more: the persona, the goal and the
script. The knowledge base, the reference answer, the required facts and the
success criterion belong to the evaluation side, and any of them in this prompt
would leak the answer key into the dialogue and from there into the judge.

The script is a mandatory ordered plan, not context: the customer owes one beat
per message and the dialogue may not end while a beat is still owed. Which beat
that is lives in :class:`~sim.script.ScriptProgress` and is rendered into every
turn's prompt, because the first version of this module sent the whole numbered
script and let the model infer its own position - the instrument bug the pilot
of 2026-09-11 exposed (``docs/pilot.md``).

Who decides what, and why, is the whole of the fix:

- **The runtime decides delivery.** A beat is consumed by a message that meets
  its contract (:class:`~sim.script.Beat`) and by nothing else - not by the
  model having written something while that beat was due, and not by what the
  model says about its own turn. Asked whether it had delivered a beat, a 4B
  answered no while sending that beat's own words, and answered yes while
  sending the next beat's.
- **The runtime decides stopping.** A reply that closes the dialogue with beats
  owed is unusable and is retried with a bumped seed. Retries are for invalid
  candidates only: a valid partial turn is committed to the dialogue and the
  same beat remains active for the next turn. If every retry attempt is invalid,
  :class:`SimulatedUserGenerationError` is raised and the run records an
  instrument failure instead of forcing synthetic continuation.

The simulated user is not evaluated. Its calls are logged under its own name and
its replies are plain user turns, never the :class:`~sim.schemas.TurnRecord` the
metrics of T-13 read.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from sim.kb import UserDataField
from sim.llm import Chat
from sim.prompts import DEFAULT_PROMPTS_DIR, load_prompt
from sim.schemas import Scenario, Turn, as_messages
from sim.script import (
    NOTHING,
    Beat,
    BeatAssessment,
    ScriptProgress,
    beats_of,
    render_beat,
    render_pending_local,
    render_plan,
    render_remaining_keywords,
    render_remaining_literals,
    render_satisfied_cumulative,
)

logger = logging.getLogger(__name__)

#: The prompt file of ``data/prompts/``, holding the brief and nothing else.
PROMPT = "simulated_user"

#: Times the model is asked again when a candidate reply is unusable. Beat
#: incompleteness alone does not trigger retries: valid partial turns are
#: committed and progress carries to the next turn. The seed is bumped per
#: attempt, as in :func:`~sim.llm.chat_parsed`, or the same prompt would replay
#: the same invalid candidate.
BEAT_RETRIES = 3

#: Extra invalid-candidate attempts for an injection beat.
INJECTION_RETRIES = 8

#: The two answers the prompt reads as a flag, so no English is built here.
YES, NO = "yes", "no"

#: Where the customer thinks the dialogue stands after the message it just wrote.
#: Three of the four stopping conditions of T-11 are these; the fourth is the
#: turn budget, which the loop of :mod:`sim.dialogue` owns.
UserStatus = Literal["continue", "goal_reached", "gave_up", "agent_ended"]


class UserError(RuntimeError):
    """The simulated user cannot speak here; the message says why."""


class SimulatedUserGenerationError(UserError):
    """The simulated user exhausted retries on invalid candidates."""

    failure_reason = "invalid_candidate_retry_exhausted"

    def __init__(
        self,
        *,
        scenario_id: str,
        attempts: int,
        invalid_reason: str,
        beat_diagnostics: dict[str, object] | None,
    ) -> None:
        self.attempts = attempts
        self.invalid_reason = invalid_reason
        self.beat_diagnostics = beat_diagnostics
        details = (
            f"{scenario_id}: exhausted {attempts} invalid candidate attempt(s): "
            f"{invalid_reason}"
        )
        if beat_diagnostics:
            details += f" | active beat diagnostics: {beat_diagnostics}"
        super().__init__(details)


class UserReply(BaseModel):
    """One customer message, what it is for, and where the dialogue stands.

    ``answering_agent_question`` is the only thing the model is asked about its
    own message, and it is asked negatively on purpose: a message delivers the
    beat that is due unless it answers something the agent asked that the beat
    does not cover. Claiming otherwise postpones the beat and never drops it.

    ``status`` is what stops a dialogue (T-11): three of the four stopping
    conditions are decided here and ``max_turns`` is decided by the loop. It is a
    stopping condition and never a metric - whether the task was completed is the
    judge's call in T-12 - because a simulated user grading its own dialogue
    would be measuring the instrument.
    """

    model_config = ConfigDict(extra="forbid")

    message: str
    answering_agent_question: bool
    status: UserStatus


@dataclass(frozen=True)
class UserTurn:
    """One accepted customer message and the beat number it delivered, if any.

    ``beat`` is ``None`` on a message that only answered the agent, and it is
    what :class:`~sim.schemas.TurnRecord` stores so the delivery of a script can
    be audited after the run.
    """

    message: str
    status: UserStatus
    beat: int | None
    beat_started_at_turn: int | None = None
    beat_completed_at_turn: int | None = None


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
        data_fields: Sequence[UserDataField],
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
        seed: int | None = None,
    ) -> None:
        self._llm = llm
        self._seed = seed
        self._scenario_id = scenario.id
        self._persona = scenario.user_persona
        self._goal = scenario.user_goal
        self._prompt = load_prompt(PROMPT, directory=prompts_dir)
        self.progress = ScriptProgress(beats_of(scenario, data_fields))

    @property
    def system_prompt(self) -> str:
        """The brief as the model reads it now, with the beat it owes marked."""
        return self._render(insist=self.progress.must_deliver, invalid_reason=NOTHING)

    def speak(self, history: Sequence[Turn]) -> UserTurn:
        """Write the customer's next message, given the dialogue so far.

        ``history`` is the transcript as the agent recorded it, which
        :func:`~sim.schemas.as_messages` mirrors for this side of the
        conversation. The client of T-07 validates the answer against
        :class:`UserReply` before handing it back; what comes out of here is the
        accepted turn, with the script advanced by it.

        Raises:
            UserError: when the history already ends with the customer's turn. The
                customer speaks first and then one message per agent reply, so
                anything else is the loop having lost the agent's turn, and a
                second customer message would paper over it. Also when the beat
                due required content word for word that never arrived, when
                every attempt came back blank, and when an injection beat still
                carried the canary without the agent-directed instruction.
        """
        _check_the_agent_spoke_last(history)
        attempts = _retries_for(self.progress.current) + 1
        invalid_reason = ""
        for attempt in range(attempts):
            reply = self._ask(history, attempt=attempt, invalid_reason=invalid_reason)
            invalid = _invalid_candidate_reason(reply, self.progress)
            if invalid is None:
                return self._accept(reply)
            invalid_reason = invalid
            logger.info(
                "%s: asking the customer again (invalid candidate): %s",
                self._scenario_id,
                invalid_reason,
            )
        raise self._generation_error(
            attempts=attempts,
            invalid_reason=invalid_reason,
        )

    def _ask(
        self, history: Sequence[Turn], *, attempt: int, invalid_reason: str
    ) -> UserReply:
        """Ask the model for one message; a retry insists on the beat and reseeds."""
        answer = self._llm.chat(
            as_messages(
                self._render(
                    insist=attempt > 0 or self.progress.must_deliver,
                    invalid_reason=invalid_reason if attempt > 0 else NOTHING,
                ),
                history,
                speaking_as="user",
            ),
            role="simulated_user",
            caller=self.name,
            schema=UserReply,
            seed=None if self._seed is None else self._seed + attempt,
        )
        return cast(UserReply, answer.parsed)

    def _generation_error(
        self, *, attempts: int, invalid_reason: str
    ) -> SimulatedUserGenerationError:
        """Build the invalid-candidate exhaustion failure."""
        diagnostics = self.progress.active_beat_diagnostics()
        beat_diagnostics = dict(diagnostics) if diagnostics is not None else None
        return SimulatedUserGenerationError(
            scenario_id=self._scenario_id,
            attempts=attempts,
            invalid_reason=invalid_reason,
            beat_diagnostics=beat_diagnostics,
        )

    def _accept(self, reply: UserReply) -> UserTurn:
        """Advance the script by ``reply`` and return the turn the loop sends."""
        turn = self.progress.commit(reply.message)
        if turn.consumed_beat is None:
            return UserTurn(message=reply.message, status=reply.status, beat=None)
        return UserTurn(
            message=reply.message,
            status=reply.status,
            beat=turn.consumed_beat,
            beat_started_at_turn=turn.beat_started_at_turn,
            beat_completed_at_turn=turn.beat_completed_at_turn,
        )

    def _render(self, *, insist: bool, invalid_reason: str) -> str:
        """Fill the brief with the persona, the goal and where the script stands."""
        beat = self.progress.current
        progress = self.progress.current_progress
        remaining_exact = render_remaining_literals(beat, progress=progress)
        remaining_words = render_remaining_keywords(beat, progress=progress)
        return self._prompt.render(
            persona=self._persona,
            goal=self._goal,
            plan=render_plan(self.progress),
            beat=render_beat(beat),
            required_exactly=remaining_exact,
            required_words=remaining_words,
            satisfied_cumulative=render_satisfied_cumulative(beat, progress=progress),
            pending_local=render_pending_local(beat),
            retry_invalid_reason=invalid_reason,
            remaining_cumulative=_remaining_requirements(
                required_exactly=remaining_exact, required_words=remaining_words
            ),
            must_ask=YES if beat is not None and beat.asks else NO,
            must_deny=YES if beat is not None and beat.negates else NO,
            deliver_now=YES if insist else NO,
            beats_left=str(max(len(self.progress.pending) - 1, 0)),
        )


def _retries_for(beat: Beat | None) -> int:
    """Return how many times a missed ``beat`` may be asked again."""
    if beat is not None and beat.verbatim is not None:
        return INJECTION_RETRIES
    return BEAT_RETRIES


def _invalid_candidate_reason(reply: UserReply, progress: ScriptProgress) -> str | None:
    """Return why ``reply`` is unusable as a delivered turn, or ``None``."""
    mismatch = _status_mismatch(reply)
    if mismatch is not None:
        return mismatch
    beat = progress.current
    if beat is None:
        return None
    assessment = progress.assess(reply.message)
    assert assessment is not None
    closing = reply.status != "continue"
    if closing and not assessment.complete:
        return (
            f"beat {beat.number}: the reply reported {reply.status!r} with "
            f"incomplete requirements still owed"
        )
    if reply.status == "goal_reached":
        return None
    if closing and not _completes_script(progress, assessment):
        return (
            f"beat {beat.number}: the reply reported {reply.status!r} before "
            "all mandatory beats were delivered"
        )
    return None


def _remaining_requirements(*, required_exactly: str, required_words: str) -> str:
    """Render one list with remaining cumulative requirements."""
    lines: list[str] = []
    if required_exactly != NOTHING:
        lines.extend(required_exactly.splitlines())
    if required_words != NOTHING:
        lines.extend(required_words.splitlines())
    if not lines:
        return NOTHING
    return "\n".join(lines)


def _status_mismatch(reply: UserReply) -> str | None:
    """Say how message and status contradict each other, or ``None``.

    ``agent_ended`` is the silent one and the only silent one: there the agent
    has closed the conversation and the customer has nothing left to answer, so
    the loop sends no further turn. Every other status has to carry a message,
    or the transcript would gain an empty customer turn and the agent would be
    answering silence. A message under ``agent_ended`` is the same fault seen
    from the other side: the loop cannot deliver it, and dropping it would lose
    a customer turn the model did write.
    """
    silent = not reply.message.strip()
    ended = reply.status == "agent_ended"
    if silent == ended:
        return None
    return (
        f"a message the loop cannot send: {reply.message!r} with status "
        f"{reply.status!r}. 'agent_ended' means the agent closed the dialogue "
        f"and the customer has nothing to add, so it is the one status that "
        f"comes empty, and the only one that may"
    )


def _completes_script(progress: ScriptProgress, assessment: BeatAssessment) -> bool:
    """Whether accepting this candidate would complete every mandatory beat."""
    beat = progress.current
    if beat is None:
        return True
    if not assessment.complete:
        return False
    return beat.number == len(progress.beats)


def _check_the_agent_spoke_last(history: Sequence[Turn]) -> None:
    """Fail unless the customer is the one whose turn it is."""
    if history and history[-1].speaker == "user":
        raise UserError(
            "the last turn of the history is the customer's, so the simulated user "
            "would be answering itself. Append the agent's reply first; two "
            "customer turns in a row mean the loop lost one"
        )
