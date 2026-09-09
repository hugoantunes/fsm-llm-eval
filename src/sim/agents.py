"""The agents under comparison: the common interface and the baseline (T-09).

Both agents answer through :meth:`Agent.respond`, produce the same
:class:`~sim.schemas.TurnRecord` and share one block of persona, tone and general
rules. What differs is the structure of the instruction: the baseline carries the
whole knowledge base from the first message, while the FSM agent of T-10 carries
the package of the current state and only the facts released for it. That
difference is the experiment, not a bug.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sim.kb import Fact, KnowledgeBase, UserDataField
from sim.llm import Chat
from sim.prompts import DEFAULT_PROMPTS_DIR, load_prompt
from sim.schemas import Turn, TurnRecord, as_messages

#: The block both agents include, word for word: persona, tone, general rules.
SHARED_PROMPT = "agent_shared"


class AgentError(RuntimeError):
    """An agent was asked for something it cannot answer; the message says what."""


@dataclass(frozen=True)
class Instruction:
    """What an agent decided to send this turn, and how it got there.

    The baseline sends the same system prompt every turn and leaves the rest
    empty; the FSM agent of T-10 fills the state and the event it detected.
    """

    system_prompt: str
    state_before: str | None = None
    state_after: str | None = None
    event: str | None = None


class Agent(ABC):
    """What the runner of T-14a talks to, and what both agents implement.

    The turn is timed, called and recorded here, once, so the two agents cannot
    drift apart in how they are measured. Only :meth:`instruct` differs.
    """

    #: The agent's name, as the run directory and the LLM log record it.
    name: str

    def __init__(self, llm: Chat, *, seed: int | None = None) -> None:
        self._llm = llm
        self._seed = seed

    def respond(self, history: Sequence[Turn]) -> TurnRecord:
        """Answer the conversation in ``history``, whose last turn is the user's.

        Raises:
            AgentError: when the history does not end with a user turn, or when
                the model returns a blank reply. The customer always speaks first
                and the agent answers one message at a time, so anything else is
                the runner losing a turn; a blank agent turn would reach the
                judge as a turn the agent took and ask the customer to answer
                silence.
        """
        _check_the_user_spoke_last(history)
        started = time.perf_counter()
        instruction = self.instruct(history)
        answer = self._llm.chat(
            as_messages(instruction.system_prompt, history, speaking_as="agent"),
            role="agent",
            caller=self.name,
            seed=self._seed,
        )
        _check_the_reply_is_not_blank(answer.text)
        return TurnRecord(
            turn=sum(1 for turn in history if turn.speaker == "user"),
            user_message=history[-1].text,
            agent_reply=answer.text,
            model=answer.model,
            prompt_hash=answer.prompt_hash,
            prompt_tokens=answer.prompt_tokens,
            output_tokens=answer.output_tokens,
            llm_latency_s=answer.latency_s,
            turn_latency_s=time.perf_counter() - started,
            cached=answer.cached,
            state_before=instruction.state_before,
            state_after=instruction.state_after,
            event=instruction.event,
        )

    @abstractmethod
    def instruct(self, history: Sequence[Turn]) -> Instruction:
        """Build the instruction this turn is answered with."""


class BaselineAgent(Agent):
    """One well-written system prompt with the whole knowledge base in it."""

    name = "baseline"

    def __init__(
        self,
        llm: Chat,
        *,
        kb: KnowledgeBase,
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
        seed: int | None = None,
    ) -> None:
        super().__init__(llm, seed=seed)
        self.system_prompt = load_prompt("baseline", directory=prompts_dir).render(
            shared=load_prompt(SHARED_PROMPT, directory=prompts_dir).template,
            user_data_fields=render_user_data_fields(kb.user_data_fields),
            knowledge_base=render_facts(kb.facts),
        )

    def instruct(self, history: Sequence[Turn]) -> Instruction:
        """Send the one system prompt, whatever the conversation has reached.

        The baseline has no state to track: everything it may need was in the
        prompt from the first message, which is the arrangement under test.
        """
        return Instruction(system_prompt=self.system_prompt)


def render_facts(facts: Iterable[Fact]) -> str:
    """Render KB facts for a prompt, grouped by intent, in the file's own shape.

    Both agents render facts through here, so the only difference between their
    prompts is *which* facts are in them: all of them for the baseline, the ones
    the current state releases for the FSM agent (T-10).
    """
    lines: list[str] = []
    intent: str | None = None
    for fact in facts:
        if fact.intent != intent:
            intent = fact.intent
            lines.extend(["", f"## {intent}", ""])
        lines.append(f"- **{fact.id}** — {fact.text}")
    return "\n".join(lines).strip()


def render_user_data_fields(fields: Sequence[UserDataField]) -> str:
    """Render the data the agent has to collect, with the requests that need it."""
    return "\n".join(
        f"- **{field.label}** — required for "
        f"{', '.join(field.required_for)} (for example: {field.example})"
        for field in fields
    )


def _check_the_reply_is_not_blank(text: str) -> None:
    """Fail unless the agent said something the customer can answer."""
    if text.strip():
        return
    raise AgentError(
        f"the agent returned a blank reply {text!r}. A blank agent turn reaches "
        f"the judge as a turn the agent took, and the simulated user would be "
        f"asked to answer silence. A thinking-mode model cut off by num_predict "
        f"or forced to think=False can emit this; it is not a turn"
    )


def _check_the_user_spoke_last(history: Sequence[Turn]) -> None:
    """Fail unless the last turn of ``history`` is the user's."""
    if not history:
        raise AgentError(
            "the agent was asked to answer an empty history. The customer opens "
            "every dialogue of this experiment, so there is nothing to answer yet"
        )
    if history[-1].speaker != "user":
        raise AgentError(
            f"the last turn of the history is the {history[-1].speaker}'s, so the "
            f"agent would be answering itself. Append the user's turn first; two "
            f"agent turns in a row mean the runner lost one"
        )
