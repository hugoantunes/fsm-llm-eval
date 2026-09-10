"""The agents under comparison: the common interface, the baseline and the FSM.

Both agents answer through :meth:`Agent.respond`, produce the same
:class:`~sim.schemas.TurnRecord` and share one block of persona, tone and general
rules. Both carry the same full knowledge base. What differs is the structure of
the instruction: the baseline is one prompt, the FSM agent is a per-state
instruction package, explicit states, transitions, guards and event
classification. That difference is the experiment, not a bug.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sim.engine import FsmEngine
from sim.fsm import DEFAULT_FSM_DIR, FsmSpec
from sim.kb import KnowledgeBase, UserDataField, render_facts
from sim.llm import Chat
from sim.prompts import DEFAULT_PROMPTS_DIR, load_prompt
from sim.schemas import (
    TransitionRecord,
    Turn,
    TurnRecord,
    as_messages,
    render_transcript,
)

#: The block both agents include, word for word: persona, tone, general rules.
SHARED_PROMPT = "agent_shared"

#: The FSM agent's prompt file in ``data/prompts/``.
FSM_TEMPLATE = "fsm_template"

#: The guard whose source is the state that still has data to collect. The name
#: is declared once in ``machine.yaml``; this is the same string the engine binds.
_COLLECT_GUARD = "required_data_collected"


class AgentError(RuntimeError):
    """An agent was asked for something it cannot answer; the message says what."""


@dataclass(frozen=True)
class Instruction:
    """What an agent decided to send this turn, and how it got there.

    The baseline sends the same system prompt every turn and leaves the rest
    empty; the FSM agent of T-10 fills the state, the event it detected and the
    edges that event walked.
    """

    system_prompt: str
    state_before: str | None = None
    state_after: str | None = None
    event: str | None = None
    transitions: tuple[TransitionRecord, ...] = ()


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
            transitions=list(instruction.transitions),
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
            user_data_fields=render_user_data_fields(
                kb.user_data_fields, prompts_dir=prompts_dir
            ),
            knowledge_base=render_facts(kb.facts),
        )

    def instruct(self, history: Sequence[Turn]) -> Instruction:
        """Send the one system prompt, whatever the conversation has reached.

        The baseline has no state to track: everything it may need was in the
        prompt from the first message, which is the arrangement under test.
        """
        return Instruction(system_prompt=self.system_prompt)


class FsmAgent(Agent):
    """Per-state instruction package, same full knowledge base as the baseline."""

    name = "fsm"

    def __init__(
        self,
        llm: Chat,
        *,
        kb: KnowledgeBase,
        spec: FsmSpec,
        fsm_dir: Path = DEFAULT_FSM_DIR,
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
        seed: int | None = None,
    ) -> None:
        super().__init__(llm, seed=seed)
        self._kb = kb
        self._prompts_dir = prompts_dir
        self._shared = load_prompt(SHARED_PROMPT, directory=prompts_dir).template
        self._template = load_prompt(FSM_TEMPLATE, directory=prompts_dir)
        self._knowledge_base = render_facts(kb.facts)
        self._packages = {
            name: (fsm_dir / state.package).read_text(encoding="utf-8")
            for name, state in spec.states.items()
        }
        self._collect_state = _data_collection_state(spec)
        self._engine = FsmEngine(
            spec, kb=kb, llm=llm, prompts_dir=prompts_dir, seed=seed
        )

    def instruct(self, history: Sequence[Turn]) -> Instruction:
        """Detect the user event, transition, and answer from the new state.

        The customer's turn moves the machine; the agent then speaks from
        ``state_after``, with that state's package and the same knowledge base
        the baseline carries. One turn can walk more than one edge, so the
        whole walk is recorded beside its endpoints: the states in between are
        where the dialogue really went, and nothing else would tell T-13 that
        it was a path.
        """
        source = self._engine.state
        walk = self._engine.step(
            history[-1].text,
            turn=sum(1 for turn in history if turn.speaker == "user"),
            transcript=render_transcript(history),
        )
        dest = self._engine.state
        return Instruction(
            system_prompt=self._render(dest),
            state_before=source,
            state_after=dest,
            event=walk[0].event,
            transitions=walk,
        )

    def _render(self, state: str) -> str:
        """Fill the FSM template for ``state``.

        The package changes with the state; the knowledge base is the full set.
        """
        return self._template.render(
            shared=self._shared,
            state_package=self._packages[state],
            user_data_fields=render_user_data_fields(
                _fields_to_collect(
                    state, self._engine.intent, self._kb, self._collect_state
                ),
                prompts_dir=self._prompts_dir,
            ),
            knowledge_base=self._knowledge_base,
        )


def render_user_data_fields(
    fields: Sequence[UserDataField],
    *,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> str:
    """Render the data the agent has to collect, from the versioned row template."""
    row = load_prompt("user_data_field", directory=prompts_dir)
    return "\n".join(
        row.render(
            label=field.label,
            required_for=", ".join(field.required_for),
            example=field.example,
        )
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


def _data_collection_state(spec: FsmSpec) -> str:
    """Return the state whose exit is guarded by ``required_data_collected``."""
    matches = [
        edge.source
        for edge in spec.transitions
        if edge.guard == _COLLECT_GUARD and not edge.from_any
    ]
    if len(matches) != 1:
        raise AgentError(
            f"machine.yaml must have exactly one transition with guard "
            f"{_COLLECT_GUARD!r} (found {len(matches)}). The FSM agent keys the "
            f"user-data field list off that state so a rename cannot silently "
            f"leave data_collection without the list the package says comes with it"
        )
    return matches[0]


def _fields_to_collect(
    state: str,
    intent: str | None,
    kb: KnowledgeBase,
    collect_state: str,
) -> list[UserDataField]:
    """Return the fields the classified intent still needs, only in collect_state."""
    if state != collect_state or intent is None:
        return []
    return [field for field in kb.user_data_fields if intent in field.required_for]


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
