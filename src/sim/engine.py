"""The runtime finite-state machine of T-08.

``machine.yaml`` is loaded by :mod:`sim.fsm`; this module is the session that
walks it: current state, collected slots, classified intent, and the history of
each attempted transition. :mod:`sim.events` classifies the user turn that
feeds :meth:`FsmEngine.apply`.
"""

from dataclasses import dataclass
from pathlib import Path

from transitions import Machine

from sim.events import NONE, detect_user_event, patterned_keys_required_for_every_intent
from sim.fsm import FsmError, FsmSpec, Transition
from sim.kb import KnowledgeBase
from sim.llm import Chat
from sim.prompts import DEFAULT_PROMPTS_DIR


@dataclass(frozen=True)
class TransitionRecord:
    """One attempted edge: whether it fired, and the states around it."""

    turn: int
    source: str
    dest: str
    event: str
    valid: bool


class FsmEngine:
    """One dialogue's walk through the machine declared in ``machine.yaml``.

    Accepting states are not terminal: ``out_of_scope`` still leaves for
    identification or closing. A failed guard or a ``none`` event keeps the
    dialogue where it is and records ``valid=False``.
    """

    def __init__(
        self,
        spec: FsmSpec,
        *,
        kb: KnowledgeBase,
        llm: Chat | None = None,
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
        seed: int | None = None,
    ) -> None:
        self._spec = spec
        self._kb = kb
        self._llm = llm
        self._prompts_dir = prompts_dir
        self._seed = seed
        self.collected: dict[str, str] = {}
        self.intent: str | None = None
        self.history: list[TransitionRecord] = []
        self.state: str
        self._machine = Machine(
            model=self,
            states=list(spec.states),
            initial=spec.initial,
            auto_transitions=False,
            ignore_invalid_triggers=True,
            transitions=[_machine_edge(edge) for edge in spec.transitions],
        )

    def park(self, state: str) -> None:
        """Move to ``state`` without recording a transition.

        Tests arm a single edge this way; a dialogue never needs it, because it
        always starts at the initial state and walks from there.
        """
        if state not in self._spec.states:
            raise FsmError(
                f"{state!r} is not a state of machine.yaml, whose states are "
                f"{sorted(self._spec.states)}"
            )
        self._machine.set_state(state)

    def step(
        self, user_message: str, *, turn: int, transcript: str | None = None
    ) -> TransitionRecord:
        """Detect the user event in ``user_message`` and apply it."""
        detection = detect_user_event(
            user_message,
            state=self.state,
            spec=self._spec,
            collected=self.collected,
            fields=self._kb.user_data_fields,
            intents=self._kb.intents(),
            intent=self.intent,
            llm=self._llm,
            transcript=transcript if transcript is not None else user_message,
            prompts_dir=self._prompts_dir,
            seed=self._seed,
        )
        self.collected.update(detection.slots)
        if detection.event == "intent_classified":
            self.intent = detection.intent
        return self.apply(detection.event, turn=turn)

    def apply(self, event: str, *, turn: int) -> TransitionRecord:
        """Fire ``event`` from the current state, or stay if it cannot fire."""
        source = self.state
        if event != NONE and event in self._spec.events_for(source):
            getattr(self, event)()
        dest = self.state
        record = TransitionRecord(
            turn=turn,
            source=source,
            dest=dest,
            event=event,
            valid=dest != source,
        )
        self.history.append(record)
        return record

    def order_and_email_present(self) -> bool:
        """Guard: every patterned field every intent requires is in hand."""
        return all(
            key in self.collected
            for key in patterned_keys_required_for_every_intent(
                self._kb.user_data_fields, self._kb.intents()
            )
        )

    def required_data_collected(self) -> bool:
        """Guard: every field the classified intent requires is in hand."""
        if self.intent is None:
            return False
        return all(
            field.key in self.collected
            for field in self._kb.user_data_fields
            if self.intent in field.required_for
        )


def _machine_edge(edge: Transition) -> dict[str, str]:
    """Turn a spec edge into the dict ``transitions.Machine`` consumes."""
    row = {"trigger": edge.event, "source": edge.source, "dest": edge.dest}
    if edge.guard is not None:
        row["conditions"] = edge.guard
    return row
