"""The runtime finite-state machine of T-08.

``machine.yaml`` is loaded by :mod:`sim.fsm`; this module is the session that
walks it: current state, collected slots, classified intent, and the history of
each attempted transition. :mod:`sim.events` classifies the user turn that
feeds :meth:`FsmEngine.apply`.
"""

from pathlib import Path

from transitions import Machine

from sim.events import (
    NONE,
    Detection,
    detect_user_event,
    machine_rules,
    patterned_keys_required_for_every_intent,
    required_data_held,
)
from sim.fsm import FsmError, FsmSpec, Transition
from sim.kb import KnowledgeBase
from sim.llm import Chat
from sim.prompts import DEFAULT_PROMPTS_DIR
from sim.schemas import FiredBy, TransitionRecord


class FsmEngine:
    """One dialogue's walk through the machine declared in ``machine.yaml``.

    Accepting states are not terminal: ``out_of_scope`` still leaves for
    identification or closing. A failed guard or a ``none`` event keeps the
    dialogue where it is and records ``valid=False``. ``history`` is every edge
    of the whole dialogue, in order, each one saying whether the customer fired
    it or the engine advanced on its own.
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
        self._rules = machine_rules(spec, kb.user_data_fields, kb.intents())
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
    ) -> tuple[TransitionRecord, ...]:
        """Detect the user event in ``user_message`` and walk what it opens.

        Returns every edge this turn attempted, in order: the customer's own
        event first, fired or refused, then whatever the machine could advance
        on its own. Each one is an edge of ``machine.yaml``, so the log keeps a
        real path rather than the endpoints of a jump (T-13).
        """
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
        self._remember_intent(detection)
        walked = [self.apply(detection.event, turn=turn)]
        walked.extend(self._advance_when_ready(turn))
        return tuple(walked)

    def apply(
        self, event: str, *, turn: int, fired_by: FiredBy = "user"
    ) -> TransitionRecord:
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
            fired_by=fired_by,
        )
        self.history.append(record)
        return record

    def _remember_intent(self, detection: Detection) -> None:
        """Keep the first request named, and revise it only where it is settled.

        The classifier reports an intent on any event, which is how a request
        named in ``greeting`` survives to the state that acts on it. Letting any
        later event overwrite it would swap the data the
        ``required_data_collected`` guard and the field list read, with nothing
        in the log to explain the change, so only the intent event itself may
        change it.
        """
        if detection.intent is None:
            return
        if self.intent is None or detection.event == self._rules.intent_event:
            self.intent = detection.intent

    def _advance_when_ready(self, turn: int) -> list[TransitionRecord]:
        """Leave every state that has nothing left to do after the user event.

        Identification already complete, every required datum already in hand:
        the agent must speak from the first state that still has work, or it
        asks for what it already holds. The edges are the machine's own, tried
        in the order a dialogue reaches them.

        ``intent_classified`` is deliberately not among them. A request the
        classifier read wrong in the opening turn would be final, because the
        one state built to settle it would never be spoken from; the classifier
        is asked there instead, with the whole transcript in front of it.
        """
        ready = (
            (self._rules.identify_event, self.order_and_email_present()),
            (self._rules.collect_event, self.required_data_collected()),
        )
        walked: list[TransitionRecord] = []
        for event, holds in ready:
            if holds and event in self._spec.events_for(self.state):
                walked.append(self.apply(event, turn=turn, fired_by="engine"))
        return walked

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
        return required_data_held(
            self._kb.user_data_fields, self.intent, self.collected
        )


def _machine_edge(edge: Transition) -> dict[str, str]:
    """Turn a spec edge into the dict ``transitions.Machine`` consumes."""
    row = {"trigger": edge.event, "source": edge.source, "dest": edge.dest}
    if edge.guard is not None:
        row["conditions"] = edge.guard
    return row
