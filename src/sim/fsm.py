"""Load the finite-state machine of ``data/fsm/`` (T-02).

``machine.yaml`` is the single source of the state and event names: the FSM engine
(T-08), the stage labeler's ``enum`` (T-13) and the Mermaid diagram and facts table of
``docs/fsm.md`` all read them from here instead of repeating them.
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sim.kb import FACT_ID, Fact, KnowledgeBase, load_kb

DEFAULT_FSM_DIR = Path("data/fsm")

#: ``from: "*"`` in ``machine.yaml``: one transition out of every state.
ANY_STATE = "*"

#: The guards a transition may name. T-08 binds each name to a callable that reads
#: the patterns and the ``required_for`` of ``data/kb/user_data_fields.json``, so the
#: rule behind a guard is written once, in the KB.
GUARDS = frozenset({"order_and_email_present", "required_data_collected"})

#: The sections every instruction package answers. Persona, tone and general rules
#: are not among them: those live in the block both agents share (T-09, T-10).
PACKAGE_SECTIONS = (
    "## Goal",
    "## Data to collect or confirm",
    "## The answer must",
    "## Never in this state",
    "## Tone in this state",
)

#: A section heading of a state package. The title of the file is a single ``#``.
_PACKAGE_HEADING = re.compile(r"^##\s+.*$", re.MULTILINE)


class FsmError(ValueError):
    """The FSM files are inconsistent; the message says what to fix."""


class State(BaseModel):
    """One state of the machine and the instruction package it carries."""

    model_config = ConfigDict(extra="forbid")

    package: str
    facts: list[str]
    facts_from_intent: bool = False


class Transition(BaseModel):
    """One edge: the user event that fires it and the guard that allows it."""

    model_config = ConfigDict(extra="forbid")

    event: str
    source: str = Field(alias="from")
    dest: str = Field(alias="to")
    guard: str | None = None
    #: True when the loader expanded this edge out of a ``from: "*"`` declaration.
    from_any: bool = False


class FsmSpec(BaseModel):
    """The machine as declared in ``machine.yaml``, cross-checked against the KB."""

    model_config = ConfigDict(extra="forbid")

    version: int
    initial: str
    accepting_states: list[str]
    states: dict[str, State]
    transitions: list[Transition]

    def events_for(self, state: str) -> list[str]:
        """Return the events that can fire in ``state``, sorted.

        This is the ``enum`` the user-event classifier of T-08 is constrained to,
        so the model can only answer with an event the current state accepts.
        """
        self._check_state(state)
        return sorted({t.event for t in self.transitions if t.source == state})

    def _check_state(self, state: str) -> None:
        """Fail unless ``state`` is one of the machine's states."""
        if state in self.states:
            return
        raise FsmError(
            f"{state!r} is not a state of machine.yaml, whose states are "
            f"{sorted(self.states)}. An unknown state would give the classifier "
            f"of T-08 an empty enum and release no fact at all"
        )

    def released_facts(
        self, state: str, intent: str | None, kb: KnowledgeBase
    ) -> list[Fact]:
        """Return the KB facts the FSM agent may state in ``state``, in ID order.

        This filter is what makes the FSM agent's context differ from the
        baseline's full knowledge base, a deliberate difference of the experiment.
        A state marked ``facts_from_intent`` also gets the whole section of the
        classified ``intent``, and demands one: ``intent`` is ``None`` only in the
        states the dialogue passes through before classification.
        """
        self._check_state(state)
        declaration = self.states[state]
        released = set(declaration.facts)
        if declaration.facts_from_intent:
            if intent is None:
                raise FsmError(
                    f"state {state!r} releases the facts of the classified intent, "
                    f"so it cannot be entered without one. Passing none would "
                    f"quietly leave the agent with the general facts alone"
                )
            from_intent = {fact.id for fact in kb.facts if fact.intent == intent}
            if not from_intent:
                raise FsmError(
                    f"state {state!r} releases the facts of the intent "
                    f"{intent!r}, which is not a section of knowledge_base.md. "
                    f"An unknown intent would silently release nothing"
                )
            released |= from_intent
        return [fact for fact in kb.facts if fact.id in released]

    def to_mermaid(self) -> str:
        """Render the machine as the Mermaid diagram of ``docs/fsm.md``.

        The edges declared out of every state become a note on their destination
        instead of one arrow per state, which is what keeps the figure readable.
        """
        lines = ["stateDiagram-v2", f"    [*] --> {self.initial}"]
        for transition in self.transitions:
            if transition.from_any:
                continue
            guard = f" [{transition.guard}]" if transition.guard else ""
            lines.append(
                f"    {transition.source} --> {transition.dest} : "
                f"{transition.event}{guard}"
            )
        lines.extend(f"    {state} --> [*]" for state in self.accepting_states)
        for dest, events in self._events_from_any().items():
            lines.append(f"    note right of {dest}")
            lines.append(f"        from every state: {', '.join(events)}")
            lines.append("    end note")
        return "\n".join(lines)

    def to_facts_table(self) -> str:
        """Render the per-state fact slices as the Markdown table of ``docs/fsm.md``."""
        intent_note = " + the section of the classified request"
        rows = ["| State | Facts released |", "|---|---|"]
        for name, state in self.states.items():
            facts = ", ".join(state.facts)
            if state.facts_from_intent:
                facts = f"{facts}{intent_note}"
            rows.append(f"| `{name}` | {facts} |")
        return "\n".join(rows)

    def _events_from_any(self) -> dict[str, list[str]]:
        """Return the events declared out of every state, by destination."""
        by_dest: dict[str, list[str]] = {}
        for transition in self.transitions:
            if not transition.from_any:
                continue
            events = by_dest.setdefault(transition.dest, [])
            if transition.event not in events:
                events.append(transition.event)
        return by_dest


def load_fsm(
    directory: Path = DEFAULT_FSM_DIR, *, kb: KnowledgeBase | None = None
) -> FsmSpec:
    """Load and cross-check the machine stored in ``directory``."""
    if kb is None:
        kb = load_kb()
    raw = yaml.safe_load((directory / "machine.yaml").read_text(encoding="utf-8"))
    try:
        spec = FsmSpec.model_validate(raw)
    except ValidationError as invalid:
        raise FsmError(
            f"machine.yaml does not match the schema of sim.fsm:\n{invalid}\n"
            f"A key the schema does not know is refused rather than ignored: a "
            f"misspelled 'guard' would silently drop the guard"
        ) from invalid
    _check_declared_states(spec)
    _check_transition_states(spec)
    _check_guards(spec)
    spec = spec.model_copy(update={"transitions": _expand_wildcards(spec)})
    _check_unambiguous_events(spec)
    _check_reachability(spec)
    _check_released_facts(spec, kb)
    _check_packages(spec, directory)
    return spec


def _check_declared_states(spec: FsmSpec) -> None:
    """Fail unless the initial and the accepting states are declared states."""
    roles = [("initial", spec.initial)]
    roles += [("accepting_states", state) for state in spec.accepting_states]
    for role, state in roles:
        if state in spec.states:
            continue
        raise FsmError(
            f"{role} names {state!r}, which machine.yaml does not declare under "
            f"'states'. The scenarios of T-05 and the flow adherence of T-13 read "
            f"these names from here"
        )


def _check_transition_states(spec: FsmSpec) -> None:
    """Fail unless every transition names declared states."""
    for transition in spec.transitions:
        endpoints = [transition.dest]
        if transition.source != ANY_STATE:
            endpoints.append(transition.source)
        for state in endpoints:
            if state in spec.states:
                continue
            raise FsmError(
                f"transition on {transition.event!r} names the state {state!r}, "
                f"which machine.yaml does not declare. Declare it under 'states' "
                f"or fix the transition"
            )


def _check_guards(spec: FsmSpec) -> None:
    """Fail unless every guard is one the engine knows how to evaluate."""
    for transition in spec.transitions:
        if transition.guard is None or transition.guard in GUARDS:
            continue
        raise FsmError(
            f"transition on {transition.event!r} guards with "
            f"{transition.guard!r}, which is not one of {sorted(GUARDS)}. "
            f"Add it to GUARDS in sim/fsm.py and implement it in the engine"
        )


def _expand_wildcards(spec: FsmSpec) -> list[Transition]:
    """Replace every ``from: "*"`` transition by one per state, minus self-loops."""
    expanded: list[Transition] = []
    for transition in spec.transitions:
        if transition.source != ANY_STATE:
            expanded.append(transition)
            continue
        expanded.extend(
            transition.model_copy(update={"source": state, "from_any": True})
            for state in spec.states
            if state != transition.dest
        )
    return expanded


def _check_unambiguous_events(spec: FsmSpec) -> None:
    """Fail unless an event in a state leads to exactly one destination.

    Two edges sharing a state and an event, whether declared or expanded out of a
    ``from: "*"``, would let the engine pick a destination and would collapse into
    one entry of the classifier's enum, hiding the choice.
    """
    seen: dict[tuple[str, str], str] = {}
    for transition in spec.transitions:
        key = (transition.source, transition.event)
        taken = seen.setdefault(key, transition.dest)
        if taken == transition.dest:
            continue
        raise FsmError(
            f"in state {transition.source!r} the event {transition.event!r} leads "
            f"both to {taken!r} and to {transition.dest!r}. Give one of them its "
            f"own event, or narrow the 'from: \"*\"' that produced it"
        )


def _check_reachability(spec: FsmSpec) -> None:
    """Fail unless every state can be reached from the initial one."""
    reached = {spec.initial}
    frontier = [spec.initial]
    while frontier:
        state = frontier.pop()
        for transition in spec.transitions:
            if transition.source != state or transition.dest in reached:
                continue
            reached.add(transition.dest)
            frontier.append(transition.dest)
    orphans = sorted(set(spec.states) - reached)
    if orphans:
        raise FsmError(
            f"no path from {spec.initial!r} to {orphans}: a state nothing "
            f"transitions into can never be visited. Add a transition into it "
            f"or drop it from machine.yaml"
        )


def _check_released_facts(spec: FsmSpec, kb: KnowledgeBase) -> None:
    """Fail unless every fact a state releases exists in the knowledge base."""
    known = kb.fact_ids()
    for name, state in spec.states.items():
        for fact_id in state.facts:
            if fact_id in known:
                continue
            raise FsmError(
                f"state {name!r} releases {fact_id}, which is not a fact of "
                f"knowledge_base.md. Fix the ID in machine.yaml; facts are never "
                f"renumbered"
            )


def _check_packages(spec: FsmSpec, directory: Path) -> None:
    """Fail unless every state's instruction package is where it says it is."""
    for name, state in spec.states.items():
        path = directory / state.package
        if not path.is_file():
            raise FsmError(
                f"state {name!r} points at the package {state.package!r}, which "
                f"does not exist under {directory}. Write it or fix the path"
            )
        package = path.read_text(encoding="utf-8")
        _check_package_sections(name, package)
        _check_package_cites_no_fact_id(name, package)


def _check_package_sections(name: str, package: str) -> None:
    """Fail unless the package carries exactly the sections of a state package."""
    headings = [heading.strip() for heading in _PACKAGE_HEADING.findall(package)]
    for section in PACKAGE_SECTIONS:
        if section in headings:
            continue
        raise FsmError(
            f"the package of state {name!r} has no '{section}' section. Every "
            f"package answers the same questions, so the two agents differ in the "
            f"structure of the instruction and in nothing else"
        )
    extra = [heading for heading in headings if heading not in PACKAGE_SECTIONS]
    if extra:
        raise FsmError(
            f"the package of state {name!r} adds the sections {extra}. Persona, "
            f"tone and the general rules live in the block both agents share, so a "
            f"package that grows a section of its own is how the two agents stop "
            f"being comparable"
        )


def _check_package_cites_no_fact_id(name: str, package: str) -> None:
    """Fail unless the package's prose is free of knowledge-base identifiers."""
    cited = FACT_ID.findall(package)
    if not cited:
        return
    raise FsmError(
        f"the package of state {name!r} cites {sorted(set(cited))}. A package says "
        f"what to do, never which fact to say it from: the slice of the knowledge "
        f"base is appended to it at run time with the IDs on it, and a package that "
        f"repeats one gets it copied to the customer, which the shared block forbids"
    )
