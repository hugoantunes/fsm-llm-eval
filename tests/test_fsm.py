"""Tests for the FSM loader and the real machine of data/fsm/ (T-02)."""

from collections.abc import Callable
from pathlib import Path

import pytest

from helpers import DOCS_DIR, GENERAL_FACT, TRACKING_FACT, make_kb
from sim.fsm import FsmError, FsmSpec, load_fsm
from sim.kb import Fact

#: The synthetic machine below is loaded against this, not against the real KB:
#: two facts are enough to tell a released one from a withheld one.
KB = make_kb(GENERAL_FACT, TRACKING_FACT)

MACHINE = """\
version: 1
initial: greeting
accepting_states: [closing, out_of_scope]
states:
  greeting:
    package: states/greeting.md
    facts: [F01]
  closing:
    package: states/closing.md
    facts: [F01]
  out_of_scope:
    package: states/out_of_scope.md
    facts: [F01]
transitions:
  - event: user_confirmed
    from: greeting
    to: closing
    guard: required_data_collected
  - event: out_of_scope_request
    from: "*"
    to: out_of_scope
"""

#: The same machine, with ``closing`` also releasing the classified intent's section.
INTENT_MACHINE = MACHINE.replace(
    "  closing:\n    package: states/closing.md\n    facts: [F01]",
    "  closing:\n    package: states/closing.md\n    facts: [F01]\n"
    "    facts_from_intent: true",
)

PACKAGE = """\
# Greeting

## Goal

Open the conversation and find out what the customer needs.

## Data to collect or confirm

Nothing yet.

## The answer must

Greet the customer and ask what the request is about.

## Never in this state

Never state anything about a specific order.

## Tone in this state

Two sentences at most.
"""

STATES = ("greeting", "closing", "out_of_scope")


def _ids(facts: list[Fact]) -> list[str]:
    return [fact.id for fact in facts]


def write_fsm(
    directory: Path,
    machine: str = MACHINE,
    packages: dict[str, str] | None = None,
) -> Path:
    """Write a synthetic machine and its state packages into ``directory``."""
    (directory / "states").mkdir(parents=True, exist_ok=True)
    (directory / "machine.yaml").write_text(machine, encoding="utf-8")
    for state, text in (dict.fromkeys(STATES, PACKAGE) | (packages or {})).items():
        (directory / "states" / f"{state}.md").write_text(text, encoding="utf-8")
    return directory


def test_load_fsm_reads_states_events_and_transitions(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")

    fsm = load_fsm(directory, kb=KB)

    assert fsm.initial == "greeting"
    assert sorted(fsm.states) == ["closing", "greeting", "out_of_scope"]
    assert (
        "greeting",
        "user_confirmed",
        "closing",
        "required_data_collected",
    ) in [(t.source, t.event, t.dest, t.guard) for t in fsm.transitions]


def test_transition_to_an_unknown_state_raises(tmp_path: Path) -> None:
    machine = MACHINE.replace("to: closing", "to: farewell")
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match="farewell"):
        load_fsm(directory, kb=KB)


def test_misspelled_optional_key_in_machine_yaml_raises(tmp_path: Path) -> None:
    machine = MACHINE.replace(
        "    guard: required_data_collected", "    guards: required_data_collected"
    )
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match="guards"):
        load_fsm(directory, kb=KB)


def test_unknown_guard_name_raises(tmp_path: Path) -> None:
    machine = MACHINE.replace(
        "guard: required_data_collected", "guard: customer_sounds_friendly"
    )
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match="customer_sounds_friendly"):
        load_fsm(directory, kb=KB)


def test_wildcard_source_expands_to_every_state(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")

    fsm = load_fsm(directory, kb=KB)

    sources = {t.source for t in fsm.transitions if t.event == "out_of_scope_request"}
    assert sources == {"greeting", "closing"}


def test_two_transitions_with_the_same_state_and_event_raise(tmp_path: Path) -> None:
    machine = MACHINE + (
        "  - event: out_of_scope_request\n    from: greeting\n    to: closing\n"
    )
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match="out_of_scope_request"):
        load_fsm(directory, kb=KB)


def test_state_unreachable_from_the_initial_state_raises(tmp_path: Path) -> None:
    machine = MACHINE.replace(
        "transitions:",
        "  solution:\n    package: states/solution.md\n    facts: [F01]\ntransitions:",
    )
    directory = write_fsm(
        tmp_path / "fsm", machine=machine, packages={"solution": PACKAGE}
    )

    with pytest.raises(FsmError, match="solution"):
        load_fsm(directory, kb=KB)


@pytest.mark.parametrize(
    ("declaration", "replacement", "unknown"),
    [
        ("initial: greeting", "initial: hello", "hello"),
        (
            "accepting_states: [closing, out_of_scope]",
            "accepting_states: [closing, escalated]",
            "escalated",
        ),
    ],
)
def test_initial_and_accepting_states_must_be_declared(
    tmp_path: Path, declaration: str, replacement: str, unknown: str
) -> None:
    machine = MACHINE.replace(declaration, replacement)
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match=unknown):
        load_fsm(directory, kb=KB)


def test_events_for_state_lists_only_that_states_events(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")

    fsm = load_fsm(directory, kb=KB)

    assert fsm.events_for("greeting") == ["out_of_scope_request", "user_confirmed"]
    assert fsm.events_for("out_of_scope") == []


@pytest.mark.parametrize(
    "ask",
    [
        lambda fsm: fsm.events_for("greetings"),
        lambda fsm: fsm.released_facts("greetings", None, KB),
    ],
    ids=["events_for", "released_facts"],
)
def test_asking_about_an_unknown_state_raises(
    tmp_path: Path, ask: Callable[[FsmSpec], object]
) -> None:
    fsm = load_fsm(write_fsm(tmp_path / "fsm"), kb=KB)

    with pytest.raises(FsmError, match="greetings"):
        ask(fsm)


def test_state_releasing_an_unknown_fact_id_raises(tmp_path: Path) -> None:
    machine = MACHINE.replace("facts: [F01]", "facts: [F01, F99]", 1)
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match="F99"):
        load_fsm(directory, kb=KB)


def test_released_facts_of_a_state_include_the_classified_intent_section(
    tmp_path: Path,
) -> None:
    directory = write_fsm(tmp_path / "fsm", machine=INTENT_MACHINE)

    fsm = load_fsm(directory, kb=KB)

    assert _ids(fsm.released_facts("greeting", "order_tracking", KB)) == ["F01"]
    assert _ids(fsm.released_facts("closing", "order_tracking", KB)) == ["F01", "F02"]
    assert _ids(fsm.released_facts("greeting", None, KB)) == ["F01"]


def test_released_facts_without_the_intent_a_state_needs_raises(
    tmp_path: Path,
) -> None:
    fsm = load_fsm(write_fsm(tmp_path / "fsm", machine=INTENT_MACHINE), kb=KB)

    with pytest.raises(FsmError, match="closing"):
        fsm.released_facts("closing", None, KB)


def test_released_facts_for_an_intent_the_kb_does_not_have_raises(
    tmp_path: Path,
) -> None:
    directory = write_fsm(tmp_path / "fsm", machine=INTENT_MACHINE)
    fsm = load_fsm(directory, kb=KB)

    with pytest.raises(FsmError, match="gift_wrapping"):
        fsm.released_facts("closing", "gift_wrapping", KB)


def test_missing_package_file_raises(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")
    (directory / "states" / "closing.md").unlink()

    with pytest.raises(FsmError, match=r"closing\.md"):
        load_fsm(directory, kb=KB)


def test_package_mentioning_a_fact_not_released_for_that_state_raises(
    tmp_path: Path,
) -> None:
    package = PACKAGE.replace(
        "Never state anything about a specific order.",
        "Never promise a delivery date (F02).",
    )
    directory = write_fsm(tmp_path / "fsm", packages={"greeting": package})

    with pytest.raises(FsmError, match="F02"):
        load_fsm(directory, kb=KB)


def test_package_with_a_persona_section_of_its_own_raises(tmp_path: Path) -> None:
    package = PACKAGE + "\n## Persona\n\nYou are the friendly voice of the store.\n"
    directory = write_fsm(tmp_path / "fsm", packages={"greeting": package})

    with pytest.raises(FsmError, match="Persona"):
        load_fsm(directory, kb=KB)


def test_package_missing_a_required_heading_raises(tmp_path: Path) -> None:
    package = PACKAGE.replace("## Never in this state", "## Watch out")
    directory = write_fsm(tmp_path / "fsm", packages={"greeting": package})

    with pytest.raises(FsmError, match="Never in this state"):
        load_fsm(directory, kb=KB)


def test_to_mermaid_draws_one_arrow_per_declared_transition(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")

    diagram = load_fsm(directory, kb=KB).to_mermaid()

    assert diagram.splitlines()[0] == "stateDiagram-v2"
    assert "    greeting --> closing : user_confirmed [required_data_collected]" in (
        diagram
    )
    assert "note right of out_of_scope" in diagram
    assert diagram.count("-->") == 4


# --- the real machine of data/fsm/ ------------------------------------------

EXPECTED_STATES = [
    "closing",
    "confirmation",
    "data_collection",
    "greeting",
    "identification",
    "intent_classification",
    "out_of_scope",
    "solution",
]


#: The events of a happy-path dialogue, in order: the flow the FSM agent is meant to
#: walk when nothing goes wrong. The universal exits are not part of it.
CANONICAL_EVENTS = (
    "request_received",
    "order_identified",
    "intent_classified",
    "data_provided",
    "solution_accepted",
    "user_confirmed",
)


def test_real_machine_has_the_eight_expected_states_and_loads_clean(
    real_fsm: FsmSpec,
) -> None:
    assert sorted(real_fsm.states) == EXPECTED_STATES
    assert real_fsm.initial == "greeting"
    assert sorted(real_fsm.accepting_states) == ["closing", "out_of_scope"]


def test_real_machine_walks_the_whole_flow_without_the_universal_exits(
    real_fsm: FsmSpec,
) -> None:
    flow = {(t.source, t.event): t.dest for t in real_fsm.transitions if not t.from_any}

    state = real_fsm.initial
    walked = [state]
    for event in CANONICAL_EVENTS:
        state = flow[(state, event)]
        walked.append(state)

    assert walked == [
        "greeting",
        "identification",
        "intent_classification",
        "data_collection",
        "solution",
        "confirmation",
        "closing",
    ]


def test_real_machine_lets_every_state_reach_out_of_scope(real_fsm: FsmSpec) -> None:
    sources = {t.source for t in real_fsm.transitions if t.dest == "out_of_scope"}
    assert sources == set(real_fsm.states) - {"out_of_scope"}


def test_docs_fsm_diagram_matches_machine_yaml(real_fsm: FsmSpec) -> None:
    doc = (DOCS_DIR / "fsm.md").read_text(encoding="utf-8")

    block = f"```mermaid\n{real_fsm.to_mermaid()}\n```"

    assert block in doc, "the diagram is stale: run `just fsm-diagram` and paste it"
