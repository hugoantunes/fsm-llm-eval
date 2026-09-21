"""Tests for the FSM loader and the real machine of data/fsm/ (T-02)."""

import re
from pathlib import Path

import pytest

from helpers import DOCS_DIR, FSM_DIR
from sim.fsm import FsmError, FsmSpec, load_fsm
from sim.kb import FACT_ID

MACHINE = """\
version: 1
initial: greeting
accepting_states: [closing, out_of_scope]
states:
  greeting:
    package: states/greeting.md
  closing:
    package: states/closing.md
  out_of_scope:
    package: states/out_of_scope.md
transitions:
  - event: user_confirmed
    from: greeting
    to: closing
    guard: required_data_collected
  - event: out_of_scope_request
    from: "*"
    to: out_of_scope
"""

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

    fsm = load_fsm(directory)

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
        load_fsm(directory)


@pytest.mark.parametrize(
    ("declaration", "replacement", "unknown"),
    [
        (
            "    guard: required_data_collected",
            "    guards: required_data_collected",
            "guards",
        ),
        (
            "    package: states/greeting.md",
            "    package: states/greeting.md\n    facts: [F01]",
            "facts",
        ),
    ],
)
def test_misspelled_optional_key_in_machine_yaml_raises(
    tmp_path: Path, declaration: str, replacement: str, unknown: str
) -> None:
    machine = MACHINE.replace(declaration, replacement)
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match=unknown):
        load_fsm(directory)


def test_unknown_guard_name_raises(tmp_path: Path) -> None:
    machine = MACHINE.replace(
        "guard: required_data_collected", "guard: customer_sounds_friendly"
    )
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match="customer_sounds_friendly"):
        load_fsm(directory)


def test_wildcard_source_expands_to_every_state(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")

    fsm = load_fsm(directory)

    sources = {t.source for t in fsm.transitions if t.event == "out_of_scope_request"}
    assert sources == {"greeting", "closing"}


def test_two_transitions_with_the_same_state_and_event_raise(tmp_path: Path) -> None:
    machine = MACHINE + (
        "  - event: out_of_scope_request\n    from: greeting\n    to: closing\n"
    )
    directory = write_fsm(tmp_path / "fsm", machine=machine)

    with pytest.raises(FsmError, match="out_of_scope_request"):
        load_fsm(directory)


def test_state_unreachable_from_the_initial_state_raises(tmp_path: Path) -> None:
    machine = MACHINE.replace(
        "transitions:",
        "  solution:\n    package: states/solution.md\ntransitions:",
    )
    directory = write_fsm(
        tmp_path / "fsm", machine=machine, packages={"solution": PACKAGE}
    )

    with pytest.raises(FsmError, match="solution"):
        load_fsm(directory)


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
        load_fsm(directory)


def test_events_for_state_lists_only_that_states_events(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")

    fsm = load_fsm(directory)

    assert fsm.events_for("greeting") == ["out_of_scope_request", "user_confirmed"]
    assert fsm.events_for("out_of_scope") == []


def test_asking_about_an_unknown_state_raises(tmp_path: Path) -> None:
    fsm = load_fsm(write_fsm(tmp_path / "fsm"))

    with pytest.raises(FsmError, match="greetings"):
        fsm.events_for("greetings")


def test_missing_package_file_raises(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")
    (directory / "states" / "closing.md").unlink()

    with pytest.raises(FsmError, match=r"closing\.md"):
        load_fsm(directory)


@pytest.mark.parametrize("fact_id", ["F01", "F02"])
def test_package_citing_a_fact_id_raises(tmp_path: Path, fact_id: str) -> None:
    package = PACKAGE.replace(
        "Never state anything about a specific order.",
        f"Never promise a delivery date ({fact_id}).",
    )
    directory = write_fsm(tmp_path / "fsm", packages={"greeting": package})

    with pytest.raises(FsmError, match=fact_id):
        load_fsm(directory)


def test_no_real_state_package_cites_a_fact_id(real_fsm: FsmSpec) -> None:
    cited = {
        name: FACT_ID.findall((FSM_DIR / state.package).read_text(encoding="utf-8"))
        for name, state in real_fsm.states.items()
    }

    assert {name: found for name, found in cited.items() if found} == {}


def test_package_with_a_persona_section_of_its_own_raises(tmp_path: Path) -> None:
    package = PACKAGE + "\n## Persona\n\nYou are the friendly voice of the store.\n"
    directory = write_fsm(tmp_path / "fsm", packages={"greeting": package})

    with pytest.raises(FsmError, match="Persona"):
        load_fsm(directory)


def test_package_missing_a_required_heading_raises(tmp_path: Path) -> None:
    package = PACKAGE.replace("## Never in this state", "## Watch out")
    directory = write_fsm(tmp_path / "fsm", packages={"greeting": package})

    with pytest.raises(FsmError, match="Never in this state"):
        load_fsm(directory)


def test_to_mermaid_draws_one_arrow_per_declared_transition(tmp_path: Path) -> None:
    directory = write_fsm(tmp_path / "fsm")

    diagram = load_fsm(directory).to_mermaid()

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


def test_intent_classification_package_asks_for_the_identity_read_back(
    real_fsm: FsmSpec,
) -> None:
    package = (FSM_DIR / real_fsm.states["intent_classification"].package).read_text(
        encoding="utf-8"
    )
    data_section, rest = package.split("## The answer must", maxsplit=1)
    answer_section = rest.split("## Never in this state", maxsplit=1)[0]

    assert "order number" in data_section.lower()
    assert "e-mail" in data_section.lower()
    assert "read back" in answer_section.lower()
    assert "order number" in answer_section.lower()
    assert "e-mail" in answer_section.lower()
    assert "before any request-specific question or answer" in answer_section.lower()
    assert "do not ask the customer to provide, repeat or" in answer_section.lower()
    assert "confirm either value again" in answer_section.lower()
    assert "unless the customer explicitly corrects one" in answer_section.lower()


def test_solution_package_does_not_reask_known_identity(real_fsm: FsmSpec) -> None:
    package = (FSM_DIR / real_fsm.states["solution"].package).read_text(
        encoding="utf-8"
    )
    data_section, rest = package.split("## The answer must", maxsplit=1)
    answer_section = rest.split("## Never in this state", maxsplit=1)[0]

    assert "order number" in data_section.lower()
    assert "e-mail" in data_section.lower()
    assert "already confirmed" in data_section.lower()
    assert (
        "do not ask the customer to provide, repeat or confirm either value again"
        in (answer_section.lower())
    )
    assert "unless the customer explicitly corrects one" in answer_section.lower()


def test_every_package_phrases_the_store_scope_the_same_way(real_fsm: FsmSpec) -> None:
    canonical = (
        "the store is an online shop for clothing, footwear and home goods, with "
        "delivery only inside the country."
    )
    packages = {
        name: re.sub(
            r"\s+",
            " ",
            (FSM_DIR / state.package).read_text(encoding="utf-8").lower(),
        )
        for name, state in real_fsm.states.items()
    }
    carrying_scope_rule = {
        name: text for name, text in packages.items() if canonical in text
    }

    assert set(carrying_scope_rule) == {
        "greeting",
        "intent_classification",
        "out_of_scope",
    }
    assert all(canonical in text for text in carrying_scope_rule.values())


def test_out_of_scope_package_puts_scope_before_escalation_and_allows_three_lines(
    real_fsm: FsmSpec,
) -> None:
    package = (FSM_DIR / real_fsm.states["out_of_scope"].package).read_text(
        encoding="utf-8"
    )
    answer_section = package.split("## The answer must", maxsplit=1)[1].split(
        "## Never in this state", maxsplit=1
    )[0]
    tone_section = package.split("## Tone in this state", maxsplit=1)[1]

    normalized = re.sub(r"\s+", " ", answer_section.lower())
    scope = normalized.index("online shop for clothing, footwear and home goods")
    escalation = normalized.index("escalate to a human specialist")
    assert scope < escalation
    assert "three short sentences" in tone_section.lower()
    assert "limit, scope, escalation" in tone_section.lower()


def test_no_package_names_a_scenario_specific_domain(real_fsm: FsmSpec) -> None:
    package_text = "\n".join(
        (FSM_DIR / state.package).read_text(encoding="utf-8").lower()
        for state in real_fsm.states.values()
    )

    assert "restaurant" not in package_text
    assert "membership" not in package_text


def test_real_machine_lets_every_state_reach_out_of_scope(real_fsm: FsmSpec) -> None:
    sources = {t.source for t in real_fsm.transitions if t.dest == "out_of_scope"}
    assert sources == set(real_fsm.states) - {"out_of_scope"}


def test_docs_fsm_diagram_matches_machine_yaml(real_fsm: FsmSpec) -> None:
    doc = (DOCS_DIR / "fsm.md").read_text(encoding="utf-8")

    block = f"```mermaid\n{real_fsm.to_mermaid()}\n```"

    assert block in doc, "the diagram is stale: run `just fsm-diagram` and paste it"
