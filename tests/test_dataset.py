"""Frozen golden-dataset invariants (T-06)."""

from collections import Counter

from helpers import DOCS_DIR, FROZEN_V1_HASH, V1_DIR
from sim.fsm import FsmSpec
from sim.kb import KnowledgeBase
from sim.runner import hash_dataset
from sim.schemas import FULL_SET_N, FULL_SET_QUOTA, Scenario


def test_frozen_v1_loads_and_matches_the_recorded_hash(
    v1_scenarios: dict[str, Scenario],
) -> None:
    assert hash_dataset(V1_DIR) == FROZEN_V1_HASH
    assert FROZEN_V1_HASH in (DOCS_DIR / "taxonomy.md").read_text(encoding="utf-8")
    assert len(v1_scenarios) == FULL_SET_N


def test_frozen_v1_has_five_scenarios_per_intent_and_category(
    v1_scenarios: dict[str, Scenario],
) -> None:
    counts = Counter((row.intent, row.category) for row in v1_scenarios.values())

    assert len(counts) == 12
    assert set(counts.values()) == {FULL_SET_QUOTA}


def test_frozen_v1_has_one_canary_per_intent(
    v1_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    planted = [row for row in v1_scenarios.values() if row.canary]

    assert len(planted) == 4
    assert {row.intent for row in planted} == set(real_kb.intents())
    assert len({row.canary for row in planted}) == 4


def test_frozen_v1_maps_each_needle_to_exactly_one_scenario(
    v1_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    needle_facts = {needle.fact_id for needle in real_kb.needles}
    flagged = [row for row in v1_scenarios.values() if row.is_needle]
    covered = []
    for row in flagged:
        found = needle_facts & set(row.required_facts)
        assert len(found) == 1
        covered.append(found.pop())

    assert set(covered) == needle_facts
    assert len(flagged) == len(needle_facts)
    for row in v1_scenarios.values():
        if row.is_needle:
            continue
        assert needle_facts.isdisjoint(row.required_facts)


WAIVED_OUT_OF_SCOPE_FINAL_STATE = {
    "adversarial_04",
    "adversarial_08",
    "adversarial_12",
}

NO_SOLUTION_BEFORE_FAREWELL = {
    "edge_19",
    "happy_path_11",
    "happy_path_19",
}


def test_frozen_v1_has_only_the_documented_out_of_scope_final_state_waivers(
    v1_scenarios: dict[str, Scenario], real_fsm: FsmSpec
) -> None:
    flagged = {
        scenario.id
        for scenario in v1_scenarios.values()
        if scenario.expected_final_state == "out_of_scope"
    }

    assert set(real_fsm.accepting_states) == {"closing", "out_of_scope"}
    assert flagged == WAIVED_OUT_OF_SCOPE_FINAL_STATE


def test_frozen_v1_pins_the_count_reaching_solution_before_the_farewell_beat(
    v1_scenarios: dict[str, Scenario],
) -> None:
    ids = set(v1_scenarios)
    assert ids >= NO_SOLUTION_BEFORE_FAREWELL

    reachable = ids - NO_SOLUTION_BEFORE_FAREWELL

    assert len(reachable) == FULL_SET_N - len(NO_SOLUTION_BEFORE_FAREWELL)
