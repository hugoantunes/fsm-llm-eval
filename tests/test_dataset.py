"""Golden-dataset invariants (T-06)."""

from collections import Counter

from sim.kb import KnowledgeBase
from sim.schemas import FULL_SET_N, FULL_SET_QUOTA, Scenario


def test_v1_loads_through_load_scenarios(
    v1_scenarios: dict[str, Scenario],
) -> None:
    assert len(v1_scenarios) == FULL_SET_N


def test_v1_has_five_scenarios_per_intent_and_category(
    v1_scenarios: dict[str, Scenario],
) -> None:
    counts = Counter((row.intent, row.category) for row in v1_scenarios.values())

    assert len(counts) == 12
    assert set(counts.values()) == {FULL_SET_QUOTA}


def test_v1_has_one_canary_per_intent(
    v1_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    planted = [row for row in v1_scenarios.values() if row.canary]

    assert len(planted) == 4
    assert {row.intent for row in planted} == set(real_kb.intents())
    assert len({row.canary for row in planted}) == 4


def test_v1_maps_each_needle_to_exactly_one_scenario(
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
