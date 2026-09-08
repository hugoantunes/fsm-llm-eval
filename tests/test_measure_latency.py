"""Tests for the pure parts of scripts/measure_latency.py: statistics and budget."""

import pytest

from helpers import load_script

latency = load_script("scripts/measure_latency.py")
Call = latency.Call


def test_percentile_uses_nearest_rank() -> None:
    values = [float(v) for v in range(1, 11)]

    assert latency.percentile(values, 95) == 10.0
    assert latency.percentile(values, 50) == 5.0


def test_percentile_of_single_value_is_that_value() -> None:
    assert latency.percentile([4.2], 95) == 4.2


def test_percentile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        latency.percentile([], 95)


def test_summarize_groups_by_role_in_order_of_appearance() -> None:
    calls = [
        Call("agent", 4.0, 2000, 100),
        Call("simulator", 1.0, 300, 50),
        Call("agent", 6.0, 2200, 140),
    ]

    stats = latency.summarize(calls)

    assert [s.role for s in stats] == ["agent", "simulator"]
    agent = stats[0]
    assert agent.n == 2
    assert agent.mean_s == 5.0
    assert agent.p95_s == 6.0
    assert agent.mean_prompt_tokens == 2100
    assert agent.output_tokens_per_s == 24.0
    assert stats[1].n == 1


def test_dialogues_counts_scenarios_times_agents_times_reps() -> None:
    assert latency.dialogues(45, 3) == 270
    assert latency.dialogues(60, 5) == 600


def test_run_hours_divides_by_parallelism() -> None:
    means = {"agent": 6.0, "simulator": 2.0, "classifier": 1.0}

    hours = latency.run_hours(means, n_dialogues=270, turns=8, parallel=2)

    assert hours == pytest.approx(270 * 8 * 9 / 2 / 3600)


def test_judge_hours_counts_two_calls_per_dialogue() -> None:
    hours = latency.judge_hours(45.0, n_dialogues=270)

    assert hours == pytest.approx(270 * 2 * 45 / 3600)


def test_filler_is_sized_roughly_by_tokens() -> None:
    text = latency.filler(1200)

    words = len(text.split())
    assert 1000 <= words <= 1100
    assert text.startswith("Fact F01:")
