"""Tests for scripts/run_stats.py: the numbers docs/pilot.md quotes (T-15)."""

from pathlib import Path

from helpers import load_script

run_stats = load_script("scripts/run_stats.py")


def test_prompt_tokens_are_averaged_per_caller(stats_run_dir: Path) -> None:
    stats = run_stats.summarize(stats_run_dir)

    assert stats.callers["baseline"].mean_prompt_tokens == 150
    assert stats.callers["baseline"].mean_output_tokens == 20
    assert stats.callers["fsm"].mean_prompt_tokens == 80
    assert stats.callers["classifier"].mean_prompt_tokens == 40
    assert stats.callers["simulated_user"].mean_prompt_tokens == 60


def test_a_cached_call_is_left_out_of_the_latency_average(stats_run_dir: Path) -> None:
    stats = run_stats.summarize(stats_run_dir)

    assert stats.callers["fsm"].mean_latency_s == 4.0
    assert stats.callers["baseline"].mean_latency_s == 2.0


def test_seconds_per_dialogue_separates_run_from_eval(stats_run_dir: Path) -> None:
    stats = run_stats.summarize(stats_run_dir)

    assert stats.run_s_per_dialogue == 10.0
    assert stats.eval_s_per_dialogue == 35.0
