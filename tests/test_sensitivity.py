"""T-25 sensitivity of the primary tests without the ambiguous-fact-ID dialogues."""

from csv import DictReader
from pathlib import Path

import pytest

from helpers import REPO_ROOT
from sim.analysis import (
    HUMAN_PRIMARY,
    N_RESAMPLES,
    OVERALL,
    PRIMARY_METRICS,
    SEED,
    SEMANTIC_PRIMARY,
    holm_adjust,
    load_metrics_csv,
)
from sim.sensitivity import sensitivity_rows

RESULTS = REPO_ROOT / "results"
UNCOMPARED_COLUMNS = frozenset({"instrument", "population", "family"})


def _row(scenario_id: str, agent: str, repetition: int, score: float) -> dict:
    return {
        "scenario_id": scenario_id,
        "agent": agent,
        "repetition": repetition,
        "task_completed": str(score > 0.5),
        "fact_f1": score,
        "claim_support": score / 2,
    }


def _frozen_overall_primary(path: Path, population: str) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["metric"]: row
            for row in DictReader(handle)
            if row["population"] == population
            and row["stratum"] == OVERALL
            and row["metric"] in PRIMARY_METRICS
        }


def _cells(row: dict) -> dict[str, str]:
    return {
        column: "" if value is None else str(value)
        for column, value in row.items()
        if column not in UNCOMPARED_COLUMNS
    }


def test_rows_are_exploratory_with_reference_holm_on_the_three_primaries() -> None:
    pytest.importorskip("scipy")
    rows = [
        row
        for index, (baseline, fsm) in enumerate(
            [(0.1, 0.9), (0.2, 0.7), (0.6, 0.3), (0.4, 0.8), (0.3, 0.35)], start=1
        )
        for row in (
            _row(f"happy_path_0{index}", "baseline", 1, baseline),
            _row(f"happy_path_0{index}", "fsm", 1, fsm),
            _row(f"happy_path_0{index}", "fsm", 2, 0.0),
        )
    ]
    ambiguous = [(f"happy_path_0{index}", "fsm", 2) for index in range(1, 6)]

    table = sensitivity_rows({"judge": rows}, ambiguous, n_resamples=20, seed=SEED)
    overall = {row["metric"]: row for row in table if row["stratum"] == OVERALL}

    assert {row["family"] for row in table} == {"exploratory"}
    assert set(overall) == set(PRIMARY_METRICS)
    assert {
        metric: row["wilcoxon_p_holm"] for metric, row in overall.items()
    } == holm_adjust({metric: row["wilcoxon_p"] for metric, row in overall.items()})
    assert all(
        row["wilcoxon_p_holm"] is None for row in table if row["stratum"] != OVERALL
    )
    assert overall["fact_f1"]["mean_diff"] == pytest.approx(0.29)


def test_empty_list_reproduces_frozen_overall_primary_rows() -> None:
    pytest.importorskip("scipy")
    frozen = {
        "judge": _frozen_overall_primary(
            RESULTS / "exp_final" / "tests.csv", SEMANTIC_PRIMARY
        ),
        "human": _frozen_overall_primary(
            RESULTS / "human_primary" / "tests.csv", HUMAN_PRIMARY
        ),
    }

    table = sensitivity_rows(
        {
            "judge": load_metrics_csv(RESULTS / "exp_final" / "metrics.csv"),
            "human": load_metrics_csv(RESULTS / "human_primary" / "metrics.csv"),
        },
        [],
        n_resamples=N_RESAMPLES,
        seed=SEED,
    )
    replayed = {
        (row["instrument"], row["metric"]): _cells(row)
        for row in table
        if row["stratum"] == OVERALL
    }

    assert replayed == {
        (instrument, metric): _cells(row)
        for instrument, rows in frozen.items()
        for metric, row in rows.items()
    }
