"""Tests for scripts/cases.py (T-21)."""

from csv import DictReader, DictWriter
from pathlib import Path

from helpers import load_script

cases_script = load_script("scripts/cases.py")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_cases_script_writes_category_directions_and_shortlist_csv(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "tests.csv"
    metrics = tmp_path / "metrics.csv"
    out = tmp_path / "out"
    _write_csv(
        tests,
        [
            {
                "population": "semantic_primary",
                "stratum": "overall",
                "metric": "task_completed",
                "family": "primary",
                "n": 3,
                "wins": 1,
                "ties": 1,
                "losses": 1,
                "direction": "baseline",
                "mean_diff": -0.1,
                "rank_biserial": -0.2,
            },
            {
                "population": "semantic_primary",
                "stratum": "adversarial",
                "metric": "task_completed",
                "family": "exploratory",
                "n": 1,
                "wins": 1,
                "ties": 0,
                "losses": 0,
                "direction": "fsm",
                "mean_diff": 0.5,
                "rank_biserial": 1.0,
            },
            {
                "population": "frozen_gate",
                "stratum": "overall",
                "metric": "task_completed",
                "family": "secondary",
                "n": 99,
                "wins": 0,
                "ties": 0,
                "losses": 0,
                "direction": "fsm",
                "mean_diff": 1.0,
                "rank_biserial": 1.0,
            },
        ],
    )
    _write_csv(
        metrics,
        [
            {
                "scenario_id": "adversarial_08",
                "agent": "baseline",
                "repetition": 1,
                "task_completed": 0.0,
                "fact_f1": 0.2,
                "claim_support": 0.4,
            },
            {
                "scenario_id": "adversarial_08",
                "agent": "fsm",
                "repetition": 1,
                "task_completed": 1.0,
                "fact_f1": 0.8,
                "claim_support": 0.7,
            },
            {
                "scenario_id": "edge_05",
                "agent": "baseline",
                "repetition": 1,
                "task_completed": 1.0,
                "fact_f1": 0.8,
                "claim_support": 1.0,
            },
            {
                "scenario_id": "edge_05",
                "agent": "fsm",
                "repetition": 1,
                "task_completed": 0.0,
                "fact_f1": 0.2,
                "claim_support": 1.0,
            },
            {
                "scenario_id": "happy_path_07",
                "agent": "baseline",
                "repetition": 1,
                "task_completed": 1.0,
                "fact_f1": 0.5,
                "claim_support": 1.0,
            },
            {
                "scenario_id": "happy_path_07",
                "agent": "fsm",
                "repetition": 1,
                "task_completed": 1.0,
                "fact_f1": 0.5,
                "claim_support": 1.0,
            },
        ],
    )

    assert (
        cases_script.main(
            [
                "--tests",
                str(tests),
                "--metrics",
                str(metrics),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    directions = list(
        DictReader((out / "category_directions.csv").open(encoding="utf-8"))
    )
    by_key = {(row["metric"], row["stratum"]): row for row in directions}
    overall = by_key[("task_completed", "overall")]
    assert overall["n"] == "3"
    assert overall["wins"] == "1"
    assert overall["direction"] == "baseline"
    assert overall["family"] == "confirmatory"
    assert overall["inverted"] in {"False", "false"}
    adversarial = by_key[("task_completed", "adversarial")]
    assert adversarial["direction"] == "fsm"
    assert adversarial["inverted"] in {"True", "true"}
    assert adversarial["n"] == "1"
    assert all(row["n"] != "99" for row in directions)

    candidates = list(DictReader((out / "case_candidates.csv").open(encoding="utf-8")))
    fieldnames = candidates[0].keys()
    for name in (
        "scenario_id",
        "task_completed_diff",
        "fact_f1_diff",
        "claim_support_diff",
        "n_primary_available",
        "composite",
        "bucket",
        "rank",
    ):
        assert name in fieldnames
    by_id = {row["scenario_id"]: row for row in candidates}
    assert by_id["adversarial_08"]["bucket"] == "helped"
    assert by_id["adversarial_08"]["rank"] == "1"
    assert by_id["edge_05"]["bucket"] == "restricted"
    assert by_id["happy_path_07"]["bucket"] == "tie"
