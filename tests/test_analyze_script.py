"""Tests for scripts/analyze.py (T-19)."""

from csv import DictWriter
from pathlib import Path

import pytest

from helpers import load_script
from sim.metrics import METRICS

pytest.importorskip("pandas")
pytest.importorskip("scipy")

analyze_script = load_script("scripts/analyze.py")

_NA_METRICS = (
    "needle_recovered",
    "injection_succeeded",
    "policy_violation",
    "stage_label_accuracy",
)


def _write_metrics(path: Path) -> None:
    fieldnames = ("scenario_id", "agent", "repetition", *METRICS)
    rows = [
        _full_row("happy_path_01", "baseline", 1, task_completed=True, fact_f1=0.4),
        _full_row("happy_path_01", "fsm", 1, task_completed=True, fact_f1=0.8),
        _full_row("edge_01", "baseline", 1, task_completed=False, fact_f1=0.2),
        _full_row("edge_01", "fsm", 1, task_completed=True, fact_f1=0.6),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _full_row(
    scenario_id: str, agent: str, repetition: int, **overrides: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario_id": scenario_id,
        "agent": agent,
        "repetition": repetition,
        **dict.fromkeys(METRICS, 0.0),
    }
    for name in _NA_METRICS:
        row[name] = ""
    row["claim_support"] = 1.0
    row["n_checkable_claims"] = 2
    row.update(overrides)
    return row


def test_analyze_writes_descriptive_tests_and_environment(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    frozen = tmp_path / "frozen.csv"
    out = tmp_path / "out"
    _write_metrics(metrics)
    _write_metrics(frozen)

    assert (
        analyze_script.main(
            [
                "--metrics",
                str(metrics),
                "--frozen",
                str(frozen),
                "--out",
                str(out),
                "--n-resamples",
                "20",
                "--seed",
                "0",
            ]
        )
        == 0
    )

    assert (out / "descriptive.csv").is_file()
    assert (out / "tests.csv").is_file()
    environment = (out / "environment.txt").read_text(encoding="utf-8")
    assert "python=" in environment
    assert "scipy=" in environment
    tests = (out / "tests.csv").read_text(encoding="utf-8")
    descriptive = (out / "descriptive.csv").read_text(encoding="utf-8")
    assert "n_nonzero" in tests
    assert "semantic_primary" in tests
    assert "frozen_gate" in tests
    assert "drop5_instrument" in tests
    assert "claim_support_na_pct" in descriptive
    assert "claim_support_complete_pairs" in descriptive
    assert "zero_checkable_claim_occurrence" in tests
    assert "diagnostic_missingness" in tests
