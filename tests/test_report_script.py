"""Tests for scripts/report.py (T-20)."""

from csv import DictWriter
from pathlib import Path

from helpers import load_script

report_script = load_script("scripts/report.py")


def _write_minimal_triple(results: Path) -> None:
    results.mkdir()
    with (results / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=(
                "scenario_id",
                "agent",
                "repetition",
                "task_completed",
                "fact_f1",
                "claim_support",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": "happy_path_01",
                "agent": "baseline",
                "repetition": 1,
                "task_completed": 1,
                "fact_f1": 0.5,
                "claim_support": 1,
            }
        )
        writer.writerow(
            {
                "scenario_id": "happy_path_01",
                "agent": "fsm",
                "repetition": 1,
                "task_completed": 1,
                "fact_f1": 0.5,
                "claim_support": 1,
            }
        )
    descriptive_fields = (
        "population",
        "stratum",
        "metric",
        "agent",
        "n_scenarios",
        "n_dialogues",
        "mean",
        "sd",
        "median",
        "ci_low",
        "ci_high",
        "mean_intra_scenario_sd",
        "claim_support_total_dialogues",
        "claim_support_na_dialogues",
        "claim_support_na_pct",
        "claim_support_aggregatable_dialogues",
        "claim_support_complete_pairs",
        "claim_support_pairs_excluded_either_na",
    )
    with (results / "descriptive.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = DictWriter(handle, fieldnames=descriptive_fields)
        writer.writeheader()
        for metric in ("task_completed", "fact_f1", "claim_support"):
            for agent in ("baseline", "fsm"):
                row = dict.fromkeys(descriptive_fields, "")
                row.update(
                    population="semantic_primary",
                    stratum="overall",
                    metric=metric,
                    agent=agent,
                    n_scenarios=1,
                    n_dialogues=1,
                    mean=0.5,
                    sd=0.0,
                    median=0.5,
                    ci_low=0.5,
                    ci_high=0.5,
                )
                if metric == "claim_support":
                    row.update(
                        claim_support_total_dialogues=1,
                        claim_support_na_dialogues=0,
                        claim_support_na_pct=0.0,
                        claim_support_aggregatable_dialogues=1,
                        claim_support_complete_pairs=1,
                        claim_support_pairs_excluded_either_na=0,
                    )
                writer.writerow(row)
    tests_fields = (
        "population",
        "stratum",
        "metric",
        "family",
        "n",
        "n_nonzero",
        "n_dropped_unpaired",
        "wilcoxon_p",
        "wilcoxon_p_holm",
        "permutation_p",
        "rank_biserial",
        "mean_diff",
        "ci_low",
        "ci_high",
        "wins",
        "ties",
        "losses",
        "direction",
        "role",
        "baseline_mean",
        "fsm_mean",
        "claim_support_complete_pairs",
        "claim_support_pairs_excluded_either_na",
    )
    with (results / "tests.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=tests_fields)
        writer.writeheader()
        for metric in ("task_completed", "fact_f1", "claim_support"):
            row = dict.fromkeys(tests_fields, "")
            row.update(
                population="semantic_primary",
                stratum="overall",
                metric=metric,
                family="primary",
                n=1,
                n_nonzero=0,
                n_dropped_unpaired=0,
                wilcoxon_p=1.0,
                wilcoxon_p_holm=1.0,
                permutation_p=1.0,
                rank_biserial=0.0,
                mean_diff=0.0,
                ci_low=0.0,
                ci_high=0.0,
                wins=0,
                ties=1,
                losses=0,
                direction="tie",
            )
            if metric == "claim_support":
                row.update(
                    claim_support_complete_pairs=1,
                    claim_support_pairs_excluded_either_na=0,
                )
            writer.writerow(row)


def test_report_script_consumes_three_frozen_t19_files(tmp_path: Path) -> None:
    results = tmp_path / "exp"
    _write_minimal_triple(results)

    assert report_script.main(["--results-dir", str(results)]) == 0

    assert (results / "tables" / "resultados_tabela-01_descritiva-geral.csv").is_file()
    assert (results / "tables" / "resultados_tabela-03_testes-pareados.csv").is_file()


def test_report_script_fails_when_a_t19_csv_is_missing(tmp_path: Path) -> None:
    results = tmp_path / "empty"
    results.mkdir()
    (results / "metrics.csv").write_text("scenario_id\n", encoding="utf-8")

    assert report_script.main(["--results-dir", str(results)]) == 1
