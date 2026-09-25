"""Scenario-level paired statistics (T-19)."""

from statistics import stdev

import pytest

from sim.analysis import (
    INSTRUMENT_DROP_SCENARIOS,
    aggregate_to_scenarios,
    bootstrap_mean_ci,
    category_of,
    descriptive_table,
    family_of,
    holm_adjust,
    mean_intra_scenario_sd,
    n_nonzero,
    pair_scenarios,
    paired_test_table,
    permutation_p,
    rank_biserial,
    rows_for_population,
    wilcoxon_p,
    wins_ties_losses,
)


def _row(
    scenario_id: str, agent: str, repetition: int, **metrics: object
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "agent": agent,
        "repetition": repetition,
        **metrics,
    }


def test_category_is_read_from_scenario_id() -> None:
    assert category_of("happy_path_07") == "happy_path"
    assert category_of("edge_01") == "edge"
    assert category_of("adversarial_19") == "adversarial"


def test_repetitions_aggregate_to_one_score_per_scenario_agent() -> None:
    rows = [
        {
            "scenario_id": "happy_path_01",
            "agent": "baseline",
            "repetition": 1,
            "fact_f1": 0.4,
        },
        {
            "scenario_id": "happy_path_01",
            "agent": "baseline",
            "repetition": 2,
            "fact_f1": 0.6,
        },
        {
            "scenario_id": "happy_path_01",
            "agent": "fsm",
            "repetition": 1,
            "fact_f1": 1.0,
        },
    ]

    scores = aggregate_to_scenarios(rows, metric="fact_f1")

    by_agent = {score.agent: score for score in scores}
    assert set(by_agent) == {"baseline", "fsm"}
    assert by_agent["baseline"].value == 0.5
    assert by_agent["baseline"].n_reps == 2
    assert by_agent["fsm"].value == 1.0
    assert by_agent["fsm"].n_reps == 1
    assert all(score.scenario_id == "happy_path_01" for score in scores)


def test_binary_aggregation_is_a_proportion() -> None:
    rows = [
        _row("edge_01", "baseline", 1, task_completed="True"),
        _row("edge_01", "baseline", 2, task_completed="False"),
        _row("edge_01", "baseline", 3, task_completed="True"),
        _row("edge_01", "fsm", 1, task_completed="True"),
        _row("edge_01", "fsm", 2, task_completed="True"),
        _row("edge_01", "fsm", 3, task_completed="True"),
    ]

    scores = aggregate_to_scenarios(rows, metric="task_completed")

    by_agent = {score.agent: score.value for score in scores}
    assert by_agent["baseline"] == 2 / 3
    assert by_agent["fsm"] == 1.0


def test_na_cells_are_omitted_from_the_scenario_mean_not_scored_zero() -> None:
    rows = [
        _row("happy_path_03", "baseline", 1, claim_support=1.0),
        _row("happy_path_03", "baseline", 2, claim_support=""),
        _row("happy_path_03", "fsm", 1, claim_support=0.5),
        _row("happy_path_03", "fsm", 2, claim_support=None),
    ]

    scores = aggregate_to_scenarios(rows, metric="claim_support")

    by_agent = {score.agent: score for score in scores}
    assert by_agent["baseline"].value == 1.0
    assert by_agent["baseline"].n_reps == 1
    assert by_agent["fsm"].value == 0.5
    assert by_agent["fsm"].n_reps == 1


def test_a_scenario_missing_one_agent_is_dropped_from_the_paired_test() -> None:
    rows = [
        _row("adversarial_12", "fsm", 1, fact_f1=0.8),
        _row("happy_path_01", "baseline", 1, fact_f1=0.4),
        _row("happy_path_01", "fsm", 1, fact_f1=0.6),
    ]

    pairs = pair_scenarios(aggregate_to_scenarios(rows, metric="fact_f1"))

    assert [pair.scenario_id for pair in pairs] == ["happy_path_01"]
    assert len(pairs) == 1


def test_incomplete_k_still_pairs_on_the_available_reps() -> None:
    rows = [
        _row("edge_01", "baseline", 1, fact_f1=0.2),
        _row("edge_01", "baseline", 2, fact_f1=0.4),
        _row("edge_01", "fsm", 1, fact_f1=1.0),
        _row("edge_01", "fsm", 2, fact_f1=0.8),
        _row("edge_01", "fsm", 3, fact_f1=0.9),
    ]

    pairs = pair_scenarios(aggregate_to_scenarios(rows, metric="fact_f1"))

    assert len(pairs) == 1
    assert pairs[0].baseline == pytest.approx(0.3)
    assert pairs[0].fsm == pytest.approx(0.9)


def test_near_zero_float_artifact_is_canonicalized_to_tie() -> None:
    rows = [
        _row("happy_path_01", "baseline", 1, fact_f1=0.3),
        _row("happy_path_01", "fsm", 1, fact_f1=0.1 + 0.2),
    ]

    pairs = pair_scenarios(aggregate_to_scenarios(rows, metric="fact_f1"))
    diffs = tuple(pair.diff for pair in pairs)

    assert len(pairs) == 1
    assert pairs[0].diff == 0.0
    assert wins_ties_losses(pairs) == (0, 1, 0)
    assert n_nonzero(pairs) == 0
    assert rank_biserial(diffs) == 0.0
    assert wilcoxon_p(diffs) == 1.0
    assert permutation_p(diffs, n_resamples=99, seed=0) == 1.0
    low, high = bootstrap_mean_ci(diffs, n_resamples=99, seed=0)
    assert low == 0.0
    assert high == 0.0


def test_wins_ties_losses_count_scenarios_on_aggregated_scores() -> None:
    rows = [
        _row("happy_path_01", "baseline", 1, fact_f1=0.0),
        _row("happy_path_01", "baseline", 2, fact_f1=0.0),
        _row("happy_path_01", "fsm", 1, fact_f1=1.0),
        _row("happy_path_01", "fsm", 2, fact_f1=1.0),
        _row("happy_path_02", "baseline", 1, fact_f1=0.5),
        _row("happy_path_02", "fsm", 1, fact_f1=0.5),
        _row("edge_01", "baseline", 1, fact_f1=1.0),
        _row("edge_01", "fsm", 1, fact_f1=0.0),
    ]

    pairs = pair_scenarios(aggregate_to_scenarios(rows, metric="fact_f1"))

    assert wins_ties_losses(pairs) == (1, 1, 1)
    assert n_nonzero(pairs) == 2


def test_rank_biserial_uses_positive_and_negative_rank_sums() -> None:
    diffs = (10.0, -1.0)

    assert rank_biserial(diffs) == pytest.approx(1 / 3)
    assert (1 - 1) / 2 == 0.0


def test_holm_adjusts_only_the_three_primary_metrics() -> None:
    raw = {
        "task_completed": 0.01,
        "fact_f1": 0.02,
        "claim_support": 0.04,
        "flow_adherence": 0.01,
        "n_turns": 0.001,
    }

    adjusted = holm_adjust(raw)

    assert set(adjusted) == {"task_completed", "fact_f1", "claim_support"}
    assert adjusted["task_completed"] == pytest.approx(0.03)
    assert adjusted["fact_f1"] == pytest.approx(0.04)
    assert adjusted["claim_support"] == pytest.approx(0.04)


def test_per_category_tests_are_labelled_exploratory() -> None:
    assert (
        family_of(
            metric="task_completed",
            population="semantic_primary",
            stratum="happy_path",
        )
        == "exploratory"
    )
    assert (
        family_of(
            metric="task_completed", population="semantic_primary", stratum="overall"
        )
        == "primary"
    )
    assert (
        family_of(
            metric="task_completed", population="human_primary", stratum="overall"
        )
        == "primary"
    )


def test_permutation_keeps_ties_is_two_sided_and_uses_plus_one_correction() -> None:
    from random import Random

    diffs = (3.0, 1.0, 0.0)
    n_resamples = 7
    seed = 0
    observed = permutation_p(diffs, n_resamples=n_resamples, seed=seed)

    rng = Random(seed)
    t_obs = abs(sum(diffs) / len(diffs))
    extreme = 0
    for _ in range(n_resamples):
        permuted = [diff * rng.choice((1, -1)) for diff in diffs]
        assert permuted[2] == 0.0
        if abs(sum(permuted) / len(permuted)) >= t_obs:
            extreme += 1
    expected = (extreme + 1) / (n_resamples + 1)

    assert observed == expected
    assert observed != extreme / n_resamples


def test_bootstrap_ci_is_scenario_level_percentile_of_the_mean_diff() -> None:
    diffs = (0.2, -0.1, 0.4)
    low, high = bootstrap_mean_ci(diffs, n_resamples=200, seed=0)

    assert low <= sum(diffs) / len(diffs) <= high
    assert low < high


def test_wilcoxon_discards_zero_differences() -> None:
    pytest.importorskip("scipy")
    from scipy.stats import wilcoxon

    diffs = (0.4, 0.0, -0.1, 0.2)
    expected = wilcoxon(
        list(diffs), zero_method="wilcox", alternative="two-sided", method="auto"
    )
    assert wilcoxon_p(diffs) == pytest.approx(float(expected.pvalue))
    assert wilcoxon_p((0.0, 0.0)) == 1.0


def test_claim_support_reports_na_counts_and_complete_pairs() -> None:
    rows = [
        _row("happy_path_01", "baseline", 1, claim_support=1.0, n_checkable_claims=2),
        _row("happy_path_01", "fsm", 1, claim_support="", n_checkable_claims=0),
        _row("happy_path_02", "baseline", 1, claim_support=0.5, n_checkable_claims=4),
        _row("happy_path_02", "fsm", 1, claim_support=1.0, n_checkable_claims=3),
        _row("edge_01", "baseline", 1, claim_support="", n_checkable_claims=0),
        _row("edge_01", "fsm", 1, claim_support="", n_checkable_claims=0),
    ]

    table = descriptive_table(
        rows,
        population="semantic_primary",
        metrics=("claim_support",),
        n_resamples=20,
        seed=0,
    )
    overall = {
        row.agent: row
        for row in table
        if row.stratum == "overall" and row.metric == "claim_support"
    }
    edge_fsm = next(
        row
        for row in table
        if row.stratum == "edge"
        and row.agent == "fsm"
        and row.metric == "claim_support"
    )

    assert overall["baseline"].claim_support_total_dialogues == 3
    assert overall["baseline"].claim_support_na_dialogues == 1
    assert overall["baseline"].claim_support_na_pct == pytest.approx(1 / 3)
    assert overall["baseline"].claim_support_aggregatable_dialogues == 2
    assert overall["fsm"].claim_support_total_dialogues == 3
    assert overall["fsm"].claim_support_na_dialogues == 2
    assert overall["fsm"].claim_support_na_pct == pytest.approx(2 / 3)
    assert overall["fsm"].claim_support_aggregatable_dialogues == 1
    assert overall["baseline"].claim_support_complete_pairs == 1
    assert overall["fsm"].claim_support_complete_pairs == 1
    assert overall["baseline"].claim_support_pairs_excluded_either_na == 2
    assert overall["fsm"].claim_support_pairs_excluded_either_na == 2
    assert overall["fsm"].mean == pytest.approx(1.0)
    assert edge_fsm.mean is None
    assert edge_fsm.claim_support_na_dialogues == 1
    assert edge_fsm.claim_support_na_pct == 1.0
    assert edge_fsm.claim_support_aggregatable_dialogues == 0
    assert edge_fsm.claim_support_complete_pairs == 0
    assert edge_fsm.claim_support_pairs_excluded_either_na == 1
    fact_f1 = descriptive_table(
        [_row("happy_path_01", "baseline", 1, fact_f1=0.5)],
        population="semantic_primary",
        metrics=("fact_f1",),
        n_resamples=20,
        seed=0,
    )
    assert fact_f1[0].claim_support_na_dialogues is None


def test_zero_checkable_claim_occurrence_is_compared_between_agents() -> None:
    pytest.importorskip("scipy")
    rows = [
        _row("happy_path_01", "baseline", 1, claim_support=1.0, n_checkable_claims=2),
        _row("happy_path_01", "fsm", 1, claim_support="", n_checkable_claims=0),
        _row("happy_path_02", "baseline", 1, claim_support=0.5, n_checkable_claims=1),
        _row("happy_path_02", "fsm", 1, claim_support=1.0, n_checkable_claims=2),
        _row("edge_01", "baseline", 1, claim_support="", n_checkable_claims=0),
        _row("edge_01", "fsm", 1, claim_support="", n_checkable_claims=0),
        _row("happy_path_03", "baseline", 1, claim_support="", n_checkable_claims=5),
        _row("happy_path_03", "fsm", 1, claim_support="", n_checkable_claims=5),
    ]

    table = paired_test_table(
        rows,
        population="semantic_primary",
        metrics=("claim_support",),
        n_resamples=20,
        seed=0,
    )
    claim = next(
        row
        for row in table
        if row.metric == "claim_support" and row.stratum == "overall"
    )
    occurrence = next(
        row
        for row in table
        if row.metric == "zero_checkable_claim_occurrence" and row.stratum == "overall"
    )

    assert claim.n == 1
    assert claim.claim_support_complete_pairs == 1
    assert claim.claim_support_pairs_excluded_either_na == 3
    assert claim.family == "primary"
    assert claim.wilcoxon_p is not None
    assert claim.wilcoxon_p_holm is not None
    assert occurrence.n == 4
    assert occurrence.baseline_mean == pytest.approx(0.25)
    assert occurrence.fsm_mean == pytest.approx(0.5)
    assert occurrence.mean_diff == pytest.approx(0.25)
    assert (occurrence.wins, occurrence.ties, occurrence.losses) == (1, 3, 0)
    assert occurrence.role == "diagnostic_missingness"
    assert occurrence.family != "primary"
    assert occurrence.wilcoxon_p is None
    assert occurrence.wilcoxon_p_holm is None
    assert occurrence.permutation_p is None
    assert occurrence.rank_biserial is None
    assert occurrence.ci_low is None
    assert occurrence.ci_high is None


def test_intra_scenario_sd_is_na_when_fewer_than_two_values() -> None:
    rows = [
        _row("happy_path_01", "baseline", 1, fact_f1=0.5),
        _row("happy_path_02", "baseline", 1, fact_f1=0.4),
        _row("happy_path_02", "baseline", 2, fact_f1=0.6),
        _row("happy_path_03", "baseline", 1, fact_f1=1.0),
        _row("happy_path_03", "baseline", 2, fact_f1=1.0),
    ]

    scores = {
        score.scenario_id: score
        for score in aggregate_to_scenarios(rows, metric="fact_f1")
    }

    assert scores["happy_path_01"].intra_sd is None
    assert scores["happy_path_02"].intra_sd == pytest.approx(stdev([0.4, 0.6]))
    assert scores["happy_path_03"].intra_sd == 0.0
    defined = [
        score.intra_sd for score in scores.values() if score.intra_sd is not None
    ]
    assert mean_intra_scenario_sd(list(scores.values())) == pytest.approx(
        sum(defined) / 2
    )


def test_descriptive_rows_include_mean_sd_median_ci_and_intra_scenario_sd() -> None:
    rows = [
        _row("happy_path_01", "baseline", 1, fact_f1=0.2),
        _row("happy_path_01", "fsm", 1, fact_f1=0.8),
        _row("happy_path_02", "baseline", 1, fact_f1=0.4),
        _row("happy_path_02", "fsm", 1, fact_f1=1.0),
        _row("edge_01", "baseline", 1, fact_f1=0.0),
        _row("edge_01", "fsm", 1, fact_f1=0.5),
    ]

    table = descriptive_table(
        rows,
        population="semantic_primary",
        metrics=("fact_f1",),
        n_resamples=40,
        seed=0,
    )
    overall_fsm = next(
        row
        for row in table
        if row.stratum == "overall" and row.agent == "fsm" and row.metric == "fact_f1"
    )

    assert overall_fsm.n_scenarios == 3
    assert overall_fsm.n_dialogues == 3
    assert overall_fsm.mean == pytest.approx((0.8 + 1.0 + 0.5) / 3)
    assert overall_fsm.median == pytest.approx(0.8)
    assert overall_fsm.sd is not None
    assert overall_fsm.ci_low <= overall_fsm.mean <= overall_fsm.ci_high
    assert any(row.stratum == "happy_path" for row in table)
    assert any(row.stratum == "edge" for row in table)


def test_paired_test_rows_have_required_columns() -> None:
    pytest.importorskip("scipy")
    rows = [
        _row("happy_path_01", "baseline", 1, task_completed="False"),
        _row("happy_path_01", "fsm", 1, task_completed="True"),
        _row("happy_path_02", "baseline", 1, task_completed="True"),
        _row("happy_path_02", "fsm", 1, task_completed="True"),
        _row("edge_01", "baseline", 1, task_completed="True"),
        _row("edge_01", "fsm", 1, task_completed="False"),
    ]

    table = paired_test_table(
        rows,
        population="semantic_primary",
        metrics=("task_completed",),
        n_resamples=40,
        seed=0,
    )
    overall = next(row for row in table if row.stratum == "overall")
    happy = next(row for row in table if row.stratum == "happy_path")

    assert overall.n == 3
    assert overall.n_nonzero == 2
    assert overall.wins + overall.ties + overall.losses == overall.n
    assert overall.family == "primary"
    assert overall.wilcoxon_p_holm is not None
    assert overall.direction in {"fsm", "baseline", "tie"}
    assert overall.ci_low <= overall.mean_diff <= overall.ci_high
    assert happy.family == "exploratory"
    assert happy.wilcoxon_p_holm is None


def test_drop5_instrument_sensitivity_excludes_the_five_named_scenarios() -> None:
    rows = [
        _row("adversarial_01", "baseline", 1, fact_f1=0.0),
        _row("adversarial_01", "fsm", 1, fact_f1=1.0),
        _row("happy_path_01", "baseline", 1, fact_f1=0.2),
        _row("happy_path_01", "fsm", 1, fact_f1=0.4),
    ]

    kept = rows_for_population(rows, "drop5_instrument")
    pairs = pair_scenarios(aggregate_to_scenarios(kept, metric="fact_f1"))

    assert [pair.scenario_id for pair in pairs] == ["happy_path_01"]
    assert "adversarial_01" in INSTRUMENT_DROP_SCENARIOS


def test_frozen_gate_is_a_second_population_not_the_confirmatory_one() -> None:
    pytest.importorskip("scipy")
    rows = [
        _row("happy_path_01", "baseline", 1, task_completed="True"),
        _row("happy_path_01", "fsm", 1, task_completed="False"),
    ]

    table = paired_test_table(
        rows,
        population="frozen_gate",
        metrics=("task_completed",),
        n_resamples=20,
        seed=0,
    )
    overall = next(row for row in table if row.stratum == "overall")

    assert overall.population == "frozen_gate"
    assert overall.family == "secondary"
    assert overall.wilcoxon_p_holm is None


def test_metrics_doc_states_n_is_scenarios_not_dialogues(metrics_doc: str) -> None:
    phrase = (
        "The n of each test is the number of paired scenarios, "
        "not the number of dialogues."
    )
    assert phrase in metrics_doc
    assert "round(fsm_score - baseline_score, 12)" in metrics_doc
    assert "n_nonzero" in metrics_doc
    assert "claim_support_na_pct" in metrics_doc
    assert "proportion in [0, 1]" in metrics_doc
    assert "zero_checkable_claim_occurrence" in metrics_doc
    assert "diagnostic_missingness" in metrics_doc
