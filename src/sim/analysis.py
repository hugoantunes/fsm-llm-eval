"""Scenario-level paired statistics for T-19.

Dialogue rows from ``metrics.csv`` aggregate to one score per
(scenario, agent, metric). Tests pair those scores by scenario. Paired
differences are canonicalized once and reused by every paired statistic.
"""

from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from csv import DictReader, DictWriter
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from io import StringIO
from pathlib import Path
from random import Random
from statistics import mean, median, stdev
from sys import version_info
from typing import Literal

from sim.io import atomic_write
from sim.metrics import METRICS, PRIMARY_METRICS
from sim.schemas import CATEGORIES, Category

DIFF_DECIMALS = 12
N_RESAMPLES = 10_000
SEED = 0
OVERALL = "overall"
SEMANTIC_PRIMARY = "semantic_primary"
HUMAN_PRIMARY = "human_primary"
FROZEN_GATE = "frozen_gate"
DROP5_INSTRUMENT = "drop5_instrument"
DROP_AMBIGUOUS_FACT_IDS = "drop_ambiguous_fact_ids"
Family = Literal["primary", "secondary", "exploratory"]
Direction = Literal["fsm", "baseline", "tie"]
ZERO_CHECKABLE_METRIC = "zero_checkable_claim_occurrence"
ROLE_DIAGNOSTIC_MISSINGNESS = "diagnostic_missingness"
INSTRUMENT_DROP_SCENARIOS = frozenset(
    {
        "adversarial_01",
        "adversarial_06",
        "adversarial_19",
        "edge_16",
        "happy_path_09",
    }
)
ENVIRONMENT_PACKAGES = ("pandas", "scipy", "numpy")


def category_of(scenario_id: str) -> Category:
    """Return the category encoded in ``scenario_id``.

    Scenario ids name the category (``happy_path_07``). That string is what
    pairs the two agents and what stratifies T-19.
    """
    for category in CATEGORIES:
        prefix = f"{category}_"
        if scenario_id.startswith(prefix) and scenario_id[len(prefix) :].isdigit():
            return category
    raise ValueError(
        f"{scenario_id!r} does not name a category: expected "
        f"'happy_path_07', 'edge_01' or 'adversarial_19'"
    )


@dataclass(frozen=True)
class ScenarioScore:
    """One aggregated (scenario, agent, metric) score."""

    scenario_id: str
    agent: str
    metric: str
    value: float
    n_reps: int
    intra_sd: float | None
    category: Category


def parse_cell(value: object) -> float | None:
    """Turn a metrics.csv cell into a float, or None for NA."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped in {"True", "true"}:
            return 1.0
        if stripped in {"False", "false"}:
            return 0.0
        return float(stripped)
    raise TypeError(f"cannot parse metric cell {value!r}")


def aggregate_to_scenarios(
    rows: Sequence[Mapping[str, object]], *, metric: str
) -> list[ScenarioScore]:
    """Mean (or proportion) of non-NA repetitions per (scenario, agent)."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        parsed = parse_cell(row[metric])
        if parsed is None:
            continue
        key = (str(row["scenario_id"]), str(row["agent"]))
        grouped[key].append(parsed)
    scores = [
        ScenarioScore(
            scenario_id=scenario_id,
            agent=agent,
            metric=metric,
            value=mean(values),
            n_reps=len(values),
            intra_sd=stdev(values) if len(values) >= 2 else None,
            category=category_of(scenario_id),
        )
        for (scenario_id, agent), values in grouped.items()
    ]
    return sorted(scores, key=lambda score: (score.scenario_id, score.agent))


@dataclass(frozen=True)
class PairedScenario:
    """One scenario with a canonical FSM-minus-baseline difference."""

    scenario_id: str
    category: Category
    metric: str
    baseline: float
    fsm: float
    diff: float


def canonical_diff(fsm_score: float, baseline_score: float) -> float:
    """Round ``fsm_score - baseline_score`` once; ``0.0`` is a tie."""
    return round(fsm_score - baseline_score, DIFF_DECIMALS)


def pair_scenarios(scores: Sequence[ScenarioScore]) -> list[PairedScenario]:
    """Pair baseline and FSM by scenario; drop scenarios missing an agent."""
    by_key: dict[tuple[str, str], dict[str, ScenarioScore]] = defaultdict(dict)
    for score in scores:
        by_key[(score.scenario_id, score.metric)][score.agent] = score
    pairs: list[PairedScenario] = []
    for (scenario_id, metric), agents in by_key.items():
        baseline = agents.get("baseline")
        fsm = agents.get("fsm")
        if baseline is None or fsm is None:
            continue
        pairs.append(
            PairedScenario(
                scenario_id=scenario_id,
                category=baseline.category,
                metric=metric,
                baseline=baseline.value,
                fsm=fsm.value,
                diff=canonical_diff(fsm.value, baseline.value),
            )
        )
    return sorted(pairs, key=lambda pair: (pair.metric, pair.scenario_id))


def wins_ties_losses(pairs: Sequence[PairedScenario]) -> tuple[int, int, int]:
    """Count FSM > / == / < baseline from canonical diffs, including ties."""
    wins = ties = losses = 0
    for pair in pairs:
        if pair.diff > 0:
            wins += 1
        elif pair.diff < 0:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def n_nonzero(pairs: Sequence[PairedScenario]) -> int:
    """Count paired canonical differences that are not ties."""
    return sum(pair.diff != 0.0 for pair in pairs)


def rank_biserial(diffs: Sequence[float]) -> float:
    """Kerby matched-pairs rank-biserial from canonical diffs.

    Zero diffs are discarded. Remaining ``|diff|`` values take average ranks
    when tied. All-zero input is ``0.0``.
    """
    nonzero = [diff for diff in diffs if diff != 0.0]
    if not nonzero:
        return 0.0
    ranks = _average_ranks([abs(diff) for diff in nonzero])
    r_plus = sum(rank for diff, rank in zip(nonzero, ranks, strict=True) if diff > 0)
    r_minus = sum(rank for diff, rank in zip(nonzero, ranks, strict=True) if diff < 0)
    return (r_plus - r_minus) / (r_plus + r_minus)


def holm_adjust(
    pvalues: Mapping[str, float],
    *,
    family: Sequence[str] = PRIMARY_METRICS,
) -> dict[str, float]:
    """Holm-adjust p-values that belong to ``family``; ignore the rest."""
    names = [name for name in family if name in pvalues]
    ordered = sorted(names, key=lambda name: pvalues[name])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        candidate = min(1.0, pvalues[name] * (len(names) - index))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def family_of(*, metric: str, population: str, stratum: str) -> Family:
    """Label a test as primary, secondary, or exploratory."""
    if stratum != OVERALL or population == DROP_AMBIGUOUS_FACT_IDS:
        return "exploratory"
    if population in {SEMANTIC_PRIMARY, HUMAN_PRIMARY} and metric in PRIMARY_METRICS:
        return "primary"
    return "secondary"


@dataclass(frozen=True)
class AgentClaimSupportMissingness:
    """Dialogue-level claim_support NA counts for one agent."""

    total_dialogues: int
    na_dialogues: int
    aggregatable_dialogues: int

    @property
    def na_pct(self) -> float | None:
        """NA dialogues / total dialogues, a proportion in [0, 1]."""
        if self.total_dialogues == 0:
            return None
        return self.na_dialogues / self.total_dialogues


@dataclass(frozen=True)
class ClaimSupportMissingness:
    """Per-agent NA counts and complete-pair n for claim_support."""

    baseline: AgentClaimSupportMissingness
    fsm: AgentClaimSupportMissingness
    complete_pairs: int
    pairs_excluded_either_na: int

    def for_agent(self, agent: str) -> AgentClaimSupportMissingness:
        """Return the per-agent NA counts."""
        if agent == "baseline":
            return self.baseline
        if agent == "fsm":
            return self.fsm
        raise ValueError(f"unknown agent {agent!r}")


def _rows_for_stratum(
    rows: Sequence[Mapping[str, object]], stratum: str
) -> list[Mapping[str, object]]:
    """Keep dialogue rows that belong to ``stratum`` (or all, if overall)."""
    if stratum == OVERALL:
        return list(rows)
    return [row for row in rows if category_of(str(row["scenario_id"])) == stratum]


def claim_support_missingness(
    rows: Sequence[Mapping[str, object]],
    *,
    stratum: str = OVERALL,
) -> ClaimSupportMissingness:
    """Count claim_support NA dialogues and how many scenario pairs remain."""
    subset = _rows_for_stratum(rows, stratum)
    totals = {"baseline": 0, "fsm": 0}
    n_na = {"baseline": 0, "fsm": 0}
    for row in subset:
        agent = str(row["agent"])
        if agent not in totals:
            continue
        totals[agent] += 1
        if parse_cell(row["claim_support"]) is None:
            n_na[agent] += 1
    pairs = pair_scenarios(aggregate_to_scenarios(subset, metric="claim_support"))
    scenarios = {str(row["scenario_id"]) for row in subset}

    def counts(agent: str) -> AgentClaimSupportMissingness:
        return AgentClaimSupportMissingness(
            total_dialogues=totals[agent],
            na_dialogues=n_na[agent],
            aggregatable_dialogues=totals[agent] - n_na[agent],
        )

    return ClaimSupportMissingness(
        baseline=counts("baseline"),
        fsm=counts("fsm"),
        complete_pairs=len(pairs),
        pairs_excluded_either_na=len(scenarios) - len(pairs),
    )


def zero_checkable_indicator(row: Mapping[str, object]) -> float | None:
    """1 if ``n_checkable_claims == 0``, else 0; None when the count is missing."""
    n_claims = parse_cell(row["n_checkable_claims"])
    if n_claims is None:
        return None
    return float(n_claims == 0.0)


def zero_checkable_pairs(
    rows: Sequence[Mapping[str, object]],
) -> list[PairedScenario]:
    """Pair the per-scenario rate of repetitions with ``n_checkable_claims == 0``."""
    tagged = [
        {**dict(row), ZERO_CHECKABLE_METRIC: zero_checkable_indicator(row)}
        for row in rows
    ]
    return pair_scenarios(aggregate_to_scenarios(tagged, metric=ZERO_CHECKABLE_METRIC))


def wilcoxon_p(diffs: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p on canonical diffs.

    All-zero diffs return ``1.0`` without calling SciPy.
    """
    if all(diff == 0.0 for diff in diffs):
        return 1.0
    wilcoxon = _scipy_wilcoxon()
    result = wilcoxon(
        list(diffs),
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.pvalue)


def permutation_p(
    diffs: Sequence[float],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> float:
    """Two-sided paired sign-flip p of the absolute mean canonical difference."""
    values = list(diffs)
    t_obs = abs(mean(values))
    rng = Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        permuted = [diff * rng.choice((1, -1)) for diff in values]
        if abs(mean(permuted)) >= t_obs:
            extreme += 1
    return (extreme + 1) / (n_resamples + 1)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling the given values with replacement."""
    data = list(values)
    n = len(data)
    rng = Random(seed)
    means = sorted(
        mean(data[rng.randrange(n)] for _ in range(n)) for _ in range(n_resamples)
    )
    return _percentile(means, 2.5), _percentile(means, 97.5)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based average ranks of ``values``, tying equal numbers."""
    n = len(values)
    order = sorted(range(n), key=lambda index: values[index])
    ranks = [0.0] * n
    start = 0
    while start < n:
        end = start
        while end + 1 < n and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + 1 + end + 1) / 2
        for index in range(start, end + 1):
            ranks[order[index]] = average
        start = end + 1
    return ranks


def _percentile(sorted_values: Sequence[float], percent: float) -> float:
    """Linear-interpolation percentile of an already sorted sample."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (percent / 100) * (n - 1)
    low = int(rank)
    high = min(low + 1, n - 1)
    fraction = rank - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def _scipy_wilcoxon() -> Callable[..., object]:
    """Import SciPy's Wilcoxon, or say how to install the analysis group."""
    try:
        from scipy.stats import wilcoxon
    except ImportError as error:
        raise ImportError(
            "scipy is required for Wilcoxon: just install-analysis"
        ) from error
    return wilcoxon


def mean_intra_scenario_sd(scores: Sequence[ScenarioScore]) -> float | None:
    """Mean of defined intra-scenario SDs; None if none are defined."""
    defined = [score.intra_sd for score in scores if score.intra_sd is not None]
    if not defined:
        return None
    return mean(defined)


def rows_for_population(
    rows: Sequence[Mapping[str, object]], population: str
) -> list[dict[str, object]]:
    """Return dialogue rows for a named analysis population."""
    copied = [dict(row) for row in rows]
    if population == DROP5_INSTRUMENT:
        return [
            row
            for row in copied
            if str(row["scenario_id"]) not in INSTRUMENT_DROP_SCENARIOS
        ]
    return copied


def rows_without_dialogues(
    rows: Sequence[Mapping[str, object]],
    dialogues: Collection[tuple[str, str, int]],
) -> list[dict[str, object]]:
    """Drop the dialogues keyed ``(scenario_id, agent, repetition)``.

    Exclusion is per dialogue: a scenario keeps the mean of its remaining
    repetitions. A key absent from ``rows`` raises instead of filtering nothing.
    """
    excluded = set(dialogues)
    missing = sorted(excluded - {dialogue_key(row) for row in rows})
    if missing:
        raise ValueError(f"dialogues not in the metrics rows: {missing}")
    return [dict(row) for row in rows if dialogue_key(row) not in excluded]


def dialogue_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    """``(scenario_id, agent, repetition)`` of a metrics row."""
    return (str(row["scenario_id"]), str(row["agent"]), int(str(row["repetition"])))


@dataclass(frozen=True)
class DescriptiveRow:
    """One descriptive cell: population x stratum x metric x agent."""

    population: str
    stratum: str
    metric: str
    agent: str
    n_scenarios: int
    n_dialogues: int
    mean: float | None
    sd: float | None
    median: float | None
    ci_low: float | None
    ci_high: float | None
    mean_intra_scenario_sd: float | None
    claim_support_total_dialogues: int | None = None
    claim_support_na_dialogues: int | None = None
    claim_support_na_pct: float | None = None
    claim_support_aggregatable_dialogues: int | None = None
    claim_support_complete_pairs: int | None = None
    claim_support_pairs_excluded_either_na: int | None = None


@dataclass(frozen=True)
class TestRow:
    """One paired test: population x stratum x metric."""

    population: str
    stratum: str
    metric: str
    family: Family
    n: int
    n_nonzero: int
    n_dropped_unpaired: int
    wilcoxon_p: float | None
    wilcoxon_p_holm: float | None
    permutation_p: float | None
    rank_biserial: float | None
    mean_diff: float
    ci_low: float | None
    ci_high: float | None
    wins: int
    ties: int
    losses: int
    direction: Direction
    role: str | None = None
    baseline_mean: float | None = None
    fsm_mean: float | None = None
    claim_support_complete_pairs: int | None = None
    claim_support_pairs_excluded_either_na: int | None = None


def descriptive_table(
    rows: Sequence[Mapping[str, object]],
    *,
    population: str,
    metrics: Sequence[str] | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> list[DescriptiveRow]:
    """Scenario-level mean, SD, median, CI and intra-scenario SD."""
    names = metrics if metrics is not None else METRICS
    table: list[DescriptiveRow] = []
    for metric in names:
        scores = aggregate_to_scenarios(rows, metric=metric)
        missingness_by_stratum = (
            {
                stratum: claim_support_missingness(rows, stratum=stratum)
                for stratum in (OVERALL, *CATEGORIES)
            }
            if metric == "claim_support"
            else {}
        )
        for stratum in (OVERALL, *CATEGORIES):
            subset = (
                scores
                if stratum == OVERALL
                else [score for score in scores if score.category == stratum]
            )
            missingness = missingness_by_stratum.get(stratum)
            for agent in ("baseline", "fsm"):
                agent_scores = [score for score in subset if score.agent == agent]
                if not agent_scores and (
                    missingness is None
                    or missingness.for_agent(agent).total_dialogues == 0
                ):
                    continue
                table.append(
                    _descriptive_row(
                        population=population,
                        stratum=stratum,
                        metric=metric,
                        agent=agent,
                        scores=agent_scores,
                        n_resamples=n_resamples,
                        seed=seed,
                        missingness=missingness,
                    )
                )
    return table


def paired_test_table(
    rows: Sequence[Mapping[str, object]],
    *,
    population: str,
    metrics: Sequence[str] | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> list[TestRow]:
    """Paired tests over scenario-level scores, Holm only on the primary family."""
    names = metrics if metrics is not None else METRICS
    table: list[TestRow] = []
    for metric in names:
        scores = aggregate_to_scenarios(rows, metric=metric)
        for stratum in (OVERALL, *CATEGORIES):
            subset = (
                scores
                if stratum == OVERALL
                else [score for score in scores if score.category == stratum]
            )
            pairs = pair_scenarios(subset)
            if not pairs:
                continue
            diffs = tuple(pair.diff for pair in pairs)
            wins, ties, losses = wins_ties_losses(pairs)
            mean_diff = mean(diffs)
            ci_low, ci_high = bootstrap_mean_ci(
                diffs, n_resamples=n_resamples, seed=seed
            )
            missingness = (
                claim_support_missingness(rows, stratum=stratum)
                if metric == "claim_support"
                else None
            )
            table.append(
                TestRow(
                    population=population,
                    stratum=stratum,
                    metric=metric,
                    family=family_of(
                        metric=metric, population=population, stratum=stratum
                    ),
                    n=len(pairs),
                    n_nonzero=n_nonzero(pairs),
                    n_dropped_unpaired=_n_unpaired(subset),
                    wilcoxon_p=wilcoxon_p(diffs),
                    wilcoxon_p_holm=None,
                    permutation_p=permutation_p(
                        diffs, n_resamples=n_resamples, seed=seed
                    ),
                    rank_biserial=rank_biserial(diffs),
                    mean_diff=mean_diff,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    wins=wins,
                    ties=ties,
                    losses=losses,
                    direction=_direction(mean_diff),
                    claim_support_complete_pairs=(
                        missingness.complete_pairs if missingness is not None else None
                    ),
                    claim_support_pairs_excluded_either_na=(
                        missingness.pairs_excluded_either_na
                        if missingness is not None
                        else None
                    ),
                )
            )
    if any("n_checkable_claims" in row for row in rows):
        table.extend(_zero_checkable_test_rows(rows, population=population))
    if population in {SEMANTIC_PRIMARY, HUMAN_PRIMARY}:
        primary = {
            row.metric: row.wilcoxon_p
            for row in table
            if row.family == "primary"
            and row.stratum == OVERALL
            and row.wilcoxon_p is not None
        }
        adjusted = holm_adjust(primary)
        table = [
            replace(row, wilcoxon_p_holm=adjusted.get(row.metric))
            if row.family == "primary" and row.stratum == OVERALL
            else row
            for row in table
        ]
    return table


def load_metrics_csv(path: Path) -> list[dict[str, object]]:
    """Load a T-18 metrics CSV as dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in DictReader(handle)]


def environment_text() -> str:
    """Python and analysis-package versions for ``environment.txt``."""
    lines = [f"python={_python_version()}"]
    for name in ENVIRONMENT_PACKAGES:
        try:
            lines.append(f"{name}={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name}=not-installed")
    return "\n".join(lines) + "\n"


def analyze(
    metrics_path: Path,
    frozen_path: Path,
    out_dir: Path,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> None:
    """Write descriptive.csv, tests.csv and environment.txt for one experiment."""
    semantic = load_metrics_csv(metrics_path)
    frozen = load_metrics_csv(frozen_path)
    populations = {
        SEMANTIC_PRIMARY: semantic,
        FROZEN_GATE: frozen,
        DROP5_INSTRUMENT: rows_for_population(semantic, DROP5_INSTRUMENT),
    }
    descriptive: list[DescriptiveRow] = []
    tests: list[TestRow] = []
    for population, pop_rows in populations.items():
        descriptive.extend(
            descriptive_table(
                pop_rows,
                population=population,
                n_resamples=n_resamples,
                seed=seed,
            )
        )
        tests.extend(
            paired_test_table(
                pop_rows,
                population=population,
                n_resamples=n_resamples,
                seed=seed,
            )
        )
    _write_csv(out_dir / "descriptive.csv", [asdict(row) for row in descriptive])
    _write_csv(out_dir / "tests.csv", [asdict(row) for row in tests])
    atomic_write(out_dir / "environment.txt", environment_text())


def _descriptive_row(
    *,
    population: str,
    stratum: str,
    metric: str,
    agent: str,
    scores: Sequence[ScenarioScore],
    n_resamples: int,
    seed: int,
    missingness: ClaimSupportMissingness | None,
) -> DescriptiveRow:
    """Build one descriptive cell, attaching claim_support missingness when given."""
    extra = _claim_support_columns(missingness, agent)
    if not scores:
        return DescriptiveRow(
            population=population,
            stratum=stratum,
            metric=metric,
            agent=agent,
            n_scenarios=0,
            n_dialogues=0,
            mean=None,
            sd=None,
            median=None,
            ci_low=None,
            ci_high=None,
            mean_intra_scenario_sd=None,
            **extra,
        )
    values = [score.value for score in scores]
    ci_low, ci_high = bootstrap_mean_ci(values, n_resamples=n_resamples, seed=seed)
    return DescriptiveRow(
        population=population,
        stratum=stratum,
        metric=metric,
        agent=agent,
        n_scenarios=len(scores),
        n_dialogues=sum(score.n_reps for score in scores),
        mean=mean(values),
        sd=stdev(values) if len(values) >= 2 else None,
        median=median(values),
        ci_low=ci_low,
        ci_high=ci_high,
        mean_intra_scenario_sd=mean_intra_scenario_sd(scores),
        **extra,
    )


def _claim_support_columns(
    missingness: ClaimSupportMissingness | None, agent: str
) -> dict[str, int | float | None]:
    """Map missingness onto the descriptive claim_support columns; else None."""
    if missingness is None:
        return {
            "claim_support_total_dialogues": None,
            "claim_support_na_dialogues": None,
            "claim_support_na_pct": None,
            "claim_support_aggregatable_dialogues": None,
            "claim_support_complete_pairs": None,
            "claim_support_pairs_excluded_either_na": None,
        }
    agent_na = missingness.for_agent(agent)
    return {
        "claim_support_total_dialogues": agent_na.total_dialogues,
        "claim_support_na_dialogues": agent_na.na_dialogues,
        "claim_support_na_pct": agent_na.na_pct,
        "claim_support_aggregatable_dialogues": agent_na.aggregatable_dialogues,
        "claim_support_complete_pairs": missingness.complete_pairs,
        "claim_support_pairs_excluded_either_na": missingness.pairs_excluded_either_na,
    }


def _zero_checkable_test_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    population: str,
) -> list[TestRow]:
    """Descriptive paired comparison of zero-checkable-claim rates; no p-values."""
    tagged = [
        {**dict(row), ZERO_CHECKABLE_METRIC: zero_checkable_indicator(row)}
        for row in rows
    ]
    scores = aggregate_to_scenarios(tagged, metric=ZERO_CHECKABLE_METRIC)
    table: list[TestRow] = []
    for stratum in (OVERALL, *CATEGORIES):
        subset = (
            scores
            if stratum == OVERALL
            else [score for score in scores if score.category == stratum]
        )
        pairs = pair_scenarios(subset)
        if not pairs:
            continue
        diffs = tuple(pair.diff for pair in pairs)
        wins, ties, losses = wins_ties_losses(pairs)
        mean_diff = mean(diffs)
        table.append(
            TestRow(
                population=population,
                stratum=stratum,
                metric=ZERO_CHECKABLE_METRIC,
                family=family_of(
                    metric=ZERO_CHECKABLE_METRIC,
                    population=population,
                    stratum=stratum,
                ),
                n=len(pairs),
                n_nonzero=n_nonzero(pairs),
                n_dropped_unpaired=_n_unpaired(subset),
                wilcoxon_p=None,
                wilcoxon_p_holm=None,
                permutation_p=None,
                rank_biserial=None,
                mean_diff=mean_diff,
                ci_low=None,
                ci_high=None,
                wins=wins,
                ties=ties,
                losses=losses,
                direction=_direction(mean_diff),
                role=ROLE_DIAGNOSTIC_MISSINGNESS,
                baseline_mean=mean(pair.baseline for pair in pairs),
                fsm_mean=mean(pair.fsm for pair in pairs),
            )
        )
    return table


def _n_unpaired(scores: Sequence[ScenarioScore]) -> int:
    """Count scenarios that have a score for exactly one agent."""
    agents: dict[str, set[str]] = defaultdict(set)
    for score in scores:
        agents[score.scenario_id].add(score.agent)
    return sum(len(present) == 1 for present in agents.values())


def _direction(mean_diff: float) -> Direction:
    """Sign of the mean canonical paired difference."""
    if mean_diff > 0:
        return "fsm"
    if mean_diff < 0:
        return "baseline"
    return "tie"


def _python_version() -> str:
    """Installed interpreter version."""
    return f"{version_info.major}.{version_info.minor}.{version_info.micro}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write ``rows`` with empty cells for None."""
    if not rows:
        raise ValueError(f"{path} would have no rows")
    fieldnames = list(rows[0].keys())
    buffer = StringIO()
    writer = DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: "" if row[key] is None else row[key] for key in fieldnames}
        )
    atomic_write(path, buffer.getvalue())
