"""Thesis tables and figures from frozen T-19 outputs (T-20).

A presentation layer: read ``metrics.csv``, ``descriptive.csv`` and
``tests.csv``, format Portuguese plates, write files. Does not import
``sim.analysis`` calculators and does not rewrite those CSVs.
"""

import json
from collections import defaultdict
from contextlib import suppress
from csv import DictReader, DictWriter
from os import environ
from pathlib import Path
from statistics import mean

from sim.io import atomic_write
from sim.metrics import PRIMARY_METRICS

PERCENT_DECIMALS = 4
NUMBER_DECIMALS = 3
P_VALUE_DECIMALS = 3
P_VALUE_FLOOR = 0.001
FIGURE_WIDTH_CM = 16.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / 2.54
FIGURE_HEIGHT_IN = 5.0
SCORE_YLIM = (0.0, 1.05)
PNG_DPI = 300
SEMANTIC_PRIMARY = "semantic_primary"
HUMAN_PRIMARY = "human_primary"
OVERALL = "overall"
UNPAIRED_SCENARIO = "adversarial_12"
CATEGORIES = ("happy_path", "edge", "adversarial")
AGENTS = ("baseline", "fsm")

METRIC_LABELS = {
    "task_completed": "Tarefa concluída",
    "fact_f1": "F1 dos fatos esperados",
    "claim_support": "Suporte das afirmações",
}
METRIC_TICK_LABELS = {
    "task_completed": "Tarefa\nconcluída",
    "fact_f1": "F1 dos fatos\nesperados",
    "claim_support": "Suporte das\nafirmações",
}
AGENT_LABELS = {"baseline": "Baseline", "fsm": "FSM"}
AGENT_FACECOLORS = {"baseline": "#56B4E9", "fsm": "#E69F00"}
AGENT_HATCHES = {"baseline": "///", "fsm": "..."}
WTL_FACECOLORS = ("#009E73", "#CCCCCC", "#D55E00")
WTL_HATCHES = ("///", "...", "xxx")
WTL_INTERNAL_LABEL_MIN = 4
COLOR_HATCH_NOTE = (
    "Cores e hachuras são usadas como codificação visual redundante; "
    "as hachuras preservam a distinção entre séries em impressão monocromática."
)
CATEGORY_LABELS = {
    "overall": "Geral",
    "happy_path": "Caminho feliz",
    "edge": "Borda",
    "adversarial": "Adversarial",
}

TABLE_01_NAME = "resultados_tabela-01_descritiva-geral.csv"
TABLE_02_NAME = "resultados_tabela-02_descritiva-categoria.csv"
TABLE_03_NAME = "resultados_tabela-03_testes-pareados.csv"
TABLE_04_NAME = "resultados_tabela-04_humano-juiz.csv"
TABLE_05_NAME = "resultados_tabela-05_conclusoes-humano-juiz.csv"
FIGURE_01_STEM = "resultados_figura-01_boxplot-metricas-primarias"
FIGURE_02_STEM = "resultados_figura-02_barras-por-categoria"
FIGURE_03_STEM = "resultados_figura-03_vitorias-empates-derrotas"
FIGURE_04_STEM = "resultados_figura-04_humano-juiz"
POPULATION_LABELS = {
    SEMANTIC_PRIMARY: "semântica primária",
    HUMAN_PRIMARY: "humana primária",
}
SOURCE_ROOTS = {
    SEMANTIC_PRIMARY: "results/exp_final",
    HUMAN_PRIMARY: "results/human_primary",
}

TABLE_01_FIELDS = (
    "Métrica",
    "Agente",
    "n cenários",
    "Média",
    "DP",
    "IC 95% inf.",
    "IC 95% sup.",
    "NA (%)",
    "Pares completos",
    "Sem par completo",
)
TABLE_02_FIELDS = (
    "Métrica",
    "Categoria",
    "Agente",
    "n cenários",
    "Média",
    "DP",
    "IC 95% inf.",
    "IC 95% sup.",
)
TABLE_03_FIELDS = (
    "Métrica",
    "n",
    "n não nulo",
    "Não pareados",
    "p Wilcoxon",
    "p Holm",
    "p permutação",
    "Rank-biserial",
    "Diferença média",
    "IC 95% inf.",
    "IC 95% sup.",
    "Vitórias",
    "Empates",
    "Derrotas",
)
TABLE_04_FIELDS = (
    "Comparação",
    "n",
    "Acordo %",
    "Cohen kappa",
    "Micro F1",
    "Observação",
)
TABLE_05_FIELDS = (
    "Métrica",
    "Direção juiz",
    "Direção humano",
    "Diferença juiz",
    "Diferença humano",
    "p Holm juiz",
    "p Holm humano",
    "Mesma direção",
    "Mesma significância",
)
DIRECTION_LABELS = {"baseline": "Baseline", "fsm": "FSM", "tie": "Empate"}


def format_pt_br_number(value: float, *, decimals: int = NUMBER_DECIMALS) -> str:
    """Format a number with a decimal comma and no thousands separator."""
    return f"{value:.{decimals}f}".replace(".", ",")


def format_as_percent(value: float) -> str:
    """Format a [0, 1] proportion as a Portuguese percentage.

    ``0.005714285714285714`` becomes ``0,5714%``. The source CSV stays a
    fraction; this is display only.
    """
    return f"{format_pt_br_number(value * 100, decimals=PERCENT_DECIMALS)}%"


def format_p_value(value: float) -> str:
    """Format a p-value with a floor of ``<0,001``."""
    if value < P_VALUE_FLOOR:
        return f"<{format_pt_br_number(P_VALUE_FLOOR, decimals=P_VALUE_DECIMALS)}"
    return format_pt_br_number(value, decimals=P_VALUE_DECIMALS)


def resolve_population(rows: list[dict[str, str]], requested: str | None = None) -> str:
    """Prefer an explicit population; else semantic-primary; else human-primary."""
    if requested:
        return requested
    pops = {row["population"] for row in rows if row.get("population", "").strip()}
    if SEMANTIC_PRIMARY in pops:
        return SEMANTIC_PRIMARY
    if HUMAN_PRIMARY in pops:
        return HUMAN_PRIMARY
    return SEMANTIC_PRIMARY


def write_thesis_artifacts(
    results_dir: Path | None = None,
    *,
    metrics_path: Path | None = None,
    descriptive_path: Path | None = None,
    tests_path: Path | None = None,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    population: str | None = None,
) -> None:
    """Write thesis tables and figures from a frozen T-19 result directory."""
    root = results_dir if results_dir is not None else Path("results/exp_final")
    metrics_file = metrics_path if metrics_path is not None else root / "metrics.csv"
    descriptive_file = (
        descriptive_path if descriptive_path is not None else root / "descriptive.csv"
    )
    tests_file = tests_path if tests_path is not None else root / "tests.csv"
    out_tables = tables_dir if tables_dir is not None else root / "tables"
    out_figures = figures_dir if figures_dir is not None else root / "figures"
    descriptive = _read_csv(descriptive_file)
    tests = _read_csv(tests_file)
    metrics = _read_csv(metrics_file)
    chosen = resolve_population(descriptive, population)
    out_tables.mkdir(parents=True, exist_ok=True)
    _write_table_01(descriptive, tests, out_tables, population=chosen)
    _write_table_02(descriptive, out_tables, population=chosen)
    _write_table_03(tests, out_tables, population=chosen)
    with suppress(ImportError):
        _write_figures(metrics, descriptive, tests, out_figures, population=chosen)
    if (root / "agreement.json").is_file():
        with suppress(ImportError):
            write_comparison_artifacts(
                root, tables_dir=out_tables, figures_dir=out_figures
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Load a UTF-8 CSV as a list of string dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(DictReader(handle))


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_float(value: str) -> float | None:
    stripped = value.strip()
    if stripped == "":
        return None
    if stripped in {"True", "true"}:
        return 1.0
    if stripped in {"False", "false"}:
        return 0.0
    return float(stripped)


def _format_optional(value: str, *, decimals: int = NUMBER_DECIMALS) -> str:
    parsed = _parse_float(value)
    if parsed is None:
        return "—"
    return format_pt_br_number(parsed, decimals=decimals)


def _format_int(value: str) -> str:
    parsed = _parse_float(value)
    if parsed is None:
        return "—"
    return str(int(parsed))


def _primary_descriptive(
    rows: list[dict[str, str]], *, stratum: str, population: str = SEMANTIC_PRIMARY
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["population"] == population
        and row["stratum"] == stratum
        and row["metric"] in PRIMARY_METRICS
    ]
    order = {metric: index for index, metric in enumerate(PRIMARY_METRICS)}
    agent_order = {agent: index for index, agent in enumerate(AGENTS)}
    return sorted(
        selected,
        key=lambda row: (order[row["metric"]], agent_order.get(row["agent"], 99)),
    )


def _primary_tests(
    rows: list[dict[str, str]], *, population: str = SEMANTIC_PRIMARY
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["population"] == population
        and row["stratum"] == OVERALL
        and row["metric"] in PRIMARY_METRICS
    ]
    order = {metric: index for index, metric in enumerate(PRIMARY_METRICS)}
    return sorted(selected, key=lambda row: order[row["metric"]])


def _write_table_01(
    descriptive: list[dict[str, str]],
    tests: list[dict[str, str]],
    tables_dir: Path,
    *,
    population: str = SEMANTIC_PRIMARY,
) -> None:
    rows: list[dict[str, str]] = []
    for source in _primary_descriptive(
        descriptive, stratum=OVERALL, population=population
    ):
        is_claim = source["metric"] == "claim_support"
        rows.append(
            {
                "Métrica": METRIC_LABELS[source["metric"]],
                "Agente": AGENT_LABELS[source["agent"]],
                "n cenários": _format_int(source["n_scenarios"]),
                "Média": _format_optional(source["mean"]),
                "DP": _format_optional(source["sd"]),
                "IC 95% inf.": _format_optional(source["ci_low"]),
                "IC 95% sup.": _format_optional(source["ci_high"]),
                "NA (%)": (
                    format_as_percent(float(source["claim_support_na_pct"]))
                    if is_claim and source["claim_support_na_pct"].strip()
                    else "—"
                ),
                "Pares completos": (
                    _format_int(source["claim_support_complete_pairs"])
                    if is_claim
                    else "—"
                ),
                "Sem par completo": (
                    _format_int(source["claim_support_pairs_excluded_either_na"])
                    if is_claim
                    else "—"
                ),
            }
        )
    path = tables_dir / TABLE_01_NAME
    _write_csv(path, TABLE_01_FIELDS, rows)
    atomic_write(
        path.with_suffix(".legenda.txt"),
        _table_01_legend(descriptive, tests, population=population),
    )


def _write_table_02(
    descriptive: list[dict[str, str]],
    tables_dir: Path,
    *,
    population: str = SEMANTIC_PRIMARY,
) -> None:
    rows: list[dict[str, str]] = []
    order = {metric: index for index, metric in enumerate(PRIMARY_METRICS)}
    category_order = {category: index for index, category in enumerate(CATEGORIES)}
    agent_order = {agent: index for index, agent in enumerate(AGENTS)}
    selected = [
        row
        for row in descriptive
        if row["population"] == population
        and row["stratum"] in CATEGORIES
        and row["metric"] in PRIMARY_METRICS
    ]
    selected.sort(
        key=lambda row: (
            order[row["metric"]],
            category_order[row["stratum"]],
            agent_order.get(row["agent"], 99),
        )
    )
    for source in selected:
        rows.append(
            {
                "Métrica": METRIC_LABELS[source["metric"]],
                "Categoria": CATEGORY_LABELS[source["stratum"]],
                "Agente": AGENT_LABELS[source["agent"]],
                "n cenários": _format_int(source["n_scenarios"]),
                "Média": _format_optional(source["mean"]),
                "DP": _format_optional(source["sd"]),
                "IC 95% inf.": _format_optional(source["ci_low"]),
                "IC 95% sup.": _format_optional(source["ci_high"]),
            }
        )
    path = tables_dir / TABLE_02_NAME
    _write_csv(path, TABLE_02_FIELDS, rows)
    atomic_write(
        path.with_suffix(".legenda.txt"),
        _table_02_legend(descriptive, population=population),
    )


def _write_table_03(
    tests: list[dict[str, str]],
    tables_dir: Path,
    *,
    population: str = SEMANTIC_PRIMARY,
) -> None:
    rows: list[dict[str, str]] = []
    for source in _primary_tests(tests, population=population):
        holm = source["wilcoxon_p_holm"].strip()
        rows.append(
            {
                "Métrica": METRIC_LABELS[source["metric"]],
                "n": _format_int(source["n"]),
                "n não nulo": _format_int(source["n_nonzero"]),
                "Não pareados": _format_int(source["n_dropped_unpaired"]),
                "p Wilcoxon": format_p_value(float(source["wilcoxon_p"])),
                "p Holm": format_p_value(float(holm)) if holm else "—",
                "p permutação": format_p_value(float(source["permutation_p"])),
                "Rank-biserial": _format_optional(source["rank_biserial"]),
                "Diferença média": _format_optional(source["mean_diff"]),
                "IC 95% inf.": _format_optional(source["ci_low"]),
                "IC 95% sup.": _format_optional(source["ci_high"]),
                "Vitórias": _format_int(source["wins"]),
                "Empates": _format_int(source["ties"]),
                "Derrotas": _format_int(source["losses"]),
            }
        )
    path = tables_dir / TABLE_03_NAME
    _write_csv(path, TABLE_03_FIELDS, rows)
    atomic_write(
        path.with_suffix(".legenda.txt"),
        _table_03_legend(tests, population=population),
    )


def _claim_support_descriptive(
    descriptive: list[dict[str, str]],
    agent: str,
    *,
    population: str = SEMANTIC_PRIMARY,
) -> dict[str, str] | None:
    for row in descriptive:
        if (
            row["population"] == population
            and row["stratum"] == OVERALL
            and row["metric"] == "claim_support"
            and row["agent"] == agent
        ):
            return row
    return None


def _claim_support_test(
    tests: list[dict[str, str]], *, population: str = SEMANTIC_PRIMARY
) -> dict[str, str] | None:
    for row in _primary_tests(tests, population=population):
        if row["metric"] == "claim_support":
            return row
    return None


def frozen_ci_yerr(mean: float, ci_low: float, ci_high: float) -> tuple[float, float]:
    """Convert frozen CI bounds into matplotlib yerr distances."""
    return (mean - ci_low, ci_high - mean)


def _count_cell(row: dict[str, str] | None, field: str) -> str:
    if row is None or not row.get(field, "").strip():
        return "—"
    return _format_int(row[field])


def _table_01_legend(
    descriptive: list[dict[str, str]],
    tests: list[dict[str, str]],
    *,
    population: str = SEMANTIC_PRIMARY,
) -> str:
    baseline = _claim_support_descriptive(
        descriptive, "baseline", population=population
    )
    fsm = _claim_support_descriptive(descriptive, "fsm", population=population)
    claim_test = _claim_support_test(tests, population=population)
    unpaired = (
        _format_int(claim_test["n_dropped_unpaired"]) if claim_test is not None else "—"
    )
    complete = _count_cell(baseline, "claim_support_complete_pairs")
    excluded = _count_cell(baseline, "claim_support_pairs_excluded_either_na")
    baseline_n = _count_cell(baseline, "n_scenarios")
    fsm_n = _count_cell(fsm, "n_scenarios")
    baseline_na = (
        format_as_percent(float(baseline["claim_support_na_pct"]))
        if baseline is not None and baseline["claim_support_na_pct"].strip()
        else "—"
    )
    fsm_na = (
        format_as_percent(float(fsm["claim_support_na_pct"]))
        if fsm is not None and fsm["claim_support_na_pct"].strip()
        else "—"
    )
    test_n = _format_int(claim_test["n"]) if claim_test is not None else "—"
    baseline_na_n = _count_cell(baseline, "claim_support_na_dialogues")
    baseline_total = _count_cell(baseline, "claim_support_total_dialogues")
    fsm_na_n = _count_cell(fsm, "claim_support_na_dialogues")
    fsm_total = _count_cell(fsm, "claim_support_total_dialogues")
    label = POPULATION_LABELS.get(population, population.replace("_", " "))
    source = SOURCE_ROOTS.get(population, f"results/{population}")
    return (
        "Título: Tabela 1 — Estatísticas descritivas das métricas primárias "
        f"(população {label}, geral).\n"
        f"Fonte: elaboração própria, a partir de {source}/descriptive.csv.\n"
        "Notas: unidade de análise = cenário. n descritivo baseline = "
        f"{baseline_n}; n descritivo FSM = {fsm_n}. {UNPAIRED_SCENARIO} não tem "
        f"linha baseline: exclusão estrutural (tests.csv n_dropped_unpaired = "
        f"{unpaired}), não um valor NA. NA de suporte das afirmações é taxa de "
        f"diálogos (coluna NA %): baseline {baseline_na} ({baseline_na_n}/"
        f"{baseline_total}); FSM {fsm_na} ({fsm_na_n}/{fsm_total}). Esse NA de "
        f"diálogo não removeu par de cenário. n inferencial pareado de "
        f"claim_support (tests.csv n / pares completos) = {test_n} / {complete}. "
        "A coluna «Sem par completo» copia claim_support_pairs_excluded_either_na, "
        f"residual = cenários distintos no quadro - pares completos (= {excluded}). "
        "Neste congelamento esse residual coincide com o não pareado estrutural; "
        "não somar os dois valores. Não interpretar o n descritivo de cada "
        "agente como n do teste pareado. claim_support_na_pct no CSV-fonte é "
        "proporção em [0, 1]; nesta tabela, percentual. Composição ABNT: título "
        "acima, fonte e notas abaixo; sem linhas verticais.\n"
    )


def _table_02_legend(
    descriptive: list[dict[str, str]], *, population: str = SEMANTIC_PRIMARY
) -> str:
    baseline_n, fsm_n = _category_n(
        descriptive, stratum="adversarial", population=population
    )
    extra = ""
    if baseline_n is not None and fsm_n is not None:
        extra = (
            f" Na categoria adversarial, n descritivo baseline = {baseline_n}, "
            f"FSM = {fsm_n}."
        )
    source = SOURCE_ROOTS.get(population, f"results/{population}")
    return (
        "Título: Tabela 2 — Estatísticas descritivas das métricas primárias por "
        "categoria (exploratório).\n"
        f"Fonte: elaboração própria, a partir de {source}/descriptive.csv.\n"
        "Notas: categorias caminho feliz, borda e adversarial. Análise por "
        f"categoria é exploratória.{extra} Composição ABNT: título acima, fonte "
        "e notas abaixo; sem linhas verticais.\n"
    )


def _table_03_legend(
    tests: list[dict[str, str]], *, population: str = SEMANTIC_PRIMARY
) -> str:
    claim_test = _claim_support_test(tests, population=population)
    extra = ""
    if claim_test is not None:
        extra = (
            f" Para suporte das afirmações, n = {_format_int(claim_test['n'])} "
            f"(pares completos). Não pareados = "
            f"{_format_int(claim_test['n_dropped_unpaired'])} "
            f"({UNPAIRED_SCENARIO}, estrutural). "
            "claim_support_pairs_excluded_either_na = "
            f"{_format_int(claim_test['claim_support_pairs_excluded_either_na'])} "
            "é o residual de cenários sem par completo, não uma segunda lista de "
            "pares com NA. NA de diálogo está na Tabela 1. "
            "Vitórias/empates/derrotas copiados de tests.csv; "
            "W+T+L = n da métrica."
        )
        if population == HUMAN_PRIMARY:
            holm_value = float(claim_test["wilcoxon_p_holm"])
            if holm_value < 0.05:
                holm = format_p_value(holm_value)
                low = _format_optional(claim_test["ci_low"])
                high = _format_optional(claim_test["ci_high"])
                extra += (
                    " Único p Holm < 0,05 desta família: suporte das afirmações "
                    f"(p Holm = {holm}); IC 95% da diferença [{low}; {high}] "
                    "exclui zero por margem estreita."
                )
    label = population
    source = SOURCE_ROOTS.get(population, f"results/{population}")
    return (
        "Título: Tabela 3 — Testes pareados (Wilcoxon, Holm, permutação) e "
        "vitórias/empates/derrotas das métricas primárias.\n"
        f"Fonte: elaboração própria, a partir de {source}/tests.csv.\n"
        f"Notas: população {label}, estrato overall. Holm só na família "
        "primária. Rank-biserial de Kerby; diferença = FSM - baseline."
        f"{extra} Composição ABNT: título acima, fonte e notas abaixo; sem "
        "linhas verticais.\n"
    )


def _category_n(
    descriptive: list[dict[str, str]],
    *,
    stratum: str,
    population: str = SEMANTIC_PRIMARY,
) -> tuple[str | None, str | None]:
    counts: dict[str, str] = {}
    for metric in PRIMARY_METRICS:
        for agent in AGENTS:
            row = _descriptive_row(
                descriptive,
                metric=metric,
                stratum=stratum,
                agent=agent,
                population=population,
            )
            if row is not None and row["n_scenarios"].strip():
                counts[agent] = _format_int(row["n_scenarios"])
        if "baseline" in counts and "fsm" in counts:
            return counts["baseline"], counts["fsm"]
        counts = {}
    return None, None


def _descriptive_row(
    descriptive: list[dict[str, str]],
    *,
    metric: str,
    stratum: str,
    agent: str,
    population: str = SEMANTIC_PRIMARY,
) -> dict[str, str] | None:
    for row in descriptive:
        if (
            row["population"] == population
            and row["metric"] == metric
            and row["stratum"] == stratum
            and row["agent"] == agent
        ):
            return row
    return None


def scenario_plot_values(
    rows: list[dict[str, str]], *, metric: str
) -> dict[str, list[float]]:
    """Collapse repetitions to the T-19 scenario score for Figura 01 only."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        parsed = _parse_float(row.get(metric, ""))
        if parsed is None:
            continue
        grouped[(row["scenario_id"], row["agent"])].append(parsed)
    by_agent: dict[str, list[float]] = {agent: [] for agent in AGENTS}
    for (_scenario_id, agent), values in sorted(grouped.items()):
        if agent in by_agent:
            by_agent[agent].append(mean(values))
    return by_agent


def _pyplot() -> object:
    """Import pyplot on a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _configure_matplotlib(plt: object) -> None:
    """Request Arial; a missing OS font must not fail the gate."""
    environ.setdefault("SOURCE_DATE_EPOCH", "0")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _write_figures(
    metrics: list[dict[str, str]],
    descriptive: list[dict[str, str]],
    tests: list[dict[str, str]],
    figures_dir: Path,
    *,
    population: str = SEMANTIC_PRIMARY,
) -> None:
    plt = _pyplot()
    _configure_matplotlib(plt)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(build_boxplot(metrics, plt), figures_dir / FIGURE_01_STEM, plt)
    atomic_write(
        figures_dir / f"{FIGURE_01_STEM}.legenda.txt",
        _figure_legend(
            number=1,
            title=(
                "Distribuição dos escores por cenário das métricas primárias, "
                "por agente"
            ),
            notes=_figura_01_notes(metrics),
        ),
    )
    _save_figure(
        build_category_bars(descriptive, plt, population=population),
        figures_dir / FIGURE_02_STEM,
        plt,
    )
    atomic_write(
        figures_dir / f"{FIGURE_02_STEM}.legenda.txt",
        _figure_legend(
            number=2,
            title="Médias por categoria e agente das métricas primárias",
            notes=_figura_02_notes(descriptive, population=population),
        ),
    )
    _save_figure(
        build_wtl_bars(tests, plt, population=population),
        figures_dir / FIGURE_03_STEM,
        plt,
    )
    atomic_write(
        figures_dir / f"{FIGURE_03_STEM}.legenda.txt",
        _figure_legend(
            number=3,
            title="Vitórias, empates e derrotas por métrica primária",
            notes=_figura_03_notes(),
        ),
    )


def _figura_01_notes(metrics: list[dict[str, str]]) -> str:
    plotted = {
        metric: {
            agent: len(scenario_plot_values(metrics, metric=metric)[agent])
            for agent in AGENTS
        }
        for metric in PRIMARY_METRICS
    }
    counts = [
        tuple(plotted[metric][agent] for agent in AGENTS) for metric in PRIMARY_METRICS
    ]
    if len(set(counts)) == 1:
        baseline_n, fsm_n = counts[0]
        n_clause = (
            "n plotado por agente (para as três métricas): "
            f"Baseline = {baseline_n}, FSM = {fsm_n}."
        )
    else:
        parts = [
            (
                f"{METRIC_LABELS[metric]} Baseline = {plotted[metric]['baseline']}, "
                f"FSM = {plotted[metric]['fsm']}"
            )
            for metric in PRIMARY_METRICS
        ]
        n_clause = "n plotado por agente: " + "; ".join(parts) + "."
    return (
        "Cada ponto da caixa é um cenário (média das repetições não-NA, "
        "definição de escore de cenário do T-19). "
        f"{n_clause} {UNPAIRED_SCENARIO} existe só no FSM. Figura 01 é "
        "descritiva/visual; o n inferencial pareado vem de tests.csv (não "
        "forçar o mesmo n nas duas caixas). Transformada só de visualização; "
        "médias publicadas vêm de descriptive.csv."
    )


def _figura_02_notes(
    descriptive: list[dict[str, str]], *, population: str = SEMANTIC_PRIMARY
) -> str:
    baseline_n, fsm_n = _category_n(
        descriptive, stratum="adversarial", population=population
    )
    extra = ""
    if baseline_n is not None and fsm_n is not None:
        extra = (
            f" Na categoria adversarial, n descritivo baseline = {baseline_n}, "
            f"FSM = {fsm_n}."
        )
    label = POPULATION_LABELS.get(population, population.replace("_", " "))
    return (
        f"Alturas = médias congeladas de descriptive.csv (população {label}). "
        "Barras de erro = IC 95% congelados das colunas ci_low e "
        "ci_high; T-20 não recalcula o IC. Análise por categoria exploratória."
        f"{extra}"
    )


def _figura_03_notes() -> str:
    return (
        "Contagens inteiras copiadas de tests.csv (wins, ties, losses), "
        "rotuladas em cada segmento. W+T+L = n de pares completos da métrica, "
        "não necessariamente o n descritivo de cada agente."
    )


def _figure_legend(*, number: int, title: str, notes: str) -> str:
    return (
        f"Título: Figura {number} — {title}.\n"
        "Fonte: elaboração própria.\n"
        f"Notas: {notes} Legenda abaixo da figura (ABNT); a imagem não traz "
        f"título embutido. {COLOR_HATCH_NOTE}\n"
    )


def _save_figure(fig: object, stem: Path, plt: object) -> None:
    fig.set_size_inches(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN)
    fig.savefig(f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_boxplot(metrics: list[dict[str, str]], plt: object | None = None) -> object:
    """Boxplots of scenario-level scores from metrics.csv (Figura 01)."""
    if plt is None:
        plt = _pyplot()
    fig, axes = plt.subplots(
        1, len(PRIMARY_METRICS), sharey=False, layout="constrained"
    )
    for axis, metric in zip(axes, PRIMARY_METRICS, strict=True):
        values = scenario_plot_values(metrics, metric=metric)
        data = [values[agent] for agent in AGENTS]
        boxes = axis.boxplot(
            data,
            tick_labels=[AGENT_LABELS[agent] for agent in AGENTS],
            patch_artist=True,
            widths=0.6,
        )
        for patch, agent in zip(boxes["boxes"], AGENTS, strict=True):
            patch.set_facecolor(AGENT_FACECOLORS[agent])
            patch.set_hatch(AGENT_HATCHES[agent])
            patch.set_edgecolor("black")
        for key in ("whiskers", "caps", "medians"):
            for line in boxes[key]:
                line.set_color("black")
        axis.set_xlabel(METRIC_LABELS[metric])
        axis.set_ylim(*SCORE_YLIM)
        axis.set_title("")
        axis.grid(False)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.suptitle("")
    return fig


def build_category_bars(
    descriptive: list[dict[str, str]],
    plt: object | None = None,
    *,
    population: str = SEMANTIC_PRIMARY,
) -> object:
    """Grouped bars from frozen descriptive.csv means (Figura 02)."""
    if plt is None:
        plt = _pyplot()
    fig, axes = plt.subplots(
        1, len(PRIMARY_METRICS), sharey=False, layout="constrained"
    )
    width = 0.35
    x_positions = list(range(len(CATEGORIES)))
    for axis, metric in zip(axes, PRIMARY_METRICS, strict=True):
        for offset, agent in zip((-width / 2, width / 2), AGENTS, strict=True):
            heights: list[float] = []
            lowers: list[float] = []
            uppers: list[float] = []
            for category in CATEGORIES:
                mean_value, ci_low, ci_high = _frozen_bar(
                    descriptive,
                    metric=metric,
                    stratum=category,
                    agent=agent,
                    population=population,
                )
                heights.append(mean_value)
                lowers.append(ci_low)
                uppers.append(ci_high)
            axis.bar(
                [position + offset for position in x_positions],
                heights,
                width=width,
                label=AGENT_LABELS[agent],
                facecolor=AGENT_FACECOLORS[agent],
                edgecolor="black",
                hatch=AGENT_HATCHES[agent],
                yerr=(lowers, uppers),
                error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 0.8},
            )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [CATEGORY_LABELS[category] for category in CATEGORIES],
            rotation=30,
            ha="right",
        )
        axis.set_xlabel(METRIC_LABELS[metric])
        axis.set_ylim(*SCORE_YLIM)
        axis.set_title("")
        axis.grid(False)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.suptitle("")
    return fig


def _mean_for(
    descriptive: list[dict[str, str]],
    *,
    metric: str,
    stratum: str,
    agent: str,
) -> float:
    mean_value, _low, _high = _frozen_bar(
        descriptive, metric=metric, stratum=stratum, agent=agent
    )
    return mean_value


def _frozen_bar(
    descriptive: list[dict[str, str]],
    *,
    metric: str,
    stratum: str,
    agent: str,
    population: str = SEMANTIC_PRIMARY,
) -> tuple[float, float, float]:
    row = _descriptive_row(
        descriptive,
        metric=metric,
        stratum=stratum,
        agent=agent,
        population=population,
    )
    if row is None:
        return (0.0, 0.0, 0.0)
    mean_value = _parse_float(row["mean"]) or 0.0
    ci_low = _parse_float(row["ci_low"])
    ci_high = _parse_float(row["ci_high"])
    if ci_low is None or ci_high is None:
        return (mean_value, 0.0, 0.0)
    low, high = frozen_ci_yerr(mean_value, ci_low, ci_high)
    return (mean_value, low, high)


def build_wtl_bars(
    tests: list[dict[str, str]],
    plt: object | None = None,
    *,
    population: str = SEMANTIC_PRIMARY,
) -> object:
    """Stacked wins/ties/losses copied from tests.csv (Figura 03)."""
    if plt is None:
        plt = _pyplot()
    fig, axis = plt.subplots(layout="constrained")
    wins: list[int] = []
    ties: list[int] = []
    losses: list[int] = []
    by_metric = {
        row["metric"]: row for row in _primary_tests(tests, population=population)
    }
    for metric in PRIMARY_METRICS:
        row = by_metric[metric]
        wins.append(int(float(row["wins"])))
        ties.append(int(float(row["ties"])))
        losses.append(int(float(row["losses"])))
    x_positions = list(range(len(PRIMARY_METRICS)))
    axis.bar(
        x_positions,
        wins,
        color=WTL_FACECOLORS[0],
        edgecolor="black",
        hatch=WTL_HATCHES[0],
        label="Vitórias (FSM)",
    )
    axis.bar(
        x_positions,
        ties,
        bottom=wins,
        color=WTL_FACECOLORS[1],
        edgecolor="black",
        hatch=WTL_HATCHES[1],
        label="Empates",
    )
    axis.bar(
        x_positions,
        losses,
        bottom=[win + tie for win, tie in zip(wins, ties, strict=True)],
        color=WTL_FACECOLORS[2],
        edgecolor="black",
        hatch=WTL_HATCHES[2],
        label="Derrotas (FSM)",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels([METRIC_TICK_LABELS[metric] for metric in PRIMARY_METRICS])
    axis.set_title("")
    axis.grid(False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    _label_stacked_counts(axis)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.suptitle("")
    return fig


def _label_stacked_counts(axis: object) -> None:
    """Write frozen integer counts on stacked bars; skip zero segments."""
    box = {"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.85}
    for container in axis.containers:
        for bar in container:
            height = bar.get_height()
            if height <= 0:
                continue
            label = str(int(height))
            x_position = bar.get_x() + bar.get_width() / 2
            if height >= WTL_INTERNAL_LABEL_MIN:
                axis.text(
                    x_position,
                    bar.get_y() + height / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    bbox=box,
                )
            else:
                axis.text(
                    x_position,
                    bar.get_y() + height,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    bbox=box,
                )


def write_comparison_artifacts(
    results_dir: Path,
    *,
    agreement_path: Path | None = None,
    conclusions_path: Path | None = None,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> None:
    """Write human vs judge agreement plates from frozen sidecar outputs."""
    root = results_dir
    agreement_file = (
        agreement_path if agreement_path is not None else root / "agreement.json"
    )
    conclusions_file = (
        conclusions_path if conclusions_path is not None else root / "conclusions.csv"
    )
    out_tables = tables_dir if tables_dir is not None else root / "tables"
    out_figures = figures_dir if figures_dir is not None else root / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    agreement = json.loads(agreement_file.read_text(encoding="utf-8"))
    _write_table_04(agreement, out_tables)
    if conclusions_file.is_file():
        _write_table_05(_read_csv(conclusions_file), out_tables)
    try:
        _write_figure_04(agreement, out_figures)
    except ImportError:
        return


def _write_table_04(agreement: dict[str, object], tables_dir: Path) -> None:
    task = _as_mapping(agreement["task_completed"])
    claims = _as_mapping(agreement["claim_support_label"])
    facts = _as_mapping(agreement["fact_ids_stated"])
    omitted = _format_int(str(facts["n_omitted_ambiguous_judge_calls"]))
    rows = [
        {
            "Comparação": "Tarefa concluída",
            "n": _format_int(str(task["n"])),
            "Acordo %": _agreement_percent(float(task["percent_agreement"])),
            "Cohen kappa": format_pt_br_number(float(task["cohen_kappa"])),
            "Micro F1": "—",
            "Observação": (
                f"{_format_int(str(task['n_uncertain_excluded']))} incerto excluído"
            ),
        },
        {
            "Comparação": "Suporte (rótulo de diálogo)",
            "n": _format_int(str(claims["n"])),
            "Acordo %": _agreement_percent(float(claims["percent_agreement"])),
            "Cohen kappa": format_pt_br_number(float(claims["cohen_kappa"])),
            "Micro F1": "—",
            "Observação": "sem casamento atômico de afirmações",
        },
        {
            "Comparação": "IDs de fato (conjunto exato)",
            "n": _format_int(str(facts["n"])),
            "Acordo %": _agreement_percent(float(facts["exact_match_rate"])),
            "Cohen kappa": "—",
            "Micro F1": "—",
            "Observação": f"{omitted} omitidos (chamadas ambíguas do juiz)",
        },
        {
            "Comparação": "IDs de fato (micro)",
            "n": _format_int(str(facts["n"])),
            "Acordo %": "—",
            "Cohen kappa": "—",
            "Micro F1": format_pt_br_number(float(facts["f1"])),
            "Observação": (
                f"TP {_format_int(str(facts['true_positive']))}; "
                f"FP {_format_int(str(facts['false_positive']))}; "
                f"FN {_format_int(str(facts['false_negative']))}"
            ),
        },
    ]
    path = tables_dir / TABLE_04_NAME
    _write_csv(path, TABLE_04_FIELDS, rows)
    atomic_write(path.with_suffix(".legenda.txt"), _table_04_legend(agreement))


def _write_table_05(conclusions: list[dict[str, str]], tables_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    order = {metric: index for index, metric in enumerate(PRIMARY_METRICS)}
    selected = [row for row in conclusions if row["metric"] in PRIMARY_METRICS]
    selected.sort(key=lambda row: order[row["metric"]])
    for source in selected:
        rows.append(
            {
                "Métrica": METRIC_LABELS[source["metric"]],
                "Direção juiz": DIRECTION_LABELS.get(
                    source["judge_direction"], source["judge_direction"]
                ),
                "Direção humano": DIRECTION_LABELS.get(
                    source["human_direction"], source["human_direction"]
                ),
                "Diferença juiz": _format_optional(source["judge_mean_diff"]),
                "Diferença humano": _format_optional(source["human_mean_diff"]),
                "p Holm juiz": format_p_value(float(source["judge_wilcoxon_p_holm"])),
                "p Holm humano": format_p_value(float(source["human_wilcoxon_p_holm"])),
                "Mesma direção": _yes_no(source["same_direction"]),
                "Mesma significância": _yes_no(source["same_significance"]),
            }
        )
    path = tables_dir / TABLE_05_NAME
    _write_csv(path, TABLE_05_FIELDS, rows)
    atomic_write(path.with_suffix(".legenda.txt"), _table_05_legend())


def _write_figure_04(agreement: dict[str, object], figures_dir: Path) -> None:
    plt = _pyplot()
    _configure_matplotlib(plt)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(
        build_agreement_bars(agreement, plt), figures_dir / FIGURE_04_STEM, plt
    )
    atomic_write(
        figures_dir / f"{FIGURE_04_STEM}.legenda.txt",
        _figure_legend(
            number=4,
            title="Concordância humano vs juiz por comparação",
            notes=_figura_04_notes(agreement),
        ),
    )


def build_agreement_bars(
    agreement: dict[str, object], plt: object | None = None
) -> object:
    """Percent-agreement bars for the three human vs judge comparisons."""
    if plt is None:
        plt = _pyplot()
    task = _as_mapping(agreement["task_completed"])
    claims = _as_mapping(agreement["claim_support_label"])
    facts = _as_mapping(agreement["fact_ids_stated"])
    labels = (
        "Tarefa\nconcluída",
        "Suporte\n(rótulo 3 vias)",
        "IDs de fato\n(conjunto exato)",
    )
    heights = (
        float(task["percent_agreement"]) * 100,
        float(claims["percent_agreement"]) * 100,
        float(facts["exact_match_rate"]) * 100,
    )
    counts = (int(task["n"]), int(claims["n"]), int(facts["n"]))
    fig, axis = plt.subplots(layout="constrained")
    x_positions = list(range(len(labels)))
    bars = axis.bar(
        x_positions,
        heights,
        facecolor=AGENT_FACECOLORS["baseline"],
        edgecolor="black",
        hatch=AGENT_HATCHES["baseline"],
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Acordo (%)")
    axis.set_ylim(0, 105)
    axis.set_title("")
    axis.grid(False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for bar, count in zip(bars, counts, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"n = {count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.suptitle("")
    return fig


def _table_04_legend(agreement: dict[str, object]) -> str:
    facts = _as_mapping(agreement["fact_ids_stated"])
    omitted = _format_int(str(facts["n_omitted_ambiguous_judge_calls"]))
    kept = _format_int(str(facts["n"]))
    return (
        "Título: Tabela 4 — Concordância humano vs juiz (grão diálogo).\n"
        "Fonte: elaboração própria, a partir de "
        "results/human_primary/agreement.json.\n"
        "Notas: humano como referência. Tarefa concluída exclui incerto. "
        "Suporte é rótulo de diálogo em três vias derivado de "
        "n_checkable_claims e claim_support, sem casamento de claim_id nem "
        "de texto. IDs de fato: conjunto exato e micro P/R/F1 sobre os "
        f"{kept} diálogos cuja reconstrução do juiz foi unívoca; {omitted} "
        "diálogos omitidos porque duas chamadas judge_facts reproduzem a "
        "linha congelada de metrics.csv e discordam nos IDs previstos "
        "(GGUF vs MLX). Não adivinhar o conjunto. Composição ABNT: título "
        "acima, fonte e notas abaixo; sem linhas verticais.\n"
    )


def _table_05_legend() -> str:
    return (
        "Título: Tabela 5 — Direção e significância Holm, juiz vs humano.\n"
        "Fonte: elaboração própria, a partir de "
        "results/human_primary/conclusions.csv.\n"
        "Notas: diferença = FSM - baseline. Significância = p Holm < 0,05 "
        "na família primária (três métricas). A escolha do humano como "
        "referência narrativa da conclusão é posterior ao congelamento e "
        "ao desvelamento. Composição ABNT: título acima, fonte e notas "
        "abaixo; sem linhas verticais.\n"
    )


def _figura_04_notes(agreement: dict[str, object]) -> str:
    facts = _as_mapping(agreement["fact_ids_stated"])
    omitted = _format_int(str(facts["n_omitted_ambiguous_judge_calls"]))
    kept = _format_int(str(facts["n"]))
    return (
        "Alturas = acordo percentual copiado de agreement.json. n anotado "
        "em cada barra. IDs de fato usam a taxa de conjunto exato nos "
        f"{kept} diálogos reconstruídos de forma unívoca; {omitted} "
        "diálogos ficaram de fora dessa comparação, não do recenseamento "
        "humano. Kappa e micro F1 estão na Tabela 4, não nesta figura."
    )


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("agreement field must be an object")
    return value


def _agreement_percent(value: float) -> str:
    return f"{format_pt_br_number(value * 100, decimals=1)}%"


def _yes_no(value: str) -> str:
    return "Sim" if value.strip().lower() in {"true", "1"} else "Não"
