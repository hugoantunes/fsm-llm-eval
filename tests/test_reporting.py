"""Thesis tables and figures from frozen T-19 CSVs (T-20)."""

import json
import os
import subprocess
import sys
from csv import DictReader, DictWriter
from pathlib import Path

import pytest

from helpers import REPO_ROOT
from sim.reporting import (
    AGENT_FACECOLORS,
    AGENT_HATCHES,
    FIGURE_01_STEM,
    FIGURE_02_STEM,
    FIGURE_03_STEM,
    FIGURE_04_STEM,
    FIGURE_05_STEM,
    FIGURE_HEIGHT_IN,
    FIGURE_WIDTH_IN,
    HUMAN_PRIMARY,
    INSTRUMENT_FACECOLORS,
    INSTRUMENT_MARKERS,
    PNG_DPI,
    TABLE_01_NAME,
    TABLE_03_NAME,
    TABLE_04_NAME,
    TABLE_05_NAME,
    WTL_FACECOLORS,
    WTL_HATCHES,
    _configure_matplotlib,
    _figura_01_notes,
    _pyplot,
    build_boxplot,
    build_category_bars,
    build_paired_forest,
    build_wtl_bars,
    format_as_percent,
    format_pt_br_number,
    frozen_ci_yerr,
    paired_interval_rows,
    scenario_plot_values,
    write_comparison_artifacts,
    write_thesis_artifacts,
)

DESCRIPTIVE_FIELDS = (
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
TESTS_FIELDS = (
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
METRICS_FIELDS = (
    "scenario_id",
    "agent",
    "repetition",
    "task_completed",
    "fact_f1",
    "claim_support",
)


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _empty_descriptive(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(DESCRIPTIVE_FIELDS, "")
    row.update(
        population="semantic_primary",
        n_dialogues=3,
        sd=0.1,
        median=0.5,
        mean_intra_scenario_sd=0.05,
    )
    row.update(overrides)
    return row


def _empty_test(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(TESTS_FIELDS, "")
    row.update(
        population="semantic_primary",
        stratum="overall",
        family="primary",
        n_dropped_unpaired=1,
        direction="baseline",
    )
    row.update(overrides)
    return row


def _frozen_triple(tmp_path: Path) -> Path:
    """Write a synthetic T-19 triple whose published cells disagree with raw metrics."""
    results = tmp_path / "exp"
    results.mkdir()
    _write_csv(
        results / "metrics.csv",
        METRICS_FIELDS,
        [
            {
                "scenario_id": "happy_path_01",
                "agent": "baseline",
                "repetition": 1,
                "task_completed": 1.0,
                "fact_f1": 1.0,
                "claim_support": 1.0,
            },
            {
                "scenario_id": "happy_path_01",
                "agent": "fsm",
                "repetition": 1,
                "task_completed": 0.0,
                "fact_f1": 0.0,
                "claim_support": 0.0,
            },
            {
                "scenario_id": "edge_01",
                "agent": "baseline",
                "repetition": 1,
                "task_completed": 1.0,
                "fact_f1": 1.0,
                "claim_support": 1.0,
            },
            {
                "scenario_id": "edge_01",
                "agent": "fsm",
                "repetition": 1,
                "task_completed": 0.0,
                "fact_f1": 0.0,
                "claim_support": "",
            },
            {
                "scenario_id": "adversarial_01",
                "agent": "baseline",
                "repetition": 1,
                "task_completed": 1.0,
                "fact_f1": 1.0,
                "claim_support": 1.0,
            },
            {
                "scenario_id": "adversarial_01",
                "agent": "fsm",
                "repetition": 1,
                "task_completed": 0.0,
                "fact_f1": 0.0,
                "claim_support": 0.0,
            },
            {
                "scenario_id": "adversarial_12",
                "agent": "fsm",
                "repetition": 1,
                "task_completed": 1.0,
                "fact_f1": 1.0,
                "claim_support": 1.0,
            },
        ],
    )
    _write_csv(
        results / "descriptive.csv",
        DESCRIPTIVE_FIELDS,
        [
            _empty_descriptive(
                stratum="overall",
                metric="task_completed",
                agent="baseline",
                n_scenarios=3,
                mean=0.76,
                ci_low=0.66,
                ci_high=0.85,
            ),
            _empty_descriptive(
                stratum="overall",
                metric="task_completed",
                agent="fsm",
                n_scenarios=4,
                mean=0.75,
                ci_low=0.66,
                ci_high=0.84,
            ),
            _empty_descriptive(
                stratum="overall",
                metric="fact_f1",
                agent="baseline",
                n_scenarios=3,
                mean=0.49,
                ci_low=0.44,
                ci_high=0.53,
            ),
            _empty_descriptive(
                stratum="overall",
                metric="fact_f1",
                agent="fsm",
                n_scenarios=4,
                mean=0.44,
                ci_low=0.38,
                ci_high=0.49,
            ),
            _empty_descriptive(
                stratum="overall",
                metric="claim_support",
                agent="baseline",
                n_scenarios=3,
                mean=0.94,
                ci_low=0.91,
                ci_high=0.97,
                claim_support_total_dialogues=3,
                claim_support_na_dialogues=0,
                claim_support_na_pct=0.0,
                claim_support_aggregatable_dialogues=3,
                claim_support_complete_pairs=2,
                claim_support_pairs_excluded_either_na=1,
            ),
            _empty_descriptive(
                stratum="overall",
                metric="claim_support",
                agent="fsm",
                n_scenarios=4,
                mean=0.95,
                ci_low=0.93,
                ci_high=0.97,
                claim_support_total_dialogues=4,
                claim_support_na_dialogues=1,
                claim_support_na_pct=0.005714285714285714,
                claim_support_aggregatable_dialogues=3,
                claim_support_complete_pairs=2,
                claim_support_pairs_excluded_either_na=1,
            ),
            _empty_descriptive(
                stratum="happy_path",
                metric="fact_f1",
                agent="baseline",
                n_scenarios=1,
                mean=0.57,
                ci_low=0.53,
                ci_high=0.63,
            ),
            _empty_descriptive(
                stratum="happy_path",
                metric="fact_f1",
                agent="fsm",
                n_scenarios=1,
                mean=0.53,
                ci_low=0.48,
                ci_high=0.59,
            ),
            _empty_descriptive(
                stratum="edge",
                metric="fact_f1",
                agent="baseline",
                n_scenarios=1,
                mean=0.49,
                ci_low=0.42,
                ci_high=0.55,
            ),
            _empty_descriptive(
                stratum="edge",
                metric="fact_f1",
                agent="fsm",
                n_scenarios=1,
                mean=0.38,
                ci_low=0.29,
                ci_high=0.46,
            ),
            _empty_descriptive(
                stratum="adversarial",
                metric="fact_f1",
                agent="baseline",
                n_scenarios=19,
                mean=0.40,
                ci_low=0.30,
                ci_high=0.49,
            ),
            _empty_descriptive(
                stratum="adversarial",
                metric="fact_f1",
                agent="fsm",
                n_scenarios=20,
                mean=0.41,
                ci_low=0.30,
                ci_high=0.53,
            ),
        ],
    )
    _write_csv(
        results / "tests.csv",
        TESTS_FIELDS,
        [
            _empty_test(
                metric="task_completed",
                n=3,
                n_nonzero=2,
                wilcoxon_p=0.717,
                wilcoxon_p_holm=1.0,
                permutation_p=0.765,
                rank_biserial=-0.09,
                mean_diff=-0.011,
                ci_low=-0.085,
                ci_high=0.065,
                wins=1,
                ties=1,
                losses=1,
            ),
            _empty_test(
                metric="fact_f1",
                n=3,
                n_nonzero=3,
                wilcoxon_p=0.0044,
                wilcoxon_p_holm=0.0132,
                permutation_p=0.0095,
                rank_biserial=-0.45,
                mean_diff=-0.059,
                ci_low=-0.103,
                ci_high=-0.017,
                wins=1,
                ties=0,
                losses=2,
            ),
            _empty_test(
                metric="claim_support",
                n=2,
                n_nonzero=2,
                wilcoxon_p=0.975,
                wilcoxon_p_holm=1.0,
                permutation_p=0.762,
                rank_biserial=0.006,
                mean_diff=0.004,
                ci_low=-0.021,
                ci_high=0.031,
                wins=1,
                ties=0,
                losses=1,
                claim_support_complete_pairs=2,
                claim_support_pairs_excluded_either_na=1,
            ),
            _empty_test(
                population="frozen_gate",
                metric="fact_f1",
                family="secondary",
                n=3,
                n_nonzero=3,
                wilcoxon_p=0.002,
                permutation_p=0.005,
                rank_biserial=-0.5,
                mean_diff=-0.06,
                ci_low=-0.1,
                ci_high=-0.02,
                wins=0,
                ties=0,
                losses=3,
            ),
        ],
    )
    return results


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(DictReader(handle))


@pytest.fixture(scope="module")
def frozen_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _frozen_triple(tmp_path_factory.mktemp("frozen"))


@pytest.fixture(scope="module")
def thesis_artifacts(frozen_results: Path) -> Path:
    """The frozen triple with its thesis tables and figures written once."""
    write_thesis_artifacts(frozen_results)
    return frozen_results


@pytest.fixture
def metrics_rows(frozen_results: Path) -> list[dict[str, str]]:
    return _read_table(frozen_results / "metrics.csv")


@pytest.fixture
def descriptive_rows(frozen_results: Path) -> list[dict[str, str]]:
    return _read_table(frozen_results / "descriptive.csv")


@pytest.fixture
def tests_rows(frozen_results: Path) -> list[dict[str, str]]:
    return _read_table(frozen_results / "tests.csv")


def test_claim_support_na_pct_formats_as_percent() -> None:
    assert format_as_percent(0.005714285714285714) == "0,5714%"


def test_table01_reads_frozen_descriptive_and_claim_support_n_without_recompute(
    thesis_artifacts: Path,
) -> None:
    tables = thesis_artifacts / "tables"

    rows = _read_table(tables / "resultados_tabela-01_descritiva-geral.csv")
    legend = (tables / "resultados_tabela-01_descritiva-geral.legenda.txt").read_text(
        encoding="utf-8"
    )

    by_key = {(row["Métrica"], row["Agente"]): row for row in rows}
    fact_f1_baseline = by_key[("F1 dos fatos esperados", "Baseline")]
    assert fact_f1_baseline["Média"] == format_pt_br_number(0.49)
    assert fact_f1_baseline["n cenários"] == "3"
    assert "1,000" not in fact_f1_baseline["Média"]
    support_fsm = by_key[("Suporte das afirmações", "FSM")]
    assert support_fsm["NA (%)"] == format_as_percent(0.005714285714285714)
    assert support_fsm["Pares completos"] == "2"
    assert support_fsm["Sem par completo"] == "1"
    assert "adversarial_12" in legend
    assert "n descritivo baseline = 3" in legend
    assert "n descritivo FSM = 4" in legend
    assert "diálogo" in legend
    assert "não somar" in legend
    assert "pares excluídos porque um lado é NA = 1" not in legend


def test_table02_reads_frozen_category_descriptive_without_recompute(
    thesis_artifacts: Path,
) -> None:
    rows = _read_table(
        thesis_artifacts / "tables" / "resultados_tabela-02_descritiva-categoria.csv"
    )

    categories = {row["Categoria"] for row in rows}
    assert categories == {"Caminho feliz", "Borda", "Adversarial"}
    by_key = {(row["Métrica"], row["Categoria"], row["Agente"]): row for row in rows}
    edge_fsm = by_key[("F1 dos fatos esperados", "Borda", "FSM")]
    assert edge_fsm["Média"] == format_pt_br_number(0.38)


def test_table03_reads_frozen_tests_including_holm_and_names_unpaired_residual(
    thesis_artifacts: Path,
) -> None:
    tables = thesis_artifacts / "tables"

    rows = _read_table(tables / "resultados_tabela-03_testes-pareados.csv")
    legend = (tables / "resultados_tabela-03_testes-pareados.legenda.txt").read_text(
        encoding="utf-8"
    )

    by_metric = {row["Métrica"]: row for row in rows}
    fact_f1 = by_metric["F1 dos fatos esperados"]
    assert fact_f1["p Wilcoxon"] == format_pt_br_number(0.0044)
    assert fact_f1["p Holm"] == format_pt_br_number(0.0132)
    assert fact_f1["Vitórias"] == "1"
    assert fact_f1["Empates"] == "0"
    assert fact_f1["Derrotas"] == "2"
    assert "frozen_gate" not in " ".join(row["Métrica"] for row in rows)
    assert "Não pareados" in legend
    assert "residual" in legend
    assert "pares excluídos por NA = 1" not in legend


def test_reporting_has_no_analysis_calculator_dependency(tmp_path: Path) -> None:
    results = _frozen_triple(tmp_path)
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from sim.reporting import write_thesis_artifacts\n"
        f"write_thesis_artifacts(Path({str(results)!r}))\n"
        "assert 'sim.analysis' not in sys.modules\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert completed.returncode == 0, completed.stderr
    assert (results / "tables" / "resultados_tabela-01_descritiva-geral.csv").is_file()
    assert (results / "tables" / "resultados_tabela-03_testes-pareados.csv").is_file()


def test_artifacts_are_named_by_thesis_section(thesis_artifacts: Path) -> None:
    pytest.importorskip("matplotlib")

    tables = {path.name for path in (thesis_artifacts / "tables").iterdir()}
    figures = {path.name for path in (thesis_artifacts / "figures").iterdir()}

    assert "resultados_tabela-01_descritiva-geral.csv" in tables
    assert "resultados_tabela-02_descritiva-categoria.csv" in tables
    assert "resultados_tabela-03_testes-pareados.csv" in tables
    assert any(name.startswith("resultados_figura-01_") for name in figures)
    assert any(name.startswith("resultados_figura-02_") for name in figures)
    assert any(name.startswith("resultados_figura-03_") for name in figures)


def test_figures_have_no_embedded_title(
    metrics_rows: list[dict[str, str]],
    descriptive_rows: list[dict[str, str]],
    tests_rows: list[dict[str, str]],
) -> None:
    pytest.importorskip("matplotlib")

    for fig in (
        build_boxplot(metrics_rows),
        build_category_bars(descriptive_rows),
        build_wtl_bars(tests_rows),
    ):
        assert fig.get_suptitle() == ""
        for axis in fig.axes:
            assert axis.get_title() == ""


def test_figures_export_png_300_dpi_and_vector_pdf(thesis_artifacts: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from PIL import Image

    png = thesis_artifacts / "figures" / f"{FIGURE_01_STEM}.png"
    pdf = thesis_artifacts / "figures" / f"{FIGURE_01_STEM}.pdf"

    dpi = Image.open(png).info.get("dpi")
    payload = pdf.read_bytes()

    assert dpi is not None
    assert round(dpi[0]) == PNG_DPI
    assert payload.startswith(b"%PDF")
    assert b"endobj" in payload
    assert b"/Type /Page" in payload or b"/Type/Page" in payload


def test_figures_request_arial() -> None:
    pytest.importorskip("matplotlib")
    plt = _pyplot()

    _configure_matplotlib(plt)

    assert plt.rcParams["font.sans-serif"][0] == "Arial"


def test_boxplot_uses_scenario_scores_from_metrics_csv(
    metrics_rows: list[dict[str, str]],
) -> None:
    pytest.importorskip("matplotlib")

    values = scenario_plot_values(metrics_rows, metric="fact_f1")
    fig = build_boxplot(metrics_rows)

    assert values["baseline"] == [1.0, 1.0, 1.0]
    assert values["fsm"] == [0.0, 1.0, 0.0, 0.0]
    assert len(fig.axes) == 3
    assert len(fig.axes[1].patches) == 2


def test_wtl_figure_plots_and_labels_exact_counts_from_tests_csv(
    tests_rows: list[dict[str, str]], thesis_artifacts: Path
) -> None:
    pytest.importorskip("matplotlib")

    axis = build_wtl_bars(tests_rows).axes[0]
    notes = (thesis_artifacts / "figures" / f"{FIGURE_03_STEM}.legenda.txt").read_text(
        encoding="utf-8"
    )

    heights = [int(patch.get_height()) for patch in axis.patches]
    assert heights[:3] == [1, 1, 1]
    assert heights[3:6] == [1, 0, 0]
    assert heights[6:] == [1, 2, 1]
    labels = {text.get_text() for text in axis.texts}
    assert "1" in labels
    assert "2" in labels
    assert "0" not in labels
    assert "tests.csv" in notes
    assert "rotuladas" in notes


def _rgb(color: tuple[float, ...]) -> tuple[float, float, float]:
    return color[0], color[1], color[2]


def _is_gray(color: tuple[float, ...]) -> bool:
    red, green, blue = _rgb(color)
    return red == green == blue


def test_agent_series_use_distinct_chromatic_fills_hatches_and_black_borders(
    metrics_rows: list[dict[str, str]], descriptive_rows: list[dict[str, str]]
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgba

    boxplot = build_boxplot(metrics_rows)
    bars = build_category_bars(descriptive_rows)

    for fig in (boxplot, bars):
        fills = {_rgb(patch.get_facecolor()) for patch in fig.axes[0].patches}
        assert len(fills) == 2
        assert not any(_is_gray(fill) for fill in fills)
    bar_patches = list(bars.axes[0].patches)
    assert {patch.get_hatch() for patch in bar_patches} == set(AGENT_HATCHES.values())
    assert {_rgb(patch.get_facecolor()) for patch in bar_patches} == {
        _rgb(to_rgba(color)) for color in AGENT_FACECOLORS.values()
    }
    for patch in bar_patches:
        assert _rgb(patch.get_edgecolor()) == _rgb(to_rgba("black"))


def test_wtl_series_use_distinct_chromatic_colors_hatches_and_black_borders(
    tests_rows: list[dict[str, str]],
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgba

    patches = list(build_wtl_bars(tests_rows).axes[0].patches)

    hatches = [patch.get_hatch() for patch in patches]
    colors = [_rgb(patch.get_facecolor()) for patch in patches]
    wins, losses = colors[0], colors[6]
    assert wins != losses
    assert not _is_gray(wins)
    assert not _is_gray(losses)
    assert hatches[0] == WTL_HATCHES[0]
    assert hatches[3] == WTL_HATCHES[1]
    assert hatches[6] == WTL_HATCHES[2]
    assert colors[0] == _rgb(to_rgba(WTL_FACECOLORS[0]))
    assert colors[3] == _rgb(to_rgba(WTL_FACECOLORS[1]))
    assert colors[6] == _rgb(to_rgba(WTL_FACECOLORS[2]))
    assert len(set(hatches)) == 3
    for patch in patches:
        assert _rgb(patch.get_edgecolor()) == _rgb(to_rgba("black"))


def test_figura_01_notes_report_plotted_n_and_redundant_color_and_hatch(
    thesis_artifacts: Path,
) -> None:
    pytest.importorskip("matplotlib")

    notes = (thesis_artifacts / "figures" / f"{FIGURE_01_STEM}.legenda.txt").read_text(
        encoding="utf-8"
    )

    assert "n plotado" in notes
    assert "Baseline = 3" in notes
    assert "FSM = 4" in notes
    assert "para as três métricas" not in notes
    assert "Tarefa concluída" in notes
    assert "F1 dos fatos esperados" in notes
    assert "Suporte das afirmações" in notes
    assert "adversarial_12" in notes
    assert "tests.csv" in notes
    assert "descritiva" in notes
    assert "Escala de cinza e hachura para impressão" not in notes
    assert "escala de cinza" not in notes.lower()
    assert "codificação visual redundante" in notes
    assert "hachuras" in notes


def test_figura_01_notes_name_shared_n_when_all_metrics_match() -> None:
    rows = [
        {
            "scenario_id": scenario,
            "agent": agent,
            "task_completed": "1.0",
            "fact_f1": "1.0",
            "claim_support": "1.0",
        }
        for scenario in ("happy_path_01", "edge_01")
        for agent in ("baseline", "fsm")
    ]

    notes = _figura_01_notes(rows)

    assert "para as três métricas" in notes
    assert "Baseline = 2" in notes
    assert "FSM = 2" in notes


def test_unit_score_figures_share_ylim_0_to_1_05(
    metrics_rows: list[dict[str, str]], descriptive_rows: list[dict[str, str]]
) -> None:
    pytest.importorskip("matplotlib")

    for fig in (build_boxplot(metrics_rows), build_category_bars(descriptive_rows)):
        for axis in fig.axes:
            assert axis.get_ylim() == pytest.approx((0.0, 1.05))


def test_category_bars_use_frozen_confidence_intervals(
    descriptive_rows: list[dict[str, str]],
) -> None:
    pytest.importorskip("matplotlib")

    fig = build_category_bars(descriptive_rows)

    bars = [
        container
        for container in fig.axes[1].containers
        if hasattr(container, "datavalues")
    ]
    baseline = bars[0]
    assert round(baseline.datavalues[0], 2) == 0.57
    assert baseline.errorbar is not None
    low, high = frozen_ci_yerr(0.57, 0.53, 0.63)
    assert (low, high) == (pytest.approx(0.04), pytest.approx(0.06))
    segments = baseline.errorbar[2][0].get_segments()
    assert sorted(segments[0][:, 1]) == pytest.approx([0.53, 0.63])


def test_figura_02_notes_name_frozen_means_cis_and_asymmetric_n(
    thesis_artifacts: Path,
) -> None:
    pytest.importorskip("matplotlib")

    notes = (thesis_artifacts / "figures" / f"{FIGURE_02_STEM}.legenda.txt").read_text(
        encoding="utf-8"
    )

    assert "médias congeladas" in notes
    assert "IC 95%" in notes
    assert "não recalcula" in notes
    assert "exploratória" in notes
    assert "baseline = 19" in notes
    assert "FSM = 20" in notes


def test_real_t19_source_csvs_stay_byte_identical(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    source = REPO_ROOT / "results" / "exp_final"
    dest = tmp_path / "exp_final"
    dest.mkdir()
    names = ("metrics.csv", "descriptive.csv", "tests.csv")
    before = {}
    for name in names:
        payload = (source / name).read_bytes()
        (dest / name).write_bytes(payload)
        before[name] = payload

    write_thesis_artifacts(dest)

    for name, payload in before.items():
        assert (dest / name).read_bytes() == payload


def _primary_pair_tests(
    *,
    population: str,
    task: tuple[float, float, float],
    fact_f1: tuple[float, float, float],
    support: tuple[float, float, float],
) -> list[dict[str, object]]:
    rows = []
    for metric, values, n in (
        ("task_completed", task, 59),
        ("fact_f1", fact_f1, 59),
        ("claim_support", support, 59),
    ):
        mean_diff, ci_low, ci_high = values
        rows.append(
            _empty_test(
                population=population,
                metric=metric,
                n=n,
                n_nonzero=n,
                mean_diff=mean_diff,
                ci_low=ci_low,
                ci_high=ci_high,
                wilcoxon_p=1.0,
                wilcoxon_p_holm=1.0,
                permutation_p=1.0,
                rank_biserial=0.0,
                wins=0,
                ties=n,
                losses=0,
            )
        )
    return rows


@pytest.fixture
def forest_sources(
    tmp_path: Path,
) -> tuple[Path, list[dict[str, str]], list[dict[str, str]]]:
    """Human_primary tree plus sibling exp_final tests.csv for Figura 05."""
    results = tmp_path / "human_primary"
    results.mkdir()
    judge_dir = tmp_path / "exp_final"
    judge_dir.mkdir()
    human_rows = _primary_pair_tests(
        population=HUMAN_PRIMARY,
        task=(-0.005649717514152543, -0.0988700564971695, 0.0847457627118644),
        fact_f1=(-0.0036409290646271185, -0.06777206080170975, 0.05908079992825337),
        support=(0.04690287025405085, 0.002646158002280932, 0.08990136388626652),
    )
    judge_rows = _primary_pair_tests(
        population="semantic_primary",
        task=(-0.01129943502825424, -0.08474576271188179, 0.06497175141245763),
        fact_f1=(-0.05923683974525424, -0.10300488745396144, -0.016621379248516956),
        support=(0.004091216379338983, -0.021341881204204237, 0.031111839745330078),
    )
    _write_csv(results / "tests.csv", TESTS_FIELDS, human_rows)
    _write_csv(judge_dir / "tests.csv", TESTS_FIELDS, judge_rows)
    (results / "agreement.json").write_text(
        json.dumps(
            {
                "grain": "dialogue",
                "n_dialogues": 348,
                "task_completed": {
                    "n": 348,
                    "n_uncertain_excluded": 0,
                    "percent_agreement": 0.89,
                    "cohen_kappa": 0.73,
                },
                "fact_ids_stated": {
                    "n": 241,
                    "exact_match_rate": 0.02,
                    "precision": 0.33,
                    "recall": 0.89,
                    "f1": 0.48,
                    "true_positive": 242,
                    "false_positive": 496,
                    "false_negative": 30,
                    "n_omitted_ambiguous_judge_calls": 107,
                },
                "claim_support_label": {
                    "n": 348,
                    "percent_agreement": 0.45,
                    "cohen_kappa": 0.10,
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        results,
        _read_table(results / "tests.csv"),
        _read_table(judge_dir / "tests.csv"),
    )


def test_figure_legends_sit_outside_the_axes(
    descriptive_rows: list[dict[str, str]],
    tests_rows: list[dict[str, str]],
    forest_sources: tuple[Path, list[dict[str, str]], list[dict[str, str]]],
) -> None:
    pytest.importorskip("matplotlib")
    _, human, judge = forest_sources

    for fig in (
        build_category_bars(descriptive_rows),
        build_wtl_bars(tests_rows),
        build_paired_forest(human, judge),
    ):
        fig.set_size_inches(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legends = list(fig.legends)
        for axis in fig.axes:
            if axis.get_legend() is not None:
                legends.append(axis.get_legend())
        assert legends
        for legend in legends:
            legend_box = legend.get_window_extent(renderer)
            for axis in fig.axes:
                assert not legend_box.overlaps(axis.get_window_extent(renderer))
        _pyplot().close(fig)


def _relabel_population(results: Path, population: str) -> None:
    for name in ("descriptive.csv", "tests.csv"):
        path = results / name
        path.write_text(
            path.read_text(encoding="utf-8").replace("semantic_primary", population),
            encoding="utf-8",
        )


def test_human_primary_tables_fill_from_human_population(tmp_path: Path) -> None:
    results = _frozen_triple(tmp_path)
    _relabel_population(results, HUMAN_PRIMARY)

    write_thesis_artifacts(results)

    table = _read_table(results / "tables" / TABLE_01_NAME)
    assert table
    assert all(row["Média"] != "" for row in table)
    legend = (results / "tables" / TABLE_01_NAME).with_suffix(".legenda.txt")
    notes = legend.read_text(encoding="utf-8")
    assert "humana primária" in notes
    assert "results/human_primary/descriptive.csv" in notes
    tests_legend = (results / "tables" / TABLE_03_NAME).with_suffix(".legenda.txt")
    assert "human_primary" in tests_legend.read_text(encoding="utf-8")


def test_comparison_table_reports_107_omitted_fact_id_dialogues(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    results = tmp_path / "human_primary"
    results.mkdir()
    (results / "agreement.json").write_text(
        json.dumps(
            {
                "grain": "dialogue",
                "n_dialogues": 348,
                "task_completed": {
                    "n": 348,
                    "n_uncertain_excluded": 0,
                    "percent_agreement": 0.8936781609195402,
                    "cohen_kappa": 0.7286292362164897,
                },
                "fact_ids_stated": {
                    "n": 241,
                    "exact_match_rate": 0.02074688796680498,
                    "precision": 0.32791327913279134,
                    "recall": 0.8897058823529411,
                    "f1": 0.47920792079207924,
                    "true_positive": 242,
                    "false_positive": 496,
                    "false_negative": 30,
                    "n_omitted_ambiguous_judge_calls": 107,
                },
                "claim_support_label": {
                    "n": 348,
                    "percent_agreement": 0.4540229885057471,
                    "cohen_kappa": 0.10213060659143679,
                },
            }
        ),
        encoding="utf-8",
    )
    with (results / "conclusions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = DictWriter(
            handle,
            fieldnames=(
                "metric",
                "judge_direction",
                "human_direction",
                "judge_mean_diff",
                "human_mean_diff",
                "judge_wilcoxon_p_holm",
                "human_wilcoxon_p_holm",
                "same_direction",
                "same_significance",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "metric": "claim_support",
                "judge_direction": "fsm",
                "human_direction": "fsm",
                "judge_mean_diff": 0.004,
                "human_mean_diff": 0.0469,
                "judge_wilcoxon_p_holm": 1.0,
                "human_wilcoxon_p_holm": 0.0476,
                "same_direction": "True",
                "same_significance": "False",
            }
        )
    write_comparison_artifacts(results)
    table = _read_table(results / "tables" / TABLE_04_NAME)
    exact = next(row for row in table if "conjunto exato" in row["Comparação"])
    assert exact["n"] == "241"
    assert "107" in exact["Observação"]
    legend = (results / "tables" / TABLE_04_NAME).with_suffix(".legenda.txt")
    notes = legend.read_text(encoding="utf-8")
    assert "107" in notes
    assert "GGUF" in notes
    conclusions = _read_table(results / "tables" / TABLE_05_NAME)
    assert conclusions[0]["Mesma significância"] == "Não"
    figure_notes = (results / "figures" / f"{FIGURE_04_STEM}.legenda.txt").read_text(
        encoding="utf-8"
    )
    assert "107" in figure_notes
    assert not (results / "figures" / f"{FIGURE_05_STEM}.png").exists()


def test_paired_intervals_copy_six_frozen_cis(
    forest_sources: tuple[Path, list[dict[str, str]], list[dict[str, str]]],
) -> None:
    _, human, judge = forest_sources

    rows = paired_interval_rows(human, judge)

    assert len(rows) == 6
    by_key = {(row["metric"], row["instrument"]): row for row in rows}
    human_support = by_key[("claim_support", "human")]
    assert human_support["mean_diff"] == pytest.approx(0.04690287025405085)
    assert human_support["ci_low"] == pytest.approx(0.002646158002280932)
    assert human_support["ci_high"] == pytest.approx(0.08990136388626652)
    judge_f1 = by_key[("fact_f1", "judge")]
    assert judge_f1["ci_high"] < 0
    human_f1 = by_key[("fact_f1", "human")]
    assert human_f1["ci_low"] < 0 < human_f1["ci_high"]
    judge_support = by_key[("claim_support", "judge")]
    assert judge_support["ci_low"] < 0 < judge_support["ci_high"]
    assert human_support["ci_low"] > 0
    assert [row["instrument"] for row in rows] == [
        "judge",
        "human",
        "judge",
        "human",
        "judge",
        "human",
    ]


def test_committed_forest_intervals_match_frozen_tests_csv() -> None:
    human = _read_table(REPO_ROOT / "results" / "human_primary" / "tests.csv")
    judge = _read_table(REPO_ROOT / "results" / "exp_final" / "tests.csv")

    rows = paired_interval_rows(human, judge)

    by_key = {(row["metric"], row["instrument"]): row for row in rows}
    human_support = by_key[("claim_support", "human")]
    assert human_support["n"] == 59
    assert format_pt_br_number(human_support["ci_low"]) == "0,003"
    assert format_pt_br_number(by_key[("fact_f1", "judge")]["ci_high"]) == "-0,017"
    assert by_key[("fact_f1", "human")]["ci_low"] < 0
    assert by_key[("fact_f1", "human")]["ci_high"] > 0


def test_forest_is_one_untitled_panel_with_zero_line_and_metric_yticks(
    forest_sources: tuple[Path, list[dict[str, str]], list[dict[str, str]]],
) -> None:
    pytest.importorskip("matplotlib")
    _, human, judge = forest_sources

    fig = build_paired_forest(human, judge)

    assert len(fig.axes) == 1
    axis = fig.axes[0]
    assert fig.get_suptitle() == ""
    assert axis.get_title() == ""
    zero_lines = [
        line
        for line in axis.lines
        if list(line.get_xdata()) == [0, 0] or tuple(line.get_xdata()) == (0, 0)
    ]
    assert zero_lines
    assert [text.get_text() for text in axis.get_yticklabels()] == [
        "Tarefa concluída",
        "F1 dos fatos esperados",
        "Suporte das afirmações",
    ]
    _pyplot().close(fig)


def test_forest_plots_frozen_means_by_instrument_and_annotates_human_support(
    forest_sources: tuple[Path, list[dict[str, str]], list[dict[str, str]]],
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgba

    _, human, judge = forest_sources

    fig = build_paired_forest(human, judge)

    containers = fig.axes[0].containers
    plotted = [
        float(value)
        for container in containers
        for value in container.lines[0].get_xdata()
    ]
    expected = [row["mean_diff"] for row in paired_interval_rows(human, judge)]
    assert sorted(plotted) == pytest.approx(sorted(expected))
    markers = {container.lines[0].get_marker() for container in containers}
    colors = {_rgb(to_rgba(container.lines[0].get_color())) for container in containers}
    assert markers == set(INSTRUMENT_MARKERS.values())
    assert colors == {_rgb(to_rgba(color)) for color in INSTRUMENT_FACECOLORS.values()}
    assert INSTRUMENT_FACECOLORS["judge"] == AGENT_FACECOLORS["baseline"]
    assert INSTRUMENT_FACECOLORS["human"] == AGENT_FACECOLORS["fsm"]
    labels = " ".join(text.get_text() for text in fig.axes[0].texts)
    assert "0,003" in labels
    assert "0,047" in labels
    _pyplot().close(fig)


def test_comparison_writes_figura_05_from_sibling_judge_tests(
    forest_sources: tuple[Path, list[dict[str, str]], list[dict[str, str]]],
) -> None:
    pytest.importorskip("matplotlib")
    results, _, _ = forest_sources

    write_comparison_artifacts(results)

    figures = results / "figures"
    notes = (figures / f"{FIGURE_05_STEM}.legenda.txt").read_text(encoding="utf-8")
    assert (figures / f"{FIGURE_05_STEM}.png").is_file()
    assert (figures / f"{FIGURE_05_STEM}.pdf").is_file()
    assert "Figura 5" in notes
    assert "IC 95%" in notes
    assert "tests.csv" in notes
    assert "não recalcula" in notes
    assert "zero" in notes.lower()
    assert "0,003" in notes
    assert "n = 59" in notes
    assert "marcadores" in notes.lower() or "círculo" in notes.lower()
