"""Notebook runner for T-20 thesis plates."""

import json
from pathlib import Path

import pytest

from helpers import REPO_ROOT
from test_reporting import _frozen_triple

NOTEBOOK = REPO_ROOT / "notebooks" / "analysis.ipynb"


def test_notebook_invokes_reporter_with_frozen_result_set() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = "".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    assert "write_thesis_artifacts" in sources
    assert "metrics.csv" in sources
    assert "descriptive.csv" in sources
    assert "tests.csv" in sources
    assert "results/exp_final" in sources
    assert "results/human_primary" in sources


def test_notebook_executes_against_synthetic_t19_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _frozen_triple(tmp_path)
    monkeypatch.setenv("SIM_RESULTS_DIR", str(results))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    for cell in payload["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec("".join(cell["source"]), namespace)
    assert (results / "tables" / "resultados_tabela-01_descritiva-geral.csv").is_file()
    assert (results / "tables" / "resultados_tabela-03_testes-pareados.csv").is_file()


def test_notebook_finds_csvs_when_kernel_cwd_is_notebooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "repo"
    (fake / "src" / "sim").mkdir(parents=True)
    (fake / "notebooks").mkdir()
    (fake / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    source = _frozen_triple(tmp_path)
    dest = fake / "results" / "exp_final"
    dest.mkdir(parents=True)
    for name in ("metrics.csv", "descriptive.csv", "tests.csv"):
        (dest / name).write_bytes((source / name).read_bytes())
    monkeypatch.chdir(fake / "notebooks")
    monkeypatch.delenv("SIM_RESULTS_DIR", raising=False)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    for cell in payload["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec("".join(cell["source"]), namespace)
    assert (dest / "tables" / "resultados_tabela-01_descritiva-geral.csv").is_file()


def test_notebook_writes_judge_and_human_into_separate_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "repo"
    (fake / "src" / "sim").mkdir(parents=True)
    (fake / "notebooks").mkdir()
    (fake / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    source = _frozen_triple(tmp_path)
    judge = fake / "results" / "exp_final"
    human = fake / "results" / "human_primary"
    for dest in (judge, human):
        dest.mkdir(parents=True)
        for name in ("metrics.csv", "descriptive.csv", "tests.csv"):
            (dest / name).write_bytes((source / name).read_bytes())
    monkeypatch.chdir(fake / "notebooks")
    monkeypatch.delenv("SIM_RESULTS_DIR", raising=False)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    for cell in payload["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec("".join(cell["source"]), namespace)
    plate = "resultados_tabela-01_descritiva-geral.csv"
    assert (judge / "tables" / plate).is_file()
    assert (human / "tables" / plate).is_file()
    assert judge / "tables" != human / "tables"
