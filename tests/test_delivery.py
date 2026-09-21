"""T-24 delivery snapshot: byte copies, fail-closed, replace-total."""

import hashlib
from pathlib import Path

import pytest

from helpers import FROZEN_V1_HASH, REPO_ROOT, V1_DIR, make_manifest
from sim.runner import hash_dataset

RESULTS_DIR = REPO_ROOT / "results" / "exp_final"
METRICS_CSV = RESULTS_DIR / "metrics.csv"


def _write_run(directory: Path, *, exp_id: str = "exp_final") -> Path:
    run_dir = directory / exp_id
    run_dir.mkdir(parents=True)
    manifest = make_manifest(exp_id=exp_id)
    (run_dir / "manifest.json").write_text(
        manifest.model_dump_json() + "\n", encoding="utf-8"
    )
    return run_dir


def test_export_delivery_copies_dataset_metrics_tables_figures_and_manifest(
    tmp_path: Path,
) -> None:
    from sim.delivery import export_delivery

    run_dir = _write_run(tmp_path)
    dest = tmp_path / "simulacao_v1"

    export_delivery(run_dir, dest)

    for source in sorted(V1_DIR.glob("*.jsonl")):
        copied = dest / "scenarios" / "v1" / source.name
        assert copied.read_bytes() == source.read_bytes()
    assert (dest / "metrics.csv").read_bytes() == METRICS_CSV.read_bytes()
    for source in sorted((RESULTS_DIR / "tables").rglob("*")):
        if source.is_file():
            copied = dest / "tables" / source.relative_to(RESULTS_DIR / "tables")
            assert copied.read_bytes() == source.read_bytes()
    for source in sorted((RESULTS_DIR / "figures").rglob("*")):
        if source.is_file():
            copied = dest / "figures" / source.relative_to(RESULTS_DIR / "figures")
            assert copied.read_bytes() == source.read_bytes()
    assert (dest / "manifest.json").read_bytes() == (
        run_dir / "manifest.json"
    ).read_bytes()
    assert not (dest / "metrics_frozen_gate.csv").exists()
    assert not (dest / "cases.md").exists()
    assert not (dest / "dialogues").exists()
    assert not (dest / "llm_calls.jsonl").exists()

    sources = (dest / "SOURCES.txt").read_text(encoding="utf-8")
    assert f"exp_id={run_dir.name}" in sources or "exp_id=exp_final" in sources
    assert f"dataset_hash={FROZEN_V1_HASH}" in sources
    assert "git_commit=" in sources
    assert "package_version=1.2.0" in sources
    assert "git_tag=v1.2" in sources
    metrics_sha = hashlib.sha256(METRICS_CSV.read_bytes()).hexdigest()
    assert f"metrics_sha256={metrics_sha}" in sources
    assert hash_dataset(V1_DIR) == FROZEN_V1_HASH


def test_export_delivery_refuses_a_dataset_hash_mismatch(tmp_path: Path) -> None:
    from sim.delivery import DeliveryError, export_delivery

    run_dir = _write_run(tmp_path)
    dest = tmp_path / "simulacao_v1"
    scenarios = tmp_path / "v1"
    scenarios.mkdir()
    for path in V1_DIR.glob("*.jsonl"):
        (scenarios / path.name).write_bytes(path.read_bytes())
    mutated = next(scenarios.glob("*.jsonl"))
    mutated.write_bytes(mutated.read_bytes() + b"\n")

    with pytest.raises(DeliveryError, match="does not match frozen v1"):
        export_delivery(run_dir, dest, scenarios_dir=scenarios)

    assert not dest.exists()


def test_export_delivery_refuses_a_missing_manifest(tmp_path: Path) -> None:
    from sim.delivery import DeliveryError, export_delivery

    run_dir = tmp_path / "exp_final"
    run_dir.mkdir()
    dest = tmp_path / "simulacao_v1"
    dest.mkdir()

    with pytest.raises(DeliveryError, match=r"manifest\.json"):
        export_delivery(run_dir, dest)

    assert list(dest.iterdir()) == []

    missing_results = tmp_path / "results"
    missing_results.mkdir()
    run_dir = _write_run(tmp_path / "ok_run")
    with pytest.raises(DeliveryError, match=r"metrics\.csv"):
        export_delivery(run_dir, dest, results_dir=missing_results)
    assert list(dest.iterdir()) == []


def test_export_delivery_refuses_a_non_empty_destination_without_replace(
    tmp_path: Path,
) -> None:
    from sim.delivery import DeliveryError, export_delivery

    run_dir = _write_run(tmp_path)
    dest = tmp_path / "simulacao_v1"
    dest.mkdir()
    stale = dest / "stale.txt"
    stale.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(DeliveryError, match="not empty"):
        export_delivery(run_dir, dest)

    assert stale.read_text(encoding="utf-8") == "keep me\n"
    assert not (dest / "metrics.csv").exists()


def test_export_delivery_replace_removes_stale_destination_files(
    tmp_path: Path,
) -> None:
    from sim.delivery import export_delivery

    run_dir = _write_run(tmp_path)
    dest = tmp_path / "simulacao_v1"
    dest.mkdir()
    stale = dest / "stale.txt"
    stale.write_text("old\n", encoding="utf-8")
    leftover = dest / "nested"
    leftover.mkdir()
    (leftover / "old.csv").write_text("nope\n", encoding="utf-8")

    export_delivery(run_dir, dest, replace=True)

    assert not stale.exists()
    assert not leftover.exists()
    assert (dest / "metrics.csv").is_file()
    assert (dest / "SOURCES.txt").is_file()


def test_delivery_script_writes_the_directory(tmp_path: Path) -> None:
    from helpers import load_script
    from sim.delivery import DEFAULT_DEST, RELEASE_TAG

    script = load_script("scripts/export_delivery.py")
    run_dir = _write_run(tmp_path)
    dest = tmp_path / "out"

    assert script.main(["--run", str(run_dir), "--out", str(dest)]) == 0

    assert (dest / "SOURCES.txt").is_file()
    assert (dest / "metrics.csv").is_file()
    assert (
        Path.home() / "Documents" / "mba" / "entregas" / "simulacao_v1"
    ) == DEFAULT_DEST
    assert RELEASE_TAG == "v1.2"
