"""Copy the T-24 delivery snapshot out of the repository.

The folder under ``~/Documents/mba/entregas/simulacao_v1`` is a distribution
copy, not another source of truth. ``just deliver`` is an operator step after
the annotated ``v1.2`` tag so ``SOURCES.txt`` names that commit.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from sim import __version__
from sim.runner import FROZEN_V1_HASH, hash_dataset
from sim.schemas import Manifest

__all__ = [
    "DEFAULT_DEST",
    "RELEASE_TAG",
    "DeliveryError",
    "export_delivery",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_DIR = _REPO_ROOT / "data" / "scenarios" / "v1"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results" / "exp_final"
DEFAULT_DEST = Path.home() / "Documents" / "mba" / "entregas" / "simulacao_v1"
RELEASE_TAG = "v1.2"
_METRICS_NAME = "metrics.csv"
_MANIFEST_NAME = "manifest.json"
_SOURCES_NAME = "SOURCES.txt"


class DeliveryError(RuntimeError):
    """The snapshot cannot be written; the message says what to fix."""


def export_delivery(
    run_dir: Path,
    dest: Path,
    *,
    replace: bool = False,
    scenarios_dir: Path | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Byte-copy the frozen dataset, scored artifacts, manifest and provenance.

    Validates every required source and the frozen dataset hash before touching
    ``dest``. ``replace=True`` deletes the whole destination tree; it does not
    merge. Returns ``dest``.
    """
    scenarios = DEFAULT_SCENARIOS_DIR if scenarios_dir is None else scenarios_dir
    results = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    sources = _required_sources(run_dir, scenarios, results)
    dataset_hash = _require_frozen_hash(scenarios)
    git_commit = _git_commit(_REPO_ROOT)
    manifest = Manifest.model_validate_json(
        sources.manifest.read_text(encoding="utf-8")
    )
    metrics_sha = hashlib.sha256(sources.metrics.read_bytes()).hexdigest()
    if dest.exists() and _is_non_empty(dest) and not replace:
        raise DeliveryError(
            f"{dest} is not empty. Pass replace=True (just deliver --replace) "
            "to delete it and write a fresh snapshot"
        )
    staging = dest.parent / f".{dest.name}.{uuid4().hex}.tmp"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_snapshot(
            staging,
            sources,
            exp_id=manifest.exp_id,
            dataset_hash=dataset_hash,
            git_commit=git_commit,
            metrics_sha=metrics_sha,
        )
        if dest.exists():
            shutil.rmtree(dest)
        staging.replace(dest)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return dest


class _Sources:
    """Resolved paths that must exist before the destination is mutated."""

    def __init__(
        self,
        *,
        scenarios: Path,
        metrics: Path,
        tables: Path,
        figures: Path,
        manifest: Path,
        jsonl: Sequence[Path],
    ) -> None:
        self.scenarios = scenarios
        self.metrics = metrics
        self.tables = tables
        self.figures = figures
        self.manifest = manifest
        self.jsonl = list(jsonl)


def _required_sources(run_dir: Path, scenarios: Path, results: Path) -> _Sources:
    """Return the snapshot inputs, or raise before ``dest`` is touched."""
    missing: list[str] = []
    jsonl = sorted(scenarios.glob("*.jsonl"))
    if not jsonl:
        missing.append(f"{scenarios}/*.jsonl")
    metrics = results / _METRICS_NAME
    tables = results / "tables"
    figures = results / "figures"
    manifest = run_dir / _MANIFEST_NAME
    for path, label in (
        (metrics, str(metrics)),
        (tables, str(tables)),
        (figures, str(figures)),
        (manifest, str(manifest)),
    ):
        if not path.exists():
            missing.append(label)
    if missing:
        raise DeliveryError(
            "delivery is missing "
            + ", ".join(missing)
            + ". Restore the frozen v1 dataset, results/exp_final/, and the "
            "run manifest before just deliver"
        )
    return _Sources(
        scenarios=scenarios,
        metrics=metrics,
        tables=tables,
        figures=figures,
        manifest=manifest,
        jsonl=jsonl,
    )


def _require_frozen_hash(scenarios: Path) -> str:
    """Return the dataset hash, or refuse a directory that is not frozen v1."""
    digest = hash_dataset(scenarios)
    if digest != FROZEN_V1_HASH:
        raise DeliveryError(
            f"dataset hash {digest} does not match frozen v1 "
            f"{FROZEN_V1_HASH}. Do not edit data/scenarios/v1/"
        )
    return digest


def _write_snapshot(
    staging: Path,
    sources: _Sources,
    *,
    exp_id: str,
    dataset_hash: str,
    git_commit: str,
    metrics_sha: str,
) -> None:
    """Populate ``staging`` with the byte copies and ``SOURCES.txt``."""
    scenario_out = staging / "scenarios" / "v1"
    scenario_out.mkdir(parents=True)
    for path in sources.jsonl:
        shutil.copy2(path, scenario_out / path.name)
    shutil.copy2(sources.metrics, staging / _METRICS_NAME)
    shutil.copytree(sources.tables, staging / "tables")
    shutil.copytree(sources.figures, staging / "figures")
    shutil.copy2(sources.manifest, staging / _MANIFEST_NAME)
    text = (
        f"exp_id={exp_id}\n"
        f"dataset_hash={dataset_hash}\n"
        f"git_commit={git_commit}\n"
        f"package_version={__version__}\n"
        f"git_tag={RELEASE_TAG}\n"
        f"metrics_sha256={metrics_sha}\n"
    )
    (staging / _SOURCES_NAME).write_text(text, encoding="utf-8")


def _is_non_empty(path: Path) -> bool:
    """Return whether ``path`` exists and holds at least one entry."""
    return path.is_dir() and any(path.iterdir())


def _git_commit(repo: Path) -> str:
    """Return the ``v1.2`` commit when tagged, otherwise ``HEAD``."""
    tagged = _git_rev_parse(repo, f"{RELEASE_TAG}^{{commit}}")
    if tagged is not None:
        return tagged
    head = _git_rev_parse(repo, "HEAD")
    if head is None:
        raise DeliveryError(
            "cannot read git HEAD. Run just deliver from a git checkout of "
            "this repository"
        )
    return head


def _git_rev_parse(repo: Path, rev: str) -> str | None:
    """Return ``git rev-parse rev`` in ``repo``, or None when git fails."""
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", rev],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
