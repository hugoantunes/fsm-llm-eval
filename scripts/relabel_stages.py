"""Relabel one frozen run with the current stage-labeler prompt only.

This script creates or refreshes a sidecar run directory from an existing run
by copying only ``manifest.json`` and ``dialogues/``. It then runs
``StageLabeler`` on each ok dialogue and writes ``metrics_turn.csv`` in the
sidecar directory.

It never runs the judge, never regenerates dialogues, and never writes into the
source run.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from sim.config import ModelsConfig, load_models_config
from sim.evaluators import LabelledTurn, StageLabeler, labelled_turns
from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.io import atomic_write
from sim.llm import LLM_CALLS_LOG, Chat, LlmClient
from sim.prompts import DEFAULT_PROMPTS_DIR
from sim.schemas import DialogueLog, Manifest

MANIFEST_JSON = "manifest.json"
METRICS_TURN_CSV = "metrics_turn.csv"
EXPECTED_LABELER_MODEL = "qwen3.5:4b"


class RelabelError(RuntimeError):
    """The sidecar relabel cannot run; the message says why."""


def relabel(
    source_run: Path,
    target_run: Path,
    *,
    llm: Chat | None = None,
    config: ModelsConfig | None = None,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    fsm_dir: Path = DEFAULT_FSM_DIR,
) -> int:
    """Write sidecar ``metrics_turn.csv`` using only ``StageLabeler``.

    Returns:
        Number of turn rows written to ``metrics_turn.csv``.
    """
    source = source_run.resolve()
    target = target_run.resolve()
    if source == target:
        raise RelabelError(
            f"--source and --target resolve to the same path ({source}). "
            "Use a sidecar target so the frozen run is not overwritten"
        )
    _copy_source_artifacts(source, target)
    _reset_sidecar_outputs(target)
    model_config = config or load_models_config()
    spec = model_config.spec("state_labeler")
    if spec.name != EXPECTED_LABELER_MODEL:
        raise RelabelError(
            "models.state_labeler must stay qwen3.5:4b for this validation; "
            f"found {spec.name!r}"
        )
    if spec.seed is None:
        raise RelabelError(
            "models.state_labeler.seed must be set so relabel runs are reproducible"
        )
    client = llm or LlmClient(
        model_config,
        cache_dir=target / "cache",
        log_path=target / LLM_CALLS_LOG,
    )
    labeler = StageLabeler(
        client,
        spec=load_fsm(fsm_dir),
        prompts_dir=prompts_dir,
        seed=spec.seed,
    )
    rows: list[LabelledTurn] = []
    for log in _load_logs(target):
        if log.status != "ok":
            continue
        rows.extend(labelled_turns(log, labeler.label(log)))
    _write_turn_csv(target / METRICS_TURN_CSV, rows)
    return len(rows)


def _copy_source_artifacts(source: Path, target: Path) -> Manifest:
    """Copy ``manifest.json`` and ``dialogues/`` from ``source`` to ``target``."""
    manifest_path = source / MANIFEST_JSON
    if not manifest_path.exists():
        raise RelabelError(f"{manifest_path} is missing")
    dialogues_dir = source / "dialogues"
    if not dialogues_dir.exists():
        raise RelabelError(f"{dialogues_dir} is missing")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, target / MANIFEST_JSON)
    target_dialogues = target / "dialogues"
    if target_dialogues.exists():
        shutil.rmtree(target_dialogues)
    shutil.copytree(dialogues_dir, target_dialogues)
    return Manifest.model_validate_json((target / MANIFEST_JSON).read_text("utf-8"))


def _reset_sidecar_outputs(run_dir: Path) -> None:
    """Remove old relabel artifacts before writing new ones."""
    for file in (run_dir / METRICS_TURN_CSV, run_dir / LLM_CALLS_LOG):
        if file.exists():
            file.unlink()
    cache = run_dir / "cache"
    if cache.exists():
        shutil.rmtree(cache)


def _load_logs(run_dir: Path) -> list[DialogueLog]:
    """Load dialogue JSONL logs from ``run_dir/dialogues``."""
    return [
        DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "dialogues").glob("*.jsonl"))
    ]


def _write_turn_csv(path: Path, rows: Sequence[LabelledTurn]) -> None:
    """Write the turn-level labels to ``path`` with eval-compatible columns."""
    fieldnames = tuple(LabelledTurn.model_fields)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        dumped = row.model_dump()
        writer.writerow(
            {
                name: "" if dumped[name] is None else str(dumped[name])
                for name in fieldnames
            }
        )
    atomic_write(path, buffer.getvalue())


def main(argv: Sequence[str] | None = None) -> int:
    """Relabel a frozen run into a sidecar directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("runs/pilot_v2"),
        help="frozen run directory that provides manifest.json and dialogues/",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("runs/pilot_v2_labeler_v2"),
        help="sidecar run directory where relabel artifacts are written",
    )
    args = parser.parse_args(argv)
    try:
        rows = relabel(args.source, args.target)
    except RelabelError as error:
        print(f"sim: {error}", file=sys.stderr)
        return 1
    print(
        f"sim: wrote {rows} turn label row(s) to {args.target} "
        f"using model role state_labeler ({EXPECTED_LABELER_MODEL})",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
