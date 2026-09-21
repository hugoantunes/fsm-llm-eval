"""Copy a frozen run's dialogues into a sidecar for re-evaluation.

Copies only ``manifest.json`` and ``dialogues/``. Never copies ``cache/``,
metrics CSVs or ``llm_calls.jsonl``. Never writes into ``runs/pilot_v2``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.schemas import Manifest

MANIFEST_JSON = "manifest.json"
FROZEN_PILOT_V2 = Path("runs/pilot_v2")


class SidecarError(RuntimeError):
    """The sidecar cannot be prepared; the message says why."""


def prepare_eval_sidecar(source_run: Path, target_run: Path) -> Manifest:
    """Copy ``manifest.json`` and ``dialogues/`` from ``source_run`` to ``target_run``.

    If the target already has the same dialogue files, the copy is skipped so a
    live eval cache is left intact.

    Returns:
        The manifest written (or already present) at the target.
    """
    source = source_run.resolve()
    target = target_run.resolve()
    frozen = FROZEN_PILOT_V2.resolve()
    if source == target:
        raise SidecarError(
            f"--source and --target resolve to the same path ({source}). "
            "Use a sidecar target so the frozen run is not overwritten"
        )
    if target == frozen:
        raise SidecarError(
            f"--target resolves to {frozen}, the frozen Pilot v2 run. "
            "Use a sidecar such as runs/pilot_v2_mlx"
        )
    manifest_path = source / MANIFEST_JSON
    if not manifest_path.exists():
        raise SidecarError(f"{manifest_path} is missing")
    dialogues_dir = source / "dialogues"
    if not dialogues_dir.exists():
        raise SidecarError(f"{dialogues_dir} is missing")
    target.mkdir(parents=True, exist_ok=True)
    target_dialogues = target / "dialogues"
    if target_dialogues.exists():
        if _dialogue_fingerprint(target_dialogues) != _dialogue_fingerprint(
            dialogues_dir
        ):
            raise SidecarError(
                f"{target_dialogues} differs from {dialogues_dir}; "
                "refusing to overwrite a sidecar that is not a copy of the source"
            )
        shutil.copy2(manifest_path, target / MANIFEST_JSON)
        return Manifest.model_validate_json((target / MANIFEST_JSON).read_text("utf-8"))
    shutil.copy2(manifest_path, target / MANIFEST_JSON)
    shutil.copytree(dialogues_dir, target_dialogues)
    return Manifest.model_validate_json((target / MANIFEST_JSON).read_text("utf-8"))


def _dialogue_fingerprint(directory: Path) -> dict[str, str]:
    """Return ``{filename: sha256}`` for every JSONL in ``directory``."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.jsonl"))
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Copy a frozen run into an eval sidecar directory."""
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
        default=Path("runs/pilot_v2_mlx"),
        help="sidecar run directory for a later sim eval",
    )
    args = parser.parse_args(argv)
    try:
        prepare_eval_sidecar(args.source, args.target)
    except SidecarError as error:
        print(f"sim: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
