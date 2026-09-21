r"""Copy the T-24 delivery snapshot.

Usage::

    uv run python scripts/export_delivery.py
    uv run python scripts/export_delivery.py --replace
    uv run python scripts/export_delivery.py --run runs/exp_final --out /tmp/out
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.delivery import DEFAULT_DEST, DeliveryError, export_delivery


def main(argv: Sequence[str] | None = None) -> int:
    """Copy the frozen dataset, scored artifacts and manifest to ``--out``."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy the T-24 delivery snapshot (frozen v1, semantic-primary "
            "metrics.csv, tables, figures, manifest, SOURCES.txt)."
        )
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/exp_final"),
        help="sim run directory that holds manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DEST,
        help="destination directory (default: ~/Documents/mba/entregas/simulacao_v1)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete a non-empty destination and write a fresh snapshot",
    )
    args = parser.parse_args(argv)
    try:
        dest = export_delivery(args.run, args.out, replace=args.replace)
    except DeliveryError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    print(f"sim: wrote delivery snapshot to {dest}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
