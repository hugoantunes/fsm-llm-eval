"""Discover instrument retry-exhausted failures for adjudication.

Writes generated discovery provenance. Does not decide semantic inclusion.

Usage::

    uv run python scripts/list_instrument_failures.py runs/exp_final
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.adjudication import (
    AdjudicationError,
    discover_instrument_failures,
    write_instrument_failures,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Discover target instrument failures of ``run_dir`` and write JSON."""
    parser = argparse.ArgumentParser(
        description=(
            "List failed dialogues with failure_kind=instrument and "
            "failure_reason=invalid_candidate_retry_exhausted. Writes generated "
            "discovery provenance, never an inclusion verdict."
        )
    )
    parser.add_argument("run_dir", type=Path, help="experiment directory")
    args = parser.parse_args(argv)
    try:
        report = discover_instrument_failures(args.run_dir)
        path = write_instrument_failures(args.run_dir, report)
    except AdjudicationError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    print(
        f"sim: wrote {path} ({len(report.candidates)} adjudication candidate(s)).",
        file=sys.stdout,
    )
    for row in report.census:
        print(
            f"{row.failure_kind} / {row.failure_reason}: {row.count}",
            file=sys.stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
