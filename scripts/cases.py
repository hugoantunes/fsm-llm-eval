r"""Write T-21 category-direction and exemplar-candidate CSVs.

Usage::

    uv run python scripts/cases.py \\
        --tests results/exp_final/tests.csv \\
        --metrics results/exp_final/metrics.csv \\
        --out results/exp_final
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.cases import write_cases_csvs


def main(argv: Sequence[str] | None = None) -> int:
    """Project T-19 tests and T-18 metrics into the T-21 CSVs."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy T-19 category directions and rank T-21 exemplar candidates. "
            "Does not overwrite results/cases.md."
        )
    )
    parser.add_argument(
        "--tests",
        type=Path,
        default=Path("results/exp_final/tests.csv"),
        help="T-19 tests.csv (authoritative direction and V/E/D)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/exp_final/metrics.csv"),
        help="T-18 semantic-primary metrics.csv (ranking only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/exp_final"),
        help="directory for category_directions.csv and case_candidates.csv",
    )
    args = parser.parse_args(argv)
    if not args.tests.is_file():
        print(f"sim: {args.tests} is missing", file=sys.stderr)
        return 1
    if not args.metrics.is_file():
        print(f"sim: {args.metrics} is missing", file=sys.stderr)
        return 1
    write_cases_csvs(args.tests, args.metrics, args.out)
    print(
        f"sim: wrote {args.out / 'category_directions.csv'} "
        f"and {args.out / 'case_candidates.csv'}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
