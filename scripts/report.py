r"""Write T-20 thesis tables and figures from frozen T-19 CSVs.

Usage::

    uv run --group analysis python scripts/report.py \\
        --results-dir results/exp_final
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.reporting import write_thesis_artifacts


def main(argv: Sequence[str] | None = None) -> int:
    """Format frozen T-19 CSVs into thesis tables and figures."""
    parser = argparse.ArgumentParser(
        description=(
            "Read metrics.csv, descriptive.csv and tests.csv from a T-19 "
            "result directory and write tables/ and figures/."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/exp_final"),
        help="directory holding the frozen T-19 CSVs (default results/exp_final)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="override path to metrics.csv",
    )
    parser.add_argument(
        "--descriptive",
        type=Path,
        default=None,
        help="override path to descriptive.csv",
    )
    parser.add_argument(
        "--tests",
        type=Path,
        default=None,
        help="override path to tests.csv",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="override directory for thesis tables",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="override directory for thesis figures",
    )
    parser.add_argument(
        "--population",
        default=None,
        help=(
            "confirmatory population in descriptive.csv "
            "(default: semantic_primary if present, else human_primary)"
        ),
    )
    args = parser.parse_args(argv)
    results_dir: Path = args.results_dir
    metrics = args.metrics if args.metrics is not None else results_dir / "metrics.csv"
    descriptive = (
        args.descriptive
        if args.descriptive is not None
        else results_dir / "descriptive.csv"
    )
    tests = args.tests if args.tests is not None else results_dir / "tests.csv"
    for path, label in (
        (metrics, "metrics.csv"),
        (descriptive, "descriptive.csv"),
        (tests, "tests.csv"),
    ):
        if not path.is_file():
            print(f"sim: {label} is missing: {path}", file=sys.stderr)
            return 1
    write_thesis_artifacts(
        results_dir,
        metrics_path=metrics,
        descriptive_path=descriptive,
        tests_path=tests,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        population=args.population,
    )
    tables = args.tables_dir if args.tables_dir is not None else results_dir / "tables"
    figures = (
        args.figures_dir if args.figures_dir is not None else results_dir / "figures"
    )
    print(f"sim: wrote {tables} and {figures}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
