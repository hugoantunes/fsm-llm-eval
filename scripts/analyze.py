r"""Write T-19 descriptive and paired-test tables from audited metrics CSVs.

Usage::

    uv run --group analysis python scripts/analyze.py \\
        --metrics results/exp_final/metrics.csv \\
        --frozen results/exp_final/metrics_frozen_gate.csv \\
        --out results/exp_final
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.analysis import N_RESAMPLES, SEED, analyze


def main(argv: Sequence[str] | None = None) -> int:
    """Run scenario-level paired statistics into ``out``."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate metrics.csv to the scenario, pair baseline vs FSM, "
            "and write descriptive.csv, tests.csv and environment.txt."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/exp_final/metrics.csv"),
        help="semantic-primary metrics CSV (T-18)",
    )
    parser.add_argument(
        "--frozen",
        type=Path,
        default=Path("results/exp_final/metrics_frozen_gate.csv"),
        help="frozen-gate metrics CSV (T-18)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/exp_final"),
        help="directory for descriptive.csv, tests.csv, environment.txt",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=N_RESAMPLES,
        help="Monte-Carlo draws for permutation and bootstrap (default 10000)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed (default 0)")
    args = parser.parse_args(argv)
    if not args.metrics.is_file():
        print(f"sim: {args.metrics} is missing", file=sys.stderr)
        return 1
    if not args.frozen.is_file():
        print(f"sim: {args.frozen} is missing", file=sys.stderr)
        return 1
    analyze(
        args.metrics,
        args.frozen,
        args.out,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    print(
        f"sim: wrote {args.out / 'descriptive.csv'}, {args.out / 'tests.csv'} "
        f"and {args.out / 'environment.txt'}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
