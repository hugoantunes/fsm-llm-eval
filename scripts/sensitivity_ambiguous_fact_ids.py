r"""T-25: primary paired tests without the ambiguous-fact-ID dialogues.

Usage::

    uv run --group analysis python scripts/sensitivity_ambiguous_fact_ids.py \\
        --judge-metrics results/exp_final/metrics.csv \\
        --human-metrics results/human_primary/metrics.csv \\
        --dialogues results/human_primary/ambiguous_fact_ids.csv \\
        --out results/sensitivity/ambiguous_fact_ids
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.analysis import N_RESAMPLES, SEED, load_metrics_csv
from sim.sensitivity import (
    ambiguous_counts,
    load_dialogue_keys,
    sensitivity_rows,
    write_sensitivity,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Drop the listed dialogues and rerun the primary tests into ``out``."""
    parser = argparse.ArgumentParser(
        description=(
            "Drop the ambiguous-fact-ID dialogues from the judge and human "
            "metrics, rerun the primary paired tests, and write tests.csv "
            "and counts.csv."
        )
    )
    parser.add_argument(
        "--judge-metrics",
        type=Path,
        default=Path("results/exp_final/metrics.csv"),
        help="semantic-primary judge metrics CSV (T-18)",
    )
    parser.add_argument(
        "--human-metrics",
        type=Path,
        default=Path("results/human_primary/metrics.csv"),
        help="human-primary metrics CSV",
    )
    parser.add_argument(
        "--dialogues",
        type=Path,
        default=Path("results/human_primary/ambiguous_fact_ids.csv"),
        help="scenario_id,agent,repetition rows to drop",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/sensitivity/ambiguous_fact_ids"),
        help="directory for tests.csv and counts.csv",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=N_RESAMPLES,
        help="Monte-Carlo draws for permutation and bootstrap (default 10000)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed (default 0)")
    args = parser.parse_args(argv)
    for path in (args.judge_metrics, args.human_metrics, args.dialogues):
        if not path.is_file():
            print(f"sim: {path} is missing", file=sys.stderr)
            return 1
    judge = load_metrics_csv(args.judge_metrics)
    keys = load_dialogue_keys(args.dialogues)
    tests = sensitivity_rows(
        {"judge": judge, "human": load_metrics_csv(args.human_metrics)},
        keys,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    write_sensitivity(args.out, tests=tests, counts=ambiguous_counts(judge, keys))
    print(
        f"sim: wrote {args.out / 'tests.csv'} and {args.out / 'counts.csv'}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
