r"""Materialize an explicit human selection of contract false positives.

Validates full run IDs against the run and the discovery artifact, then writes
the executable inclusion JSON. Does not decide semantic inclusion.

Usage::

    uv run python scripts/freeze_contract_false_positives.py \\
        runs/<exp_id> \\
        <scenario_id>__<agent>__repNN
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.adjudication import AdjudicationError, freeze_contract_false_positives


def main(argv: Sequence[str] | None = None) -> int:
    """Validate ``ids`` and write the run-local inclusion artifact."""
    parser = argparse.ArgumentParser(
        description=(
            "Materialize an explicit human-selected set of full dialogue-run "
            "IDs into runs/<exp_id>/adjudication/contract_false_positives.json. "
            "Does not infer IDs from audit flags. --replace changes a frozen "
            "methodological decision."
        )
    )
    parser.add_argument("run_dir", type=Path, help="experiment directory")
    parser.add_argument(
        "ids",
        nargs="+",
        help="full dialogue-run ids of the form {scenario}__{agent}__repNN",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "overwrite a different existing freeze; this changes the frozen "
            "methodological decision"
        ),
    )
    args = parser.parse_args(argv)
    try:
        path = freeze_contract_false_positives(
            args.run_dir, args.ids, replace=args.replace
        )
    except AdjudicationError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    print(
        f"sim: wrote {path} ({len(set(args.ids))} id(s)). "
        "This file is the executable representation of an already-documented "
        "human adjudication.",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
