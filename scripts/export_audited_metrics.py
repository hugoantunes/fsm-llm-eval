r"""Export T-18 audited scored CSVs as atomic byte copies.

Usage::

    uv run python scripts/export_audited_metrics.py \\
        --run runs/exp_final \\
        --sidecar runs/exp_final_semantic
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.audit import AuditError, audited_export_dir, export_audited_metrics


def main(argv: Sequence[str] | None = None) -> int:
    """Audit ``run`` against ``sidecar`` and copy scored CSVs into ``out``."""
    parser = argparse.ArgumentParser(
        description=(
            "Census every manifest job, fail closed on integrity anomalies, "
            "and atomically byte-copy the semantic sidecar and frozen-gate "
            "metrics CSVs into results/<exp_id>/."
        )
    )
    parser.add_argument("--run", type=Path, required=True, help="sim run directory")
    parser.add_argument(
        "--sidecar",
        type=Path,
        required=True,
        help="semantic sidecar eval directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="export directory (default: results/<exp_id>)",
    )
    args = parser.parse_args(argv)
    try:
        report = export_audited_metrics(args.run, args.sidecar, args.out)
        destination = args.out if args.out is not None else audited_export_dir(args.run)
    except AuditError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    n_primary = sum(job.primary_status == "included_semantic" for job in report.jobs)
    n_frozen = sum(job.frozen_status == "included_frozen" for job in report.jobs)
    n_outliers = sum(len(group.outlier_ids) for group in report.latency)
    print(
        f"sim: wrote {destination / 'metrics.csv'} ({n_primary} scored) and "
        f"{destination / 'metrics_frozen_gate.csv'} ({n_frozen} scored). "
        f"{len(report.events)} llm-call event(s); "
        f"{n_outliers} latency outlier flag(s).",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
