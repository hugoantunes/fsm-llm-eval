r"""Split the frozen ``fact_f1`` effect into its counts, without re-judging (T-22).

Exploratory post hoc. Re-reads the judge output already in the run's
``llm_calls.jsonl``, verifies every reconstruction against the exported
``metrics.csv``, and reports TP, FN and the FP total split into the
knowledge-base facts the scenario did not require and the claims the knowledge
base does not support. Makes no LLM call and writes nothing under ``runs/``.

Usage::

    uv run python scripts/decompose_fact_f1.py \\
        --run runs/exp_final \\
        --metrics results/exp_final/metrics.csv \\
        --out results/exp_final
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.decomposition import (
    DECOMPOSITION_CSV,
    DECOMPOSITION_JSON,
    DecompositionError,
    decompose,
    load_frozen_rows,
    reconstruct_fact_counts,
    write_decomposition,
    write_provenance,
)
from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.judge import redact_canary
from sim.kb import load_kb
from sim.schemas import DialogueLog, load_scenarios, render_transcript


def main(argv: Sequence[str] | None = None) -> int:
    """Reconstruct the fact counts of one run and write the decomposition."""
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the frozen fact_f1 effect into TP, FN and the two kinds "
            "of false positive, re-reading judge output instead of re-judging."
        )
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/exp_final"),
        help="run directory holding dialogues/ and llm_calls.jsonl",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/exp_final/metrics.csv"),
        help="exported metrics CSV every reconstruction must reproduce (T-18)",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("data/scenarios/v1"),
        help="scenario directory the run was played against",
    )
    parser.add_argument(
        "--fsm-dir", type=Path, default=DEFAULT_FSM_DIR, help="FSM directory"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/exp_final"),
        help=f"directory for {DECOMPOSITION_CSV} and {DECOMPOSITION_JSON}",
    )
    args = parser.parse_args(argv)
    for path in (args.metrics, args.run / "llm_calls.jsonl"):
        if not path.is_file():
            print(f"sim: {path} is missing", file=sys.stderr)
            return 1

    kb = load_kb()
    fsm = load_fsm(args.fsm_dir)
    scenarios = {
        scenario.id: scenario
        for scenario in load_scenarios(args.scenarios, kb=kb, fsm=fsm)
    }
    frozen_rows = load_frozen_rows(args.metrics)
    transcripts = {}
    for path in sorted((args.run / "dialogues").glob("*.jsonl")):
        log = DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        key = (log.scenario_id, log.agent, log.repetition)
        if key not in frozen_rows or not log.transcript:
            continue
        transcripts[key] = redact_canary(
            render_transcript(log.transcript), scenarios[log.scenario_id].canary
        )

    missing = sorted(set(frozen_rows) - set(transcripts))
    if missing:
        print(
            f"sim: {len(missing)} scored dialogues have no transcript under "
            f"{args.run / 'dialogues'}: {missing[:3]}",
            file=sys.stderr,
        )
        return 1
    try:
        counts = reconstruct_fact_counts(
            run_dir=args.run,
            transcripts=transcripts,
            required={
                scenario_id: scenario.required_facts
                for scenario_id, scenario in scenarios.items()
            },
            frozen_rows=frozen_rows,
        )
    except DecompositionError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1

    rows = decompose(counts)
    args.out.mkdir(parents=True, exist_ok=True)
    write_decomposition(args.out / DECOMPOSITION_CSV, rows)
    write_provenance(
        args.out / DECOMPOSITION_JSON,
        {
            "run_dir": str(args.run),
            "metrics_csv": str(args.metrics),
            "scenarios_dir": str(args.scenarios),
            "n_dialogues_scored": len(frozen_rows),
            "n_dialogues_reconstructed": len(counts),
            "n_unmatched": len(frozen_rows) - len(counts),
            "n_ambiguous": 0,
            "n_scenarios_paired": max(row.n for row in rows),
            "judge_calls_replayed": True,
            "judge_calls_made": 0,
        },
    )
    print(
        f"sim: wrote {args.out / DECOMPOSITION_CSV} and "
        f"{args.out / DECOMPOSITION_JSON} from {len(counts)} verified dialogues",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
