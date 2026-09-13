"""Summarize a ``sim run`` / ``sim eval`` directory for the T-15 pilot notes.

Reads ``manifest.json`` and ``llm_calls.jsonl`` and reports, per caller,
call counts, mean prompt tokens, mean output tokens and mean uncached
latency, plus seconds per dialogue for the run phase and the eval phase.

Usage::

    uv run python scripts/run_stats.py runs/exp_pilot
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sim.llm import LLM_CALLS_LOG, LlmCallRecord
from sim.schemas import Manifest

EVAL_CALLERS = frozenset({"judge_facts", "judge_global", "stage_labeler"})


@dataclass(frozen=True)
class CallerStats:
    """Averages over the calls of one ``caller`` in ``llm_calls.jsonl``.

    Latency is averaged over uncached calls only: a cache hit has no
    meaningful stopwatch. Token averages include every call.
    """

    n_calls: int
    mean_prompt_tokens: float
    mean_output_tokens: float
    mean_latency_s: float | None


@dataclass(frozen=True)
class RunStats:
    """The numbers ``docs/pilot.md`` quotes from one experiment directory."""

    callers: dict[str, CallerStats]
    run_s_per_dialogue: float | None
    eval_s_per_dialogue: float | None


def load_calls(path: Path) -> list[LlmCallRecord]:
    """Load every line of ``llm_calls.jsonl``; a missing file is an empty list."""
    if not path.exists():
        return []
    return [
        LlmCallRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def summarize(run_dir: Path) -> RunStats:
    """Return per-caller averages and seconds per dialogue from ``run_dir``."""
    grouped: dict[str, list[LlmCallRecord]] = defaultdict(list)
    for record in load_calls(run_dir / LLM_CALLS_LOG):
        grouped[record.caller].append(record)
    callers = {name: _caller_stats(rows) for name, rows in grouped.items()}
    manifest = Manifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    played = manifest.n_ok + manifest.n_failed
    eval_latency = sum(
        row.latency_s
        for rows in grouped.values()
        for row in rows
        if row.caller in EVAL_CALLERS and not row.cached
    )
    return RunStats(
        callers=callers,
        run_s_per_dialogue=(manifest.elapsed_s / played) if played else None,
        eval_s_per_dialogue=(eval_latency / manifest.n_ok) if manifest.n_ok else None,
    )


def fsm_turn_prompt_tokens(stats: RunStats) -> float | None:
    """Return FSM turn prompt tokens as agent plus classifier means."""
    fsm = stats.callers.get("fsm")
    classifier = stats.callers.get("classifier")
    if fsm is None or classifier is None:
        return None
    return fsm.mean_prompt_tokens + classifier.mean_prompt_tokens


def _caller_stats(rows: Sequence[LlmCallRecord]) -> CallerStats:
    """Average tokens over every call and latency over uncached ones."""
    n_calls = len(rows)
    live = [row.latency_s for row in rows if not row.cached]
    return CallerStats(
        n_calls=n_calls,
        mean_prompt_tokens=sum(row.prompt_tokens for row in rows) / n_calls,
        mean_output_tokens=sum(row.output_tokens for row in rows) / n_calls,
        mean_latency_s=sum(live) / len(live) if live else None,
    )


def render(stats: RunStats) -> str:
    """Format ``stats`` as a table plus the two seconds-per-dialogue lines."""
    lines = [
        f"{'caller':<16} {'n':>5} {'prompt':>8} {'output':>8} {'latency_s':>10}",
    ]
    for name in sorted(stats.callers):
        row = stats.callers[name]
        latency = "-" if row.mean_latency_s is None else f"{row.mean_latency_s:.2f}"
        lines.append(
            f"{name:<16} {row.n_calls:>5} {row.mean_prompt_tokens:>8.1f} "
            f"{row.mean_output_tokens:>8.1f} {latency:>10}"
        )
    run = "-" if stats.run_s_per_dialogue is None else f"{stats.run_s_per_dialogue:.1f}"
    ev = (
        "-" if stats.eval_s_per_dialogue is None else f"{stats.eval_s_per_dialogue:.1f}"
    )
    fsm_turn = fsm_turn_prompt_tokens(stats)
    fsm_turn_text = "-" if fsm_turn is None else f"{fsm_turn:.1f}"
    lines.append(f"run s/dialogue    {run}")
    lines.append(f"eval s/dialogue   {ev}")
    lines.append(f"fsm turn prompt   {fsm_turn_text}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the summary of the experiment directory in ``argv``."""
    parser = argparse.ArgumentParser(
        description="Summarize llm_calls.jsonl and the manifest of a sim run."
    )
    parser.add_argument(
        "run_dir", type=Path, help="experiment directory (runs/<exp_id>)"
    )
    args = parser.parse_args(argv)
    run_dir = args.run_dir
    if not (run_dir / "manifest.json").exists():
        print(f"sim: {run_dir / 'manifest.json'} is missing", file=sys.stderr)
        return 1
    print(render(summarize(run_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
