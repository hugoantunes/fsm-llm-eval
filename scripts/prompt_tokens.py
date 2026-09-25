"""Summarize the prompt size of each agent condition from ``llm_calls.jsonl``.

Ollama truncates the beginning of a prompt that does not fit the context window,
and the ``prompt_eval_count`` it reports then drops below the length of the text
sent. Per condition, this reports how many tokens the server evaluated (mean and
maximum), how many calls went over the client budget, and the range of tokens
evaluated per character sent: a truncated call shows up as an outlier below the
rest.

Usage::

    uv run python scripts/prompt_tokens.py runs/exp_final
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from sim.config import load_models_config
from sim.llm import LLM_CALLS_LOG, LlmCallRecord

AGENT_ROLE = "agent"


@dataclass(frozen=True)
class PromptSummary:
    """Prompt size of the successful calls of one agent condition."""

    n_calls: int
    mean_prompt_tokens: float
    max_prompt_tokens: int
    over_budget: int
    min_tokens_per_char: float
    max_tokens_per_char: float


def load_calls(path: Path) -> list[LlmCallRecord]:
    """Load every line of ``llm_calls.jsonl``."""
    return [
        LlmCallRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def summarize(calls: Sequence[LlmCallRecord], budget: int) -> dict[str, PromptSummary]:
    """Return one summary per agent caller, over calls that got an answer."""
    grouped: dict[str, list[LlmCallRecord]] = defaultdict(list)
    for call in calls:
        if call.role == AGENT_ROLE and call.error is None:
            grouped[call.caller].append(call)
    return {name: _summary(rows, budget) for name, rows in sorted(grouped.items())}


def _summary(rows: Sequence[LlmCallRecord], budget: int) -> PromptSummary:
    tokens = [row.prompt_tokens for row in rows]
    ratios = [row.prompt_tokens / _chars(row) for row in rows]
    return PromptSummary(
        n_calls=len(rows),
        mean_prompt_tokens=sum(tokens) / len(tokens),
        max_prompt_tokens=max(tokens),
        over_budget=sum(1 for value in tokens if value > budget),
        min_tokens_per_char=min(ratios),
        max_tokens_per_char=max(ratios),
    )


def _chars(row: LlmCallRecord) -> int:
    return sum(len(message.get("content", "")) for message in row.messages)


def write_csv(
    rows: dict[str, PromptSummary], path: Path, num_ctx: int, budget: int
) -> None:
    """Write one CSV row per condition, with the window and the budget used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", *PromptSummary.__dataclass_fields__, "num_ctx", "budget"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, summary in rows.items():
            writer.writerow(
                {
                    "condition": condition,
                    **asdict(summary),
                    "num_ctx": num_ctx,
                    "budget": budget,
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``prompt_tokens.csv`` for the run directory in ``argv``."""
    parser = argparse.ArgumentParser(
        description="Prompt tokens per agent condition, from llm_calls.jsonl."
    )
    parser.add_argument(
        "run_dir", type=Path, help="experiment directory (runs/<exp_id>)"
    )
    parser.add_argument(
        "--out", type=Path, help="CSV path (default results/<exp_id>/prompt_tokens.csv)"
    )
    args = parser.parse_args(argv)
    log = args.run_dir / LLM_CALLS_LOG
    if not log.exists():
        print(f"sim: {log} is missing", file=sys.stderr)
        return 1
    config = load_models_config()
    budget = config.max_prompt_tokens
    rows = summarize(load_calls(log), budget)
    out = args.out or Path("results") / args.run_dir.name / "prompt_tokens.csv"
    write_csv(rows, out, config.num_ctx, budget)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
