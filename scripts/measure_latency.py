"""Measure the run-phase latency budget: agent and simulator alternating, in parallel.

T-03 deliverable. Reproduces the call pattern of ``sim run`` without the real agents:
for each of P parallel dialogues and T turns, one simulator call (persona plus the
growing history), one classifier call (short, schema-constrained) and one agent call
(a KB-sized system prompt plus the history). Prints mean and p95 latency per role with
token counts, then the machine-hours budget for the N and K options of TICKETS.md,
assuming the planned turns per dialogue (``--budget-turns``, 8 by default) rather than
the turns measured. Run it on the Pro with the models of ``configs/models.yaml`` pulled
and ``scripts/ollama_env.sh`` applied; numbers from the Air are only indicative.

Usage::

    uv run python scripts/measure_latency.py --parallel 2 --turns 8 [--judge]
"""

import argparse
import asyncio
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from ollama import AsyncClient
from pydantic import BaseModel

AGENT_PROMPT_TOKENS = 2500
SIMULATOR_PROMPT_TOKENS = 300
JUDGE_PROMPT_TOKENS = 3500
BUDGET_OPTIONS = ((45, 3), (45, 5), (60, 3), (60, 5))
POLICY_SENTENCES = (
    "Fact F{n:02d}: standard delivery takes {a} to {b} business days after dispatch.",
    "Fact F{n:02d}: an item can be returned within {a} calendar days of delivery.",
    "Fact F{n:02d}: sale items can be exchanged within {a} days but not returned.",
    "Fact F{n:02d}: a payment link can be reissued {a} times, each valid {b} days.",
    "Fact F{n:02d}: an order can be cancelled for free until it is marked shipped.",
    "Fact F{n:02d}: confirm the order number and the email before sharing any details.",
)


class Event(BaseModel):
    """Schema of the classifier call, mirroring the event enum of T-08."""

    event: Literal["order_identified", "intent_classified", "user_dissatisfied", "none"]


class Verdict(BaseModel):
    """Schema of the judge probe, a stand-in for the global judgement of T-12."""

    accuracy: Literal["correct", "partial", "incorrect"]
    task_completed: bool
    justification: str


@dataclass(frozen=True)
class Call:
    """One measured LLM call."""

    role: str
    seconds: float
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RoleStats:
    """Latency and token statistics of one role."""

    role: str
    n: int
    mean_s: float
    p95_s: float
    mean_prompt_tokens: float
    mean_output_tokens: float
    output_tokens_per_s: float


def percentile(values: Sequence[float], q: float) -> float:
    """Return the nearest-rank ``q``-th percentile of ``values``."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(calls: Iterable[Call]) -> list[RoleStats]:
    """Aggregate calls per role, in order of first appearance."""
    by_role: dict[str, list[Call]] = {}
    for call in calls:
        by_role.setdefault(call.role, []).append(call)
    stats = []
    for role, group in by_role.items():
        seconds = [c.seconds for c in group]
        stats.append(
            RoleStats(
                role=role,
                n=len(group),
                mean_s=statistics.fmean(seconds),
                p95_s=percentile(seconds, 95),
                mean_prompt_tokens=statistics.fmean(c.prompt_tokens for c in group),
                mean_output_tokens=statistics.fmean(c.output_tokens for c in group),
                output_tokens_per_s=sum(c.output_tokens for c in group) / sum(seconds),
            )
        )
    return stats


def dialogues(scenarios: int, reps: int, agents: int = 2) -> int:
    """Number of dialogues in the experiment: scenarios x agents x repetitions."""
    return scenarios * agents * reps


def run_hours(
    role_mean_s: Mapping[str, float], n_dialogues: int, turns: int, parallel: int
) -> float:
    """Run-phase hours: each role once per turn, ``parallel`` dialogues at once."""
    seconds_per_turn = sum(role_mean_s.values())
    return n_dialogues * turns * seconds_per_turn / parallel / 3600


def judge_hours(
    mean_s: float, n_dialogues: int, calls_per_dialogue: int = 2, parallel: int = 1
) -> float:
    """Hours of the eval phase: ``calls_per_dialogue`` judge calls per dialogue."""
    return n_dialogues * calls_per_dialogue * mean_s / parallel / 3600


def filler(target_tokens: int) -> str:
    """Synthetic policy text of roughly ``target_tokens`` tokens (1.2 per word)."""
    words_needed = int(target_tokens / 1.2)
    sentences: list[str] = []
    words = 0
    n = 1
    while words < words_needed:
        template = POLICY_SENTENCES[(n - 1) % len(POLICY_SENTENCES)]
        sentence = template.format(n=n, a=3 + n % 7, b=10 + n % 5)
        sentences.append(sentence)
        words += len(sentence.split())
        n += 1
    return " ".join(sentences)


def as_messages(history: Sequence[tuple[str, str]], me: str) -> list[dict[str, str]]:
    """Render the shared history from one speaker's point of view."""
    return [
        {"role": "assistant" if speaker == me else "user", "content": text}
        for speaker, text in history
    ]


async def supports_thinking(client: AsyncClient, model: str) -> bool:
    """Tell whether the model has a thinking mode that must be switched off."""
    info = await client.show(model)
    return "thinking" in (info.capabilities or [])


async def timed_chat(
    client: AsyncClient,
    model: str,
    messages: list[dict[str, str]],
    *,
    role: str,
    num_ctx: int,
    num_predict: int,
    seed: int,
    think: bool,
    schema: dict[str, object] | None = None,
) -> tuple[Call, str]:
    """Run one chat call and return its measurement with the reply text."""
    options = {
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "temperature": 0,
        "seed": seed,
    }
    kwargs: dict[str, object] = {}
    if think:
        kwargs["think"] = False
    if schema is not None:
        kwargs["format"] = schema
    start = time.perf_counter()
    response = await client.chat(
        model=model, messages=messages, options=options, **kwargs
    )
    seconds = time.perf_counter() - start
    call = Call(
        role, seconds, response.prompt_eval_count or 0, response.eval_count or 0
    )
    return call, response.message.content or ""


async def dialogue(
    client: AsyncClient,
    agent_model: str,
    simulator_model: str,
    *,
    turns: int,
    num_ctx: int,
    seed: int,
    agent_think: bool,
    simulator_think: bool,
) -> list[Call]:
    """Simulate one dialogue: simulator, classifier and agent calls on every turn."""
    agent_system = (
        "You are the customer-service agent of an online store. Answer briefly and use "
        "only these facts:\n" + filler(AGENT_PROMPT_TOKENS)
    )
    simulator_system = (
        "You play a customer of an online store who wants to return a damaged "
        "item. Stay in character and write one short message per turn. Background:\n"
        + filler(SIMULATOR_PROMPT_TOKENS)
    )
    classifier_system = "Classify the customer's last message into exactly one event."
    history: list[tuple[str, str]] = []
    calls: list[Call] = []
    for turn in range(turns):
        instruction = "Write your next message." if turn else "Open the conversation."
        call, customer = await timed_chat(
            client,
            simulator_model,
            [
                {"role": "system", "content": simulator_system},
                *as_messages(history, me="customer"),
                {"role": "user", "content": instruction},
            ],
            role="simulator",
            num_ctx=num_ctx,
            num_predict=60,
            seed=seed + turn,
            think=simulator_think,
        )
        calls.append(call)
        history.append(("customer", customer))

        call, _ = await timed_chat(
            client,
            simulator_model,
            [
                {"role": "system", "content": classifier_system},
                {"role": "user", "content": f"Customer: {customer}"},
            ],
            role="classifier",
            num_ctx=num_ctx,
            num_predict=20,
            seed=seed,
            think=simulator_think,
            schema=Event.model_json_schema(),
        )
        calls.append(call)

        call, reply = await timed_chat(
            client,
            agent_model,
            [
                {"role": "system", "content": agent_system},
                *as_messages(history, me="agent"),
            ],
            role="agent",
            num_ctx=num_ctx,
            num_predict=120,
            seed=seed,
            think=agent_think,
        )
        calls.append(call)
        history.append(("agent", reply))
    return calls


async def judge_probe(
    client: AsyncClient, judge_model: str, *, num_ctx: int, seed: int, think: bool
) -> list[Call]:
    """Time two judge-sized calls: a long transcript prompt, a structured verdict."""
    prompt = (
        "Judge the agent's answers in this transcript against the knowledge base "
        "below. Transcript and knowledge base:\n" + filler(JUDGE_PROMPT_TOKENS)
    )
    calls = []
    for i in range(2):
        call, _ = await timed_chat(
            client,
            judge_model,
            [{"role": "user", "content": prompt}],
            role="judge",
            num_ctx=num_ctx,
            num_predict=300,
            seed=seed + i,
            think=think,
            schema=Verdict.model_json_schema(),
        )
        calls.append(call)
    return calls


def load_config(path: Path) -> tuple[dict[str, str | None], int]:
    """Read model names and ``num_ctx`` from ``configs/models.yaml``."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = {
        role: config["models"][role]["name"] for role in ("agent", "simulator", "judge")
    }
    return names, int(config["num_ctx"])


def report(stats: Sequence[RoleStats], budget_turns: int, parallel: int) -> str:
    """Render the per-role table and the budget for the N and K options."""
    lines = [
        f"{'role':<11}{'n':>4}{'mean s':>9}{'p95 s':>8}{'prompt tok':>12}"
        f"{'out tok':>9}{'out tok/s':>11}"
    ]
    for s in stats:
        lines.append(
            f"{s.role:<11}{s.n:>4}{s.mean_s:>9.2f}{s.p95_s:>8.2f}"
            f"{s.mean_prompt_tokens:>12.0f}{s.mean_output_tokens:>9.0f}"
            f"{s.output_tokens_per_s:>11.1f}"
        )
    means = {s.role: s.mean_s for s in stats}
    judge_mean = means.pop("judge", None)
    lines.append("")
    lines.append(
        f"Budget assuming {budget_turns} turns per dialogue and {parallel} dialogues "
        "in parallel:"
    )
    for scenarios, reps in BUDGET_OPTIONS:
        n = dialogues(scenarios, reps)
        run = run_hours(means, n, budget_turns, parallel)
        judge = f", judge {judge_hours(judge_mean, n):.1f} h" if judge_mean else ""
        lines.append(f"  N={scenarios} K={reps}: {n} dialogues, run {run:.1f} h{judge}")
    return "\n".join(lines)


async def measure(args: argparse.Namespace) -> str:
    """Run the measurement described by the command-line arguments."""
    names, num_ctx = load_config(args.config)
    agent_model = args.agent_model or names["agent"]
    simulator_model = args.simulator_model or names["simulator"]
    judge_model = args.judge_model or names["judge"]
    if not agent_model or not simulator_model:
        raise SystemExit(
            "agent and simulator models are not set: fill configs/models.yaml"
        )
    client = AsyncClient()
    agent_think = await supports_thinking(client, agent_model)
    simulator_think = await supports_thinking(client, simulator_model)
    runs = await asyncio.gather(
        *(
            dialogue(
                client,
                agent_model,
                simulator_model,
                turns=args.turns,
                num_ctx=num_ctx,
                seed=args.seed + 100 * i,
                agent_think=agent_think,
                simulator_think=simulator_think,
            )
            for i in range(args.parallel)
        )
    )
    calls = [call for run in runs for call in run]
    if args.judge:
        if not judge_model:
            raise SystemExit("judge model is not set: fill configs/models.yaml")
        judge_think = await supports_thinking(client, judge_model)
        calls.extend(
            await judge_probe(
                client, judge_model, num_ctx=num_ctx, seed=args.seed, think=judge_think
            )
        )
    header = (
        f"agent={agent_model} simulator={simulator_model}"
        f"{' judge=' + judge_model if args.judge else ''} num_ctx={num_ctx}\n"
    )
    return header + report(summarize(calls), args.budget_turns, args.parallel)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--parallel", type=int, default=2, help="dialogues at once")
    parser.add_argument(
        "--turns", type=int, default=8, help="turns per measured dialogue"
    )
    parser.add_argument(
        "--budget-turns",
        type=int,
        default=8,
        help="turns per dialogue assumed by the budget",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--judge", action="store_true", help="also time two judge calls"
    )
    parser.add_argument("--agent-model", help="override configs/models.yaml")
    parser.add_argument("--simulator-model", help="override configs/models.yaml")
    parser.add_argument("--judge-model", help="override configs/models.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    print(asyncio.run(measure(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
