"""Command-line entry point: ``python -m sim {run,eval}``.

``sim run`` (T-14a) plays dialogues into ``runs/<exp_id>/``. ``sim eval``
(T-14b) is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from sim import __version__
from sim.config import ModelsConfig, load_models_config
from sim.fsm import load_fsm
from sim.kb import load_kb
from sim.llm import LLM_CALLS_LOG, LlmClient
from sim.runner import AGENTS, LlmFactory, RunnerError, run_experiment
from sim.schemas import load_scenarios

__all__ = ["AGENTS", "build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the ``sim`` parser with the ``run`` and ``eval`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="sim",
        description="Simulation and evaluation of LLM agents (FSM vs. plain prompt).",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="{run,eval}"
    )

    run = subparsers.add_parser(
        "run",
        help="run dialogues (agent x scenarios x repetitions) into runs/<exp_id>/",
    )
    run.add_argument(
        "--agent",
        choices=AGENTS,
        action="append",
        help="agent to run; omit to run both, in baseline then fsm order",
    )
    run.add_argument(
        "--scenarios", required=True, help="directory with the scenarios (JSONL)"
    )
    run.add_argument("--exp-id", required=True, help="experiment id under --runs-dir")
    run.add_argument(
        "--runs-dir",
        default="runs",
        help="parent directory for experiment folders (default: runs)",
    )
    run.add_argument("--reps", type=int, default=1, help="repetitions per scenario (K)")
    run.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="dialogues in parallel (match OLLAMA_NUM_PARALLEL)",
    )
    run.add_argument(
        "--resume", action="store_true", help="skip dialogues already written"
    )

    ev = subparsers.add_parser(
        "eval",
        help="read runs/<exp_id>/, run the evaluators and write metrics.csv",
    )
    ev.add_argument("--run", required=True, help="experiment directory (runs/<exp_id>)")
    return parser


def main(
    argv: Sequence[str] | None = None, *, llm_factory: LlmFactory | None = None
) -> int:
    """Parse ``argv`` and dispatch to the requested subcommand.

    ``llm_factory`` is a test seam: production builds one :class:`LlmClient`
    shared by every dialogue.

    Returns the process exit code.
    """
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        print(
            "sim: 'eval' is not implemented yet (see T-14b in TICKETS.md)",
            file=sys.stderr,
        )
        return 1
    return _run(args, llm_factory=llm_factory)


def _run(args: argparse.Namespace, *, llm_factory: LlmFactory | None) -> int:
    """Load config and data, play the jobs, write ``runs/<exp_id>/``."""
    config = load_models_config()
    kb = load_kb()
    fsm = load_fsm(kb=kb)
    scenarios_dir = Path(args.scenarios)
    scenarios = load_scenarios(scenarios_dir, kb=kb, fsm=fsm)
    run_dir = Path(args.runs_dir) / args.exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = llm_factory or _shared_client(config, run_dir)
    agents = args.agent or list(AGENTS)
    bar = tqdm(
        total=0,
        unit="dlg",
        disable=not sys.stderr.isatty(),
        desc=args.exp_id,
    )

    def on_progress(done: int, total: int) -> None:
        if bar.total != total:
            bar.total = total
        bar.n = done
        bar.refresh()

    try:
        manifest = run_experiment(
            scenarios=scenarios,
            agents=agents,
            reps=args.reps,
            parallel=args.parallel,
            resume=args.resume,
            run_dir=run_dir,
            llm_factory=factory,
            kb=kb,
            fsm=fsm,
            config=config,
            scenarios_dir=scenarios_dir,
            exp_id=args.exp_id,
            on_progress=on_progress,
        )
    except RunnerError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    finally:
        bar.close()
    print(
        f"sim: wrote {manifest.n_ok} dialogue(s) to {run_dir} "
        f"({manifest.n_failed} failed, {manifest.n_skipped} skipped, "
        f"{manifest.n_llm_cached}/{manifest.n_llm_calls} llm cache hits, "
        f"{manifest.dialogues_per_hour:.1f} dlg/h)",
        file=sys.stdout,
    )
    return 1 if manifest.n_failed else 0


def _shared_client(config: ModelsConfig, run_dir: Path) -> LlmFactory:
    """One cached client for the process, shared across the thread pool."""
    client = LlmClient(
        config,
        cache_dir=run_dir / "cache",
        log_path=run_dir / LLM_CALLS_LOG,
    )
    return lambda _job: client


if __name__ == "__main__":
    sys.exit(main())
