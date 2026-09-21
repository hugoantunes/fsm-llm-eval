"""Command-line entry point: ``python -m sim {run,eval}``.

``sim run`` (T-14a) plays dialogues into ``runs/<exp_id>/``. ``sim eval``
(T-14b) scores those logs into ``metrics.csv`` and ``metrics_turn.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from sim import __version__
from sim.adjudication import (
    CONTRACT_FALSE_POSITIVES_RELATIVE,
    resolve_include_failed_from,
)
from sim.config import ModelsConfig, load_models_config
from sim.eval import EvalError, EvalProgress, evaluate_run, sidecar_out_dir
from sim.fsm import load_fsm
from sim.kb import load_kb
from sim.llm import LLM_CALLS_LOG, Chat, LlmClient
from sim.logs import PACKAGE_LOGGER, configure_run_log
from sim.runner import AGENTS, LlmFactory, RunnerError, run_experiment, select_scenarios
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
    run.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        metavar="ID",
        help="play only this scenario; repeat to name several. Omit to play all",
    )

    ev = subparsers.add_parser(
        "eval",
        help="read runs/<exp_id>/, run the evaluators and write metrics.csv",
    )
    ev.add_argument("--run", required=True, help="experiment directory (runs/<exp_id>)")
    ev.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "dialogues scored in parallel (match OLLAMA_NUM_PARALLEL); "
            "the bar counts completed dialogues, so an uncached job can sit "
            "at N/total while active workers are still inside judge/labeler calls"
        ),
    )
    ev.add_argument(
        "--include-failed-from",
        type=Path,
        nargs="?",
        const=CONTRACT_FALSE_POSITIVES_RELATIVE,
        default=None,
        metavar="ALLOWLIST",
        help=(
            "JSON allowlist of failed dialogue ids to score in a sidecar eval; "
            "omit the path to use <run>/adjudication/contract_false_positives.json; "
            "ordinary sim eval still skips every failed log"
        ),
    )
    ev.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "directory for sidecar CSVs; defaults to a sibling named "
            "<run>_semantic; must be distinct from --run"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    llm_factory: LlmFactory | None = None,
    eval_llm: Chat | None = None,
) -> int:
    """Parse ``argv`` and dispatch to the requested subcommand.

    ``llm_factory`` is the test seam for ``run`` (one fake per dialogue job).
    ``eval_llm`` is the seam for ``eval`` (one client for the judge and labeler).
    Production builds one :class:`LlmClient` on the experiment cache.

    Returns the process exit code.
    """
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        return _eval(args, llm=eval_llm)
    return _run(args, llm_factory=llm_factory)


def _run(args: argparse.Namespace, *, llm_factory: LlmFactory | None) -> int:
    """Load config and data, play the jobs, write ``runs/<exp_id>/``."""
    config = load_models_config()
    kb = load_kb()
    fsm = load_fsm()
    scenarios_dir = Path(args.scenarios)
    scenarios = load_scenarios(scenarios_dir, kb=kb, fsm=fsm)
    run_dir = Path(args.runs_dir) / args.exp_id
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
        if args.scenario_ids:
            scenarios = select_scenarios(scenarios, args.scenario_ids)
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
    configure_run_log(run_dir, phase="run").info(
        "wrote %d dialogue(s) (%d failed, %d skipped, %d/%d llm cache hits)",
        manifest.n_ok,
        manifest.n_failed,
        manifest.n_skipped,
        manifest.n_llm_cached,
        manifest.n_llm_calls,
    )
    print(
        f"sim: wrote {manifest.n_ok} dialogue(s) to {run_dir} "
        f"({manifest.n_failed} failed, {manifest.n_skipped} skipped, "
        f"{manifest.n_llm_cached}/{manifest.n_llm_calls} llm cache hits, "
        f"{manifest.dialogues_per_hour:.1f} dlg/h)",
        file=sys.stdout,
    )
    return 1 if manifest.n_failed else 0


#: While a worker is inside ``score()``, omit rate/ETA. A cache burst would
#: otherwise freeze the last frame at hundreds of dlg/s and remaining 00:00.
_ACTIVE_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}"
_IDLE_BAR_FORMAT = "{l_bar}{bar}{r_bar}"
_ELAPSED_REFRESH_S = 1.0


def apply_eval_progress(bar: tqdm, progress: EvalProgress) -> None:
    """Update ``bar`` from an eval snapshot.

    ``bar.n`` is completed dialogue evaluations. ``set_postfix_str`` redraws
    when ``active`` changes so an in-flight judge/labeler call is visible even
    if ``n`` is unchanged. The worker emits the snapshot before
    ``as_completed()`` observes the Future.
    """
    if bar.total != progress.total:
        bar.total = progress.total
    postfix = f"active={progress.active}"
    if progress.active:
        postfix += " | evaluating"
        bar.bar_format = _ACTIVE_BAR_FORMAT
    else:
        bar.bar_format = _IDLE_BAR_FORMAT
    delta = progress.completed - bar.n
    if delta > 0:
        bar.set_postfix_str(postfix, refresh=False)
        bar.update(delta)
    else:
        bar.set_postfix_str(postfix)


def _eval_progress_callback(
    bar: tqdm,
) -> tuple[Callable[[EvalProgress], None], threading.Event]:
    """Return a lock-serializing callback and an event that stops elapsed redraws.

    Workers emit snapshots; this lock is the only tqdm writer. A 1s refresh
    keeps ``elapsed`` moving during a long judge/labeler call. It does not poll
    Ollama or ``llm_calls.jsonl``.
    """
    lock = threading.Lock()
    stop = threading.Event()

    def on_progress(progress: EvalProgress) -> None:
        with lock:
            apply_eval_progress(bar, progress)

    def refresh_elapsed() -> None:
        while not stop.wait(_ELAPSED_REFRESH_S):
            with lock:
                if not bar.disable:
                    bar.refresh()

    threading.Thread(
        target=refresh_elapsed, name="eval-tqdm-elapsed", daemon=True
    ).start()
    return on_progress, stop


def _eval(args: argparse.Namespace, *, llm: Chat | None) -> int:
    """Score the dialogues under ``--run`` and write the two CSVs.

    The tqdm counter is completed dialogue evaluations. ``active`` in the
    postfix means workers may still be inside judge/labeler LLM calls, so
    ``--parallel 1`` can sit at ``N/total`` for a long time on an uncached
    dialogue. Snapshots are emitted in the worker before ``as_completed()``
    observes the Future. Elapsed time keeps redrawing while that happens;
    rate/ETA are omitted while ``active > 0`` because a cache burst would
    otherwise freeze the last frame at remaining 00:00.
    """
    run_dir = Path(args.run)
    config = load_models_config()
    kb = load_kb()
    fsm = load_fsm()
    client = llm or _client(config, run_dir)
    log = configure_run_log(run_dir, phase="eval")
    bar = tqdm(
        total=0,
        unit="dlg",
        disable=not sys.stderr.isatty(),
        desc=run_dir.name,
        miniters=1,
        dynamic_ncols=True,
    )
    on_progress, stop_elapsed = _eval_progress_callback(bar)

    try:
        with logging_redirect_tqdm(loggers=[logging.getLogger(PACKAGE_LOGGER)]):
            result = evaluate_run(
                run_dir,
                llm=client,
                kb=kb,
                fsm=fsm,
                config=config,
                parallel=args.parallel,
                on_progress=on_progress,
                include_failed_from=resolve_include_failed_from(
                    args.include_failed_from, run_dir
                ),
                out_dir=args.out,
            )
    except EvalError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    finally:
        stop_elapsed.set()
        bar.close()
    log.info(
        "scored %d dialogue(s) (%d failed, %d unscored)",
        result.n_scored,
        result.n_failed,
        result.n_unscored,
    )
    if args.include_failed_from is not None:
        destination = sidecar_out_dir(run_dir) if args.out is None else args.out
    else:
        destination = run_dir
    extra = (
        f", {result.n_included_failed} included from allowlist"
        if result.n_included_failed
        else ""
    )
    print(
        f"sim: scored {result.n_scored} dialogue(s) in {destination} "
        f"({result.n_failed} failed, {result.n_unscored} unscored{extra}, "
        f"metrics.csv, metrics_turn.csv)",
        file=sys.stdout,
    )
    return 1 if result.n_unscored else 0


def _client(config: ModelsConfig, run_dir: Path) -> LlmClient:
    """One cached client on the experiment directory, shared by run and eval."""
    return LlmClient(
        config,
        cache_dir=run_dir / "cache",
        log_path=run_dir / LLM_CALLS_LOG,
    )


def _shared_client(config: ModelsConfig, run_dir: Path) -> LlmFactory:
    """One cached client for the process, shared across the thread pool."""
    client = _client(config, run_dir)
    return lambda _job: client


if __name__ == "__main__":
    sys.exit(main())
