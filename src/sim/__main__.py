"""Command-line entry point: ``python -m sim {run,eval}``.

Only the interface specified in TICKETS.md is defined here. The subcommands are
implemented by the runner (``sim run``, T-14a) and the evaluation pipeline
(``sim eval``, T-14b).
"""

import argparse
import sys
from collections.abc import Sequence

from sim import __version__

AGENTS = ("baseline", "fsm")


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
    run.add_argument("--agent", choices=AGENTS, required=True, help="agent to run")
    run.add_argument(
        "--scenarios", required=True, help="directory with the scenarios (JSONL)"
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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the requested subcommand.

    Returns the process exit code.
    """
    args = build_parser().parse_args(argv)
    print(
        f"sim: '{args.command}' is not implemented yet (see T-14a/T-14b in TICKETS.md)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
