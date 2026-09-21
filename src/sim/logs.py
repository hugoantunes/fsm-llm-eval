"""The log file both phases append to, inside the run directory.

``sim run`` and ``sim eval`` write to ``runs/<exp_id>/sim.log``, so a run keeps
its own history and rsync moves it with everything else. What each call sent and
got back belongs to ``llm_calls.jsonl``: this file holds phase boundaries and
what went wrong, which the terminal loses behind the progress bar.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

#: Written under the run directory, next to ``metrics.csv``.
LOG_FILE = "sim.log"

#: Every logger in the package descends from this one.
PACKAGE_LOGGER = "sim"


def configure_run_log(run_dir: Path, *, phase: str) -> logging.Logger:
    """Send the package's records to ``run_dir/sim.log`` and return the logger.

    The file takes INFO and above, which is the phase boundaries plus every
    warning; the terminal keeps the WARNING and above it showed before this
    module existed. Handlers of an earlier call are closed and replaced, so
    configuring twice in one process does not log everything twice.

    Args:
        run_dir: the experiment directory; created when it does not exist yet.
        phase: what to tag every record with, ``run`` or ``eval``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s {phase} %(name)s: %(message)s"
    )
    to_file = logging.FileHandler(run_dir / LOG_FILE, encoding="utf-8")
    to_file.setLevel(logging.INFO)
    to_file.setFormatter(formatter)
    to_terminal = logging.StreamHandler(sys.stderr)
    to_terminal.setLevel(logging.WARNING)
    to_terminal.setFormatter(formatter)

    logger.addHandler(to_file)
    logger.addHandler(to_terminal)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
