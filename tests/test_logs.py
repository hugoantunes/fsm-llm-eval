"""Tests for the per-run log file both phases append to."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from sim.logs import LOG_FILE, configure_run_log


@pytest.fixture(autouse=True)
def restore_sim_logger() -> Iterator[None]:
    """Leave the ``sim`` logger as the suite found it, handlers included."""
    logger = logging.getLogger("sim")
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate

    yield

    for handler in list(logger.handlers):
        handler.close()
    logger.handlers = handlers
    logger.level = level
    logger.propagate = propagate


def test_configure_run_log_writes_warnings_into_the_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "exp"

    configure_run_log(run_dir, phase="eval")
    logging.getLogger("sim.llm").warning("attempt 1 of 3 failed")

    text = (run_dir / LOG_FILE).read_text(encoding="utf-8")

    assert "attempt 1 of 3 failed" in text
    assert "eval" in text
    assert "WARNING" in text
    assert "sim.llm" in text


def test_configure_run_log_records_the_phase_boundary(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp"

    logger = configure_run_log(run_dir, phase="run")
    logger.info("done")

    text = (run_dir / LOG_FILE).read_text(encoding="utf-8")

    assert "done" in text
    assert "run" in text


def test_a_second_phase_appends_instead_of_replacing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp"

    configure_run_log(run_dir, phase="run").info("played 20 dialogues")
    configure_run_log(run_dir, phase="eval").info("scored 19 dialogues")

    lines = [
        line
        for line in (run_dir / LOG_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(lines) == 2
    assert "played 20 dialogues" in lines[0]
    assert "scored 19 dialogues" in lines[1]


def test_configuring_twice_does_not_double_every_record(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp"

    configure_run_log(run_dir, phase="run")
    configure_run_log(run_dir, phase="eval")
    logging.getLogger("sim.llm").warning("said once")

    text = (run_dir / LOG_FILE).read_text(encoding="utf-8")

    assert text.count("said once") == 1


def test_the_terminal_still_sees_warnings_and_not_info(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "runs" / "exp"

    logger = configure_run_log(run_dir, phase="eval")
    logger.info("a phase boundary")
    logging.getLogger("sim.llm").warning("a retry")

    err = capsys.readouterr().err

    assert "a retry" in err
    assert "a phase boundary" not in err
