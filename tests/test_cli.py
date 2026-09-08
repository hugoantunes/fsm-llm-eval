"""Smoke tests for the ``sim`` command-line surface specified in T-14a/T-14b."""

import pytest

from sim import __version__
from sim.__main__ import AGENTS, build_parser, main


def test_version_flag_reports_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"sim {__version__}"


def test_help_lists_both_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "run" in out
    assert "eval" in out


def test_subcommand_is_required() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args([])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("agent", AGENTS)
def test_run_accepts_each_agent_with_defaults(agent: str) -> None:
    argv = ["run", "--agent", agent, "--scenarios", "data/scenarios/examples"]

    args = build_parser().parse_args(argv)

    assert args.command == "run"
    assert args.agent == agent
    assert args.scenarios == "data/scenarios/examples"
    assert (args.reps, args.parallel, args.resume) == (1, 1, False)


def test_run_rejects_unknown_agent(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["run", "--agent", "gpt", "--scenarios", "x"])

    assert exc_info.value.code == 2
    assert "--agent" in capsys.readouterr().err


def test_eval_requires_run_directory() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["eval"])

    assert exc_info.value.code == 2


def test_eval_parses_run_directory() -> None:
    args = build_parser().parse_args(["eval", "--run", "runs/exp_pilot"])

    assert args.command == "eval"
    assert args.run == "runs/exp_pilot"


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--agent", "baseline", "--scenarios", "data/scenarios/examples"],
        ["eval", "--run", "runs/exp_pilot"],
    ],
)
def test_unimplemented_subcommands_fail_loudly(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 1
    assert "not implemented" in capsys.readouterr().err
