"""Smoke tests for the ``sim`` command-line surface specified in T-14a/T-14b."""

from pathlib import Path

import pytest

from helpers import CannedEvalLlm, RunCanned
from helpers import canned_llm_factory as make_canned_llm
from sim import __version__
from sim.__main__ import AGENTS, build_parser, main
from sim.schemas import Manifest


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
    argv = [
        "run",
        "--agent",
        agent,
        "--scenarios",
        "data/scenarios/examples",
        "--exp-id",
        "exp",
    ]

    args = build_parser().parse_args(argv)

    assert args.command == "run"
    assert args.agent == [agent]
    assert args.scenarios == "data/scenarios/examples"
    assert args.exp_id == "exp"
    assert args.runs_dir == "runs"
    assert (args.reps, args.parallel, args.resume) == (1, 1, False)


def test_run_without_agent_selects_both() -> None:
    args = build_parser().parse_args(
        ["run", "--scenarios", "data/scenarios/examples", "--exp-id", "exp"]
    )

    assert args.agent is None


def test_run_rejects_unknown_agent(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(
            ["run", "--agent", "gpt", "--scenarios", "x", "--exp-id", "exp"]
        )

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


def test_eval_writes_metrics_into_the_run_directory(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_canned()

    code = main(["eval", "--run", str(run_dir)], eval_llm=canned_eval_llm)

    assert code == 0
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "metrics_turn.csv").exists()
    assert "0 failed" in capsys.readouterr().out


def test_eval_cli_exits_1_when_the_manifest_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["eval", "--run", str(tmp_path / "ghost")])

    assert code == 1
    assert "manifest" in capsys.readouterr().err


def test_run_writes_into_the_experiment_directory(
    tmp_path: Path, two_scenario_dir: Path
) -> None:
    runs_dir = tmp_path / "runs"

    code = main(
        [
            "run",
            "--scenarios",
            str(two_scenario_dir),
            "--exp-id",
            "exp",
            "--runs-dir",
            str(runs_dir),
            "--reps",
            "1",
            "--parallel",
            "1",
        ],
        llm_factory=make_canned_llm,
    )

    assert code == 0
    exp = runs_dir / "exp"
    manifest = Manifest.model_validate_json(
        (exp / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.n_ok == 4
    assert len(list((exp / "dialogues").glob("*.jsonl"))) == 4


def test_run_with_zero_reps_does_not_create_the_experiment_directory(
    tmp_path: Path, two_scenario_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"

    code = main(
        [
            "run",
            "--scenarios",
            str(two_scenario_dir),
            "--exp-id",
            "ghost",
            "--runs-dir",
            str(runs_dir),
            "--reps",
            "0",
        ],
        llm_factory=make_canned_llm,
    )

    assert code == 1
    assert "reps" in capsys.readouterr().err
    assert not (runs_dir / "ghost").exists()
