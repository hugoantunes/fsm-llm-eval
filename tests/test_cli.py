"""Smoke tests for the ``sim`` command-line surface specified in T-14a/T-14b."""

from io import StringIO
from pathlib import Path

import pytest
from tqdm import tqdm

from helpers import CannedEvalLlm, RunCanned
from helpers import canned_llm_factory as make_canned_llm
from sim import __version__
from sim.__main__ import apply_eval_progress, build_parser, main
from sim.adjudication import CONTRACT_FALSE_POSITIVES_RELATIVE
from sim.eval import EvalProgress
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


RUN_ARGV = ["run", "--scenarios", "data/scenarios/examples", "--exp-id", "exp"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(
            [*RUN_ARGV, "--agent", "baseline"],
            {
                "command": "run",
                "agent": ["baseline"],
                "scenarios": "data/scenarios/examples",
                "exp_id": "exp",
                "runs_dir": "runs",
                "reps": 1,
                "parallel": 1,
                "resume": False,
            },
            id="run_with_agent_uses_defaults",
        ),
        pytest.param(RUN_ARGV, {"agent": None}, id="run_without_agent_selects_both"),
        pytest.param(
            ["eval", "--run", "runs/exp_pilot"],
            {"command": "eval", "run": "runs/exp_pilot", "parallel": 1},
            id="eval_parses_run_directory",
        ),
        pytest.param(
            ["eval", "--run", "runs/exp_pilot", "--parallel", "2"],
            {"parallel": 2},
            id="eval_parses_parallel",
        ),
        pytest.param(
            [
                "eval",
                "--run",
                "runs/exp_final",
                "--include-failed-from",
                "runs/exp_final/adjudication/contract_false_positives.json",
                "--out",
                "runs/exp_final_semantic",
            ],
            {
                "include_failed_from": Path(
                    "runs/exp_final/adjudication/contract_false_positives.json"
                ),
                "out": Path("runs/exp_final_semantic"),
            },
            id="eval_parses_sidecar_inclusion_flags",
        ),
        pytest.param(
            ["eval", "--run", "runs/exp_final", "--include-failed-from"],
            {"include_failed_from": CONTRACT_FALSE_POSITIVES_RELATIVE, "out": None},
            id="eval_include_failed_from_defaults_to_the_contract_path",
        ),
    ],
)
def test_parser_maps_argv_to_fields(
    argv: list[str], expected: dict[str, object]
) -> None:
    args = build_parser().parse_args(argv)

    assert {name: getattr(args, name) for name in expected} == expected


@pytest.mark.parametrize(
    ("argv", "stderr_substrings"),
    [
        pytest.param([], (), id="subcommand_is_required"),
        pytest.param(
            ["run", "--agent", "gpt", "--scenarios", "x", "--exp-id", "exp"],
            ("--agent",),
            id="run_rejects_unknown_agent",
        ),
        pytest.param(["eval"], (), id="eval_requires_run_directory"),
    ],
)
def test_parser_exits_2_on_invalid_argv(
    argv: list[str],
    stderr_substrings: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    for substring in stderr_substrings:
        assert substring in err


def test_eval_cli_exits_1_on_a_parallel_below_one(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_canned()

    code = main(
        ["eval", "--run", str(run_dir), "--parallel", "0"], eval_llm=canned_eval_llm
    )

    assert code == 1
    assert "--parallel must be at least 1" in capsys.readouterr().err


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
    assert (run_dir / "sim.log").exists()
    out = capsys.readouterr().out
    assert "0 failed" in out
    assert "0 unscored" in out


def test_eval_cli_reports_the_dialogues_the_judge_could_not_grade(
    run_canned: RunCanned,
    run_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_canned()
    llm = CannedEvalLlm(invalid_facts_marker="Hello from support.")

    code = main(["eval", "--run", str(run_dir)], eval_llm=llm)

    assert code == 1
    assert "4 unscored" in capsys.readouterr().out
    assert "unscored" in (run_dir / "sim.log").read_text(encoding="utf-8")


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


def test_run_with_scenario_id_plays_only_the_named_scenarios(
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
            "--scenario-id",
            "happy_path_01",
        ],
        llm_factory=make_canned_llm,
    )

    assert code == 0
    exp = runs_dir / "exp"
    manifest = Manifest.model_validate_json(
        (exp / "manifest.json").read_text(encoding="utf-8")
    )
    assert [(job.scenario_id, job.agent) for job in manifest.jobs] == [
        ("happy_path_01", "baseline"),
        ("happy_path_01", "fsm"),
    ]
    names = sorted(path.name for path in (exp / "dialogues").glob("*.jsonl"))
    assert names == [
        "happy_path_01__baseline__rep01.jsonl",
        "happy_path_01__fsm__rep01.jsonl",
    ]


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


def test_eval_progress_bar_counts_completed_and_shows_active() -> None:
    bar = tqdm(file=StringIO(), total=0)

    apply_eval_progress(bar, EvalProgress(completed=0, active=1, total=10))

    assert bar.n == 0
    assert bar.total == 10
    assert "active=1" in bar.postfix
    assert "evaluating" in bar.postfix
    assert "{rate_fmt}" not in (bar.bar_format or "")
    assert "{remaining}" not in (bar.bar_format or "")

    apply_eval_progress(bar, EvalProgress(completed=1, active=0, total=10))

    assert bar.n == 1
    assert "active=0" in bar.postfix
    assert "evaluating" not in bar.postfix
    assert bar.bar_format == "{l_bar}{bar}{r_bar}"
    bar.close()
