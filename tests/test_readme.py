"""Release README is enough to install, run the T-15 pilot, and regenerate analysis."""

from helpers import REPO_ROOT

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

PILOT_SCENARIOS = (
    "happy_path_06",
    "edge_02",
    "edge_19",
    "adversarial_04",
    "adversarial_13",
)


def test_readme_installs_with_uv_just_ollama_env_and_model_digests() -> None:
    assert "just install" in README
    assert "scripts/ollama_env.sh" in README
    assert "just pull-models" in README
    assert "just verify-models" in README


def test_readme_runs_the_t15_pilot_from_frozen_v1() -> None:
    heading = "## Run the T-15 pilot"
    assert heading in README
    section = README.split(heading, 1)[1].split("\n## ", 1)[0]

    assert "data/scenarios/v1" in section
    assert "readme_pilot" in section
    assert "exp_pilot" not in section
    for scenario_id in PILOT_SCENARIOS:
        assert f"--scenario-id {scenario_id}" in section
    assert "--reps 2" in section or " 2 2" in section
    assert "just adherence" in section
    assert "just eval" in section
    assert "data/scenarios/examples" not in section


def _section(heading: str) -> str:
    assert heading in README, f"missing {heading}"
    return README.split(heading, 1)[1].split("\n## ", 1)[0]


def test_readme_documents_full_experiment_and_air_only_fallback() -> None:
    section = _section("## Full experiment")

    assert "Air" in section
    assert "Pro" in section
    assert "run" in section.lower()
    assert "eval" in section.lower()
    assert "fallback" in section.lower()
    assert "runs/exp_final/" in section
    assert "do not replay" in section.lower() or "Do not replay" in section
    remaining = "remaining" in section.lower() and "judge" in section.lower()
    assert not remaining


def test_readme_regenerates_analysis_from_frozen_csvs() -> None:
    section = _section("## Regenerate analysis")

    assert "just analyze" in section
    assert "just figures" in section
    assert "just cases" in section
    assert "just appendices" in section
    assert "do not rerun" in section.lower() or "does not rerun" in section.lower()


def test_readme_embeds_architecture_mermaid() -> None:
    assert "```mermaid" in README
    diagram = README.split("```mermaid", 1)[1].split("```", 1)[0]
    for name in (
        "runPhase",
        "sim_run",
        "frozen_v1",
        "simulated_user",
        "baseline",
        "fsm_agent",
        "engine_and_classifier",
        "runs_exp_id",
        "evalPhase",
        "sim_eval",
        "judge_two_calls",
        "deterministic_evaluators",
        "stage_labeler",
        "analyze_figures_cases",
    ):
        assert name in diagram
