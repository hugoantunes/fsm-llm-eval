"""Tests for the experiment runner of T-14a."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from helpers import CONFIG, EXAMPLES_DIR, FakeLlm
from helpers import canned_llm_factory as make_canned_llm
from sim.config import load_models_config
from sim.fsm import FsmSpec
from sim.io import atomic_write
from sim.kb import KnowledgeBase
from sim.llm import LLM_CALLS_LOG, Chat, LlmCallRecord, LlmClient, LlmError
from sim.runner import (
    AGENTS,
    DialogueJob,
    dialogue_path,
    dialogue_seed,
    hash_dataset,
    iter_jobs,
    run_experiment,
)
from sim.schemas import DialogueLog, Manifest, Scenario

RunCanned = Callable[..., Manifest]


@pytest.fixture
def run_canned(
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    run_dir: Path,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> RunCanned:
    """Run the experiment on the 2x2 pair with a one-turn fake per job."""
    config = load_models_config(CONFIG)

    def _run(**kwargs: Any) -> Manifest:
        params: dict[str, Any] = {
            "scenarios": list(two_example_scenarios),
            "agents": AGENTS,
            "reps": 1,
            "parallel": 1,
            "resume": False,
            "run_dir": run_dir,
            "llm_factory": make_canned_llm,
            "kb": real_kb,
            "fsm": real_fsm,
            "config": config,
            "scenarios_dir": EXAMPLES_DIR,
            "exp_id": "exp",
        }
        params.update(kwargs)
        return run_experiment(**params)

    return _run


def test_jobs_are_ordered_repetition_then_scenario_then_agent(
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    first, second = two_example_scenarios

    jobs = iter_jobs(two_example_scenarios, agents=AGENTS, reps=2)

    assert [(job.repetition, job.scenario.id, job.agent) for job in jobs] == [
        (1, first.id, "baseline"),
        (1, first.id, "fsm"),
        (1, second.id, "baseline"),
        (1, second.id, "fsm"),
        (2, first.id, "baseline"),
        (2, first.id, "fsm"),
        (2, second.id, "baseline"),
        (2, second.id, "fsm"),
    ]


def test_dialogue_seed_is_identical_for_both_agents(
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    jobs = iter_jobs(two_example_scenarios, agents=AGENTS, reps=1)

    seeds = {
        (job.scenario.id, job.repetition, job.agent): dialogue_seed(
            42, job.scenario.id, job.repetition
        )
        for job in jobs
    }

    for scenario in two_example_scenarios:
        assert seeds[(scenario.id, 1, "baseline")] == seeds[(scenario.id, 1, "fsm")]
    assert (
        seeds[(two_example_scenarios[0].id, 1, "baseline")]
        != seeds[(two_example_scenarios[1].id, 1, "baseline")]
    )
    assert dialogue_seed(42, two_example_scenarios[0].id, 1) != dialogue_seed(
        42, two_example_scenarios[0].id, 2
    )


def test_a_dialogue_file_is_written_atomically(tmp_path: Path) -> None:
    path = tmp_path / "dialogues" / "happy_path_01__baseline__rep01.jsonl"

    atomic_write(path, '{"status":"ok"}\n')

    assert path.read_text(encoding="utf-8") == '{"status":"ok"}\n'
    assert list(path.parent.glob("*.tmp")) == []
    atomic_write(path, '{"status":"failed"}\n')
    assert path.read_text(encoding="utf-8") == '{"status":"failed"}\n'
    assert list(path.parent.glob("*.tmp")) == []


def test_resume_skips_a_complete_file_and_retries_a_missing_one(
    two_example_scenarios: tuple[Scenario, Scenario],
    run_dir: Path,
    run_canned: RunCanned,
) -> None:
    jobs = iter_jobs(two_example_scenarios, agents=("baseline",), reps=1)
    done, missing = jobs
    atomic_write(
        dialogue_path(run_dir, done),
        DialogueLog(
            scenario_id=done.scenario.id,
            agent=done.agent,
            repetition=done.repetition,
            seed=0,
            status="ok",
            stop_reason="goal_reached",
        ).model_dump_json()
        + "\n",
    )
    leftover = dialogue_path(run_dir, missing).with_name(
        f"{dialogue_path(run_dir, missing).name}.partial.tmp"
    )
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_text("{", encoding="utf-8")
    played: list[str] = []

    def factory(job: DialogueJob) -> Chat:
        played.append(job.scenario.id)
        return make_canned_llm(job)

    run_canned(agents=("baseline",), resume=True, llm_factory=factory)

    assert played == [missing.scenario.id]
    assert dialogue_path(run_dir, missing).exists()
    log = DialogueLog.model_validate_json(
        dialogue_path(run_dir, missing).read_text(encoding="utf-8")
    )
    assert log.status == "ok"
    assert leftover.exists()


class _FailingLlm(FakeLlm):
    """A Chat that always raises, so one job can fail without aborting the run."""

    def chat(
        self,
        messages: object,
        *,
        role: object,
        caller: object,
        schema: object = None,
        seed: object = None,
        num_predict: object = None,
    ) -> object:
        raise LlmError("the server went away")


def _read_log(run_dir: Path, job: DialogueJob) -> DialogueLog:
    """Load the dialogue JSONL of ``job``."""
    return DialogueLog.model_validate_json(
        dialogue_path(run_dir, job).read_text(encoding="utf-8")
    )


def test_a_failed_dialogue_is_recorded_and_does_not_abort_the_run(
    two_example_scenarios: tuple[Scenario, Scenario],
    run_dir: Path,
    run_canned: RunCanned,
) -> None:
    jobs = iter_jobs(two_example_scenarios, agents=("baseline",), reps=1)
    failing = jobs[0].scenario.id

    def factory(job: DialogueJob) -> Chat:
        if job.scenario.id == failing:
            return _FailingLlm()
        return make_canned_llm(job)

    manifest = run_canned(agents=("baseline",), llm_factory=factory)

    assert (manifest.n_failed, manifest.n_ok) == (1, 1)
    failed = _read_log(run_dir, jobs[0])
    ok = _read_log(run_dir, jobs[1])
    assert failed.status == "failed"
    assert failed.error is not None
    assert "server went away" in failed.error
    assert failed.records == []
    assert ok.status == "ok"
    assert ok.records


def test_two_scenarios_two_agents_one_rep_write_four_dialogue_files(
    two_example_scenarios: tuple[Scenario, Scenario],
    run_dir: Path,
    run_canned: RunCanned,
) -> None:
    manifest = run_canned()
    jobs = iter_jobs(two_example_scenarios, agents=AGENTS, reps=1)

    assert len(jobs) == 4
    assert manifest.n_ok == 4
    dumped = ""
    for job in jobs:
        log = _read_log(run_dir, job)
        assert log.status == "ok"
        assert log.scenario_id == job.scenario.id
        assert log.agent == job.agent
        assert log.repetition == 1
        assert log.seed == dialogue_seed(42, job.scenario.id, job.repetition)
        assert log.records
        record = log.records[0]
        if job.agent == "baseline":
            assert (record.state_before, record.state_after, record.event) == (
                None,
                None,
                None,
            )
        else:
            assert record.state_before is not None
            assert record.state_after is not None
            assert record.event is not None
        dumped += log.model_dump_json()
    for scenario in two_example_scenarios:
        assert scenario.reference_answer not in dumped
        assert scenario.success_criterion not in dumped


def test_parallel_overlaps_dialogues_and_records_throughput(
    run_canned: RunCanned,
) -> None:
    delay_s = 0.2

    def factory(job: DialogueJob) -> Chat:
        return make_canned_llm(job, delay_s=delay_s)

    manifest = run_canned(agents=("baseline",), parallel=2, llm_factory=factory)

    assert manifest.parallel == 2
    assert manifest.n_ok == 2
    assert manifest.elapsed_s > 0
    assert manifest.dialogues_per_hour > 0
    assert manifest.elapsed_s < 3 * delay_s


def test_manifest_records_config_hash_digests_prompts_and_parallel(
    two_example_scenarios: tuple[Scenario, Scenario],
    run_dir: Path,
    run_canned: RunCanned,
) -> None:
    config = load_models_config(CONFIG)
    first, second = two_example_scenarios

    manifest = run_canned()

    assert manifest.num_ctx == config.num_ctx
    assert manifest.ollama_version == config.ollama.version
    assert manifest.dataset_hash == hash_dataset(EXAMPLES_DIR)
    assert manifest.model_names["agent"] == config.models.agent.name
    assert manifest.model_digests["agent"] == config.models.agent.digest
    assert manifest.prompt_versions["baseline"] >= 1
    assert manifest.prompt_versions["fsm_template"] >= 1
    assert manifest.prompt_versions["simulated_user"] >= 1
    assert manifest.parallel == 1
    assert manifest.reps == 1
    assert (manifest.n_llm_calls, manifest.n_llm_cached) == (0, 0)
    assert [(job.repetition, job.scenario_id, job.agent) for job in manifest.jobs] == [
        (1, first.id, "baseline"),
        (1, first.id, "fsm"),
        (1, second.id, "baseline"),
        (1, second.id, "fsm"),
    ]
    saved = Manifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert saved == manifest


def test_manifest_counts_cached_calls_from_the_llm_log(
    run_dir: Path,
    run_canned: RunCanned,
) -> None:
    (run_dir / LLM_CALLS_LOG).write_text(
        "\n".join(
            [
                _llm_log_line(cached=False, prompt_hash="aa"),
                _llm_log_line(cached=True, prompt_hash="aa"),
                _llm_log_line(cached=False, prompt_hash="bb"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = run_canned(agents=("baseline",))

    assert (manifest.n_llm_calls, manifest.n_llm_cached) == (3, 1)


def _llm_log_line(*, cached: bool, prompt_hash: str) -> str:
    """One valid ``llm_calls.jsonl`` row for the manifest counter."""
    return LlmCallRecord(
        timestamp="2026-09-09T00:00:00+00:00",
        caller="simulated_user",
        role="simulator",
        model="small-model",
        prompt_hash=prompt_hash,
        messages=[],
        text="{}",
        prompt_tokens=1,
        output_tokens=1,
        latency_s=0.1,
        cached=cached,
        attempts=0 if cached else 1,
    ).model_dump_json()


def test_progress_is_reported_once_per_finished_dialogue(
    run_canned: RunCanned,
) -> None:
    seen: list[tuple[int, int]] = []

    run_canned(on_progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


# --- Integration: a real Ollama with the models of configs/models.yaml -------


@pytest.mark.integration
def test_two_example_scenarios_run_end_to_end(
    two_example_scenarios: tuple[Scenario, Scenario],
    two_scenario_dir: Path,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    real_client: LlmClient,
) -> None:
    config = load_models_config(CONFIG)

    manifest = run_experiment(
        scenarios=list(two_example_scenarios),
        agents=AGENTS,
        reps=1,
        parallel=1,
        run_dir=run_dir,
        llm_factory=lambda _job: real_client,
        kb=real_kb,
        fsm=real_fsm,
        config=config,
        scenarios_dir=two_scenario_dir,
        exp_id="exp",
    )

    assert manifest.n_ok == 4
    assert manifest.n_failed == 0
    for job in iter_jobs(two_example_scenarios, agents=AGENTS, reps=1):
        log = _read_log(run_dir, job)
        assert log.status == "ok"
        assert log.records
        assert all(record.agent_reply.strip() for record in log.records)
        if job.agent == "fsm":
            assert log.records[-1].state_after is not None
        else:
            assert log.records[-1].state_after is None
