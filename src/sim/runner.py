"""Run the experiment: scenarios x agents x repetitions into ``runs/<exp_id>/``.

The dialogue loop of T-11 is unchanged. This module is the CLI, the job list
(repetition then scenario then agent), the JSONL per dialogue, ``--resume``,
the thread pool and the manifest. One new agent instance per dialogue, so FSM
state never crosses the pair boundary.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from sim import __version__
from sim.agents import Agent, AgentError, BaselineAgent, FsmAgent
from sim.config import ROLES, ModelsConfig
from sim.dialogue import DialogueError, run_dialogue
from sim.events import EventError
from sim.fsm import DEFAULT_FSM_DIR, FsmError, FsmSpec
from sim.io import atomic_write
from sim.kb import KnowledgeBase
from sim.llm import LLM_CALLS_LOG, Chat, LlmCallRecord, LlmError
from sim.prompts import DEFAULT_PROMPTS_DIR, load_prompt_versions
from sim.schemas import DialogueLog, JobRef, Manifest, Scenario, dialogue_filename
from sim.user import SimulatedUser, SimulatedUserGenerationError, UserError

#: The two agents under comparison, in the order the inner loop always emits.
AGENTS = ("baseline", "fsm")

#: SHA-256 of ``data/scenarios/v1/*.jsonl`` (DECISOES.md 2026-09-10). One
#: literal; ``tests.helpers`` re-exports it. ``just check`` fails if the files
#: move.
FROZEN_V1_HASH = "0778a90110e6dd67685c1e6768934483cb4405213dad35ec172f3ddcf21d47c9"

Progress = Callable[[int, int], None]
LlmFactory = Callable[["DialogueJob"], Chat]


class RunnerError(RuntimeError):
    """The run cannot start; the message says what to fix."""


@dataclass(frozen=True)
class DialogueJob:
    """One (repetition, scenario, agent) the runner will play or skip."""

    scenario: Scenario
    agent: str
    repetition: int


def select_scenarios(
    scenarios: Sequence[Scenario], ids: Sequence[str]
) -> list[Scenario]:
    """Keep the named scenarios in dataset order, not the order of ``ids``."""
    known = {scenario.id for scenario in scenarios}
    unknown = [scenario_id for scenario_id in ids if scenario_id not in known]
    if unknown:
        raise RunnerError(
            f"unknown scenario {unknown[0]!r}. Choose an id from the loaded "
            f"dataset, or omit --scenario-id to run every scenario"
        )
    wanted = set(ids)
    return [scenario for scenario in scenarios if scenario.id in wanted]


def iter_jobs(
    scenarios: Sequence[Scenario],
    *,
    agents: Sequence[str] = AGENTS,
    reps: int,
) -> list[DialogueJob]:
    """Build the schedule: repetition, then scenario, then agent.

    After repetition K the paired dataset is complete; extra repetitions are
    incremental (cut 1a). Agents are always emitted in :data:`AGENTS` order,
    even if the caller listed them backwards.
    """
    unknown = sorted({name for name in agents if name not in AGENTS})
    if unknown:
        raise RunnerError(
            f"unknown agent {unknown[0]!r}. Choose one of {AGENTS}, or omit "
            f"--agent to run both"
        )
    selected = [name for name in AGENTS if name in agents]
    return [
        DialogueJob(scenario=scenario, agent=agent, repetition=repetition)
        for repetition in range(1, reps + 1)
        for scenario in scenarios
        for agent in selected
    ]


def dialogue_seed(base: int, scenario_id: str, repetition: int) -> int:
    """Derive the per-dialogue seed from the base, scenario and repetition.

    The agent name is not in the key: baseline and FSM of the same
    (scenario, repetition) share one seed, which is the pairing.
    """
    digest = hashlib.sha256(f"{base}:{scenario_id}:{repetition}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def dialogue_path(run_dir: Path, job: DialogueJob) -> Path:
    """Return the JSONL path of ``job`` under ``run_dir``."""
    return (
        run_dir
        / "dialogues"
        / dialogue_filename(job.scenario.id, job.agent, job.repetition)
    )


def hash_dataset(directory: Path) -> str:
    """SHA-256 the scenario files in ``directory``, in sorted path order."""
    return _hash_paths(
        (path.name, path.read_bytes()) for path in sorted(directory.glob("*.jsonl"))
    )


def hash_fsm(directory: Path) -> str:
    """SHA-256 the machine and its state packages, in sorted path order."""
    paths = [directory / "machine.yaml", *sorted((directory / "states").glob("*.md"))]
    return _hash_paths(
        (path.relative_to(directory).as_posix(), path.read_bytes()) for path in paths
    )


def _hash_paths(parts: Iterable[tuple[str, bytes]]) -> str:
    """SHA-256 ``(name, content)`` pairs so two hashers share one encoding."""
    digest = hashlib.sha256()
    for name, content in parts:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()


def count_llm_log(path: Path) -> tuple[int, int]:
    """Return ``(n_calls, n_cached)`` from ``llm_calls.jsonl``.

    A missing file is ``(0, 0)``: canned tests never write the log. Empty
    lines are skipped so a trailing newline does not become a parse error.
    """
    if not path.exists():
        return 0, 0
    n_calls = 0
    n_cached = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = LlmCallRecord.model_validate_json(line)
        n_calls += 1
        if record.cached:
            n_cached += 1
    return n_calls, n_cached


def run_experiment(
    *,
    scenarios: Sequence[Scenario],
    agents: Sequence[str] = AGENTS,
    reps: int,
    parallel: int = 1,
    resume: bool = False,
    run_dir: Path,
    llm_factory: LlmFactory,
    kb: KnowledgeBase,
    fsm: FsmSpec,
    config: ModelsConfig,
    scenarios_dir: Path,
    exp_id: str,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    fsm_dir: Path = DEFAULT_FSM_DIR,
    on_progress: Progress | None = None,
) -> Manifest:
    """Play every job, write one JSONL per dialogue, and return the manifest.

    A job whose final file already exists is skipped when ``resume`` is true.
    A leftover ``.tmp`` is not a complete file. Operational failures become a
    ``status=failed`` log and do not abort the rest of the run.
    """
    if parallel < 1:
        raise RunnerError(f"--parallel must be at least 1, not {parallel}")
    if reps < 1:
        raise RunnerError(f"--reps must be at least 1, not {reps}")
    seed_base = _seed_base(config)
    selected = [name for name in AGENTS if name in agents]
    jobs = iter_jobs(scenarios, agents=agents, reps=reps)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n_ok = n_failed = n_skipped = 0
    done = 0

    def work(job: DialogueJob) -> DialogueLog | None:
        path = dialogue_path(run_dir, job)
        if resume and path.exists():
            return None
        log = _play(
            job,
            llm_factory(job),
            kb=kb,
            fsm=fsm,
            seed_base=seed_base,
            prompts_dir=prompts_dir,
            fsm_dir=fsm_dir,
        )
        atomic_write(path, log.model_dump_json() + "\n")
        return log

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for future in as_completed(futures):
            log = future.result()
            if log is None:
                n_skipped += 1
            elif log.status == "ok":
                n_ok += 1
            else:
                n_failed += 1
            done += 1
            if on_progress is not None:
                on_progress(done, len(jobs))

    elapsed_s = time.perf_counter() - started
    played = n_ok + n_failed
    n_llm_calls, n_llm_cached = count_llm_log(run_dir / LLM_CALLS_LOG)
    manifest = Manifest(
        exp_id=exp_id,
        package_version=__version__,
        ollama_version=config.ollama.version,
        num_ctx=config.num_ctx,
        dataset_hash=hash_dataset(scenarios_dir),
        fsm_hash=hash_fsm(fsm_dir),
        scenarios_dir=str(scenarios_dir),
        prompt_versions=load_prompt_versions(prompts_dir),
        model_names={role: config.spec(role).name for role in ROLES},
        model_digests={role: config.spec(role).digest for role in ROLES},
        agents=selected,
        reps=reps,
        parallel=parallel,
        seed_base=seed_base,
        jobs=[
            JobRef(
                scenario_id=job.scenario.id,
                agent=job.agent,
                repetition=job.repetition,
            )
            for job in jobs
        ],
        elapsed_s=elapsed_s,
        dialogues_per_hour=(played / elapsed_s * 3600) if elapsed_s else 0.0,
        n_ok=n_ok,
        n_failed=n_failed,
        n_skipped=n_skipped,
        n_llm_calls=n_llm_calls,
        n_llm_cached=n_llm_cached,
    )
    atomic_write(run_dir / "manifest.json", manifest.model_dump_json() + "\n")
    return manifest


def _play(
    job: DialogueJob,
    llm: Chat,
    *,
    kb: KnowledgeBase,
    fsm: FsmSpec,
    seed_base: int,
    prompts_dir: Path,
    fsm_dir: Path,
) -> DialogueLog:
    """Run one dialogue and return its log, catching operational failures."""
    seed = dialogue_seed(seed_base, job.scenario.id, job.repetition)
    agent = _make_agent(
        job.agent,
        llm,
        kb=kb,
        fsm=fsm,
        seed=seed,
        prompts_dir=prompts_dir,
        fsm_dir=fsm_dir,
    )
    user = SimulatedUser(
        llm,
        scenario=job.scenario,
        data_fields=kb.user_data_fields,
        prompts_dir=prompts_dir,
        seed=seed,
    )
    try:
        result = run_dialogue(agent, user, max_turns=job.scenario.max_turns)
        if result.stop_reason == "max_turns" and not user.progress.complete:
            diagnostics = user.progress.active_beat_diagnostics()
            metadata = dict(diagnostics) if diagnostics is not None else {}
            return DialogueLog(
                scenario_id=job.scenario.id,
                agent=job.agent,
                repetition=job.repetition,
                seed=seed,
                status="failed",
                error="the dialogue reached max_turns with an incomplete active beat",
                records=result.records,
                termination_reason="max_turns",
                active_beat_complete=False,
                goal_reached_seen=result.goal_reached_seen,
                goal_reached_at_turn=result.goal_reached_at_turn,
                script_complete_at_goal_reached=result.script_complete_at_goal_reached,
                failure_kind="simulation",
                failure_reason="max_turns_with_incomplete_beat",
                failure_metadata=metadata,
            )
    except (
        LlmError,
        AgentError,
        UserError,
        DialogueError,
        EventError,
        FsmError,
    ) as failure:
        kind, reason, metadata = _classify_failure(failure)
        return DialogueLog(
            scenario_id=job.scenario.id,
            agent=job.agent,
            repetition=job.repetition,
            seed=seed,
            status="failed",
            error=str(failure),
            records=list(getattr(failure, "records", [])),
            failure_kind=kind,
            failure_reason=reason,
            failure_metadata=metadata,
        )
    return DialogueLog(
        scenario_id=job.scenario.id,
        agent=job.agent,
        repetition=job.repetition,
        seed=seed,
        status="ok",
        stop_reason=result.stop_reason,
        termination_reason=result.stop_reason,
        active_beat_complete=user.progress.complete,
        goal_reached_seen=result.goal_reached_seen,
        goal_reached_at_turn=result.goal_reached_at_turn,
        script_complete_at_goal_reached=result.script_complete_at_goal_reached,
        records=result.records,
    )


def _classify_failure(failure: Exception) -> tuple[str, str, dict[str, object]]:
    """Return ``(failure_kind, failure_reason, failure_metadata)``."""
    wrapped = failure.__cause__ if isinstance(failure, DialogueError) else None
    if isinstance(failure, SimulatedUserGenerationError) or isinstance(
        wrapped, SimulatedUserGenerationError
    ):
        error = (
            failure if isinstance(failure, SimulatedUserGenerationError) else wrapped
        )
        assert isinstance(error, SimulatedUserGenerationError)
        metadata: dict[str, object] = {
            "retry_count": error.attempts,
            "invalid_reason": error.invalid_reason,
        }
        if error.beat_diagnostics is not None:
            metadata.update(error.beat_diagnostics)
        return ("instrument", "invalid_candidate_retry_exhausted", metadata)
    return ("operational", "runtime_error", {})


def _make_agent(
    name: str,
    llm: Chat,
    *,
    kb: KnowledgeBase,
    fsm: FsmSpec,
    seed: int,
    prompts_dir: Path,
    fsm_dir: Path,
) -> Agent:
    """Build a fresh agent for one dialogue."""
    if name == "baseline":
        return BaselineAgent(llm, kb=kb, prompts_dir=prompts_dir, seed=seed)
    if name == "fsm":
        return FsmAgent(
            llm,
            kb=kb,
            spec=fsm,
            fsm_dir=fsm_dir,
            prompts_dir=prompts_dir,
            seed=seed,
        )
    raise RunnerError(
        f"unknown agent {name!r}. Choose one of {AGENTS}, or omit --agent to run both"
    )


def _seed_base(config: ModelsConfig) -> int:
    """Return the configured agent seed, which every dialogue seed derives from."""
    seed = config.models.agent.seed
    if seed is None:
        raise RunnerError(
            "configs/models.yaml must set models.agent.seed; it is the base "
            "the runner derives each dialogue seed from"
        )
    return seed
