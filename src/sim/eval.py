"""Evaluate a ``sim run`` directory into ``metrics.csv`` (T-14b).

Reads the dialogue JSONL of T-14a, scores each ok log with the two judge calls
of T-12 and the deterministic evaluators of T-13, and writes one row per
dialogue. P/R/F1 come from :func:`sim.metrics.fact_scores` on claims the judge
already validated. Re-eval of the same directory is identical: the client of
T-07 caches by prompt hash and the judge seed is the one in the config.
"""

from __future__ import annotations

import csv
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sim.config import ModelsConfig, Role
from sim.evaluators import (
    LabelledTurn,
    StageLabeler,
    flow_scores,
    injection_succeeded,
    labelled_turns,
    llm_latency_s,
    n_turns,
    policy_violation,
    turn_latency_s,
)
from sim.fsm import DEFAULT_FSM_DIR, FsmSpec
from sim.io import atomic_write
from sim.judge import Judge, JudgeSchemaError
from sim.kb import KnowledgeBase
from sim.llm import Chat
from sim.metrics import ACCURACY_SCORE, METRICS, fact_scores
from sim.prompts import DEFAULT_PROMPTS_DIR
from sim.runner import hash_dataset, hash_fsm
from sim.schemas import DialogueLog, JobRef, Manifest, Scenario, load_scenarios

logger = logging.getLogger(__name__)

METRICS_CSV = "metrics.csv"
METRICS_TURN_CSV = "metrics_turn.csv"
UNSCORED_CSV = "unscored.csv"
IDENTITY = ("scenario_id", "agent", "repetition")
METRICS_FIELDS = (*IDENTITY, *METRICS)
UNSCORED_FIELDS = (*IDENTITY, "reason")
TURN_FIELDS = tuple(LabelledTurn.model_fields)


class EvalError(RuntimeError):
    """The eval cannot start or a log cannot be scored; the message says why."""


Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class EvalResult:
    """How many dialogues were scored, already failed, or the judge could not grade."""

    n_scored: int
    n_failed: int
    n_unscored: int


@dataclass(frozen=True)
class _Graded:
    """One finished dialogue: its row and turns, or the reason the judge failed it."""

    log: DialogueLog
    row: dict[str, Any] | None
    turns: list[LabelledTurn]
    reason: str | None


def evaluate_run(
    run_dir: Path,
    *,
    llm: Chat,
    kb: KnowledgeBase,
    fsm: FsmSpec,
    config: ModelsConfig,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    fsm_dir: Path = DEFAULT_FSM_DIR,
    parallel: int = 1,
    on_progress: Progress | None = None,
) -> EvalResult:
    """Score every ok dialogue under ``run_dir`` and write the metrics CSVs.

    ``parallel`` scores that many dialogues at once. It is a machine choice and
    never a measurement one: each dialogue is graded on its own, the seeds come
    from the config rather than from the position in the queue, and the CSV rows
    are ordered by the manifest, so the files do not depend on which thread
    finished first.
    """
    if parallel < 1:
        raise EvalError(f"--parallel must be at least 1, not {parallel}")
    manifest = _load_manifest(run_dir)
    scenarios_dir = Path(manifest.scenarios_dir)
    _check_recorded_hash(
        hash_dataset(scenarios_dir),
        manifest.dataset_hash,
        artifact="scenarios",
        path=scenarios_dir,
    )
    _check_recorded_hash(
        hash_fsm(fsm_dir),
        manifest.fsm_hash,
        artifact="the FSM",
        path=fsm_dir,
    )
    scenarios = {
        scenario.id: scenario
        for scenario in load_scenarios(scenarios_dir, kb=kb, fsm=fsm)
    }
    judge_seed = _seed(config, "judge")
    labeler_seed = _seed(config, "simulator")
    judge = Judge(llm, kb=kb, prompts_dir=prompts_dir, seed=judge_seed)
    labeler = StageLabeler(llm, spec=fsm, prompts_dir=prompts_dir, seed=labeler_seed)
    logs = _load_logs(run_dir)
    n_failed = sum(1 for log in logs if log.status == "failed")
    ok_logs = [log for log in logs if log.status == "ok"]
    scored: dict[tuple[str, str, int], dict[str, Any]] = {}
    turns_by_key: dict[tuple[str, str, int], list[LabelledTurn]] = {}
    unscored: dict[tuple[str, str, int], dict[str, Any]] = {}
    random.Random(judge_seed).shuffle(ok_logs)
    pairs = [
        (log, _scenario_of(log, scenarios, manifest.scenarios_dir)) for log in ok_logs
    ]

    def score(pair: tuple[DialogueLog, Scenario]) -> _Graded:
        log, scenario = pair
        try:
            row, turns = _score(
                log, scenario, judge=judge, labeler=labeler, kb=kb, fsm=fsm
            )
        except JudgeSchemaError as failure:
            return _Graded(log=log, row=None, turns=[], reason=str(failure))
        return _Graded(log=log, row=row, turns=turns, reason=None)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(score, pair) for pair in pairs]
        for index, future in enumerate(as_completed(futures), start=1):
            graded = future.result()
            key = _key(graded.log)
            if graded.row is None:
                unscored[key] = {
                    "scenario_id": graded.log.scenario_id,
                    "agent": graded.log.agent,
                    "repetition": graded.log.repetition,
                    "reason": graded.reason,
                }
                logger.warning(
                    "%s/%s/%s unscored: %s",
                    graded.log.scenario_id,
                    graded.log.agent,
                    graded.log.repetition,
                    graded.reason,
                )
            else:
                scored[key], turns_by_key[key] = graded.row, graded.turns
            if on_progress is not None:
                on_progress(index, len(pairs))
    keys = _csv_keys(manifest.jobs, scored)
    _write_csv(run_dir / METRICS_CSV, METRICS_FIELDS, [scored[key] for key in keys])
    _write_csv(
        run_dir / METRICS_TURN_CSV,
        TURN_FIELDS,
        [turn.model_dump() for key in keys for turn in turns_by_key[key]],
    )
    _write_csv(
        run_dir / UNSCORED_CSV,
        UNSCORED_FIELDS,
        [unscored[key] for key in _csv_keys(manifest.jobs, unscored)],
    )
    return EvalResult(n_scored=len(keys), n_failed=n_failed, n_unscored=len(unscored))


def _score(
    log: DialogueLog,
    scenario: Scenario,
    *,
    judge: Judge,
    labeler: StageLabeler,
    kb: KnowledgeBase,
    fsm: FsmSpec,
) -> tuple[dict[str, Any], list[LabelledTurn]]:
    """Return one metrics row and the turn rows for ``log``."""
    verdict = judge.evaluate(log.transcript, scenario)
    if scenario.is_needle and verdict.facts.needle_recovered is None:
        raise EvalError(
            f"{log.scenario_id} is a needle scenario but the judge returned "
            "needle_recovered=null. The facts call must answer yes or no when "
            "the needle placeholder is a fact ID"
        )
    scores = fact_scores(required=scenario.required_facts, claims=verdict.facts.claims)
    labels = labeler.label(log)
    flow = flow_scores(labels, scenario.expected_final_state, fsm)
    global_judgement = verdict.global_judgement
    return (
        {
            "scenario_id": log.scenario_id,
            "agent": log.agent,
            "repetition": log.repetition,
            "fact_precision": scores.fact_precision,
            "fact_recall": scores.fact_recall,
            "fact_f1": scores.fact_f1,
            "claim_support": scores.claim_support,
            "unsupported_claim_rate": scores.unsupported_claim_rate,
            "n_checkable_claims": scores.n_checkable_claims,
            "needle_recovered": (
                verdict.facts.needle_recovered if scenario.is_needle else None
            ),
            "accuracy_score": ACCURACY_SCORE[global_judgement.accuracy],
            "relevance": global_judgement.relevance,
            "task_completed": global_judgement.task_completed,
            "offensive_content": global_judgement.offensive_content,
            "injection_succeeded": injection_succeeded(log, scenario),
            "policy_violation": policy_violation(log, scenario, kb),
            "n_turns": n_turns(log),
            "n_stage_transitions": flow.n_stage_transitions,
            "n_self_loops": flow.n_self_loops,
            "llm_latency_s": llm_latency_s(log),
            "turn_latency_s": turn_latency_s(log),
            "ended_in_expected_state": flow.ended_in_expected_state,
            "valid_flow_path": flow.valid_flow_path,
            "flow_adherence": flow.flow_adherence,
        },
        labelled_turns(log, labels),
    )


def _check_recorded_hash(
    found: str, recorded: str, *, artifact: str, path: Path
) -> None:
    """Fail unless ``found`` is the hash the run recorded for ``artifact``."""
    if found == recorded:
        return
    raise EvalError(
        f"{artifact} in {path} hash to {found}, but the run recorded {recorded}. "
        "The files must be the ones the dialogues were played against; check out "
        "the same commit on both machines"
    )


def _csv_keys(
    jobs: Sequence[JobRef],
    scored: Mapping[tuple[str, str, int], object],
) -> list[tuple[str, str, int]]:
    """Order scored rows by the manifest, then extras in file order."""
    known = [(job.scenario_id, job.agent, job.repetition) for job in jobs]
    listed = [key for key in known if key in scored]
    extras = sorted(key for key in scored if key not in set(known))
    return listed + extras


def _load_manifest(run_dir: Path) -> Manifest:
    """Load ``manifest.json``, which names the scenario directory to join against."""
    path = run_dir / "manifest.json"
    if not path.exists():
        raise EvalError(
            f"{path} is missing. sim eval reads the T-14a manifest to find the "
            "scenarios that produced the dialogues"
        )
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as invalid:
        if any(
            error["type"] == "missing" and error["loc"] == ("fsm_hash",)
            for error in invalid.errors()
        ):
            raise EvalError(
                f"{path} has no fsm_hash: this run predates the FSM hash. "
                "Delete the directory and re-run sim run; sim eval cannot score "
                "a manifest it cannot identify"
            ) from invalid
        raise EvalError(
            f"{path} does not match the manifest schema:\n{invalid}"
        ) from invalid


def _load_logs(run_dir: Path) -> list[DialogueLog]:
    """Load every dialogue JSONL under ``run_dir/dialogues``."""
    directory = run_dir / "dialogues"
    if not directory.exists():
        raise EvalError(
            f"{directory} is missing. sim eval scores the JSONL files T-14a wrote"
        )
    return [
        DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.jsonl"))
    ]


def _scenario_of(
    log: DialogueLog, scenarios: Mapping[str, Scenario], scenarios_dir: str
) -> Scenario:
    """Return the scenario ``log`` was played against, before any scoring starts."""
    scenario = scenarios.get(log.scenario_id)
    if scenario is None:
        raise EvalError(
            f"{log.scenario_id} is not in {scenarios_dir}. Eval joins "
            "each log to its scenario by id; a missing scenario cannot be scored"
        )
    return scenario


def _key(log: DialogueLog) -> tuple[str, str, int]:
    """The identity columns that pair a log with a manifest job."""
    return (log.scenario_id, log.agent, log.repetition)


def _seed(config: ModelsConfig, role: Role) -> int:
    """Return the configured seed for ``role``, which makes re-eval identical."""
    seed = config.spec(role).seed
    if seed is None:
        raise EvalError(
            f"configs/models.yaml must set models.{role}.seed; eval uses it so "
            "a re-run of the same dialogues is identical"
        )
    return seed


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write ``rows`` atomically so a reader never sees a half file."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_cell(row[field]) for field in fieldnames})
    atomic_write(path, buffer.getvalue())


def _csv_cell(value: object) -> str:
    """Render one CSV cell: empty for NA, otherwise ``str(value)``."""
    if value is None:
        return ""
    return str(value)
