"""Tests for the eval pipeline of T-14b."""

import csv
import json
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from helpers import (
    CONFIG,
    CannedEvalLlm,
    FakeOllama,
    RunCanned,
    canned_eval_transport_replies,
    judge_facts_reply,
    make_dialogue_log,
    make_turn_record,
)
from sim.config import Role, load_models_config
from sim.eval import EvalError, EvalProgress, Progress, evaluate_run
from sim.evaluators import LabelledTurn
from sim.fsm import FsmSpec
from sim.io import atomic_write
from sim.kb import KnowledgeBase
from sim.llm import Chat, LlmClient, LlmResponse, Message
from sim.metrics import METRICS, JudgeFacts, fact_scores
from sim.runner import AGENTS, iter_jobs
from sim.schemas import DialogueLog, Manifest, Scenario, TurnRecord, dialogue_filename

IDENTITY = ("scenario_id", "agent", "repetition")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return fieldnames and rows of a CSV written by eval."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_log(run_dir: Path, log: DialogueLog) -> None:
    """Write ``log`` under the T-14a file name."""
    atomic_write(
        run_dir
        / "dialogues"
        / dialogue_filename(log.scenario_id, log.agent, log.repetition),
        log.model_dump_json() + "\n",
    )


def _eval(
    run_dir: Path,
    llm: Chat,
    kb: KnowledgeBase,
    fsm: FsmSpec,
    parallel: int = 1,
    on_progress: Progress | None = None,
) -> None:
    """Score ``run_dir`` with the canned eval client and the real config."""
    evaluate_run(
        run_dir,
        llm=llm,
        kb=kb,
        fsm=fsm,
        config=load_models_config(CONFIG),
        parallel=parallel,
        on_progress=on_progress,
    )


def test_eval_writes_metrics_csv_with_all_t04_columns_for_ok_dialogues(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    run_canned()
    happy, attack = two_example_scenarios
    claims = JudgeFacts.model_validate_json(
        judge_facts_reply(needle_recovered=False)
    ).claims

    _eval(run_dir, canned_eval_llm, real_kb, real_fsm)

    fieldnames, rows = _read_csv(run_dir / "metrics.csv")

    assert fieldnames == [*IDENTITY, *METRICS]
    assert len(rows) == 4
    assert {
        (row["scenario_id"], row["agent"], int(row["repetition"])) for row in rows
    } == {
        (happy.id, "baseline", 1),
        (happy.id, "fsm", 1),
        (attack.id, "baseline", 1),
        (attack.id, "fsm", 1),
    }
    for row in rows:
        scenario = happy if row["scenario_id"] == happy.id else attack
        scores = fact_scores(required=scenario.required_facts, claims=claims)
        assert float(row["fact_f1"]) == scores.fact_f1
        assert float(row["claim_support"]) == scores.claim_support
        assert row["fact_id_leak"] == "False"
        if row["agent"] == "baseline":
            assert row["stage_label_accuracy"] == ""
        else:
            assert 0.0 <= float(row["stage_label_accuracy"]) <= 1.0
        if scenario is happy:
            assert row["needle_recovered"] == ""
            assert row["injection_succeeded"] == ""
            assert row["policy_violation"] == ""
        else:
            assert row["needle_recovered"] == "False"
            assert row["injection_succeeded"] == "False"
            assert row["policy_violation"] == "False"


def test_eval_sets_fact_id_leak_when_a_reply_contains_a_fact_identifier(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    _write_log(
        run_dir,
        make_dialogue_log(
            [make_turn_record(agent_reply="That policy is covered by fact F31.")],
            scenario_id="happy_path_01",
            agent="baseline",
        ),
    )

    _eval(run_dir, canned_eval_llm, real_kb, real_fsm)
    _, rows = _read_csv(run_dir / "metrics.csv")
    leaked = next(
        row
        for row in rows
        if row["scenario_id"] == "happy_path_01" and row["agent"] == "baseline"
    )

    assert leaked["fact_id_leak"] == "True"


def test_eval_writes_metrics_turn_csv_with_one_row_per_agent_turn(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_turn_fsm_records: list[TurnRecord],
) -> None:
    run_canned()
    log = make_dialogue_log(
        two_turn_fsm_records, scenario_id="happy_path_01", agent="fsm"
    )
    _write_log(run_dir, log)

    _eval(run_dir, canned_eval_llm, real_kb, real_fsm)

    fieldnames, rows = _read_csv(run_dir / "metrics_turn.csv")
    fsm_rows = [
        row
        for row in rows
        if row["scenario_id"] == "happy_path_01" and row["agent"] == "fsm"
    ]

    assert fieldnames == list(LabelledTurn.model_fields)
    assert len(rows) == 5
    assert [int(row["turn"]) for row in fsm_rows] == [1, 2]
    assert [row["labelled_stage"] for row in fsm_rows] == ["greeting", "greeting"]
    assert [row["true_state_after"] for row in fsm_rows] == [
        "identification",
        "closing",
    ]


def test_eval_skips_failed_dialogues(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    _write_log(
        run_dir,
        make_dialogue_log(
            [],
            scenario_id="happy_path_01",
            agent="baseline",
            status="failed",
            stop_reason=None,
        ),
    )

    result = evaluate_run(
        run_dir,
        llm=canned_eval_llm,
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
    )

    _, rows = _read_csv(run_dir / "metrics.csv")
    _, turn_rows = _read_csv(run_dir / "metrics_turn.csv")
    judged = [call for call in canned_eval_llm.calls if call["caller"] == "judge_facts"]

    assert result.n_scored == 3
    assert result.n_failed == 1
    assert len(rows) == 3
    assert ("happy_path_01", "baseline") not in {
        (row["scenario_id"], row["agent"]) for row in rows
    }
    assert len(turn_rows) == 3
    assert len(judged) == 3


def test_eval_skips_the_dialogue_whose_judge_reply_never_validates(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    run_canned()
    _mark_replies(run_dir, two_example_scenarios)
    happy, _ = two_example_scenarios
    llm = CannedEvalLlm(invalid_facts_marker=f"{happy.id}:baseline")

    result = evaluate_run(
        run_dir,
        llm=llm,
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
    )

    _, rows = _read_csv(run_dir / "metrics.csv")
    _, turn_rows = _read_csv(run_dir / "metrics_turn.csv")
    unscored_fields, unscored = _read_csv(run_dir / "unscored.csv")

    assert result.n_scored == 3
    assert result.n_unscored == 1
    assert len(rows) == 3
    assert (happy.id, "baseline") not in {
        (row["scenario_id"], row["agent"]) for row in rows
    }
    assert len(turn_rows) == 3
    assert unscored_fields == [*IDENTITY, "reason"]
    assert [
        (row["scenario_id"], row["agent"], int(row["repetition"])) for row in unscored
    ] == [(happy.id, "baseline", 1)]
    assert "fact_id" in unscored[0]["reason"]


def test_eval_skips_the_dialogue_whose_labeler_count_mismatches_turns(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    run_canned()
    _mark_replies(run_dir, two_example_scenarios)
    happy, _ = two_example_scenarios
    llm = CannedEvalLlm(extra_label_marker=f"{happy.id}:baseline")

    result = evaluate_run(
        run_dir,
        llm=llm,
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
    )

    _, rows = _read_csv(run_dir / "metrics.csv")
    _, turn_rows = _read_csv(run_dir / "metrics_turn.csv")
    unscored_fields, unscored = _read_csv(run_dir / "unscored.csv")

    assert result.n_scored == 3
    assert result.n_unscored == 1
    assert len(rows) == 3
    assert (happy.id, "baseline") not in {
        (row["scenario_id"], row["agent"]) for row in rows
    }
    assert len(turn_rows) == 3
    assert unscored_fields == [*IDENTITY, "reason"]
    assert [
        (row["scenario_id"], row["agent"], int(row["repetition"])) for row in unscored
    ] == [(happy.id, "baseline", 1)]
    assert "label" in unscored[0]["reason"]
    assert "turn" in unscored[0]["reason"]


def test_eval_skips_the_dialogue_whose_judge_call_times_out(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    run_canned()
    _mark_replies(run_dir, two_example_scenarios)
    happy, _ = two_example_scenarios
    llm = CannedEvalLlm(timeout_marker=f"{happy.id}:baseline")

    result = evaluate_run(
        run_dir,
        llm=llm,
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
    )

    _, rows = _read_csv(run_dir / "metrics.csv")
    _, turn_rows = _read_csv(run_dir / "metrics_turn.csv")
    unscored_fields, unscored = _read_csv(run_dir / "unscored.csv")

    assert result.n_scored == 3
    assert result.n_unscored == 1
    assert len(rows) == 3
    assert (happy.id, "baseline") not in {
        (row["scenario_id"], row["agent"]) for row in rows
    }
    assert len(turn_rows) == 3
    assert unscored_fields == [*IDENTITY, "reason"]
    assert [
        (row["scenario_id"], row["agent"], int(row["repetition"])) for row in unscored
    ] == [(happy.id, "baseline", 1)]
    assert "timed out" in unscored[0]["reason"]


def test_eval_writes_a_header_only_unscored_csv_when_every_dialogue_scores(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()

    _eval(run_dir, canned_eval_llm, real_kb, real_fsm)

    fieldnames, rows = _read_csv(run_dir / "unscored.csv")

    assert fieldnames == [*IDENTITY, "reason"]
    assert rows == []


def _mark_replies(run_dir: Path, scenarios: tuple[Scenario, Scenario]) -> None:
    """Give each dialogue a unique agent reply so call order is observable."""
    for scenario in scenarios:
        for agent in AGENTS:
            _write_log(
                run_dir,
                make_dialogue_log(
                    [make_turn_record(agent_reply=f"{scenario.id}:{agent}")],
                    scenario_id=scenario.id,
                    agent=agent,
                ),
            )


def _dialogue_of(call: dict[str, object]) -> tuple[str, str]:
    """Return (scenario_id, agent) planted in the canned agent reply."""
    messages = call["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert isinstance(content, str)
    for scenario_id in ("happy_path_01", "adversarial_01"):
        for agent in AGENTS:
            if f"{scenario_id}:{agent}" in content:
                return scenario_id, agent
    raise AssertionError(f"planted reply not in judge prompt: {content!r}")


def test_eval_calls_the_judge_in_shuffled_order(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    run_canned()
    _mark_replies(run_dir, two_example_scenarios)
    jobs_order = [
        (job.scenario.id, job.agent)
        for job in iter_jobs(two_example_scenarios, agents=AGENTS, reps=1)
    ]
    file_order = [
        (scenario_id, agent)
        for scenario_id in sorted(scenario.id for scenario in two_example_scenarios)
        for agent in AGENTS
    ]
    first = CannedEvalLlm()
    _eval(run_dir, first, real_kb, real_fsm)
    second = CannedEvalLlm()
    _eval(run_dir, second, real_kb, real_fsm)

    order = [
        _dialogue_of(call) for call in first.calls if call["caller"] == "judge_facts"
    ]
    _, rows = _read_csv(run_dir / "metrics.csv")
    csv_order = [(row["scenario_id"], row["agent"]) for row in rows]

    assert order != jobs_order
    assert order != file_order
    assert sorted(order) == sorted(jobs_order)
    assert [
        _dialogue_of(call) for call in second.calls if call["caller"] == "judge_facts"
    ] == order
    assert csv_order == jobs_order


def test_eval_in_parallel_writes_the_same_csvs_as_a_sequential_eval(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    """Concurrency is a machine choice, never a measurement one."""
    run_canned()
    _mark_replies(run_dir, two_example_scenarios)
    sequential = CannedEvalLlm()
    _eval(run_dir, sequential, real_kb, real_fsm)
    metrics = (run_dir / "metrics.csv").read_bytes()
    turns = (run_dir / "metrics_turn.csv").read_bytes()
    unscored = (run_dir / "unscored.csv").read_bytes()

    concurrent = CannedEvalLlm()
    _eval(run_dir, concurrent, real_kb, real_fsm, parallel=2)

    assert (run_dir / "metrics.csv").read_bytes() == metrics
    assert (run_dir / "metrics_turn.csv").read_bytes() == turns
    assert (run_dir / "unscored.csv").read_bytes() == unscored
    assert sorted(call["prompt_hash"] for call in concurrent.calls) == sorted(
        call["prompt_hash"] for call in sequential.calls
    )
    assert sorted(call["seed"] or 0 for call in concurrent.calls) == sorted(
        call["seed"] or 0 for call in sequential.calls
    )


def test_eval_rejects_a_parallel_below_one(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()

    with pytest.raises(EvalError, match="--parallel must be at least 1"):
        _eval(run_dir, canned_eval_llm, real_kb, real_fsm, parallel=0)


def test_reeval_of_the_same_run_hits_the_prompt_hash_cache(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    transport = FakeOllama(canned_eval_transport_replies(4))
    client = LlmClient(
        load_models_config(CONFIG),
        cache_dir=run_dir / "cache",
        log_path=run_dir / "llm_calls.jsonl",
        transport=transport,
    )
    config = load_models_config(CONFIG)

    evaluate_run(run_dir, llm=client, kb=real_kb, fsm=real_fsm, config=config)
    first_metrics = (run_dir / "metrics.csv").read_bytes()
    first_turns = (run_dir / "metrics_turn.csv").read_bytes()
    n_calls = len(transport.calls)

    evaluate_run(run_dir, llm=client, kb=real_kb, fsm=real_fsm, config=config)

    assert len(transport.calls) == n_calls
    assert (run_dir / "metrics.csv").read_bytes() == first_metrics
    assert (run_dir / "metrics_turn.csv").read_bytes() == first_turns


def test_eval_writes_a_row_for_every_ok_log_not_only_manifest_jobs(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    run_canned(agents=("baseline",))
    run_canned(agents=("fsm",), resume=True)
    happy, attack = two_example_scenarios

    _eval(run_dir, canned_eval_llm, real_kb, real_fsm)

    _, rows = _read_csv(run_dir / "metrics.csv")

    assert [(row["scenario_id"], row["agent"]) for row in rows[:2]] == [
        (happy.id, "fsm"),
        (attack.id, "fsm"),
    ]
    assert {(row["scenario_id"], row["agent"]) for row in rows} == {
        (happy.id, "baseline"),
        (happy.id, "fsm"),
        (attack.id, "baseline"),
        (attack.id, "fsm"),
    }


@pytest.mark.parametrize("field", ["dataset_hash", "fsm_hash"])
def test_eval_rejects_a_directory_that_does_not_match_the_recorded_hash(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    field: str,
) -> None:
    run_canned()
    path = run_dir / "manifest.json"
    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    atomic_write(
        path,
        manifest.model_copy(update={field: "0" * 64}).model_dump_json() + "\n",
    )

    with pytest.raises(EvalError, match="hash"):
        _eval(run_dir, canned_eval_llm, real_kb, real_fsm)


def test_eval_rejects_a_manifest_that_predates_the_fsm_hash(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    path = run_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["fsm_hash"]
    atomic_write(path, json.dumps(payload) + "\n")

    with pytest.raises(EvalError, match="predates the FSM hash"):
        _eval(run_dir, canned_eval_llm, real_kb, real_fsm)


def test_eval_rejects_a_null_needle_recovered_on_a_needle_scenario(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()

    with pytest.raises(EvalError, match="needle_recovered"):
        _eval(run_dir, CannedEvalLlm(needle_recovered=None), real_kb, real_fsm)

    assert not (run_dir / "unscored.csv").exists()


def test_eval_rejects_a_log_whose_scenario_is_missing(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    _write_log(
        run_dir,
        make_dialogue_log(
            [make_turn_record()],
            scenario_id="edge_99",
            agent="baseline",
        ),
    )

    with pytest.raises(EvalError, match="edge_99"):
        _eval(run_dir, canned_eval_llm, real_kb, real_fsm)


def test_eval_rejects_a_missing_manifest(
    tmp_path: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    with pytest.raises(EvalError, match="manifest"):
        _eval(tmp_path / "ghost", canned_eval_llm, real_kb, real_fsm)


def test_eval_rejects_a_missing_dialogues_directory(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    (run_dir / "dialogues").rename(run_dir / "dialogues.bak")

    with pytest.raises(EvalError, match="dialogues"):
        _eval(run_dir, canned_eval_llm, real_kb, real_fsm)


def test_eval_rejects_a_missing_judge_seed(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    config = load_models_config(CONFIG)
    judge = config.models.judge.model_copy(update={"seed": None})
    models = config.models.model_copy(update={"judge": judge})
    bare = config.model_copy(update={"models": models})

    with pytest.raises(EvalError, match=r"judge\.seed"):
        evaluate_run(
            run_dir,
            llm=canned_eval_llm,
            kb=real_kb,
            fsm=real_fsm,
            config=bare,
        )


class _GatedEvalLlm:
    """A Chat that blocks in ``chat()`` until ``gate`` is set."""

    def __init__(self, inner: CannedEvalLlm, gate: threading.Event) -> None:
        self._inner = inner
        self._gate = gate

    def chat(
        self,
        messages: Sequence[Message],
        *,
        role: Role,
        caller: str,
        schema: type[BaseModel] | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> LlmResponse:
        self._gate.wait()
        return self._inner.chat(
            messages,
            role=role,
            caller=caller,
            schema=schema,
            seed=seed,
            num_predict=num_predict,
        )


class _BoomLlm:
    """A Chat that always raises, so ``score()`` cannot produce a ``_Graded``."""

    def chat(
        self,
        messages: Sequence[Message],
        *,
        role: Role,
        caller: str,
        schema: type[BaseModel] | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> LlmResponse:
        raise RuntimeError("boom")


class _CallProgressLlm:
    """Record ``completed`` at each LLM call, to show it is not a per-call counter."""

    def __init__(self, inner: CannedEvalLlm, snapshots: list[EvalProgress]) -> None:
        self._inner = inner
        self._snapshots = snapshots
        self.completed_at_call: list[int] = []

    def chat(
        self,
        messages: Sequence[Message],
        *,
        role: Role,
        caller: str,
        schema: type[BaseModel] | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> LlmResponse:
        self.completed_at_call.append(
            self._snapshots[-1].completed if self._snapshots else 0
        )
        return self._inner.chat(
            messages,
            role=role,
            caller=caller,
            schema=schema,
            seed=seed,
            num_predict=num_predict,
        )


def _run_eval_in_thread(
    run_dir: Path,
    llm: Chat,
    kb: KnowledgeBase,
    fsm: FsmSpec,
    *,
    parallel: int,
    on_progress: Progress,
) -> tuple[threading.Thread, list[BaseException]]:
    """Start ``evaluate_run`` on a thread and return it plus a slot for exceptions."""
    errors: list[BaseException] = []

    def run() -> None:
        try:
            evaluate_run(
                run_dir,
                llm=llm,
                kb=kb,
                fsm=fsm,
                config=load_models_config(CONFIG),
                parallel=parallel,
                on_progress=on_progress,
            )
        except BaseException as failure:
            errors.append(failure)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, errors


def test_progress_completed_counts_dialogues_not_llm_calls(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    snapshots: list[EvalProgress] = []
    inner = CannedEvalLlm()
    llm = _CallProgressLlm(inner, snapshots)

    evaluate_run(
        run_dir,
        llm=llm,
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
        on_progress=snapshots.append,
    )

    n_dialogues = snapshots[-1].total
    assert llm.completed_at_call == [
        dialogue for dialogue in range(n_dialogues) for _ in range(3)
    ]
    assert len(inner.calls) > snapshots[-1].completed
    assert snapshots[-1].completed == n_dialogues


def test_progress_reports_active_while_a_dialogue_is_blocked(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    gate = threading.Event()
    snapshots: list[EvalProgress] = []
    lock = threading.Lock()
    started = threading.Event()

    def on_progress(progress: EvalProgress) -> None:
        with lock:
            snapshots.append(progress)
        if progress.active >= 1:
            started.set()

    thread, errors = _run_eval_in_thread(
        run_dir,
        _GatedEvalLlm(CannedEvalLlm(), gate),
        real_kb,
        real_fsm,
        parallel=1,
        on_progress=on_progress,
    )
    assert started.wait(timeout=5)
    with lock:
        blocked = list(snapshots)
    assert any(snapshot.active >= 1 for snapshot in blocked)
    assert all(snapshot.completed == 0 for snapshot in blocked)

    gate.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert snapshots[-1].completed == snapshots[-1].total
    assert snapshots[-1].active == 0


def test_progress_active_stays_within_parallel_bounds(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    parallel = 2
    gate = threading.Event()
    snapshots: list[EvalProgress] = []
    lock = threading.Lock()
    overlapped = threading.Event()

    def on_progress(progress: EvalProgress) -> None:
        with lock:
            snapshots.append(progress)
        if progress.active > 1:
            overlapped.set()

    thread, errors = _run_eval_in_thread(
        run_dir,
        _GatedEvalLlm(CannedEvalLlm(), gate),
        real_kb,
        real_fsm,
        parallel=parallel,
        on_progress=on_progress,
    )
    assert overlapped.wait(timeout=5)
    gate.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert all(0 <= snapshot.active <= parallel for snapshot in snapshots)
    assert snapshots[-1].completed == snapshots[-1].total
    assert snapshots[-1].active == 0


def test_progress_final_state_is_exact_after_fast_jobs(
    run_canned: RunCanned,
    run_dir: Path,
    canned_eval_llm: CannedEvalLlm,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    snapshots: list[EvalProgress] = []

    _eval(run_dir, canned_eval_llm, real_kb, real_fsm, on_progress=snapshots.append)

    assert snapshots[-1].completed == snapshots[-1].total
    assert snapshots[-1].active == 0


def test_progress_counts_unscored_dialogues_as_completed(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
    two_example_scenarios: tuple[Scenario, Scenario],
) -> None:
    run_canned()
    _mark_replies(run_dir, two_example_scenarios)
    happy, _ = two_example_scenarios
    snapshots: list[EvalProgress] = []

    result = evaluate_run(
        run_dir,
        llm=CannedEvalLlm(invalid_facts_marker=f"{happy.id}:baseline"),
        kb=real_kb,
        fsm=real_fsm,
        config=load_models_config(CONFIG),
        on_progress=snapshots.append,
    )

    assert result.n_unscored == 1
    assert snapshots[-1].completed == snapshots[-1].total
    assert snapshots[-1].active == 0


def test_progress_cleans_up_active_when_score_raises(
    run_canned: RunCanned,
    run_dir: Path,
    real_kb: KnowledgeBase,
    real_fsm: FsmSpec,
) -> None:
    run_canned()
    snapshots: list[EvalProgress] = []

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_run(
            run_dir,
            llm=_BoomLlm(),
            kb=real_kb,
            fsm=real_fsm,
            config=load_models_config(CONFIG),
            on_progress=snapshots.append,
        )

    assert snapshots[-1].active == 0
    assert snapshots[-1].completed == 0
    assert not (run_dir / "unscored.csv").exists()
