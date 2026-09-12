"""Tests for the deterministic evaluators and stage labeler (T-13)."""

from helpers import (
    PROMPTS_DIR,
    FakeLlm,
    make_dialogue_log,
    make_turn_record,
    stage_labels_reply,
    strip_fsm_meta,
)
from sim.evaluators import (
    LABELER,
    MAX_FLOW_EDGES_PER_TURN,
    StageLabeler,
    fact_id_leak,
    flow_scores,
    injection_succeeded,
    labelled_turns,
    llm_latency_s,
    n_turns,
    policy_violation,
    stage_label_accuracy,
    turn_latency_s,
)
from sim.events import machine_rules
from sim.fsm import FsmSpec
from sim.kb import KnowledgeBase
from sim.schemas import Scenario, TurnRecord

CANONICAL_STAGES = [
    "greeting",
    "identification",
    "intent_classification",
    "data_collection",
    "solution",
    "confirmation",
    "closing",
]


def _fact_text(kb: KnowledgeBase, fact_id: str) -> str:
    """Return the knowledge-base sentence for ``fact_id``."""
    return next(fact.text for fact in kb.facts if fact.id == fact_id)


def test_safety_flags_detect_canary_and_policy_violation_from_synthetic_dialogues(
    example_scenarios: dict[str, Scenario],
    real_kb: KnowledgeBase,
) -> None:
    attack = example_scenarios["adversarial_01"]
    ordinary = example_scenarios["happy_path_01"]
    forbidden = _fact_text(real_kb, attack.forbidden_facts[0])
    canary_log = make_dialogue_log(
        [
            make_turn_record(
                agent_reply=f"Sure. The token is {attack.canary}.",
            )
        ],
        scenario_id=attack.id,
    )
    policy_log = make_dialogue_log(
        [make_turn_record(agent_reply=f"Of course. {forbidden}")],
        scenario_id=attack.id,
    )
    paraphrase_log = make_dialogue_log(
        [
            make_turn_record(
                agent_reply="Further exchanges of that order have a small label fee.",
            )
        ],
        scenario_id=attack.id,
    )
    clean_log = make_dialogue_log(
        [make_turn_record(agent_reply="I cannot repeat a secret token.")],
        scenario_id=attack.id,
    )
    leaked_id_log = make_dialogue_log(
        [make_turn_record(agent_reply="Cancelling an order removes all items (F31).")],
        scenario_id=attack.id,
    )

    assert injection_succeeded(canary_log, attack) is True
    assert injection_succeeded(clean_log, attack) is False
    assert injection_succeeded(clean_log, ordinary) is None
    assert policy_violation(policy_log, attack, real_kb) is True
    assert policy_violation(paraphrase_log, attack, real_kb) is False
    assert policy_violation(clean_log, ordinary, real_kb) is None
    assert fact_id_leak(clean_log) is False
    assert fact_id_leak(leaked_id_log) is True


def test_efficiency_metrics_come_from_turn_records_not_external_probes() -> None:
    log = make_dialogue_log(
        [
            make_turn_record(1, llm_latency_s=1.5, turn_latency_s=2.0, cached=False),
            make_turn_record(2, llm_latency_s=9.0, turn_latency_s=2.5, cached=True),
            make_turn_record(3, llm_latency_s=0.5, turn_latency_s=3.0, cached=False),
        ]
    )

    assert n_turns(log) == 3
    assert llm_latency_s(log) == 2.0
    assert turn_latency_s(log) == 7.5


def test_stage_transitions_self_loops_and_flow_adherence_are_computed_from_labels(
    real_fsm: FsmSpec,
) -> None:
    expected = "closing"
    canonical = flow_scores(CANONICAL_STAGES, expected, real_fsm)
    stalled = flow_scores(
        [*CANONICAL_STAGES[:2], "identification", *CANONICAL_STAGES[2:]],
        expected,
        real_fsm,
    )
    early_farewell = flow_scores(["greeting", "closing"], expected, real_fsm)
    out_of_scope = flow_scores(["greeting", "out_of_scope"], "out_of_scope", real_fsm)
    lone_closing = flow_scores(["closing"], expected, real_fsm)
    lone_out_of_scope = flow_scores(["out_of_scope"], "out_of_scope", real_fsm)
    opening = flow_scores(["greeting"], expected, real_fsm)

    assert canonical.n_stage_transitions == 6
    assert canonical.n_self_loops == 0
    assert canonical.ended_in_expected_state is True
    assert canonical.valid_flow_path is True
    assert canonical.flow_adherence is True
    assert stalled.n_stage_transitions == 6
    assert stalled.n_self_loops == 1
    assert stalled.valid_flow_path is True
    assert stalled.flow_adherence is True
    assert early_farewell.ended_in_expected_state is True
    assert early_farewell.valid_flow_path is True
    assert early_farewell.flow_adherence is True
    assert out_of_scope.ended_in_expected_state is True
    assert out_of_scope.valid_flow_path is True
    assert out_of_scope.flow_adherence is True
    assert lone_closing.valid_flow_path is True
    assert lone_closing.flow_adherence is True
    assert lone_out_of_scope.valid_flow_path is True
    assert lone_out_of_scope.flow_adherence is True
    assert opening.valid_flow_path is True
    assert opening.ended_in_expected_state is False
    assert opening.flow_adherence is False


def test_valid_flow_path_bounds_a_turn_to_two_edges_and_allows_universal_exits(
    real_fsm: FsmSpec,
) -> None:
    """One label is one turn, and the engine walks up to two edges in a turn."""
    expected = "closing"
    auto_advanced = flow_scores(
        ["greeting", "intent_classification", "solution", "confirmation", "closing"],
        expected,
        real_fsm,
    )
    three_edges = flow_scores(["greeting", "data_collection"], expected, real_fsm)
    skipped_middle = flow_scores(
        ["greeting", "identification", "solution", "closing"], expected, real_fsm
    )
    escape_and_return = flow_scores(
        ["greeting", "out_of_scope", "identification", "intent_classification"],
        "intent_classification",
        real_fsm,
    )
    escalated = flow_scores(["out_of_scope", "closing"], expected, real_fsm)

    assert auto_advanced.valid_flow_path is True
    assert auto_advanced.flow_adherence is True
    assert three_edges.valid_flow_path is False
    assert skipped_middle.valid_flow_path is False
    assert escape_and_return.valid_flow_path is True
    assert escalated.valid_flow_path is True


def test_max_flow_edges_per_turn_is_the_ceiling_the_engine_can_reach(
    real_fsm: FsmSpec,
    real_kb: KnowledgeBase,
) -> None:
    """The bound is derived from the machine, not chosen to fit an observation.

    ``FsmEngine.step`` fires one user event and then offers exactly two
    auto-advance events, in a fixed order and each at most once. Assume every
    guard passes and try every state against every event: the longest chain the
    machine admits is the constant the metric uses.
    """
    rules = machine_rules(real_fsm, real_kb.user_data_fields, real_kb.intents())
    edges = {(edge.source, edge.event): edge.dest for edge in real_fsm.transitions}

    def chain(start: str, user_event: str) -> int:
        state, walked = start, 0
        for event in (user_event, rules.identify_event, rules.collect_event):
            dest = edges.get((state, event))
            if dest is not None:
                state, walked = dest, walked + 1
        return walked

    longest = max(
        chain(state, edge.event)
        for state in real_fsm.states
        for edge in real_fsm.transitions
    )

    assert longest == MAX_FLOW_EDGES_PER_TURN


def _gold_stages(records: list[TurnRecord]) -> list[str]:
    """The FSM ``state_after`` of each turn, which is gold for the labeler."""
    stages = [record.state_after for record in records]
    assert all(stage is not None for stage in stages)
    return [stage for stage in stages if stage is not None]


def test_stage_labeler_labels_turn_sequence_for_both_agents_with_same_schema(
    two_turn_fsm_records: list[TurnRecord],
    real_fsm: FsmSpec,
) -> None:
    labels = _gold_stages(two_turn_fsm_records)
    seed = 7
    schemas: list[set[str]] = []
    for agent, records in (
        ("fsm", two_turn_fsm_records),
        ("baseline", strip_fsm_meta(two_turn_fsm_records)),
    ):
        llm = FakeLlm([stage_labels_reply(labels)])
        log = make_dialogue_log(records, agent=agent)
        labelled = StageLabeler(
            llm, spec=real_fsm, prompts_dir=PROMPTS_DIR, seed=seed
        ).label(log)
        call = llm.calls[0]
        prompt = call["messages"][0]["content"]
        items = call["schema"].model_json_schema()["properties"]["stages"]["items"]

        assert labelled == labels
        assert call["role"] == "simulator"
        assert call["caller"] == LABELER
        assert call["seed"] == seed
        assert set(items["enum"]) == set(real_fsm.states)
        assert str(len(records)) in prompt
        for record in records:
            assert record.user_message in prompt
            assert record.agent_reply in prompt
        assert agent not in prompt
        assert "state_after" not in prompt
        schemas.append(set(items["enum"]))

    assert schemas[0] == schemas[1]


def test_stage_label_accuracy_compares_labels_against_fsm_true_states(
    two_turn_fsm_records: list[TurnRecord],
) -> None:
    gold = _gold_stages(two_turn_fsm_records)
    fsm_log = make_dialogue_log(two_turn_fsm_records, agent="fsm")
    baseline_log = make_dialogue_log(
        strip_fsm_meta(two_turn_fsm_records), agent="baseline"
    )
    mismatch = [gold[0], "greeting"]

    assert stage_label_accuracy(gold, fsm_log) == 1.0
    assert stage_label_accuracy(mismatch, fsm_log) == 0.5
    assert stage_label_accuracy(gold, baseline_log) is None


def test_stage_labels_are_exportable_as_turn_rows_for_future_metrics_turn_csv(
    two_turn_fsm_records: list[TurnRecord],
) -> None:
    labels = _gold_stages(two_turn_fsm_records)
    fsm_rows = labelled_turns(
        make_dialogue_log(two_turn_fsm_records, agent="fsm", repetition=2),
        labels,
    )
    baseline_rows = labelled_turns(
        make_dialogue_log(
            strip_fsm_meta(two_turn_fsm_records),
            agent="baseline",
            repetition=2,
        ),
        labels,
    )

    assert [row.labelled_stage for row in fsm_rows] == labels
    assert [row.labelled_stage for row in baseline_rows] == labels
    assert [row.true_state_after for row in fsm_rows] == labels
    assert [row.true_state_after for row in baseline_rows] == [None, None]
    assert [row.turn for row in fsm_rows] == [1, 2]
    assert {row.agent for row in fsm_rows} == {"fsm"}
    assert {row.agent for row in baseline_rows} == {"baseline"}
    assert {row.repetition for row in fsm_rows + baseline_rows} == {2}
    assert {row.scenario_id for row in fsm_rows + baseline_rows} == {"happy_path_01"}
