"""Trade-off table and exemplar selection (T-21)."""

import inspect
import re
from pathlib import Path

import pytest

from helpers import make_dialogue_log, make_turn_record
from sim.cases import (
    CasesError,
    category_direction_table,
    inverted,
    load_frozen_log,
    lowest_common_repetition,
    rank_case_candidates,
    render_transcript,
    select_exemplars,
)


def _test_row(
    metric: str,
    stratum: str,
    *,
    population: str = "semantic_primary",
    family: str = "exploratory",
    n: int = 20,
    wins: int = 5,
    ties: int = 10,
    losses: int = 5,
    direction: str = "baseline",
    mean_diff: float = -0.1,
    rank_biserial: float | None = -0.2,
) -> dict[str, object]:
    return {
        "population": population,
        "stratum": stratum,
        "metric": metric,
        "family": family,
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "direction": direction,
        "mean_diff": mean_diff,
        "rank_biserial": rank_biserial,
        "wilcoxon_p": 0.01,
        "permutation_p": 0.02,
    }


def _by_key(rows: list[object]) -> dict[tuple[str, str], object]:
    return {(row.metric, row.stratum): row for row in rows}


def test_inversion_flags_category_direction_that_disagrees_with_overall() -> None:
    assert inverted("happy_path", "baseline", "fsm") is True
    assert inverted("adversarial", "fsm", "baseline") is True
    assert inverted("edge", "tie", "fsm") is True
    assert inverted("happy_path", "baseline", "baseline") is False
    assert inverted("overall", "fsm", "baseline") is False
    assert inverted("overall", "baseline", "baseline") is False


def test_category_direction_table_copies_t19_ved_and_marks_inversions() -> None:
    rows = [
        _test_row(
            "task_completed",
            "overall",
            family="primary",
            n=59,
            wins=9,
            ties=39,
            losses=11,
            direction="baseline",
            mean_diff=-0.0113,
            rank_biserial=-0.09,
        ),
        _test_row(
            "task_completed",
            "happy_path",
            n=20,
            wins=2,
            ties=14,
            losses=4,
            direction="baseline",
            mean_diff=-0.0833,
            rank_biserial=-0.62,
        ),
        _test_row(
            "task_completed",
            "adversarial",
            n=19,
            wins=4,
            ties=11,
            losses=4,
            direction="fsm",
            mean_diff=0.0789,
            rank_biserial=0.44,
        ),
        _test_row(
            "fact_precision",
            "overall",
            family="secondary",
            direction="baseline",
        ),
        _test_row(
            "task_completed",
            "overall",
            population="frozen_gate",
            family="secondary",
            n=99,
            direction="fsm",
        ),
    ]

    table = category_direction_table(rows)
    by_key = _by_key(table)

    overall = by_key[("task_completed", "overall")]
    assert overall.n == 59
    assert (overall.wins, overall.ties, overall.losses) == (9, 39, 11)
    assert overall.direction == "baseline"
    assert overall.mean_diff == -0.0113
    assert overall.rank_biserial == -0.09
    assert overall.inverted is False

    happy = by_key[("task_completed", "happy_path")]
    assert happy.n == 20
    assert (happy.wins, happy.ties, happy.losses) == (2, 14, 4)
    assert happy.direction == "baseline"
    assert happy.mean_diff == -0.0833
    assert happy.inverted is False

    adversarial = by_key[("task_completed", "adversarial")]
    assert adversarial.direction == "fsm"
    assert adversarial.inverted is True
    assert adversarial.n == 19

    assert ("fact_precision", "overall") not in by_key
    assert all(row.n != 99 for row in table)


def test_category_table_family_is_confirmatory_exploratory_or_diagnostic() -> None:
    rows = [
        _test_row("task_completed", "overall", family="primary", direction="baseline"),
        _test_row("task_completed", "edge", direction="baseline"),
        _test_row("fact_f1", "overall", family="primary", direction="baseline"),
        _test_row("fact_f1", "happy_path", direction="baseline"),
        _test_row("claim_support", "overall", family="primary", direction="fsm"),
        _test_row("claim_support", "adversarial", direction="fsm"),
        _test_row("flow_adherence", "overall", family="secondary", direction="fsm"),
        _test_row("flow_adherence", "edge", direction="fsm"),
        _test_row(
            "task_completed",
            "overall",
            population="drop5_instrument",
            family="secondary",
            direction="fsm",
        ),
    ]

    table = category_direction_table(rows)
    by_key = _by_key(table)

    assert by_key[("task_completed", "overall")].family == "confirmatory"
    assert by_key[("task_completed", "edge")].family == "exploratory"
    assert by_key[("fact_f1", "overall")].family == "confirmatory"
    assert by_key[("fact_f1", "happy_path")].family == "exploratory"
    assert by_key[("claim_support", "overall")].family == "confirmatory"
    assert by_key[("claim_support", "adversarial")].family == "exploratory"
    assert by_key[("flow_adherence", "overall")].family == "diagnostic"
    assert by_key[("flow_adherence", "edge")].family == "diagnostic"
    assert all(row.family != "secondary" for row in table)
    assert all(row.family != "primary" for row in table)


def _metrics_row(
    scenario_id: str,
    agent: str,
    repetition: int,
    *,
    task_completed: float,
    fact_f1: float,
    claim_support: object,
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "agent": agent,
        "repetition": repetition,
        "task_completed": task_completed,
        "fact_f1": fact_f1,
        "claim_support": claim_support,
    }


def _pair(
    scenario_id: str,
    *,
    task_completed: tuple[float, float],
    fact_f1: tuple[float, float],
    claim_support: tuple[float, float],
) -> list[dict[str, object]]:
    baseline, fsm = 0, 1
    return [
        _metrics_row(
            scenario_id,
            "baseline",
            1,
            task_completed=task_completed[baseline],
            fact_f1=fact_f1[baseline],
            claim_support=claim_support[baseline],
        ),
        _metrics_row(
            scenario_id,
            "fsm",
            1,
            task_completed=task_completed[fsm],
            fact_f1=fact_f1[fsm],
            claim_support=claim_support[fsm],
        ),
    ]


def test_exemplar_ranking_orders_fsm_win_then_loss_then_tie() -> None:
    rows = [
        *_pair(
            "adversarial_02",
            task_completed=(0.0, 1.0),
            fact_f1=(0.2, 0.4),
            claim_support=(0.5, 0.5),
        ),
        *_pair(
            "adversarial_08",
            task_completed=(0.0, 1.0),
            fact_f1=(0.2, 0.8),
            claim_support=(0.4, 0.7),
        ),
        *_pair(
            "edge_05",
            task_completed=(1.0, 0.0),
            fact_f1=(0.8, 0.2),
            claim_support=(1.0, 1.0),
        ),
        *_pair(
            "edge_13",
            task_completed=(1.0, 0.0),
            fact_f1=(0.6, 0.4),
            claim_support=(0.5, 0.5),
        ),
        *_pair(
            "happy_path_07",
            task_completed=(1.0, 1.0),
            fact_f1=(0.5, 0.5),
            claim_support=(1.0, 1.0),
        ),
    ]

    exemplars = select_exemplars(rank_case_candidates(rows))

    assert exemplars["helped"].scenario_id == "adversarial_08"
    assert exemplars["restricted"].scenario_id == "edge_05"
    assert exemplars["tie"].scenario_id == "happy_path_07"
    assert exemplars["helped"].rank == 1
    assert exemplars["restricted"].rank == 1
    assert exemplars["tie"].rank == 1


def test_exemplar_ranking_drops_instrument_drop_and_unpaired_scenarios() -> None:
    rows = [
        *_pair(
            "happy_path_01",
            task_completed=(0.0, 1.0),
            fact_f1=(0.1, 0.2),
            claim_support=(0.5, 0.5),
        ),
        *_pair(
            "happy_path_09",
            task_completed=(0.0, 1.0),
            fact_f1=(0.0, 1.0),
            claim_support=(0.0, 1.0),
        ),
        _metrics_row(
            "adversarial_12",
            "fsm",
            1,
            task_completed=1.0,
            fact_f1=1.0,
            claim_support=1.0,
        ),
    ]

    candidates = rank_case_candidates(rows)
    ids = {row.scenario_id for row in candidates}

    assert ids == {"happy_path_01"}
    assert "happy_path_09" not in ids
    assert "adversarial_12" not in ids


def test_zero_composite_from_cancellation_is_not_a_tie() -> None:
    rows = [
        *_pair(
            "happy_path_03",
            task_completed=(0.5, 0.5),
            fact_f1=(0.2, 0.4),
            claim_support=(0.6, 0.4),
        ),
        *_pair(
            "happy_path_07",
            task_completed=(1.0, 1.0),
            fact_f1=(0.5, 0.5),
            claim_support=(1.0, 1.0),
        ),
    ]

    candidates = rank_case_candidates(rows)
    by_id = {row.scenario_id: row for row in candidates}

    assert by_id["happy_path_03"].composite == 0.0
    assert by_id["happy_path_03"].bucket != "tie"
    assert by_id["happy_path_07"].bucket == "tie"
    assert select_exemplars(candidates)["tie"].scenario_id == "happy_path_07"


def test_tie_bucket_requires_exact_three_primary_zeros() -> None:
    rows = [
        _metrics_row(
            "happy_path_04",
            "baseline",
            1,
            task_completed=0.5,
            fact_f1=0.4,
            claim_support="",
        ),
        _metrics_row(
            "happy_path_04",
            "fsm",
            1,
            task_completed=0.5,
            fact_f1=0.4,
            claim_support="",
        ),
        *_pair(
            "happy_path_07",
            task_completed=(1.0, 1.0),
            fact_f1=(1.0, 1.0),
            claim_support=(1.0, 1.0),
        ),
    ]

    candidates = rank_case_candidates(rows)
    by_id = {row.scenario_id: row for row in candidates}

    assert by_id["happy_path_04"].n_primary_available == 2
    assert by_id["happy_path_04"].task_completed_diff == 0.0
    assert by_id["happy_path_04"].fact_f1_diff == 0.0
    assert by_id["happy_path_04"].claim_support_diff is None
    assert by_id["happy_path_04"].bucket != "tie"
    assert select_exemplars(candidates)["tie"].scenario_id == "happy_path_07"


def _exact_tie(scenario_id: str) -> list[dict[str, object]]:
    return _pair(
        scenario_id,
        task_completed=(1.0, 1.0),
        fact_f1=(0.5, 0.5),
        claim_support=(1.0, 1.0),
    )


def test_tie_bucket_prefers_happy_path_then_scenario_id() -> None:
    rows = [
        *_exact_tie("adversarial_05"),
        *_exact_tie("happy_path_10"),
        *_exact_tie("happy_path_07"),
        *_exact_tie("edge_03"),
    ]

    candidates = rank_case_candidates(rows)
    ties = [row for row in candidates if row.bucket == "tie"]

    assert [row.scenario_id for row in sorted(ties, key=lambda row: row.rank or 0)] == [
        "happy_path_07",
        "happy_path_10",
        "adversarial_05",
        "edge_03",
    ]
    assert select_exemplars(candidates)["tie"].scenario_id == "happy_path_07"


def test_helped_and_restricted_equal_composite_order_by_scenario_id() -> None:
    rows = [
        *_pair(
            "adversarial_04",
            task_completed=(0.0, 1.0),
            fact_f1=(0.5, 0.5),
            claim_support=(0.5, 0.5),
        ),
        *_pair(
            "adversarial_08",
            task_completed=(0.0, 1.0),
            fact_f1=(0.5, 0.5),
            claim_support=(0.5, 0.5),
        ),
        *_pair(
            "edge_13",
            task_completed=(1.0, 0.0),
            fact_f1=(0.5, 0.5),
            claim_support=(0.5, 0.5),
        ),
        *_pair(
            "edge_05",
            task_completed=(1.0, 0.0),
            fact_f1=(0.5, 0.5),
            claim_support=(0.5, 0.5),
        ),
    ]

    candidates = rank_case_candidates(rows)
    helped = [row for row in candidates if row.bucket == "helped"]
    restricted = [row for row in candidates if row.bucket == "restricted"]

    assert [
        row.scenario_id for row in sorted(helped, key=lambda row: row.rank or 0)
    ] == [
        "adversarial_04",
        "adversarial_08",
    ]
    assert [
        row.scenario_id for row in sorted(restricted, key=lambda row: row.rank or 0)
    ] == ["edge_05", "edge_13"]
    assert select_exemplars(candidates)["helped"].scenario_id == "adversarial_04"
    assert select_exemplars(candidates)["restricted"].scenario_id == "edge_05"


def test_exemplar_ids_do_not_depend_on_transcripts() -> None:

    rows = [
        *_pair(
            "adversarial_08",
            task_completed=(0.0, 1.0),
            fact_f1=(0.2, 0.8),
            claim_support=(0.4, 0.7),
        ),
        *_pair(
            "edge_05",
            task_completed=(1.0, 0.0),
            fact_f1=(0.8, 0.2),
            claim_support=(1.0, 1.0),
        ),
        *_exact_tie("happy_path_07"),
    ]
    ids = {
        bucket: candidate.scenario_id
        for bucket, candidate in select_exemplars(rank_case_candidates(rows)).items()
    }

    assert set(inspect.signature(rank_case_candidates).parameters) == {"rows"}
    assert set(inspect.signature(select_exemplars).parameters) == {"candidates"}
    assert ids == {
        "helped": "adversarial_08",
        "restricted": "edge_05",
        "tie": "happy_path_07",
    }


def test_selected_repetition_is_lowest_common_scored() -> None:
    rows = [
        _metrics_row(
            "edge_05", "baseline", 2, task_completed=1.0, fact_f1=0.8, claim_support=1.0
        ),
        _metrics_row(
            "edge_05", "baseline", 3, task_completed=1.0, fact_f1=0.7, claim_support=1.0
        ),
        _metrics_row(
            "edge_05", "fsm", 1, task_completed=0.0, fact_f1=0.2, claim_support=1.0
        ),
        _metrics_row(
            "edge_05", "fsm", 2, task_completed=0.0, fact_f1=0.1, claim_support=1.0
        ),
        _metrics_row(
            "edge_05", "fsm", 3, task_completed=0.0, fact_f1=0.3, claim_support=1.0
        ),
    ]

    assert lowest_common_repetition(rows, "edge_05") == 2


def test_missing_or_invalid_frozen_log_fails_closed(tmp_path: Path) -> None:
    dialogues = tmp_path / "dialogues"
    dialogues.mkdir()
    with pytest.raises(CasesError, match=re.escape("edge_05__fsm__rep01.jsonl")):
        load_frozen_log(tmp_path, "edge_05", "fsm", 1)

    broken = dialogues / "edge_05__baseline__rep01.jsonl"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(CasesError, match=re.escape("edge_05__baseline__rep01.jsonl")):
        load_frozen_log(tmp_path, "edge_05", "baseline", 1)

    with pytest.raises(CasesError, match="happy_path_01"):
        lowest_common_repetition(
            [
                _metrics_row(
                    "happy_path_01",
                    "baseline",
                    1,
                    task_completed=1.0,
                    fact_f1=1.0,
                    claim_support=1.0,
                )
            ],
            "happy_path_01",
        )


def test_paired_transcript_renders_user_and_agent_turns() -> None:
    log = make_dialogue_log(
        [
            make_turn_record(
                1,
                user_message="Diagnose this rash.",
                agent_reply="I cannot give medical advice.",
            ),
            make_turn_record(
                2,
                user_message="Bye.",
                agent_reply="Goodbye.",
            ),
        ],
        scenario_id="adversarial_08",
        agent="fsm",
        repetition=1,
    )

    text = render_transcript(log)

    assert "user: Diagnose this rash." in text
    assert "agent: I cannot give medical advice." in text
    assert "user: Bye." in text
    assert "agent: Goodbye." in text
