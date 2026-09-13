"""Tests for scripts/validate_judge.py (T-16)."""

import csv
import json
from pathlib import Path

import pytest

from helpers import (
    judge_facts_reply,
    load_script,
    make_dialogue_log,
    make_llm_call_record,
    make_turn_record,
    write_llm_calls,
    write_run,
)
from sim.evaluators import labelled_turns
from sim.metrics import judge_claim

validate = load_script("scripts/validate_judge.py")


def test_percent_agreement_is_the_share_of_equal_pairs() -> None:
    assert (
        validate.percent_agreement(
            ["correct", "partial", "incorrect", "correct"],
            ["correct", "partial", "correct", "correct"],
        )
        == 0.75
    )


def test_cohen_kappa_is_one_on_perfect_agreement_and_zero_at_chance() -> None:
    perfect = ["yes", "no", "yes"]
    chance_left = ["a", "a", "b", "b"]
    chance_right = ["a", "b", "a", "b"]

    assert validate.cohen_kappa(perfect, perfect) == 1.0
    assert validate.cohen_kappa(chance_left, chance_right) == 0.0


def test_quadratic_weighted_kappa_penalises_ordinal_accuracy_by_distance() -> None:
    human = ["incorrect", "partial", "correct", "correct"]
    near = ["partial", "incorrect", "correct", "correct"]
    far = ["correct", "partial", "incorrect", "correct"]

    near_kappa = validate.quadratic_weighted_kappa(human, near)
    far_kappa = validate.quadratic_weighted_kappa(human, far)
    unweighted_near = validate.cohen_kappa(human, near)
    unweighted_far = validate.cohen_kappa(human, far)

    assert near_kappa > far_kappa
    assert unweighted_near == unweighted_far
    assert validate.quadratic_weighted_kappa(human, human) == 1.0


def test_fact_id_set_agreement_reports_exact_match_and_micro_f1() -> None:
    report = validate.fact_id_set_agreement(
        [{"F01", "F02"}, {"F03"}, set()],
        [{"F01", "F02"}, {"F03", "F04"}, set()],
    )

    assert report.n == 3
    assert report.exact_match_rate == pytest.approx(2 / 3)
    assert report.precision == pytest.approx(0.75)
    assert report.recall == pytest.approx(1.0)
    assert report.f1 == pytest.approx(0.8571428571)


def test_turn_claims_are_attributed_by_claim_text_and_unmatched_are_reported() -> None:
    turns = (
        (1, "UniqueAlpha. Refunds take five days."),
        (2, "Refunds take five days after dispatch."),
    )
    hit = judge_claim(
        text="UniqueAlpha",
        fact_id="F10",
        supported_by_kb="yes",
    )
    double = judge_claim(
        text="Refunds take five days",
        fact_id=None,
        supported_by_kb="unverifiable",
    )
    miss = judge_claim(
        text="a membership programme",
        fact_id=None,
        supported_by_kb="no",
    )

    attributed, unmatched = validate.attribute_claims(
        [hit, double, miss],
        turns,
    )

    assert attributed[1] == [hit]
    assert attributed[2] == []
    assert {item.claim.text for item in unmatched} == {double.text, miss.text}
    assert validate.claim_support_label(attributed[1]) == "all_supported"
    assert validate.claim_support_label(attributed[2]) == "none_checkable"


def test_agreement_joins_response_and_dialogue_sheets_on_sample_json(
    agreement_sample: Path,
) -> None:
    report = validate.agree(
        agreement_sample / "sample",
        agreement_sample / "run",
    )

    assert report.dialogue.n == 2
    assert report.dialogue.task_completed.percent == 1.0
    assert report.dialogue.task_completed.kappa == 1.0
    assert report.dialogue.accuracy.percent == 1.0
    assert report.dialogue.accuracy.weighted_kappa == 1.0
    assert report.response.n == 1
    assert report.response.claim_support.percent == 1.0
    assert report.response.fact_ids.exact_match_rate == 1.0
    assert {item.claim.text for item in report.response.unmatched_claims} == {
        "Refunds take five days"
    }


def test_labeler_report_scores_fsm_turns_and_path_agreement(
    labeler_run_dir: Path,
) -> None:
    report = validate.labeler_report(labeler_run_dir)

    assert report.n_fsm_turns == 2
    assert report.n_correct == 1
    assert report.turn_accuracy == 0.5
    assert report.n_fsm_dialogues == 1
    assert report.n_gold_valid_paths == 1
    assert report.n_labelled_valid_paths == 0
    assert report.n_path_agreement == 0


@pytest.fixture
def agreement_sample(tmp_path: Path) -> Path:
    """Two ok dialogues, one sampled response, dialogue census of two rows."""
    baseline = make_dialogue_log(
        [
            make_turn_record(
                1,
                user_message="Where is the order?",
                agent_reply="UniqueAlpha. Refunds take five days.",
            ),
            make_turn_record(
                2,
                user_message="Thanks, goodbye.",
                agent_reply="Refunds take five days after dispatch.",
            ),
        ],
        scenario_id="happy_path_01",
        agent="baseline",
        repetition=1,
    )
    fsm = make_dialogue_log(
        [
            make_turn_record(
                1,
                user_message="Do you sell pizza?",
                agent_reply="Restaurant requests are out of scope.",
            )
        ],
        scenario_id="happy_path_01",
        agent="fsm",
        repetition=1,
    )
    run_dir = write_run(
        tmp_path / "run",
        [baseline, fsm],
        scenarios_dir="data/scenarios/examples",
    )

    facts = judge_facts_reply(
        [
            judge_claim(
                text="UniqueAlpha",
                fact_id="F10",
                supported_by_kb="yes",
            ),
            judge_claim(
                text="Refunds take five days",
                fact_id=None,
                supported_by_kb="unverifiable",
            ),
        ]
    )
    other = judge_facts_reply(
        [
            judge_claim(
                text="Restaurant requests are out of scope.",
                fact_id="F01",
                supported_by_kb="yes",
            )
        ]
    )
    write_llm_calls(
        run_dir,
        [
            make_llm_call_record(
                caller="judge_facts",
                role="judge",
                prompt_hash="facts-baseline",
            ).model_copy(
                update={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Transcript\n\n"
                                "user: Where is the order?\n"
                                "agent: UniqueAlpha. Refunds take five days.\n"
                                "user: Thanks, goodbye.\n"
                                "agent: Refunds take five days after dispatch."
                            ),
                        }
                    ],
                    "text": facts,
                }
            ),
            make_llm_call_record(
                caller="judge_facts",
                role="judge",
                prompt_hash="facts-fsm",
            ).model_copy(
                update={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Transcript\n\n"
                                "user: Do you sell pizza?\n"
                                "agent: Restaurant requests are out of scope."
                            ),
                        }
                    ],
                    "text": other,
                }
            ),
        ],
    )
    _write_csv(
        run_dir / "metrics.csv",
        [
            {
                "scenario_id": "happy_path_01",
                "agent": "baseline",
                "repetition": "1",
                "accuracy_score": "1.0",
                "task_completed": "True",
            },
            {
                "scenario_id": "happy_path_01",
                "agent": "fsm",
                "repetition": "1",
                "accuracy_score": "0.5",
                "task_completed": "False",
            },
        ],
    )

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "sample.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "annotation_id": "A01",
                        "dialogue_id": "happy_path_01__baseline__rep01",
                        "scenario_id": "happy_path_01",
                        "agent": "baseline",
                        "repetition": 1,
                        "turn": 1,
                    }
                ],
                "dialogues": [
                    {
                        "dialogue_annotation_id": "D01",
                        "dialogue_id": "happy_path_01__baseline__rep01",
                        "scenario_id": "happy_path_01",
                        "agent": "baseline",
                        "repetition": 1,
                    },
                    {
                        "dialogue_annotation_id": "D02",
                        "dialogue_id": "happy_path_01__fsm__rep01",
                        "scenario_id": "happy_path_01",
                        "agent": "fsm",
                        "repetition": 1,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        sample_dir / "dialogue_annotations.csv",
        [
            {
                "dialogue_annotation_id": "D01",
                "accuracy": "correct",
                "task_completed": "yes",
                "notes": "",
            },
            {
                "dialogue_annotation_id": "D02",
                "accuracy": "partial",
                "task_completed": "no",
                "notes": "",
            },
        ],
    )
    _write_csv(
        sample_dir / "response_annotations.csv",
        [
            {
                "annotation_id": "A01",
                "fact_ids_stated": "F10",
                "claim_support": "all_supported",
                "notes": "",
            }
        ],
    )
    return tmp_path


@pytest.fixture
def labeler_run_dir(tmp_path: Path, two_turn_fsm_records: list) -> Path:
    """One FSM dialogue where one stage label is wrong and path validity differs."""
    log = make_dialogue_log(
        two_turn_fsm_records,
        scenario_id="happy_path_01",
        agent="fsm",
    )
    run_dir = write_run(
        tmp_path / "run",
        [log],
        scenarios_dir="data/scenarios/examples",
    )
    labels = [two_turn_fsm_records[0].state_after, "solution"]
    turns = labelled_turns(log, labels)
    _write_csv(run_dir / "metrics_turn.csv", [turn.model_dump() for turn in turns])
    return run_dir


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write ``rows`` to ``path`` with the first row's keys as the header."""
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
