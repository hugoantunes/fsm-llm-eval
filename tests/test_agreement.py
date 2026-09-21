"""Tests for scripts/validate_judge.py (T-16)."""

import csv
import json
import shutil
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
    assert report.true_positive == 3
    assert report.false_positive == 1
    assert report.false_negative == 0


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


def test_agree_out_writes_json_and_prediction_csvs(
    agreement_sample: Path, tmp_path: Path
) -> None:
    report = validate.agree(
        agreement_sample / "sample",
        agreement_sample / "run",
    )
    out = tmp_path / "out"
    validate.write_agreement_artifacts(report, out)

    payload = json.loads((out / "agreement.json").read_text(encoding="utf-8"))
    assert payload["dialogue"]["n"] == report.dialogue.n
    assert payload["dialogue"]["task_completed"]["percent"] == (
        report.dialogue.task_completed.percent
    )
    assert payload["dialogue"]["task_completed"]["kappa"] == (
        report.dialogue.task_completed.kappa
    )
    assert payload["response"]["fact_ids"]["f1"] == report.response.fact_ids.f1
    assert payload["response"]["unmatched_claims"] == len(
        report.response.unmatched_claims
    )
    dialogue_rows = _read_csv(out / "predictions_dialogue.csv")
    assert [row["dialogue_annotation_id"] for row in dialogue_rows] == ["D01", "D02"]
    assert dialogue_rows[0]["judge_task_completed"] == "yes"
    assert dialogue_rows[0]["judge_accuracy"] == "correct"
    response_rows = _read_csv(out / "predictions_response.csv")
    assert response_rows[0]["annotation_id"] == "A01"
    assert response_rows[0]["judge_claim_support"] == "all_supported"
    assert response_rows[0]["judge_fact_ids"] == "F10"


def test_compare_reuses_agree_and_lists_only_changed_judge_labels(
    agreement_sample: Path, tmp_path: Path
) -> None:
    sample = agreement_sample / "sample"
    original = agreement_sample / "run"
    mlx = _mlx_variant(agreement_sample)
    compared = validate.compare_runs(sample, original, mlx)
    original_report = validate.agree(sample, original)
    mlx_report = validate.agree(sample, mlx)

    assert compared.original.dialogue == original_report.dialogue
    assert compared.original.response.claim_support == (
        original_report.response.claim_support
    )
    assert compared.original.response.fact_ids == original_report.response.fact_ids
    assert compared.mlx.dialogue == mlx_report.dialogue
    assert compared.mlx.response.fact_ids == mlx_report.response.fact_ids

    out = tmp_path / "out"
    validate.write_comparison_artifacts(
        compared,
        out,
        environment={"model_tag_observed": "gemma4:12b-mlx"},
    )
    comparison = _read_csv(out / "comparison.csv")
    by_metric = {row["metric"]: row for row in comparison}
    assert float(by_metric["task_completed_agreement"]["delta"]) == pytest.approx(
        mlx_report.dialogue.task_completed.percent
        - original_report.dialogue.task_completed.percent
    )
    assert float(by_metric["accuracy_agreement"]["delta"]) == 0.0
    changed = _read_csv(out / "changed_cases.csv")
    assert changed == [
        {
            "unit": "D02",
            "field": "task_completed",
            "original": "no",
            "mlx": "yes",
        },
        {
            "unit": "A01",
            "field": "fact_ids_stated",
            "original": "F10",
            "mlx": "F11",
        },
    ]
    env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert env["model_tag_observed"] == "gemma4:12b-mlx"


def test_judge_environment_separates_observed_from_configured(
    agreement_sample: Path,
) -> None:
    run_dir = agreement_sample / "run"
    records = [
        record.model_copy(
            update={
                "model": "gemma4:12b-mlx",
                "digest": "ab" * 32,
            }
        )
        for record in validate._load_calls(run_dir / "llm_calls.jsonl")
    ]
    write_llm_calls(run_dir, records)

    env = validate.judge_environment(run_dir, ollama_version="0.99.0")

    assert env["model_tag_observed"] == "gemma4:12b-mlx"
    assert env["model_digest_observed"] == "ab" * 32
    assert env["ollama_version_observed"] == "0.99.0"
    assert env["temperature_configured"] == 0
    assert env["seed_configured"] == 42
    assert env["num_ctx_configured"] == 8192
    assert env["judge_facts_prompt_version"] == "v3"
    assert env["judge_global_prompt_version"] == "v2"
    assert env["judge_shared_prompt_version"] == "v2"


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


def test_compare_labeler_counts_corrected_remaining_and_new_errors(
    tmp_path: Path,
) -> None:
    records = [
        make_turn_record(
            1,
            state_before="greeting",
            state_after="identification",
        ),
        make_turn_record(
            2,
            state_before="identification",
            state_after="intent_classification",
        ),
        make_turn_record(
            3,
            state_before="intent_classification",
            state_after="closing",
        ),
    ]
    log = make_dialogue_log(records, scenario_id="happy_path_01", agent="fsm")
    original = write_run(
        tmp_path / "original",
        [log],
        scenarios_dir="data/scenarios/examples",
    )
    sidecar = write_run(
        tmp_path / "sidecar",
        [log],
        scenarios_dir="data/scenarios/examples",
    )
    _write_csv(
        original / "metrics_turn.csv",
        [
            turn.model_dump()
            for turn in labelled_turns(log, ["identification", "greeting", "solution"])
        ],
    )
    _write_csv(
        sidecar / "metrics_turn.csv",
        [
            turn.model_dump()
            for turn in labelled_turns(
                log, ["greeting", "intent_classification", "solution"]
            )
        ],
    )

    report = validate.compare_labeler(original, sidecar)

    assert report.original.n_correct == 1
    assert report.sidecar.n_correct == 1
    assert report.n_corrected == 1
    assert report.n_remaining == 1
    assert report.n_new == 1
    kinds = {row["kind"]: row for row in report.changed_turns}
    assert kinds["corrected"]["turn"] == "2"
    assert kinds["corrected"]["sidecar"] == "intent_classification"
    assert kinds["remaining"]["turn"] == "3"
    assert kinds["new"]["turn"] == "1"
    assert kinds["new"]["original"] == "identification"
    assert kinds["new"]["sidecar"] == "greeting"

    out = tmp_path / "out"
    validate.write_labeler_comparison_artifacts(report, out)
    comparison = _read_csv(out / "labeler_comparison.csv")
    changed = _read_csv(out / "labeler_changed_turns.csv")
    assert {row["metric"] for row in comparison} >= {
        "turn_correct",
        "path_agreement",
        "corrected",
        "remaining",
        "new",
    }
    assert {row["kind"] for row in changed} == {"corrected", "remaining", "new"}


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


def _mlx_variant(agreement_sample: Path) -> Path:
    """Copy the fixture run and change one dialogue field and one fact id."""
    original = agreement_sample / "run"
    mlx = agreement_sample / "mlx"
    shutil.copytree(original, mlx)
    _write_csv(
        mlx / "metrics.csv",
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
                "task_completed": "True",
            },
        ],
    )
    facts = judge_facts_reply(
        [
            judge_claim(
                text="UniqueAlpha",
                fact_id="F11",
                supported_by_kb="yes",
            ),
            judge_claim(
                text="Refunds take five days",
                fact_id=None,
                supported_by_kb="unverifiable",
            ),
        ]
    )
    records = []
    for record in validate._load_calls(mlx / "llm_calls.jsonl"):
        if record.prompt_hash == "facts-baseline":
            records.append(record.model_copy(update={"text": facts}))
            continue
        records.append(record)
    write_llm_calls(mlx, records)
    return mlx


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write ``rows`` to ``path`` with the first row's keys as the header."""
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
