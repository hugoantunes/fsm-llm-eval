"""Lightweight checks for scripts/score_human_validation.py."""

import json
from pathlib import Path

import pytest

from helpers import REPO_ROOT, load_script
from sim.audit import file_sha256
from sim.kb import load_kb
from sim.metrics import judge_claim
from sim.sensitivity import load_dialogue_keys

scorer = load_script("scripts/score_human_validation.py")


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _frozen_packet(tmp_path: Path) -> Path:
    """Two complete frozen sheets plus SHA-256."""
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    _write_csv(
        frozen / "dialogue_annotations.csv",
        "blind_dialogue_id,task_completed,notes",
        ["D001,no,", "D002,yes,"],
    )
    _write_csv(
        frozen / "response_annotations.csv",
        "blind_dialogue_id,blind_response_id,fact_ids_stated,notes",
        ["D001,D001-A01,F28,", "D002,D002-A01,F26;F28,"],
    )
    _write_csv(
        frozen / "claims_annotations.csv",
        "blind_dialogue_id,blind_response_id,claim_id,claim_text,support,notes",
        [
            "D001,D001-A01,D001-A01-C01,refund in 5 days,supported,",
            "D001,D001-A01,D001-A01-C02,cancelled immediately,unsupported,",
        ],
    )
    lines = ["# freeze"]
    for name in (
        "dialogue_annotations.csv",
        "response_annotations.csv",
        "claims_annotations.csv",
    ):
        lines.append(f"{file_sha256(frozen / name)}  {name}")
    (frozen / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frozen


def test_validate_frozen_sheets_rejects_empty_task_completed(tmp_path: Path) -> None:
    frozen = _frozen_packet(tmp_path)
    path = frozen / "dialogue_annotations.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("D001,no,", "D001,,"),
        encoding="utf-8",
    )
    digest = file_sha256(path)
    rewritten = []
    for line in (frozen / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.endswith("dialogue_annotations.csv"):
            rewritten.append(f"{digest}  dialogue_annotations.csv")
        else:
            rewritten.append(line)
    (frozen / "SHA256SUMS.txt").write_text(
        "\n".join(rewritten) + "\n", encoding="utf-8"
    )
    with pytest.raises(scorer.ScoreError, match="task_completed"):
        scorer.validate_frozen_sheets(
            frozen, kb=load_kb(), n_dialogues=2, n_responses=2
        )


def test_validate_frozen_sheets_rejects_unknown_fact_id(tmp_path: Path) -> None:
    frozen = _frozen_packet(tmp_path)
    path = frozen / "response_annotations.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("F28", "F99"), encoding="utf-8"
    )
    digest = file_sha256(path)
    rewritten = []
    for line in (frozen / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.endswith("response_annotations.csv"):
            rewritten.append(f"{digest}  response_annotations.csv")
        else:
            rewritten.append(line)
    (frozen / "SHA256SUMS.txt").write_text(
        "\n".join(rewritten) + "\n", encoding="utf-8"
    )
    with pytest.raises(scorer.ScoreError, match="unknown fact"):
        scorer.validate_frozen_sheets(
            frozen, kb=load_kb(), n_dialogues=2, n_responses=2
        )


def test_validate_frozen_sheets_does_not_rewrite_csvs(tmp_path: Path) -> None:
    frozen = _frozen_packet(tmp_path)
    before = file_sha256(frozen / "dialogue_annotations.csv")
    scorer.validate_frozen_sheets(frozen, kb=load_kb(), n_dialogues=2, n_responses=2)
    assert file_sha256(frozen / "dialogue_annotations.csv") == before


def test_derive_human_scores_uses_fact_scores_and_leaves_uncertain() -> None:
    mapping = [
        scorer.MappingRow(
            blind_id="D001",
            dialogue_id="happy_path_09__baseline__rep01",
            scenario_id="happy_path_09",
            condition="baseline",
            repetition=1,
        )
    ]
    scores = scorer.derive_human_scores(
        mapping=mapping,
        dialogues=[{"blind_dialogue_id": "D001", "task_completed": "uncertain"}],
        responses=[
            {
                "blind_dialogue_id": "D001",
                "blind_response_id": "D001-A01",
                "fact_ids_stated": "F28",
            }
        ],
        claims=[
            {
                "blind_dialogue_id": "D001",
                "claim_text": "refund in 5 days",
                "support": "supported",
            },
            {
                "blind_dialogue_id": "D001",
                "claim_text": "immediately",
                "support": "unsupported",
            },
            {
                "blind_dialogue_id": "D001",
                "claim_text": "maybe",
                "support": "uncertain",
            },
        ],
        required={"happy_path_09": ["F26", "F28"]},
    )
    assert len(scores) == 1
    row = scores[0]
    assert row.task_completed == "uncertain"
    assert row.predicted_ids == frozenset({"F28"})
    assert row.n_checkable_claims == 2
    assert row.n_uncertain_claims == 1
    assert row.claim_support == pytest.approx(0.5)
    assert row.claim_support_label == "some_unsupported"
    assert row.n_true_positives == 1
    assert row.n_false_negatives == 1


def test_human_metrics_rows_leave_uncertain_task_completed_blank() -> None:
    mapping = scorer.MappingRow(
        blind_id="D001",
        dialogue_id="x",
        scenario_id="happy_path_09",
        condition="baseline",
        repetition=1,
    )
    scores = scorer.derive_human_scores(
        mapping=[mapping],
        dialogues=[{"blind_dialogue_id": "D001", "task_completed": "uncertain"}],
        responses=[
            {
                "blind_dialogue_id": "D001",
                "blind_response_id": "D001-A01",
                "fact_ids_stated": "",
            }
        ],
        claims=[],
        required={"happy_path_09": ["F26"]},
    )
    rows = scorer.human_metrics_rows(scores)
    assert rows[0]["task_completed"] == ""
    assert rows[0]["claim_support"] is None


def test_compare_instrument_drops_uncertain_task_completed() -> None:
    mapping = scorer.MappingRow(
        blind_id="D001",
        dialogue_id="happy_path_09__baseline__rep01",
        scenario_id="happy_path_09",
        condition="baseline",
        repetition=1,
    )
    human = scorer.derive_human_scores(
        mapping=[mapping],
        dialogues=[{"blind_dialogue_id": "D001", "task_completed": "uncertain"}],
        responses=[
            {
                "blind_dialogue_id": "D001",
                "blind_response_id": "D001-A01",
                "fact_ids_stated": "F28",
            }
        ],
        claims=[],
        required={"happy_path_09": ["F28"]},
    )
    judge_rows = {
        mapping.key: {
            "task_completed": "True",
            "fact_precision": "1",
            "fact_recall": "1",
            "claim_support": "1",
            "n_checkable_claims": "1",
        }
    }
    claims = {
        mapping.key: [judge_claim(text="refund", fact_id="F28", supported_by_kb="yes")]
    }
    report = scorer.compare_instrument(human, judge_rows, claims)
    assert report["task_completed"]["n"] == 0
    assert report["task_completed"]["n_uncertain_excluded"] == 1
    assert report["fact_ids_stated"]["exact_match_rate"] == 1.0
    assert report["claim_support_label"]["alignment"].startswith("dialogue-level")


def test_fact_id_set_agreement_treats_human_as_gold() -> None:
    stats = scorer.fact_id_set_agreement(
        [{"F01", "F02"}],
        [{"F01", "F03"}],
    )
    assert stats["true_positive"] == 1
    assert stats["false_positive"] == 1
    assert stats["false_negative"] == 1


def test_frozen_census_matches_committed_sheets() -> None:
    frozen = REPO_ROOT / "results" / "human_validation" / "exp_final" / "frozen"
    scorer.validate_frozen_sheets(
        frozen, kb=load_kb(), n_dialogues=348, n_responses=1306
    )


def test_committed_ambiguous_list_matches_agreement_count() -> None:
    human_primary = REPO_ROOT / "results" / "human_primary"
    keys = load_dialogue_keys(human_primary / "ambiguous_fact_ids.csv")
    agreement = json.loads((human_primary / "agreement.json").read_text("utf-8"))

    omitted = agreement["fact_ids_stated"]["n_omitted_ambiguous_judge_calls"]
    assert len(set(keys)) == len(keys) == omitted
