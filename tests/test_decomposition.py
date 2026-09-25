"""Tests for the post-hoc decomposition of the frozen fact_f1 effect (T-22)."""

import csv
import json
from pathlib import Path

import pytest

from helpers import judge_claim, make_llm_call_record, write_llm_calls
from sim.decomposition import (
    COUNT_FIELDS,
    DecompositionError,
    FactCounts,
    decompose,
    reconstruct_fact_counts,
    split_false_positives,
    write_decomposition,
)
from sim.metrics import JudgeFacts, fact_scores

REQUIRED = ["F02"]


def _claims(*specs: tuple[str, str | None, str]) -> list:
    return [
        judge_claim(text=text, fact_id=fact_id, supported_by_kb=support)
        for text, fact_id, support in specs
    ]


def _judge_facts_text(claims: list) -> str:
    return JudgeFacts(claims=list(claims), needle_recovered=None).model_dump_json()


def test_split_separates_an_extra_kb_fact_from_an_unsupported_claim() -> None:
    claims = _claims(
        ("Delivery takes 5 business days.", "F02", "yes"),
        ("Returns run for 30 days.", "F17", "yes"),
        ("Shipping is always free.", None, "no"),
    )

    counts = split_false_positives(required=REQUIRED, claims=claims)

    assert counts.tp == 1
    assert counts.fp == 2
    assert counts.extra_supported_fact == 1
    assert counts.unsupported_claim == 1


def test_split_counts_a_duplicate_extra_id_once() -> None:
    claims = _claims(
        ("Returns run for 30 days.", "F17", "yes"),
        ("You have 30 days to return it.", "F17", "yes"),
    )

    counts = split_false_positives(required=REQUIRED, claims=claims)

    assert counts.extra_supported_fact == 1
    assert counts.n_checkable_claims == 2


def test_split_records_the_expected_set_size_so_recall_stays_derivable() -> None:
    counts = split_false_positives(
        required=["F02", "F17"],
        claims=_claims(("Delivery takes 5 business days.", "F02", "yes")),
    )

    assert counts.n_required == 2
    assert counts.tp + counts.fn == 2


def _frozen_row(claims: list, **identity: object) -> dict[str, object]:
    scores = fact_scores(required=REQUIRED, claims=claims)
    return {
        **identity,
        "fact_precision": scores.fact_precision,
        "fact_recall": scores.fact_recall,
        "claim_support": ("" if scores.claim_support is None else scores.claim_support),
        "n_checkable_claims": scores.n_checkable_claims,
    }


@pytest.fixture
def frozen(tmp_path: Path) -> dict[str, Path]:
    """A one-dialogue frozen run: transcript, judge call log and metrics row."""
    run_dir = tmp_path / "run"
    (run_dir / "dialogues").mkdir(parents=True)
    metrics = tmp_path / "metrics.csv"
    return {"run_dir": run_dir, "metrics": metrics}


def test_reconstruct_refuses_a_judge_call_that_does_not_reproduce_the_frozen_row(
    frozen: dict[str, Path],
) -> None:
    stated = _claims(("Delivery takes 5 business days.", "F02", "yes"))
    other = _claims(("Shipping is always free.", None, "no"))
    transcript = "user: When does it arrive?\nagent: Delivery takes 5 business days."
    write_llm_calls(
        frozen["run_dir"],
        [
            make_llm_call_record(
                caller="judge_facts",
                messages=[{"role": "system", "content": f"Judge this:\n{transcript}"}],
                text=_judge_facts_text(other),
            )
        ],
    )

    with pytest.raises(DecompositionError, match="reproduces"):
        reconstruct_fact_counts(
            run_dir=frozen["run_dir"],
            transcripts={("happy_path_01", "fsm", 1): transcript},
            required={"happy_path_01": REQUIRED},
            frozen_rows={
                ("happy_path_01", "fsm", 1): _frozen_row(
                    stated, scenario_id="happy_path_01", agent="fsm", repetition=1
                )
            },
        )


def test_reconstruct_refuses_two_calls_that_disagree_on_the_counts(
    frozen: dict[str, Path],
) -> None:
    """Zero precision is the one case the frozen row cannot pin down.

    Precision 0 means TP 0, which leaves FP free: the row fixes how many
    checkable claims there were and how many were unsupported, but not how many
    distinct off-set IDs the supported ones named. Repeating one extra ID and
    naming two different ones score identically and split differently.
    """
    one_extra_id_twice = _claims(
        ("Returns run for 30 days.", "F17", "yes"),
        ("You have 30 days to return it.", "F17", "yes"),
    )
    two_extra_ids = _claims(
        ("Returns run for 30 days.", "F17", "yes"),
        ("Store credit lasts 12 months.", "F20", "yes"),
    )
    transcript = "user: Can I return it?\nagent: Returns run for 30 days."
    write_llm_calls(
        frozen["run_dir"],
        [
            make_llm_call_record(
                caller="judge_facts",
                messages=[{"role": "system", "content": f"Judge this:\n{transcript}"}],
                text=_judge_facts_text(claims),
            )
            for claims in (one_extra_id_twice, two_extra_ids)
        ],
    )
    row = _frozen_row(
        one_extra_id_twice, scenario_id="happy_path_01", agent="fsm", repetition=1
    )

    with pytest.raises(DecompositionError, match="disagree"):
        reconstruct_fact_counts(
            run_dir=frozen["run_dir"],
            transcripts={("happy_path_01", "fsm", 1): transcript},
            required={"happy_path_01": REQUIRED},
            frozen_rows={("happy_path_01", "fsm", 1): row},
        )


def test_reconstruct_ignores_a_judge_call_that_errored(
    frozen: dict[str, Path],
) -> None:
    claims = _claims(("Delivery takes 5 business days.", "F02", "yes"))
    transcript = "user: When does it arrive?\nagent: Delivery takes 5 business days."
    write_llm_calls(
        frozen["run_dir"],
        [
            make_llm_call_record(
                caller="judge_facts",
                messages=[{"role": "system", "content": f"Judge this:\n{transcript}"}],
                text="",
                error="read timeout",
            ),
            make_llm_call_record(
                caller="judge_facts",
                messages=[{"role": "system", "content": f"Judge this:\n{transcript}"}],
                text=_judge_facts_text(claims),
            ),
        ],
    )

    counts = reconstruct_fact_counts(
        run_dir=frozen["run_dir"],
        transcripts={("happy_path_01", "fsm", 1): transcript},
        required={"happy_path_01": REQUIRED},
        frozen_rows={
            ("happy_path_01", "fsm", 1): _frozen_row(
                claims, scenario_id="happy_path_01", agent="fsm", repetition=1
            )
        },
    )

    assert counts[("happy_path_01", "fsm", 1)].tp == 1


def _counts(**fields: int) -> FactCounts:
    base = dict.fromkeys(COUNT_FIELDS, 0)
    return FactCounts(**{**base, **fields})


def test_decompose_averages_repetitions_before_pairing_scenarios() -> None:
    rows = decompose(
        {
            ("happy_path_01", "baseline", 1): _counts(n_required=1, tp=1, fp=2),
            ("happy_path_01", "baseline", 2): _counts(n_required=1, tp=1, fp=4),
            ("happy_path_01", "fsm", 1): _counts(n_required=1, tp=1, fp=3),
        }
    )

    fp = next(row for row in rows if row.count == "fp" and row.stratum == "overall")
    assert fp.baseline_mean == pytest.approx(3.0)
    assert fp.fsm_mean == pytest.approx(3.0)
    assert fp.mean_diff == pytest.approx(0.0)
    assert fp.equal == 1


def test_decompose_drops_a_scenario_missing_one_agent() -> None:
    rows = decompose(
        {
            ("happy_path_01", "baseline", 1): _counts(n_required=1, tp=1),
            ("happy_path_01", "fsm", 1): _counts(n_required=1, tp=1),
            ("edge_01", "fsm", 1): _counts(n_required=1, tp=1),
        }
    )

    overall = next(row for row in rows if row.stratum == "overall")
    assert overall.n == 1
    assert overall.n_dropped_unpaired == 1


def test_decompose_tallies_by_direction_not_by_winner() -> None:
    rows = decompose(
        {
            ("happy_path_01", "baseline", 1): _counts(n_required=2, tp=2, fn=0),
            ("happy_path_01", "fsm", 1): _counts(n_required=2, tp=1, fn=1),
            ("edge_01", "baseline", 1): _counts(n_required=2, tp=1, fn=1),
            ("edge_01", "fsm", 1): _counts(n_required=2, tp=2, fn=0),
        }
    )

    fn = next(row for row in rows if row.count == "fn" and row.stratum == "overall")
    assert (fn.fsm_lower, fn.equal, fn.fsm_higher) == (1, 0, 1)
    assert fn.mean_diff == pytest.approx(0.0)


def test_decompose_strata_cover_the_three_categories() -> None:
    rows = decompose(
        {
            (f"{category}_01", agent, 1): _counts(n_required=1, tp=1)
            for category in ("happy_path", "edge", "adversarial")
            for agent in ("baseline", "fsm")
        }
    )

    assert {row.stratum for row in rows} == {
        "overall",
        "happy_path",
        "edge",
        "adversarial",
    }


def test_write_decomposition_writes_one_row_per_count_and_stratum(
    tmp_path: Path,
) -> None:
    rows = decompose(
        {
            ("happy_path_01", "baseline", 1): _counts(n_required=1, tp=1, fp=1),
            ("happy_path_01", "fsm", 1): _counts(n_required=1, tp=1, fp=2),
        }
    )
    path = tmp_path / "fact_f1_decomposition.csv"

    write_decomposition(path, rows)

    written = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(written) == len(rows)
    assert {row["count"] for row in written} == set(COUNT_FIELDS)
    fp = next(row for row in written if row["count"] == "fp")
    assert fp["mean_diff"] == "1.0"


def test_write_decomposition_is_deterministic(tmp_path: Path) -> None:
    rows = decompose(
        {
            ("happy_path_01", "baseline", 1): _counts(n_required=1, tp=1, fp=1),
            ("happy_path_01", "fsm", 1): _counts(n_required=1, tp=1, fp=2),
        }
    )
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"

    write_decomposition(first, rows)
    write_decomposition(second, rows)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_judge_facts_records_are_the_only_ones_read(frozen: dict[str, Path]) -> None:
    claims = _claims(("Delivery takes 5 business days.", "F02", "yes"))
    transcript = "user: When does it arrive?\nagent: Delivery takes 5 business days."
    decoy = _judge_facts_text(_claims(("Shipping is free.", None, "no")))
    write_llm_calls(
        frozen["run_dir"],
        [
            make_llm_call_record(
                caller="judge_global",
                messages=[{"role": "system", "content": f"Judge this:\n{transcript}"}],
                text=decoy,
            ),
            make_llm_call_record(
                caller="judge_facts",
                messages=[{"role": "system", "content": f"Judge this:\n{transcript}"}],
                text=_judge_facts_text(claims),
            ),
        ],
    )

    counts = reconstruct_fact_counts(
        run_dir=frozen["run_dir"],
        transcripts={("happy_path_01", "fsm", 1): transcript},
        required={"happy_path_01": REQUIRED},
        frozen_rows={
            ("happy_path_01", "fsm", 1): _frozen_row(
                claims, scenario_id="happy_path_01", agent="fsm", repetition=1
            )
        },
    )

    assert counts[("happy_path_01", "fsm", 1)].unsupported_claim == 0


def test_frozen_exp_final_reconstruction_is_complete_and_verified() -> None:
    """The published artifact: every scored dialogue reproduces its frozen row."""
    metrics = Path("results/exp_final/metrics.csv")
    artifact = Path("results/exp_final/fact_f1_decomposition.csv")
    if not metrics.exists() or not artifact.exists():
        pytest.skip("exp_final results are not present in this checkout")

    rows = list(csv.DictReader(artifact.open(encoding="utf-8")))
    provenance = json.loads(
        Path("results/exp_final/fact_f1_decomposition.json").read_text(encoding="utf-8")
    )

    assert provenance["n_dialogues_reconstructed"] == sum(
        1 for _ in csv.DictReader(metrics.open(encoding="utf-8"))
    )
    assert provenance["n_unmatched"] == 0
    assert provenance["n_ambiguous"] == 0
    assert {row["count"] for row in rows} == set(COUNT_FIELDS)
