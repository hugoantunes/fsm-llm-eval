"""Tests for docs/decisions_and_limitations.md (T-23)."""

import json
import re
from csv import DictReader

import pytest

from helpers import DOCS_DIR, FROZEN_V1_HASH, REPO_ROOT
from sim.kb import KnowledgeBase
from sim.metrics import PRIMARY_METRICS
from sim.reporting import UNPAIRED_SCENARIO

TESTS_CSV = REPO_ROOT / "results" / "exp_final" / "tests.csv"


def _markdown_table_section(doc: str, heading: str) -> str:
    _, _, rest = doc.partition(heading)
    section, _, _ = rest.partition("\n### ")
    return section


def _kappa_cell(section: str, field: str) -> str:
    for line in section.splitlines():
        if line.startswith(f"| `{field}`"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            return cells[-1].split()[0]
    raise AssertionError(f"no kappa row for {field}")


def _census_int(audit: str, label: str) -> int:
    match = re.search(rf"^{re.escape(label)}\s+(\d+)\s*$", audit, flags=re.M)
    if match is None:
        raise AssertionError(f"missing audit census line {label!r}")
    return int(match.group(1))


def _tests_csv_n(stratum: str) -> int:
    with TESTS_CSV.open(encoding="utf-8") as handle:
        rows = list(DictReader(handle))
    matches = [
        int(row["n"])
        for row in rows
        if row["population"] == "semantic_primary"
        and row["stratum"] == stratum
        and row["metric"] == PRIMARY_METRICS[0]
    ]
    assert len(matches) == 1, stratum
    return matches[0]


def test_quoted_instrument_numbers_match_frozen_sources(
    decisions_doc: str,
    judge_validation_doc: str,
    metrics_doc: str,
    real_kb: KnowledgeBase,
) -> None:
    audit = (DOCS_DIR / "audit.md").read_text(encoding="utf-8")
    pilot_v2 = _markdown_table_section(
        judge_validation_doc, "### Agreement table (Pilot v2)"
    )
    freeze = _markdown_table_section(judge_validation_doc, "### Agreement table\n")
    labeler = _markdown_table_section(
        judge_validation_doc, "### Stage labeler on Pilot v2"
    )
    turn_match = re.search(r": (\d+/\d+) = ", labeler)
    flow_match = re.search(r"`valid_flow_path`[^\n]*?(\d+/\d+)", labeler)
    dz_match = re.search(r"dz ≈ 0\.37", metrics_doc)
    n_overall = _tests_csv_n("overall")
    n_happy = _tests_csv_n("happy_path")
    n_edge = _tests_csv_n("edge")
    n_adversarial = _tests_csv_n("adversarial")
    n_facts = len(real_kb.facts)
    frozen_eligible = _census_int(audit, "frozen-gate eligible")
    semantic_eligible = _census_int(audit, "semantic-primary eligible")
    frozen_scored = _census_int(audit, "frozen-gate scored")
    semantic_scored = _census_int(audit, "semantic-primary scored")
    cfp = _census_int(audit, "semantic CFP rows")
    unscored = _census_int(audit, "locked eligible-but-unscored")
    max_turns = re.search(r"^\+ (\d+) max_turns_with_incomplete_beat$", audit, re.M)
    tif = re.search(r"^\+ (\d+) instrument_true_failure$", audit, re.M)

    assert turn_match is not None
    assert flow_match is not None
    assert dz_match is not None
    assert max_turns is not None
    assert tif is not None
    assert UNPAIRED_SCENARIO == "adversarial_12"

    task_k = _kappa_cell(pilot_v2, "task_completed")
    claim_k = _kappa_cell(pilot_v2, "claim_support")
    accuracy_k = _kappa_cell(pilot_v2, "accuracy")
    freeze_task_k = _kappa_cell(freeze, "task_completed")

    assert task_k == "0.667"
    assert claim_k == "0.533"
    assert accuracy_k == "0.569"
    assert turn_match.group(1) == "25/39"
    assert flow_match.group(1) == "8/10"
    sidecar_9b = _markdown_table_section(
        judge_validation_doc, "### Stage labeler (4B vs 9B)"
    )
    assert "27/39" in sidecar_9b
    assert "6/10" in sidecar_9b
    classifier_9b = _markdown_table_section(
        judge_validation_doc, "### Event classifier"
    )
    assert "23/28" in classifier_9b
    assert "21/21" in classifier_9b
    assert "19/21" in classifier_9b
    assert n_overall == 59
    assert (n_happy, n_edge, n_adversarial) == (20, 20, 19)
    assert frozen_eligible == 342
    assert semantic_eligible == 350
    assert frozen_scored == 340
    assert semantic_scored == 348
    assert cfp == 8
    assert unscored == 2
    assert max_turns.group(1) == "9"
    assert tif.group(1) == "1"

    assert task_k in decisions_doc
    assert claim_k in decisions_doc
    assert accuracy_k in decisions_doc
    assert freeze_task_k in decisions_doc
    assert "25/39" in decisions_doc
    assert "8/10" in decisions_doc
    assert "27/39" in decisions_doc
    assert "19/21" in decisions_doc
    assert FROZEN_V1_HASH in decisions_doc
    assert "qwen3.5:9b" in decisions_doc
    assert f"{n_facts} facts" in decisions_doc
    assert "N = 60" in decisions_doc
    assert f"n = {n_overall}" in decisions_doc
    assert UNPAIRED_SCENARIO in decisions_doc
    assert "dz ≈ 0.37" in decisions_doc
    assert f"{semantic_eligible} eligible / {semantic_scored} scored" in decisions_doc
    assert f"{frozen_eligible} eligible / {frozen_scored} scored" in decisions_doc
    assert f"{cfp} CFP" in decisions_doc
    assert f"{tif.group(1)} TIF" in decisions_doc or "One TIF" in decisions_doc
    assert f"{max_turns.group(1)} `max_turns`" in decisions_doc or (
        "Nine `max_turns`" in decisions_doc
    )
    assert "edge_01/fsm/2" in decisions_doc
    assert "edge_13/fsm/2" in decisions_doc
    assert f"{n_happy}/{n_edge}/{n_adversarial}" in decisions_doc
    assert "`gemma4:12b`" in decisions_doc
    assert "`gemma4:12b-mlx`" in decisions_doc


def test_quoted_human_primary_numbers_match_frozen_sources(
    decisions_doc: str,
) -> None:
    human_tests = REPO_ROOT / "results" / "human_primary" / "tests.csv"
    agreement_path = REPO_ROOT / "results" / "human_primary" / "agreement.json"
    with human_tests.open(encoding="utf-8") as handle:
        rows = list(DictReader(handle))
    claim = next(
        row
        for row in rows
        if row["population"] == "human_primary"
        and row["stratum"] == "overall"
        and row["metric"] == "claim_support"
    )
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    omitted = agreement["fact_ids_stated"]["n_omitted_ambiguous_judge_calls"]
    holm = float(claim["wilcoxon_p_holm"])
    low = float(claim["ci_low"])
    high = float(claim["ci_high"])

    assert omitted == 107
    assert f"{omitted}" in decisions_doc
    assert "0.048" in decisions_doc or "0,048" in decisions_doc
    assert holm == pytest.approx(0.04763857690616527)
    assert low == pytest.approx(0.002646158002280932)
    assert high == pytest.approx(0.08990136388626652)
    assert "2026-09-20T21:11:58Z" in decisions_doc
