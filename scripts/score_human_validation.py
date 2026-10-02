"""Score frozen human sheets, compare them to the judge, rerun T-19.

Reads ``results/human_validation/exp_final/frozen/`` and
``private/mapping.csv``. Does not edit the frozen CSVs or retune labels
from disagreements.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from csv import DictReader, DictWriter
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path

from sim.analysis import (
    HUMAN_PRIMARY,
    N_RESAMPLES,
    PRIMARY_METRICS,
    SEED,
    aggregate_to_scenarios,
    descriptive_table,
    environment_text,
    pair_scenarios,
    paired_test_table,
    parse_cell,
)
from sim.audit import file_sha256
from sim.decomposition import (
    DecompositionError,
    DialogueKey,
    reconstruct_judge_claims,
)
from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.io import atomic_write
from sim.judge import redact_canary
from sim.kb import DEFAULT_KB_DIR, FACT_ID_SCHEMA_PATTERN, KnowledgeBase, load_kb
from sim.metrics import JudgeClaim, fact_scores, judge_claim
from sim.schemas import DialogueLog, Scenario, load_scenarios, render_transcript
from sim.sensitivity import write_dialogue_keys

DEFAULT_PACKET = Path("results/human_validation/exp_final")
DEFAULT_METRICS = Path("results/exp_final/metrics.csv")
DEFAULT_JUDGE_TESTS = Path("results/exp_final/tests.csv")
DEFAULT_RUN = Path("runs/exp_final")
DEFAULT_SIDECAR = Path("runs/exp_final_semantic")
DEFAULT_OUT = Path("results/human_primary")
DEFAULT_SCENARIOS = Path("data/scenarios/v1")
DIALOGUE_CSV = "dialogue_annotations.csv"
RESPONSE_CSV = "response_annotations.csv"
CLAIMS_CSV = "claims_annotations.csv"
MAPPING_CSV = "mapping.csv"
MANIFEST_JSON = "manifest.json"
SHA256SUMS = "SHA256SUMS.txt"
TASK_VALUES = frozenset({"yes", "no", "uncertain"})
SUPPORT_VALUES = frozenset({"supported", "unsupported", "uncertain"})
HUMAN_METRICS = (
    "task_completed",
    "fact_precision",
    "fact_recall",
    "fact_f1",
    "claim_support",
    "unsupported_claim_rate",
    "n_checkable_claims",
)
SIGNIFICANT = 0.05


class ScoreError(RuntimeError):
    """Frozen sheets or join failed; the message says what to fix."""


@dataclass(frozen=True)
class MappingRow:
    """One unblinded dialogue identity."""

    blind_id: str
    dialogue_id: str
    scenario_id: str
    condition: str
    repetition: int

    @property
    def key(self) -> DialogueKey:
        """Identity tuple used by metrics.csv and the judge reconstruction."""
        return (self.scenario_id, self.condition, self.repetition)


@dataclass(frozen=True)
class HumanScores:
    """Derived dialogue-level scores from the frozen human sheets."""

    mapping: MappingRow
    task_completed: str
    predicted_ids: frozenset[str]
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    fact_precision: float
    fact_recall: float
    fact_f1: float
    claim_support: float | None
    unsupported_claim_rate: float | None
    n_checkable_claims: int
    n_uncertain_claims: int
    claim_support_label: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Load a UTF-8 CSV as dictionaries."""
    if not path.is_file():
        raise ScoreError(f"{path} is missing")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(DictReader(handle))


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    """Atomically write ``rows`` with empty cells for None."""
    buffer = StringIO()
    writer = DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: "" if row.get(key) is None else row.get(key) for key in fieldnames}
        )
    atomic_write(path, buffer.getvalue())


def parse_fact_ids(raw: str) -> list[str]:
    """Split ``F03;F07`` into catalogue IDs, rejecting empties in the middle."""
    stripped = raw.strip()
    if not stripped:
        return []
    ids = [part.strip() for part in stripped.split(";")]
    if any(not item for item in ids):
        raise ScoreError(f"empty token in fact_ids_stated {raw!r}")
    return ids


def parse_sha256sums(path: Path) -> dict[str, str]:
    """Read ``SHA256SUMS.txt`` as ``{filename: digest}``."""
    if not path.is_file():
        raise ScoreError(f"{path} is missing")
    recorded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, _, name = stripped.partition("  ")
        if len(digest) != 64 or not name:
            raise ScoreError(f"{path} has a malformed line: {line!r}")
        recorded[name] = digest
    return recorded


def validate_frozen_sheets(
    frozen_dir: Path,
    *,
    kb: KnowledgeBase,
    n_dialogues: int,
    n_responses: int,
) -> None:
    """Fail closed if the frozen packet is incomplete or uses unknown IDs."""
    recorded = parse_sha256sums(frozen_dir / SHA256SUMS)
    for name in (DIALOGUE_CSV, RESPONSE_CSV, CLAIMS_CSV):
        path = frozen_dir / name
        if not path.is_file():
            raise ScoreError(f"{path} is missing")
        expected = recorded.get(name)
        if expected is None:
            raise ScoreError(f"{frozen_dir / SHA256SUMS} does not hash {name}")
        digest = file_sha256(path)
        if digest != expected:
            raise ScoreError(
                f"{path} sha256 is {digest}, expected {expected}. "
                "Score the frozen copy; do not edit it"
            )
    catalogue = {fact.id for fact in kb.facts}
    dialogues = _read_csv(frozen_dir / DIALOGUE_CSV)
    responses = _read_csv(frozen_dir / RESPONSE_CSV)
    claims = _read_csv(frozen_dir / CLAIMS_CSV)
    ids = [row["blind_dialogue_id"] for row in dialogues]
    expected_ids = [f"D{n:03d}" for n in range(1, n_dialogues + 1)]
    if ids != expected_ids:
        raise ScoreError(f"{DIALOGUE_CSV} ids are not D001…D{n_dialogues:03d} in order")
    empty_task = [
        row["blind_dialogue_id"]
        for row in dialogues
        if row.get("task_completed", "").strip() not in TASK_VALUES
    ]
    if empty_task:
        raise ScoreError(f"task_completed missing or invalid on {empty_task[:5]}")
    if len(responses) != n_responses:
        raise ScoreError(
            f"{RESPONSE_CSV} has {len(responses)} rows, expected {n_responses}"
        )
    response_ids = [row["blind_response_id"] for row in responses]
    if len(response_ids) != len(set(response_ids)):
        raise ScoreError(f"{RESPONSE_CSV} has duplicate blind_response_id")
    dialogue_set = set(ids)
    for row in responses:
        if row["blind_dialogue_id"] not in dialogue_set:
            raise ScoreError(
                f"{row['blind_response_id']} names unknown {row['blind_dialogue_id']}"
            )
        if not row["blind_response_id"].startswith(row["blind_dialogue_id"] + "-"):
            raise ScoreError(
                f"{row['blind_response_id']} is not a child of "
                f"{row['blind_dialogue_id']}"
            )
        for fact_id in parse_fact_ids(row.get("fact_ids_stated", "")):
            if (
                not re.fullmatch(FACT_ID_SCHEMA_PATTERN, fact_id)
                or fact_id not in catalogue
            ):
                raise ScoreError(
                    f"{row['blind_response_id']} names unknown fact {fact_id}"
                )
    claim_ids = [row.get("claim_id", "") for row in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ScoreError(f"{CLAIMS_CSV} has duplicate claim_id")
    response_set = set(response_ids)
    for row in claims:
        if row.get("claim_text", "").strip() == "":
            raise ScoreError(f"{row.get('claim_id')} is missing claim_text")
        if row.get("support", "").strip() not in SUPPORT_VALUES:
            raise ScoreError(
                f"{row.get('claim_id')} support is "
                f"{row.get('support')!r}, not {sorted(SUPPORT_VALUES)}"
            )
        if row["blind_response_id"] not in response_set:
            raise ScoreError(
                f"{row.get('claim_id')} points at unknown {row['blind_response_id']}"
            )
        if not row.get("claim_id", "").startswith(row["blind_response_id"] + "-"):
            raise ScoreError(
                f"{row.get('claim_id')} is not prefixed by {row['blind_response_id']}"
            )


def load_mapping(path: Path, *, n_dialogues: int) -> list[MappingRow]:
    """Load ``private/mapping.csv`` and check it is a D001… bijection."""
    rows = _read_csv(path)
    expected = [f"D{n:03d}" for n in range(1, n_dialogues + 1)]
    found = [row["blind_dialogue_id"] for row in rows]
    if found != expected:
        raise ScoreError(f"{path} is not a bijection with D001…D{n_dialogues:03d}")
    real_ids = [row["dialogue_id"] for row in rows]
    if len(real_ids) != len(set(real_ids)):
        raise ScoreError(f"{path} has duplicate dialogue_id")
    mapped = []
    for row in rows:
        mapped.append(
            MappingRow(
                blind_id=row["blind_dialogue_id"],
                dialogue_id=row["dialogue_id"],
                scenario_id=row["scenario_id"],
                condition=row["condition"],
                repetition=int(row["repetition"]),
            )
        )
    return mapped


def _claim_support_label(*, n_supported: int, n_unsupported: int) -> str:
    """Map atomic claims onto the T-16 three-way response label, at dialogue grain."""
    if n_supported + n_unsupported == 0:
        return "none_checkable"
    if n_unsupported:
        return "some_unsupported"
    return "all_supported"


def derive_human_scores(
    *,
    mapping: Sequence[MappingRow],
    dialogues: Sequence[Mapping[str, str]],
    responses: Sequence[Mapping[str, str]],
    claims: Sequence[Mapping[str, str]],
    required: Mapping[str, Sequence[str]],
) -> list[HumanScores]:
    """Turn frozen sheets into dialogue-level scores. Uncertain stays uncertain."""
    task = {
        row["blind_dialogue_id"]: row["task_completed"].strip() for row in dialogues
    }
    predicted: dict[str, set[str]] = defaultdict(set)
    for row in responses:
        predicted[row["blind_dialogue_id"]].update(
            parse_fact_ids(row.get("fact_ids_stated", ""))
        )
    n_supported: dict[str, int] = defaultdict(int)
    n_unsupported: dict[str, int] = defaultdict(int)
    n_uncertain: dict[str, int] = defaultdict(int)
    unsupported_texts: dict[str, list[str]] = defaultdict(list)
    for row in claims:
        support = row["support"].strip()
        blind_id = row["blind_dialogue_id"]
        if support == "supported":
            n_supported[blind_id] += 1
        elif support == "unsupported":
            n_unsupported[blind_id] += 1
            unsupported_texts[blind_id].append(row["claim_text"])
        else:
            n_uncertain[blind_id] += 1
    scores: list[HumanScores] = []
    for item in mapping:
        gold = list(required[item.scenario_id])
        stated = predicted[item.blind_id]
        constructed: list[JudgeClaim] = [
            judge_claim(text=fact_id, fact_id=fact_id, supported_by_kb="yes")
            for fact_id in sorted(stated)
        ]
        constructed.extend(
            judge_claim(text=text, fact_id=None, supported_by_kb="no")
            for text in unsupported_texts[item.blind_id]
        )
        facts = fact_scores(required=gold, claims=constructed)
        checkable = n_supported[item.blind_id] + n_unsupported[item.blind_id]
        if checkable == 0:
            support = None
            rate = None
        else:
            support = n_supported[item.blind_id] / checkable
            rate = 1.0 - support
        scores.append(
            HumanScores(
                mapping=item,
                task_completed=task[item.blind_id],
                predicted_ids=frozenset(stated),
                n_true_positives=facts.n_true_positives,
                n_false_positives=facts.n_false_positives,
                n_false_negatives=facts.n_false_negatives,
                fact_precision=facts.fact_precision,
                fact_recall=facts.fact_recall,
                fact_f1=facts.fact_f1,
                claim_support=support,
                unsupported_claim_rate=rate,
                n_checkable_claims=checkable,
                n_uncertain_claims=n_uncertain[item.blind_id],
                claim_support_label=_claim_support_label(
                    n_supported=n_supported[item.blind_id],
                    n_unsupported=n_unsupported[item.blind_id],
                ),
            )
        )
    return scores


def human_metrics_rows(scores: Sequence[HumanScores]) -> list[dict[str, object]]:
    """Rows ``paired_test_table`` can aggregate, identity plus human metrics."""
    rows: list[dict[str, object]] = []
    for item in scores:
        task = item.task_completed
        rows.append(
            {
                "scenario_id": item.mapping.scenario_id,
                "agent": item.mapping.condition,
                "repetition": item.mapping.repetition,
                "blind_dialogue_id": item.mapping.blind_id,
                "dialogue_id": item.mapping.dialogue_id,
                "task_completed": "" if task == "uncertain" else task == "yes",
                "task_completed_label": task,
                "fact_precision": item.fact_precision,
                "fact_recall": item.fact_recall,
                "fact_f1": item.fact_f1,
                "claim_support": item.claim_support,
                "unsupported_claim_rate": item.unsupported_claim_rate,
                "n_checkable_claims": item.n_checkable_claims,
                "n_true_positives": item.n_true_positives,
                "n_false_positives": item.n_false_positives,
                "n_false_negatives": item.n_false_negatives,
                "n_uncertain_claims": item.n_uncertain_claims,
                "fact_ids_stated": ";".join(sorted(item.predicted_ids)),
            }
        )
    return rows


def percent_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    """Share of equal pairs; same definition as T-16."""
    if len(left) != len(right) or not left:
        raise ScoreError("agreement needs two equally long non-empty sequences")
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    """Unweighted Cohen's kappa; same definition as T-16."""
    labels = sorted(set(left) | set(right))
    if not labels:
        raise ScoreError("kappa needs at least one labelled row")
    index = {label: number for number, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for a, b in zip(left, right, strict=True):
        matrix[index[a]][index[b]] += 1
    n = len(left)
    observed = sum(matrix[i][i] for i in range(len(labels))) / n
    row = [sum(values) / n for values in matrix]
    col = [
        sum(matrix[i][j] for i in range(len(labels))) / n for j in range(len(labels))
    ]
    expected = sum(a * b for a, b in zip(row, col, strict=True))
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def fact_id_set_agreement(
    human_sets: Sequence[set[str]],
    judge_sets: Sequence[set[str]],
) -> dict[str, float | int]:
    """Exact-set rate and micro P/R/F1, human as gold, same as T-16."""
    if len(human_sets) != len(judge_sets) or not human_sets:
        raise ScoreError("fact-id agreement needs paired non-empty sets")
    exact = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for human, judge in zip(human_sets, judge_sets, strict=True):
        if human == judge:
            exact += 1
        true_positive += len(human & judge)
        false_positive += len(judge - human)
        false_negative += len(human - judge)
    precision_den = true_positive + false_positive
    recall_den = true_positive + false_negative
    precision = true_positive / precision_den if precision_den else 0.0
    recall = true_positive / recall_den if recall_den else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    n = len(human_sets)
    return {
        "n": n,
        "exact_match_rate": exact / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _bool_to_yes_no(value: object) -> str:
    """Map a metrics.csv bool cell to yes/no."""
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return "yes"
    if text in {"false", "0"}:
        return "no"
    raise ScoreError(f"judge task_completed is not bool-like: {value!r}")


def judge_support_label_from_row(row: Mapping[str, str]) -> str:
    """Three-way claim_support label from the frozen metrics row, not claim text."""
    n_checkable = int(str(row["n_checkable_claims"]))
    if n_checkable == 0:
        return "none_checkable"
    support = parse_cell(row["claim_support"])
    if support is None:
        return "none_checkable"
    if abs(support - 1.0) < 1e-9:
        return "all_supported"
    return "some_unsupported"


def predicted_ids_from_claims(claims: Sequence[JudgeClaim]) -> set[str]:
    """Judge predicted fact-id set, the predicted side of ``fact_scores``."""
    return {
        claim.fact_id
        for claim in claims
        if claim.supported_by_kb == "yes" and claim.fact_id is not None
    }


def load_transcripts(
    run_dir: Path,
    keys: Sequence[DialogueKey],
    scenarios: Mapping[str, Scenario],
) -> dict[DialogueKey, str]:
    """Load canary-redacted transcripts for the census keys."""
    transcripts: dict[DialogueKey, str] = {}
    for scenario_id, agent, repetition in keys:
        path = (
            run_dir / "dialogues" / f"{scenario_id}__{agent}__rep{repetition:02d}.jsonl"
        )
        if not path.is_file():
            raise ScoreError(f"{path} is missing")
        log = DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        scenario = scenarios[scenario_id]
        transcripts[(scenario_id, agent, repetition)] = redact_canary(
            render_transcript(log.transcript), scenario.canary
        )
    return transcripts


def compare_instrument(
    scores: Sequence[HumanScores],
    judge_rows: Mapping[DialogueKey, Mapping[str, str]],
    judge_claims: Mapping[DialogueKey, Sequence[JudgeClaim]],
) -> dict[str, object]:
    """Human vs judge at dialogue grain. Claims are not aligned by text."""
    task_human: list[str] = []
    task_judge: list[str] = []
    n_uncertain = 0
    human_sets: list[set[str]] = []
    judge_sets: list[set[str]] = []
    support_human: list[str] = []
    support_judge: list[str] = []
    for item in scores:
        judge = judge_rows.get(item.mapping.key)
        if judge is None:
            raise ScoreError(
                f"{item.mapping.dialogue_id} is not in the judge metrics CSV"
            )
        if item.task_completed == "uncertain":
            n_uncertain += 1
        else:
            task_human.append(item.task_completed)
            task_judge.append(_bool_to_yes_no(judge["task_completed"]))
        claims = judge_claims.get(item.mapping.key)
        if claims is not None:
            human_sets.append(set(item.predicted_ids))
            judge_sets.append(predicted_ids_from_claims(claims))
        support_human.append(item.claim_support_label)
        support_judge.append(judge_support_label_from_row(judge))
    facts = (
        fact_id_set_agreement(human_sets, judge_sets)
        if human_sets
        else {
            "n": 0,
            "exact_match_rate": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
    )
    if task_human:
        task_report: dict[str, object] = {
            "n": len(task_human),
            "n_uncertain_excluded": n_uncertain,
            "percent_agreement": percent_agreement(task_human, task_judge),
            "cohen_kappa": cohen_kappa(task_human, task_judge),
        }
    else:
        task_report = {
            "n": 0,
            "n_uncertain_excluded": n_uncertain,
            "percent_agreement": None,
            "cohen_kappa": None,
        }
    return {
        "grain": "dialogue",
        "n_dialogues": len(scores),
        "task_completed": task_report,
        "fact_ids_stated": {
            **facts,
            "n_omitted_ambiguous_judge_calls": len(scores) - int(facts["n"]),
        },
        "claim_support_label": {
            "n": len(support_human),
            "percent_agreement": percent_agreement(support_human, support_judge),
            "cohen_kappa": cohen_kappa(support_human, support_judge),
            "alignment": (
                "dialogue-level three-way label from n_checkable_claims and "
                "claim_support; no claim_id or claim_text matching"
            ),
        },
    }


def _test_index(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Index tests.csv by population, stratum, metric."""
    return {(row["population"], row["stratum"], row["metric"]): row for row in rows}


def compare_conclusions(
    human_tests: Sequence[Mapping[str, object]],
    judge_tests: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Overall primary-family direction and Holm significance, judge vs human."""
    judge = _test_index(judge_tests)
    rows: list[dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        human_row = next(
            (
                row
                for row in human_tests
                if row["population"] == HUMAN_PRIMARY
                and row["stratum"] == "overall"
                and row["metric"] == metric
            ),
            None,
        )
        judge_row = judge.get(("semantic_primary", "overall", metric))
        if human_row is None or judge_row is None:
            raise ScoreError(f"missing overall {metric} tests row")
        human_p = human_row["wilcoxon_p_holm"]
        judge_p = judge_row["wilcoxon_p_holm"]
        human_sig = human_p is not None and float(human_p) < SIGNIFICANT
        judge_sig = judge_p != "" and float(judge_p) < SIGNIFICANT
        rows.append(
            {
                "metric": metric,
                "judge_direction": judge_row["direction"],
                "human_direction": human_row["direction"],
                "judge_mean_diff": judge_row["mean_diff"],
                "human_mean_diff": human_row["mean_diff"],
                "judge_wilcoxon_p_holm": judge_p,
                "human_wilcoxon_p_holm": human_p,
                "same_direction": judge_row["direction"] == human_row["direction"],
                "same_significance": human_sig == judge_sig,
            }
        )
    return rows


def freeze_manifest(packet_dir: Path, frozen_dir: Path) -> None:
    """Copy the packet manifest next to the frozen CSVs; do not touch the CSVs."""
    source = packet_dir / MANIFEST_JSON
    dest = frozen_dir / MANIFEST_JSON
    if not source.is_file():
        raise ScoreError(f"{source} is missing")
    if dest.is_file() and dest.read_bytes() == source.read_bytes():
        return
    shutil.copyfile(source, dest)


def score_packet(
    *,
    packet_dir: Path,
    metrics_path: Path,
    judge_tests_path: Path,
    run_dir: Path,
    sidecar_dir: Path,
    scenarios_dir: Path,
    out_dir: Path,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> None:
    """Validate, unblind, derive, compare, and write human_primary outputs."""
    frozen_dir = packet_dir / "frozen"
    packet_manifest = json.loads(
        (packet_dir / MANIFEST_JSON).read_text(encoding="utf-8")
    )
    n_dialogues = int(packet_manifest["n_dialogues"])
    n_responses = int(packet_manifest["n_assistant_responses"])
    kb = load_kb(DEFAULT_KB_DIR)
    before = {
        name: file_sha256(frozen_dir / name)
        for name in (DIALOGUE_CSV, RESPONSE_CSV, CLAIMS_CSV)
    }
    validate_frozen_sheets(
        frozen_dir, kb=kb, n_dialogues=n_dialogues, n_responses=n_responses
    )
    freeze_manifest(packet_dir, frozen_dir)
    mapping = load_mapping(
        packet_dir / "private" / MAPPING_CSV, n_dialogues=n_dialogues
    )
    fsm = load_fsm(DEFAULT_FSM_DIR)
    scenarios = {
        scenario.id: scenario
        for scenario in load_scenarios(scenarios_dir, kb=kb, fsm=fsm)
    }
    required = {
        scenario_id: scenario.required_facts
        for scenario_id, scenario in scenarios.items()
    }
    human = derive_human_scores(
        mapping=mapping,
        dialogues=_read_csv(frozen_dir / DIALOGUE_CSV),
        responses=_read_csv(frozen_dir / RESPONSE_CSV),
        claims=_read_csv(frozen_dir / CLAIMS_CSV),
        required=required,
    )
    metrics_rows = human_metrics_rows(human)
    descriptive = [
        asdict(row)
        for row in descriptive_table(
            metrics_rows,
            population=HUMAN_PRIMARY,
            metrics=HUMAN_METRICS,
            n_resamples=n_resamples,
            seed=seed,
        )
    ]
    tests = paired_test_table(
        metrics_rows,
        population=HUMAN_PRIMARY,
        metrics=HUMAN_METRICS,
        n_resamples=n_resamples,
        seed=seed,
    )
    paired_rows: list[dict[str, object]] = []
    for metric in HUMAN_METRICS:
        pairs = pair_scenarios(aggregate_to_scenarios(metrics_rows, metric=metric))
        paired_rows.extend(asdict(pair) for pair in pairs)
    judge_metrics = {
        (row["scenario_id"], row["agent"], int(row["repetition"])): row
        for row in _read_csv(metrics_path)
    }
    keys = [item.mapping.key for item in human]
    transcripts = load_transcripts(run_dir, keys, scenarios)
    run_dirs = [run_dir]
    if sidecar_dir.is_dir() and (sidecar_dir / "llm_calls.jsonl").is_file():
        run_dirs.append(sidecar_dir)
    try:
        judge_claims = reconstruct_judge_claims(
            run_dirs=run_dirs,
            transcripts=transcripts,
            required=required,
            frozen_rows=judge_metrics,
            ambiguous="skip",
        )
    except DecompositionError as failure:
        raise ScoreError(str(failure)) from failure
    agreement = compare_instrument(human, judge_metrics, judge_claims)
    ambiguous_fact_ids = [key for key in keys if key not in judge_claims]
    test_dicts = [asdict(row) for row in tests]
    conclusions = compare_conclusions(test_dicts, _read_csv(judge_tests_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = list(metrics_rows[0].keys())
    _write_csv(out_dir / "metrics.csv", metric_fields, metrics_rows)
    _write_csv(
        out_dir / "paired.csv",
        ("scenario_id", "category", "metric", "baseline", "fsm", "diff"),
        paired_rows,
    )
    _write_csv(out_dir / "descriptive.csv", list(descriptive[0].keys()), descriptive)
    _write_csv(out_dir / "tests.csv", list(test_dicts[0].keys()), test_dicts)
    _write_csv(
        out_dir / "conclusions.csv",
        (
            "metric",
            "judge_direction",
            "human_direction",
            "judge_mean_diff",
            "human_mean_diff",
            "judge_wilcoxon_p_holm",
            "human_wilcoxon_p_holm",
            "same_direction",
            "same_significance",
        ),
        conclusions,
    )
    atomic_write(
        out_dir / "agreement.json",
        json.dumps(agreement, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write(out_dir / "environment.txt", environment_text())
    write_dialogue_keys(out_dir / "ambiguous_fact_ids.csv", ambiguous_fact_ids)
    after = {
        name: file_sha256(frozen_dir / name)
        for name in (DIALOGUE_CSV, RESPONSE_CSV, CLAIMS_CSV)
    }
    if after != before:
        raise ScoreError("frozen CSVs changed while scoring; aborting")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: freeze-check, unblind, derive, compare, confirmatory rerun."""
    parser = argparse.ArgumentParser(
        description=(
            "Score frozen human annotations against the judge and rerun the "
            "confirmatory paired tests. Does not edit frozen/"
        )
    )
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--judge-tests", type=Path, default=DEFAULT_JUDGE_TESTS)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        score_packet(
            packet_dir=args.packet,
            metrics_path=args.metrics,
            judge_tests_path=args.judge_tests,
            run_dir=args.run,
            sidecar_dir=args.sidecar,
            scenarios_dir=args.scenarios,
            out_dir=args.out,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
    except ScoreError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    print(f"sim: wrote {args.out}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
