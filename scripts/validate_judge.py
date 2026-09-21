"""Compute T-16 agreement and labeler reports from a pilot run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path

import httpx

from sim.config import ModelsConfig, load_models_config
from sim.evaluators import flow_scores
from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.io import atomic_write
from sim.judge import redact_canary
from sim.kb import DEFAULT_KB_DIR, load_kb
from sim.llm import LLM_CALLS_LOG, LlmCallRecord
from sim.metrics import ACCURACY_SCORE, JudgeClaim, JudgeFacts
from sim.prompts import load_prompt
from sim.schemas import (
    DialogueLog,
    Manifest,
    Scenario,
    load_scenarios,
    render_transcript,
)

SAMPLE_JSON = "sample.json"
RESPONSE_CSV = "response_annotations.csv"
DIALOGUE_CSV = "dialogue_annotations.csv"
METRICS_CSV = "metrics.csv"
METRICS_TURN_CSV = "metrics_turn.csv"
ACCURACY_ORDER = ("incorrect", "partial", "correct")
JUDGE_CALLERS = frozenset({"judge_facts", "judge_global"})
DIALOGUE_PREDICTION_FIELDS = (
    "dialogue_annotation_id",
    "human_task_completed",
    "judge_task_completed",
    "human_accuracy",
    "judge_accuracy",
)
RESPONSE_PREDICTION_FIELDS = (
    "annotation_id",
    "human_claim_support",
    "judge_claim_support",
    "human_fact_ids",
    "judge_fact_ids",
)
COMPARISON_CSV_FIELDS = ("metric", "original", "mlx", "delta")
CHANGED_CASE_FIELDS = ("unit", "field", "original", "mlx")
LABELER_COMPARISON_FIELDS = (
    "metric",
    "original",
    "sidecar",
    "delta",
)
LABELER_CHANGED_TURN_FIELDS = (
    "scenario_id",
    "repetition",
    "turn",
    "gold",
    "original",
    "sidecar",
    "kind",
)


class ValidationError(RuntimeError):
    """The validation input is malformed; the message says what to fix."""


@dataclass(frozen=True)
class CategoricalAgreement:
    """Agreement of one categorical field."""

    n: int
    percent: float
    kappa: float


@dataclass(frozen=True)
class OrdinalAgreement:
    """Agreement of one ordinal field."""

    n: int
    percent: float
    weighted_kappa: float


@dataclass(frozen=True)
class FactIdSetAgreement:
    """Exact-set and micro scores for fact-id sets."""

    n: int
    exact_match_rate: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class UnmatchedClaim:
    """One judge claim that could not be tied to exactly one turn."""

    dialogue_id: str
    claim: JudgeClaim
    matching_turns: tuple[int, ...]


@dataclass(frozen=True)
class DialogueAgreement:
    """Dialogue-level agreement results."""

    n: int
    task_completed: CategoricalAgreement
    accuracy: OrdinalAgreement


@dataclass(frozen=True)
class ResponseAgreement:
    """Response-level agreement results."""

    n: int
    claim_support: CategoricalAgreement
    fact_ids: FactIdSetAgreement
    unmatched_claims: list[UnmatchedClaim]


@dataclass(frozen=True)
class DialoguePrediction:
    """One dialogue's human labels and the judge labels from one run."""

    dialogue_annotation_id: str
    human_task_completed: str
    judge_task_completed: str
    human_accuracy: str
    judge_accuracy: str


@dataclass(frozen=True)
class ResponsePrediction:
    """One response's human labels and the judge labels from one run."""

    annotation_id: str
    human_claim_support: str
    judge_claim_support: str
    human_fact_ids: str
    judge_fact_ids: str


@dataclass(frozen=True)
class AgreementReport:
    """The full agreement output of T-16."""

    dialogue: DialogueAgreement
    response: ResponseAgreement
    dialogue_predictions: tuple[DialoguePrediction, ...]
    response_predictions: tuple[ResponsePrediction, ...]


@dataclass(frozen=True)
class ComparisonReport:
    """Human-vs-judge agreement on two runs, plus the judge-vs-judge diff."""

    original: AgreementReport
    mlx: AgreementReport
    comparison: tuple[dict[str, str], ...]
    changed_cases: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class LabelerReport:
    """Stage-labeler results measured on the final pilot eval."""

    n_fsm_turns: int
    n_correct: int
    turn_accuracy: float
    n_fsm_dialogues: int
    n_gold_valid_paths: int
    n_labelled_valid_paths: int
    n_path_agreement: int
    path_agreement: float
    n_gold_true_labelled_false: int


@dataclass(frozen=True)
class LabelerCompareReport:
    """Turn-level 4B vs sidecar labels against FSM gold."""

    original: LabelerReport
    sidecar: LabelerReport
    n_corrected: int
    n_remaining: int
    n_new: int
    changed_turns: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class _SampledResponse:
    annotation_id: str
    dialogue_id: str
    turn: int


@dataclass(frozen=True)
class _SampledDialogue:
    dialogue_annotation_id: str
    dialogue_id: str
    scenario_id: str
    agent: str
    repetition: int


COMPARISON_METRICS: tuple[tuple[str, Callable[[AgreementReport], float]], ...] = (
    ("task_completed_agreement", lambda r: r.dialogue.task_completed.percent),
    ("task_completed_kappa", lambda r: r.dialogue.task_completed.kappa),
    ("accuracy_agreement", lambda r: r.dialogue.accuracy.percent),
    ("accuracy_qwk", lambda r: r.dialogue.accuracy.weighted_kappa),
    ("claim_support_agreement", lambda r: r.response.claim_support.percent),
    ("claim_support_kappa", lambda r: r.response.claim_support.kappa),
    ("fact_ids_stated_exact_set", lambda r: r.response.fact_ids.exact_match_rate),
    ("precision", lambda r: r.response.fact_ids.precision),
    ("recall", lambda r: r.response.fact_ids.recall),
    ("f1", lambda r: r.response.fact_ids.f1),
)


def percent_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    """Return the share of equal pairs in two equally long sequences."""
    if len(left) != len(right):
        raise ValidationError(
            f"cannot compare {len(left)} and {len(right)} values: agreement pairs "
            "rows one by one"
        )
    if not left:
        raise ValidationError("agreement needs at least one row")
    matched = sum(a == b for a, b in zip(left, right, strict=True))
    return matched / len(left)


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    """Return unweighted Cohen's kappa for two categorical labellers."""
    labels = sorted(set(left) | set(right))
    if not labels:
        raise ValidationError("kappa needs at least one labelled row")
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


def quadratic_weighted_kappa(
    left: Sequence[str],
    right: Sequence[str],
    *,
    order: Sequence[str] = ACCURACY_ORDER,
) -> float:
    """Return quadratic weighted Cohen's kappa for ordinal labels."""
    if len(left) != len(right):
        raise ValidationError(
            f"cannot compare {len(left)} and {len(right)} values: agreement pairs "
            "rows one by one"
        )
    if not left:
        raise ValidationError("weighted kappa needs at least one labelled row")
    k = len(order)
    if k < 2:
        raise ValidationError("weighted kappa needs at least two ordered labels")
    index = {label: number for number, label in enumerate(order)}
    missing = (set(left) | set(right)) - set(order)
    if missing:
        raise ValidationError(
            f"labels {sorted(missing)} are outside the ordinal scale {list(order)}"
        )
    matrix = [[0 for _ in range(k)] for _ in range(k)]
    for a, b in zip(left, right, strict=True):
        matrix[index[a]][index[b]] += 1
    n = len(left)
    row = [sum(values) for values in matrix]
    col = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    observed = 0.0
    expected = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) ** 2) / ((k - 1) ** 2)
            observed += weight * (matrix[i][j] / n)
            expected += weight * ((row[i] / n) * (col[j] / n))
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - (observed / expected)


def fact_id_set_agreement(
    human_sets: Sequence[set[str]],
    judge_sets: Sequence[set[str]],
) -> FactIdSetAgreement:
    """Return exact-match rate and micro P/R/F1 for fact-id sets."""
    if len(human_sets) != len(judge_sets):
        raise ValidationError(
            f"cannot compare {len(human_sets)} and {len(judge_sets)} responses"
        )
    if not human_sets:
        raise ValidationError("fact-id agreement needs at least one response")
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
    return FactIdSetAgreement(
        n=n,
        exact_match_rate=exact / n,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def attribute_claims(
    claims: Sequence[JudgeClaim],
    turns: Sequence[tuple[int, str]],
) -> tuple[dict[int, list[JudgeClaim]], list[UnmatchedClaim]]:
    """Attribute each claim by substring to one turn, or mark it unmatched."""
    attributed = {turn: [] for turn, _ in turns}
    unmatched: list[UnmatchedClaim] = []
    lowered = [(turn, text.lower()) for turn, text in turns]
    for claim in claims:
        text = claim.text.strip().lower()
        if not text:
            unmatched.append(
                UnmatchedClaim(
                    dialogue_id="",
                    claim=claim,
                    matching_turns=(),
                )
            )
            continue
        hits = tuple(turn for turn, reply in lowered if text in reply)
        if len(hits) == 1:
            attributed[hits[0]].append(claim)
            continue
        unmatched.append(
            UnmatchedClaim(
                dialogue_id="",
                claim=claim,
                matching_turns=hits,
            )
        )
    return attributed, unmatched


def claim_support_label(claims: Sequence[JudgeClaim]) -> str:
    """Map claim-level judge labels to one response-level claim_support value."""
    if not claims:
        return "none_checkable"
    if any(claim.supported_by_kb == "no" for claim in claims):
        return "some_unsupported"
    if all(claim.supported_by_kb == "unverifiable" for claim in claims):
        return "none_checkable"
    return "all_supported"


def agree(sample_dir: Path, run_dir: Path) -> AgreementReport:
    """Compute T-16 agreement on one frozen sample against one eval directory."""
    sampled = _load_sample(sample_dir / SAMPLE_JSON)
    response_rows = _read_csv(sample_dir / RESPONSE_CSV)
    dialogue_rows = _read_csv(sample_dir / DIALOGUE_CSV)
    metrics = _metrics_by_dialogue(run_dir / METRICS_CSV)
    logs = _dialogues_by_id(run_dir)
    scenarios = _scenarios_by_id(run_dir)
    facts_of = _judge_facts_by_dialogue(run_dir, logs, scenarios)

    dialogue_predictions: list[DialoguePrediction] = []
    dialogue_meta = {item.dialogue_annotation_id: item for item in sampled["dialogues"]}
    for row in dialogue_rows:
        dialogue_id = row["dialogue_annotation_id"]
        if dialogue_id not in dialogue_meta:
            raise ValidationError(
                f"{dialogue_id} is in {DIALOGUE_CSV} but not in {SAMPLE_JSON}"
            )
        meta = dialogue_meta[dialogue_id]
        key = (meta.scenario_id, meta.agent, meta.repetition)
        metrics_row = metrics.get(key)
        if metrics_row is None:
            raise ValidationError(
                f"{key} is in {SAMPLE_JSON} but not in {METRICS_CSV} of {run_dir}"
            )
        dialogue_predictions.append(
            DialoguePrediction(
                dialogue_annotation_id=dialogue_id,
                human_task_completed=row["task_completed"].strip().lower(),
                judge_task_completed=_bool_to_yes_no(metrics_row["task_completed"]),
                human_accuracy=row["accuracy"].strip().lower(),
                judge_accuracy=_accuracy_from_score(metrics_row["accuracy_score"]),
            )
        )
    human_task = [item.human_task_completed for item in dialogue_predictions]
    judge_task = [item.judge_task_completed for item in dialogue_predictions]
    human_accuracy = [item.human_accuracy for item in dialogue_predictions]
    judge_accuracy = [item.judge_accuracy for item in dialogue_predictions]

    dialogue_report = DialogueAgreement(
        n=len(human_task),
        task_completed=CategoricalAgreement(
            n=len(human_task),
            percent=percent_agreement(human_task, judge_task),
            kappa=cohen_kappa(human_task, judge_task),
        ),
        accuracy=OrdinalAgreement(
            n=len(human_accuracy),
            percent=percent_agreement(human_accuracy, judge_accuracy),
            weighted_kappa=quadratic_weighted_kappa(human_accuracy, judge_accuracy),
        ),
    )

    sampled_responses = {item.annotation_id: item for item in sampled["responses"]}
    sampled_dialogues = {item.dialogue_id for item in sampled["responses"]}
    cached_attribution: dict[str, dict[int, list[JudgeClaim]]] = {}
    unmatched: list[UnmatchedClaim] = []
    response_predictions: list[ResponsePrediction] = []
    for row in response_rows:
        annotation_id = row["annotation_id"]
        sampled_response = sampled_responses.get(annotation_id)
        if sampled_response is None:
            raise ValidationError(
                f"{annotation_id} is in {RESPONSE_CSV} but not in {SAMPLE_JSON}"
            )
        dialogue_id = sampled_response.dialogue_id
        if dialogue_id not in cached_attribution:
            log = logs[dialogue_id]
            turns = [(record.turn, record.agent_reply) for record in log.records]
            attributed, misses = attribute_claims(facts_of[dialogue_id].claims, turns)
            cached_attribution[dialogue_id] = attributed
            for miss in misses:
                unmatched.append(
                    UnmatchedClaim(
                        dialogue_id=dialogue_id,
                        claim=miss.claim,
                        matching_turns=miss.matching_turns,
                    )
                )
        attributed = cached_attribution[dialogue_id]
        on_turn = attributed.get(sampled_response.turn, [])
        judge_facts = {
            claim.fact_id
            for claim in on_turn
            if claim.supported_by_kb == "yes" and claim.fact_id is not None
        }
        response_predictions.append(
            ResponsePrediction(
                annotation_id=annotation_id,
                human_claim_support=row["claim_support"].strip().lower(),
                judge_claim_support=claim_support_label(on_turn),
                human_fact_ids=row["fact_ids_stated"].strip(),
                judge_fact_ids=_format_fact_ids(judge_facts),
            )
        )
    for dialogue_id in sampled_dialogues:
        if dialogue_id not in facts_of:
            raise ValidationError(
                f"{dialogue_id} has sampled responses but no judge_facts call"
            )

    human_support = [item.human_claim_support for item in response_predictions]
    judge_support = [item.judge_claim_support for item in response_predictions]
    human_fact_sets = [_fact_ids(item.human_fact_ids) for item in response_predictions]
    judge_fact_sets = [_fact_ids(item.judge_fact_ids) for item in response_predictions]
    response_report = ResponseAgreement(
        n=len(human_support),
        claim_support=CategoricalAgreement(
            n=len(human_support),
            percent=percent_agreement(human_support, judge_support),
            kappa=cohen_kappa(human_support, judge_support),
        ),
        fact_ids=fact_id_set_agreement(human_fact_sets, judge_fact_sets),
        unmatched_claims=unmatched,
    )
    return AgreementReport(
        dialogue=dialogue_report,
        response=response_report,
        dialogue_predictions=tuple(dialogue_predictions),
        response_predictions=tuple(response_predictions),
    )


def compare_runs(
    sample_dir: Path, original_run: Path, mlx_run: Path
) -> ComparisonReport:
    """Compare two evals using the same ``agree()`` path on each run."""
    original = agree(sample_dir, original_run)
    mlx = agree(sample_dir, mlx_run)
    return ComparisonReport(
        original=original,
        mlx=mlx,
        comparison=tuple(_comparison_rows(original, mlx)),
        changed_cases=tuple(_changed_cases(original, mlx)),
    )


def write_agreement_artifacts(report: AgreementReport, out_dir: Path) -> None:
    """Write agreement.json and the two prediction CSVs under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        out_dir / "agreement.json",
        json.dumps(_agreement_payload(report), indent=2, ensure_ascii=False) + "\n",
    )
    _write_csv(
        out_dir / "predictions_dialogue.csv",
        DIALOGUE_PREDICTION_FIELDS,
        [asdict(row) for row in report.dialogue_predictions],
    )
    _write_csv(
        out_dir / "predictions_response.csv",
        RESPONSE_PREDICTION_FIELDS,
        [asdict(row) for row in report.response_predictions],
    )


def write_comparison_artifacts(
    report: ComparisonReport,
    out_dir: Path,
    *,
    environment: Mapping[str, object] | None = None,
) -> None:
    """Write MLX agreement artifacts plus comparison, changed cases, environment."""
    write_agreement_artifacts(report.mlx, out_dir)
    _write_csv(out_dir / "comparison.csv", COMPARISON_CSV_FIELDS, report.comparison)
    _write_csv(out_dir / "changed_cases.csv", CHANGED_CASE_FIELDS, report.changed_cases)
    if environment is not None:
        atomic_write(
            out_dir / "environment.json",
            json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        )


def judge_environment(
    run_dir: Path,
    *,
    config: ModelsConfig | None = None,
    ollama_version: str | None = None,
) -> dict[str, object]:
    """Return observed log fields and configured judge parameters.

    Observed model identity comes from ``llm_calls.jsonl``. Configured sampling
    comes from ``configs/models.yaml``. YAML is not treated as proof of what
    the server ran.
    """
    loaded = config or load_models_config()
    spec = loaded.spec("judge")
    tag, digest = _observed_judge_identity(run_dir)
    version = ollama_version if ollama_version is not None else fetch_ollama_version()
    return {
        "model_tag_observed": tag,
        "model_digest_observed": digest,
        "ollama_version_observed": version,
        "temperature_configured": spec.temperature,
        "seed_configured": spec.seed,
        "num_ctx_configured": loaded.num_ctx,
        "judge_facts_prompt_version": f"v{load_prompt('judge_facts').version}",
        "judge_global_prompt_version": f"v{load_prompt('judge_global').version}",
        "judge_shared_prompt_version": f"v{load_prompt('judge_shared').version}",
    }


def fetch_ollama_version(*, timeout_s: float = 2.0) -> str | None:
    """Return the Ollama server version, or None if it cannot be read."""
    try:
        response = httpx.get("http://localhost:11434/api/version", timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    if isinstance(version, str) and version:
        return version
    return None


def labeler_report(run_dir: Path) -> LabelerReport:
    """Compute turn and path agreement of the stage labeler on FSM dialogues."""
    rows = [
        row for row in _read_csv(run_dir / METRICS_TURN_CSV) if row["agent"] == "fsm"
    ]
    if not rows:
        raise ValidationError(f"{METRICS_TURN_CSV} has no FSM rows in {run_dir}")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario_id"], int(row["repetition"]))].append(row)
    fsm = load_fsm(DEFAULT_FSM_DIR)
    scenarios = _scenarios_by_id(run_dir)
    n_correct = 0
    n_gold_valid = 0
    n_labelled_valid = 0
    n_agree = 0
    n_downward = 0
    for (scenario_id, _), group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item["turn"]))
        labelled = [row["labelled_stage"] for row in ordered]
        gold = [row["true_state_after"] for row in ordered]
        if any(not item for item in gold):
            raise ValidationError(
                f"{scenario_id} has empty true_state_after in {METRICS_TURN_CSV}"
            )
        n_correct += sum(a == b for a, b in zip(labelled, gold, strict=True))
        scenario = scenarios[scenario_id]
        gold_flow = flow_scores(gold, scenario.expected_final_state, fsm)
        labelled_flow = flow_scores(labelled, scenario.expected_final_state, fsm)
        n_gold_valid += int(gold_flow.valid_flow_path)
        n_labelled_valid += int(labelled_flow.valid_flow_path)
        n_agree += int(gold_flow.valid_flow_path == labelled_flow.valid_flow_path)
        if gold_flow.valid_flow_path and not labelled_flow.valid_flow_path:
            n_downward += 1
    n_turns = len(rows)
    n_dialogues = len(grouped)
    return LabelerReport(
        n_fsm_turns=n_turns,
        n_correct=n_correct,
        turn_accuracy=n_correct / n_turns,
        n_fsm_dialogues=n_dialogues,
        n_gold_valid_paths=n_gold_valid,
        n_labelled_valid_paths=n_labelled_valid,
        n_path_agreement=n_agree,
        path_agreement=n_agree / n_dialogues,
        n_gold_true_labelled_false=n_downward,
    )


def compare_labeler(original_run: Path, sidecar_run: Path) -> LabelerCompareReport:
    """Join FSM turn labels from two runs and count corrected/remaining/new."""
    original_rows = _fsm_turn_index(original_run)
    sidecar_rows = _fsm_turn_index(sidecar_run)
    if original_rows.keys() != sidecar_rows.keys():
        raise ValidationError(
            "original and sidecar FSM turn keys differ; compare the same dialogues"
        )
    n_corrected = 0
    n_remaining = 0
    n_new = 0
    changed: list[dict[str, str]] = []
    for key in sorted(original_rows, key=lambda item: (item[0], item[1], item[2])):
        original = original_rows[key]
        sidecar = sidecar_rows[key]
        gold = original["true_state_after"]
        if gold != sidecar["true_state_after"]:
            raise ValidationError(
                f"{key[0]} rep {key[1]} turn {key[2]} has different gold labels"
            )
        orig_label = original["labelled_stage"]
        side_label = sidecar["labelled_stage"]
        orig_ok = orig_label == gold
        side_ok = side_label == gold
        kind: str | None = None
        if not orig_ok and side_ok:
            kind = "corrected"
            n_corrected += 1
        elif not orig_ok and not side_ok:
            kind = "remaining"
            n_remaining += 1
        elif orig_ok and not side_ok:
            kind = "new"
            n_new += 1
        if kind is not None:
            changed.append(
                {
                    "scenario_id": key[0],
                    "repetition": str(key[1]),
                    "turn": str(key[2]),
                    "gold": gold,
                    "original": orig_label,
                    "sidecar": side_label,
                    "kind": kind,
                }
            )
    return LabelerCompareReport(
        original=labeler_report(original_run),
        sidecar=labeler_report(sidecar_run),
        n_corrected=n_corrected,
        n_remaining=n_remaining,
        n_new=n_new,
        changed_turns=tuple(changed),
    )


def write_labeler_comparison_artifacts(
    report: LabelerCompareReport, out_dir: Path
) -> None:
    """Write labeler comparison CSV files under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    original = report.original
    sidecar = report.sidecar
    rows = (
        {
            "metric": "turn_correct",
            "original": f"{original.n_correct}/{original.n_fsm_turns}",
            "sidecar": f"{sidecar.n_correct}/{sidecar.n_fsm_turns}",
            "delta": str(sidecar.n_correct - original.n_correct),
        },
        {
            "metric": "turn_accuracy",
            "original": _csv_float(original.turn_accuracy),
            "sidecar": _csv_float(sidecar.turn_accuracy),
            "delta": _csv_float(sidecar.turn_accuracy - original.turn_accuracy),
        },
        {
            "metric": "path_agreement",
            "original": (f"{original.n_path_agreement}/{original.n_fsm_dialogues}"),
            "sidecar": f"{sidecar.n_path_agreement}/{sidecar.n_fsm_dialogues}",
            "delta": str(sidecar.n_path_agreement - original.n_path_agreement),
        },
        {
            "metric": "downward_bias",
            "original": str(original.n_gold_true_labelled_false),
            "sidecar": str(sidecar.n_gold_true_labelled_false),
            "delta": str(
                sidecar.n_gold_true_labelled_false - original.n_gold_true_labelled_false
            ),
        },
        {
            "metric": "corrected",
            "original": "",
            "sidecar": str(report.n_corrected),
            "delta": "",
        },
        {
            "metric": "remaining",
            "original": "",
            "sidecar": str(report.n_remaining),
            "delta": "",
        },
        {
            "metric": "new",
            "original": "",
            "sidecar": str(report.n_new),
            "delta": "",
        },
    )
    _write_csv(out_dir / "labeler_comparison.csv", LABELER_COMPARISON_FIELDS, rows)
    _write_csv(
        out_dir / "labeler_changed_turns.csv",
        LABELER_CHANGED_TURN_FIELDS,
        report.changed_turns,
    )


def _fsm_turn_index(run_dir: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    """Index FSM ``metrics_turn.csv`` rows by scenario, repetition and turn."""
    rows = [
        row for row in _read_csv(run_dir / METRICS_TURN_CSV) if row["agent"] == "fsm"
    ]
    if not rows:
        raise ValidationError(f"{METRICS_TURN_CSV} has no FSM rows in {run_dir}")
    indexed: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = (row["scenario_id"], int(row["repetition"]), int(row["turn"]))
        if key in indexed:
            raise ValidationError(
                f"duplicate FSM turn {key[0]} rep {key[1]} turn {key[2]} in {run_dir}"
            )
        indexed[key] = row
    return indexed


def _load_sample(path: Path) -> dict[str, list[object]]:
    """Load sampled responses and dialogues from the frozen sample file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    responses = [
        _SampledResponse(
            annotation_id=item["annotation_id"],
            dialogue_id=item["dialogue_id"],
            turn=int(item["turn"]),
        )
        for item in payload["responses"]
    ]
    dialogues = [
        _SampledDialogue(
            dialogue_annotation_id=item["dialogue_annotation_id"],
            dialogue_id=item["dialogue_id"],
            scenario_id=item["scenario_id"],
            agent=item["agent"],
            repetition=int(item["repetition"]),
        )
        for item in payload["dialogues"]
    ]
    return {"responses": responses, "dialogues": dialogues}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV into dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _dialogues_by_id(run_dir: Path) -> dict[str, DialogueLog]:
    """Load every dialogue log of a run directory."""
    dialogues_dir = run_dir / "dialogues"
    if not dialogues_dir.exists():
        raise ValidationError(f"{dialogues_dir} is missing")
    logs = [
        DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(dialogues_dir.glob("*.jsonl"))
    ]
    return {log.dialogue_id: log for log in logs}


def _manifest(run_dir: Path) -> Manifest:
    """Load the run manifest."""
    path = run_dir / "manifest.json"
    if not path.exists():
        raise ValidationError(f"{path} is missing")
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def _scenarios_by_id(run_dir: Path) -> dict[str, Scenario]:
    """Load the scenarios that the run manifest records."""
    manifest = _manifest(run_dir)
    kb = load_kb(DEFAULT_KB_DIR)
    fsm = load_fsm(DEFAULT_FSM_DIR)
    scenarios = load_scenarios(Path(manifest.scenarios_dir), kb=kb, fsm=fsm)
    return {scenario.id: scenario for scenario in scenarios}


def _metrics_by_dialogue(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    """Index metrics rows by (scenario_id, agent, repetition)."""
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in _read_csv(path):
        key = (row["scenario_id"], row["agent"], int(row["repetition"]))
        if key in indexed:
            raise ValidationError(f"duplicate metrics row for {key}")
        indexed[key] = row
    return indexed


def _judge_facts_by_dialogue(
    run_dir: Path,
    dialogues: dict[str, DialogueLog],
    scenarios: dict[str, Scenario],
) -> dict[str, JudgeFacts]:
    """Map each dialogue to its judge_facts output via transcript containment."""
    expected = {
        dialogue_id: redact_canary(
            render_transcript(log.transcript), scenarios[log.scenario_id].canary
        )
        for dialogue_id, log in dialogues.items()
        if log.status == "ok"
    }
    mapped: dict[str, JudgeFacts] = {}
    for record in _load_calls(run_dir / LLM_CALLS_LOG):
        if record.caller != "judge_facts":
            continue
        message = "\n".join(item["content"] for item in record.messages)
        matches = [
            dialogue_id
            for dialogue_id, transcript in expected.items()
            if transcript in message
        ]
        if len(matches) != 1:
            continue
        mapped[matches[0]] = JudgeFacts.model_validate_json(record.text)
    missing = sorted(set(expected) - set(mapped))
    if missing:
        raise ValidationError(
            "missing judge_facts rows for dialogues: "
            + ", ".join(missing[:3])
            + ("..." if len(missing) > 3 else "")
        )
    return mapped


def _load_calls(path: Path) -> list[LlmCallRecord]:
    """Load every line of llm_calls.jsonl."""
    if not path.exists():
        raise ValidationError(f"{path} is missing")
    return [
        LlmCallRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _accuracy_from_score(value: str) -> str:
    """Map the numeric accuracy score back to its ordinal label."""
    score = float(value)
    for label, mapped in ACCURACY_SCORE.items():
        if mapped == score:
            return label
    raise ValidationError(f"unknown accuracy_score {value!r}")


def _bool_to_yes_no(value: str) -> str:
    """Map a bool-like CSV value to yes/no."""
    lowered = value.strip().lower()
    if lowered in {"true", "1"}:
        return "yes"
    if lowered in {"false", "0"}:
        return "no"
    raise ValidationError(f"task_completed must be bool-like, got {value!r}")


def _fact_ids(raw: str) -> set[str]:
    """Parse `F01;F02` style text into a set of ids."""
    parts = [item.strip() for item in raw.split(";")]
    return {item for item in parts if item}


def _format_fact_ids(ids: set[str]) -> str:
    """Render a fact-id set as a stable semicolon-separated string."""
    return ";".join(sorted(ids))


def _agreement_payload(report: AgreementReport) -> dict[str, object]:
    """JSON-ready agreement metrics, with unmatched claims as a count."""
    return {
        "dialogue": {
            "n": report.dialogue.n,
            "task_completed": asdict(report.dialogue.task_completed),
            "accuracy": asdict(report.dialogue.accuracy),
        },
        "response": {
            "n": report.response.n,
            "claim_support": asdict(report.response.claim_support),
            "fact_ids": asdict(report.response.fact_ids),
            "unmatched_claims": len(report.response.unmatched_claims),
        },
    }


def _comparison_rows(
    original: AgreementReport, mlx: AgreementReport
) -> list[dict[str, str]]:
    """One row per published agreement metric, with mlx minus original."""
    rows: list[dict[str, str]] = []
    for name, getter in COMPARISON_METRICS:
        left = getter(original)
        right = getter(mlx)
        rows.append(
            {
                "metric": name,
                "original": _csv_float(left),
                "mlx": _csv_float(right),
                "delta": _csv_float(right - left),
            }
        )
    return rows


def _changed_cases(
    original: AgreementReport, mlx: AgreementReport
) -> list[dict[str, str]]:
    """Units whose judge label differs between the two ``agree()`` reports."""
    rows: list[dict[str, str]] = []
    orig_dialogues = {
        item.dialogue_annotation_id: item for item in original.dialogue_predictions
    }
    mlx_dialogues = {
        item.dialogue_annotation_id: item for item in mlx.dialogue_predictions
    }
    if orig_dialogues.keys() != mlx_dialogues.keys():
        raise ValidationError("original and mlx dialogue prediction ids differ")
    for item in original.dialogue_predictions:
        other = mlx_dialogues[item.dialogue_annotation_id]
        if item.judge_task_completed != other.judge_task_completed:
            rows.append(
                {
                    "unit": item.dialogue_annotation_id,
                    "field": "task_completed",
                    "original": item.judge_task_completed,
                    "mlx": other.judge_task_completed,
                }
            )
        if item.judge_accuracy != other.judge_accuracy:
            rows.append(
                {
                    "unit": item.dialogue_annotation_id,
                    "field": "accuracy",
                    "original": item.judge_accuracy,
                    "mlx": other.judge_accuracy,
                }
            )
    orig_responses = {
        item.annotation_id: item for item in original.response_predictions
    }
    mlx_responses = {item.annotation_id: item for item in mlx.response_predictions}
    if orig_responses.keys() != mlx_responses.keys():
        raise ValidationError("original and mlx response prediction ids differ")
    for item in original.response_predictions:
        other = mlx_responses[item.annotation_id]
        if item.judge_claim_support != other.judge_claim_support:
            rows.append(
                {
                    "unit": item.annotation_id,
                    "field": "claim_support",
                    "original": item.judge_claim_support,
                    "mlx": other.judge_claim_support,
                }
            )
        if _fact_ids(item.judge_fact_ids) != _fact_ids(other.judge_fact_ids):
            rows.append(
                {
                    "unit": item.annotation_id,
                    "field": "fact_ids_stated",
                    "original": item.judge_fact_ids,
                    "mlx": other.judge_fact_ids,
                }
            )
    return rows


def _csv_float(value: float) -> str:
    """Render a float for CSV without rounding away the computed value."""
    return f"{value:.12g}"


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write ``rows`` to ``path`` with a fixed header, including an empty body."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue())


def _observed_judge_identity(run_dir: Path) -> tuple[str | None, str | None]:
    """Return the unique judge model tag and digest from ``llm_calls.jsonl``."""
    path = run_dir / LLM_CALLS_LOG
    if not path.exists():
        return None, None
    pairs: list[tuple[str, str | None]] = []
    for record in _load_calls(path):
        if record.caller not in JUDGE_CALLERS:
            continue
        pair = (record.model, record.digest)
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        return None, None
    if len(pairs) > 1:
        raise ValidationError(
            f"judge calls in {path} use more than one model or digest"
        )
    return pairs[0]


def render_agreement(report: AgreementReport) -> str:
    """Render agreement metrics as plain text."""
    lines = [
        "dialogue-level",
        f"  n={report.dialogue.n}",
        (
            "  task_completed "
            f"agreement={report.dialogue.task_completed.percent:.3f} "
            f"kappa={report.dialogue.task_completed.kappa:.3f}"
        ),
        (
            "  accuracy "
            f"agreement={report.dialogue.accuracy.percent:.3f} "
            f"quadratic_kappa={report.dialogue.accuracy.weighted_kappa:.3f}"
        ),
        "response-level",
        f"  n={report.response.n}",
        (
            "  claim_support "
            f"agreement={report.response.claim_support.percent:.3f} "
            f"kappa={report.response.claim_support.kappa:.3f}"
        ),
        (
            "  fact_ids "
            f"exact_set={report.response.fact_ids.exact_match_rate:.3f} "
            f"precision={report.response.fact_ids.precision:.3f} "
            f"recall={report.response.fact_ids.recall:.3f} "
            f"f1={report.response.fact_ids.f1:.3f}"
        ),
        f"  unmatched_claims={len(report.response.unmatched_claims)}",
    ]
    return "\n".join(lines)


def render_labeler(report: LabelerReport) -> str:
    """Render stage-labeler metrics as plain text."""
    turn_ratio = f"{report.n_correct}/{report.n_fsm_turns}"
    path_ratio = f"{report.n_path_agreement}/{report.n_fsm_dialogues}"
    return "\n".join(
        (
            f"fsm_turns={report.n_fsm_turns}",
            f"turn_accuracy={report.turn_accuracy:.3f} ({turn_ratio})",
            f"fsm_dialogues={report.n_fsm_dialogues}",
            f"valid_flow_path_agreement={report.path_agreement:.3f} ({path_ratio})",
            f"gold_valid_paths={report.n_gold_valid_paths}",
            f"labelled_valid_paths={report.n_labelled_valid_paths}",
            f"gold_true_labelled_false={report.n_gold_true_labelled_false}",
        )
    )


def render_labeler_compare(report: LabelerCompareReport) -> str:
    """Render the 4B vs sidecar labeler comparison as plain text."""
    return "\n".join(
        (
            "original",
            render_labeler(report.original),
            "sidecar",
            render_labeler(report.sidecar),
            f"corrected={report.n_corrected}",
            f"remaining={report.n_remaining}",
            f"new={report.n_new}",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run agreement or labeler validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    agree_parser = sub.add_parser("agree", help="compute T-16 human-vs-judge agreement")
    agree_parser.add_argument(
        "--sample",
        type=Path,
        default=Path("results/judge_validation/pilot_v1"),
        help="directory with sample.json and the two annotation sheets",
    )
    agree_parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/exp_pilot"),
        help="run directory with metrics.csv, llm_calls.jsonl and dialogue logs",
    )

    agree_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional directory for agreement.json and prediction CSVs",
    )

    compare_parser = sub.add_parser(
        "compare",
        help="compare two agree() runs (original vs gemma4:12b-mlx sidecar)",
    )
    compare_parser.add_argument(
        "--original",
        type=Path,
        default=Path("runs/pilot_v2"),
        help="run directory of the published validation",
    )
    compare_parser.add_argument(
        "--mlx",
        type=Path,
        default=Path("runs/pilot_v2_mlx"),
        help="sidecar run directory scored with gemma4:12b-mlx",
    )
    compare_parser.add_argument(
        "--sample",
        type=Path,
        default=Path("results/judge_validation/pilot_v2"),
        help="directory with sample.json and the two annotation sheets",
    )
    compare_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/judge_validation/pilot_v2_mlx"),
        help="directory for comparison artifacts",
    )

    labeler_parser = sub.add_parser(
        "labeler", help="measure stage labeler on metrics_turn.csv"
    )
    labeler_parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/exp_pilot"),
        help="run directory with metrics_turn.csv and the manifest",
    )

    labeler_compare_parser = sub.add_parser(
        "labeler-compare",
        help="compare two stage-labeler metrics_turn.csv files against FSM gold",
    )
    labeler_compare_parser.add_argument(
        "--original",
        type=Path,
        default=Path("runs/pilot_v2"),
        help="run directory of the published 4B labels",
    )
    labeler_compare_parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("runs/pilot_v2_labeler_9b"),
        help="sidecar run directory relabelled with the comparison model",
    )
    labeler_compare_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/judge_validation/pilot_v2_qwen9b"),
        help="directory for labeler comparison CSVs",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "agree":
            report = agree(args.sample, args.run)
            print(render_agreement(report))
            if args.out is not None:
                write_agreement_artifacts(report, args.out)
            return 0
        if args.command == "compare":
            report = compare_runs(args.sample, args.original, args.mlx)
            print("original")
            print(render_agreement(report.original))
            print("gemma4:12b-mlx")
            print(render_agreement(report.mlx))
            write_comparison_artifacts(
                report,
                args.out,
                environment=judge_environment(args.mlx),
            )
            return 0
        if args.command == "labeler":
            print(render_labeler(labeler_report(args.run)))
            return 0
        if args.command == "labeler-compare":
            report = compare_labeler(args.original, args.sidecar)
            print(render_labeler_compare(report))
            write_labeler_comparison_artifacts(report, args.out)
            return 0
        raise ValidationError(f"unknown command {args.command}")
    except ValidationError as error:
        print(f"sim: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
