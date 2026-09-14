"""Compute T-16 agreement and labeler reports from a pilot run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sim.evaluators import flow_scores
from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.judge import redact_canary
from sim.kb import DEFAULT_KB_DIR, load_kb
from sim.llm import LLM_CALLS_LOG, LlmCallRecord
from sim.metrics import ACCURACY_SCORE, JudgeClaim, JudgeFacts
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
class AgreementReport:
    """The full agreement output of T-16."""

    dialogue: DialogueAgreement
    response: ResponseAgreement


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

    human_task: list[str] = []
    judge_task: list[str] = []
    human_accuracy: list[str] = []
    judge_accuracy: list[str] = []
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
        human_task.append(row["task_completed"].strip().lower())
        judge_task.append(_bool_to_yes_no(metrics_row["task_completed"]))
        human_accuracy.append(row["accuracy"].strip().lower())
        judge_accuracy.append(_accuracy_from_score(metrics_row["accuracy_score"]))

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
    human_support: list[str] = []
    judge_support: list[str] = []
    human_fact_sets: list[set[str]] = []
    judge_fact_sets: list[set[str]] = []
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
        human_support.append(row["claim_support"].strip().lower())
        judge_support.append(claim_support_label(on_turn))
        human_fact_sets.append(_fact_ids(row["fact_ids_stated"]))
        judge_fact_sets.append(
            {
                claim.fact_id
                for claim in on_turn
                if claim.supported_by_kb == "yes" and claim.fact_id is not None
            }
        )
    for dialogue_id in sampled_dialogues:
        if dialogue_id not in facts_of:
            raise ValidationError(
                f"{dialogue_id} has sampled responses but no judge_facts call"
            )

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
    return AgreementReport(dialogue=dialogue_report, response=response_report)


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

    labeler_parser = sub.add_parser(
        "labeler", help="measure stage labeler on metrics_turn.csv"
    )
    labeler_parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/exp_pilot"),
        help="run directory with metrics_turn.csv and the manifest",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "agree":
            print(render_agreement(agree(args.sample, args.run)))
            return 0
        if args.command == "labeler":
            print(render_labeler(labeler_report(args.run)))
            return 0
        raise ValidationError(f"unknown command {args.command}")
    except ValidationError as error:
        print(f"sim: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
