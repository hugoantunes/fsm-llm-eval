"""Post-hoc decomposition of the frozen ``fact_f1`` effect (T-22, exploratory).

``fact_f1`` measures correspondence to one scenario's expected fact set, and it
falls for two unrelated reasons: a required ID never stated, and a predicted item
outside the expected set. The second is itself two things — a knowledge-base
fact the scenario did not ask for, and a checkable claim the knowledge base does
not support — which the score collapses into one FP count. This module takes them
apart from the frozen artifacts alone: the judge output already in
``llm_calls.jsonl`` is re-read, never re-requested, and every reconstruction has
to reproduce the frozen ``metrics.csv`` row before it is used. Nothing here
changes how ``fact_f1`` is computed; :func:`sim.metrics.fact_scores` stays the
one definition.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from statistics import mean

from pydantic import ValidationError

from sim.analysis import CATEGORIES, canonical_diff, category_of, parse_cell
from sim.io import atomic_write
from sim.llm import LLM_CALLS_LOG, LlmCallRecord
from sim.metrics import JudgeClaim, JudgeFacts, fact_scores

#: One dialogue: the identity columns of ``metrics.csv``.
DialogueKey = tuple[str, str, int]

JUDGE_FACTS_CALLER = "judge_facts"

#: The counts written one row each, in reading order: the expected set, the two
#: sides of recall, the FP total and the two kinds it is made of, then how much
#: the agent said that could be checked at all.
#:
#: Unverifiable claims are deliberately absent. They are out of both fact scores,
#: so the frozen row does not pin their number down, and the two judge backends of
#: this run disagree about it on at least one dialogue while agreeing on every
#: checkable count. Reporting them would be reporting a number the frozen
#: artifacts cannot verify.
COUNT_FIELDS = (
    "n_required",
    "tp",
    "fn",
    "fp",
    "extra_supported_fact",
    "unsupported_claim",
    "n_checkable_claims",
)

DECOMPOSITION_CSV = "fact_f1_decomposition.csv"
DECOMPOSITION_JSON = "fact_f1_decomposition.json"

_VERIFIED_COLUMNS = ("fact_precision", "fact_recall", "claim_support")


class DecompositionError(RuntimeError):
    """A reconstruction cannot be trusted; the message says which dialogue."""


@dataclass(frozen=True)
class FactCounts:
    """The TP/FP/FN counts of one dialogue, with the FP total split by kind.

    ``fp == extra_supported_fact + unsupported_claim`` by construction, the same
    two disjoint addends :func:`sim.metrics.fact_scores` sums.
    """

    n_required: int
    tp: int
    fn: int
    fp: int
    extra_supported_fact: int
    unsupported_claim: int
    n_checkable_claims: int


@dataclass(frozen=True)
class DecompositionRow:
    """One count, one stratum: the paired scenario-level comparison of it.

    The tallies are named by direction rather than by winner: for ``tp`` a lower
    FSM count is worse and for every other count it is better, so "win" would
    mean two different things in one column.
    """

    stratum: str
    count: str
    n: int
    n_dropped_unpaired: int
    baseline_mean: float | None
    fsm_mean: float | None
    mean_diff: float | None
    fsm_lower: int
    equal: int
    fsm_higher: int


def split_false_positives(
    *, required: Sequence[str], claims: Sequence[JudgeClaim]
) -> FactCounts:
    """Return the counts of ``fact_scores``, with FP split into its two kinds.

    ``extra_supported_fact`` is a ``yes`` claim whose KB id the scenario did not
    require: true in the knowledge base, off the expected set.
    ``unsupported_claim`` is a ``no`` claim: checkable and unsupported. Only the
    second is evidence about grounding.
    """
    scores = fact_scores(required=list(required), claims=list(claims))
    unsupported = sum(1 for claim in claims if claim.supported_by_kb == "no")
    return FactCounts(
        n_required=len(set(required)),
        tp=scores.n_true_positives,
        fn=scores.n_false_negatives,
        fp=scores.n_false_positives,
        extra_supported_fact=scores.n_false_positives - unsupported,
        unsupported_claim=unsupported,
        n_checkable_claims=scores.n_checkable_claims,
    )


def reconstruct_judge_claims(
    *,
    run_dirs: Sequence[Path],
    transcripts: Mapping[DialogueKey, str],
    required: Mapping[str, Sequence[str]],
    frozen_rows: Mapping[DialogueKey, Mapping[str, object]],
    ambiguous: str = "error",
) -> dict[DialogueKey, list[JudgeClaim]]:
    """Return the judge claim list that uniquely reproduces each frozen row.

    Searches every ``llm_calls.jsonl`` in ``run_dirs``. Join is transcript
    containment; the frozen metrics row is the arbiter. Refuses to guess.
    """
    if not run_dirs:
        raise DecompositionError(
            "reconstruct_judge_claims needs at least one run directory"
        )
    candidates: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        candidates.extend(_judge_facts_texts(run_dir))
    claims_by_key: dict[DialogueKey, list[JudgeClaim]] = {}
    for key, transcript in transcripts.items():
        row = frozen_rows.get(key)
        if row is None:
            continue
        expected = required[key[0]]
        found = [
            claims
            for claims in _claims_for(candidates, transcript)
            if _reproduces(row, required=expected, claims=claims)
        ]
        unique = {_claims_signature(claims) for claims in found}
        if not unique:
            searched = ", ".join(str(path / LLM_CALLS_LOG) for path in run_dirs)
            raise DecompositionError(
                f"{_name(key)}: no judge_facts call in {searched} reproduces "
                "the frozen metrics row. The reconstruction only re-reads "
                "judge output, so it cannot score this dialogue"
            )
        if len(unique) > 1:
            if ambiguous == "skip":
                continue
            raise DecompositionError(
                f"{_name(key)}: {len(unique)} judge_facts calls reproduce the "
                "frozen metrics row but disagree on predicted fact IDs. "
                "The reconstruction will not guess among them"
            )
        claims_by_key[key] = found[0]
    return claims_by_key


def reconstruct_fact_counts(
    *,
    run_dir: Path,
    transcripts: Mapping[DialogueKey, str],
    required: Mapping[str, Sequence[str]],
    frozen_rows: Mapping[DialogueKey, Mapping[str, object]],
) -> dict[DialogueKey, FactCounts]:
    """Recover the counts behind every frozen row, from the judge log alone."""
    claims_by_key = reconstruct_judge_claims(
        run_dirs=(run_dir,),
        transcripts=transcripts,
        required=required,
        frozen_rows=frozen_rows,
    )
    return {
        key: split_false_positives(required=required[key[0]], claims=claims)
        for key, claims in claims_by_key.items()
    }


def _claims_signature(
    claims: Sequence[JudgeClaim],
) -> tuple[frozenset[str], int, int, int]:
    """Predicted IDs and checkable counts; wording differences do not matter."""
    predicted = frozenset(
        claim.fact_id
        for claim in claims
        if claim.supported_by_kb == "yes" and claim.fact_id is not None
    )
    n_yes = sum(1 for claim in claims if claim.supported_by_kb == "yes")
    n_no = sum(1 for claim in claims if claim.supported_by_kb == "no")
    n_unverifiable = sum(
        1 for claim in claims if claim.supported_by_kb == "unverifiable"
    )
    return (predicted, n_yes, n_no, n_unverifiable)


def decompose(counts: Mapping[DialogueKey, FactCounts]) -> list[DecompositionRow]:
    """Aggregate ``counts`` to scenarios and pair them, one row per count.

    Repetitions become a scenario mean before pairing, the T-19 unit of analysis.
    A scenario missing either agent is dropped and reported in
    ``n_dropped_unpaired``.
    """
    by_scenario: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (scenario_id, agent, _), value in counts.items():
        for field, number in asdict(value).items():
            by_scenario[(scenario_id, agent)][field].append(number)
    scenarios = sorted({scenario_id for scenario_id, _ in by_scenario})
    rows: list[DecompositionRow] = []
    for stratum in ("overall", *CATEGORIES):
        in_stratum = [
            scenario_id
            for scenario_id in scenarios
            if stratum == "overall" or category_of(scenario_id) == stratum
        ]
        paired = [
            scenario_id
            for scenario_id in in_stratum
            if all((scenario_id, agent) in by_scenario for agent in ("baseline", "fsm"))
        ]
        for field in COUNT_FIELDS:
            rows.append(
                _row(
                    stratum=stratum,
                    count=field,
                    paired=paired,
                    n_dropped_unpaired=len(in_stratum) - len(paired),
                    by_scenario=by_scenario,
                )
            )
    return rows


def write_decomposition(path: Path, rows: Sequence[DecompositionRow]) -> None:
    """Write ``rows`` as CSV, atomically, in the order :func:`decompose` built."""
    fieldnames = tuple(DecompositionRow.__dataclass_fields__)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: "" if value is None else value
                for name, value in asdict(row).items()
            }
        )
    atomic_write(path, buffer.getvalue())


def write_provenance(path: Path, payload: Mapping[str, object]) -> None:
    """Write the reconstruction's provenance next to the CSV, atomically."""
    atomic_write(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def load_frozen_rows(path: Path) -> dict[DialogueKey, dict[str, str]]:
    """Load an exported ``metrics.csv`` keyed by the identity columns."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["scenario_id"], row["agent"], int(row["repetition"])): row
            for row in csv.DictReader(handle)
        }


def _row(
    *,
    stratum: str,
    count: str,
    paired: Sequence[str],
    n_dropped_unpaired: int,
    by_scenario: Mapping[tuple[str, str], Mapping[str, Sequence[int]]],
) -> DecompositionRow:
    """Build one paired row: the two means, their difference and the tallies."""
    if not paired:
        return DecompositionRow(
            stratum=stratum,
            count=count,
            n=0,
            n_dropped_unpaired=n_dropped_unpaired,
            baseline_mean=None,
            fsm_mean=None,
            mean_diff=None,
            fsm_lower=0,
            equal=0,
            fsm_higher=0,
        )
    scores = {
        agent: [
            float(mean(by_scenario[(scenario, agent)][count])) for scenario in paired
        ]
        for agent in ("baseline", "fsm")
    }
    diffs = [
        canonical_diff(fsm, baseline)
        for fsm, baseline in zip(scores["fsm"], scores["baseline"], strict=True)
    ]
    return DecompositionRow(
        stratum=stratum,
        count=count,
        n=len(paired),
        n_dropped_unpaired=n_dropped_unpaired,
        baseline_mean=float(mean(scores["baseline"])),
        fsm_mean=float(mean(scores["fsm"])),
        mean_diff=float(mean(diffs)),
        fsm_lower=sum(1 for diff in diffs if diff < 0),
        equal=sum(1 for diff in diffs if diff == 0),
        fsm_higher=sum(1 for diff in diffs if diff > 0),
    )


def _judge_facts_texts(run_dir: Path) -> list[tuple[str, str]]:
    """Return ``(prompt, answer)`` of every judge_facts call that answered."""
    path = run_dir / LLM_CALLS_LOG
    if not path.exists():
        raise DecompositionError(
            f"{path} is missing. The decomposition re-reads the judge output of "
            f"the run instead of calling the judge again"
        )
    calls: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = LlmCallRecord.model_validate_json(line)
            if record.caller != JUDGE_FACTS_CALLER or record.error or not record.text:
                continue
            calls.append((record.messages[0]["content"], record.text))
    return calls


def _claims_for(
    candidates: Sequence[tuple[str, str]], transcript: str
) -> list[list[JudgeClaim]]:
    """Return the claim lists of every judge call whose prompt holds ``transcript``."""
    found = []
    for prompt, text in candidates:
        if transcript not in prompt:
            continue
        try:
            found.append(list(JudgeFacts.model_validate_json(text).claims))
        except ValidationError:
            continue
    return found


def _reproduces(
    row: Mapping[str, object], *, required: Sequence[str], claims: Sequence[JudgeClaim]
) -> bool:
    """Tell whether ``claims`` rescore to the frozen row, cell by cell."""
    scores = fact_scores(required=list(required), claims=list(claims))
    if int(str(row["n_checkable_claims"])) != scores.n_checkable_claims:
        return False
    found = {
        "fact_precision": scores.fact_precision,
        "fact_recall": scores.fact_recall,
        "claim_support": scores.claim_support,
    }
    for column in _VERIFIED_COLUMNS:
        frozen = parse_cell(row[column])
        if frozen is None or found[column] is None:
            if frozen is not None or found[column] is not None:
                return False
            continue
        if abs(frozen - found[column]) > 1e-9:
            return False
    return True


def _name(key: DialogueKey) -> str:
    """Render a dialogue key the way the run names its JSONL."""
    scenario_id, agent, repetition = key
    return f"{scenario_id}__{agent}__rep{repetition:02d}"
