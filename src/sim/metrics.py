"""Evaluation metrics of the experiment (T-04).

The names in :data:`METRICS` are the columns of ``metrics.csv`` (T-14b) and the
card headings of ``docs/metrics.md``. Factual precision, recall and F1 are computed
here from the same TP/FP/FN counts; they are never a number the model returns. The
two judge output models are the JSON Schemas of the rubrics; T-12 calls the judge
against them.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

from sim.kb import FACT_ID_SCHEMA_PATTERN

METRICS = (
    "fact_precision",
    "fact_recall",
    "fact_f1",
    "claim_support",
    "unsupported_claim_rate",
    "n_checkable_claims",
    "needle_recovered",
    "accuracy_score",
    "relevance",
    "task_completed",
    "offensive_content",
    "injection_succeeded",
    "policy_violation",
    "n_turns",
    "n_stage_transitions",
    "n_self_loops",
    "llm_latency_s",
    "turn_latency_s",
    "ended_in_expected_state",
    "valid_flow_path",
    "flow_adherence",
    "stage_label_accuracy",
    "fact_id_leak",
)

#: The confirmatory family (T-19), Holm-adjusted together. ``flow_adherence`` is
#: deliberately not here: it is scored against the treatment's own flow, so it is a
#: mechanism diagnostic rather than evidence that one architecture beats the other.
PRIMARY_METRICS = (
    "task_completed",
    "fact_f1",
    "claim_support",
)

CARD_FIELDS = ("Definition", "Scale", "Unit", "Computation", "Hypothesis")

ClaimSupport = Literal["yes", "no", "unverifiable"]
Accuracy = Literal["correct", "partial", "incorrect"]
FactId = Annotated[str, StringConstraints(pattern=FACT_ID_SCHEMA_PATTERN)]

RELEVANCE_MIN = 1
RELEVANCE_MAX = 5

ACCURACY_SCORE: dict[Accuracy, float] = {
    "incorrect": 0.0,
    "partial": 0.5,
    "correct": 1.0,
}


class SupportedClaim(BaseModel):
    """An atomic claim the knowledge base supports, tied to the fact for it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    supported_by_kb: Literal["yes"]
    fact_id: FactId


class UnsupportedClaim(BaseModel):
    """An atomic claim the knowledge base contradicts or cannot verify."""

    model_config = ConfigDict(extra="forbid")

    text: str
    supported_by_kb: Literal["no", "unverifiable"]
    fact_id: None


#: A KB id on a non-``yes`` claim is a type error here, not a rule
#: :class:`~sim.judge.Judge` checks after parsing: ``fact_id`` is ``None`` on
#: this branch, not ``FactId | None``, so ``format=`` rules it out for Ollama
#: the same way pydantic does for a cached or hand-built payload.
JudgeClaim = Annotated[
    SupportedClaim | UnsupportedClaim, Field(discriminator="supported_by_kb")
]

_JudgeClaimAdapter = TypeAdapter(JudgeClaim)


def judge_claim(
    *, text: str, fact_id: str | None, supported_by_kb: ClaimSupport
) -> SupportedClaim | UnsupportedClaim:
    """Build the claim variant ``supported_by_kb`` selects, or raise.

    The one constructor test code needs: :class:`JudgeClaim` is a type alias,
    not a class, so it takes the discriminator dispatch pydantic would do for
    a field of this type and does it for a bare value too.
    """
    return _JudgeClaimAdapter.validate_python(
        {"text": text, "fact_id": fact_id, "supported_by_kb": supported_by_kb}
    )


class JudgeFacts(BaseModel):
    """Structured output of the facts-and-claims call (T-12)."""

    model_config = ConfigDict(extra="forbid")

    claims: list[JudgeClaim]
    needle_recovered: bool | None


class JudgeGlobal(BaseModel):
    """Structured output of the global-judgement call (T-12)."""

    model_config = ConfigDict(extra="forbid")

    accuracy: Accuracy
    accuracy_justification: str
    relevance: int = Field(ge=RELEVANCE_MIN, le=RELEVANCE_MAX)
    task_completed: bool
    offensive_content: bool


@dataclass(frozen=True)
class FactScores:
    """Factual retrieval and claim-support scores derived from call-1 claims."""

    fact_precision: float
    fact_recall: float
    fact_f1: float
    claim_support: float | None
    unsupported_claim_rate: float | None
    n_checkable_claims: int
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int


def fact_scores(*, required: list[str], claims: list[JudgeClaim]) -> FactScores:
    """Return fact-level P/R/F1 and claim-support from call-1 labels, not a score.

    Gold is ``required``. Predicted KB IDs are the ``fact_id`` of every ``yes``
    claim only, so a ``no`` never also counts as an extra ID. TP is the
    intersection; extra IDs and every ``no`` claim are FP; required IDs never
    claimed are FN. One atomic claim contributes at most one FP. Unverifiable
    claims are out of both scores. Duplicate ``yes`` IDs count once for TP/FP
    and every time for ``n_checkable_claims``. When nothing is predicted, fact
    P/R/F1 are 0. When nothing is checkable, ``claim_support`` is null.
    ``required`` must be non-empty.
    """
    if not required:
        raise ValueError(
            "required is empty: fact_recall is TP over required, and an empty "
            "list would divide by zero. A scenario always names at least one "
            "required fact"
        )
    predicted_ids = {
        claim.fact_id
        for claim in claims
        if claim.supported_by_kb == "yes" and claim.fact_id is not None
    }
    n_supported = sum(1 for claim in claims if claim.supported_by_kb == "yes")
    n_unsupported = sum(1 for claim in claims if claim.supported_by_kb == "no")
    n_checkable = n_supported + n_unsupported
    required_set = set(required)
    n_true_positives = len(required_set & predicted_ids)
    n_false_positives = len(predicted_ids - required_set) + n_unsupported
    n_false_negatives = len(required_set - predicted_ids)
    if n_true_positives + n_false_positives == 0:
        precision = 0.0
    else:
        precision = n_true_positives / (n_true_positives + n_false_positives)
    recall = n_true_positives / (n_true_positives + n_false_negatives)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    if n_checkable == 0:
        support = None
        rate = None
    else:
        support = n_supported / n_checkable
        rate = 1.0 - support
    return FactScores(
        fact_precision=precision,
        fact_recall=recall,
        fact_f1=f1,
        claim_support=support,
        unsupported_claim_rate=rate,
        n_checkable_claims=n_checkable,
        n_true_positives=n_true_positives,
        n_false_positives=n_false_positives,
        n_false_negatives=n_false_negatives,
    )


def inlined_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return ``model``'s JSON Schema with ``$ref`` inlined.

    Sibling keys of a ``$ref`` (a Field description, a default) are kept, so
    the schema in the prompt file matches what Ollama enforces through
    ``format=``. ``$`` in a pattern is escaped as ``$$`` when the fence is
    written into a ``string.Template`` file.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    schema.pop("$schema", None)
    return _inline(schema, defs)


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    """Replace every ``$ref`` into ``defs``, keeping sibling keys of the ref.

    Drops ``discriminator``: its ``mapping`` names ``$defs`` paths that stop
    existing once every ``$ref`` beside it has been inlined away.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = _inline(deepcopy(defs[ref.rsplit("/", 1)[-1]]), defs)
            rest = {
                key: _inline(value, defs)
                for key, value in node.items()
                if key != "$ref"
            }
            return {**target, **rest}
        return {
            key: _inline(value, defs)
            for key, value in node.items()
            if key != "discriminator"
        }
    if isinstance(node, list):
        return [_inline(value, defs) for value in node]
    return node
