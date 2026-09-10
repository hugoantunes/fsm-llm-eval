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

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

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


class JudgeClaim(BaseModel):
    """One atomic assertion the assistant made, optionally tied to a KB ID."""

    model_config = ConfigDict(extra="forbid")

    text: str
    fact_id: FactId | None
    supported_by_kb: ClaimSupport

    @model_validator(mode="after")
    def _fact_id_matches_support(self) -> "JudgeClaim":
        """A supported claim names its KB ID; a non-supported claim has none."""
        if self.supported_by_kb == "yes":
            if self.fact_id is not None:
                return self
            raise ValueError(
                "supported_by_kb is yes but fact_id is null: a claim the knowledge "
                "base supports must name that fact, or fact_precision cannot tell a "
                "required ID from an unmatched leftover"
            )
        if self.fact_id is None:
            return self
        raise ValueError(
            f"supported_by_kb is {self.supported_by_kb!r} but fact_id is "
            f"{self.fact_id}: only a yes claim is associated with a knowledge-base ID"
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
    """Replace every ``$ref`` into ``defs``, keeping sibling keys of the ref."""
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
        return {key: _inline(value, defs) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline(value, defs) for value in node]
    return node
