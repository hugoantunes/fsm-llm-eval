"""Category inversions and exemplar ranking for T-21.

The category-direction table copies T-19 ``tests.csv`` cells. Exemplar ranking
uses scenario-level scores from ``metrics.csv``. Dialogues are loaded only after
IDs and repetition are frozen.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from csv import DictWriter
from dataclasses import asdict, dataclass, replace
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import ValidationError

from sim.analysis import (
    INSTRUMENT_DROP_SCENARIOS,
    OVERALL,
    SEMANTIC_PRIMARY,
    aggregate_to_scenarios,
    category_of,
    load_metrics_csv,
    pair_scenarios,
)
from sim.io import atomic_write
from sim.metrics import PRIMARY_METRICS
from sim.schemas import (
    CATEGORIES,
    DialogueLog,
    dialogue_filename,
    transcript_from_records,
)

DIRECTION_METRICS = (*PRIMARY_METRICS, "flow_adherence")
STRATA = (OVERALL, *CATEGORIES)
FLOW_ADHERENCE = "flow_adherence"


class CasesError(RuntimeError):
    """T-21 cannot freeze a table, ranking, or transcript."""


BUCKET_HELPED = "helped"
BUCKET_RESTRICTED = "restricted"
BUCKET_TIE = "tie"
Bucket = Literal["helped", "restricted", "tie", ""]


@dataclass(frozen=True)
class DirectionCell:
    """One T-21 category-direction cell, copied from a T-19 test row."""

    metric: str
    stratum: str
    family: str
    n: int
    wins: int
    ties: int
    losses: int
    direction: str
    inverted: bool
    mean_diff: float
    rank_biserial: float | None


def inverted(stratum: str, direction: str, overall_direction: str) -> bool:
    """True when a category cell disagrees with the overall direction."""
    return stratum != OVERALL and direction != overall_direction


def cases_family(*, metric: str, stratum: str) -> str:
    """T-21 family label; independent of the T-19 ``family`` column."""
    if metric == FLOW_ADHERENCE:
        return "diagnostic"
    if metric in PRIMARY_METRICS:
        return "confirmatory" if stratum == OVERALL else "exploratory"
    raise ValueError(f"{metric!r} is not a T-21 table metric")


def category_direction_table(
    rows: Sequence[Mapping[str, object]],
) -> list[DirectionCell]:
    """Project ``semantic_primary`` T-19 cells into the inversion table."""
    selected = [
        row
        for row in rows
        if str(row["population"]) == SEMANTIC_PRIMARY
        and str(row["metric"]) in DIRECTION_METRICS
    ]
    overall_direction = {
        str(row["metric"]): str(row["direction"])
        for row in selected
        if str(row["stratum"]) == OVERALL
    }
    by_key = {(str(row["metric"]), str(row["stratum"])): row for row in selected}
    table: list[DirectionCell] = []
    for metric in DIRECTION_METRICS:
        overall = overall_direction.get(metric)
        if overall is None:
            continue
        for stratum in STRATA:
            row = by_key.get((metric, stratum))
            if row is None:
                continue
            direction = str(row["direction"])
            table.append(
                DirectionCell(
                    metric=metric,
                    stratum=stratum,
                    family=cases_family(metric=metric, stratum=stratum),
                    n=_as_int(row["n"]),
                    wins=_as_int(row["wins"]),
                    ties=_as_int(row["ties"]),
                    losses=_as_int(row["losses"]),
                    direction=direction,
                    inverted=inverted(stratum, direction, overall),
                    mean_diff=_as_float(row["mean_diff"]),
                    rank_biserial=_optional_float(row.get("rank_biserial")),
                )
            )
    return table


@dataclass(frozen=True)
class CaseCandidate:
    """One eligible scenario under the T-21 ranking contract."""

    scenario_id: str
    task_completed_diff: float | None
    fact_f1_diff: float | None
    claim_support_diff: float | None
    n_primary_available: int
    composite: float | None
    bucket: Bucket
    rank: int | None


def rank_case_candidates(
    rows: Sequence[Mapping[str, object]],
) -> list[CaseCandidate]:
    """Rank eligible scenarios into helped, restricted, and tie buckets."""
    agents: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        agents[str(row["scenario_id"])].add(str(row["agent"]))
    diffs: dict[str, dict[str, float]] = defaultdict(dict)
    for metric in PRIMARY_METRICS:
        for pair in pair_scenarios(aggregate_to_scenarios(rows, metric=metric)):
            diffs[pair.scenario_id][metric] = pair.diff
    unranked: list[CaseCandidate] = []
    for scenario_id, present in agents.items():
        if scenario_id in INSTRUMENT_DROP_SCENARIOS:
            continue
        if present != {"baseline", "fsm"}:
            continue
        metric_diffs = diffs[scenario_id]
        available = tuple(
            metric_diffs[name] for name in PRIMARY_METRICS if name in metric_diffs
        )
        task = metric_diffs.get("task_completed")
        fact_f1 = metric_diffs.get("fact_f1")
        claim_support = metric_diffs.get("claim_support")
        unranked.append(
            CaseCandidate(
                scenario_id=scenario_id,
                task_completed_diff=task,
                fact_f1_diff=fact_f1,
                claim_support_diff=claim_support,
                n_primary_available=len(available),
                composite=mean(available) if available else None,
                bucket=_bucket(task, fact_f1, claim_support),
                rank=None,
            )
        )
    return _assign_ranks(unranked)


def select_exemplars(
    candidates: Sequence[CaseCandidate],
) -> dict[str, CaseCandidate]:
    """Return the rank-1 candidate of each bucket."""
    exemplars: dict[str, CaseCandidate] = {}
    for candidate in candidates:
        if candidate.bucket and candidate.rank == 1:
            exemplars[candidate.bucket] = candidate
    return exemplars


def lowest_common_repetition(
    rows: Sequence[Mapping[str, object]], scenario_id: str
) -> int:
    """Lowest repetition scored for both baseline and FSM on ``scenario_id``."""
    reps: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if str(row["scenario_id"]) != scenario_id:
            continue
        reps[str(row["agent"])].add(_as_int(row["repetition"]))
    common = reps.get("baseline", set()) & reps.get("fsm", set())
    if not common:
        raise CasesError(
            f"{scenario_id} has no repetition scored for both baseline and FSM"
        )
    return min(common)


def load_frozen_log(
    run_dir: Path, scenario_id: str, agent: str, repetition: int
) -> DialogueLog:
    """Load one frozen dialogue; missing or invalid JSONL fails closed."""
    name = dialogue_filename(scenario_id, agent, repetition)
    path = run_dir / "dialogues" / name
    if not path.is_file():
        raise CasesError(f"missing frozen log {name}")
    try:
        return DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise CasesError(f"invalid frozen log {name}") from exc


def render_transcript(log: DialogueLog) -> str:
    """Render user and agent turns as ``speaker: text`` lines."""
    return "\n".join(
        f"{turn.speaker}: {turn.text}" for turn in transcript_from_records(log.records)
    )


def write_cases_csvs(tests_path: Path, metrics_path: Path, out_dir: Path) -> None:
    """Write category_directions.csv and case_candidates.csv under ``out_dir``."""
    cells = [
        asdict(cell) for cell in category_direction_table(load_metrics_csv(tests_path))
    ]
    candidates = [
        _candidate_csv_row(row)
        for row in rank_case_candidates(load_metrics_csv(metrics_path))
    ]
    _write_csv(out_dir / "category_directions.csv", cells)
    _write_csv(out_dir / "case_candidates.csv", candidates)


def _candidate_csv_row(row: CaseCandidate) -> dict[str, object]:
    data = asdict(row)
    if data["rank"] is None:
        data["rank"] = ""
    return data


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise CasesError(f"{path} would have no rows")
    fieldnames = list(rows[0].keys())
    buffer = StringIO()
    writer = DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: "" if row[key] is None else row[key] for key in fieldnames}
        )
    atomic_write(path, buffer.getvalue())


def _bucket(
    task: float | None, fact_f1: float | None, claim_support: float | None
) -> Bucket:
    if (
        task == 0.0
        and fact_f1 == 0.0
        and claim_support == 0.0
        and task is not None
        and fact_f1 is not None
        and claim_support is not None
    ):
        return BUCKET_TIE
    if task is not None and task > 0:
        return BUCKET_HELPED
    if task is not None and task < 0:
        return BUCKET_RESTRICTED
    return ""


def _assign_ranks(candidates: Sequence[CaseCandidate]) -> list[CaseCandidate]:
    ranked: list[CaseCandidate] = []
    remaining = [row for row in candidates if not row.bucket]
    helped = _sorted_bucket(
        [row for row in candidates if row.bucket == BUCKET_HELPED],
        key=lambda row: (-_composite(row), row.scenario_id),
    )
    restricted = _sorted_bucket(
        [row for row in candidates if row.bucket == BUCKET_RESTRICTED],
        key=lambda row: (_composite(row), row.scenario_id),
    )
    ties = _sorted_bucket(
        [row for row in candidates if row.bucket == BUCKET_TIE],
        key=lambda row: (
            0 if category_of(row.scenario_id) == "happy_path" else 1,
            row.scenario_id,
        ),
    )
    ranked.extend(helped)
    ranked.extend(restricted)
    ranked.extend(ties)
    ranked.extend(remaining)
    return sorted(ranked, key=lambda row: row.scenario_id)


def _sorted_bucket(
    candidates: Sequence[CaseCandidate],
    *,
    key: Callable[[CaseCandidate], tuple[object, ...]],
) -> list[CaseCandidate]:
    ordered = sorted(candidates, key=key)
    return [replace(row, rank=index) for index, row in enumerate(ordered, start=1)]


def _composite(candidate: CaseCandidate) -> float:
    if candidate.composite is None:
        raise ValueError(f"{candidate.scenario_id} has no composite")
    return candidate.composite


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError(f"cannot parse int from {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"cannot parse int from {value!r}")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"cannot parse float from {value!r}")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"cannot parse float from {value!r}")


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return _as_float(value)
