"""Primary tests without the ambiguous-fact-ID dialogues (T-25, post-hoc).

The advisor asked whether the paired tests hold without the census
dialogues whose judge fact IDs could not be recovered unambiguously.
In each, an extra GGUF ``judge_facts`` call reproduced the same
frozen score with other predicted IDs. The list comes from
``scripts/score_human_validation.py``. This module drops those dialogues from
the judge and human metrics rows and reruns T-19's ``paired_test_table`` on the
three primary metrics. The population is exploratory, and Holm is computed here
only as a reference for comparison with the confirmatory tables.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from csv import DictReader, DictWriter
from csv import writer as csv_writer
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path

from sim.analysis import (
    CATEGORIES,
    DROP_AMBIGUOUS_FACT_IDS,
    N_RESAMPLES,
    OVERALL,
    PRIMARY_METRICS,
    SEED,
    TestRow,
    category_of,
    dialogue_key,
    holm_adjust,
    paired_test_table,
    rows_without_dialogues,
)
from sim.decomposition import DialogueKey
from sim.io import atomic_write
from sim.runner import AGENTS

KEY_COLUMNS = ("scenario_id", "agent", "repetition")


def load_dialogue_keys(path: Path) -> list[DialogueKey]:
    """Read ``scenario_id,agent,repetition`` rows, keeping duplicates."""
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            (row["scenario_id"], row["agent"], int(row["repetition"]))
            for row in DictReader(handle)
        ]


def write_dialogue_keys(path: Path, keys: Collection[DialogueKey]) -> None:
    """Write ``keys`` sorted, one ``scenario_id,agent,repetition`` row each."""
    buffer = StringIO()
    writer = csv_writer(buffer)
    writer.writerow(KEY_COLUMNS)
    writer.writerows(sorted(keys))
    atomic_write(path, buffer.getvalue())


def ambiguous_counts(
    rows: Sequence[Mapping[str, object]], keys: Collection[DialogueKey]
) -> list[dict[str, object]]:
    """Dialogues and excluded dialogues per stratum and agent."""
    excluded = set(keys)
    dialogues = [dialogue_key(row) for row in rows]
    counts: list[dict[str, object]] = []
    for stratum in (OVERALL, *CATEGORIES):
        for agent in AGENTS:
            in_cell = [
                key
                for key in dialogues
                if key[1] == agent and stratum in {OVERALL, category_of(key[0])}
            ]
            counts.append(
                {
                    "stratum": stratum,
                    "agent": agent,
                    "n_dialogues": len(in_cell),
                    "n_ambiguous": sum(key in excluded for key in in_cell),
                }
            )
    return counts


def sensitivity_rows(
    rows_by_instrument: Mapping[str, Sequence[Mapping[str, object]]],
    keys: Collection[DialogueKey],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> list[dict[str, object]]:
    """Primary paired tests per instrument after dropping ``keys``."""
    table: list[dict[str, object]] = []
    for instrument, rows in rows_by_instrument.items():
        tests = paired_test_table(
            rows_without_dialogues(rows, keys),
            population=DROP_AMBIGUOUS_FACT_IDS,
            metrics=PRIMARY_METRICS,
            n_resamples=n_resamples,
            seed=seed,
        )
        primary = [row for row in tests if row.metric in PRIMARY_METRICS]
        table.extend(
            {"instrument": instrument, **asdict(row)}
            for row in with_reference_holm(primary)
        )
    return table


def with_reference_holm(tests: Sequence[TestRow]) -> list[TestRow]:
    """Holm over the overall primary Wilcoxon p-values; category rows untouched."""
    overall = {
        row.metric: row.wilcoxon_p
        for row in tests
        if row.stratum == OVERALL and row.wilcoxon_p is not None
    }
    adjusted = holm_adjust(overall)
    return [
        replace(row, wilcoxon_p_holm=adjusted.get(row.metric))
        if row.stratum == OVERALL
        else row
        for row in tests
    ]


def write_sensitivity(
    out_dir: Path,
    *,
    tests: Sequence[Mapping[str, object]],
    counts: Sequence[Mapping[str, object]],
) -> None:
    """Write ``tests.csv`` and ``counts.csv`` into ``out_dir``."""
    _write_csv(out_dir / "tests.csv", tests)
    _write_csv(out_dir / "counts.csv", counts)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write ``rows`` with empty cells for None."""
    if not rows:
        raise ValueError(f"{path} would have no rows")
    fieldnames = list(rows[0].keys())
    buffer = StringIO()
    writer = DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: "" if row[key] is None else row[key] for key in fieldnames}
        )
    atomic_write(path, buffer.getvalue())
