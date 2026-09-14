"""Tests for scripts/relabel_stages.py."""

import csv
from pathlib import Path

from helpers import (
    FakeLlm,
    load_script,
    make_dialogue_log,
    make_turn_record,
    stage_labels_reply,
    strip_fsm_meta,
    write_run,
)

relabel = load_script("scripts/relabel_stages.py")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_relabel_copies_manifest_and_dialogues_and_writes_metrics_turn(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    fsm_records = [
        make_turn_record(
            turn=1,
            user_message="Need tracking and maybe cancel. me@example.com",
            agent_reply="Track or cancel?",
            state_after="intent_classification",
        ),
        make_turn_record(
            turn=2,
            user_message="Track only.",
            agent_reply="Here is the tracking policy.",
            state_after="solution",
        ),
    ]
    logs = [
        make_dialogue_log(fsm_records, scenario_id="edge_02", agent="fsm"),
        make_dialogue_log(
            strip_fsm_meta(fsm_records),
            scenario_id="edge_02",
            agent="baseline",
        ),
    ]
    write_run(source, logs, scenarios_dir="data/scenarios/examples")

    llm = FakeLlm(
        [
            stage_labels_reply(["intent_classification", "solution"]),
            stage_labels_reply(["intent_classification", "solution"]),
        ]
    )
    target = tmp_path / "sidecar"

    rows_written = relabel.relabel(source, target, llm=llm)

    assert rows_written == 4
    assert (target / "manifest.json").read_text(encoding="utf-8") == (
        source / "manifest.json"
    ).read_text(encoding="utf-8")
    assert sorted(path.name for path in (target / "dialogues").glob("*.jsonl")) == (
        sorted(path.name for path in (source / "dialogues").glob("*.jsonl"))
    )
    rows = _rows(target / "metrics_turn.csv")
    assert len(rows) == 4
    fsm_rows = [row for row in rows if row["agent"] == "fsm"]
    assert [row["true_state_after"] for row in fsm_rows] == [
        "intent_classification",
        "solution",
    ]
    assert all(call["role"] == "state_labeler" for call in llm.calls)
    assert all(call["caller"] == "stage_labeler" for call in llm.calls)
    assert not (source / "metrics_turn.csv").exists()


def test_relabel_skips_failed_dialogues(tmp_path: Path) -> None:
    source = tmp_path / "source"
    ok_log = make_dialogue_log(
        [
            make_turn_record(
                turn=1,
                user_message="Need a slip.",
                agent_reply="I can help.",
                state_after="intent_classification",
            )
        ],
        scenario_id="adversarial_13",
        agent="fsm",
        repetition=1,
        status="ok",
    )
    failed_log = make_dialogue_log(
        [],
        scenario_id="adversarial_13",
        agent="fsm",
        repetition=2,
        status="failed",
    )
    write_run(source, [ok_log, failed_log], scenarios_dir="data/scenarios/examples")
    llm = FakeLlm([stage_labels_reply(["intent_classification"])])
    target = tmp_path / "sidecar"

    rows_written = relabel.relabel(source, target, llm=llm)

    assert rows_written == 1
    assert len(llm.calls) == 1
    rows = _rows(target / "metrics_turn.csv")
    assert len(rows) == 1
    assert rows[0]["repetition"] == "1"
