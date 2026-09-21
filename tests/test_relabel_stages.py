"""Tests for scripts/relabel_stages.py."""

import csv
from pathlib import Path

import pytest

from helpers import (
    FakeLlm,
    load_script,
    make_dialogue_log,
    make_turn_record,
    stage_labels_reply,
    strip_fsm_meta,
    write_models_config,
    write_run,
)
from sim.config import load_models_config

relabel = load_script("scripts/relabel_stages.py")

_NINE_B_LABELER_YAML = """\
ollama:
  version: "0.33.3"
  env:
    OLLAMA_NUM_PARALLEL: 2
num_ctx: 8192
max_prompt_fraction: 0.8
client:
  timeout_s: 300
  retries: 2
  backoff_s: 2
models:
  agent:
    name: "qwen3.5:9b"
    digest: "digest-9b"
    temperature: 0.7
    seed: 42
  simulated_user:
    name: "qwen3.5:9b"
    digest: "digest-9b"
  classifier:
    name: "qwen3.5:4b"
    digest: "digest-4b"
    temperature: 0.7
    seed: 42
  state_labeler:
    name: "qwen3.5:9b"
    digest: "digest-9b"
    temperature: 0.7
    seed: 42
  judge:
    name: "judge-model"
    temperature: 0
    seed: 42
"""


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


def test_relabel_refuses_frozen_pilot_v2_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_run(
        source,
        [
            make_dialogue_log(
                [
                    make_turn_record(
                        turn=1,
                        state_after="closing",
                    )
                ],
                scenario_id="happy_path_01",
                agent="fsm",
            )
        ],
        scenarios_dir="data/scenarios/examples",
    )
    frozen = Path("runs/pilot_v2")
    before = (
        (frozen / "manifest.json").read_bytes()
        if (frozen / "manifest.json").exists()
        else None
    )
    llm = FakeLlm([stage_labels_reply(["closing"])])

    with pytest.raises(relabel.RelabelError, match="pilot_v2"):
        relabel.relabel(source, frozen, llm=llm)

    if before is not None:
        assert (frozen / "manifest.json").read_bytes() == before


def test_relabel_default_refuses_a_non_4b_labeler(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_run(
        source,
        [
            make_dialogue_log(
                [make_turn_record(turn=1, state_after="closing")],
                scenario_id="happy_path_01",
                agent="fsm",
            )
        ],
        scenarios_dir="data/scenarios/examples",
    )
    config = load_models_config(write_models_config(tmp_path, _NINE_B_LABELER_YAML))
    llm = FakeLlm([stage_labels_reply(["closing"])])

    with pytest.raises(relabel.RelabelError, match=r"qwen3\.5:4b"):
        relabel.relabel(source, tmp_path / "sidecar", llm=llm, config=config)


def test_relabel_model_override_accepts_9b_and_writes_turns(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_run(
        source,
        [
            make_dialogue_log(
                [make_turn_record(turn=1, state_after="closing")],
                scenario_id="happy_path_01",
                agent="fsm",
            )
        ],
        scenarios_dir="data/scenarios/examples",
    )
    config = load_models_config(write_models_config(tmp_path, _NINE_B_LABELER_YAML))
    llm = FakeLlm([stage_labels_reply(["closing"])])
    target = tmp_path / "sidecar"

    rows_written = relabel.relabel(
        source, target, llm=llm, config=config, model="qwen3.5:9b"
    )

    assert rows_written == 1
    assert _rows(target / "metrics_turn.csv")[0]["labelled_stage"] == "closing"
    assert not (source / "metrics_turn.csv").exists()
