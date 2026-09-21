"""Tests for scripts/prepare_eval_sidecar.py."""

from pathlib import Path

import pytest

from helpers import load_script, make_dialogue_log, make_turn_record, write_run

sidecar = load_script("scripts/prepare_eval_sidecar.py")


def _one_log() -> object:
    return make_dialogue_log(
        [
            make_turn_record(
                1,
                user_message="Where is the order?",
                agent_reply="I can look that up.",
            )
        ],
        scenario_id="happy_path_01",
        agent="baseline",
    )


def test_prepare_copies_only_manifest_and_dialogues(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_run(source, [_one_log()], scenarios_dir="data/scenarios/examples")
    (source / "cache").mkdir()
    (source / "cache" / "ab.json").write_text("{}", encoding="utf-8")
    (source / "metrics.csv").write_text("scenario_id\n", encoding="utf-8")
    (source / "llm_calls.jsonl").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "pilot_v2_mlx"

    sidecar.prepare_eval_sidecar(source, target)

    assert (target / "manifest.json").read_bytes() == (
        source / "manifest.json"
    ).read_bytes()
    source_names = sorted(path.name for path in (source / "dialogues").glob("*.jsonl"))
    target_names = sorted(path.name for path in (target / "dialogues").glob("*.jsonl"))
    assert target_names == source_names
    for name in source_names:
        assert (target / "dialogues" / name).read_bytes() == (
            source / "dialogues" / name
        ).read_bytes()
    assert not (target / "cache").exists()
    assert not (target / "metrics.csv").exists()
    assert not (target / "llm_calls.jsonl").exists()


def test_prepare_refuses_source_equal_to_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_run(source, [_one_log()], scenarios_dir="data/scenarios/examples")

    with pytest.raises(sidecar.SidecarError, match="same path"):
        sidecar.prepare_eval_sidecar(source, source)


def test_prepare_refuses_frozen_pilot_v2_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_run(source, [_one_log()], scenarios_dir="data/scenarios/examples")
    frozen = Path("runs/pilot_v2")
    before = (
        (frozen / "manifest.json").read_bytes()
        if (frozen / "manifest.json").exists()
        else None
    )

    with pytest.raises(sidecar.SidecarError, match="pilot_v2"):
        sidecar.prepare_eval_sidecar(source, frozen)

    if before is not None:
        assert (frozen / "manifest.json").read_bytes() == before


def test_prepare_leaves_existing_cache_when_dialogues_match(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_run(source, [_one_log()], scenarios_dir="data/scenarios/examples")
    target = tmp_path / "pilot_v2_mlx"
    sidecar.prepare_eval_sidecar(source, target)
    cache_file = target / "cache" / "ab.json"
    cache_file.parent.mkdir()
    cache_file.write_text('{"kept": true}\n', encoding="utf-8")

    sidecar.prepare_eval_sidecar(source, target)

    assert cache_file.read_text(encoding="utf-8") == '{"kept": true}\n'


def test_prepare_refuses_when_target_dialogues_differ(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_run(source, [_one_log()], scenarios_dir="data/scenarios/examples")
    target = tmp_path / "pilot_v2_mlx"
    sidecar.prepare_eval_sidecar(source, target)
    dialogue = next((target / "dialogues").glob("*.jsonl"))
    dialogue.write_text('{"altered": true}\n', encoding="utf-8")

    with pytest.raises(sidecar.SidecarError, match="differ"):
        sidecar.prepare_eval_sidecar(source, target)
