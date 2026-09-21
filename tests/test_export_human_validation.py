"""Lightweight checks for scripts/export_human_validation.py."""

import csv
import hashlib
import json
from io import StringIO
from pathlib import Path

import pytest

from helpers import (
    REPO_ROOT,
    load_script,
    make_dialogue_log,
    make_turn_record,
    write_run,
)
from sim.audit import SIDECAR_METRICS_SHA256, file_sha256
from sim.kb import KnowledgeBase
from sim.reporting import UNPAIRED_SCENARIO
from sim.runner import FROZEN_V1_HASH
from sim.schemas import DialogueLog, Scenario, dialogue_id

exporter = load_script("scripts/export_human_validation.py")

CANONICAL_METRICS = REPO_ROOT / "results" / "exp_final" / "metrics.csv"
SCORE_SENTINEL = "JUDGE_OUTPUT_SENTINEL"
GENERATED_AT = "2026-09-20T19:00:00Z"


def _write_metrics(path: Path, logs: list[DialogueLog]) -> Path:
    """Write identity rows plus a score-column sentinel the packet must ignore."""
    path.write_text(
        "scenario_id,agent,repetition,task_completed,claim_support\n"
        + "".join(
            f"{log.scenario_id},{log.agent},{log.repetition},"
            f"{SCORE_SENTINEL},{SCORE_SENTINEL}\n"
            for log in logs
        ),
        encoding="utf-8",
    )
    return path


def _make_log(
    scenario_id: str,
    agent: str,
    repetition: int,
    *,
    n_turns: int = 2,
    status: str = "ok",
    canary: str | None = None,
    state_after: str | None = "greeting",
) -> DialogueLog:
    """A short dialogue; replies never name the agent."""
    records = [
        make_turn_record(
            turn,
            user_message=f"question {turn}"
            + (f" {canary}" if canary and turn == 1 else ""),
            agent_reply=f"answer {turn}"
            + (f" token {canary}" if canary and turn == 1 else ""),
            state_after=state_after if agent == "fsm" else None,
            state_before="identification" if agent == "fsm" else None,
        )
        for turn in range(1, n_turns + 1)
    ]
    return make_dialogue_log(
        records,
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
        status=status,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[DialogueLog]]:
    """Four census dialogues: both conditions, one failed CFP-style log, one canary."""
    logs = [
        _make_log("happy_path_01", "baseline", 1, n_turns=2),
        _make_log("happy_path_01", "fsm", 1, n_turns=3, status="failed"),
        _make_log(
            "adversarial_01", "baseline", 1, n_turns=2, canary="ZX4-LANTERN-9021"
        ),
        _make_log("adversarial_01", "fsm", 1, n_turns=1),
    ]
    run_dir = write_run(
        tmp_path / "exp_final",
        logs,
        scenarios_dir="data/scenarios/examples",
    )
    metrics = _write_metrics(tmp_path / "metrics.csv", logs)
    out_dir = tmp_path / "packet"
    return run_dir, metrics, out_dir, logs


def _with_canary(scenarios: dict[str, Scenario]) -> dict[str, Scenario]:
    """Copy example scenarios, planting a canary on adversarial_01."""
    planted = scenarios["adversarial_01"].model_copy(
        update={"canary": "ZX4-LANTERN-9021"}
    )
    return {**scenarios, "adversarial_01": planted}


def _rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dictionaries."""
    return list(csv.DictReader(StringIO(path.read_text(encoding="utf-8"))))


def test_canonical_metrics_census_is_348_and_59_paired() -> None:
    refs = exporter.load_identity(CANONICAL_METRICS)

    exporter.require_canonical_census(refs, CANONICAL_METRICS)

    n_scenarios, n_paired, unpaired = exporter.scenario_counts(refs)
    assert len(refs) == 348
    assert n_paired == 59
    assert unpaired == frozenset({UNPAIRED_SCENARIO})
    assert n_scenarios == 60
    assert file_sha256(CANONICAL_METRICS) == SIDECAR_METRICS_SHA256
    assert {ref.condition for ref in refs} == {"baseline", "fsm"}


def test_canonical_mismatch_raises_before_writing(
    tmp_path: Path, example_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    run_dir, metrics, out_dir, _logs = _fixture(tmp_path)

    with pytest.raises(exporter.PacketError, match="sha256"):
        exporter.export_packet(
            run_dir=run_dir,
            metrics_path=metrics,
            out_dir=out_dir,
            kb=real_kb,
            scenarios=_with_canary(example_scenarios),
            enforce_canonical=True,
        )

    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_packet_has_unique_blind_ids_empty_judgments_and_header_only_claims(
    tmp_path: Path, example_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    run_dir, metrics, out_dir, logs = _fixture(tmp_path)

    packet = exporter.export_packet(
        run_dir=run_dir,
        metrics_path=metrics,
        out_dir=out_dir,
        kb=real_kb,
        scenarios=_with_canary(example_scenarios),
        generated_at=GENERATED_AT,
    )

    dialogue_rows = _rows(out_dir / exporter.DIALOGUE_CSV)
    response_rows = _rows(out_dir / exporter.RESPONSE_CSV)
    claims_rows = _rows(out_dir / exporter.CLAIMS_CSV)
    mapping = _rows(out_dir / exporter.PRIVATE_DIR / exporter.MAPPING_CSV)
    n_replies = sum(
        1 for log in logs for record in log.records if record.agent_reply.strip()
    )
    assert [row["blind_dialogue_id"] for row in dialogue_rows] == [
        "D001",
        "D002",
        "D003",
        "D004",
    ]
    assert len(packet.dialogues) == 4
    assert len(response_rows) == n_replies
    assert len({row["blind_response_id"] for row in response_rows}) == n_replies
    assert all(
        row["task_completed"] == "" and row["notes"] == "" for row in dialogue_rows
    )
    assert all(
        row["fact_ids_stated"] == "" and row["notes"] == "" for row in response_rows
    )
    assert claims_rows == []
    assert list(response_rows[0]) == list(exporter.RESPONSE_FIELDS)
    assert "unsupported_claims" not in response_rows[0]
    assert "claim_support" not in response_rows[0]
    assert [row["blind_dialogue_id"] for row in mapping] == [
        "D001",
        "D002",
        "D003",
        "D004",
    ]
    assert {row["dialogue_id"] for row in mapping} == {
        dialogue_id(log.scenario_id, log.agent, log.repetition) for log in logs
    }
    assert {row["condition"] for row in mapping} == {"baseline", "fsm"}
    failed = next(log for log in logs if log.status == "failed")
    assert any(
        row["dialogue_id"]
        == dialogue_id(failed.scenario_id, failed.agent, failed.repetition)
        for row in mapping
    )
    md = (out_dir / exporter.DIALOGUES_DIR / "D001.md").read_text(encoding="utf-8")
    assert "## Success criterion" in md
    assert "## Expected facts" in md
    assert "F09" in md or "F10" in md


def test_same_seed_reproduces_mapping_and_public_files_write_before_private(
    tmp_path: Path, example_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    run_dir, metrics, out_dir, _logs = _fixture(tmp_path)
    packet = exporter.build_packet(
        run_dir=run_dir,
        metrics_path=metrics,
        kb=real_kb,
        scenarios=_with_canary(example_scenarios),
        generated_at=GENERATED_AT,
        seed=20260920,
    )
    again = exporter.build_packet(
        run_dir=run_dir,
        metrics_path=metrics,
        kb=real_kb,
        scenarios=_with_canary(example_scenarios),
        generated_at=GENERATED_AT,
        seed=20260920,
    )
    other = exporter.build_packet(
        run_dir=run_dir,
        metrics_path=metrics,
        kb=real_kb,
        scenarios=_with_canary(example_scenarios),
        generated_at=GENERATED_AT,
        seed=1,
    )

    assert [item.ref.dialogue_id for item in packet.dialogues] == [
        item.ref.dialogue_id for item in again.dialogues
    ]
    assert [item.ref.dialogue_id for item in packet.dialogues] != [
        item.ref.dialogue_id for item in other.dialogues
    ]
    assert [item.ref.condition for item in packet.dialogues] != sorted(
        item.ref.condition for item in packet.dialogues
    )

    exporter.write_public(packet, out_dir)

    assert (out_dir / exporter.DIALOGUE_CSV).exists()
    assert (out_dir / exporter.MANIFEST_JSON).exists()
    assert not (out_dir / exporter.PRIVATE_DIR / exporter.MAPPING_CSV).exists()
    exporter.validate_packet(out_dir, mapping_required=False)
    manifest = json.loads(
        (out_dir / exporter.MANIFEST_JSON).read_text(encoding="utf-8")
    )
    assert "mapping_sha256" not in manifest
    assert "dialogue_ids" not in manifest
    assert manifest["seed"] == 20260920
    assert manifest["frozen_v1_hash"] == FROZEN_V1_HASH

    exporter.write_private(packet, out_dir)
    exporter.validate_packet(out_dir, mapping_required=True)


def test_metadata_leak_checks_headers_not_spoken_text(
    tmp_path: Path, example_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    log = make_dialogue_log(
        [
            make_turn_record(
                1,
                user_message="please speak to baseline staff",
                agent_reply="the fsm procedure is not named here",
                state_after="greeting",
                state_before="greeting",
            )
        ],
        scenario_id="happy_path_01",
        agent="fsm",
        repetition=1,
    )
    pair = make_dialogue_log(
        [make_turn_record(1, agent_reply="ok")],
        scenario_id="happy_path_01",
        agent="baseline",
        repetition=1,
    )
    run_dir = write_run(
        tmp_path / "exp",
        [log, pair],
        scenarios_dir="data/scenarios/examples",
    )
    metrics = _write_metrics(tmp_path / "metrics.csv", [log, pair])
    out_dir = tmp_path / "out"

    exporter.export_packet(
        run_dir=run_dir,
        metrics_path=metrics,
        out_dir=out_dir,
        kb=real_kb,
        scenarios=_with_canary(example_scenarios),
        generated_at=GENERATED_AT,
    )

    for name in (exporter.DIALOGUE_CSV, exporter.RESPONSE_CSV, exporter.CLAIMS_CSV):
        header = (out_dir / name).read_text(encoding="utf-8").splitlines()[0]
        assert "condition" not in header
        assert "true_state_after" not in header
        assert "judge_facts" not in header
    manifest = (out_dir / exporter.MANIFEST_JSON).read_text(encoding="utf-8")
    assert "true_state_after" not in manifest
    assert "labelled_stage" not in manifest
    spoken = False
    for path in (out_dir / exporter.DIALOGUES_DIR).glob("D*.md"):
        text = path.read_text(encoding="utf-8")
        prefix, _, body = text.partition("## Dialogue")
        assert "true_state_after" not in prefix
        assert "labelled_stage" not in prefix
        assert "baseline" not in prefix.lower()
        assert "fsm" not in prefix.lower()
        spoken = spoken or "baseline" in body or "fsm" in body
    assert spoken


def test_score_sentinel_never_reaches_the_packet(
    tmp_path: Path, example_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    run_dir, metrics, out_dir, _logs = _fixture(tmp_path)

    exporter.export_packet(
        run_dir=run_dir,
        metrics_path=metrics,
        out_dir=out_dir,
        kb=real_kb,
        scenarios=_with_canary(example_scenarios),
        generated_at=GENERATED_AT,
    )

    for path in [out_dir / exporter.DIALOGUE_CSV, out_dir / exporter.RESPONSE_CSV]:
        assert SCORE_SENTINEL not in path.read_text(encoding="utf-8")
    for path in (out_dir / exporter.DIALOGUES_DIR).glob("D*.md"):
        assert SCORE_SENTINEL not in path.read_text(encoding="utf-8")
    assert SCORE_SENTINEL not in (out_dir / exporter.MANIFEST_JSON).read_text(
        encoding="utf-8"
    )


def test_frozen_artifacts_are_unchanged_and_canary_is_redacted(
    tmp_path: Path, example_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    run_dir, metrics, out_dir, _logs = _fixture(tmp_path)
    before = {
        path: file_sha256(path) for path in sorted((run_dir / "dialogues").iterdir())
    }
    metrics_before = file_sha256(metrics)

    exporter.export_packet(
        run_dir=run_dir,
        metrics_path=metrics,
        out_dir=out_dir,
        kb=real_kb,
        scenarios=_with_canary(example_scenarios),
        generated_at=GENERATED_AT,
    )

    assert file_sha256(metrics) == metrics_before
    after = {
        path: file_sha256(path) for path in sorted((run_dir / "dialogues").iterdir())
    }
    assert after == before
    canary = _with_canary(example_scenarios)["adversarial_01"].canary
    assert canary is not None
    leaked = False
    redacted = False
    for path in (out_dir / exporter.DIALOGUES_DIR).glob("D*.md"):
        text = path.read_text(encoding="utf-8")
        leaked = leaked or canary in text
        redacted = redacted or "[redacted]" in text
        for fact in example_scenarios["adversarial_01"].required_facts:
            if fact in text:
                assert any(listed.id == fact for listed in real_kb.facts)
    assert not leaked
    assert redacted


def test_missing_dialogue_jsonl_raises(
    tmp_path: Path, example_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    run_dir, metrics, out_dir, logs = _fixture(tmp_path)
    dropped = logs[0]
    (
        run_dir
        / "dialogues"
        / f"{dialogue_id(dropped.scenario_id, dropped.agent, dropped.repetition)}.jsonl"
    ).unlink()

    with pytest.raises(exporter.PacketError, match="missing"):
        exporter.export_packet(
            run_dir=run_dir,
            metrics_path=metrics,
            out_dir=out_dir,
            kb=real_kb,
            scenarios=_with_canary(example_scenarios),
        )


def test_refuse_overwrite_when_annotation_cells_are_filled(
    tmp_path: Path, example_scenarios: dict[str, Scenario], real_kb: KnowledgeBase
) -> None:
    run_dir, metrics, out_dir, _logs = _fixture(tmp_path)
    exporter.export_packet(
        run_dir=run_dir,
        metrics_path=metrics,
        out_dir=out_dir,
        kb=real_kb,
        scenarios=_with_canary(example_scenarios),
        generated_at=GENERATED_AT,
    )
    sheet = out_dir / exporter.DIALOGUE_CSV
    rows = _rows(sheet)
    rows[0]["task_completed"] = "yes"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=exporter.DIALOGUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(exporter.PacketError, match="overwrite"):
        exporter.export_packet(
            run_dir=run_dir,
            metrics_path=metrics,
            out_dir=out_dir,
            kb=real_kb,
            scenarios=_with_canary(example_scenarios),
            generated_at=GENERATED_AT,
        )


def test_main_writes_the_package_layout(
    tmp_path: Path, example_scenarios: dict[str, Scenario]
) -> None:
    run_dir, metrics, out_dir, _logs = _fixture(tmp_path)

    code = exporter.main(
        [
            "--run",
            str(run_dir),
            "--metrics",
            str(metrics),
            "--out",
            str(out_dir),
            "--seed",
            "20260920",
        ]
    )

    assert code == 1
    assert not (out_dir / exporter.DIALOGUE_CSV).exists()


def test_transcripts_hash_is_ordered_by_dialogue_id(
    tmp_path: Path, example_scenarios: dict[str, Scenario]
) -> None:
    run_dir, metrics, _out_dir, logs = _fixture(tmp_path)
    refs = exporter.load_identity(metrics)

    digest = exporter.transcripts_sha256(run_dir, refs)

    lines = []
    for log in sorted(
        logs,
        key=lambda item: dialogue_id(item.scenario_id, item.agent, item.repetition),
    ):
        path = (
            run_dir
            / "dialogues"
            / f"{dialogue_id(log.scenario_id, log.agent, log.repetition)}.jsonl"
        )
        lines.append(file_sha256(path) + "\n")
    assert digest == hashlib.sha256("".join(lines).encode("ascii")).hexdigest()
