"""T-17 workflow: discover 9, audit evidence, freeze 8, eligibility 342 + 8."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from helpers import (
    load_script,
    make_dialogue_log,
    make_turn_record,
    write_run,
    write_scenarios,
)
from sim.adjudication import (
    freeze_contract_false_positives,
    iter_dialogue_logs,
    load_failed_inclusion_allowlist,
    resolve_included_failed,
)
from sim.schemas import DialogueLog, Manifest, Scenario

audit = load_script("scripts/audit_failed_dialogues.py")
discover = load_script("scripts/list_instrument_failures.py")

SELECTED = (
    "adversarial_01__baseline__rep03",
    "edge_16__baseline__rep02",
    "edge_16__baseline__rep03",
    "happy_path_09__baseline__rep02",
    "adversarial_19__fsm__rep02",
    "edge_16__fsm__rep02",
    "edge_16__fsm__rep03",
    "happy_path_09__fsm__rep02",
)
TRUE_INSTRUMENT = "adversarial_06__fsm__rep02"
INSTRUMENT_IDS = (*SELECTED, TRUE_INSTRUMENT)


def _scenario(scenario_id: str) -> Scenario:
    """A valid scenario whose id carries its category."""
    category = scenario_id.rsplit("_", 1)[0]
    return Scenario(
        id=scenario_id,
        category=category,  # type: ignore[arg-type]
        intent="order_tracking",
        user_persona=f"A customer in {scenario_id}.",
        user_goal=f"Track the order for {scenario_id}.",
        script=["Where is order NL-20260145?"],
        reference_answer="Standard delivery takes 5 business days.",
        required_facts=["F09"],
        expected_final_state="closing",
        success_criterion="The estimate is provided.",
        max_turns=6,
    )


def _log(
    run_id: str,
    *,
    status: str,
    kind: str | None = None,
    reason: str | None = None,
) -> DialogueLog:
    """Build one log whose dialogue id is ``run_id``."""
    scenario_id, agent, rep = run_id.rsplit("__", 2)
    repetition = int(rep.removeprefix("rep"))
    log = make_dialogue_log(
        [make_turn_record(user_message="Where is order NL-20260145?")],
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
        status=status,  # type: ignore[arg-type]
        stop_reason=None if status == "failed" else "goal_reached",
    )
    if kind is None:
        return log
    return log.model_copy(update={"failure_kind": kind, "failure_reason": reason})


def _t17_fixture(tmp_path: Path) -> Path:
    """Nine instrument failures, nine max-turn failures, and 342 ok logs."""
    logs: list[DialogueLog] = []
    for item in INSTRUMENT_IDS:
        logs.append(
            _log(
                item,
                status="failed",
                kind="instrument",
                reason="invalid_candidate_retry_exhausted",
            )
        )
    max_ids = [f"happy_path_{50 + index:02d}__fsm__rep01" for index in range(9)]
    for item in max_ids:
        logs.append(
            _log(
                item,
                status="failed",
                kind="simulation",
                reason="max_turns_with_incomplete_beat",
            )
        )
    for index in range(342):
        logs.append(_log(f"ok_{index:03d}__baseline__rep01", status="ok"))
    scenario_ids = sorted({log.scenario_id for log in logs if log.status == "failed"})
    scenarios: Sequence[Scenario] = [_scenario(item) for item in scenario_ids]
    scenario_dir = write_scenarios(
        tmp_path / "scenarios",
        cases=[item.model_dump(mode="json") for item in scenarios],
    )
    run_dir = write_run(tmp_path / "exp_final", logs, scenarios_dir=str(scenario_dir))
    manifest = Manifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    (run_dir / "manifest.json").write_text(
        manifest.model_copy(update={"exp_id": "exp_final"}).model_dump_json(),
        encoding="utf-8",
    )
    return run_dir


def test_t17_discover_audit_freeze_eligibility_population(tmp_path: Path) -> None:
    run_dir = _t17_fixture(tmp_path)
    jsonls = sorted((run_dir / "dialogues").glob("*.jsonl"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in jsonls}
    manifest_before = (run_dir / "manifest.json").read_bytes()

    assert discover.main([str(run_dir)]) == 0
    discovery = json.loads(
        (run_dir / "adjudication" / "instrument_failures.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["run_id"] for row in discovery["candidates"]] == sorted(INSTRUMENT_IDS)
    census = {
        (row["failure_kind"], row["failure_reason"]): row["count"]
        for row in discovery["census"]
    }
    assert census[("instrument", "invalid_candidate_retry_exhausted")] == 9
    assert census[("simulation", "max_turns_with_incomplete_beat")] == 9
    assert all(row["needs_adjudication"] for row in discovery["candidates"])
    blob = json.dumps(discovery)
    assert "CONTRACT_FALSE_POSITIVE" not in blob
    assert "TRUE_INSTRUMENT_FAILURE" not in blob

    assert audit.main(["--run", str(run_dir)]) == 0
    audit_json = json.loads(
        (run_dir / "adjudication" / "audit.json").read_text(encoding="utf-8")
    )
    audit_md = (run_dir / "adjudication" / "audit.md").read_text(encoding="utf-8")
    assert audit_json["not_an_adjudication"] is True
    assert all(
        item["review_status"] == "needs_adjudication"
        for item in audit_json["dialogues"]
    )
    assert "CONTRACT_FALSE_POSITIVE" not in audit_md
    assert "Needs adjudication" in audit_md
    freeze_path = run_dir / "adjudication" / "contract_false_positives.json"
    assert not freeze_path.exists()

    path = freeze_contract_false_positives(run_dir, SELECTED)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ids"] == sorted(SELECTED)
    assert TRUE_INSTRUMENT not in payload["ids"]
    assert all(item in INSTRUMENT_IDS for item in payload["ids"])
    assert len(payload["ids"]) == 8
    first_bytes = path.read_bytes()
    freeze_contract_false_positives(run_dir, SELECTED)
    assert path.read_bytes() == first_bytes

    allowlist = load_failed_inclusion_allowlist(path)
    logs = [log for _source, log in iter_dialogue_logs(run_dir)]
    ok = [log for log in logs if log.status == "ok"]
    included = resolve_included_failed(logs, allowlist)
    assert len(ok) == 342
    assert len(included) == 8
    assert {log.dialogue_id for log in included} == set(SELECTED)
    assert TRUE_INSTRUMENT not in {log.dialogue_id for log in included}
    assert not any(
        log.failure_reason == "max_turns_with_incomplete_beat" for log in included
    )

    assert (run_dir / "manifest.json").read_bytes() == manifest_before
    for path_jsonl, digest in before.items():
        assert hashlib.sha256(path_jsonl.read_bytes()).hexdigest() == digest
