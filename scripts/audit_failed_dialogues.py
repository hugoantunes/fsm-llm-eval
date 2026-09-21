"""Audit failed dialogues for possible contract false-positive mechanisms.

This script reads a ``sim run`` directory and writes generated evidence under
``<run>/adjudication/``. It does not decide experimental inclusion, does not
recode original JSONL files, and does not write or regenerate the inclusion
artifact.

Usage::

    uv run python scripts/audit_failed_dialogues.py --run runs/exp_final
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from sim.adjudication import (
    CONTRACT_FALSE_POSITIVES_NAME,
    adjudication_dir,
)
from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.io import atomic_write
from sim.kb import DEFAULT_KB_DIR, UserDataField, load_kb
from sim.llm import LLM_CALLS_LOG, LlmCallRecord
from sim.schemas import (
    DialogueLog,
    Manifest,
    Scenario,
    as_messages,
    load_scenarios,
    transcript_from_records,
)
from sim.script import (
    _PAST_SHIPMENT,
    _PRESENT_SHIPMENT,
    CLOSING_WORDS,
    QUESTION,
    ScriptProgress,
    _carries,
    _denies,
    _words,
    beats_of,
)
from sim.user import UserReply, _invalid_candidate_reason

REVIEW_STATUS = "needs_adjudication"
SEED_SOURCE = "inferred: dialogue.seed + attempt"
THANKS_WORDS = frozenset(("thanks", "thank"))
SHIPMENT_WORDS = _PRESENT_SHIPMENT | _PAST_SHIPMENT
ALLOWLIST_NOTE = (
    "The systematic audit produces evidence for review; it does not determine "
    "inclusion. Audit flags are evidence-retrieval heuristics, not semantic "
    "classifiers. Their presence or absence does not determine experimental "
    "inclusion. The generated inclusion JSON encodes the result of the completed "
    "adjudication; it is not the adjudication process itself. The audit must "
    "never overwrite or regenerate that file."
)


class AuditError(RuntimeError):
    """The audit cannot start; the message says what is missing."""


class DialogueAudit(BaseModel):
    """Generated evidence for one failed dialogue. Not an inclusion verdict."""

    model_config = ConfigDict(extra="forbid")

    dialogue_id: str
    run_id: str
    scenario_id: str
    agent: str
    repetition: int
    seed: int
    failure_kind: str | None
    failure_reason: str | None
    recorded_failure_metadata: dict[str, Any]
    active_beat: dict[str, Any] | None
    stop_status: str | None
    invalid_reason: str | None
    retry_recovery: dict[str, Any]
    possible_mechanisms: list[dict[str, Any]]
    review_status: Literal["needs_adjudication"] = REVIEW_STATUS


class AuditReport(BaseModel):
    """Machine-readable failed-dialogue audit for one run."""

    model_config = ConfigDict(extra="forbid")

    artifact_kind: Literal["generated_audit_evidence"] = "generated_audit_evidence"
    not_an_adjudication: Literal[True] = True
    allowlist_note: str = ALLOWLIST_NOTE
    run: dict[str, Any]
    dialogues: list[DialogueAudit]

    @property
    def dialogue_map(self) -> dict[str, DialogueAudit]:
        """Index audits by dialogue id."""
        return {item.dialogue_id: item for item in self.dialogues}


def default_out_dir(run_dir: Path) -> Path:
    """Return the default generated-evidence directory for ``run_dir``."""
    return adjudication_dir(run_dir)


def audit_run(run_dir: Path) -> AuditReport:
    """Inspect every failed dialogue of ``run_dir`` and return generated evidence."""
    manifest = _load_manifest(run_dir)
    kb = load_kb(DEFAULT_KB_DIR)
    fsm = load_fsm(DEFAULT_FSM_DIR)
    scenarios = {
        scenario.id: scenario
        for scenario in load_scenarios(Path(manifest.scenarios_dir), kb=kb, fsm=fsm)
    }
    logs = _load_failed_logs(run_dir)
    calls = _load_calls(run_dir / LLM_CALLS_LOG)
    dialogues = [
        _audit_dialogue(
            log,
            scenario=_scenario_of(log, scenarios, manifest.scenarios_dir),
            fields=kb.user_data_fields,
            run_id=manifest.exp_id,
            calls=calls,
        )
        for log in logs
    ]
    return AuditReport(
        run={
            "exp_id": manifest.exp_id,
            "run_dir": str(run_dir),
            "n_failed": len(dialogues),
            "n_ok_skipped": _count_ok(run_dir),
        },
        dialogues=dialogues,
    )


def write_reports(report: AuditReport, out_dir: Path) -> None:
    """Write ``audit.json`` and ``audit.md`` under ``out_dir``.

    Refuses to write the frozen T-17 allowlist filename or path.
    """
    json_path = out_dir / "audit.json"
    markdown_path = out_dir / "audit.md"
    for path in (json_path, markdown_path):
        if path.name == CONTRACT_FALSE_POSITIVES_NAME:
            raise AuditError(
                f"{path} is the executable inclusion artifact. The audit writes "
                "generated evidence only and must not overwrite that file"
            )
    atomic_write(json_path, report.model_dump_json(indent=2) + "\n")
    atomic_write(markdown_path, render_markdown(report) + "\n")


def render_markdown(report: AuditReport) -> str:
    """Render a human-readable audit that stops at Needs adjudication."""
    lines = [
        "# Failed-dialogue audit",
        "",
        ALLOWLIST_NOTE,
        "",
        f"Run `{report.run['exp_id']}` at `{report.run['run_dir']}`: "
        f"{report.run['n_failed']} failed dialogue(s). "
        "OK logs are omitted from this census.",
        "",
        "This report is not part of the semantic judge.",
        "",
    ]
    for item in report.dialogues:
        lines.extend(_render_dialogue(item))
    return "\n".join(lines).rstrip()


def _render_dialogue(item: DialogueAudit) -> list[str]:
    """Render the eight review sections for one failed dialogue."""
    beat = item.active_beat or {}
    missing = beat.get("missing_cumulative_requirements", {})
    pending = beat.get("pending_local_predicates", [])
    flags = [str(flag.get("mechanism")) for flag in item.possible_mechanisms]
    cfp_hints = [
        name for name in flags if name not in ("possible_required_content_absent",)
    ]
    absent = "possible_required_content_absent" in flags
    return [
        f"## `{item.dialogue_id}`",
        "",
        "Review status: **Needs adjudication**",
        "",
        "### 1. Runtime failure",
        "",
        f"- kind: `{item.failure_kind}`",
        f"- reason: `{item.failure_reason}`",
        f"- seed: `{item.seed}`",
        f"- stop status: `{item.stop_status}`",
        f"- invalid reason: {item.invalid_reason or 'none recorded'}",
        "",
        "### 2. Current active beat",
        "",
        f"- number: `{beat.get('number')}`",
        f"- start turn: `{beat.get('start_turn')}`",
        f"- delivered turns on active beat: `{beat.get('delivered_turns')}`",
        "",
        "### 3. Requirements still pending",
        "",
        f"- satisfied cumulative: `{beat.get('satisfied_cumulative_requirements')}`",
        f"- missing cumulative: `{missing}`",
        f"- pending local predicates: `{pending}`",
        "",
        "### 4. Relevant earlier transcript evidence",
        "",
        *_bullet_evidence(item, kind="transcript"),
        "",
        "### 5. Recovered rejected candidates",
        "",
        *_render_retries(item.retry_recovery),
        "",
        "### 6. Possible mechanisms",
        "",
        *([f"- `{name}`" for name in flags] or ["- none flagged"]),
        "",
        "### 7. Evidence suggesting a true missing requirement",
        "",
        (
            "- `possible_required_content_absent` fired; content was not found "
            "in the committed transcript or recovered candidates."
            if absent
            else "- none flagged."
        ),
        "",
        "### 8. Evidence suggesting a contract false positive",
        "",
        *([f"- `{name}`" for name in cfp_hints] or ["- none flagged."]),
        "",
        "Audit flags are evidence-retrieval heuristics, not semantic "
        "classifiers. Their presence or absence does not determine "
        "experimental inclusion.",
        "",
    ]


def _bullet_evidence(item: DialogueAudit, *, kind: str) -> list[str]:
    """Return markdown bullets for transcript-side mechanism evidence."""
    del kind
    bullets: list[str] = []
    for flag in item.possible_mechanisms:
        for row in flag.get("evidence") or []:
            if not isinstance(row, dict):
                continue
            if "earlier_turns" in row or "turns" in row:
                bullets.append(f"- `{flag['mechanism']}`: {row}")
    if not bullets:
        return ["- none recorded beyond the census fields above."]
    return bullets


def _render_retries(recovery: Mapping[str, Any]) -> list[str]:
    """Render recovered retry attempts, or why they are unavailable."""
    if not recovery.get("applicable"):
        return [
            "- not applicable: this failure class is not "
            "`invalid_candidate_retry_exhausted`."
        ]
    if recovery.get("status") == "unavailable":
        return [f"- unavailable: {recovery.get('note', 'no matching calls')}"]
    lines: list[str] = []
    for candidate in recovery.get("candidates") or []:
        lines.append(
            f"- attempt {candidate.get('attempt_index')}: "
            f"status={candidate.get('status')!r} "
            f"seed={candidate.get('seed')} "
            f"({candidate.get('seed_source')}) "
            f"invalid_reason={candidate.get('invalid_reason')!r}"
        )
        lines.append(f"  text: {candidate.get('message')!r}")
    return lines or ["- none recovered."]


def _audit_dialogue(
    log: DialogueLog,
    *,
    scenario: Scenario,
    fields: Sequence[UserDataField],
    run_id: str,
    calls: Sequence[LlmCallRecord],
) -> DialogueAudit:
    """Build generated evidence for one failed log."""
    progress = ScriptProgress(beats_of(scenario, fields))
    for record in log.records:
        progress.commit(record.user_message)
    diagnostics = _diagnostics(progress)
    retry = _retry_recovery(log, scenario, progress, calls)
    candidates = [
        UserReply.model_validate(
            {
                "message": row.get("message") or "",
                "answering_agent_question": False,
                "status": row.get("status") or "continue",
            }
        )
        if row.get("message") is not None and row.get("status") is not None
        else None
        for row in retry.get("candidates") or []
    ]
    recovered_texts = [
        str(row.get("message") or row.get("recorded_text") or "")
        for row in retry.get("candidates") or []
    ]
    mechanisms = _possible_mechanisms(
        log,
        progress=progress,
        diagnostics=diagnostics,
        recovered_texts=recovered_texts,
        recovered_replies=[item for item in candidates if item is not None],
        retry=retry,
    )
    metadata = dict(log.failure_metadata)
    return DialogueAudit(
        dialogue_id=log.dialogue_id,
        run_id=run_id,
        scenario_id=log.scenario_id,
        agent=log.agent,
        repetition=log.repetition,
        seed=log.seed,
        failure_kind=log.failure_kind,
        failure_reason=log.failure_reason,
        recorded_failure_metadata=metadata,
        active_beat=diagnostics,
        stop_status=log.stop_reason or log.termination_reason,
        invalid_reason=_invalid_reason(metadata),
        retry_recovery=retry,
        possible_mechanisms=mechanisms,
    )


def _diagnostics(progress: ScriptProgress) -> dict[str, Any] | None:
    """Recompute active-beat census from frozen ``ScriptProgress`` semantics."""
    snapshot = progress.active_beat_diagnostics()
    if snapshot is None:
        return None
    beat = progress.current
    assert beat is not None
    return {
        "number": snapshot["active_beat_index"],
        "start_turn": snapshot["beat_start_turn"],
        "delivered_turns": snapshot["delivered_turns_on_active_beat"],
        "satisfied_cumulative_requirements": dict(
            snapshot["satisfied_cumulative_requirements"]
        ),
        "missing_cumulative_requirements": dict(
            snapshot["missing_cumulative_requirements"]
        ),
        "pending_local_predicates": list(snapshot["pending_local_predicates"]),
        "text": beat.text,
    }


def _retry_recovery(
    log: DialogueLog,
    scenario: Scenario,
    progress: ScriptProgress,
    calls: Sequence[LlmCallRecord],
) -> dict[str, Any]:
    """Recover rejected simulated-user candidates when that failure class applies."""
    if log.failure_reason != "invalid_candidate_retry_exhausted":
        return {"applicable": False, "status": "not_applicable", "candidates": []}
    matched = _matching_user_calls(log, scenario, calls)
    if not matched:
        return {
            "applicable": True,
            "status": "unavailable",
            "candidates": [],
            "note": (
                "llm_calls.jsonl missing or no matching simulated_user calls; "
                "no candidate text was reconstructed"
            ),
        }
    candidates: list[dict[str, Any]] = []
    for index, call in enumerate(matched):
        row: dict[str, Any] = {
            "attempt_index": index,
            "seed": log.seed + index,
            "seed_source": SEED_SOURCE,
            "recorded_text": call.text,
        }
        try:
            reply = UserReply.model_validate_json(call.text)
        except (ValidationError, ValueError) as failure:
            row["parse_error"] = str(failure)
            candidates.append(row)
            continue
        row["message"] = reply.message
        row["status"] = reply.status
        row["invalid_reason"] = _invalid_candidate_reason(reply, progress)
        candidates.append(row)
    return {"applicable": True, "status": "recovered", "candidates": candidates}


def _matching_user_calls(
    log: DialogueLog,
    scenario: Scenario,
    calls: Sequence[LlmCallRecord],
) -> list[LlmCallRecord]:
    """Return simulated-user calls whose history matches this failed speak()."""
    expected = as_messages(
        "unused", transcript_from_records(log.records), speaking_as="user"
    )[1:]
    matched: list[LlmCallRecord] = []
    for call in calls:
        if call.caller != "simulated_user" or not call.messages:
            continue
        system = call.messages[0].get("content", "")
        if scenario.user_persona not in system or scenario.user_goal not in system:
            continue
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in call.messages[1:]
        ]
        if history != expected:
            continue
        matched.append(call)
    matched.sort(key=lambda item: (item.timestamp, item.prompt_hash))
    return matched


def _possible_mechanisms(
    log: DialogueLog,
    *,
    progress: ScriptProgress,
    diagnostics: dict[str, Any] | None,
    recovered_texts: Sequence[str],
    recovered_replies: Sequence[UserReply],
    retry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flag possible mechanisms. These are hints, never verdicts."""
    flags: list[dict[str, Any]] = []
    flags.extend(_future_beat_flags(log, progress=progress, diagnostics=diagnostics))
    flags.extend(_nonsticky_flags(log, progress, diagnostics))
    flags.extend(_case_mismatch_flags(log, diagnostics, recovered_texts))
    flags.extend(_denial_gap_flags(log, progress, recovered_texts))
    flags.extend(_absent_flags(log, diagnostics, recovered_texts, case_mismatch=flags))
    flags.extend(_semantic_completion_flags(log, recovered_replies, flags))
    flags.extend(_empty_stop_flags(recovered_replies, retry))
    if log.failure_reason == "invalid_candidate_retry_exhausted" and not _named(flags):
        flags.append(
            {
                "mechanism": "possible_other_contract_mismatch",
                "evidence": [
                    {
                        "invalid_reason": _invalid_reason(log.failure_metadata),
                    }
                ],
            }
        )
    return flags


def _future_beat_flags(
    log: DialogueLog,
    *,
    progress: ScriptProgress,
    diagnostics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Flag contract material that appeared off the beat the runtime is scoring."""
    if diagnostics is None:
        return []
    start = int(diagnostics["start_turn"])
    earlier = [record for record in log.records if record.turn < start]
    missing = diagnostics["missing_cumulative_requirements"]
    evidence: list[dict[str, Any]] = []
    for literal in missing.get("exact_strings") or []:
        turns = [record.turn for record in earlier if literal in record.user_message]
        if turns:
            evidence.append(
                _content_hit(
                    requirement=f"the exact string {literal!r}",
                    active_beat=diagnostics["number"],
                    turns=turns,
                    sent_to_agent=True,
                )
            )
    for keyword in missing.get("keywords") or []:
        turns = [
            record.turn for record in earlier if _carries(keyword, record.user_message)
        ]
        if turns:
            evidence.append(
                _content_hit(
                    requirement=f"the word {keyword!r}",
                    active_beat=diagnostics["number"],
                    turns=turns,
                    sent_to_agent=True,
                )
            )
    evidence.extend(_later_beat_content_on_open_beat(log, progress, diagnostics))
    if not evidence:
        return []
    return [{"mechanism": "possible_future_beat_content", "evidence": evidence}]


def _later_beat_content_on_open_beat(
    log: DialogueLog,
    progress: ScriptProgress,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flag later-beat contract material already present on an open earlier beat."""
    current = progress.current
    if current is None:
        return []
    evidence: list[dict[str, Any]] = []
    for beat in progress.beats:
        if beat.number <= current.number:
            continue
        for literal in beat.literals:
            turns = [
                record.turn for record in log.records if literal in record.user_message
            ]
            if turns:
                evidence.append(
                    _content_hit(
                        requirement=(
                            f"later beat {beat.number} exact string {literal!r}"
                        ),
                        active_beat=diagnostics["number"],
                        turns=turns,
                        sent_to_agent=True,
                    )
                )
        if beat.verbatim is not None:
            continue
        for keyword in beat.keywords:
            turns = [
                record.turn
                for record in log.records
                if _carries(keyword, record.user_message)
            ]
            if turns:
                evidence.append(
                    _content_hit(
                        requirement=f"later beat {beat.number} word {keyword!r}",
                        active_beat=diagnostics["number"],
                        turns=turns,
                        sent_to_agent=True,
                    )
                )
    return evidence


def _content_hit(
    *,
    requirement: str,
    active_beat: int,
    turns: Sequence[int],
    sent_to_agent: bool,
) -> dict[str, Any]:
    """One piece of earlier content the active beat did not credit."""
    return {
        "requirement": requirement,
        "active_beat": active_beat,
        "earlier_turns": list(turns),
        "sent_to_agent": sent_to_agent,
        "credited_to_active_beat": False,
    }


def _nonsticky_flags(
    log: DialogueLog,
    progress: ScriptProgress,
    diagnostics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Flag turn-local predicates seen earlier but pending on the last turn."""
    if diagnostics is None or not log.records:
        return []
    last = log.records[-1]
    pending = set(diagnostics["pending_local_predicates"])
    flags: list[dict[str, Any]] = []
    beat = progress.current
    if beat is not None and beat.asks and "a question" in pending:
        earlier = [
            record.turn
            for record in log.records[:-1]
            if QUESTION in record.user_message
        ]
        if QUESTION not in last.user_message and earlier:
            flags.append(
                {
                    "mechanism": "possible_nonsticky_question",
                    "active_beat": diagnostics["number"],
                    "pending_now": True,
                    "seen_earlier_on_turns": earlier,
                }
            )
    if beat is not None and beat.negates and "a denial" in pending:
        earlier = [
            record.turn for record in log.records[:-1] if _denies(record.user_message)
        ]
        if not _denies(last.user_message) and earlier:
            flags.append(
                {
                    "mechanism": "possible_nonsticky_denial",
                    "active_beat": diagnostics["number"],
                    "pending_now": True,
                    "seen_earlier_on_turns": earlier,
                }
            )
    return flags


def _case_mismatch_flags(
    log: DialogueLog,
    diagnostics: dict[str, Any] | None,
    recovered_texts: Sequence[str],
) -> list[dict[str, Any]]:
    """Flag missing exact strings that match only after case folding."""
    if diagnostics is None:
        return []
    committed = [record.user_message for record in log.records]
    corpus = [*committed, *recovered_texts]
    evidence: list[dict[str, Any]] = []
    for literal in (
        diagnostics["missing_cumulative_requirements"].get("exact_strings") or []
    ):
        search = _literal_search(literal, corpus)
        if search["exact_match"] or not search["case_insensitive_match"]:
            continue
        turns = [
            record.turn
            for record in log.records
            if literal.lower() in record.user_message.lower()
        ]
        sent = bool(turns)
        evidence.append(
            {
                "requirement": literal,
                "exact_match": False,
                "case_insensitive_match": True,
                "whitespace_normalized_match": search["whitespace_normalized_match"],
                "turns": turns,
                "sent_to_agent": sent,
            }
        )
    if not evidence:
        return []
    return [
        {
            "mechanism": "possible_exact_string_case_mismatch",
            "evidence": evidence,
        }
    ]


def _literal_search(literal: str, messages: Sequence[str]) -> dict[str, bool]:
    """Exact, case-insensitive, and whitespace-normalized substring search."""
    folded = literal.lower()
    normalized = _whitespace_normalized(literal)
    return {
        "exact_match": any(literal in message for message in messages),
        "case_insensitive_match": any(
            folded in message.lower() for message in messages
        ),
        "whitespace_normalized_match": any(
            normalized in _whitespace_normalized(message) for message in messages
        ),
    }


def _denial_gap_flags(
    log: DialogueLog,
    progress: ScriptProgress,
    recovered_texts: Sequence[str],
) -> list[dict[str, Any]]:
    """Flag pre-shipment wording the frozen detector does not treat as a denial."""
    beat = progress.current
    if beat is None or not beat.negates:
        return []
    texts = [record.user_message for record in log.records]
    texts.extend(recovered_texts)
    hits = [
        text for text in texts if _looks_like_pre_shipment(text) and not _denies(text)
    ]
    if not hits:
        return []
    return [
        {
            "mechanism": "possible_denial_detector_gap",
            "evidence": [{"text": text, "detector": False} for text in hits],
        }
    ]


def _looks_like_pre_shipment(text: str) -> bool:
    """Whether ``text`` uses ``before`` plus a frozen shipment word."""
    words = _words(text)
    return "before" in words and bool(set(words) & SHIPMENT_WORDS)


def _absent_flags(
    log: DialogueLog,
    diagnostics: dict[str, Any] | None,
    recovered_texts: Sequence[str],
    *,
    case_mismatch: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flag cumulative requirements with no matching evidence anywhere."""
    if diagnostics is None:
        return []
    messages = [record.user_message for record in log.records]
    messages.extend(recovered_texts)
    mismatched = {
        row["requirement"]
        for flag in case_mismatch
        if flag.get("mechanism") == "possible_exact_string_case_mismatch"
        for row in flag.get("evidence") or []
    }
    missing = diagnostics["missing_cumulative_requirements"]
    evidence: list[dict[str, Any]] = []
    for literal in missing.get("exact_strings") or []:
        if literal in mismatched:
            continue
        search = _literal_search(literal, messages)
        if search["exact_match"] or search["case_insensitive_match"]:
            continue
        if search["whitespace_normalized_match"]:
            continue
        if any(
            literal.lower() in _whitespace_normalized(item).lower() for item in messages
        ):
            continue
        evidence.append(
            {
                "requirement": f"the exact string {literal!r}",
                "searched": ["committed_transcript", "rejected_candidates"],
            }
        )
    for keyword in missing.get("keywords") or []:
        if any(_carries(keyword, message) for message in messages):
            continue
        evidence.append(
            {
                "requirement": f"the word {keyword!r}",
                "searched": ["committed_transcript", "rejected_candidates"],
            }
        )
    if not evidence:
        return []
    return [{"mechanism": "possible_required_content_absent", "evidence": evidence}]


def _semantic_completion_flags(
    log: DialogueLog,
    recovered_replies: Sequence[UserReply],
    already: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flag structural hints that the conversation may already have ended."""
    hints: list[str] = []
    if any(reply.status in ("goal_reached", "gave_up") for reply in recovered_replies):
        hints.append("a recovered candidate reported goal_reached or gave_up")
    if log.goal_reached_seen:
        hints.append("goal_reached_seen is true on the log")
    if log.stop_reason in ("goal_reached", "gave_up") or log.termination_reason in (
        "goal_reached",
        "gave_up",
    ):
        hints.append(
            f"recorded stop status is {log.stop_reason or log.termination_reason}"
        )
    last = log.records[-1].user_message if log.records else ""
    words = set(_words(last))
    if words & (CLOSING_WORDS | THANKS_WORDS):
        hints.append("last committed user turn uses thanks/goodbye wording")
    reason = _invalid_reason(log.failure_metadata) or ""
    if "goal_reached" in reason or "gave_up" in reason:
        hints.append("invalid_reason names a closing status")
    if any(flag.get("mechanism") == "possible_future_beat_content" for flag in already):
        hints.append("later-beat content was already delivered and the agent responded")
    if not hints:
        return []
    return [
        {
            "mechanism": "possible_semantic_completion_before_ledger_completion",
            "evidence": hints,
        }
    ]


def _empty_stop_flags(
    recovered_replies: Sequence[UserReply], retry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Flag empty or agent_ended recovered candidates."""
    hits = [
        reply
        for reply in recovered_replies
        if not reply.message.strip() or reply.status == "agent_ended"
    ]
    if not hits:
        return []
    return [
        {
            "mechanism": "possible_empty_stop_candidate",
            "evidence": [
                {"status": reply.status, "empty": not reply.message.strip()}
                for reply in hits
            ],
            "retry_status": retry.get("status"),
        }
    ]


def _named(flags: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return mechanism names already flagged."""
    return {str(flag.get("mechanism")) for flag in flags}


def _invalid_reason(metadata: Mapping[str, Any]) -> str | None:
    """Return the recorded invalid-candidate reason, if any."""
    value = metadata.get("invalid_reason")
    if value is None or value == "":
        return None
    return str(value)


def _whitespace_normalized(text: str) -> str:
    """Collapse runs of whitespace."""
    return " ".join(text.split())


def _load_manifest(run_dir: Path) -> Manifest:
    """Load ``manifest.json`` from ``run_dir``."""
    path = run_dir / "manifest.json"
    if not path.exists():
        raise AuditError(
            f"{path} is missing. The audit reads a sim run directory, never metrics.csv"
        )
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def _load_failed_logs(run_dir: Path) -> list[DialogueLog]:
    """Load every ``status=failed`` dialogue, in file-name order."""
    directory = run_dir / "dialogues"
    if not directory.exists():
        raise AuditError(
            f"{directory} is missing. The census is built from dialogue JSONL, "
            "never from judge output"
        )
    logs: list[DialogueLog] = []
    for path in sorted(directory.glob("*.jsonl")):
        log = DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        if log.status == "failed":
            logs.append(log)
    return logs


def _count_ok(run_dir: Path) -> int:
    """Count ``status=ok`` logs so the census can say what it skipped."""
    directory = run_dir / "dialogues"
    return sum(
        1
        for path in directory.glob("*.jsonl")
        if DialogueLog.model_validate_json(path.read_text(encoding="utf-8")).status
        == "ok"
    )


def _load_calls(path: Path) -> list[LlmCallRecord]:
    """Load ``llm_calls.jsonl``; a missing file is an empty list."""
    if not path.exists():
        return []
    return [
        LlmCallRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _scenario_of(
    log: DialogueLog, scenarios: Mapping[str, Scenario], scenarios_dir: str
) -> Scenario:
    """Return the frozen scenario this dialogue was given."""
    scenario = scenarios.get(log.scenario_id)
    if scenario is None:
        raise AuditError(
            f"{log.dialogue_id} plays {log.scenario_id}, which is not in "
            f"{scenarios_dir}"
        )
    return scenario


def main(argv: Sequence[str] | None = None) -> int:
    """Audit failed dialogues of ``--run`` and write generated evidence."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit failed dialogues for possible contract false-positive "
            "mechanisms. Writes generated evidence, never an allowlist."
        )
    )
    parser.add_argument("--run", type=Path, required=True, help="experiment directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=("directory for audit.json and audit.md (default: <run>/adjudication)"),
    )
    args = parser.parse_args(argv)
    try:
        report = audit_run(args.run)
        out_dir = args.out if args.out is not None else default_out_dir(args.run)
        write_reports(report, out_dir)
    except AuditError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    print(
        f"sim: wrote {out_dir / 'audit.json'} and {out_dir / 'audit.md'} "
        f"({report.run['n_failed']} failed dialogue(s)). Needs adjudication. "
        "This is generated evidence, not an inclusion decision.",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
