"""Replay the user-event classifier on frozen logs or T-08 gold phrases.

The hybrid detector is unchanged: rules first, LLM fallback, same schema. This
script never re-simulates agents, never writes into ``runs/pilot_v2``, and never
edits ``configs/models.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from sim.config import (
    ConfigError,
    ModelsConfig,
    Role,
    load_models_config,
    with_role_model,
)
from sim.engine import FsmEngine
from sim.events import CLASSIFIER, INTENT_EVENT, detect_user_event
from sim.fsm import DEFAULT_FSM_DIR, FsmSpec, load_fsm
from sim.io import atomic_write
from sim.kb import DEFAULT_KB_DIR, KnowledgeBase, load_kb
from sim.llm import LLM_CALLS_LOG, Chat, LlmCallRecord, LlmClient, LlmResponse
from sim.prompts import DEFAULT_PROMPTS_DIR
from sim.schemas import DialogueLog, Turn, render_transcript, transcript_from_records

FROZEN_PILOT_V2 = Path("runs/pilot_v2")
EXPECTED_PILOT_TURNS = 39
EXPECTED_PILOT_RULE = 11
EXPECTED_PILOT_LLM = 28
LABELED_EVENTS = Path("tests/fixtures/user_events.jsonl")
TURNS_CSV = "classifier_turns.csv"
SUMMARY_JSON = "classifier_summary.json"


class ReplayError(RuntimeError):
    """The classifier replay cannot run; the message says why."""


class GoldUtterance(Protocol):
    """The T-08 gold fields ``score_fixtures`` reads."""

    state: str
    text: str
    event: str
    intent: str | None


@dataclass(frozen=True)
class ReplayTurn:
    """One FSM turn replayed against the logged 4B event."""

    scenario_id: str
    repetition: int
    turn: int
    via: Literal["rule", "llm"]
    logged_event: str
    logged_intent: str | None
    replayed_event: str
    replayed_intent: str | None
    event_match: bool


@dataclass(frozen=True)
class PilotReplayReport:
    """Agreement of a replayed classifier with the frozen Pilot v2 log."""

    n_turns: int
    n_rule: int
    n_llm: int
    n_llm_event_agree: int
    n_llm_event_change: int
    n_intent_classified: int
    n_intent_change: int
    turns: tuple[ReplayTurn, ...]


@dataclass(frozen=True)
class FixtureReport:
    """Accuracy of the classifier on T-08 gold phrases."""

    n: int
    n_correct: int
    accuracy: float
    misses: tuple[str, ...]


class CountingChat:
    """Wrap a ``Chat`` and count how many times it is called."""

    def __init__(self, inner: Chat) -> None:
        self._inner = inner
        self.n_calls = 0

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        role: Role,
        caller: str,
        schema: type[BaseModel] | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> LlmResponse:
        """Forward the call after incrementing the counter."""
        self.n_calls += 1
        return self._inner.chat(
            messages,
            role=role,
            caller=caller,
            schema=schema,
            seed=seed,
            num_predict=num_predict,
        )


def replay_pilot(
    source_run: Path,
    target_run: Path,
    *,
    llm: Chat | None = None,
    config: ModelsConfig | None = None,
    model: str | None = None,
    fsm: FsmSpec | None = None,
    kb: KnowledgeBase | None = None,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> PilotReplayReport:
    """Replay ``detect_user_event`` on frozen FSM turns and write sidecar CSVs."""
    source = source_run.resolve()
    target = target_run.resolve()
    if source == target:
        raise ReplayError(
            f"--source and --target resolve to the same path ({source}). "
            "Use a sidecar target so the frozen run is not overwritten"
        )
    frozen = FROZEN_PILOT_V2.resolve()
    if target == frozen:
        raise ReplayError(
            f"--target resolves to {frozen}, the frozen Pilot v2 run. "
            "Use a sidecar such as runs/pilot_v2_classifier_9b"
        )
    spec = fsm if fsm is not None else load_fsm(DEFAULT_FSM_DIR)
    knowledge = kb if kb is not None else load_kb(DEFAULT_KB_DIR)
    model_config = _classifier_config(config or load_models_config(), model)
    client = llm or LlmClient(
        model_config,
        cache_dir=target / "cache",
        log_path=target / LLM_CALLS_LOG,
    )
    turns: list[ReplayTurn] = []
    for log in _load_fsm_logs(source):
        turns.extend(
            _replay_log(
                log,
                llm=client,
                spec=spec,
                kb=knowledge,
                prompts_dir=prompts_dir,
            )
        )
    report = _pilot_report(turns)
    _check_pilot_split(source, report)
    _write_pilot_artifacts(target, report)
    return report


def score_fixtures(
    cases: Sequence[GoldUtterance],
    *,
    llm: Chat,
    fsm: FsmSpec,
    kb: KnowledgeBase,
    seed: int = 42,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> FixtureReport:
    """Score gold utterances the same way the T-08 integration test does."""
    misses: list[str] = []
    n_correct = 0
    for case in cases:
        engine = FsmEngine(fsm, kb=kb, llm=llm, prompts_dir=prompts_dir, seed=seed)
        engine.park(case.state)
        walk = engine.step(case.text, turn=1)
        matched = walk[0].event == case.event
        if matched and case.event == INTENT_EVENT:
            matched = engine.intent == case.intent
        if matched:
            n_correct += 1
            continue
        misses.append(
            f"{case.state}: {case.text!r} → {walk[0].event}"
            f"{f'/{engine.intent}' if engine.intent else ''} "
            f"(want {case.event}"
            f"{f'/{case.intent}' if case.intent else ''})"
        )
    n = len(cases)
    if n == 0:
        raise ReplayError("fixture scoring needs at least one gold utterance")
    return FixtureReport(
        n=n,
        n_correct=n_correct,
        accuracy=n_correct / n,
        misses=tuple(misses),
    )


def _classifier_config(config: ModelsConfig, model: str | None) -> ModelsConfig:
    """Return ``config`` with an optional classifier model override."""
    if model is None:
        return config
    try:
        return with_role_model(config, "classifier", model)
    except ConfigError as error:
        raise ReplayError(str(error)) from error


def _load_fsm_logs(run_dir: Path) -> list[DialogueLog]:
    """Load ok FSM dialogue logs from ``run_dir/dialogues``."""
    dialogues = run_dir / "dialogues"
    if not dialogues.exists():
        raise ReplayError(f"{dialogues} is missing")
    logs = [
        DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(dialogues.glob("*.jsonl"))
    ]
    return [log for log in logs if log.status == "ok" and log.agent == "fsm"]


def _replay_log(
    log: DialogueLog,
    *,
    llm: Chat,
    spec: FsmSpec,
    kb: KnowledgeBase,
    prompts_dir: Path,
) -> list[ReplayTurn]:
    """Replay every turn of one frozen FSM dialogue."""
    rows: list[ReplayTurn] = []
    for index, record in enumerate(log.records):
        if record.state_before is None or record.event is None:
            raise ReplayError(
                f"{log.scenario_id} fsm rep {log.repetition} turn {record.turn} "
                "is missing state_before or event"
            )
        prior = log.records[index - 1] if index else None
        collected = dict(prior.collected) if prior is not None else {}
        intent = prior.intent if prior is not None else None
        history = [
            *transcript_from_records(log.records[:index]),
            Turn(speaker="user", text=record.user_message),
        ]
        counter = CountingChat(llm)
        before = counter.n_calls
        detection = detect_user_event(
            record.user_message,
            state=record.state_before,
            spec=spec,
            collected=collected,
            fields=kb.user_data_fields,
            intents=kb.intents(),
            intent=intent,
            llm=counter,
            transcript=render_transcript(history),
            prompts_dir=prompts_dir,
            seed=log.seed,
        )
        via: Literal["rule", "llm"] = "llm" if counter.n_calls > before else "rule"
        event_match = detection.event == record.event
        if via == "rule" and not event_match:
            raise ReplayError(
                f"rule-fired turn {log.scenario_id} rep {log.repetition} "
                f"turn {record.turn} replayed {detection.event!r} but the log "
                f"has {record.event!r}"
            )
        rows.append(
            ReplayTurn(
                scenario_id=log.scenario_id,
                repetition=log.repetition,
                turn=record.turn,
                via=via,
                logged_event=record.event,
                logged_intent=record.intent,
                replayed_event=detection.event,
                replayed_intent=detection.intent,
                event_match=event_match,
            )
        )
    return rows


def _pilot_report(turns: Sequence[ReplayTurn]) -> PilotReplayReport:
    """Aggregate rule/LLM denominators and 9B-vs-log agreement."""
    if not turns:
        raise ReplayError("no FSM turns to replay")
    n_rule = sum(row.via == "rule" for row in turns)
    llm_rows = [row for row in turns if row.via == "llm"]
    n_llm = len(llm_rows)
    n_llm_event_agree = sum(row.event_match for row in llm_rows)
    intent_rows = [row for row in llm_rows if row.logged_event == INTENT_EVENT]
    n_intent_change = sum(
        row.replayed_intent != row.logged_intent for row in intent_rows
    )
    return PilotReplayReport(
        n_turns=len(turns),
        n_rule=n_rule,
        n_llm=n_llm,
        n_llm_event_agree=n_llm_event_agree,
        n_llm_event_change=n_llm - n_llm_event_agree,
        n_intent_classified=len(intent_rows),
        n_intent_change=n_intent_change,
        turns=tuple(turns),
    )


def _check_pilot_split(source: Path, report: PilotReplayReport) -> None:
    """Fail if a frozen Pilot v2 replay does not recover the 11/28 split."""
    if source.resolve() != FROZEN_PILOT_V2.resolve():
        return
    if (
        report.n_turns,
        report.n_rule,
        report.n_llm,
    ) != (EXPECTED_PILOT_TURNS, EXPECTED_PILOT_RULE, EXPECTED_PILOT_LLM):
        raise ReplayError(
            "Pilot v2 replay split is "
            f"rule={report.n_rule}/{report.n_turns} "
            f"llm={report.n_llm}/{report.n_turns}; "
            f"expected rule={EXPECTED_PILOT_RULE}/{EXPECTED_PILOT_TURNS} "
            f"llm={EXPECTED_PILOT_LLM}/{EXPECTED_PILOT_TURNS}"
        )
    calls_path = source / LLM_CALLS_LOG
    if not calls_path.exists():
        raise ReplayError(f"{calls_path} is missing; cannot verify classifier n=28")
    n_logged = sum(
        1
        for line in calls_path.read_text(encoding="utf-8").splitlines()
        if line and LlmCallRecord.model_validate_json(line).caller == CLASSIFIER
    )
    if n_logged != report.n_llm:
        raise ReplayError(
            f"replay LLM-fallback turns={report.n_llm} but {calls_path} has "
            f"{n_logged} classifier calls"
        )


def _write_pilot_artifacts(target: Path, report: PilotReplayReport) -> None:
    """Write turn CSV and a JSON summary into the sidecar directory."""
    target.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "scenario_id",
        "repetition",
        "turn",
        "via",
        "logged_event",
        "logged_intent",
        "replayed_event",
        "replayed_intent",
        "event_match",
    )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for row in report.turns:
        writer.writerow(
            {
                "scenario_id": row.scenario_id,
                "repetition": str(row.repetition),
                "turn": str(row.turn),
                "via": row.via,
                "logged_event": row.logged_event,
                "logged_intent": row.logged_intent or "",
                "replayed_event": row.replayed_event,
                "replayed_intent": row.replayed_intent or "",
                "event_match": str(row.event_match),
            }
        )
    atomic_write(target / TURNS_CSV, buffer.getvalue())
    atomic_write(
        target / SUMMARY_JSON,
        json.dumps(
            {
                "n_turns": report.n_turns,
                "n_rule": report.n_rule,
                "n_llm": report.n_llm,
                "n_llm_event_agree": report.n_llm_event_agree,
                "n_llm_event_change": report.n_llm_event_change,
                "n_intent_classified": report.n_intent_classified,
                "n_intent_change": report.n_intent_change,
            },
            indent=2,
        )
        + "\n",
    )


def render_pilot(report: PilotReplayReport) -> str:
    """Render Pilot v2 classifier replay metrics as plain text."""
    return "\n".join(
        (
            f"turns={report.n_turns}",
            f"rule-fired={report.n_rule}/{report.n_turns}",
            f"LLM-fallback={report.n_llm}/{report.n_turns}",
            (f"llm_event_agreement={report.n_llm_event_agree}/{report.n_llm}"),
            f"llm_event_changes={report.n_llm_event_change}",
            (f"intent_classified_logged={report.n_intent_classified}"),
            f"intent_changes={report.n_intent_change}",
        )
    )


def render_fixtures(report: FixtureReport) -> str:
    """Render T-08 fixture accuracy as plain text."""
    lines = [
        f"n={report.n}",
        f"correct={report.n_correct}/{report.n}",
        f"accuracy={report.accuracy:.3f}",
    ]
    lines.extend(report.misses)
    return "\n".join(lines)


def _load_gold_utterances(path: Path) -> list[_GoldRow]:
    """Load T-08 gold phrases from ``path``."""
    if not path.exists():
        raise ReplayError(f"{path} is missing")
    rows: list[_GoldRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            _GoldRow(
                state=payload["state"],
                text=payload["text"],
                event=payload["event"],
                intent=payload.get("intent"),
            )
        )
    return rows


@dataclass(frozen=True)
class _GoldRow:
    """One T-08 gold utterance loaded from JSONL."""

    state: str
    text: str
    event: str
    intent: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    """Replay the classifier on a frozen run or on T-08 fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot", help="replay FSM turns from a frozen run")
    pilot.add_argument("--source", type=Path, default=Path("runs/pilot_v2"))
    pilot.add_argument(
        "--target", type=Path, default=Path("runs/pilot_v2_classifier_9b")
    )
    pilot.add_argument("--model", default="qwen3.5:9b")
    fixtures = sub.add_parser("fixtures", help="score tests/fixtures/user_events.jsonl")
    fixtures.add_argument("--model", required=True)
    fixtures.add_argument("--out", type=Path, default=None)
    fixtures.add_argument("--gold", type=Path, default=LABELED_EVENTS)
    args = parser.parse_args(argv)
    try:
        if args.command == "pilot":
            report = replay_pilot(args.source, args.target, model=args.model)
            print(render_pilot(report))
            return 0
        if args.command == "fixtures":
            config = _classifier_config(load_models_config(), args.model)
            out_dir = args.out or Path("results/judge_validation/pilot_v2_qwen9b")
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = args.model.replace(":", "_")
            run_dir = Path("runs") / f"pilot_v2_classifier_fixtures_{stem}"
            run_dir.mkdir(parents=True, exist_ok=True)
            client = LlmClient(
                config,
                cache_dir=run_dir / "cache",
                log_path=run_dir / LLM_CALLS_LOG,
            )
            report = score_fixtures(
                _load_gold_utterances(args.gold),
                llm=client,
                fsm=load_fsm(DEFAULT_FSM_DIR),
                kb=load_kb(DEFAULT_KB_DIR),
            )
            print(render_fixtures(report))
            stem = "classifier_fixtures_" + args.model.replace(":", "_")
            atomic_write(
                out_dir / f"{stem}.json",
                json.dumps(
                    {
                        "model": args.model,
                        "n": report.n,
                        "n_correct": report.n_correct,
                        "accuracy": report.accuracy,
                        "misses": list(report.misses),
                    },
                    indent=2,
                )
                + "\n",
            )
            return 0
        raise ReplayError(f"unknown command {args.command}")
    except ReplayError as error:
        print(f"sim: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
