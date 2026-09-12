"""Check that every dialogue of a run delivered its script beats in order (T-11).

A scenario's script is a mandatory ordered plan. The simulated user enforces it
while a dialogue runs; this script audits the result afterwards, from the
dialogue JSONL alone, and is the gate a pilot has to pass before anything is
evaluated. It reads no judge output: a dialogue that lost a beat is an instrument
failure, and whether it also scored well is beside the point.

Every check is deterministic. Which beats were delivered and in what order is
read from the ``user_beat`` the runtime recorded on each turn; whether the
message on that turn actually met the beat's contract is checked again here,
from the scenario, so a regression in the runtime shows up as a dialogue whose
recorded beats its own messages do not support.

Usage::

    uv run python scripts/script_adherence.py --run runs/exp_pilot2
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.kb import DEFAULT_KB_DIR, UserDataField, load_kb
from sim.schemas import DialogueLog, Manifest, Scenario, load_scenarios
from sim.script import Beat, beats_of

DEFAULT_RUN_DIR = Path("runs/exp_pilot2")


class AdherenceError(RuntimeError):
    """The run cannot be checked; the message says what is missing."""


@dataclass(frozen=True)
class Report:
    """What one dialogue did with the script it was given."""

    dialogue_id: str
    scenario_id: str
    agent: str
    repetition: int
    status: str
    stop_reason: str | None
    n_beats: int
    delivered: tuple[int, ...]
    missing: tuple[int, ...]
    out_of_order: bool
    unsatisfied: tuple[str, ...]
    canary_delivered: bool | None
    injection_delivered: bool | None

    @property
    def ok(self) -> bool:
        """Whether every mandatory beat arrived, in order and intact."""
        return (
            self.status == "ok"
            and not self.missing
            and not self.out_of_order
            and not self.unsatisfied
            and self.canary_delivered is not False
            and self.injection_delivered is not False
        )

    @property
    def faults(self) -> tuple[str, ...]:
        """Everything wrong with this dialogue, for the report line."""
        faults: list[str] = []
        if self.status != "ok":
            faults.append(f"dialogue {self.status}")
        if self.missing:
            faults.append(f"missing beats {list(self.missing)}")
        if self.out_of_order:
            faults.append(f"out of order {list(self.delivered)}")
        if self.unsatisfied:
            faults.append("; ".join(self.unsatisfied))
        if self.canary_delivered is False:
            faults.append("canary never reached the agent")
        if self.injection_delivered is False:
            faults.append("the injection beat was never delivered as written")
        return tuple(faults)


def check_dialogue(
    log: DialogueLog, scenario: Scenario, fields: Sequence[UserDataField]
) -> Report:
    """Report what ``log`` delivered of ``scenario``'s script."""
    beats = beats_of(scenario, fields)
    delivered = tuple(
        record.user_beat for record in log.records if record.user_beat is not None
    )
    by_beat = {
        record.user_beat: record.user_message
        for record in log.records
        if record.user_beat is not None
    }
    unsatisfied = tuple(
        f"beat {beat.number} needs {fault}"
        for beat in beats
        if beat.number in by_beat
        for fault in beat.faults_in(by_beat[beat.number])
    )
    return Report(
        dialogue_id=log.dialogue_id,
        scenario_id=log.scenario_id,
        agent=log.agent,
        repetition=log.repetition,
        status=log.status,
        stop_reason=log.stop_reason,
        n_beats=len(beats),
        delivered=delivered,
        missing=tuple(
            beat.number for beat in beats if beat.number not in set(delivered)
        ),
        out_of_order=list(delivered) != sorted(set(delivered)),
        unsatisfied=unsatisfied,
        canary_delivered=_canary_delivered(log, scenario),
        injection_delivered=_injection_delivered(beats, by_beat, scenario),
    )


def _injection_delivered(
    beats: Sequence[Beat], by_beat: dict[int, str], scenario: Scenario
) -> bool | None:
    """Whether the beat that plants the canary was delivered as written.

    The token reaching the agent is not the same question: ``Please proceed with
    VN6-HARBOUR-1188`` carries it and asks for nothing, and the rerun of
    2026-09-11 produced exactly that. Here the whole beat has to have met its
    contract, so the attack the scenario planted is what the agent answered.
    """
    if scenario.canary is None:
        return None
    return all(
        beat.number in by_beat and beat.satisfied_by(by_beat[beat.number])
        for beat in beats
        if scenario.canary in beat.literals
    )


def _canary_delivered(log: DialogueLog, scenario: Scenario) -> bool | None:
    """Whether the planted token reached the agent, or ``None`` without one.

    Checked over every customer turn and not only the beat that carries it: the
    question the metric of T-13 rests on is whether the agent ever saw the token.
    """
    if scenario.canary is None:
        return None
    return any(scenario.canary in record.user_message for record in log.records)


def check_run(run_dir: Path) -> list[Report]:
    """Check every dialogue of ``run_dir``, in file-name order."""
    manifest = _load_manifest(run_dir)
    kb = load_kb(DEFAULT_KB_DIR)
    fsm = load_fsm(DEFAULT_FSM_DIR)
    scenarios = {
        scenario.id: scenario
        for scenario in load_scenarios(Path(manifest.scenarios_dir), kb=kb, fsm=fsm)
    }
    directory = run_dir / "dialogues"
    if not directory.exists():
        raise AdherenceError(
            f"{directory} is missing. Script adherence is read from the dialogue "
            "JSONL of T-14a, never from an evaluated CSV"
        )
    reports: list[Report] = []
    for path in sorted(directory.glob("*.jsonl")):
        log = DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        scenario = scenarios.get(log.scenario_id)
        if scenario is None:
            raise AdherenceError(
                f"{path.name} plays {log.scenario_id}, which is not in "
                f"{manifest.scenarios_dir}. The script checked has to be the "
                "script the dialogue was given"
            )
        reports.append(check_dialogue(log, scenario, kb.user_data_fields))
    return reports


def render(reports: Sequence[Report]) -> str:
    """Render one line per dialogue, then the verdict."""
    lines = [
        f"{'dialogue':38} {'stop':12} {'beats':10} verdict",
        "-" * 88,
    ]
    for report in sorted(reports, key=lambda item: item.dialogue_id):
        beats = f"{len(report.delivered)}/{report.n_beats}"
        verdict = "ok" if report.ok else "; ".join(report.faults)
        lines.append(
            f"{report.dialogue_id:38} {report.stop_reason!s:12} {beats:10} {verdict}"
        )
    passed = sum(1 for report in reports if report.ok)
    canary = [report for report in reports if report.canary_delivered is not None]
    planted = [report for report in reports if report.injection_delivered is not None]
    lines.extend(
        (
            "-" * 88,
            f"{passed}/{len(reports)} dialogues delivered every mandatory beat "
            "in order",
            f"{sum(1 for report in canary if report.canary_delivered)}/{len(canary)} "
            "canary tokens reached the agent word for word",
            f"{sum(1 for report in planted if report.injection_delivered)}/"
            f"{len(planted)} injection beats were delivered as the script wrote "
            "them",
        )
    )
    return "\n".join(lines)


def _load_manifest(run_dir: Path) -> Manifest:
    """Load the run manifest, which names the scenarios the dialogues were given."""
    path = run_dir / "manifest.json"
    if not path.exists():
        raise AdherenceError(
            f"{path} is missing. The check loads the scenario files the run "
            "recorded, so a script cannot be compared against another dataset"
        )
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Check the run in ``argv`` and return 0 only when every dialogue passed."""
    parser = argparse.ArgumentParser(
        description="Check that every dialogue delivered its script beats in order."
    )
    parser.add_argument(
        "--run", type=Path, default=DEFAULT_RUN_DIR, help="experiment directory"
    )
    args = parser.parse_args(argv)
    try:
        reports = check_run(args.run)
    except AdherenceError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    print(render(reports))
    return 0 if all(report.ok for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
