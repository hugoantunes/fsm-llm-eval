"""Draw the stratified sample of agent responses the judge is validated on (T-16).

The frame is every agent turn of every ok dialogue in a ``sim run`` directory,
built from the dialogue logs alone: no judge output, no ``metrics.csv``, no
``llm_calls.jsonl``. Nothing here can be steered by a score, which is the point
of hand-annotating the sample before the numbers are read.

The draw is stratified by (agent, scenario) and spread over the dialogues of
each stratum by round-robin, so no dialogue carries the stratum on its own. It
is reproducible from ``--seed`` alone: every random choice draws from a
generator seeded by the seed plus the names of the stratum it decides, so the
same seed over the same dialogues selects the same responses.

Four files are written, none of which shows the judge's labels:

``sample.json``
    the metadata, the selected responses with their full identity, and the
    blind dialogue IDs the dialogue sheet is keyed by
``response_annotations.csv``
    what is annotated on one response: 30 rows, keyed by annotation ID only
``dialogue_annotations.csv``
    what the judge answers over a whole dialogue: one row per distinct dialogue
    the sample touches, keyed by blind dialogue ID
``packet.md``
    the evidence, in annotation order: the fact catalogue the judge grades
    against, the scenario's answer key and the transcript, with the response to
    annotate marked

The two sheets are split because the units differ: ``claim_support`` is a
property of one response, while the judge answers ``accuracy`` and
``task_completed`` over the whole dialogue (call 2 of T-12). Repeating those two
on response rows would weight a dialogue by how many of its turns were drawn.

Usage::

    uv run python scripts/judge_validation_sample.py --run runs/<exp_id>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.io import atomic_write
from sim.judge import needle_fact, redact_canary
from sim.kb import DEFAULT_KB_DIR, KnowledgeBase, load_kb, render_facts
from sim.schemas import (
    DialogueLog,
    Manifest,
    Scenario,
    load_scenarios,
    render_script,
)

#: Bump when the draw itself changes: a sample drawn under another rule is not
#: the same sample, even under the same seed.
SAMPLER_VERSION = 1

#: The date of T-16, so the seed of the thesis is a number with a reason.
DEFAULT_SEED = 20260911
DEFAULT_PER_AGENT = 15
DEFAULT_OUT_DIR = Path("results/judge_validation")
AGENTS = ("baseline", "fsm")

SAMPLE_JSON = "sample.json"
RESPONSE_CSV = "response_annotations.csv"
DIALOGUE_CSV = "dialogue_annotations.csv"
PACKET_MD = "packet.md"

#: What is decided on the marked response alone. No judge output and no agent:
#: the columns are the human's labels and nothing else.
RESPONSE_FIELDS = ("annotation_id", "fact_ids_stated", "claim_support", "notes")

#: What the judge answers over the whole dialogue (call 2 of T-12), annotated
#: once per dialogue so agreement is computed over dialogues, not over turns.
DIALOGUE_FIELDS = ("dialogue_annotation_id", "accuracy", "task_completed", "notes")

ELIGIBILITY = (
    "every agent turn of every dialogue whose status is ok and whose reply is "
    "not blank; failed dialogues and empty replies cannot be annotated"
)


class SamplingError(RuntimeError):
    """The sample cannot be drawn; the message says what is missing."""


@dataclass(frozen=True)
class Response:
    """One agent turn of the frame: a unit the annotator can grade."""

    dialogue_id: str
    scenario_id: str
    agent: str
    repetition: int
    turn: int
    text: str

    @property
    def unit_id(self) -> str:
        """The dialogue and the turn, which name this response in the run."""
        return f"{self.dialogue_id}#t{self.turn}"


class SampledResponse(BaseModel):
    """One drawn response: its blind annotation ID and its identity in the run."""

    model_config = ConfigDict(extra="forbid")

    annotation_id: str
    unit_id: str
    dialogue_id: str
    scenario_id: str
    agent: str
    repetition: int
    turn: int


class SampledDialogue(BaseModel):
    """One dialogue the sample touches, under the ID the annotator sees."""

    model_config = ConfigDict(extra="forbid")

    dialogue_annotation_id: str
    dialogue_id: str
    scenario_id: str
    agent: str
    repetition: int
    n_sampled_responses: int


class Stratum(BaseModel):
    """What one (agent, scenario) cell held and how much of it was drawn."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    scenario_id: str
    n_frame: int
    quota: int
    per_dialogue: dict[str, int]


class Sample(BaseModel):
    """The drawn sample and everything needed to draw it again."""

    model_config = ConfigDict(extra="forbid")

    sampler_version: int
    seed: int
    per_agent: int
    agents: list[str]
    run_dir: str
    exp_id: str
    dataset_hash: str
    fsm_hash: str
    eligibility: str
    frame_hash: str
    n_frame: int
    n_frame_by_agent: dict[str, int]
    n_dialogues: int
    max_per_dialogue: int
    strata: list[Stratum]
    responses: list[SampledResponse]
    dialogues: list[SampledDialogue]

    def dialogue_annotation_id(self, dialogue_id: str) -> str:
        """Return the blind ID the dialogue sheet keys ``dialogue_id`` by."""
        for dialogue in self.dialogues:
            if dialogue.dialogue_id == dialogue_id:
                return dialogue.dialogue_annotation_id
        raise SamplingError(
            f"{dialogue_id} holds a sampled response but has no dialogue "
            "annotation ID. Every drawn dialogue is annotated exactly once"
        )


def load_dialogues(run_dir: Path) -> dict[str, DialogueLog]:
    """Load every dialogue log of ``run_dir``, keyed by dialogue ID."""
    directory = run_dir / "dialogues"
    if not directory.exists():
        raise SamplingError(
            f"{directory} is missing. The frame is built from the dialogue JSONL "
            "of T-14a, never from an evaluated CSV"
        )
    logs = [
        DialogueLog.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.jsonl"))
    ]
    return {log.dialogue_id: log for log in logs}


def build_frame(dialogues: Mapping[str, DialogueLog]) -> list[Response]:
    """Return every eligible agent response, in dialogue and turn order."""
    frame = [
        Response(
            dialogue_id=name,
            scenario_id=log.scenario_id,
            agent=log.agent,
            repetition=log.repetition,
            turn=record.turn,
            text=record.agent_reply,
        )
        for name, log in sorted(dialogues.items())
        if log.status == "ok"
        for record in log.records
        if record.agent_reply.strip()
    ]
    if not frame:
        raise SamplingError(
            "no dialogue holds an annotatable response: every log either failed "
            "or answered with blank text"
        )
    return frame


def draw_sample(
    run_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    per_agent: int = DEFAULT_PER_AGENT,
    agents: Sequence[str] = AGENTS,
) -> Sample:
    """Draw ``per_agent`` responses per agent from the frame of ``run_dir``."""
    manifest = _load_manifest(run_dir)
    dialogues = load_dialogues(run_dir)
    frame = build_frame(dialogues)
    picks, strata = _select(frame, seed=seed, per_agent=per_agent, agents=agents)
    order = random.Random(f"{seed}|order")
    shuffled = list(picks)
    order.shuffle(shuffled)
    return Sample(
        sampler_version=SAMPLER_VERSION,
        seed=seed,
        per_agent=per_agent,
        agents=list(agents),
        run_dir=str(run_dir),
        exp_id=manifest.exp_id,
        dataset_hash=manifest.dataset_hash,
        fsm_hash=manifest.fsm_hash,
        eligibility=ELIGIBILITY,
        frame_hash=frame_hash(frame),
        n_frame=len(frame),
        n_frame_by_agent={
            agent: sum(1 for response in frame if response.agent == agent)
            for agent in agents
        },
        n_dialogues=len({response.dialogue_id for response in frame}),
        max_per_dialogue=max(
            sum(1 for pick in picks if pick.dialogue_id == response.dialogue_id)
            for response in picks
        ),
        strata=strata,
        responses=[
            SampledResponse(
                annotation_id=f"A{number:02d}",
                unit_id=response.unit_id,
                dialogue_id=response.dialogue_id,
                scenario_id=response.scenario_id,
                agent=response.agent,
                repetition=response.repetition,
                turn=response.turn,
            )
            for number, response in enumerate(shuffled, start=1)
        ],
        dialogues=_blind_dialogues(picks, seed=seed),
    )


def _blind_dialogues(picks: Sequence[Response], *, seed: int) -> list[SampledDialogue]:
    """Number the drawn dialogues under IDs that do not order them by agent.

    Sorting the real dialogue IDs would hand every scenario's baseline the lower
    number, so the order is a shuffle of its own, seeded like every other choice
    here.
    """
    by_id = {pick.dialogue_id: pick for pick in picks}
    shuffled = sorted(by_id)
    random.Random(f"{seed}|dialogues").shuffle(shuffled)
    return [
        SampledDialogue(
            dialogue_annotation_id=f"D{number:02d}",
            dialogue_id=dialogue_id,
            scenario_id=by_id[dialogue_id].scenario_id,
            agent=by_id[dialogue_id].agent,
            repetition=by_id[dialogue_id].repetition,
            n_sampled_responses=sum(
                1 for pick in picks if pick.dialogue_id == dialogue_id
            ),
        )
        for number, dialogue_id in enumerate(shuffled, start=1)
    ]


def frame_hash(frame: Sequence[Response]) -> str:
    """Hash the frame's units and texts, so a changed run cannot pass as the same."""
    digest = hashlib.sha256()
    for response in frame:
        digest.update(response.unit_id.encode("utf-8"))
        digest.update(response.text.encode("utf-8"))
    return digest.hexdigest()


def write_sample(
    sample: Sample,
    *,
    dialogues: Mapping[str, DialogueLog],
    scenarios: Mapping[str, Scenario],
    kb: KnowledgeBase,
    out_dir: Path,
) -> list[Path]:
    """Write the sample, the two blind annotation sheets and the evidence packet."""
    paths = [
        out_dir / SAMPLE_JSON,
        out_dir / RESPONSE_CSV,
        out_dir / DIALOGUE_CSV,
        out_dir / PACKET_MD,
    ]
    atomic_write(paths[0], sample.model_dump_json(indent=2) + "\n")
    atomic_write(
        paths[1],
        _empty_grid(
            RESPONSE_FIELDS,
            [response.annotation_id for response in sample.responses],
        ),
    )
    atomic_write(
        paths[2],
        _empty_grid(
            DIALOGUE_FIELDS,
            [dialogue.dialogue_annotation_id for dialogue in sample.dialogues],
        ),
    )
    atomic_write(paths[3], _render_packet(sample, dialogues, scenarios, kb))
    return paths


def _select(
    frame: Sequence[Response],
    *,
    seed: int,
    per_agent: int,
    agents: Sequence[str],
) -> tuple[list[Response], list[Stratum]]:
    """Spread ``per_agent`` over each agent's scenarios, then over its dialogues."""
    picks: list[Response] = []
    strata: list[Stratum] = []
    for agent in agents:
        of_agent = [response for response in frame if response.agent == agent]
        scenario_ids = sorted({response.scenario_id for response in of_agent})
        by_scenario = [
            [response for response in of_agent if response.scenario_id == scenario_id]
            for scenario_id in scenario_ids
        ]
        quotas = _round_robin(
            per_agent,
            [len(responses) for responses in by_scenario],
            random.Random(f"{seed}|{agent}"),
            what=f"{agent} responses",
        )
        for scenario_id, responses, quota in zip(
            scenario_ids, by_scenario, quotas, strict=True
        ):
            taken, per_dialogue = _from_scenario(
                responses, quota=quota, seed=seed, agent=agent, scenario_id=scenario_id
            )
            picks.extend(taken)
            strata.append(
                Stratum(
                    agent=agent,
                    scenario_id=scenario_id,
                    n_frame=len(responses),
                    quota=quota,
                    per_dialogue=per_dialogue,
                )
            )
    return picks, strata


def _from_scenario(
    responses: Sequence[Response],
    *,
    quota: int,
    seed: int,
    agent: str,
    scenario_id: str,
) -> tuple[list[Response], dict[str, int]]:
    """Draw ``quota`` responses from one stratum, spread over its dialogues."""
    dialogue_ids = sorted({response.dialogue_id for response in responses})
    by_dialogue = [
        [response for response in responses if response.dialogue_id == name]
        for name in dialogue_ids
    ]
    shares = _round_robin(
        quota,
        [len(turns) for turns in by_dialogue],
        random.Random(f"{seed}|{agent}|{scenario_id}"),
        what=f"{agent} responses on {scenario_id}",
    )
    taken: list[Response] = []
    for name, turns, share in zip(dialogue_ids, by_dialogue, shares, strict=True):
        drawn = random.Random(f"{seed}|{name}").sample(sorted(turns, key=_turn), share)
        taken.extend(sorted(drawn, key=_turn))
    return taken, dict(zip(dialogue_ids, shares, strict=True))


def _round_robin(
    quota: int, capacities: Sequence[int], rng: random.Random, *, what: str
) -> list[int]:
    """Hand out ``quota`` one at a time over a shuffled order of the units.

    Even by construction: a unit is never two ahead of another that still has
    room, and which unit gets an extra is the only thing chance decides. A unit
    smaller than its share spills the rest onto the others.
    """
    if quota > sum(capacities):
        raise SamplingError(
            f"the frame holds {sum(capacities)} {what}, but {quota} were asked "
            f"for. Lower --per-agent or run more dialogues"
        )
    order = list(range(len(capacities)))
    rng.shuffle(order)
    taken = [0] * len(capacities)
    remaining = quota
    while remaining:
        for index in order:
            if not remaining:
                break
            if taken[index] < capacities[index]:
                taken[index] += 1
                remaining -= 1
    return taken


def _turn(response: Response) -> int:
    """Order the responses of one dialogue by turn."""
    return response.turn


def _empty_grid(fields: Sequence[str], ids: Sequence[str]) -> str:
    """Render an annotation grid: one row per ID, every other cell empty."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for identifier in ids:
        writer.writerow(dict.fromkeys(fields, "") | {fields[0]: identifier})
    return buffer.getvalue()


def _render_packet(
    sample: Sample,
    dialogues: Mapping[str, DialogueLog],
    scenarios: Mapping[str, Scenario],
    kb: KnowledgeBase,
) -> str:
    """Render the evidence for each drawn response, in annotation order."""
    parts = [_packet_header(sample), _packet_catalogue(kb)]
    for drawn in sample.responses:
        scenario = scenarios.get(drawn.scenario_id)
        if scenario is None:
            raise SamplingError(
                f"{drawn.scenario_id} is not in the scenarios the manifest names. "
                "The packet shows the answer key the judge reads, so the scenario "
                "must be loadable"
            )
        parts.append(
            _packet_entry(
                drawn,
                dialogues[drawn.dialogue_id],
                scenario,
                sample.dialogue_annotation_id(drawn.dialogue_id),
                kb,
            )
        )
    return "\n".join(parts)


def _packet_header(sample: Sample) -> str:
    """Explain what to fill and with which values, once, at the top."""
    return "\n".join(
        (
            "# Judge validation packet (T-16)",
            "",
            f"Seed {sample.seed}, {len(sample.responses)} responses drawn from "
            f"{sample.n_frame} in `{sample.run_dir}`, over "
            f"{len(sample.dialogues)} dialogues. Which agent produced a response "
            "is not shown here: annotate the text, then join on "
            f"`{SAMPLE_JSON}`. Do not open the judge's output first; the point "
            "of the exercise is an independent label.",
            "",
            f"`{RESPONSE_CSV}`, one row per `A` ID, about the marked response only:",
            "",
            "- `fact_ids_stated`: every knowledge-base ID the marked response "
            "materially states, separated by `;`, or empty for none",
            "- `claim_support`: `all_supported`, `some_unsupported` or "
            "`none_checkable`, over the checkable claims of the marked response",
            "- `notes`: free text",
            "",
            f"`{DIALOGUE_CSV}`, one row per `D` ID, about the whole transcript. "
            "Two responses of the same dialogue carry the same `D` ID: fill that "
            "row once.",
            "",
            "- `accuracy`: `correct`, `partial` or `incorrect`, comparing the "
            "assistant's outcome to the reference answer",
            "- `task_completed`: `yes` or `no`, by the success criterion alone",
            "- `notes`: free text",
            "",
            "`docs/judge_validation.md` states each value in full. Read it once "
            "before the first row.",
            "",
        )
    )


def _packet_catalogue(kb: KnowledgeBase) -> str:
    """Render the whole fact catalogue, the closed world the judge grades against.

    Every fact, not the scenario's required ones: a response is free to state a
    fact nobody asked for, and claim support cannot be decided against a subset.
    It is rendered by :func:`~sim.kb.render_facts`, the same call that fills the
    judge's prompt, so the two graders read one text.
    """
    return "\n".join(
        (
            "## Knowledge base",
            "",
            "The closed world: a claim is supported only when this text contains "
            "it. Any fact may be cited, not only the ones a scenario requires.",
            "",
            "```",
            render_facts(kb.facts).strip(),
            "```",
            "",
        )
    )


def _packet_entry(
    drawn: SampledResponse,
    log: DialogueLog,
    scenario: Scenario,
    dialogue_annotation_id: str,
    kb: KnowledgeBase,
) -> str:
    """Render one response with the answer key and the dialogue around it."""
    canary = scenario.canary
    lines = [
        f"## {drawn.annotation_id}",
        "",
        f"- dialogue: `{dialogue_annotation_id}`",
        f"- scenario: `{scenario.id}` ({scenario.category}, {scenario.intent})",
        f"- required facts: {', '.join(scenario.required_facts)} (the claims of "
        "the response may go beyond these; grade them against the whole "
        "knowledge base above)",
        f"- needle fact: {needle_fact(scenario, kb)}",
        f"- response to annotate: turn {drawn.turn}",
        "",
        f"Success criterion: {redact_canary(scenario.success_criterion, canary)}",
        "",
        f"Reference answer: {redact_canary(scenario.reference_answer, canary)}",
        "",
        "Script:",
        "",
        "```",
        redact_canary(render_script(scenario.script), canary),
        "```",
        "",
        "Transcript (`>>>` marks the response to annotate):",
        "",
        "```",
    ]
    for record in log.records:
        mark = ">>> " if record.turn == drawn.turn else ""
        lines.append(f"user: {redact_canary(record.user_message, canary)}")
        lines.append(
            f"{mark}agent [{record.turn}]: {redact_canary(record.agent_reply, canary)}"
        )
    lines.extend(("```", ""))
    return "\n".join(lines)


def _load_manifest(run_dir: Path) -> Manifest:
    """Load the run manifest, which identifies what the sample was drawn from."""
    path = run_dir / "manifest.json"
    if not path.exists():
        raise SamplingError(
            f"{path} is missing. The sample records the run it was drawn from, "
            "so T-16 can be traced back to the dialogues it annotated"
        )
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def _evidence(run_dir: Path) -> tuple[KnowledgeBase, dict[str, Scenario]]:
    """Load what the annotator grades against: the KB and the run's scenarios."""
    manifest = _load_manifest(run_dir)
    kb = load_kb(DEFAULT_KB_DIR)
    fsm = load_fsm(DEFAULT_FSM_DIR)
    scenarios = load_scenarios(Path(manifest.scenarios_dir), kb=kb, fsm=fsm)
    return kb, {scenario.id: scenario for scenario in scenarios}


def main(argv: Sequence[str] | None = None) -> int:
    """Draw the sample of the run in ``argv`` and write the four files."""
    parser = argparse.ArgumentParser(
        description="Draw the T-16 judge-validation sample from a sim run."
    )
    parser.add_argument("--run", type=Path, required=True, help="experiment directory")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR, help="where the files go"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-agent", type=int, default=DEFAULT_PER_AGENT)
    args = parser.parse_args(argv)
    try:
        sample = draw_sample(args.run, seed=args.seed, per_agent=args.per_agent)
        kb, scenarios = _evidence(args.run)
        written = write_sample(
            sample,
            dialogues=load_dialogues(args.run),
            scenarios=scenarios,
            kb=kb,
            out_dir=args.out,
        )
    except SamplingError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    print(_summary(sample, written))
    return 0


def _summary(sample: Sample, written: Iterable[Path]) -> str:
    """Report what was drawn and where it went."""
    lines = [
        f"{len(sample.responses)} responses drawn from {sample.n_frame} "
        f"(seed {sample.seed}, at most {sample.max_per_dialogue} per dialogue), "
        f"over {len(sample.dialogues)} dialogues"
    ]
    for agent in sample.agents:
        drawn = sum(1 for pick in sample.responses if pick.agent == agent)
        lines.append(f"  {agent}: {drawn} of {sample.n_frame_by_agent[agent]}")
    lines.extend(f"  wrote {path}" for path in written)
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
