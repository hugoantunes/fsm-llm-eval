"""Build the blinded human-evaluation sidecar for frozen ``exp_final``.

The packet is an independent extra evaluation of the scored
``semantic_primary`` population. It does not retune the judge, rescore
dialogues, or compare human labels with judge output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from csv import DictReader
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from sim.analysis import load_metrics_csv
from sim.audit import SIDECAR_METRICS_SHA256, file_sha256
from sim.fsm import DEFAULT_FSM_DIR, load_fsm
from sim.io import atomic_write
from sim.judge import redact_canary
from sim.kb import DEFAULT_KB_DIR, KnowledgeBase, load_kb, render_facts
from sim.reporting import UNPAIRED_SCENARIO
from sim.runner import FROZEN_V1_HASH
from sim.schemas import (
    DialogueLog,
    Manifest,
    Scenario,
    TurnRecord,
    dialogue_filename,
    dialogue_id,
    load_scenarios,
)

DEFAULT_RUN_DIR = Path("runs/exp_final")
DEFAULT_METRICS = Path("results/exp_final/metrics.csv")
DEFAULT_OUT_DIR = Path("results/human_validation/exp_final")
DEFAULT_SEED = 20260920
EXPECTED_N_DIALOGUES = 348
EXPECTED_N_PAIRED_SCENARIOS = 59
DIALOGUE_CSV = "dialogue_annotations.csv"
RESPONSE_CSV = "response_annotations.csv"
CLAIMS_CSV = "claims_annotations.csv"
MAPPING_CSV = "mapping.csv"
MANIFEST_JSON = "manifest.json"
README_MD = "README.md"
RUBRIC_MD = "rubric.md"
DIALOGUES_DIR = "dialogues"
PRIVATE_DIR = "private"
DIALOGUE_FIELDS = ("blind_dialogue_id", "task_completed", "notes")
RESPONSE_FIELDS = (
    "blind_dialogue_id",
    "blind_response_id",
    "fact_ids_stated",
    "notes",
)
CLAIMS_FIELDS = (
    "blind_dialogue_id",
    "blind_response_id",
    "claim_id",
    "claim_text",
    "support",
    "notes",
)
MAPPING_FIELDS = (
    "blind_dialogue_id",
    "dialogue_id",
    "scenario_id",
    "condition",
    "repetition",
)
METADATA_LEAK_TOKENS = (
    "true_state_after",
    "labelled_stage",
    "judge_facts",
    "judge_global",
)


class PacketError(RuntimeError):
    """The human-evaluation packet cannot be written; the message says why."""


@dataclass(frozen=True)
class DialogueRef:
    """One scored ``semantic_primary`` dialogue, identity only."""

    dialogue_id: str
    scenario_id: str
    condition: str
    repetition: int


@dataclass(frozen=True)
class BlindDialogue:
    """One census dialogue under the ID the annotator sees."""

    blind_id: str
    ref: DialogueRef
    response_ids: tuple[str, ...]


@dataclass(frozen=True)
class Packet:
    """In-memory blind packet, ready to write public files then the mapping."""

    dialogues: tuple[BlindDialogue, ...]
    logs: Mapping[str, DialogueLog]
    scenarios: Mapping[str, Scenario]
    kb: KnowledgeBase
    seed: int
    generated_at: str
    population_source: str
    metrics_sha256: str
    transcripts_sha256: str
    frozen_v1_hash: str
    n_scenarios: int
    n_paired_scenarios: int


def load_identity(metrics_path: Path) -> list[DialogueRef]:
    """Return census identity rows from a metrics CSV, scores discarded."""
    seen: set[str] = set()
    refs: list[DialogueRef] = []
    for row in load_metrics_csv(metrics_path):
        scenario_id = str(row["scenario_id"])
        condition = str(row["agent"])
        repetition = int(row["repetition"])
        name = dialogue_id(scenario_id, condition, repetition)
        if name in seen:
            raise PacketError(f"duplicate dialogue_id {name} in {metrics_path}")
        seen.add(name)
        refs.append(
            DialogueRef(
                dialogue_id=name,
                scenario_id=scenario_id,
                condition=condition,
                repetition=repetition,
            )
        )
    if not refs:
        raise PacketError(f"{metrics_path} has no dialogue rows")
    return refs


def scenario_counts(refs: Sequence[DialogueRef]) -> tuple[int, int, frozenset[str]]:
    """Return unique scenarios, paired count, and unpaired scenario ids."""
    agents: dict[str, set[str]] = defaultdict(set)
    for ref in refs:
        agents[ref.scenario_id].add(ref.condition)
    paired = [
        scenario_id
        for scenario_id, names in agents.items()
        if {"baseline", "fsm"} <= names
    ]
    unpaired = frozenset(
        scenario_id
        for scenario_id, names in agents.items()
        if not {"baseline", "fsm"} <= names
    )
    return len(agents), len(paired), unpaired


def require_canonical_census(refs: Sequence[DialogueRef], metrics_path: Path) -> None:
    """Fail unless ``metrics_path`` is the frozen ``semantic_primary`` CSV."""
    digest = file_sha256(metrics_path)
    if digest != SIDECAR_METRICS_SHA256:
        raise PacketError(
            f"{metrics_path} sha256 is {digest}, expected "
            f"{SIDECAR_METRICS_SHA256}. The packet uses the frozen scored "
            "semantic_primary CSV and no other population"
        )
    _, n_paired, unpaired = scenario_counts(refs)
    conditions = {ref.condition for ref in refs}
    if (
        len(refs) != EXPECTED_N_DIALOGUES
        or n_paired != EXPECTED_N_PAIRED_SCENARIOS
        or unpaired != frozenset({UNPAIRED_SCENARIO})
        or not {"baseline", "fsm"} <= conditions
    ):
        raise PacketError(
            "semantic_primary census mismatch: "
            f"n_dialogues={len(refs)} (expected {EXPECTED_N_DIALOGUES}), "
            f"n_paired_scenarios={n_paired} "
            f"(expected {EXPECTED_N_PAIRED_SCENARIOS}), "
            f"unpaired={sorted(unpaired)} (expected [{UNPAIRED_SCENARIO}]), "
            f"conditions={sorted(conditions)}"
        )


def transcripts_sha256(run_dir: Path, refs: Sequence[DialogueRef]) -> str:
    """SHA-256 of per-file digests, ordered by real dialogue_id.

    For each census JSONL, in ASCII ``dialogue_id`` order, take the SHA-256
    hex digest of the file bytes. Concatenate those 64-character lines, each
    followed by a newline. The result is the SHA-256 hex digest of that
    concatenation.
    """
    lines: list[str] = []
    for ref in sorted(refs, key=lambda item: item.dialogue_id):
        path = (
            run_dir
            / "dialogues"
            / dialogue_filename(ref.scenario_id, ref.condition, ref.repetition)
        )
        lines.append(file_sha256(path) + "\n")
    return hashlib.sha256("".join(lines).encode("ascii")).hexdigest()


def assign_blind_ids(
    refs: Sequence[DialogueRef],
    logs: Mapping[str, DialogueLog],
    *,
    seed: int,
) -> list[BlindDialogue]:
    """Shuffle real IDs with ``seed`` and number them ``D001``…."""
    ordered = sorted(ref.dialogue_id for ref in refs)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    by_id = {ref.dialogue_id: ref for ref in refs}
    dialogues: list[BlindDialogue] = []
    for number, name in enumerate(shuffled, start=1):
        blind_id = f"D{number:03d}"
        log = logs[name]
        n_replies = len(_annotatable_records(log))
        response_ids = tuple(
            f"{blind_id}-A{index:02d}" for index in range(1, n_replies + 1)
        )
        dialogues.append(
            BlindDialogue(blind_id=blind_id, ref=by_id[name], response_ids=response_ids)
        )
    return dialogues


def load_logs(run_dir: Path, refs: Sequence[DialogueRef]) -> dict[str, DialogueLog]:
    """Load the census JSONL files; include failed logs that are in the CSV."""
    directory = run_dir / "dialogues"
    if not directory.exists():
        raise PacketError(
            f"{directory} is missing. The packet reads frozen dialogue JSONL "
            "and does not rerun the experiment"
        )
    missing: list[str] = []
    logs: dict[str, DialogueLog] = {}
    for ref in refs:
        path = directory / dialogue_filename(
            ref.scenario_id, ref.condition, ref.repetition
        )
        if not path.exists():
            missing.append(ref.dialogue_id)
            continue
        logs[ref.dialogue_id] = DialogueLog.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    if missing:
        preview = ", ".join(missing[:8])
        extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise PacketError(
            f"{len(missing)} census dialogue JSONL file(s) missing under "
            f"{directory}: {preview}{extra}"
        )
    return logs


def build_packet(
    *,
    run_dir: Path,
    metrics_path: Path,
    seed: int = DEFAULT_SEED,
    generated_at: str | None = None,
    kb: KnowledgeBase | None = None,
    scenarios: Mapping[str, Scenario] | None = None,
    enforce_canonical: bool = False,
) -> Packet:
    """Build the in-memory packet from frozen metrics identity and JSONL."""
    refs = load_identity(metrics_path)
    if enforce_canonical:
        require_canonical_census(refs, metrics_path)
    logs = load_logs(run_dir, refs)
    knowledge = kb if kb is not None else load_kb(DEFAULT_KB_DIR)
    loaded_scenarios = (
        dict(scenarios) if scenarios is not None else _load_run_scenarios(run_dir)
    )
    n_scenarios, n_paired, _unpaired = scenario_counts(refs)
    dialogues = assign_blind_ids(refs, logs, seed=seed)
    return Packet(
        dialogues=tuple(dialogues),
        logs=logs,
        scenarios=loaded_scenarios,
        kb=knowledge,
        seed=seed,
        generated_at=generated_at or _utc_now(),
        population_source=str(metrics_path),
        metrics_sha256=file_sha256(metrics_path),
        transcripts_sha256=transcripts_sha256(run_dir, refs),
        frozen_v1_hash=FROZEN_V1_HASH,
        n_scenarios=n_scenarios,
        n_paired_scenarios=n_paired,
    )


def write_public(packet: Packet, out_dir: Path) -> list[Path]:
    """Write annotator-facing files from the in-memory mapping.

    Does not read or write ``private/mapping.csv``.
    """
    _refuse_filled_annotations(out_dir)
    dialogues_dir = out_dir / DIALOGUES_DIR
    dialogues_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / README_MD,
        out_dir / RUBRIC_MD,
        out_dir / MANIFEST_JSON,
        out_dir / DIALOGUE_CSV,
        out_dir / RESPONSE_CSV,
        out_dir / CLAIMS_CSV,
    ]
    atomic_write(paths[0], render_readme())
    atomic_write(paths[1], render_rubric(packet.kb))
    atomic_write(paths[2], _manifest_json(packet))
    atomic_write(paths[3], _dialogue_csv(packet))
    atomic_write(paths[4], _response_csv(packet))
    atomic_write(paths[5], _claims_csv_header())
    for dialogue in packet.dialogues:
        path = dialogues_dir / f"{dialogue.blind_id}.md"
        atomic_write(path, render_dialogue_markdown(dialogue, packet))
        paths.append(path)
    return paths


def write_private(packet: Packet, out_dir: Path) -> Path:
    """Persist the unblinding table; not an input to annotator files."""
    path = out_dir / PRIVATE_DIR / MAPPING_CSV
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MAPPING_FIELDS, extrasaction="raise")
    writer.writeheader()
    for dialogue in packet.dialogues:
        writer.writerow(
            {
                "blind_dialogue_id": dialogue.blind_id,
                "dialogue_id": dialogue.ref.dialogue_id,
                "scenario_id": dialogue.ref.scenario_id,
                "condition": dialogue.ref.condition,
                "repetition": str(dialogue.ref.repetition),
            }
        )
    atomic_write(path, buffer.getvalue())
    return path


def export_packet(
    *,
    run_dir: Path,
    metrics_path: Path,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    generated_at: str | None = None,
    kb: KnowledgeBase | None = None,
    scenarios: Mapping[str, Scenario] | None = None,
    enforce_canonical: bool = False,
) -> Packet:
    """Build the packet, write public files, then persist the private mapping."""
    packet = build_packet(
        run_dir=run_dir,
        metrics_path=metrics_path,
        seed=seed,
        generated_at=generated_at,
        kb=kb,
        scenarios=scenarios,
        enforce_canonical=enforce_canonical,
    )
    write_public(packet, out_dir)
    write_private(packet, out_dir)
    validate_packet(out_dir, mapping_required=True)
    return packet


def validate_packet(out_dir: Path, *, mapping_required: bool = True) -> None:
    """Check that the written sidecar is complete, empty, and metadata-blind."""
    dialogue_rows = _read_csv(out_dir / DIALOGUE_CSV)
    response_rows = _read_csv(out_dir / RESPONSE_CSV)
    claims_rows = _read_csv(out_dir / CLAIMS_CSV)
    if list(_csv_fields(out_dir / DIALOGUE_CSV)) != list(DIALOGUE_FIELDS):
        raise PacketError("dialogue_annotations.csv fields are not the declared set")
    if list(_csv_fields(out_dir / RESPONSE_CSV)) != list(RESPONSE_FIELDS):
        raise PacketError("response_annotations.csv fields are not the declared set")
    if list(_csv_fields(out_dir / CLAIMS_CSV)) != list(CLAIMS_FIELDS):
        raise PacketError("claims_annotations.csv fields are not the declared set")
    if "unsupported_claims" in _csv_fields(out_dir / RESPONSE_CSV):
        raise PacketError("response_annotations.csv must not carry unsupported_claims")
    if "claim_support" in _csv_fields(out_dir / RESPONSE_CSV):
        raise PacketError("response_annotations.csv must not carry claim_support")
    if claims_rows:
        raise PacketError("claims_annotations.csv must be header-only at generation")
    n_dialogues = len(dialogue_rows)
    ids = [row["blind_dialogue_id"] for row in dialogue_rows]
    expected_ids = [f"D{number:03d}" for number in range(1, n_dialogues + 1)]
    if ids != expected_ids:
        raise PacketError(
            "blind dialogue IDs are not D001… in order, without duplicates"
        )
    for row in dialogue_rows:
        if row["task_completed"] or row["notes"]:
            raise PacketError(
                "dialogue_annotations.csv judgment cells must start empty"
            )
    response_ids: list[str] = []
    for row in response_rows:
        if row["fact_ids_stated"] or row["notes"]:
            raise PacketError(
                "response_annotations.csv judgment cells must start empty"
            )
        response_ids.append(row["blind_response_id"])
    if len(response_ids) != len(set(response_ids)):
        raise PacketError("blind response IDs are not unique")
    manifest = json.loads((out_dir / MANIFEST_JSON).read_text(encoding="utf-8"))
    if "mapping_sha256" in manifest or "dialogue_ids" in manifest:
        raise PacketError("public manifest must not carry mapping or real dialogue IDs")
    if manifest.get("n_dialogues") != n_dialogues:
        raise PacketError("manifest n_dialogues does not match the dialogue sheet")
    _assert_no_metadata_leak(out_dir, dialogue_rows, response_rows, manifest)
    if mapping_required:
        mapping = _read_csv(out_dir / PRIVATE_DIR / MAPPING_CSV)
        mapped = [row["blind_dialogue_id"] for row in mapping]
        if mapped != ids:
            raise PacketError("private mapping is not a bijection with D001… order")
        real_ids = [row["dialogue_id"] for row in mapping]
        if len(real_ids) != len(set(real_ids)):
            raise PacketError("private mapping has duplicate real dialogue IDs")
        if any(not row["condition"] for row in mapping):
            raise PacketError("private mapping is missing condition")


def render_dialogue_markdown(dialogue: BlindDialogue, packet: Packet) -> str:
    """Render one annotator-facing dialogue file."""
    log = packet.logs[dialogue.ref.dialogue_id]
    scenario = packet.scenarios.get(dialogue.ref.scenario_id)
    if scenario is None:
        raise PacketError(
            f"{dialogue.ref.scenario_id} is not in the scenarios the run names"
        )
    canary = scenario.canary
    fact_by_id = {fact.id: fact for fact in packet.kb.facts}
    expected: list[str] = []
    for fact_id in scenario.required_facts:
        fact = fact_by_id.get(fact_id)
        if fact is None:
            raise PacketError(
                f"{scenario.id} required_facts names {fact_id}, which is not "
                "in the knowledge-base catalogue"
            )
        expected.append(f"- {fact.id} — {redact_canary(fact.text, canary)}")
    lines = [
        f"# {dialogue.blind_id}",
        "",
        "## Success criterion",
        "",
        redact_canary(scenario.success_criterion, canary),
        "",
        "## Reference answer",
        "",
        redact_canary(scenario.reference_answer, canary),
        "",
        "## Expected facts",
        "",
        *expected,
        "",
        "## Dialogue",
        "",
    ]
    reply_index = 0
    for record in log.records:
        lines.append("### USER")
        lines.append(redact_canary(record.user_message, canary))
        lines.append("")
        if not record.agent_reply.strip():
            continue
        response_id = dialogue.response_ids[reply_index]
        reply_index += 1
        lines.append(f"### ASSISTANT — {response_id}")
        lines.append(redact_canary(record.agent_reply, canary))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_readme() -> str:
    """Return the annotator-facing README."""
    return """# Human evaluation sidecar (Final Experiment)

This is an additional human evaluation of the frozen final experiment. It does
not modify or replace the existing judge evaluation at this stage.

The packet covers the same 348 scored dialogues used by `semantic_primary`.
Blind IDs hide the experimental condition. Judge predictions, explanations,
and aggregate scores are not in these files.

## Workflow

1. Read `rubric.md`.
2. Annotate dialogues in blind-ID order (`D001` … `D348`).
3. Fill `dialogue_annotations.csv` (`task_completed`: `yes` / `no` / `uncertain`).
4. Fill `response_annotations.csv` (`fact_ids_stated` as `F03;F07`, or empty).
5. Add one row per atomic checkable claim to `claims_annotations.csv`.
   Do not fill `claim_support`; it will be derived later.
6. Do not inspect `private/mapping.csv` (or anything under `private/`).
7. Do not inspect old judge outputs, `metrics.csv` scores, or aggregate
   baseline/FSM results for individual cases while annotating.

## Post-annotation freeze

After all 348 dialogues are annotated, and **before** any look at judge
results or `private/mapping.csv`, freeze the filled CSVs by copy plus
SHA-256. Human vs judge comparison and unblinding happen only after that
freeze, in a later ticket.
"""


def render_rubric(kb: KnowledgeBase) -> str:
    """Return the operational rubric plus the closed-world fact catalogue."""
    catalogue = render_facts(kb.facts)
    return f"""# Human evaluation rubric

Grade the dialogue text against this rubric and the knowledge base below.
Do not open judge outputs. Do not infer the experimental condition.

## `task_completed` (dialogue)

Judge the scenario success criterion printed in the `Dxxx.md` file, not
overall conversational quality.

- `yes`: the criterion was satisfied.
- `no`: the criterion was not satisfied.
- `uncertain`: the evidence does not permit a sufficiently confident binary
  decision.

## `fact_ids_stated` (response)

For each assistant response, list the expected-fact IDs that response
actually communicated, separated by `;`. Empty if none.

- Semantic paraphrases count; lexical identity is not required.
- Do not mark a fact only because it appears under Expected facts.
- Partial statements count only if the required information was actually
  conveyed.
- Repeated facts may be marked in every response where they are expressed.
- A claim that contradicts a fact gets no ID.
- If a response states the core fact but adds an incorrect qualifier or
  condition that materially changes its meaning, do not mark that fact ID.
  Example: the catalogue says “the deadline is 30 days”; the reply says
  “the deadline is 30 days only for premium customers”.

## Claims (`claims_annotations.csv`)

One row per atomic checkable assertion about a policy, a deadline, a fee,
an order status, or a next step.

- `support`: `supported` / `unsupported` / `uncertain`.
- Store the claim text. Do not force uncertain cases into `unsupported`.
- Do not record greetings, questions, conversational filler, purely
  subjective statements, or restated customer data.
- Repeated factual assertions are annotated per response occurrence. Do
  not deduplicate claims across assistant responses.
- Within the same response, do not invent extra rows for a single
  assertion that is only restated redundantly.

`claim_support` is not filled by hand. A later ticket will derive it as
supported / (supported + unsupported) over checkable claims.

## Knowledge base

The closed world. A claim is supported only when this text contains it.

```
{catalogue}
```
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Build the sidecar from frozen ``exp_final`` artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    metrics_before = file_sha256(args.metrics) if args.metrics.exists() else None
    run_metrics = args.run / "metrics.csv"
    run_metrics_before = file_sha256(run_metrics) if run_metrics.exists() else None
    try:
        packet = export_packet(
            run_dir=args.run,
            metrics_path=args.metrics,
            out_dir=args.out,
            seed=args.seed,
            enforce_canonical=True,
        )
    except PacketError as failure:
        print(f"sim: {failure}", file=sys.stderr)
        return 1
    if metrics_before is not None and file_sha256(args.metrics) != metrics_before:
        print(f"sim: {args.metrics} changed during generation", file=sys.stderr)
        return 1
    if (
        run_metrics_before is not None
        and file_sha256(run_metrics) != run_metrics_before
    ):
        print(f"sim: {run_metrics} changed during generation", file=sys.stderr)
        return 1
    mapping = args.out / PRIVATE_DIR / MAPPING_CSV
    print(
        f"wrote {len(packet.dialogues)} dialogues, "
        f"{sum(len(item.response_ids) for item in packet.dialogues)} "
        f"assistant responses, seed {packet.seed}"
    )
    print(f"metrics_sha256={packet.metrics_sha256}")
    print(f"transcripts_sha256={packet.transcripts_sha256}")
    print(f"mapping_sha256={file_sha256(mapping)}")
    return 0


def _annotatable_records(log: DialogueLog) -> list[TurnRecord]:
    """Return assistant turns whose reply is not blank."""
    return [record for record in log.records if record.agent_reply.strip()]


def _load_run_scenarios(run_dir: Path) -> dict[str, Scenario]:
    """Load the scenario files the run manifest names."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise PacketError(f"{manifest_path} is missing")
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    kb = load_kb(DEFAULT_KB_DIR)
    fsm = load_fsm(DEFAULT_FSM_DIR)
    return {
        scenario.id: scenario
        for scenario in load_scenarios(Path(manifest.scenarios_dir), kb=kb, fsm=fsm)
    }


def _utc_now() -> str:
    """Return a UTC ISO-8601 timestamp with a ``Z`` suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _refuse_filled_annotations(out_dir: Path) -> None:
    """Fail if a previous annotation pass already wrote judgments."""
    dialogue_path = out_dir / DIALOGUE_CSV
    response_path = out_dir / RESPONSE_CSV
    claims_path = out_dir / CLAIMS_CSV
    if dialogue_path.exists():
        for row in _read_csv(dialogue_path):
            if row.get("task_completed") or row.get("notes"):
                raise PacketError(
                    f"{dialogue_path} already has filled judgment cells; "
                    "refusing to overwrite"
                )
    if response_path.exists():
        for row in _read_csv(response_path):
            if row.get("fact_ids_stated") or row.get("notes"):
                raise PacketError(
                    f"{response_path} already has filled judgment cells; "
                    "refusing to overwrite"
                )
    if claims_path.exists() and _read_csv(claims_path):
        raise PacketError(
            f"{claims_path} already has claim rows; refusing to overwrite"
        )


def _dialogue_csv(packet: Packet) -> str:
    """Render the empty dialogue annotation sheet."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=DIALOGUE_FIELDS, extrasaction="raise")
    writer.writeheader()
    for dialogue in packet.dialogues:
        writer.writerow(
            {
                "blind_dialogue_id": dialogue.blind_id,
                "task_completed": "",
                "notes": "",
            }
        )
    return buffer.getvalue()


def _response_csv(packet: Packet) -> str:
    """Render the empty response annotation sheet."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RESPONSE_FIELDS, extrasaction="raise")
    writer.writeheader()
    for dialogue in packet.dialogues:
        for response_id in dialogue.response_ids:
            writer.writerow(
                {
                    "blind_dialogue_id": dialogue.blind_id,
                    "blind_response_id": response_id,
                    "fact_ids_stated": "",
                    "notes": "",
                }
            )
    return buffer.getvalue()


def _claims_csv_header() -> str:
    """Render the claims sheet with a header and no data rows."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CLAIMS_FIELDS, extrasaction="raise")
    writer.writeheader()
    return buffer.getvalue()


def _manifest_json(packet: Packet) -> str:
    """Render public provenance; no mapping and no real dialogue IDs."""
    n_responses = sum(len(item.response_ids) for item in packet.dialogues)
    payload = {
        "population_source": packet.population_source,
        "n_dialogues": len(packet.dialogues),
        "n_assistant_responses": n_responses,
        "n_scenarios": packet.n_scenarios,
        "n_paired_scenarios": packet.n_paired_scenarios,
        "seed": packet.seed,
        "generated_at": packet.generated_at,
        "frozen_v1_hash": packet.frozen_v1_hash,
        "metrics_sha256": packet.metrics_sha256,
        "transcripts_sha256": packet.transcripts_sha256,
    }
    return json.dumps(payload, indent=2) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(DictReader(handle))


def _csv_fields(path: Path) -> tuple[str, ...]:
    """Return the header fields of ``path``."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        if reader.fieldnames is None:
            return ()
        return tuple(reader.fieldnames)


def _assert_no_metadata_leak(
    out_dir: Path,
    dialogue_rows: Sequence[Mapping[str, str]],
    response_rows: Sequence[Mapping[str, str]],
    manifest: Mapping[str, object],
) -> None:
    """Fail if generated fields or headers expose condition or judge metadata."""
    for key in (*DIALOGUE_FIELDS, *RESPONSE_FIELDS, *CLAIMS_FIELDS, *manifest):
        lowered = str(key).lower()
        if lowered in METADATA_LEAK_TOKENS or lowered == "condition":
            raise PacketError(f"annotator-facing field {key!r} leaks metadata")
    for row in (*dialogue_rows, *response_rows):
        for key, value in row.items():
            if key == "blind_dialogue_id" or key == "blind_response_id":
                continue
            text = str(value).lower()
            if any(
                token in text for token in ("baseline", "fsm", *METADATA_LEAK_TOKENS)
            ):
                raise PacketError("annotator-facing CSV cell leaks metadata")
    blob = json.dumps(manifest)
    for token in METADATA_LEAK_TOKENS:
        if token in blob:
            raise PacketError(f"manifest contains {token}")
    for path in sorted((out_dir / DIALOGUES_DIR).glob("D*.md")):
        text = path.read_text(encoding="utf-8")
        prefix, _, _body = text.partition("## Dialogue")
        lowered = prefix.lower()
        if any(token in lowered for token in METADATA_LEAK_TOKENS):
            raise PacketError(f"{path.name} prefix leaks judge or state metadata")
        if "baseline" in lowered or "fsm" in lowered:
            raise PacketError(f"{path.name} prefix names an experimental condition")


if __name__ == "__main__":
    sys.exit(main())
