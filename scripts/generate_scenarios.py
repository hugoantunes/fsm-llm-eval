"""Generate the golden scenario dataset from the cell plan (T-06)."""

import argparse
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sim.config import DEFAULT_CONFIG_PATH, ModelsConfig, load_models_config
from sim.io import atomic_write
from sim.kb import load_kb
from sim.llm import LLM_CALLS_LOG, Chat, LlmClient
from sim.prompts import Prompt, load_prompt
from sim.runner import hash_dataset
from sim.schemas import (
    CATEGORIES,
    FIRST_BLOCK_N,
    FIRST_BLOCK_QUOTA,
    FULL_SET_N,
    FULL_SET_QUOTA,
    Category,
    Scenario,
    render_script,
)


class GenerateError(ValueError):
    """The generator was asked to do something the experiment forbids."""


SCENARIOS_ROOT = Path("data/scenarios")
_VERSION_DIR = re.compile(r"^v(\d+)$")


class Slot(BaseModel):
    """One scenario draft: the gabarito plus the seed persona, goal and script."""

    model_config = ConfigDict(extra="forbid")

    required_facts: list[str] = Field(min_length=1)
    forbidden_facts: list[str] = []
    expected_final_state: str
    success_criterion: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    user_persona: str = Field(min_length=1)
    user_goal: str = Field(min_length=1)
    script: list[str] = Field(min_length=1)
    is_needle: bool = False
    canary: str | None = None
    unanswerable_id: str | None = None
    max_turns: int = 8


class Cell(BaseModel):
    """The slots of one intent x category cell, first block then the extra."""

    model_config = ConfigDict(extra="forbid")

    intent: str
    category: Category
    slots: list[Slot] = Field(min_length=FIRST_BLOCK_QUOTA)


class Plan(BaseModel):
    """The cell plan that expands into ``data/scenarios/vN/``."""

    model_config = ConfigDict(extra="forbid")

    version: int
    cells: list[Cell] = Field(min_length=1)


def resolve_model(config: ModelsConfig, model: str | None = None) -> str:
    """Return the model that will phrase scenarios; refuse the agents' model."""
    agent = config.spec("agent").name
    chosen = config.spec("judge").name if model is None else model
    if chosen == agent:
        raise GenerateError(
            f"scenarios are generated with the judge model, not {agent}. "
            f"Using the agents' model would evaluate them on their own phrasing"
        )
    return chosen


def next_dataset_dir(root: Path) -> Path:
    """Return the next unused ``vN`` directory under ``root``.

    The counter is the highest existing ``vN`` folder, not a separate file, so
    a clone that already has ``v1`` writes ``v2`` without extra state. Gaps
    are not filled: ``v1`` and ``v3`` yield ``v4``.
    """
    versions: list[int] = []
    if root.is_dir():
        for path in root.iterdir():
            matched = _VERSION_DIR.fullmatch(path.name)
            if matched is not None and path.is_dir():
                versions.append(int(matched.group(1)))
    return root / f"v{max(versions, default=0) + 1}"


def resolve_out(out: Path | None, *, root: Path = SCENARIOS_ROOT) -> Path:
    """Use ``out`` when given, otherwise the next unused ``vN`` under ``root``."""
    if out is not None:
        return out
    return next_dataset_dir(root)


def load_plan(path: Path) -> Plan:
    """Load and validate the cell plan YAML at ``path``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as missing:
        raise GenerateError(
            f"{path} is missing. Author the cell plan (gabarito plus seed "
            f"persona, goal and script) before generating v1"
        ) from missing
    try:
        return Plan.model_validate(yaml.safe_load(text))
    except ValidationError as invalid:
        raise GenerateError(
            f"{path} does not match the cell-plan schema:\n{invalid}"
        ) from invalid


def check_unanswerable_coverage(plan: Plan, unanswerable_ids: Sequence[str]) -> None:
    """Fail unless each unanswerable ID seeds exactly one adversarial slot.

    ``unanswerable_id`` is authoring metadata on the plan (T-06). It does not
    enter ``Scenario`` or the generated JSONL. The taxonomy asks every
    question of unanswerable.json to seed at least one adversarial scenario;
    this dataset maps each ID once.
    """
    expected = list(unanswerable_ids)
    seeded: list[str] = []
    for cell in plan.cells:
        for slot in cell.slots:
            if slot.unanswerable_id is None:
                continue
            if cell.category != "adversarial":
                raise GenerateError(
                    f"unanswerable_id {slot.unanswerable_id} is on a "
                    f"{cell.category} slot; unanswerable questions seed "
                    f"adversarial scenarios"
                )
            seeded.append(slot.unanswerable_id)
    counts = Counter(seeded)
    missing = [uid for uid in expected if counts.get(uid, 0) == 0]
    repeated = [uid for uid in expected if counts.get(uid, 0) > 1]
    unknown = [uid for uid in counts if uid not in expected]
    if not (missing or repeated or unknown):
        return
    raise GenerateError(
        f"unanswerable coverage is not 1:1 with unanswerable.json: "
        f"missing {missing}, repeated {repeated}, unknown {unknown}. "
        f"Each Uxx seeds exactly one adversarial slot"
    )


def expand_plan(plan: Plan, n: int) -> list[Scenario]:
    """Turn the plan into scenarios, first block per category, extras last.

    ``n`` is the experiment size: 48 keeps four slots per cell, 60 appends the
    extra of each cell after that category's first block.
    """
    per_cell = _slots_per_cell(n)
    intents = list(dict.fromkeys(cell.intent for cell in plan.cells))
    cells = {(cell.intent, cell.category): cell for cell in plan.cells}
    scenarios: list[Scenario] = []
    for category in CATEGORIES:
        present = [intent for intent in intents if (intent, category) in cells]
        if not present:
            continue
        index = 1
        for intent in present:
            for slot in cells[(intent, category)].slots[:FIRST_BLOCK_QUOTA]:
                scenarios.append(_to_scenario(category, intent, index, slot))
                index += 1
        if per_cell > FIRST_BLOCK_QUOTA:
            for intent in present:
                cell = cells[(intent, category)]
                if len(cell.slots) < per_cell:
                    raise GenerateError(
                        f"cell {intent}/{category} has {len(cell.slots)} slots; "
                        f"n={n} needs {per_cell} (the extra is last)"
                    )
                scenarios.append(
                    _to_scenario(category, intent, index, cell.slots[FIRST_BLOCK_QUOTA])
                )
                index += 1
    return scenarios


def _slots_per_cell(n: int) -> int:
    """Map the experiment N onto how many slots of each cell to emit."""
    if n == FIRST_BLOCK_N:
        return FIRST_BLOCK_QUOTA
    if n == FULL_SET_N:
        return FULL_SET_QUOTA
    raise GenerateError(
        f"n must be {FIRST_BLOCK_N} (first block) or {FULL_SET_N} (with extras), "
        f"not {n}"
    )


def _to_scenario(category: Category, intent: str, index: int, slot: Slot) -> Scenario:
    """Fill a Scenario from one slot; the ID is the category plus a two-digit n."""
    return Scenario(
        id=f"{category}_{index:02d}",
        category=category,
        intent=intent,
        user_persona=slot.user_persona,
        user_goal=slot.user_goal,
        script=slot.script,
        reference_answer=slot.reference_answer,
        required_facts=slot.required_facts,
        forbidden_facts=slot.forbidden_facts,
        expected_final_state=slot.expected_final_state,
        success_criterion=slot.success_criterion,
        max_turns=slot.max_turns,
        is_needle=slot.is_needle,
        canary=slot.canary,
    )


class ScenarioPhrasing(BaseModel):
    """The user-side wording the judge may rewrite; never the gabarito."""

    model_config = ConfigDict(extra="forbid")

    user_persona: str = Field(min_length=1)
    user_goal: str = Field(min_length=1)
    script: list[str] = Field(min_length=1)


def phrase_slot[T: (Slot, Scenario)](
    row: T,
    *,
    intent: str,
    category: Category,
    llm: Chat,
    seed: int,
    prompt: Prompt,
) -> T:
    """Replace persona, goal and script with a judge-phrased variant of ``row``.

    Gabarito fields are copied through unchanged. Adversarial rows are not
    sent here: the caller skips them so attacks stay as authored.
    """
    phrasing = _request_phrasing(
        intent=intent,
        category=category,
        user_persona=row.user_persona,
        user_goal=row.user_goal,
        script=row.script,
        llm=llm,
        seed=seed,
        prompt=prompt,
    )
    return row.model_copy(
        update={
            "user_persona": phrasing.user_persona,
            "user_goal": phrasing.user_goal,
            "script": phrasing.script,
        }
    )


def _request_phrasing(
    *,
    intent: str,
    category: Category,
    user_persona: str,
    user_goal: str,
    script: list[str],
    llm: Chat,
    seed: int,
    prompt: Prompt,
) -> ScenarioPhrasing:
    """Call the judge once for a schema-valid persona, goal and script."""
    rendered = prompt.render(
        intent=intent,
        category=category,
        user_persona=user_persona,
        user_goal=user_goal,
        script=render_script(script),
    )
    response = llm.chat(
        [{"role": "system", "content": rendered}],
        role="judge",
        caller="scenario_phrasing",
        schema=ScenarioPhrasing,
        seed=seed,
    )
    phrasing = response.parsed
    if not isinstance(phrasing, ScenarioPhrasing):
        raise GenerateError(
            "the phrasing call did not return a ScenarioPhrasing object"
        )
    return phrasing


def generate_from_plan(
    plan: Plan,
    n: int,
    *,
    phrasing: Literal["seed", "llm"] = "seed",
    llm: Chat | None = None,
    seed: int = 42,
    prompt: Prompt | None = None,
    unanswerable_ids: Sequence[str] | None = None,
) -> list[Scenario]:
    """Expand ``plan`` to ``n`` scenarios, optionally rephrasing non-attacks."""
    if unanswerable_ids is not None:
        check_unanswerable_coverage(plan, unanswerable_ids)
    scenarios = expand_plan(plan, n)
    if phrasing == "seed":
        return scenarios
    if llm is None or prompt is None:
        raise GenerateError(
            "phrasing=llm needs an LLM client and the scenario_phrasing prompt"
        )
    phrased: list[Scenario] = []
    for scenario in scenarios:
        if scenario.category == "adversarial":
            phrased.append(scenario)
            continue
        phrased.append(
            phrase_slot(
                scenario,
                intent=scenario.intent,
                category=scenario.category,
                llm=llm,
                seed=seed,
                prompt=prompt,
            )
        )
    return phrased


def write_dataset(scenarios: Sequence[Scenario], directory: Path) -> str:
    """Write one JSONL per category under ``directory`` and return its hash."""
    existing = sorted(directory.glob("*.jsonl")) if directory.exists() else []
    if existing:
        names = ", ".join(path.name for path in existing)
        raise GenerateError(
            f"{directory} already has {names}. Frozen datasets are not "
            f"overwritten; omit --out to write the next vN, or pass a new "
            f"directory"
        )
    grouped: dict[Category, list[Scenario]] = {category: [] for category in CATEGORIES}
    for scenario in scenarios:
        grouped[scenario.category].append(scenario)
    for category, rows in grouped.items():
        if not rows:
            continue
        text = "".join(row.model_dump_json() + "\n" for row in rows)
        atomic_write(directory / f"{category}.jsonl", text)
    return hash_dataset(directory)


def main(argv: Sequence[str] | None = None) -> int:
    """Expand the cell plan into the next unused ``data/scenarios/vN/``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", type=Path, default=Path("data/scenarios/plan.yaml"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="dataset directory (default: next unused data/scenarios/vN)",
    )
    parser.add_argument(
        "--n", type=int, default=FULL_SET_N, choices=(FIRST_BLOCK_N, FULL_SET_N)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phrasing",
        choices=("seed", "llm"),
        default="seed",
        help=(
            "seed uses the plan wording; llm asks the judge to vary persona and script"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="must be the configured judge (default); the agents' model is refused",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    config = load_models_config(args.config)
    try:
        resolve_model(config, args.model)
        plan = load_plan(args.plan)
        kb = load_kb()
        prompt: Prompt | None = None
        llm: Chat | None = None
        if args.phrasing == "llm":
            prompt = load_prompt("scenario_phrasing")
            llm = LlmClient(
                config,
                cache_dir=Path("runs") / "generate_scenarios" / "cache",
                log_path=Path("runs") / "generate_scenarios" / LLM_CALLS_LOG,
            )
        scenarios = generate_from_plan(
            plan,
            args.n,
            phrasing=args.phrasing,
            llm=llm,
            seed=args.seed,
            prompt=prompt,
            unanswerable_ids=[question.id for question in kb.unanswerable],
        )
        out = resolve_out(args.out)
        digest = write_dataset(scenarios, out)
    except GenerateError as failure:
        print(f"generate-scenarios: {failure}", file=sys.stderr)
        return 1
    print(f"wrote {len(scenarios)} scenarios to {out} hash {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
