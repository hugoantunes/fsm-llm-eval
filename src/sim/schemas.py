"""Load and cross-check the scenarios of ``data/scenarios/`` (T-05).

A scenario is one test case of the experiment: the brief the simulated user of
T-11 plays, and the answer key the judge of T-12 and the deterministic
evaluators of T-13 grade against. It is also the unit of analysis, so its ``id``
is what pairs the two agents in T-19.
"""

import json
import re
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sim.fsm import FsmSpec
from sim.kb import KnowledgeBase

#: The three kinds of test case, from the taxonomy of ``docs/taxonomy.md``.
Category = Literal["happy_path", "edge", "adversarial"]

#: The same three as values: the columns of the quota matrix and the strata the
#: analysis of T-19 repeats itself over.
CATEGORIES: tuple[Category, ...] = get_args(Category)

#: The turns a dialogue may be given. Fewer than four cannot reach a solution
#: through greeting, identification and data collection; the machine budget of
#: T-03 is drawn for eight, and twelve is where a dialogue is going nowhere.
MIN_TURNS = 4
MAX_TURNS = 12


class ScenarioError(ValueError):
    """A scenario is malformed; the message says what to fix and where."""


class Turn(BaseModel):
    """One line of a dialogue, by whoever said it.

    The speakers are named rather than given Ollama's chat roles: the agent and
    the simulated user of T-11 read the same history from opposite sides, and each
    maps it to ``user`` and ``assistant`` from its own point of view.
    """

    model_config = ConfigDict(extra="forbid")

    speaker: Literal["user", "agent"]
    text: str


class TurnRecord(BaseModel):
    """What one agent turn produced: the reply and what the metrics need (T-09).

    Both agents fill the same record, which is what makes them comparable;
    ``state_before``, ``state_after`` and ``event`` are the FSM bookkeeping of
    T-10 and stay ``None`` for the baseline, whose path is reconstructed by the
    stage labeler of T-13 instead.

    The prompt itself is not stored here: ``prompt_hash`` addresses it in the
    ``llm_calls.jsonl`` of T-07, which is the same key its cache uses, so the text
    lives in exactly one place and ``runs/`` stays small enough to rsync (T-17).
    """

    model_config = ConfigDict(extra="forbid")

    turn: int
    user_message: str
    agent_reply: str
    model: str
    prompt_hash: str
    prompt_tokens: int
    output_tokens: int
    #: Seconds inside the agent's own LLM call. T-13 reports it as the agent's
    #: latency, and it only means anything when ``cached`` is false.
    llm_latency_s: float
    #: Seconds for the whole turn: the LLM call plus everything around it, which
    #: for the FSM agent includes the extra classifier call of T-08.
    turn_latency_s: float
    cached: bool = False
    state_before: str | None = None
    state_after: str | None = None
    event: str | None = None


class Scenario(BaseModel):
    """One scenario: the user's brief, the answer key and the success test.

    The simulated user sees ``user_persona``, ``user_goal`` and ``script`` and
    nothing else (T-11); every other field belongs to the evaluation side and
    would leak the answer into the dialogue.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    category: Category
    intent: str
    user_persona: str
    user_goal: str
    script: list[str]
    reference_answer: str
    required_facts: list[str]
    forbidden_facts: list[str] = []
    expected_final_state: str
    success_criterion: str
    max_turns: int = Field(ge=MIN_TURNS, le=MAX_TURNS)
    is_needle: bool = False
    canary: str | None = None

    @model_validator(mode="after")
    def _id_names_the_category(self) -> "Scenario":
        """Fail unless the ID is the category plus a two-digit number."""
        if re.fullmatch(rf"{self.category}_\d{{2}}", self.id):
            return self
        raise ValueError(
            f"the id {self.id!r} does not name the category {self.category!r}: "
            f"an id reads '{self.category}_07'. The id carries the category into "
            f"the per-category analysis of T-19, so the two cannot disagree"
        )

    @model_validator(mode="after")
    def _answer_key_does_not_contradict_itself(self) -> "Scenario":
        """Fail unless the required and the forbidden facts are disjoint."""
        both = sorted(set(self.required_facts) & set(self.forbidden_facts))
        if not both:
            return self
        raise ValueError(
            f"{both} are required and forbidden at once. The judge measures recall "
            f"on the required facts and the policy check fires on the forbidden "
            f"ones (T-12, T-13), so such a scenario cannot be passed"
        )

    @model_validator(mode="after")
    def _canary_belongs_to_an_attack(self) -> "Scenario":
        """Fail unless the canary is an adversarial scenario's planted token."""
        if self.canary is None or self.category == "adversarial":
            return self
        raise ValueError(
            f"a {self.category} scenario carries the canary {self.canary!r}. The "
            f"token is how T-13 sees an injection succeed, so it belongs to the "
            f"attack that asks for it and nowhere else"
        )

    @model_validator(mode="after")
    def _canary_is_planted_by_the_script(self) -> "Scenario":
        """Fail unless the script asks the agent for the canary, word for word."""
        if self.canary is None or any(self.canary in beat for beat in self.script):
            return self
        raise ValueError(
            f"no beat of the script asks for the canary {self.canary!r}. An attack "
            f"the simulated user never delivers can never succeed, and the "
            f"scenario would report perfect safety for free (T-13)"
        )

    @model_validator(mode="after")
    def _canary_stays_out_of_the_reference_answer(self) -> "Scenario":
        """Fail unless the answer key is free of the canary."""
        if self.canary is None or self.canary not in self.reference_answer:
            return self
        raise ValueError(
            f"the reference answer contains the canary {self.canary!r}. A correct "
            f"dialogue never repeats the token, so keeping it out of every "
            f"evaluated text is what makes an occurrence mean a successful attack"
        )


def load_scenarios(path: Path, *, kb: KnowledgeBase, fsm: FsmSpec) -> list[Scenario]:
    """Load and cross-check every scenario under ``path``.

    ``path`` is a directory of ``.jsonl`` files, read in file-name order, one
    scenario per line.
    """
    scenarios: list[Scenario] = []
    seen: dict[str, str] = {}
    for file in sorted(path.glob("*.jsonl")):
        for number, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            where = f"{file.name}, line {number}"
            entry = _parse(line, file, number)
            _check_unique(entry, where, seen)
            _check_intent(entry, where, kb)
            _check_facts(entry, where, kb)
            _check_final_state(entry, where, fsm)
            _check_needle(entry, where, kb)
            scenarios.append(entry)
            seen[entry.id] = where
    return scenarios


def _check_intent(entry: Scenario, where: str, kb: KnowledgeBase) -> None:
    """Fail unless the scenario addresses one of the knowledge base's intents."""
    intents = kb.intents()
    if entry.intent in intents:
        return
    raise ScenarioError(
        f"{where}: the intent {entry.intent!r} is not one of {intents}. The FSM "
        f"agent releases the facts of the classified intent by that name (T-02), "
        f"and 'general' is not one: its facts are released in every state"
    )


def _check_facts(entry: Scenario, where: str, kb: KnowledgeBase) -> None:
    """Fail unless every fact the scenario names is a fact of the knowledge base."""
    known = kb.fact_ids()
    for fact_id in entry.required_facts + entry.forbidden_facts:
        if fact_id in known:
            continue
        raise ScenarioError(
            f"{where}: {entry.id} names {fact_id}, which is not a fact of "
            f"knowledge_base.md. Recall and fidelity are measured by ID (T-12), "
            f"so an ID nobody can state would score the scenario as failed"
        )


def _check_final_state(entry: Scenario, where: str, fsm: FsmSpec) -> None:
    """Fail unless the dialogue is expected to end where it legitimately can."""
    if entry.expected_final_state in fsm.accepting_states:
        return
    raise ScenarioError(
        f"{where}: {entry.id} expects to end in "
        f"{entry.expected_final_state!r}, which is not one of the accepting states "
        f"{fsm.accepting_states} of machine.yaml. Flow adherence asks whether the "
        f"dialogue ended there (T-13), and no dialogue ends mid-flow on purpose"
    )


def _check_needle(entry: Scenario, where: str, kb: KnowledgeBase) -> None:
    """Fail unless a needle scenario requires a fact that is a needle."""
    if not entry.is_needle:
        return
    needles = {needle.fact_id for needle in kb.needles}
    if needles & set(entry.required_facts):
        return
    raise ScenarioError(
        f"{where}: {entry.id} is flagged is_needle but requires none of the "
        f"needle facts {sorted(needles)} of needles.json. The needle metric asks "
        f"whether the specific fact was recovered (T-04), so it needs one"
    )


def _check_unique(entry: Scenario, where: str, seen: dict[str, str]) -> None:
    """Fail unless ``entry`` carries an ID no earlier scenario has used."""
    if entry.id not in seen:
        return
    raise ScenarioError(
        f"{where}: the id {entry.id!r} is already used by {seen[entry.id]}. The ID "
        f"is what pairs the two agents on the same scenario in T-19, so a repeat "
        f"would silently drop one of the two from the analysis"
    )


def _parse(line: str, file: Path, number: int) -> Scenario:
    """Parse one JSONL line into a ``Scenario``, or say where it is wrong."""
    try:
        return Scenario.model_validate(json.loads(line))
    except ValidationError as invalid:
        raise ScenarioError(
            f"{file.name}, line {number}: does not match the schema of "
            f"sim.schemas:\n{invalid}\nA key the schema does not know is refused "
            f"rather than ignored: a misspelled 'required_facts' would leave the "
            f"answer key empty and every scenario would score a perfect recall"
        ) from invalid
