"""Blind two-call LLM judge (T-12).

The judge never sees agent metadata: a clean transcript of user and agent turns,
the scenario's evaluation fields, and the knowledge base. Two independent chats
per dialogue, each constrained by a JSON Schema; numeric P/R/F1 are not asked
of the model.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from sim.kb import KnowledgeBase, render_facts
from sim.llm import Chat, LlmError
from sim.metrics import JudgeFacts, JudgeGlobal
from sim.prompts import DEFAULT_PROMPTS_DIR, load_prompt
from sim.schemas import Scenario, Turn, render_script, render_transcript

FACTS_PROMPT = "judge_facts"
GLOBAL_PROMPT = "judge_global"
SHARED_PROMPT = "judge_shared"

#: Extra attempts when pydantic refuses a payload JSON Schema cannot express
#: (`fact_id` only on ``yes``). Transport retries stay in T-07.
SCHEMA_RETRIES = 2

_Parsed = TypeVar("_Parsed", bound=BaseModel)


class JudgeError(RuntimeError):
    """The judge cannot grade this dialogue; the message says why."""


class JudgeResult(BaseModel):
    """The two schema-validated judge calls for one dialogue."""

    model_config = ConfigDict(extra="forbid")

    facts: JudgeFacts
    global_judgement: JudgeGlobal


class Judge:
    """The facts-and-claims call and the global-judgement call, in that order."""

    def __init__(
        self,
        llm: Chat,
        *,
        kb: KnowledgeBase,
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
        seed: int | None = None,
    ) -> None:
        self._llm = llm
        self._kb = kb
        self._seed = seed
        self._shared = load_prompt(SHARED_PROMPT, directory=prompts_dir).template
        self._facts = load_prompt(FACTS_PROMPT, directory=prompts_dir)
        self._global = load_prompt(GLOBAL_PROMPT, directory=prompts_dir)

    def evaluate(self, transcript: Sequence[Turn], scenario: Scenario) -> JudgeResult:
        """Grade ``transcript`` against ``scenario`` in two schema-constrained calls."""
        if not transcript:
            raise JudgeError(
                "the judge was asked to grade an empty transcript. A dialogue "
                "with no turns has nothing to score; skip failed logs in T-14b "
                "instead of sending silence to the model"
            )
        rendered = _redact(render_transcript(transcript), scenario.canary)
        facts = self._chat(
            self._facts.render(
                shared=self._shared,
                transcript=rendered,
                knowledge_base=render_facts(self._kb.facts),
                required_facts=", ".join(scenario.required_facts),
                needle_fact=_needle_placeholder(scenario, self._kb),
            ),
            caller="judge_facts",
            schema=JudgeFacts,
        )
        global_judgement = self._chat(
            self._global.render(
                shared=self._shared,
                transcript=rendered,
                script=_redact(render_script(scenario.script), scenario.canary),
                reference_answer=_redact(scenario.reference_answer, scenario.canary),
                success_criterion=_redact(scenario.success_criterion, scenario.canary),
            ),
            caller="judge_global",
            schema=JudgeGlobal,
        )
        return JudgeResult(facts=facts, global_judgement=global_judgement)

    def _chat(self, prompt: str, *, caller: str, schema: type[_Parsed]) -> _Parsed:
        """Send one judge call and return the parsed object, never raw JSON.

        A ``fact_id`` on a ``no`` claim is valid JSON Schema (Ollama will emit
        it) and invalid pydantic. Retry with a bumped seed; do not score the
        raw text. Transport failures are retried inside the client instead.
        """
        last_error: BaseException | None = None
        for attempt in range(SCHEMA_RETRIES + 1):
            seed = None if self._seed is None else self._seed + attempt
            try:
                answer = self._llm.chat(
                    [{"role": "system", "content": prompt}],
                    role="judge",
                    caller=caller,
                    schema=schema,
                    seed=seed,
                )
            except ValidationError as error:
                last_error = error
                continue
            except LlmError as error:
                if not isinstance(error.__cause__, ValidationError):
                    raise
                last_error = error
                continue
            parsed = answer.parsed
            if isinstance(parsed, schema):
                return parsed
            last_error = JudgeError(
                f"{caller} returned no parsed {schema.__name__}. The client of "
                "T-07 validates against the schema; scoring the raw JSON would "
                "let a fact_id on a no claim into fact_scores"
            )
        assert last_error is not None
        raise last_error


def _redact(text: str, canary: str | None) -> str:
    """Strip the canary token so injection scoring stays a T-13 literal match."""
    if canary is None:
        return text
    return text.replace(canary, "[redacted]")


def _needle_placeholder(scenario: Scenario, kb: KnowledgeBase) -> str:
    """Return the needle ID or ``none``, never ``str(is_needle)``."""
    if not scenario.is_needle:
        return "none"
    found = sorted(
        {needle.fact_id for needle in kb.needles} & set(scenario.required_facts)
    )
    if len(found) != 1:
        raise JudgeError(
            f"{scenario.id} is a needle scenario but required_facts intersect "
            f"needles is {found or 'empty'}. needle_recovered is one boolean, "
            f"so the placeholder is exactly one ID"
        )
    return found[0]
