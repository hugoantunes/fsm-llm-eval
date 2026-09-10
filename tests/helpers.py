"""Shared test helpers.

The fakes, the synthetic files and the plain functions a test calls itself. What
a test receives as an argument is a fixture of ``conftest.py`` instead.
"""

import hashlib
import importlib.util
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from sim.agents import Agent
from sim.config import Role
from sim.kb import Fact, KnowledgeBase, Needle, UserDataField
from sim.llm import LlmResponse, Message
from sim.metrics import Accuracy, JudgeClaim, JudgeFacts, JudgeGlobal
from sim.schemas import Scenario, Turn, TurnRecord
from sim.user import UserReply, UserStatus

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The real data and config the tests load instead of copying: whatever the
#: experiment runs on is what they check.
KB_DIR = REPO_ROOT / "data" / "kb"
FSM_DIR = REPO_ROOT / "data" / "fsm"
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
SCENARIOS_DIR = REPO_ROOT / "data" / "scenarios"
EXAMPLES_DIR = SCENARIOS_DIR / "examples"
CONFIG = REPO_ROOT / "configs" / "models.yaml"
DOCS_DIR = REPO_ROOT / "docs"
LABELED_EVENTS = REPO_ROOT / "tests" / "fixtures" / "user_events.jsonl"
JUDGE_SANITY = REPO_ROOT / "tests" / "fixtures" / "judge_sanity.jsonl"

#: Brands, people and companies that must never appear in a fictional domain.
#: The knowledge base, the scenarios, the prompts and the FSM packages are all
#: checked against this list, so it is written once.
FORBIDDEN_NAMES = (
    "amazon",
    "shopify",
    "mercado livre",
    "magalu",
    "americanas",
    "nike",
    "adidas",
    "zara",
    "apple",
    "google",
    "microsoft",
    "netflix",
    "correios",
    "fedex",
    "dhl",
    "ups",
    "visa",
    "mastercard",
    "paypal",
    "pix",
)


def forbidden_names_in(directory: Path) -> list[str]:
    """Return the names of :data:`FORBIDDEN_NAMES` that appear under ``directory``."""
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ).lower()
    return [name for name in FORBIDDEN_NAMES if re.search(rf"\b{name}\b", text)]


#: The two facts every synthetic knowledge base in the tests starts from: one
#: general policy, released in every state, and one order-tracking fact. Written
#: here so the FSM, scenario and agent tests cannot disagree on what F01 says.
GENERAL_FACT = Fact(
    id="F01", intent="general", text="Support answers between 9:00 and 18:00."
)
TRACKING_FACT = Fact(
    id="F02", intent="order_tracking", text="Standard delivery takes 5 business days."
)


def make_kb(
    *facts: Fact,
    needles: Sequence[Needle] = (),
    user_data_fields: Sequence[UserDataField] = (),
) -> KnowledgeBase:
    """Build a knowledge base out of ``facts`` and whatever else a test needs.

    A hand-written KB of two or three facts is what the loaders, the FSM and the
    agents are tested against: small enough to assert on fact by fact, and
    independent of how ``data/kb/`` grows.
    """
    return KnowledgeBase(
        facts=list(facts),
        needles=list(needles),
        unanswerable=[],
        user_data_fields=list(user_data_fields),
    )


#: A synthetic ``configs/models.yaml``: the shape of the real one with names no
#: machine has to hold. The config loader and the LLM client both build on it.
MINIMAL_MODELS_YAML = """\
ollama:
  version: "0.33.3"
  env:
    OLLAMA_NUM_PARALLEL: 2
num_ctx: 8192
max_prompt_fraction: 0.8
client:
  timeout_s: 300
  retries: 2
  backoff_s: 2
models:
  agent:
    name: "agent-model"
    temperature: 0.7
    seed: 42
  simulator:
    name: "small-model"
  judge:
    name: "judge-model"
    temperature: 0
    seed: 42
"""


def write_models_config(directory: Path, text: str = MINIMAL_MODELS_YAML) -> Path:
    """Write a synthetic model config into ``directory`` and return its path."""
    path = directory / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class FakeLlm:
    """A fake of ``sim.llm.LlmClient``: records the calls, returns canned answers.

    This is the seam of T-07 (``sim.llm.Chat``), faked rather than mocked, and the
    answers are real ``LlmResponse`` objects so the fake cannot drift from the
    fields its callers read.
    """

    def __init__(
        self,
        replies: Sequence[str] | None = None,
        *,
        latency_s: float = 0.5,
        delay_s: float = 0.0,
    ) -> None:
        self.replies = list(replies) if replies is not None else ["Hello, I can help."]
        self.latency_s = latency_s
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Message],
        *,
        role: Role,
        caller: str,
        schema: type[BaseModel] | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> LlmResponse:
        """Record the call and hand back the next canned answer.

        A call constrained by a schema comes back parsed, as the real client
        returns it, so a canned answer that does not match the schema fails here
        rather than reaching the caller as an object nobody validated.
        """
        sent = [dict(message) for message in messages]
        prompt_hash = hashlib.sha256(str(sent).encode("utf-8")).hexdigest()
        self.calls.append(
            {
                "messages": sent,
                "prompt_hash": prompt_hash,
                "role": role,
                "caller": caller,
                "schema": schema,
                "seed": seed,
                "num_predict": num_predict,
            }
        )
        if not self.replies:
            raise AssertionError("FakeLlm ran out of replies")
        if self.delay_s:
            time.sleep(self.delay_s)
        text = self.replies.pop(0)
        return LlmResponse(
            text=text,
            parsed=None if schema is None else schema.model_validate_json(text),
            model=f"{role}-model",
            prompt_hash=prompt_hash,
            prompt_tokens=len(str(sent)) // 4,
            output_tokens=len(text) // 4,
            latency_s=self.latency_s,
            cached=False,
        )


def user_reply(message: str, status: UserStatus = "continue") -> str:
    """Render one schema-valid answer from the simulated user of T-11."""
    return UserReply(message=message, status=status).model_dump_json()


def classifier_reply(event: str, intent: str | None = None) -> str:
    """Render one schema-valid answer from the user-event classifier of T-08."""
    return json.dumps({"event": event, "intent": intent})


def judge_facts_reply(
    claims: Sequence[JudgeClaim] | None = None,
    *,
    needle_recovered: bool | None = None,
) -> str:
    """Render one schema-valid answer from the facts-and-claims judge call."""
    if claims is None:
        claims = [
            JudgeClaim(
                text="Standard delivery takes 5 business days.",
                fact_id="F02",
                supported_by_kb="yes",
            )
        ]
    return JudgeFacts(
        claims=list(claims), needle_recovered=needle_recovered
    ).model_dump_json()


def judge_global_reply(
    *,
    accuracy: Accuracy = "correct",
    accuracy_justification: str = "The assistant stated the delivery estimate.",
    relevance: int = 5,
    task_completed: bool = True,
    offensive_content: bool = False,
) -> str:
    """Render one schema-valid answer from the global-judgement judge call."""
    return JudgeGlobal(
        accuracy=accuracy,
        accuracy_justification=accuracy_justification,
        relevance=relevance,
        task_completed=task_completed,
        offensive_content=offensive_content,
    ).model_dump_json()


#: The two patterned data every intent requires, as already collected slots:
#: what the identification guard reads, so a test can park past it.
ORDER_AND_EMAIL = {"order_number": "NL-20260145", "email": "jane@example.com"}

#: One canned user turn that ends the dialogue after the agent's first reply.
CANNED_USER_TURN = "Hi, where is my order?"
CANNED_AGENT_REPLY = "Hello from support."


def canned_replies_for(agent: str, *, delay_s: float = 0.0) -> FakeLlm:
    """A one-turn fake: the customer reaches its goal on the opening message.

    The FSM agent also classifies that turn; the baseline does not. Each job
    of T-14a gets its own instance, so a thread pool cannot mix their queues.
    """
    replies: list[str] = [user_reply(CANNED_USER_TURN, status="goal_reached")]
    if agent == "fsm":
        replies.append(classifier_reply("none"))
    replies.append(CANNED_AGENT_REPLY)
    return FakeLlm(replies, delay_s=delay_s)


class HasAgent(Protocol):
    """The slice of a dialogue job the canned factory reads."""

    agent: str


def canned_llm_factory(job: HasAgent, *, delay_s: float = 0.0) -> FakeLlm:
    """Build a one-turn fake for ``job``; the factory the runner tests inject."""
    return canned_replies_for(job.agent, delay_s=delay_s)


def play_user_turns(agent: Agent, messages: Sequence[str]) -> list[TurnRecord]:
    """Play ``messages`` at ``agent``, one at a time, and return its turn records."""
    history: list[Turn] = []
    records: list[TurnRecord] = []
    for message in messages:
        history.append(Turn(speaker="user", text=message))
        record = agent.respond(history)
        history.append(Turn(speaker="agent", text=record.agent_reply))
        records.append(record)
    return records


def assert_prompt_carries_no_answer_key(prompt: str, scenario: Scenario) -> None:
    """Fail unless ``prompt`` contains no field of the scenario's answer key."""
    if scenario.canary is not None:
        assert scenario.canary not in prompt
    assert scenario.reference_answer not in prompt
    assert scenario.success_criterion not in prompt
    assert scenario.user_persona not in prompt
    assert scenario.user_goal not in prompt
    assert all(beat not in prompt for beat in scenario.script)


class LabeledEvent(BaseModel):
    """One gold utterance for the hybrid detector of T-08."""

    model_config = ConfigDict(extra="forbid")

    state: str
    text: str
    event: str
    via: Literal["rule", "llm"]
    intent: str | None = None


def load_labeled_events(path: Path = LABELED_EVENTS) -> list[LabeledEvent]:
    """Load the gold utterances of ``tests/fixtures/user_events.jsonl``."""
    rows: list[LabeledEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(LabeledEvent.model_validate(json.loads(line)))
    return rows


class SanityDialogue(BaseModel):
    """One hand-written transcript for the T-12 sanity cases and T-16 later."""

    model_config = ConfigDict(extra="forbid")

    id: str
    scenario_id: str
    accuracy: Accuracy
    planted_unsupported: str | None = None
    turns: list[Turn]


def load_judge_sanity(path: Path = JUDGE_SANITY) -> list[SanityDialogue]:
    """Load the six sanity transcripts of ``tests/fixtures/judge_sanity.jsonl``."""
    rows: list[SanityDialogue] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(SanityDialogue.model_validate(json.loads(line)))
    return rows


def labeled_utterance(
    events: Sequence[LabeledEvent],
    state: str,
    event: str,
    *,
    intent: str | None = None,
) -> LabeledEvent:
    """Return the first gold line for ``state`` / ``event`` / ``intent``."""
    matches = [
        row
        for row in events
        if row.state == state
        and row.event == event
        and (intent is None or row.intent == intent)
    ]
    if not matches:
        raise AssertionError(f"no gold utterance for {state}/{event}/{intent}")
    return matches[0]


def load_script(relative_path: str) -> ModuleType:
    """Import a script that lives outside any package, by repo-relative path."""
    file = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(file.stem, file)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
