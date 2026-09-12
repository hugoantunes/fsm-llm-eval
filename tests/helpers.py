"""Shared test helpers.

The fakes, the synthetic files and the plain functions a test calls itself. What
a test receives as an argument is a fixture of ``conftest.py`` instead.
"""

import hashlib
import importlib.util
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

from ollama import ChatResponse, ShowResponse
from ollama._types import Message as OllamaMessage
from pydantic import BaseModel, ConfigDict

from sim.agents import Agent
from sim.config import Role
from sim.events import CLASSIFIER
from sim.kb import Fact, KnowledgeBase, Needle, UserDataField, load_kb
from sim.llm import LlmCallRecord, LlmResponse, Message
from sim.metrics import Accuracy, JudgeClaim, JudgeFacts, JudgeGlobal, judge_claim
from sim.schemas import (
    DialogueLog,
    DialogueStatus,
    Manifest,
    Scenario,
    StopReason,
    Turn,
    TurnRecord,
    dialogue_filename,
)
from sim.script import beats_of
from sim.user import SimulatedUser, UserReply, UserStatus

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The real data and config the tests load instead of copying: whatever the
#: experiment runs on is what they check.
KB_DIR = REPO_ROOT / "data" / "kb"
FSM_DIR = REPO_ROOT / "data" / "fsm"
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
SCENARIOS_DIR = REPO_ROOT / "data" / "scenarios"
EXAMPLES_DIR = SCENARIOS_DIR / "examples"
V1_DIR = SCENARIOS_DIR / "v1"
PLAN_PATH = SCENARIOS_DIR / "plan.yaml"
#: SHA-256 of ``data/scenarios/v1/*.jsonl`` (DECISOES.md 2026-09-10). ``just check``
#: fails if the files move; ``docs/taxonomy.md`` must quote the same digest.
FROZEN_V1_HASH = "0778a90110e6dd67685c1e6768934483cb4405213dad35ec172f3ddcf21d47c9"
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


def write_scenarios(directory: Path, **files: list[dict[str, Any]]) -> Path:
    """Write one JSONL per keyword into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, scenarios in files.items():
        lines = [json.dumps(entry) for entry in scenarios]
        (directory / f"{name}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return directory


def forbidden_names_in(directory: Path) -> list[str]:
    """Return the names of :data:`FORBIDDEN_NAMES` that appear under ``directory``."""
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ).lower()
    return [name for name in FORBIDDEN_NAMES if re.search(rf"\b{name}\b", text)]


#: The two facts every synthetic knowledge base in the tests starts from: one
#: general policy and one order-tracking fact. Written here so the FSM, scenario
#: and agent tests cannot disagree on what F01 says.
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


def reply(
    text: str = "hello",
    *,
    prompt_tokens: int | None = 100,
    output_tokens: int = 5,
) -> ChatResponse:
    """Build the response the real Ollama would return for one chat call."""
    return ChatResponse(
        model="agent-model",
        message=OllamaMessage(role="assistant", content=text),
        done=True,
        prompt_eval_count=prompt_tokens,
        eval_count=output_tokens,
    )


class FakeOllama:
    """Stand-in for ``ollama.Client``: records the calls, returns canned replies.

    The replies are real ``ChatResponse`` objects, so the fake cannot drift from
    the fields the client reads.
    """

    def __init__(
        self,
        replies: list[ChatResponse | Exception] | None = None,
        *,
        capabilities: list[str] | None = None,
        show_errors: list[Exception] | None = None,
    ) -> None:
        self.replies = replies if replies is not None else [reply()]
        self.capabilities = (
            ["completion", "thinking"] if capabilities is None else capabilities
        )
        self.show_errors = list(show_errors or [])
        self.calls: list[dict[str, Any]] = []
        self.shown: list[str] = []

    def chat(self, **kwargs: Any) -> ChatResponse:
        """Record the call and hand back the next canned reply."""
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("FakeOllama ran out of replies")
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def show(self, model: str) -> ShowResponse:
        """Record the capability probe and report the configured capabilities."""
        self.shown.append(model)
        if self.show_errors:
            raise self.show_errors.pop(0)
        return ShowResponse(model_info={}, capabilities=self.capabilities)


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
        if self.delay_s:
            time.sleep(self.delay_s)
        text = self._next(caller)
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

    def _next(self, caller: str) -> str:
        """Return the next canned answer, in the order they were given."""
        if not self.replies:
            raise AssertionError("FakeLlm ran out of replies")
        return self.replies.pop(0)


class CannedJobLlm(FakeLlm):
    """A fake that answers by caller instead of by call order.

    The event classifier of T-08 is rule-first: whether it reaches the model at
    all depends on the words the customer used, so a queue in call order
    desyncs as soon as a beat happens to name an order or an item. The customer
    still speaks from a queue, one message per beat.
    """

    def __init__(self, user_replies: Sequence[str], *, delay_s: float = 0.0) -> None:
        super().__init__(list(user_replies), delay_s=delay_s)

    def _next(self, caller: str) -> str:
        if caller == SimulatedUser.name:
            return super()._next(caller)
        if caller == CLASSIFIER:
            return classifier_reply("none")
        return CANNED_AGENT_REPLY


def real_data_fields() -> list[UserDataField]:
    """The patterned slots of ``data/kb/``, as the simulated user of T-11 reads them.

    What a script beat requires word for word is derived from these patterns, so
    a test hands over the real ones rather than a second copy of the regexes.
    """
    return load_kb(KB_DIR).user_data_fields


def user_reply(
    message: str, status: UserStatus = "continue", *, answering: bool = False
) -> str:
    """Render one schema-valid answer from the simulated user of T-11.

    ``answering`` defaults to false, as the field does: a customer turn carries
    the beat it owes unless it says it is answering the agent instead.
    """
    return UserReply(
        message=message, answering_agent_question=answering, status=status
    ).model_dump_json()


def classifier_reply(event: str, intent: str | None = None) -> str:
    """Render one schema-valid answer from the user-event classifier of T-08."""
    return json.dumps({"event": event, "intent": intent})


def invalid_classifier_reply() -> str:
    """Render ``intent_classified`` with no intent.

    JSON Schema allows the null; pydantic refuses it.
    """
    return classifier_reply("intent_classified")


def make_turn_record(
    turn: int = 1,
    *,
    user_message: str = "Where is my order?",
    agent_reply: str = "I can help with that.",
    llm_latency_s: float = 0.5,
    turn_latency_s: float = 0.8,
    cached: bool = False,
    state_before: str | None = None,
    state_after: str | None = None,
    event: str | None = None,
    user_beat: int | None = None,
) -> TurnRecord:
    """Build one turn record; only the fields a test cares about need saying."""
    return TurnRecord(
        turn=turn,
        user_message=user_message,
        user_beat=user_beat,
        agent_reply=agent_reply,
        model="agent-model",
        prompt_hash="a" * 64,
        prompt_tokens=10,
        output_tokens=4,
        llm_latency_s=llm_latency_s,
        turn_latency_s=turn_latency_s,
        cached=cached,
        state_before=state_before,
        state_after=state_after,
        event=event,
    )


def make_dialogue_log(
    records: Sequence[TurnRecord],
    *,
    scenario_id: str = "happy_path_01",
    agent: str = "baseline",
    repetition: int = 1,
    seed: int = 42,
    status: DialogueStatus = "ok",
    stop_reason: StopReason | None = "goal_reached",
) -> DialogueLog:
    """Build one dialogue log from turn records, for the evaluators of T-13."""
    return DialogueLog(
        scenario_id=scenario_id,
        agent=agent,
        repetition=repetition,
        seed=seed,
        status=status,
        stop_reason=stop_reason,
        records=list(records),
    )


def strip_fsm_meta(records: Sequence[TurnRecord]) -> list[TurnRecord]:
    """Copy ``records`` without FSM bookkeeping, as a baseline log stores them."""
    return [
        record.model_copy(
            update={
                "state_before": None,
                "state_after": None,
                "event": None,
                "transitions": [],
            }
        )
        for record in records
    ]


def stage_labels_reply(stages: Sequence[str]) -> str:
    """Render one schema-valid answer from the stage labeler of T-13."""
    return json.dumps({"stages": list(stages)})


def judge_facts_reply(
    claims: Sequence[JudgeClaim] | None = None,
    *,
    needle_recovered: bool | None = None,
) -> str:
    """Render one schema-valid answer from the facts-and-claims judge call."""
    if claims is None:
        claims = [
            judge_claim(
                text="Standard delivery takes 5 business days.",
                fact_id="F02",
                supported_by_kb="yes",
            )
        ]
    return JudgeFacts(
        claims=list(claims), needle_recovered=needle_recovered
    ).model_dump_json()


def invalid_judge_facts_reply() -> str:
    """Render a hand-built call-1 payload pydantic refuses, bypassing ``format=``.

    ``format=<schema>`` now forbids a ``fact_id`` on a ``no`` claim structurally
    (the ``oneOf`` on :class:`~sim.metrics.JudgeClaim`), so no real judge call can
    produce this. It stands in for one anyway, to exercise the ``chat_parsed``
    retry path a defective client or a stale cache entry could still hit.
    """
    return json.dumps(
        {
            "claims": [
                {
                    "text": "Shipping is always free.",
                    "fact_id": "F10",
                    "supported_by_kb": "no",
                }
            ],
            "needle_recovered": None,
        }
    )


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

#: The canned agent turn, one per beat of the script the job plays.
CANNED_AGENT_REPLY = "Hello from support."


class HasScenario(Protocol):
    """The slice of a dialogue job the canned factory reads."""

    agent: str
    scenario: Scenario


def canned_llm_factory(job: HasScenario, *, delay_s: float = 0.0) -> FakeLlm:
    """Build a canned fake for ``job``; the factory the runner tests inject.

    The script is a mandatory ordered plan (T-11), so the fake customer plays
    every beat and reaches its goal on the last one. Each message is the beat's
    own words, which is the one message every beat contract accepts, so a canned
    dialogue never spends a retry. A one-turn dialogue therefore needs a
    one-beat brief, which is what ``run_canned`` cuts them to.

    Each job of T-14a gets its own instance, so a thread pool cannot mix their
    queues.
    """
    beats = beats_of(job.scenario, real_data_fields())
    return CannedJobLlm(
        [
            user_reply(
                beat.text,
                "goal_reached" if beat.number == len(beats) else "continue",
            )
            for beat in beats
        ],
        delay_s=delay_s,
    )


RunCanned = Callable[..., Manifest]


class CannedEvalLlm:
    """A Chat that answers the judge and the stage labeler, and never runs out.

    Re-eval of the same run does not pop a queue dry. Stage labels default to
    one ``greeting`` per agent turn, matching the one-turn canned logs of T-14a.
    """

    def __init__(
        self,
        stages: Sequence[str] | None = None,
        *,
        needle_recovered: bool | None = False,
        invalid_facts_marker: str | None = None,
    ) -> None:
        self.stages = list(stages) if stages is not None else None
        self.needle_recovered = needle_recovered
        self.invalid_facts_marker = invalid_facts_marker
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
        """Record the call and return the canned answer for ``caller``."""
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
        text = self._reply(caller, sent)
        return LlmResponse(
            text=text,
            parsed=None if schema is None else schema.model_validate_json(text),
            model=f"{role}-model",
            prompt_hash=prompt_hash,
            prompt_tokens=len(str(sent)) // 4,
            output_tokens=len(text) // 4,
            latency_s=0.5,
            cached=False,
        )

    def _reply(self, caller: str, messages: list[dict[str, str]]) -> str:
        """Return the schema-valid payload the caller expects.

        The facts call answers with the payload pydantic refuses when the
        dialogue under the marker is the one being graded, so a test can single
        out one dialogue the judge never gets right.
        """
        if caller == "judge_facts":
            if self.invalid_facts_marker is not None and any(
                self.invalid_facts_marker in message["content"] for message in messages
            ):
                return invalid_judge_facts_reply()
            return judge_facts_reply(needle_recovered=self.needle_recovered)
        if caller == "judge_global":
            return judge_global_reply()
        if caller == "stage_labeler":
            labels = self.stages or ["greeting"] * _n_turns(messages)
            return stage_labels_reply(labels)
        raise AssertionError(f"unexpected eval caller {caller!r}")


def make_llm_call_record(
    *,
    caller: str = "baseline",
    role: str = "agent",
    prompt_tokens: int = 100,
    output_tokens: int = 10,
    latency_s: float = 1.0,
    cached: bool = False,
    prompt_hash: str = "aa",
) -> LlmCallRecord:
    """Build one ``llm_calls.jsonl`` row; only fields a test cares about need saying."""
    return LlmCallRecord(
        timestamp="2026-09-10T00:00:00+00:00",
        caller=caller,
        role=role,
        model=f"{role}-model",
        prompt_hash=prompt_hash,
        messages=[],
        text="{}",
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        latency_s=latency_s,
        cached=cached,
        attempts=0 if cached else 1,
    )


def write_llm_calls(run_dir: Path, records: Sequence[LlmCallRecord]) -> Path:
    """Write ``records`` as ``llm_calls.jsonl`` under ``run_dir`` and return it."""
    path = run_dir / "llm_calls.jsonl"
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def make_manifest(**fields: Any) -> Manifest:
    """Build a run manifest; only the fields a test cares about need saying."""
    values: dict[str, Any] = {
        "exp_id": "exp",
        "package_version": "0.1.0",
        "ollama_version": "0.33.3",
        "num_ctx": 8192,
        "dataset_hash": "a" * 64,
        "fsm_hash": "b" * 64,
        "scenarios_dir": "data/scenarios/v1",
        "prompt_versions": {"baseline": 2},
        "model_names": {"agent": "agent-model"},
        "model_digests": {"agent": None},
        "agents": ["baseline", "fsm"],
        "reps": 2,
        "parallel": 2,
        "seed_base": 42,
        "jobs": [],
        "elapsed_s": 20.0,
        "dialogues_per_hour": 360.0,
        "n_ok": 2,
        "n_failed": 0,
        "n_skipped": 0,
        "n_llm_calls": 0,
        "n_llm_cached": 0,
    }
    values.update(fields)
    return Manifest(**values)


def write_run(
    directory: Path,
    logs: Sequence[DialogueLog],
    *,
    scenarios_dir: str = "data/scenarios/v1",
) -> Path:
    """Write ``logs`` and a manifest as ``sim run`` would, and return the directory."""
    dialogues = directory / "dialogues"
    dialogues.mkdir(parents=True)
    for log in logs:
        (
            dialogues / dialogue_filename(log.scenario_id, log.agent, log.repetition)
        ).write_text(log.model_dump_json() + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(
        make_manifest(scenarios_dir=scenarios_dir).model_dump_json(),
        encoding="utf-8",
    )
    return directory


def canned_eval_transport_replies(n_dialogues: int) -> list[ChatResponse]:
    """One facts, global and labeler reply per dialogue, for an ``LlmClient`` fake."""
    cycle = (
        judge_facts_reply(needle_recovered=False),
        judge_global_reply(),
        stage_labels_reply(["greeting"]),
    )
    return [reply(text) for _ in range(n_dialogues) for text in cycle]


def _n_turns(messages: Sequence[Mapping[str, str]]) -> int:
    """Read the turn count the stage-labeler prompt declares."""
    content = messages[0]["content"]
    match = re.search(r"There are (\d+) turn", content)
    if match is None:
        raise AssertionError("stage labeler prompt has no turn count")
    return int(match.group(1))


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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
