"""Shared test helpers.

The fakes, the synthetic files and the plain functions a test calls itself. What
a test receives as an argument is a fixture of ``conftest.py`` instead.
"""

import hashlib
import importlib.util
import re
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel

from sim.config import Role
from sim.kb import Fact, KnowledgeBase, Needle, UserDataField
from sim.llm import LlmResponse, Message
from sim.user import UserReply, UserStatus

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The real data and config the tests load instead of copying: whatever the
#: experiment runs on is what they check.
KB_DIR = REPO_ROOT / "data" / "kb"
FSM_DIR = REPO_ROOT / "data" / "fsm"
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
EXAMPLES_DIR = REPO_ROOT / "data" / "scenarios" / "examples"
CONFIG = REPO_ROOT / "configs" / "models.yaml"
DOCS_DIR = REPO_ROOT / "docs"

#: Brands, people and companies that must never appear in a fictional domain.
#: The knowledge base and the scenarios are both checked against this list, so
#: it is written once (T-01, T-05).
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
        path.read_text(encoding="utf-8") for path in sorted(directory.iterdir())
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
        self, replies: Sequence[str] | None = None, *, latency_s: float = 0.5
    ) -> None:
        self.replies = list(replies) if replies is not None else ["Hello, I can help."]
        self.latency_s = latency_s
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
        text = self.replies.pop(0)
        return LlmResponse(
            text=text,
            parsed=None if schema is None else schema.model_validate_json(text),
            model=f"{role}-model",
            caller=caller,
            prompt_hash=prompt_hash,
            prompt_tokens=len(str(sent)) // 4,
            output_tokens=len(text) // 4,
            latency_s=self.latency_s,
            cached=False,
        )


def user_reply(message: str, status: UserStatus = "continue") -> str:
    """Render one schema-valid answer from the simulated user of T-11."""
    return UserReply(message=message, status=status).model_dump_json()


def load_script(relative_path: str) -> ModuleType:
    """Import a script that lives outside any package, by repo-relative path."""
    file = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(file.stem, file)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
