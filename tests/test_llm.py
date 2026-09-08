"""Tests for the Ollama client of T-07: parameters, schema, budget, log, cache."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
from ollama import ChatResponse, ResponseError, ShowResponse
from ollama._types import Message
from pydantic import BaseModel

from helpers import REPO_ROOT, write_models_config
from sim.config import load_models_config
from sim.llm import LlmCallError, LlmClient, LlmError, PromptTooLongError


class Event(BaseModel):
    """A schema with an enum, the shape the classifier of T-08 will use."""

    event: Literal["order_identified", "intent_classified", "none"]


def reply(
    text: str = "hello", *, prompt_tokens: int = 100, output_tokens: int = 5
) -> ChatResponse:
    """Build the response the real Ollama would return for one chat call."""
    return ChatResponse(
        model="agent-model",
        message=Message(role="assistant", content=text),
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
    ) -> None:
        self.replies = replies if replies is not None else [reply()]
        self.capabilities = (
            ["completion", "thinking"] if capabilities is None else capabilities
        )
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
        return ShowResponse(model_info={}, capabilities=self.capabilities)


def read_log(tmp_path: Path) -> list[dict[str, Any]]:
    """Return the records the client appended to its JSONL log."""
    lines = (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def make_client(
    tmp_path: Path, transport: FakeOllama, *, slept: list[float] | None = None
) -> LlmClient:
    """Build a client on the synthetic config, writing under ``tmp_path``.

    ``slept`` collects the backoff delays instead of waiting them out.
    """
    config = load_models_config(write_models_config(tmp_path))
    return LlmClient(
        config,
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm_calls.jsonl",
        transport=transport,
        sleep=(slept if slept is None else slept.append),
    )


def test_chat_sends_the_configured_model_num_ctx_temperature_and_seed(
    tmp_path: Path,
) -> None:
    transport = FakeOllama()
    client = make_client(tmp_path, transport)

    client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    sent = transport.calls[0]
    assert sent["model"] == "agent-model"
    assert sent["options"]["num_ctx"] == 8192
    assert sent["options"]["temperature"] == 0.7
    assert sent["options"]["seed"] == 42


def test_chat_sends_think_false_when_the_model_supports_thinking(
    tmp_path: Path,
) -> None:
    transport = FakeOllama(capabilities=["completion", "thinking"])
    client = make_client(tmp_path, transport)

    client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    assert transport.calls[0]["think"] is False


def test_chat_omits_think_when_the_model_does_not_support_it(tmp_path: Path) -> None:
    transport = FakeOllama(capabilities=["completion"])
    client = make_client(tmp_path, transport)

    client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    assert transport.calls[0]["think"] is None


def test_the_capability_probe_runs_once_per_model(tmp_path: Path) -> None:
    transport = FakeOllama([reply("one"), reply("two")])
    client = make_client(tmp_path, transport)

    client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")
    client.chat([{"role": "user", "content": "again"}], role="agent", caller="agent")

    assert transport.shown == ["agent-model"]


def test_a_pydantic_schema_becomes_format_and_comes_back_parsed(
    tmp_path: Path,
) -> None:
    transport = FakeOllama([reply('{"event": "order_identified"}')])
    client = make_client(tmp_path, transport)

    response = client.chat(
        [{"role": "user", "content": "my order is NL-12345678"}],
        role="simulator",
        caller="classifier",
        schema=Event,
    )

    assert transport.calls[0]["format"] == Event.model_json_schema()
    assert isinstance(response.parsed, Event)
    assert response.parsed.event == "order_identified"


def test_prompt_over_eighty_percent_of_num_ctx_raises(tmp_path: Path) -> None:
    transport = FakeOllama([reply("hello", prompt_tokens=7000)])
    client = make_client(tmp_path, transport)

    with pytest.raises(PromptTooLongError, match="7000 of 8192"):
        client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")


def test_a_transport_error_is_retried_and_then_succeeds(tmp_path: Path) -> None:
    transport = FakeOllama([httpx.ConnectError("connection refused"), reply("late")])
    client = make_client(tmp_path, transport, slept=[])

    response = client.chat(
        [{"role": "user", "content": "hi"}], role="agent", caller="agent"
    )

    assert response.text == "late"
    assert len(transport.calls) == 2
    (record,) = read_log(tmp_path)
    assert record["attempts"] == 2


def test_exhausted_retries_raise_with_the_last_error(tmp_path: Path) -> None:
    refused = [httpx.ConnectError("connection refused") for _ in range(3)]
    transport = FakeOllama(list(refused))
    slept: list[float] = []
    client = make_client(tmp_path, transport, slept=slept)

    with pytest.raises(LlmCallError, match="connection refused"):
        client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    assert len(transport.calls) == 3
    assert slept == [2, 4]
    (record,) = read_log(tmp_path)
    assert "connection refused" in record["error"]


def test_a_missing_model_is_not_retried_and_records_one_attempt(
    tmp_path: Path,
) -> None:
    transport = FakeOllama([ResponseError("model not found", 404), reply("second")])
    slept: list[float] = []
    client = make_client(tmp_path, transport, slept=slept)

    with pytest.raises(LlmCallError, match="model not found"):
        client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    assert len(transport.calls) == 1
    assert slept == []
    (record,) = read_log(tmp_path)
    assert record["attempts"] == 1


def test_a_prompt_too_long_error_is_not_retried(tmp_path: Path) -> None:
    transport = FakeOllama([reply("hello", prompt_tokens=7000), reply("second")])
    client = make_client(tmp_path, transport, slept=[])

    with pytest.raises(PromptTooLongError):
        client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    assert len(transport.calls) == 1


def test_concurrent_calls_write_whole_jsonl_lines(tmp_path: Path) -> None:
    long_answer = "x" * 20_000
    transport = FakeOllama([reply(long_answer) for _ in range(20)])
    client = make_client(tmp_path, transport)

    def ask(number: int) -> None:
        client.chat(
            [{"role": "user", "content": f"question {number}"}],
            role="agent",
            caller="agent",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(ask, range(20)))

    records = read_log(tmp_path)
    assert len(records) == 20
    assert all(record["text"] == long_answer for record in records)


def test_the_refused_call_is_still_logged(tmp_path: Path) -> None:
    transport = FakeOllama([reply("hello", prompt_tokens=7000)])
    client = make_client(tmp_path, transport)

    with pytest.raises(PromptTooLongError):
        client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    (record,) = read_log(tmp_path)
    assert record["prompt_tokens"] == 7000
    assert "7000 of 8192" in record["error"]


def test_a_reply_that_does_not_match_the_schema_raises(tmp_path: Path) -> None:
    transport = FakeOllama([reply('{"event": "sang_a_song"}')])
    client = make_client(tmp_path, transport)

    with pytest.raises(LlmError, match="Event"):
        client.chat(
            [{"role": "user", "content": "hi"}],
            role="simulator",
            caller="classifier",
            schema=Event,
        )


def test_an_identical_call_hits_the_cache_instead_of_the_transport(
    tmp_path: Path,
) -> None:
    transport = FakeOllama([reply("the same answer")])
    client = make_client(tmp_path, transport)
    messages = [{"role": "user", "content": "where is my order?"}]

    first = client.chat(messages, role="agent", caller="agent")
    second = client.chat(messages, role="agent", caller="agent")

    assert len(transport.calls) == 1
    assert second.text == first.text == "the same answer"
    assert first.cached is False
    assert second.cached is True


def test_the_cache_key_separates_calls_that_differ_by_seed_messages_or_schema(
    tmp_path: Path,
) -> None:
    transport = FakeOllama([reply("a"), reply("b"), reply('{"event": "none"}')])
    client = make_client(tmp_path, transport)
    messages = [{"role": "user", "content": "where is my order?"}]

    client.chat(messages, role="agent", caller="agent")
    client.chat(messages, role="agent", caller="agent", seed=7)
    client.chat(messages, role="agent", caller="agent", schema=Event)

    hashes = {record["prompt_hash"] for record in read_log(tmp_path)}
    assert len(transport.calls) == 3
    assert len(hashes) == 3


def test_a_cached_answer_is_parsed_again_through_the_schema(tmp_path: Path) -> None:
    transport = FakeOllama([reply('{"event": "intent_classified"}')])
    client = make_client(tmp_path, transport)
    messages = [{"role": "user", "content": "I want to return a lamp"}]

    client.chat(messages, role="simulator", caller="classifier", schema=Event)
    second = client.chat(messages, role="simulator", caller="classifier", schema=Event)

    assert isinstance(second.parsed, Event)
    assert second.parsed.event == "intent_classified"


def test_one_jsonl_record_per_call_carries_the_tokens_latency_and_caller(
    tmp_path: Path,
) -> None:
    transport = FakeOllama([reply("hi there", prompt_tokens=120, output_tokens=7)])
    client = make_client(tmp_path, transport)

    client.chat(
        [{"role": "user", "content": "hi"}], role="simulator", caller="classifier"
    )

    (record,) = read_log(tmp_path)
    assert record["caller"] == "classifier"
    assert record["role"] == "simulator"
    assert record["model"] == "small-model"
    assert record["messages"] == [{"role": "user", "content": "hi"}]
    assert record["text"] == "hi there"
    assert record["prompt_tokens"] == 120
    assert record["output_tokens"] == 7
    assert record["latency_s"] >= 0
    assert record["cached"] is False
    assert record["error"] is None
    assert record["prompt_hash"]


# --- Integration: a real Ollama with the models of configs/models.yaml -------


def real_client(tmp_path: Path) -> LlmClient:
    """Build a client on the real config and the real Ollama server."""
    config = load_models_config(REPO_ROOT / "configs" / "models.yaml")
    return LlmClient(
        config,
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm_calls.jsonl",
    )


@pytest.mark.integration
def test_a_real_call_returns_text_and_a_prompt_eval_count(tmp_path: Path) -> None:
    client = real_client(tmp_path)

    response = client.chat(
        [{"role": "user", "content": "Say hello in three words."}],
        role="simulator",
        caller="probe",
        num_predict=20,
    )

    assert response.text.strip()
    assert response.prompt_tokens > 0
    assert response.latency_s > 0
    (record,) = read_log(tmp_path)
    assert record["prompt_tokens"] == response.prompt_tokens


@pytest.mark.integration
def test_a_real_schema_call_returns_a_valid_enum_member(tmp_path: Path) -> None:
    client = real_client(tmp_path)

    response = client.chat(
        [
            {
                "role": "user",
                "content": (
                    "The customer wrote: 'my order is NL-12345678'. "
                    "Classify what the customer just did."
                ),
            }
        ],
        role="simulator",
        caller="probe",
        schema=Event,
        num_predict=40,
    )

    assert isinstance(response.parsed, Event)


@pytest.mark.integration
def test_a_real_reply_carries_no_reasoning_text(tmp_path: Path) -> None:
    client = real_client(tmp_path)

    response = client.chat(
        [{"role": "user", "content": "How much is 17 times 23? Answer with a number."}],
        role="simulator",
        caller="probe",
        num_predict=80,
    )

    assert "<think>" not in response.text
    assert "</think>" not in response.text
