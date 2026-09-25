"""Tests for the Ollama client of T-07: parameters, schema, budget, log, cache."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
from ollama import ResponseError
from pydantic import BaseModel

from helpers import MINIMAL_MODELS_YAML, FakeOllama, reply, write_models_config
from sim.config import load_models_config
from sim.llm import LlmCallError, LlmClient, LlmError, PromptTooLongError


class Event(BaseModel):
    """A schema with an enum, the shape the classifier of T-08 will use."""

    event: Literal["order_identified", "intent_classified", "none"]


def read_log(tmp_path: Path) -> list[dict[str, Any]]:
    """Return the records the client appended to its JSONL log."""
    lines = (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def make_client(
    tmp_path: Path,
    transport: FakeOllama,
    *,
    slept: list[float] | None = None,
    config_text: str = MINIMAL_MODELS_YAML,
) -> LlmClient:
    """Build a client on the synthetic config, writing under ``tmp_path``.

    ``slept`` collects the backoff delays instead of waiting them out.
    """
    config = load_models_config(write_models_config(tmp_path, config_text))
    return LlmClient(
        config,
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "llm_calls.jsonl",
        transport=transport,
        sleep=(slept if slept is None else slept.append),
    )


def _with_agent_digest(digest: str) -> str:
    """Return the synthetic config with a recorded digest on the agent model."""
    return MINIMAL_MODELS_YAML.replace(
        '    name: "agent-model"\n',
        f'    name: "agent-model"\n    digest: "{digest}"\n',
        1,
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


@pytest.mark.parametrize(
    ("capabilities", "think"),
    [(["completion", "thinking"], False), (["completion"], None)],
    ids=["a thinking model", "a model without the capability"],
)
def test_think_false_is_sent_only_to_a_model_that_supports_thinking(
    tmp_path: Path, capabilities: list[str], think: bool | None
) -> None:
    transport = FakeOllama(capabilities=capabilities)
    client = make_client(tmp_path, transport)

    client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    assert transport.calls[0]["think"] is think


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
        role="classifier",
        caller="classifier",
        schema=Event,
    )

    assert transport.calls[0]["format"] == Event.model_json_schema()
    assert isinstance(response.parsed, Event)
    assert response.parsed.event == "order_identified"


def test_a_missing_prompt_eval_count_raises(tmp_path: Path) -> None:
    transport = FakeOllama([reply("hello", prompt_tokens=None)])
    client = make_client(tmp_path, transport)

    with pytest.raises(LlmError, match="prompt_eval_count"):
        client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    (record,) = read_log(tmp_path)
    assert "prompt_eval_count" in record["error"]
    assert list((tmp_path / "cache").rglob("*.json")) == []


#: Global 300 s / 2 retries, with the judge override of the real models.yaml.
_JUDGE_TIMEOUT_YAML = MINIMAL_MODELS_YAML.replace(
    "  timeout_s: 300\n  retries: 2\n  backoff_s: 2\n",
    (
        "  timeout_s: 300\n"
        "  retries: 2\n"
        "  backoff_s: 2\n"
        "  by_role:\n"
        "    judge:\n"
        "      timeout_s: 600\n"
        "      retries: 1\n"
    ),
)


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


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection refused"),
        ResponseError("server busy", 503),
    ],
    ids=["connection-refused", "server-busy"],
)
def test_a_failing_capability_probe_is_retried_wrapped_and_logged(
    tmp_path: Path, failure: Exception
) -> None:
    transport = FakeOllama(show_errors=[failure, failure, failure])
    slept: list[float] = []
    client = make_client(tmp_path, transport, slept=slept)

    with pytest.raises(LlmCallError, match="failed after 3 attempt") as caught:
        client.chat([{"role": "user", "content": "hi"}], role="agent", caller="agent")

    assert str(failure) in str(caught.value)
    assert transport.calls == []
    assert transport.shown == ["agent-model"] * 3
    assert slept == [2, 4]
    (record,) = read_log(tmp_path)
    assert str(failure) in record["error"]
    assert record["attempts"] == 3


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


def test_judge_role_stops_after_one_retry(tmp_path: Path) -> None:
    timed_out = httpx.ReadTimeout("timed out")
    transport = FakeOllama([timed_out, timed_out, reply("late")])
    slept: list[float] = []
    client = make_client(
        tmp_path, transport, slept=slept, config_text=_JUDGE_TIMEOUT_YAML
    )

    with pytest.raises(LlmCallError, match="failed after 2 attempt") as caught:
        client.chat(
            [{"role": "user", "content": "grade this"}],
            role="judge",
            caller="judge_facts",
        )

    assert caught.value.attempts == 2
    assert len(transport.calls) == 2
    assert slept == [2]
    (record,) = read_log(tmp_path)
    assert record["attempts"] == 2


def test_agent_role_keeps_three_attempts_when_the_judge_is_overridden(
    tmp_path: Path,
) -> None:
    timed_out = httpx.ReadTimeout("timed out")
    transport = FakeOllama([timed_out, timed_out, reply("late")])
    client = make_client(tmp_path, transport, slept=[], config_text=_JUDGE_TIMEOUT_YAML)

    response = client.chat(
        [{"role": "user", "content": "hi"}], role="agent", caller="agent"
    )

    assert response.text == "late"
    assert len(transport.calls) == 3
    (record,) = read_log(tmp_path)
    assert record["attempts"] == 3


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
            role="classifier",
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


def test_a_replayed_call_records_zero_attempts(tmp_path: Path) -> None:
    transport = FakeOllama([httpx.ConnectError("connection refused"), reply("late")])
    client = make_client(tmp_path, transport, slept=[])
    messages = [{"role": "user", "content": "hi"}]

    client.chat(messages, role="agent", caller="agent")
    client.chat(messages, role="agent", caller="agent")

    miss, hit = read_log(tmp_path)
    assert miss["cached"] is False
    assert miss["attempts"] == 2
    assert hit["cached"] is True
    assert hit["attempts"] == 0


def test_transport_config_is_outside_the_cache_key(tmp_path: Path) -> None:
    transport = FakeOllama([reply("cached")])
    messages = [{"role": "user", "content": "where is my order?"}]
    first = make_client(tmp_path, transport)
    first.chat(messages, role="agent", caller="agent")
    other_transport = MINIMAL_MODELS_YAML.replace(
        "  timeout_s: 300\n  retries: 2\n  backoff_s: 2\n",
        (
            "  timeout_s: 600\n"
            "  retries: 0\n"
            "  backoff_s: 8\n"
            "  by_role:\n"
            "    judge:\n"
            "      timeout_s: 900\n"
            "      retries: 1\n"
            "      backoff_s: 16\n"
        ),
    )
    second = make_client(tmp_path, transport, config_text=other_transport)

    response = second.chat(messages, role="agent", caller="agent")

    assert len(transport.calls) == 1
    assert response.cached is True
    miss, hit = read_log(tmp_path)
    assert miss["prompt_hash"] == hit["prompt_hash"]


def test_the_cache_key_separates_the_same_tag_under_two_digests(
    tmp_path: Path,
) -> None:
    transport = FakeOllama([reply("old weights"), reply("new weights")])
    messages = [{"role": "user", "content": "where is my order?"}]
    first = make_client(tmp_path, transport, config_text=_with_agent_digest("aaa"))
    first.chat(messages, role="agent", caller="agent")
    second = make_client(tmp_path, transport, config_text=_with_agent_digest("bbb"))

    response = second.chat(messages, role="agent", caller="agent")

    assert len(transport.calls) == 2
    assert response.text == "new weights"
    assert response.cached is False
    records = read_log(tmp_path)
    assert records[0]["prompt_hash"] != records[1]["prompt_hash"]
    assert records[0]["digest"] == "aaa"
    assert records[1]["digest"] == "bbb"


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

    client.chat(messages, role="classifier", caller="classifier", schema=Event)
    second = client.chat(messages, role="classifier", caller="classifier", schema=Event)

    assert isinstance(second.parsed, Event)
    assert second.parsed.event == "intent_classified"


def test_one_jsonl_record_per_call_carries_the_tokens_latency_and_caller(
    tmp_path: Path,
) -> None:
    transport = FakeOllama([reply("hi there", prompt_tokens=120, output_tokens=7)])
    client = make_client(tmp_path, transport)

    client.chat(
        [{"role": "user", "content": "hi"}], role="classifier", caller="classifier"
    )

    (record,) = read_log(tmp_path)
    assert record["caller"] == "classifier"
    assert record["role"] == "classifier"
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


@pytest.mark.integration
def test_a_real_call_returns_text_and_a_prompt_eval_count(
    real_client: LlmClient, tmp_path: Path
) -> None:
    response = real_client.chat(
        [{"role": "user", "content": "Say hello in three words."}],
        role="simulated_user",
        caller="probe",
        num_predict=20,
    )

    assert response.text.strip()
    assert response.prompt_tokens > 0
    assert response.latency_s > 0
    (record,) = read_log(tmp_path)
    assert record["prompt_tokens"] == response.prompt_tokens


@pytest.mark.integration
def test_a_real_schema_call_returns_a_valid_enum_member(
    real_client: LlmClient,
) -> None:
    response = real_client.chat(
        [
            {
                "role": "user",
                "content": (
                    "The customer wrote: 'my order is NL-12345678'. "
                    "Classify what the customer just did."
                ),
            }
        ],
        role="classifier",
        caller="probe",
        schema=Event,
        num_predict=40,
    )

    assert isinstance(response.parsed, Event)


@pytest.mark.integration
def test_a_real_reply_carries_no_reasoning_text(real_client: LlmClient) -> None:
    response = real_client.chat(
        [{"role": "user", "content": "How much is 17 times 23? Answer with a number."}],
        role="simulated_user",
        caller="probe",
        num_predict=80,
    )

    assert "<think>" not in response.text
    assert "</think>" not in response.text
