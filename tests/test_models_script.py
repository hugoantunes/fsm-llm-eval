"""Tests for scripts/models.py: the digest comparison behind ``just verify-models``."""

from pathlib import Path

import pytest

from helpers import MINIMAL_MODELS_YAML, load_script, write_models_config
from sim.config import ConfigError

models = load_script("scripts/models.py")

CONFIG = {
    "agent": {"name": "qwen:1b", "digest": "a" * 64},
    "judge": {"name": "gemma:1b", "digest": "b" * 64},
}


def test_configured_models_refuses_a_key_the_schema_does_not_know(
    tmp_path: Path,
) -> None:
    misspelled = MINIMAL_MODELS_YAML.replace("    temperature: 0.7", "    temp: 0.7")

    with pytest.raises(ConfigError, match="temp"):
        models.configured_models(write_models_config(tmp_path, misspelled))


def test_matching_digests_report_no_problems() -> None:
    local = {"qwen:1b": "a" * 64, "gemma:1b": "b" * 64}

    assert models.compare(CONFIG, local) == []


def test_model_not_pulled_is_reported() -> None:
    local = {"qwen:1b": "a" * 64}

    assert models.compare(CONFIG, local) == ["judge: gemma:1b is not pulled"]


def test_differing_digest_is_reported() -> None:
    local = {"qwen:1b": "a" * 64, "gemma:1b": "c" * 64}

    problems = models.compare(CONFIG, local)

    assert len(problems) == 1
    assert problems[0].startswith("judge: gemma:1b digest differs")


def test_missing_recorded_digest_is_reported() -> None:
    config = {"agent": {"name": "qwen:1b", "digest": None}}

    problems = models.compare(config, {"qwen:1b": "a" * 64})

    assert problems == [
        f"agent: qwen:1b has no digest in the config (local {'a' * 12})"
    ]


def test_missing_name_is_reported() -> None:
    config = {"agent": {"name": None, "digest": None}}

    assert models.compare(config, {}) == ["agent: no model name in the config"]


def test_extra_loaded_models_are_those_other_than_the_judge() -> None:
    resident = ["qwen3.5:4b", "gemma4:12b", "qwen3.5:4b"]

    assert models.extra_loaded_models(resident, "gemma4:12b") == ["qwen3.5:4b"]


def test_no_extra_loaded_models_when_only_the_judge_is_resident() -> None:
    assert models.extra_loaded_models(["gemma4:12b"], "gemma4:12b") == []


def test_extra_loaded_warning_names_the_resident_models() -> None:
    text = models.extra_loaded_warning(["qwen3.5:9b"])

    assert "qwen3.5:9b" in text
    assert "`ollama stop qwen3.5:9b`" in text
    assert "qwen3.5:4b" not in text


class _FakeWarmupClient:
    """Record the chat kwargs warmup_judge would send to Ollama."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def test_warmup_judge_uses_the_configured_name_and_num_ctx(tmp_path: Path) -> None:
    path = write_models_config(tmp_path)
    client = _FakeWarmupClient()

    code = models.warmup_judge(path, client=client)

    assert code == 0
    (call,) = client.calls
    assert call["model"] == "judge-model"
    assert call["options"] == {
        "num_ctx": 8192,
        "num_predict": 1,
        "temperature": 0,
    }
    assert call["think"] is False


def test_warmup_judge_writes_no_cache_files_and_no_llm_log(tmp_path: Path) -> None:
    path = write_models_config(tmp_path)
    client = _FakeWarmupClient()

    models.warmup_judge(path, client=client)

    assert list(tmp_path.rglob("llm_calls.jsonl")) == []
    assert list(tmp_path.rglob("cache")) == []
    assert list(tmp_path.rglob("*.json")) == []
