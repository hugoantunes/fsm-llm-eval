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
