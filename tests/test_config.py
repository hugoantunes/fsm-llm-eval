"""Tests for the loader of configs/models.yaml (T-07)."""

from pathlib import Path

import pytest

from helpers import CONFIG, MINIMAL_MODELS_YAML, write_models_config
from sim.config import ConfigError, load_models_config


def test_loads_the_real_models_yaml() -> None:
    config = load_models_config(CONFIG)

    assert config.num_ctx == 8192
    assert config.max_prompt_fraction == 0.8
    assert config.models.agent.name
    assert config.models.simulator.name
    assert config.models.judge.name
    assert config.models.agent.temperature is not None
    assert config.models.agent.seed is not None
    # The simulated user of T-11 samples like the agents: nothing is left to
    # Ollama's default, or a rerun would not reproduce the same customer.
    assert config.models.simulator.temperature == config.models.agent.temperature
    assert config.models.simulator.seed is not None


def test_rejects_an_unknown_key(tmp_path: Path) -> None:
    misspelled = MINIMAL_MODELS_YAML.replace("    temperature: 0.7", "    temp: 0.7")
    path = write_models_config(tmp_path, misspelled)

    with pytest.raises(ConfigError, match="temp"):
        load_models_config(path)


def test_rejects_a_missing_role(tmp_path: Path) -> None:
    without_judge = MINIMAL_MODELS_YAML.split("  judge:")[0]
    path = write_models_config(tmp_path, without_judge)

    with pytest.raises(ConfigError, match="judge"):
        load_models_config(path)


def test_the_prompt_budget_is_the_configured_fraction_of_num_ctx(
    tmp_path: Path,
) -> None:
    config = load_models_config(write_models_config(tmp_path))

    assert config.max_prompt_tokens == 6553


def test_spec_returns_the_model_of_each_role(tmp_path: Path) -> None:
    config = load_models_config(write_models_config(tmp_path))

    assert config.spec("agent").name == "agent-model"
    assert config.spec("judge").seed == 42
