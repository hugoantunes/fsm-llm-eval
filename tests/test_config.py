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
    assert config.models.simulated_user.name
    assert config.models.classifier.name
    assert config.models.state_labeler.name
    assert config.models.judge.name
    assert config.models.agent.temperature is not None
    assert config.models.agent.seed is not None


def test_split_small_roles_sample_with_the_same_default_temperature_as_the_agents() -> (
    None
):
    config = load_models_config(CONFIG)

    assert config.models.simulated_user.temperature == config.models.agent.temperature
    assert config.models.classifier.temperature == config.models.agent.temperature
    assert config.models.state_labeler.temperature == config.models.agent.temperature
    assert config.models.simulated_user.seed is not None
    assert config.models.classifier.seed is not None
    assert config.models.state_labeler.seed is not None


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
    assert config.spec("simulated_user").name == "small-model"
    assert config.spec("classifier").name == "small-model"
    assert config.spec("state_labeler").name == "small-model"
    assert config.spec("judge").seed == 42


def test_split_roles_can_be_configured_independently(tmp_path: Path) -> None:
    path = write_models_config(
        tmp_path,
        text=(
            "ollama:\n"
            '  version: "0.33.3"\n'
            "  env:\n"
            "    OLLAMA_NUM_PARALLEL: 2\n"
            "num_ctx: 8192\n"
            "max_prompt_fraction: 0.8\n"
            "client:\n"
            "  timeout_s: 300\n"
            "  retries: 2\n"
            "  backoff_s: 2\n"
            "models:\n"
            "  agent:\n"
            '    name: "agent-model"\n'
            "    temperature: 0.7\n"
            "    seed: 42\n"
            "  simulated_user:\n"
            '    name: "sim-user-model"\n'
            "  classifier:\n"
            '    name: "classifier-model"\n'
            "  state_labeler:\n"
            '    name: "labeler-model"\n'
            "  judge:\n"
            '    name: "judge-model"\n'
            "    temperature: 0\n"
            "    seed: 42\n"
        ),
    )
    config = load_models_config(path)

    assert config.spec("simulated_user").name == "sim-user-model"
    assert config.spec("classifier").name == "classifier-model"
    assert config.spec("state_labeler").name == "labeler-model"
