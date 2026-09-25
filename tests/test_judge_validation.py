"""Tests for docs/judge_validation.md (T-16)."""

from helpers import CONFIG
from sim.config import load_models_config


def test_validation_doc_names_the_configured_judge(judge_validation_doc: str) -> None:
    config = load_models_config(CONFIG)

    assert config.models.judge.name in judge_validation_doc
