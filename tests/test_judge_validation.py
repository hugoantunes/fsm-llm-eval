"""Tests for docs/judge_validation.md (T-16)."""

from helpers import CONFIG
from sim.config import load_models_config


def test_validation_doc_voids_the_first_sample_not_the_path(
    judge_validation_doc: str,
) -> None:
    config = load_models_config(CONFIG)

    assert (
        "What is void is that draw, not the path `runs/exp_pilot`"
        in judge_validation_doc
    )
    assert "census of all 20 ok dialogues" in judge_validation_doc
    assert "30/76 eligible agent responses" in judge_validation_doc
    assert "sample touches" not in judge_validation_doc
    assert "Discard any sheet written from `runs/exp_pilot`" not in judge_validation_doc
    assert config.models.judge.name in judge_validation_doc
    assert "judge_facts.md v3" in judge_validation_doc
    assert "judge_global.md v2" in judge_validation_doc
    assert "## Results" in judge_validation_doc
