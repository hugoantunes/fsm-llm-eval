"""Tests for the knowledge base loader and the real KB files (T-01)."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from helpers import (
    FSM_DIR,
    KB_DIR,
    PROMPTS_DIR,
    SCENARIOS_DIR,
    forbidden_names_in,
)
from sim.kb import (
    Fact,
    KnowledgeBase,
    KnowledgeBaseError,
    Needle,
    UnanswerableQuestion,
    load_kb,
)

MARKDOWN = """\
# Knowledge base - Test Store

## General policies · `general`

- **F01** — Support answers between 9:00 and 18:00.

## Order tracking · `order_tracking`

- **F02** — Standard delivery takes 5 business days.
"""

NEEDLES = '{"version": 1, "needles": []}'
UNANSWERABLE = '{"version": 1, "unanswerable": []}'
USER_DATA_FIELDS = '{"version": 1, "fields": []}'


def write_kb(
    directory: Path,
    markdown: str = MARKDOWN,
    needles: str = NEEDLES,
    unanswerable: str = UNANSWERABLE,
    user_data_fields: str = USER_DATA_FIELDS,
) -> Path:
    """Write a synthetic KB into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "knowledge_base.md").write_text(markdown, encoding="utf-8")
    (directory / "needles.json").write_text(needles, encoding="utf-8")
    (directory / "unanswerable.json").write_text(unanswerable, encoding="utf-8")
    (directory / "user_data_fields.json").write_text(user_data_fields, encoding="utf-8")
    return directory


def test_load_kb_parses_id_intent_and_text_of_each_fact(tmp_path: Path) -> None:
    directory = write_kb(tmp_path / "kb")

    kb = load_kb(directory)

    assert [(f.id, f.intent, f.text) for f in kb.facts] == [
        ("F01", "general", "Support answers between 9:00 and 18:00."),
        ("F02", "order_tracking", "Standard delivery takes 5 business days."),
    ]


def test_duplicate_fact_id_raises(tmp_path: Path) -> None:
    markdown = MARKDOWN.replace("**F02**", "**F01**")
    directory = write_kb(tmp_path / "kb", markdown=markdown)

    with pytest.raises(KnowledgeBaseError, match="F01"):
        load_kb(directory)


def test_gap_in_fact_numbering_raises(tmp_path: Path) -> None:
    markdown = MARKDOWN.replace("**F02**", "**F03**")
    directory = write_kb(tmp_path / "kb", markdown=markdown)

    with pytest.raises(KnowledgeBaseError, match="F02"):
        load_kb(directory)


def test_fact_ids_may_appear_out_of_file_order(tmp_path: Path) -> None:
    markdown = """\
# Knowledge base - Test Store

## General policies · `general`

- **F01** — Support answers between 9:00 and 18:00.
- **F03** — The agent states only what the knowledge base contains.

## Order tracking · `order_tracking`

- **F02** — Standard delivery takes 5 business days.
"""
    directory = write_kb(tmp_path / "kb", markdown=markdown)

    kb = load_kb(directory)

    assert [fact.id for fact in kb.facts] == ["F01", "F03", "F02"]


def test_a_repeated_section_heading_raises(tmp_path: Path) -> None:
    markdown = MARKDOWN + "\n## General policies · `general`\n\n- **F03** — Extra.\n"
    directory = write_kb(tmp_path / "kb", markdown=markdown)

    with pytest.raises(KnowledgeBaseError, match="general"):
        load_kb(directory)


def test_section_heading_without_intent_slug_raises(tmp_path: Path) -> None:
    markdown = MARKDOWN.replace(
        "## General policies · `general`", "## General policies"
    )
    directory = write_kb(tmp_path / "kb", markdown=markdown)

    with pytest.raises(KnowledgeBaseError, match="F01"):
        load_kb(directory)


def test_needle_referencing_an_unknown_fact_raises(tmp_path: Path) -> None:
    needles = (
        '{"version": 1, "needles": [{"fact_id": "F41", "why": "specific", '
        '"probe_question": "?"}]}'
    )
    directory = write_kb(tmp_path / "kb", needles=needles)

    with pytest.raises(KnowledgeBaseError, match="F41"):
        load_kb(directory)


def test_unanswerable_questions_with_a_repeated_id_raise(tmp_path: Path) -> None:
    entry = (
        '{"id": "U01", "question": "?", "intent": "general", '
        '"why_unanswerable": "not covered", "expected_behavior": "escalate"}'
    )
    unanswerable = f'{{"version": 1, "unanswerable": [{entry}, {entry}]}}'
    directory = write_kb(tmp_path / "kb", unanswerable=unanswerable)

    with pytest.raises(KnowledgeBaseError, match="U01"):
        load_kb(directory)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (Fact, {"id": "F01", "intent": "general", "text": "a fact", "junk": 1}),
        (
            Needle,
            {
                "fact_id": "F01",
                "why": "specific",
                "probe_question": "?",
                "junk": 1,
            },
        ),
        (
            UnanswerableQuestion,
            {
                "id": "U01",
                "question": "?",
                "intent": "general",
                "why_unanswerable": "not covered",
                "expected_behavior": "escalate",
                "junk": 1,
            },
        ),
        (
            KnowledgeBase,
            {
                "facts": [],
                "needles": [],
                "unanswerable": [],
                "user_data_fields": [],
                "junk": 1,
            },
        ),
    ],
    ids=["Fact", "Needle", "UnanswerableQuestion", "KnowledgeBase"],
)
def test_kb_models_refuse_a_key_the_schema_does_not_know(
    model: type[BaseModel], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="junk"):
        model.model_validate(payload)


def test_unknown_key_in_a_user_data_field_file_raises(tmp_path: Path) -> None:
    fields = (
        '{"version": 1, "fields": [{"key": "order_number", "label": "Order number", '
        '"patern": "\\\\bNL-\\\\d{8}\\\\b", "example": "NL-20260145", '
        '"required_for": ["order_tracking"]}]}'
    )
    directory = write_kb(tmp_path / "kb", user_data_fields=fields)

    with pytest.raises(KnowledgeBaseError, match="patern"):
        load_kb(directory)


def test_user_data_field_pattern_that_misses_its_own_example_raises(
    tmp_path: Path,
) -> None:
    fields = (
        '{"version": 1, "fields": [{"key": "order_number", "label": "Order number", '
        '"pattern": "\\\\bNL-\\\\d{8}\\\\b", "example": "NL-1", '
        '"required_for": ["order_tracking"]}]}'
    )
    directory = write_kb(tmp_path / "kb", user_data_fields=fields)

    with pytest.raises(KnowledgeBaseError, match="order_number"):
        load_kb(directory)


def test_user_data_field_requiring_an_unknown_intent_raises(tmp_path: Path) -> None:
    fields = (
        '{"version": 1, "fields": [{"key": "email", "label": "E-mail", '
        '"pattern": "\\\\S+@\\\\S+", "example": "a@b.com", '
        '"required_for": ["gift_wrapping"]}]}'
    )
    directory = write_kb(tmp_path / "kb", user_data_fields=fields)

    with pytest.raises(KnowledgeBaseError, match="gift_wrapping"):
        load_kb(directory)


# --- the real KB of data/kb/ ------------------------------------------------


def test_real_kb_has_30_to_40_facts_covering_the_four_intents(
    real_kb: KnowledgeBase,
) -> None:
    assert 30 <= len(real_kb.facts) <= 40
    assert {
        "order_tracking",
        "exchange_return",
        "cancellation",
        "payment_reissue",
    } <= {fact.intent for fact in real_kb.facts}


def test_real_kb_declares_five_to_eight_needles_and_some_unanswerable_questions(
    real_kb: KnowledgeBase,
) -> None:
    assert 5 <= len(real_kb.needles) <= 8
    assert len(real_kb.unanswerable) >= 8


def test_forbidden_names_in_reads_nested_files(tmp_path: Path) -> None:
    (tmp_path / "states").mkdir()
    (tmp_path / "machine.yaml").write_text("initial: greeting\n", encoding="utf-8")
    (tmp_path / "states" / "greeting.md").write_text(
        "Contact amazon support.\n", encoding="utf-8"
    )

    assert forbidden_names_in(tmp_path) == ["amazon"]


@pytest.mark.parametrize(
    "directory",
    [KB_DIR, SCENARIOS_DIR, PROMPTS_DIR, FSM_DIR],
    ids=["kb", "scenarios", "prompts", "fsm"],
)
def test_the_experiment_names_no_real_brand_or_person(directory: Path) -> None:
    assert forbidden_names_in(directory) == []
