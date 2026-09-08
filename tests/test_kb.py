"""Tests for the knowledge base loader and the real KB files (T-01)."""

import re
from pathlib import Path

import pytest

from sim.kb import KnowledgeBaseError, load_kb

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

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = REPO_ROOT / "data" / "kb"

#: Brands, people and companies that must never appear in a fictional domain.
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

#: Character ceiling for the KB markdown, roughly 3 000 tokens. The real token
#: counter arrives with T-07; until then this guards T-09's criterion that the
#: baseline's full-KB prompt still fits num_ctx 8192 with 8 turns of history.
MAX_KB_CHARS = 12_000


def test_real_kb_has_30_to_40_facts_covering_the_four_intents() -> None:
    kb = load_kb(KB_DIR)

    assert 30 <= len(kb.facts) <= 40
    assert {
        "order_tracking",
        "exchange_return",
        "cancellation",
        "payment_reissue",
    } <= {fact.intent for fact in kb.facts}


def test_real_kb_declares_five_to_eight_needles_and_some_unanswerable_questions() -> (
    None
):
    kb = load_kb(KB_DIR)

    assert 5 <= len(kb.needles) <= 8
    assert len(kb.unanswerable) >= 8


def test_real_kb_names_no_real_brand_or_person() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(KB_DIR.iterdir())
    ).lower()

    found = [name for name in FORBIDDEN_NAMES if re.search(rf"\b{name}\b", text)]

    assert found == []


def test_real_kb_markdown_stays_within_the_prompt_budget() -> None:
    markdown = (KB_DIR / "knowledge_base.md").read_text(encoding="utf-8")

    assert len(markdown) <= MAX_KB_CHARS
