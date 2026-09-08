"""Load the knowledge base of ``data/kb/`` (T-01).

``knowledge_base.md`` is the single source of the fact IDs (``F01``, ``F02``...):
scenarios, the judge and the policy checks address facts by those IDs, so the
markdown is parsed here once and everything else imports the result.
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel

DEFAULT_KB_DIR = Path("data/kb")

#: The section whose facts every state releases: the general policies, which are
#: not an intent a customer arrives with and which no scenario addresses (T-05).
GENERAL_INTENT = "general"

#: ``## Order tracking · `order_tracking` `` — human title plus the intent slug.
_SECTION = re.compile(r"^##\s+(?P<title>.+?)\s+·\s+`(?P<intent>[a-z_]+)`\s*$")
#: ``- **F14** — Items bought on promotion...`` — one atomic fact per line.
_FACT = re.compile(r"^-\s+\*\*(?P<id>F\d{2})\*\*\s+—\s+(?P<text>.+?)\s*$")


class KnowledgeBaseError(ValueError):
    """The KB files are inconsistent; the message says what to fix."""


class Fact(BaseModel):
    """One atomic statement the agents are allowed to make."""

    id: str
    intent: str
    text: str


class Needle(BaseModel):
    """A specific, non-obvious fact, and the question that probes it."""

    fact_id: str
    why: str
    probe_question: str


class UnanswerableQuestion(BaseModel):
    """A plausible question the KB does not answer: a hallucination trap."""

    id: str
    question: str
    intent: str
    why_unanswerable: str
    expected_behavior: str


class UserDataField(BaseModel):
    """A datum the agent has to collect from the user before solving.

    ``pattern`` is the regex that detects the datum in a user turn; it is
    ``None`` for free text (an item name, a reason), which no rule can spot.
    The rule-first event detection of T-08 reads these patterns.
    """

    key: str
    label: str
    pattern: str | None = None
    example: str
    required_for: list[str]


class KnowledgeBase(BaseModel):
    """Everything under ``data/kb/``, parsed and cross-checked."""

    facts: list[Fact]
    needles: list[Needle]
    unanswerable: list[UnanswerableQuestion]
    user_data_fields: list[UserDataField]

    def fact_ids(self) -> set[str]:
        """Return the IDs of every fact in the knowledge base."""
        return {fact.id for fact in self.facts}

    def intents(self) -> list[str]:
        """Return the intents a customer arrives with, in file order.

        ``general`` is not among them: those facts are released in every state
        and belong to no single request, so no scenario addresses them (T-05).
        """
        sections = (f.intent for f in self.facts if f.intent != GENERAL_INTENT)
        return list(dict.fromkeys(sections))

    def by_id(self, fact_id: str) -> Fact:
        """Return the fact with ``fact_id``, or raise ``KnowledgeBaseError``."""
        for fact in self.facts:
            if fact.id == fact_id:
                return fact
        raise KnowledgeBaseError(f"no fact {fact_id} in knowledge_base.md")


def parse_facts(markdown: str) -> list[Fact]:
    """Parse the facts of ``knowledge_base.md``, in file order."""
    facts: list[Fact] = []
    intent: str | None = None
    for line in markdown.splitlines():
        section = _SECTION.match(line)
        if section:
            intent = section["intent"]
            continue
        fact = _FACT.match(line)
        if not fact:
            continue
        if intent is None:
            raise KnowledgeBaseError(
                f"fact {fact['id']} appears before any '## Title · `intent`' heading"
            )
        facts.append(Fact(id=fact["id"], intent=intent, text=fact["text"]))
    _check_numbering(facts)
    return facts


def _check_numbering(facts: list[Fact]) -> None:
    """Fail unless the fact IDs are ``F01``, ``F02``... with no gap or repeat."""
    for position, fact in enumerate(facts, start=1):
        expected = f"F{position:02d}"
        if fact.id == expected:
            continue
        raise KnowledgeBaseError(
            f"fact IDs must run F01, F02... in order: expected {expected}, "
            f"found {fact.id}. Renumber the facts in knowledge_base.md"
        )


def load_kb(directory: Path = DEFAULT_KB_DIR) -> KnowledgeBase:
    """Load and cross-check the knowledge base stored in ``directory``."""
    markdown = (directory / "knowledge_base.md").read_text(encoding="utf-8")
    facts = parse_facts(markdown)
    needles = json.loads((directory / "needles.json").read_text(encoding="utf-8"))
    unanswerable = json.loads(
        (directory / "unanswerable.json").read_text(encoding="utf-8")
    )
    fields = json.loads(
        (directory / "user_data_fields.json").read_text(encoding="utf-8")
    )
    kb = KnowledgeBase(
        facts=facts,
        needles=needles["needles"],
        unanswerable=unanswerable["unanswerable"],
        user_data_fields=fields["fields"],
    )
    _check_needles(kb)
    _check_unanswerable(kb)
    _check_user_data_fields(kb)
    return kb


def _check_needles(kb: KnowledgeBase) -> None:
    """Fail unless every needle points at a fact of ``knowledge_base.md``."""
    known = kb.fact_ids()
    for needle in kb.needles:
        if needle.fact_id not in known:
            raise KnowledgeBaseError(
                f"needles.json references {needle.fact_id}, which is not in "
                f"knowledge_base.md"
            )


def _check_unanswerable(kb: KnowledgeBase) -> None:
    """Fail unless the unanswerable questions have distinct IDs."""
    seen: set[str] = set()
    for question in kb.unanswerable:
        if question.id in seen:
            raise KnowledgeBaseError(
                f"unanswerable.json repeats the id {question.id}; ids identify "
                f"the scenarios generated from them (T-06)"
            )
        seen.add(question.id)


def _check_user_data_fields(kb: KnowledgeBase) -> None:
    """Fail unless each field's pattern matches its example and its intents exist."""
    intents = {fact.intent for fact in kb.facts}
    for field in kb.user_data_fields:
        if field.pattern is not None and not re.search(field.pattern, field.example):
            raise KnowledgeBaseError(
                f"user_data_fields.json: the pattern of {field.key} does not match "
                f"its own example {field.example!r}; fix one of the two"
            )
        for intent in field.required_for:
            if intent not in intents:
                raise KnowledgeBaseError(
                    f"user_data_fields.json: {field.key} is required for "
                    f"{intent!r}, which is not a section of knowledge_base.md"
                )
