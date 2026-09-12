"""Deliver a scenario's script as an ordered plan of mandatory beats (T-11).

``scenario.script`` is a plan and not a hint: the customer sends one beat per
message, in the order they are written, and a dialogue may not end while a beat
is still owed. The first simulated user handed the model the whole numbered
script every turn and let it infer its own position from the history, which is
what the pilot of 2026-09-11 measured the cost of — beats dropped, a verbatim
canary paraphrased, a dialogue closed while the injection it existed to deliver
had never been sent (``docs/pilot.md``).

Position is explicit state here instead, and a beat is consumed by a message
that carries it rather than by the model having written something while that
beat was due. What carries it is the **contract** of :class:`Beat`, derived from
the beat's own words: the strings it spells out, the content words it is made
of, a question where it asks one and a denial where it denies. The rerun of
2026-09-11 showed why every part is needed — the canary arrived inside ``Please
proceed with VN6-HARBOUR-1188``, which is not the attack, and a beat asking
where a slip appears was consumed by the customer answering its own question.

The contract holds because a beat of the frozen dataset is one customer
utterance. It is checked without a second model: a judge in the loop of the
simulator would make the instrument depend on the thing T-16 is validating.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sim.kb import UserDataField
from sim.schemas import Scenario

#: Consecutive messages that may answer the agent without delivering the beat
#: they owe before the simulator is told to deliver it anyway. Two leaves room
#: for a clarification and one follow-up inside the turn budget of T-05, and
#: stops a customer from dodging a beat until ``max_turns`` runs out.
MAX_DEFERRALS = 2

#: Rendered where a turn has no such field: a plan with every beat behind it, or
#: a beat that requires nothing word for word. The prompt file says what it
#: means, so no instructional English is written here.
NOTHING = "none"

#: How the plan marks a beat that is behind, the one due now, and one still
#: ahead. The prompt file explains the three; they are labels, not instructions.
SENT, NOW, LATER = "sent", "now", "later"

#: What a beat is made of that a message need not repeat: the grammar around its
#: content. A word wrongly in here is a requirement lost, so the list holds only
#: words that carry no content of their own, and no verb, noun or number.
FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "about",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "mine",
        "yours",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "why",
        "please",
        "also",
        "again",
        "already",
        "still",
        "yet",
        "too",
        "very",
        "much",
        "more",
        "most",
        "some",
        "any",
        "each",
        "one",
        "two",
        "three",
    ]
)

#: Words that deny or restrict. A beat written with one of them is not delivered
#: by a message that agrees, which is what ``edge_02`` lost when "not a cancel"
#: was consumed by a question about tracking. They are a class and not a
#: requirement word for word, because "no e-mail" is answered by "I don't have
#: one" as faithfully as by "no".
DENIALS = frozenset(
    [
        "no",
        "not",
        "none",
        "never",
        "nothing",
        "nobody",
        "neither",
        "nor",
        "without",
        "only",
        "just",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "isn't",
        "aren't",
        "wasn't",
        "haven't",
        "hasn't",
        "hadn't",
        "refuse",
        "refuses",
        "refusing",
    ]
)

#: What a message asks with. A beat that asks something and a message that does
#: not are two different turns of the dialogue.
QUESTION = "?"

#: Shortest prefix two words share before one counts as an inflection of the
#: other, so "cancel" is carried by "cancelling" and "not" is not by "nothing".
STEM = 4


class ScriptError(RuntimeError):
    """The script cannot advance here; the message says why."""


@dataclass(frozen=True)
class Beat:
    """One beat of the script and the contract a message has to meet to send it.

    ``literals`` is what a paraphrase would destroy: the canary an adversarial
    script plants (T-05) and the patterned customer data the beat spells out.
    ``verbatim`` is the whole beat when it plants the canary: the attack is an
    instruction *to the agent*, so content-word coverage is not enough — ``I
    need to ignore my previous instructions`` has the words and still aims the
    attack at the customer. ``keywords`` / ``asks`` / ``negates`` are the
    contract of an ordinary beat, where a paraphrase is still a delivery.
    """

    number: int
    text: str
    literals: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    asks: bool = False
    negates: bool = False
    verbatim: str | None = None

    def faults_in(self, message: str) -> tuple[str, ...]:
        """Return what this beat requires that ``message`` does not carry."""
        faults = [
            f"the exact string {literal!r}"
            for literal in self.literals
            if literal not in message
        ]
        if self.verbatim is not None:
            if not _contains_normalized(self.verbatim, message):
                faults.append(f"the instruction {self.verbatim!r}")
            return tuple(faults)
        faults += [
            f"the word {keyword!r}"
            for keyword in self.keywords
            if not _carries(keyword, message)
        ]
        if self.negates and not _denies(message):
            faults.append("a denial")
        if self.asks and QUESTION not in message:
            faults.append("a question")
        return tuple(faults)

    def satisfied_by(self, message: str) -> bool:
        """Whether ``message`` delivers this beat and may consume it."""
        return not self.faults_in(message)


def beats_of(scenario: Scenario, fields: Sequence[UserDataField]) -> tuple[Beat, ...]:
    """Split ``scenario.script`` into numbered beats, in the order written."""
    return tuple(
        _beat(number, text, scenario.canary, fields)
        for number, text in enumerate(scenario.script, start=1)
    )


def _beat(
    number: int, text: str, canary: str | None, fields: Sequence[UserDataField]
) -> Beat:
    """Read one beat's contract off the words it is written with."""
    literals = _literals(text, canary, fields)
    return Beat(
        number=number,
        text=text,
        literals=literals,
        keywords=_keywords(text, literals),
        asks=QUESTION in text,
        negates=_denies(text),
        verbatim=text if canary is not None and canary in text else None,
    )


def _keywords(text: str, literals: Sequence[str]) -> tuple[str, ...]:
    """Return the content words of ``text``, in the order it is written.

    What the beat spells out is left out, because ``literals`` already requires
    it character for character, and so are the function words, the denials and
    the single letters left by a word like "e-mail": the first carry no content,
    the second are checked as a class and the third are not words.
    """
    rest = text
    for literal in literals:
        rest = rest.replace(literal, " ")
    return tuple(
        dict.fromkeys(
            word
            for word in _words(rest)
            if len(word) > 1 and word not in FUNCTION_WORDS and word not in DENIALS
        )
    )


def _normalized(text: str) -> str:
    """Lowercase ``text`` and keep only its words, so case and punctuation drop out."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _contains_normalized(required: str, message: str) -> bool:
    """Whether ``message`` carries ``required`` once case and punctuation are gone."""
    return _normalized(required) in _normalized(message)


def _words(text: str) -> tuple[str, ...]:
    """Split ``text`` into lowercase words, keeping the apostrophe of "don't"."""
    return tuple(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()))


def _carries(keyword: str, message: str) -> bool:
    """Whether ``message`` uses ``keyword`` or an inflection of it."""
    return any(_same_word(keyword, word) for word in _words(message))


def _same_word(keyword: str, word: str) -> bool:
    """Whether two words are the same one, allowing a different ending."""
    shared = min(len(keyword), len(word))
    return keyword == word or (shared >= STEM and keyword[:shared] == word[:shared])


def _denies(text: str) -> bool:
    """Whether ``text`` says no to something, or restricts what it asks for."""
    return bool(set(_words(text)) & DENIALS) or "n't" in text.lower()


def _literals(
    text: str, canary: str | None, fields: Sequence[UserDataField]
) -> tuple[str, ...]:
    """Return what ``text`` spells out and a message has to repeat unchanged.

    The patterns come from ``user_data_fields.json``, the same file the
    rule-first event detection of T-08 reads, so the shape of an order number is
    never written twice.
    """
    found = [canary] if canary is not None and canary in text else []
    for field in fields:
        if field.pattern is None:
            continue
        found.extend(
            match.group(0)
            for match in re.finditer(field.pattern, text, flags=re.IGNORECASE)
        )
    return tuple(dict.fromkeys(found))


class ScriptProgress:
    """Where one dialogue stands in its script: the beat due now and what is left.

    Mutable by design, one instance per dialogue: the simulated user advances it
    as messages are accepted and the runtime reads it to decide whether the
    dialogue may end.
    """

    def __init__(self, beats: Sequence[Beat]) -> None:
        self._beats = tuple(beats)
        self._index = 0
        self._deferrals = 0

    @property
    def beats(self) -> tuple[Beat, ...]:
        """Every beat of the script, in order."""
        return self._beats

    @property
    def current(self) -> Beat | None:
        """The beat this message owes, or ``None`` once every beat is delivered."""
        if self.complete:
            return None
        return self._beats[self._index]

    @property
    def pending(self) -> tuple[Beat, ...]:
        """The beats still owed, the current one first."""
        return self._beats[self._index :]

    @property
    def delivered(self) -> tuple[int, ...]:
        """The numbers of the beats already sent, in the order they went out."""
        return tuple(range(1, self._index + 1))

    @property
    def complete(self) -> bool:
        """Whether every beat of the script has been delivered."""
        return self._index >= len(self._beats)

    @property
    def deferrals(self) -> int:
        """Messages that answered the agent without delivering the current beat."""
        return self._deferrals

    @property
    def must_deliver(self) -> bool:
        """Whether the current beat has been put off as long as it may be."""
        return not self.complete and self._deferrals >= MAX_DEFERRALS

    def deliver(self) -> None:
        """Consume the current beat; the next message owes the next one."""
        if self.complete:
            raise ScriptError(
                "every beat of the script is already delivered. Advance only "
                "while a beat is still owed"
            )
        self._index += 1
        self._deferrals = 0

    def defer(self) -> None:
        """Record a message that answered the agent and left the beat for later."""
        self._deferrals += 1


def render_plan(progress: ScriptProgress) -> str:
    """Number the beats and mark the one due now, so the plan shows its position."""
    return "\n".join(
        f"{beat.number}. [{_mark(beat, progress)}] {beat.text}"
        for beat in progress.beats
    )


def _mark(beat: Beat, progress: ScriptProgress) -> str:
    """Return the marker of ``beat`` relative to where the dialogue stands."""
    current = progress.current
    if current is None or beat.number < current.number:
        return SENT
    if beat.number == current.number:
        return NOW
    return LATER


def render_beat(beat: Beat | None) -> str:
    """Render the beat due now with its number, or :data:`NOTHING`."""
    if beat is None:
        return NOTHING
    return f"{beat.number}. {beat.text}"


def render_literals(beat: Beat | None) -> str:
    """Render what the beat due now requires word for word, or :data:`NOTHING`."""
    if beat is None:
        return NOTHING
    items = ((beat.verbatim,) if beat.verbatim else ()) + beat.literals
    if not items:
        return NOTHING
    return "\n".join(f"- {item}" for item in dict.fromkeys(items))


def render_keywords(beat: Beat | None) -> str:
    """Render the content words the beat due now needs, or :data:`NOTHING`."""
    if beat is None or beat.verbatim is not None or not beat.keywords:
        return NOTHING
    return "\n".join(f"- {keyword}" for keyword in beat.keywords)
