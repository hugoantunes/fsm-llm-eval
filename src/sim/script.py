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
from types import MappingProxyType
from typing import Any

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

#: Closing speech-act words that must appear on the completing turn when present
#: in the beat text.
CLOSING_WORDS = frozenset(("goodbye", "bye"))


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
    local_keywords: tuple[str, ...] = ()

    def faults_in(self, message: str) -> tuple[str, ...]:
        """Return what this beat requires that one message does not carry."""
        assessment = self.assess(message, progress=BeatProgress())
        return assessment.cumulative_missing + assessment.local_missing

    def satisfied_by(self, message: str) -> bool:
        """Whether ``message`` delivers this beat and may consume it."""
        return self.assess(message, progress=BeatProgress()).complete

    def assess(self, message: str, *, progress: "BeatProgress") -> "BeatAssessment":
        """Evaluate ``message`` against cumulative progress and local predicates."""
        updated = progress.with_message(self, message)
        return BeatAssessment(
            progress=updated,
            progress_gained=updated != progress,
            cumulative_missing=self.cumulative_faults(updated),
            local_missing=self.local_faults_in(message),
        )

    def cumulative_faults(self, progress: "BeatProgress") -> tuple[str, ...]:
        """Return cumulative requirements still missing in ``progress``."""
        faults = [
            f"the exact string {literal!r}"
            for literal in self.literals
            if literal not in progress.matched_literals
        ]
        if self.verbatim is not None:
            return tuple(faults)
        faults += [
            f"the word {keyword!r}"
            for keyword in self.keywords
            if keyword not in progress.matched_keywords
        ]
        return tuple(faults)

    def local_faults_in(self, message: str) -> tuple[str, ...]:
        """Return turn-local requirements ``message`` does not satisfy."""
        faults: list[str] = []
        if self.verbatim is not None and not _contains_normalized(
            self.verbatim, message
        ):
            faults.append(f"the instruction {self.verbatim!r}")
        if self.negates and not _denies(message):
            faults.append("a denial")
        if self.asks and QUESTION not in message:
            faults.append("a question")
        if self.local_keywords and not any(
            _carries(keyword, message) for keyword in self.local_keywords
        ):
            faults.append(_local_words_requirement(self.local_keywords))
        return tuple(faults)

    def local_requirements(self) -> tuple[str, ...]:
        """Return turn-local predicates still required for completion."""
        required: list[str] = []
        if self.verbatim is not None:
            required.append(f"the instruction {self.verbatim!r}")
        if self.asks:
            required.append("a question")
        if self.negates:
            required.append("a denial")
        if self.local_keywords:
            required.append(_local_words_requirement(self.local_keywords))
        return tuple(required)


@dataclass(frozen=True)
class BeatProgress:
    """Cumulative requirements matched by delivered turns of the active beat."""

    matched_literals: frozenset[str] = frozenset()
    matched_keywords: frozenset[str] = frozenset()

    def with_message(self, beat: Beat, message: str) -> "BeatProgress":
        """Return progress updated by one delivered ``message``."""
        return BeatProgress(
            matched_literals=self.matched_literals
            | {literal for literal in beat.literals if literal in message},
            matched_keywords=self.matched_keywords
            | {keyword for keyword in beat.keywords if _carries(keyword, message)},
        )


@dataclass(frozen=True)
class BeatAssessment:
    """Beat check result for one candidate/delivered message."""

    progress: BeatProgress
    progress_gained: bool
    cumulative_missing: tuple[str, ...]
    local_missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Whether cumulative and local requirements are both satisfied."""
        return not self.cumulative_missing and not self.local_missing


@dataclass(frozen=True)
class TurnProgress:
    """Outcome of committing one delivered user message."""

    turn: int
    consumed_beat: int | None
    beat_started_at_turn: int | None
    beat_completed_at_turn: int | None
    assessment: BeatAssessment | None
    active_beat_complete: bool


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
    keywords = _keywords(text, literals)
    return Beat(
        number=number,
        text=text,
        literals=literals,
        keywords=keywords,
        asks=QUESTION in text,
        negates=_denies(text),
        verbatim=text if canary is not None and canary in text else None,
        local_keywords=_closing_keywords(keywords),
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
    if keyword == word:
        return True
    stemmed_keyword = _stem(keyword)
    stemmed_word = _stem(word)
    if stemmed_keyword == stemmed_word:
        return True
    shared = min(len(stemmed_keyword), len(stemmed_word))
    return shared >= STEM and stemmed_keyword[:shared] == stemmed_word[:shared]


def _stem(word: str) -> str:
    """Return a light stem for common English inflections."""
    for suffix in ("ing", "ed", "es", "s"):
        if not word.endswith(suffix):
            continue
        stem = word[: -len(suffix)]
        if len(stem) < STEM:
            continue
        if suffix in ("ing", "ed") and len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    return word


#: Present/future shipment words. Past ``shipped`` / ``dispatched`` are not here:
#: they only count as not-yet when a present auxiliary makes them passive
#: (``before it is shipped``), not when they are simple past (``before it shipped``).
_PRESENT_SHIPMENT = frozenset(("ship", "ships", "shipping", "dispatch"))
_PAST_SHIPMENT = frozenset(("shipped", "dispatched"))
_PRESENT_AUX = frozenset(("am", "is", "are", "be", "being"))
_SHIPMENT_BLOCKERS = frozenset(("after", "when", "once", "already"))
_FUTURE_TIME = frozenset(("tomorrow",))
_BEFORE_SKIP = FUNCTION_WORDS | frozenset(("order",))


def _denies(text: str) -> bool:
    """Whether ``text`` denies something or says shipment has not happened yet."""
    words = _words(text)
    if bool(set(words) & DENIALS) or "n't" in text.lower():
        return True
    return _shipment_has_not_happened(words)


def _shipment_has_not_happened(words: tuple[str, ...]) -> bool:
    """Whether ``words`` describe a current pre-shipment state."""
    return _before_present_shipment(words) or _upcoming_shipment(words)


def _before_present_shipment(words: tuple[str, ...]) -> bool:
    """Whether ``before`` is followed by a present/future shipment construction."""
    try:
        start = words.index("before")
    except ValueError:
        return False
    saw_present_aux = False
    for word in words[start + 1 :]:
        if word in _PRESENT_AUX:
            saw_present_aux = True
            continue
        if word in _PRESENT_SHIPMENT:
            return True
        if word in _PAST_SHIPMENT:
            return saw_present_aux
        if word in _BEFORE_SKIP:
            continue
        return False
    return False


def _upcoming_shipment(words: tuple[str, ...]) -> bool:
    """Whether shipment is scheduled ahead, so it has not happened yet."""
    present = set(words)
    if present & _SHIPMENT_BLOCKERS:
        return False
    if not present & _PRESENT_SHIPMENT:
        return False
    return "will" in present or bool(present & _FUTURE_TIME)


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
        self._turn = 0
        self._progress = BeatProgress()
        self._beat_start_turn = 1 if beats else 0

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

    @property
    def turn(self) -> int:
        """Delivered user turns committed so far."""
        return self._turn

    @property
    def current_progress(self) -> BeatProgress:
        """Cumulative matches gathered for the active beat."""
        return self._progress

    @property
    def beat_start_turn(self) -> int:
        """Turn index where the active beat started."""
        return self._beat_start_turn

    def assess(self, message: str) -> BeatAssessment | None:
        """Preview ``message`` against current beat without mutating progress."""
        beat = self.current
        if beat is None:
            return None
        return beat.assess(message, progress=self._progress)

    def commit(self, message: str) -> TurnProgress:
        """Commit one delivered user ``message`` into progress and beat order."""
        self._turn += 1
        beat = self.current
        if beat is None:
            return TurnProgress(
                turn=self._turn,
                consumed_beat=None,
                beat_started_at_turn=None,
                beat_completed_at_turn=None,
                assessment=None,
                active_beat_complete=True,
            )
        assessment = beat.assess(message, progress=self._progress)
        self._progress = assessment.progress
        if assessment.complete:
            start = self._beat_start_turn
            end = self._turn
            self._index += 1
            self._deferrals = 0
            self._progress = BeatProgress()
            if not self.complete:
                self._beat_start_turn = self._turn + 1
            return TurnProgress(
                turn=self._turn,
                consumed_beat=beat.number,
                beat_started_at_turn=start,
                beat_completed_at_turn=end,
                assessment=assessment,
                active_beat_complete=True,
            )
        if assessment.progress_gained:
            self._deferrals = 0
        else:
            self._deferrals += 1
        return TurnProgress(
            turn=self._turn,
            consumed_beat=None,
            beat_started_at_turn=None,
            beat_completed_at_turn=None,
            assessment=assessment,
            active_beat_complete=False,
        )

    def active_beat_diagnostics(self) -> MappingProxyType[str, Any] | None:
        """Return immutable diagnostics for the active incomplete beat."""
        beat = self.current
        if beat is None:
            return None
        satisfied = {
            "exact_strings": [
                literal
                for literal in beat.literals
                if literal in self._progress.matched_literals
            ],
            "keywords": [
                keyword
                for keyword in beat.keywords
                if keyword in self._progress.matched_keywords
            ],
        }
        missing = {
            "exact_strings": [
                literal
                for literal in beat.literals
                if literal not in self._progress.matched_literals
            ],
            "keywords": [
                keyword
                for keyword in beat.keywords
                if keyword not in self._progress.matched_keywords
            ],
        }
        turns_on_beat = max(0, self._turn - self._beat_start_turn + 1)
        return MappingProxyType(
            {
                "active_beat_index": beat.number,
                "satisfied_cumulative_requirements": satisfied,
                "missing_cumulative_requirements": missing,
                "pending_local_predicates": list(beat.local_requirements()),
                "beat_start_turn": self._beat_start_turn,
                "delivered_turns_on_active_beat": turns_on_beat,
            }
        )

    def deliver(self) -> None:
        """Consume the current beat; the next message owes the next one."""
        if self.complete:
            raise ScriptError(
                "every beat of the script is already delivered. Advance only "
                "while a beat is still owed"
            )
        self._index += 1
        self._deferrals = 0
        self._progress = BeatProgress()
        if not self.complete:
            self._beat_start_turn = self._turn + 1

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
    return render_remaining_literals(beat, progress=None)


def render_remaining_literals(
    beat: Beat | None, *, progress: BeatProgress | None
) -> str:
    """Render missing exact strings for the active beat."""
    if beat is None:
        return NOTHING
    seen = frozenset() if progress is None else progress.matched_literals
    items = ((beat.verbatim,) if beat.verbatim is not None else ()) + tuple(
        literal for literal in beat.literals if literal not in seen
    )
    if not items:
        return NOTHING
    return "\n".join(f"- {item}" for item in dict.fromkeys(items))


def render_keywords(beat: Beat | None) -> str:
    """Render the content words the beat due now needs, or :data:`NOTHING`."""
    return render_remaining_keywords(beat, progress=None)


def render_remaining_keywords(
    beat: Beat | None, *, progress: BeatProgress | None
) -> str:
    """Render missing cumulative keywords for the active beat."""
    if beat is None or beat.verbatim is not None or not beat.keywords:
        return NOTHING
    seen = frozenset() if progress is None else progress.matched_keywords
    pending = [keyword for keyword in beat.keywords if keyword not in seen]
    if not pending:
        return NOTHING
    return "\n".join(f"- {keyword}" for keyword in pending)


def render_satisfied_cumulative(beat: Beat | None, *, progress: BeatProgress) -> str:
    """Render cumulative requirements already satisfied for the active beat."""
    if beat is None:
        return NOTHING
    items = [
        *[
            f"exact: {literal}"
            for literal in beat.literals
            if literal in progress.matched_literals
        ],
        *[
            f"word: {keyword}"
            for keyword in beat.keywords
            if keyword in progress.matched_keywords
        ],
    ]
    if not items:
        return NOTHING
    return "\n".join(f"- {item}" for item in items)


def render_pending_local(beat: Beat | None) -> str:
    """Render turn-local predicates still required for beat completion."""
    if beat is None:
        return NOTHING
    local = beat.local_requirements()
    if not local:
        return NOTHING
    return "\n".join(f"- {item}" for item in local)


def _closing_keywords(keywords: Sequence[str]) -> tuple[str, ...]:
    """Return closing speech-act words present in ``keywords``."""
    return tuple(keyword for keyword in keywords if keyword in CLOSING_WORDS)


def _local_words_requirement(words: Sequence[str]) -> str:
    """Render one local keyword requirement for diagnostics/prompts."""
    if len(words) == 1:
        return f"the word {words[0]!r} in this message"
    options = ", ".join(repr(word) for word in words)
    return f"one of {options} in this message"
