"""Hybrid user-event detection for the FSM engine (T-08).

Rules read the slot patterns of ``data/kb/user_data_fields.json`` and a short
farewell list first. Only when they do not fire does the small model classify,
constrained to the events the current state accepts plus ``none``. Pattern-less
fields (item, reason, preferred resolution) come back on that schema so the
guard of ``data_collection`` can see them.

Every rule here is about what the customer just did. What the dialogue already
holds — the intent, the required data — moves the machine through
:meth:`sim.engine.FsmEngine._advance_when_ready` instead, after the classifier
has had its turn to answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, create_model

from sim.fsm import FsmSpec, Transition
from sim.kb import UserDataField
from sim.llm import Chat
from sim.prompts import DEFAULT_PROMPTS_DIR, load_prompt

#: What the detector returns when no user event fired this turn.
NONE = "none"

#: The prompt file of ``data/prompts/``.
PROMPT = "event_classifier"

_APOSTROPHE = "'\u2019"
_FAREWELL_EVENT = "farewell"
#: High-precision goodbye cues; only consulted when ``farewell`` is allowed.
_FAREWELL = re.compile(
    rf"(?i)\bgood\s*bye\b|\bbye-bye\b|\bbye\b|\bthat[{_APOSTROPHE}]s all\b"
)

_IDENTIFY_GUARD = "order_and_email_present"
_COLLECT_GUARD = "required_data_collected"

#: The unguarded edge that settles which of the four requests this is. It is the
#: one rule edge no guard names, so it is looked up by event; ``machine.yaml``
#: stays the only place the name is written, and a rename raises there.
INTENT_EVENT = "intent_classified"

CLASSIFIER = "classifier"

#: Keys of the classifier schema that are not slot values.
_NOT_SLOTS = frozenset({"event", "intent"})


@dataclass(frozen=True)
class Detection:
    """The event the user fired this turn, plus anything the rules extracted."""

    event: str
    intent: str | None = None
    slots: dict[str, str] = field(default_factory=dict)


class EventError(RuntimeError):
    """The detector cannot classify this turn; the message says what is missing."""


@dataclass(frozen=True)
class MachineRules:
    """The edges the detector and the engine key their rules off.

    Every name is read from ``machine.yaml``, so renaming an event or a guard
    there raises here instead of silently switching a rule off.
    """

    identify_state: str
    identify_event: str
    identify_keys: tuple[str, ...]
    collect_state: str
    collect_event: str
    intent_event: str


def patterned_keys_required_for_every_intent(
    fields: Sequence[UserDataField], intents: Sequence[str]
) -> tuple[str, ...]:
    """Return patterned field keys every intent requires.

    Identification and the ``order_and_email_present`` guard both read this
    so a rename in ``user_data_fields.json`` cannot silently skip the rule.
    """
    every = set(intents)
    keys = tuple(
        item.key
        for item in fields
        if item.pattern is not None and every <= set(item.required_for)
    )
    if not keys:
        raise EventError(
            "user_data_fields.json has no patterned field required_for every "
            "intent; the identification rule would never fire"
        )
    return keys


def required_data_held(
    fields: Sequence[UserDataField],
    intent: str | None,
    held: Mapping[str, str],
) -> bool:
    """Return whether every field the classified intent requires is in ``held``."""
    if intent is None:
        return False
    return all(item.key in held for item in fields if intent in item.required_for)


def detect_user_event(
    message: str,
    *,
    state: str,
    spec: FsmSpec,
    collected: dict[str, str],
    fields: Sequence[UserDataField],
    intents: Sequence[str],
    intent: str | None,
    llm: Chat | None,
    transcript: str,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    seed: int | None = None,
) -> Detection:
    """Return the user event for ``message`` in ``state``, rules first.

    A rule fires only on something the customer did this turn: said goodbye,
    completed the identification, sent or corrected a datum. That a state has
    nothing left to do is not a user event and is not detected here; the engine
    advances on its own afterwards, so every state keeps listening for the
    classifier's answer instead of being short-circuited by what is already held.
    """
    allowed = spec.events_for(state)
    slots = _extract_slots(message, fields)
    held = {**collected, **slots}
    changed = [key for key, value in slots.items() if collected.get(key) != value]
    free_text = _patternless_for_intent(fields, intent)
    rules = machine_rules(spec, fields, intents)

    if _FAREWELL_EVENT in allowed and _FAREWELL.search(message):
        return Detection(_FAREWELL_EVENT, slots=slots)
    if (
        state == rules.identify_state
        and rules.identify_event in allowed
        and all(key in held for key in rules.identify_keys)
    ):
        return Detection(rules.identify_event, slots=slots)
    if state == rules.collect_state and rules.collect_event in allowed and changed:
        return Detection(rules.collect_event, slots=slots)
    return _classify_with_llm(
        state=state,
        allowed=allowed,
        intents=intents,
        llm=llm,
        transcript=transcript,
        prompts_dir=prompts_dir,
        seed=seed,
        slots=slots,
        free_text=free_text,
    )


def user_event_schema(
    events: Sequence[str],
    intents: Sequence[str],
    slot_keys: Sequence[str] = (),
) -> type[BaseModel]:
    """Build the JSON Schema whose enum is ``events`` plus ``none``.

    Pattern-less slot keys of the current intent become optional string fields
    so the classifier can fill what no regex can see.
    """
    event_literal = Literal[*events, NONE]
    intent_literal = Literal[*intents] | None
    shape: dict[str, object] = {
        "event": (event_literal, ...),
        "intent": (intent_literal, None),
    }
    for key in slot_keys:
        shape[key] = (str | None, None)
    return create_model(
        "UserEvent",
        __config__=ConfigDict(extra="forbid"),
        **shape,
    )


def machine_rules(
    spec: FsmSpec, fields: Sequence[UserDataField], intents: Sequence[str]
) -> MachineRules:
    """Derive the rule edges from the guarded edges and ``required_for``."""
    names = {edge.event for edge in spec.transitions}
    if _FAREWELL_EVENT not in names:
        raise EventError(
            f"machine.yaml has no {_FAREWELL_EVENT!r} event (found "
            f"{sorted(names)}). The goodbye rule would never fire and every "
            f"matching turn would fall through to the classifier"
        )
    identify = _unique_guarded_edge(spec, _IDENTIFY_GUARD)
    collect = _unique_guarded_edge(spec, _COLLECT_GUARD)
    intent_edge = _unique_named_event(spec, INTENT_EVENT)
    return MachineRules(
        identify_state=identify.source,
        identify_event=identify.event,
        identify_keys=patterned_keys_required_for_every_intent(fields, intents),
        collect_state=collect.source,
        collect_event=collect.event,
        intent_event=intent_edge.event,
    )


def _unique_guarded_edge(spec: FsmSpec, guard: str) -> Transition:
    """Return the single transition that carries ``guard``, or raise."""
    matches = [edge for edge in spec.transitions if edge.guard == guard]
    if len(matches) != 1:
        raise EventError(
            f"machine.yaml must have exactly one transition with guard {guard!r} "
            f"(found {len(matches)}). The detector keys the slot rules off that "
            f"edge so a rename in the YAML cannot silently fall through to the "
            f"classifier"
        )
    return matches[0]


def _unique_named_event(spec: FsmSpec, event: str) -> Transition:
    """Return the single non-star transition for ``event``, or raise."""
    matches = [
        edge for edge in spec.transitions if edge.event == event and not edge.from_any
    ]
    if len(matches) != 1:
        raise EventError(
            f"machine.yaml must have exactly one non-star transition for "
            f"{event!r} (found {len(matches)}). The detector keys the already-"
            f"known-intent rule off that edge so a rename cannot silently fall "
            f"through to the classifier"
        )
    return matches[0]


def _extract_slots(message: str, fields: Sequence[UserDataField]) -> dict[str, str]:
    """Return the patterned fields that appear in ``message``."""
    found: dict[str, str] = {}
    for item in fields:
        if item.pattern is None:
            continue
        match = re.search(item.pattern, message, flags=re.IGNORECASE)
        if match:
            found[item.key] = match.group(0)
    return found


def _patternless_for_intent(
    fields: Sequence[UserDataField], intent: str | None
) -> list[UserDataField]:
    """Return free-text fields the classified intent still has to collect."""
    if intent is None:
        return []
    return [
        item for item in fields if item.pattern is None and intent in item.required_for
    ]


def _slots_from_payload(
    payload: dict[str, object], patterned: dict[str, str]
) -> dict[str, str]:
    """Merge regex slots with any free-text values the classifier copied out."""
    from_llm = {
        key: value.strip()
        for key, value in payload.items()
        if key not in _NOT_SLOTS and isinstance(value, str) and value.strip()
    }
    return {**patterned, **from_llm}


def _render_slot_fields(fields: Sequence[UserDataField]) -> str:
    """Render the free-text slot list for the classifier prompt."""
    if not fields:
        return "None this turn."
    return "\n".join(f"- `{item.key}` — {item.label}" for item in fields)


def _classify_with_llm(
    *,
    state: str,
    allowed: Sequence[str],
    intents: Sequence[str],
    llm: Chat | None,
    transcript: str,
    prompts_dir: Path,
    seed: int | None,
    slots: dict[str, str],
    free_text: Sequence[UserDataField],
) -> Detection:
    """Ask the small model, constrained to ``allowed`` plus ``none``."""
    if llm is None:
        raise EventError(
            "rules did not fire and no LLM client was given to fall back on. "
            "Pass llm= to FsmEngine when the turn is not a slot or a farewell"
        )
    prompt = load_prompt(PROMPT, directory=prompts_dir).render(
        state=state,
        allowed_events="\n".join(f"- `{name}`" for name in [*allowed, NONE]),
        transcript=transcript,
        slot_fields=_render_slot_fields(free_text),
    )
    schema = user_event_schema(allowed, intents, [item.key for item in free_text])
    answer = llm.chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        role="simulator",
        caller=CLASSIFIER,
        schema=schema,
        seed=seed,
    )
    parsed = answer.parsed
    if parsed is None:
        raise EventError(
            "the classifier returned no parsed object; the client of T-07 should "
            "have validated the schema before this"
        )
    payload = parsed.model_dump()
    event = payload["event"]
    classified_intent = payload["intent"]
    if event == INTENT_EVENT and classified_intent is None:
        raise EventError(
            f"event intent_classified in state {state!r} with no intent "
            f"(transcript: {transcript!r}). The classifier must name one of "
            f"{list(intents)}; a null intent would wedge data_collection "
            f"because required_data_collected stays false"
        )
    return Detection(
        event=event,
        intent=classified_intent,
        slots=_slots_from_payload(payload, slots),
    )
