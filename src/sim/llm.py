"""The one module that talks to Ollama (T-07).

Agents, the simulated user, the event classifier, the stage labeler and the judge
all call :meth:`LlmClient.chat`. Model, ``num_ctx``, temperature and seed come from
``configs/models.yaml``; output is constrained by a JSON Schema, so no caller ever
parses model text defensively.
"""

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
from ollama import ChatResponse, Client, ResponseError
from pydantic import BaseModel, ValidationError

from sim.config import ModelsConfig, Role
from sim.io import atomic_write

logger = logging.getLogger(__name__)

#: One chat message, as Ollama takes it: ``{"role": ..., "content": ...}``.
Message = Mapping[str, str]

#: Audit log of every call, including cache hits, under ``runs/<exp_id>/``.
LLM_CALLS_LOG = "llm_calls.jsonl"

#: Extra attempts when pydantic refuses a payload JSON Schema cannot express
#: (``fact_id`` only on ``yes``; ``intent`` on ``intent_classified``). Transport
#: retries stay in :meth:`LlmClient.chat`.
SCHEMA_RETRIES = 2


class LlmError(RuntimeError):
    """Something went wrong in a call to the model."""


class LlmCallError(LlmError):
    """The server could not be reached, or kept failing.

    ``attempts`` is how many were actually made: the log has to say one when the
    error was not worth retrying, or the audit trail overstates the flakiness.
    """

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class PromptTooLongError(LlmError):
    """The prompt ate too much of ``num_ctx`` to trust what came back.

    Ollama truncates a prompt longer than the context window without a word,
    which would quietly delete the beginning of the knowledge base or of the
    history. Failing at a fraction of the window is the defence against it.
    """


class CacheEntry(BaseModel):
    """A stored answer, keyed by the hash of everything that produced it.

    The cache is what keeps ``run`` and ``eval`` separate phases: a rubric can
    change and the judge can be run again without re-executing a single
    dialogue. ``latency_s`` is the original call's, kept so a replayed record
    still says how long the real call took. ``digest`` is the recorded
    identity of the weights; the tag in ``model`` is not enough, because an
    upstream retag keeps the name and changes the bits.
    """

    model: str
    digest: str | None = None
    text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    created_at: str


class LlmCallRecord(BaseModel):
    """One line of ``llm_calls.jsonl``: the audit trail of a single call.

    ``prompt_tokens`` is Ollama's ``prompt_eval_count``, which T-16 uses for the
    parity table; ``latency_s`` is the per-call stopwatch T-13 measures agent
    efficiency with, and it only means anything when ``cached`` is false.
    ``attempts`` is how many times this log line talked to the server: one when
    the first try answered, more when it retried, zero when the answer was
    replayed from the cache.
    """

    timestamp: str
    caller: str
    role: str
    model: str
    digest: str | None = None
    prompt_hash: str
    messages: list[dict[str, str]]
    text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool
    attempts: int = 1
    error: str | None = None


@dataclass(frozen=True)
class LlmResponse:
    """The outcome of one call, plus what the metrics of T-13 need to measure it."""

    text: str
    parsed: BaseModel | None
    model: str
    #: The key of this call in the cache and in ``llm_calls.jsonl``. The turn
    #: record of T-09 stores it instead of a second copy of the prompt.
    prompt_hash: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool


class Chat(Protocol):
    """The seam every caller of a model depends on, rather than on ``LlmClient``.

    Agents (T-09, T-10), the simulated user (T-11) and the judge (T-12) are typed
    against this, so their tests hand them a fake of *our* client instead of
    mocking the ``ollama`` package.
    """

    def chat(
        self,
        messages: Sequence[Message],
        *,
        role: Role,
        caller: str,
        schema: type[BaseModel] | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> "LlmResponse":
        """Send ``messages`` to the model configured for ``role``."""
        ...


def chat_parsed[Parsed: BaseModel](
    llm: Chat,
    messages: Sequence[Message],
    *,
    role: Role,
    caller: str,
    schema: type[Parsed],
    seed: int | None = None,
) -> Parsed:
    """Call ``llm`` until the answer matches ``schema``, or the retries run out.

    JSON Schema cannot express every pydantic check. A ``ValidationError`` is
    retried with a bumped seed; transport failures raise immediately.
    """
    last_error: BaseException | None = None
    for attempt in range(SCHEMA_RETRIES + 1):
        call_seed = None if seed is None else seed + attempt
        try:
            answer = llm.chat(
                messages,
                role=role,
                caller=caller,
                schema=schema,
                seed=call_seed,
            )
        except ValidationError as error:
            last_error = error
            continue
        except LlmError as error:
            if not isinstance(error.__cause__, ValidationError):
                raise
            last_error = error
            continue
        parsed = answer.parsed
        if isinstance(parsed, schema):
            return parsed
        last_error = LlmError(
            f"{caller} returned no parsed {schema.__name__}. The client of "
            "T-07 validates against the schema; an unparsed payload would let "
            "a contract-invalid answer through"
        )
    assert last_error is not None
    raise last_error


class ChatTransport(Protocol):
    """The slice of ``ollama.Client`` this module uses."""

    def chat(self, **kwargs: Any) -> ChatResponse:
        """Send one chat request."""
        ...

    def show(self, model: str) -> Any:
        """Return the model's metadata, including its capabilities."""
        ...


class LlmClient:
    """A thin, cached, logged wrapper over one Ollama server."""

    def __init__(
        self,
        config: ModelsConfig,
        *,
        cache_dir: Path,
        log_path: Path,
        transport: ChatTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config
        self._cache_dir = cache_dir
        self._log_path = log_path
        self._transport = transport or Client(timeout=config.client.timeout_s)
        self._sleep = sleep or time.sleep
        self._thinking: dict[str, bool] = {}
        self._lock = threading.Lock()

    def chat(
        self,
        messages: Sequence[Message],
        *,
        role: Role,
        caller: str,
        schema: type[BaseModel] | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> LlmResponse:
        """Send ``messages`` to the model configured for ``role``.

        Args:
            messages: the conversation, oldest first.
            role: which configured model answers: agent, simulator or judge.
            caller: what to record in the log, e.g. ``classifier`` or
                ``judge_facts``; several callers share one model.
            schema: a pydantic model the answer must match.
            seed: overrides the seed of the config, per dialogue (T-14a).
            num_predict: caps the answer's length.
        """
        spec = self._config.spec(role)
        options = _options(
            num_ctx=self._config.num_ctx,
            temperature=spec.temperature,
            seed=spec.seed if seed is None else seed,
            num_predict=num_predict,
        )
        fmt = None if schema is None else schema.model_json_schema()
        prompt_hash = _prompt_hash(spec.name, spec.digest, messages, options, fmt)

        entry = self._read_cache(prompt_hash)
        if entry is not None:
            return self._finish(
                _record(
                    entry,
                    caller,
                    role,
                    prompt_hash,
                    messages,
                    cached=True,
                    attempts=0,
                ),
                schema,
            )

        try:
            response, latency_s, attempts = self._call(
                model=spec.name,
                messages=messages,
                options=options,
                fmt=fmt,
                caller=caller,
            )
        except LlmCallError as failure:
            self._append_log(
                _record(
                    _no_answer(spec.name, spec.digest),
                    caller,
                    role,
                    prompt_hash,
                    messages,
                    cached=False,
                    attempts=failure.attempts,
                    error=str(failure),
                )
            )
            raise

        if response.prompt_eval_count is None:
            error = (
                f"{caller}: Ollama returned no prompt_eval_count. An unmeasurable "
                f"prompt cannot be checked against the "
                f"{self._config.max_prompt_fraction:.0%} budget of "
                f"{self._config.max_prompt_tokens} tokens; past the window Ollama "
                f"truncates in silence"
            )
            self._append_log(
                _record(
                    _no_answer(spec.name, spec.digest),
                    caller,
                    role,
                    prompt_hash,
                    messages,
                    cached=False,
                    attempts=attempts,
                    error=error,
                )
            )
            raise LlmError(error)

        entry = CacheEntry(
            model=spec.name,
            digest=spec.digest,
            text=response.message.content or "",
            prompt_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count or 0,
            latency_s=latency_s,
            created_at=datetime.now(UTC).isoformat(),
        )
        record = _record(
            entry, caller, role, prompt_hash, messages, cached=False, attempts=attempts
        )
        answer = self._finish(record, schema)
        self._write_cache(prompt_hash, entry)
        return answer

    def _call(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        fmt: Mapping[str, Any] | None,
        caller: str,
    ) -> tuple[ChatResponse, float, int]:
        """Send the request, retrying only what is worth retrying.

        Returns the response, the seconds it took and the number of attempts it
        needed. The stopwatch times the attempt that answered and not the
        waiting in between: T-13 reports it as the agent's latency. The
        capability probe lives here so a failing ``/api/show`` is retried and
        logged like any other transport error.
        """
        attempts = self._config.client.retries + 1
        delay = self._config.client.backoff_s
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                think = False if self._supports_thinking(model) else None
                response = self._transport.chat(
                    model=model,
                    messages=list(messages),
                    options=options,
                    format=fmt,
                    think=think,
                )
            except (httpx.RequestError, ResponseError) as failure:
                if attempt == attempts or not _is_transient(failure):
                    raise LlmCallError(
                        f"{caller}: {model} failed after {attempt} attempt(s): "
                        f"{failure}. Check that Ollama is running and holds the "
                        f"models of configs/models.yaml",
                        attempts=attempt,
                    ) from failure
                logger.warning(
                    "%s: attempt %d of %d to %s failed (%s), retrying in %.1fs",
                    caller,
                    attempt,
                    attempts,
                    model,
                    failure,
                    delay,
                )
                self._sleep(delay)
                delay *= 2
            else:
                return response, time.perf_counter() - started, attempt
        raise AssertionError("the loop above either returns or raises")

    def _finish(
        self, record: LlmCallRecord, schema: type[BaseModel] | None
    ) -> LlmResponse:
        """Check the answer, log the call either way, and hand it back."""
        try:
            self._check_prompt_budget(record.prompt_tokens, record.caller)
            parsed = _parse(record.text, schema)
        except LlmError as failure:
            self._append_log(record.model_copy(update={"error": str(failure)}))
            raise
        self._append_log(record)
        return LlmResponse(
            text=record.text,
            parsed=parsed,
            model=record.model,
            prompt_hash=record.prompt_hash,
            prompt_tokens=record.prompt_tokens,
            output_tokens=record.output_tokens,
            latency_s=record.latency_s,
            cached=record.cached,
        )

    def _read_cache(self, prompt_hash: str) -> CacheEntry | None:
        """Return the stored answer for ``prompt_hash``, if there is one."""
        path = self._cache_path(prompt_hash)
        if not path.exists():
            return None
        return CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_cache(self, prompt_hash: str, entry: CacheEntry) -> None:
        """Store ``entry`` under ``prompt_hash``, atomically.

        A reader must never see half a file: ``sim eval`` re-reads this cache
        while ``sim run`` writes it, and rsync copies the directory mid-run.
        """
        atomic_write(self._cache_path(prompt_hash), entry.model_dump_json())

    def _cache_path(self, prompt_hash: str) -> Path:
        """Return the file of ``prompt_hash``, sharded by its first two digits."""
        return self._cache_dir / prompt_hash[:2] / f"{prompt_hash}.json"

    def _append_log(self, record: LlmCallRecord) -> None:
        """Append one record to the JSONL log, whole, one writer at a time."""
        line = record.model_dump_json() + "\n"
        with self._lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as log:
                log.write(line)

    def _check_prompt_budget(self, prompt_tokens: int, caller: str) -> None:
        """Refuse an answer whose prompt came too close to the context window."""
        budget = self._config.max_prompt_tokens
        if prompt_tokens <= budget:
            return
        fraction = self._config.max_prompt_fraction
        raise PromptTooLongError(
            f"{caller}: prompt uses {prompt_tokens} of {self._config.num_ctx} tokens, "
            f"over the {fraction:.0%} budget of {budget}: shorten the prompt (fewer "
            f"KB facts, fewer turns of history) or raise num_ctx in "
            f"configs/models.yaml. Past the window Ollama truncates in silence"
        )

    def _supports_thinking(self, model: str) -> bool:
        """Tell whether ``model`` has a thinking mode, asking the server once.

        Every call passes ``think=False`` (T-03), but a model without the
        capability refuses the parameter, so it is only sent where it applies.
        Ollama drops ``think=None`` from the request, so the two cases differ
        on the wire exactly as they should.
        """
        if model not in self._thinking:
            info = self._transport.show(model)
            self._thinking[model] = "thinking" in (info.capabilities or [])
        return self._thinking[model]


def _record(
    entry: CacheEntry,
    caller: str,
    role: Role,
    prompt_hash: str,
    messages: Sequence[Message],
    *,
    cached: bool,
    attempts: int,
    error: str | None = None,
) -> LlmCallRecord:
    """Build the log record of one call, fresh or replayed from the cache."""
    return LlmCallRecord(
        timestamp=datetime.now(UTC).isoformat(),
        caller=caller,
        role=role,
        model=entry.model,
        digest=entry.digest,
        prompt_hash=prompt_hash,
        messages=[dict(message) for message in messages],
        text=entry.text,
        prompt_tokens=entry.prompt_tokens,
        output_tokens=entry.output_tokens,
        latency_s=entry.latency_s,
        cached=cached,
        attempts=attempts,
        error=error,
    )


def _no_answer(model: str, digest: str | None = None) -> CacheEntry:
    """The empty stand-in a call that never answered is logged with."""
    return CacheEntry(
        model=model,
        digest=digest,
        text="",
        prompt_tokens=0,
        output_tokens=0,
        latency_s=0.0,
        created_at=datetime.now(UTC).isoformat(),
    )


def _is_transient(failure: Exception) -> bool:
    """Tell whether another attempt could plausibly succeed.

    A timeout or a refused connection is worth retrying, and so is a 5xx: the
    server was busy loading or evicting a model. A 4xx is not: a missing model
    or a malformed request will fail the same way three times.
    """
    if isinstance(failure, httpx.RequestError):
        return True
    return isinstance(failure, ResponseError) and failure.status_code >= 500


def _prompt_hash(
    model: str,
    digest: str | None,
    messages: Sequence[Message],
    options: Mapping[str, Any],
    fmt: Mapping[str, Any] | None,
) -> str:
    """Hash everything that decides the answer, so the cache can key on it.

    ``think`` is not part of the key: every call passes ``think=False`` (T-03),
    and leaving it out is what lets a cached re-evaluation run without asking
    the server which capabilities the model has. The digest is: it is the
    identity of the weights, and the tag in ``model`` is reused across retags.
    """
    payload = json.dumps(
        {
            "model": model,
            "digest": digest,
            "messages": [dict(message) for message in messages],
            "options": dict(options),
            "format": fmt,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse(text: str, schema: type[BaseModel] | None) -> BaseModel | None:
    """Validate ``text`` against ``schema``, or fail saying what came back."""
    if schema is None:
        return None
    try:
        return schema.model_validate_json(text)
    except ValidationError as invalid:
        raise LlmError(
            f"the answer does not match {schema.__name__}, the schema it was "
            f"constrained by:\n{invalid}\nThe answer was: {text!r}. Ollama enforces "
            f"the schema, so this is a broken server or a schema it could not "
            f"honour, not something to parse around"
        ) from invalid


def _options(
    *,
    num_ctx: int,
    temperature: float | None,
    seed: int | None,
    num_predict: int | None,
) -> dict[str, Any]:
    """Build the Ollama options, leaving out what the config does not fix."""
    options: dict[str, Any] = {"num_ctx": num_ctx}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed
    if num_predict is not None:
        options["num_predict"] = num_predict
    return options
