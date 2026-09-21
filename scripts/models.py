"""Pull and verify the models of ``configs/models.yaml`` against the local Ollama.

Usage::

    uv run python scripts/models.py pull            # ollama pull every configured model
    uv run python scripts/models.py digests         # print role, name and local digest
    uv run python scripts/models.py verify          # exit 1 unless every digest matches
    uv run python scripts/models.py eval-preflight  # show judge name and ollama ps
    uv run python scripts/models.py warmup-judge    # load the judge; not via LlmClient

``verify`` is the reproducibility check of T-03: both machines must hold the same
digests, and those must be the ones recorded in the config.
"""

import argparse
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import ollama

from sim.config import DEFAULT_CONFIG_PATH, ROLES, load_models_config

ModelSpec = Mapping[str, str | None]


class WarmupClient(Protocol):
    """The slice of ``ollama.Client`` that ``warmup_judge`` uses."""

    def chat(self, **kwargs: Any) -> object:
        """Send one chat request."""
        ...


def configured_models(config: Path) -> dict[str, dict[str, str | None]]:
    """Return ``{role: {"name": ..., "digest": ...}}`` from the config file."""
    loaded = load_models_config(config)
    return {
        role: {"name": loaded.spec(role).name, "digest": loaded.spec(role).digest}
        for role in ROLES
    }


def local_digests() -> dict[str, str]:
    """Return ``{model name: digest}`` for every model the local Ollama holds."""
    return {m.model: m.digest for m in ollama.list().models if m.model and m.digest}


def compare(configured: Mapping[str, ModelSpec], local: Mapping[str, str]) -> list[str]:
    """Return one problem per configured model whose digest is missing or differs."""
    problems = []
    for role, spec in configured.items():
        name, expected = spec.get("name"), spec.get("digest")
        if not name:
            problems.append(f"{role}: no model name in the config")
        elif name not in local:
            problems.append(f"{role}: {name} is not pulled")
        elif not expected:
            problems.append(
                f"{role}: {name} has no digest in the config (local {local[name][:12]})"
            )
        elif local[name] != expected:
            problems.append(
                f"{role}: {name} digest differs, config {expected[:12]} vs local "
                f"{local[name][:12]}"
            )
    return problems


def pull(configured: Mapping[str, ModelSpec]) -> int:
    """Run ``ollama pull`` for every configured model."""
    for role, spec in configured.items():
        name = spec.get("name")
        if not name:
            print(f"{role}: no model name in the config", file=sys.stderr)
            return 1
        subprocess.run(["ollama", "pull", name], check=True)
    return 0


def digests(configured: Mapping[str, ModelSpec], local: Mapping[str, str]) -> int:
    """Print role, name and local digest of every configured model."""
    for role, spec in configured.items():
        name = spec.get("name") or "-"
        print(f"{role:<10} {name:<16} {local.get(name, 'not pulled')}")
    return 0


def verify(configured: Mapping[str, ModelSpec], local: Mapping[str, str]) -> int:
    """Exit non-zero with one line per problem unless every digest matches."""
    problems = compare(configured, local)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"all {len(configured)} configured models match the recorded digests")
    return 0


def extra_loaded_models(resident: Sequence[str], judge_name: str) -> list[str]:
    """Return resident model names that are not the configured judge, once each."""
    extras: list[str] = []
    seen: set[str] = set()
    for name in resident:
        if not name or name == judge_name or name in seen:
            continue
        extras.append(name)
        seen.add(name)
    return extras


def extra_loaded_warning(extras: Sequence[str]) -> str:
    """Return the observational warning naming the stop command for each extra."""
    stops = "; ".join(f"`ollama stop {name}`" for name in extras)
    return (
        "warning: models other than the judge are resident: "
        + ", ".join(extras)
        + f". Stop them explicitly before a memory-tight eval: {stops}. "
        "just eval does not stop models."
    )


def resident_model_names(processes: Sequence[object] | None = None) -> list[str]:
    """Return the names reported by ``ollama ps``, or by ``processes`` in tests."""
    if processes is None:
        processes = ollama.ps().models
    names: list[str] = []
    for item in processes:
        if isinstance(item, str):
            name = item
        else:
            name = getattr(item, "model", None) or getattr(item, "name", None)
        if name:
            names.append(name)
    return names


def eval_preflight(
    config: Path,
    *,
    resident: Sequence[str] | None = None,
) -> int:
    """Print the judge model and current Ollama residency; warn, do not stop.

    Observational: extra resident models are a warning on stderr and the process
    still exits 0. ``sim eval`` does not unload anything.
    """
    loaded = load_models_config(config)
    judge_name = loaded.spec("judge").name
    print(f"judge model: {judge_name}")
    print("resident models (`ollama ps`):")
    completed = subprocess.run(["ollama", "ps"], check=False)
    if completed.returncode != 0:
        print(
            f"warning: `ollama ps` exited {completed.returncode}",
            file=sys.stderr,
        )
    if resident is None:
        try:
            resident = resident_model_names()
        except Exception as failure:
            print(
                f"warning: could not list resident models via the Ollama API: "
                f"{failure}",
                file=sys.stderr,
            )
            resident = []
    extras = extra_loaded_models(resident, judge_name)
    if extras:
        print(extra_loaded_warning(extras), file=sys.stderr)
    else:
        print("no extra models reported besides the judge (or none listed)")
    return 0


def warmup_judge(config: Path, *, client: WarmupClient | None = None) -> int:
    """Load the configured judge with ``num_ctx``, outside the experiment cache.

    Goes through Ollama directly, never :class:`sim.llm.LlmClient`, so it cannot
    write ``runs/*/cache`` or ``llm_calls.jsonl``.
    """
    loaded = load_models_config(config)
    spec = loaded.spec("judge")
    transport = client if client is not None else ollama.Client()
    transport.chat(
        model=spec.name,
        messages=[{"role": "user", "content": "ok"}],
        options={
            "num_ctx": loaded.num_ctx,
            "num_predict": 1,
            "temperature": 0,
        },
        think=False,
    )
    print(f"warmed {spec.name} (num_ctx={loaded.num_ctx}, num_predict=1)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command",
        choices=("pull", "digests", "verify", "eval-preflight", "warmup-judge"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.command == "eval-preflight":
        return eval_preflight(args.config)
    if args.command == "warmup-judge":
        return warmup_judge(args.config)
    configured = configured_models(args.config)
    if args.command == "pull":
        return pull(configured)
    local = local_digests()
    if args.command == "digests":
        return digests(configured, local)
    return verify(configured, local)


if __name__ == "__main__":
    raise SystemExit(main())
