"""Pull and verify the models of ``configs/models.yaml`` against the local Ollama.

Usage::

    uv run python scripts/models.py pull      # ollama pull every configured model
    uv run python scripts/models.py digests   # print role, name and local digest
    uv run python scripts/models.py verify    # exit 1 unless every digest matches

``verify`` is the reproducibility check of T-03: both machines must hold the same
digests, and those must be the ones recorded in the config.
"""

import argparse
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import ollama
import yaml

ROLES = ("agent", "simulator", "judge")
ModelSpec = Mapping[str, str | None]


def configured_models(config: Path) -> dict[str, dict[str, str | None]]:
    """Return ``{role: {"name": ..., "digest": ...}}`` from the config file."""
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    return {role: dict(data["models"][role]) for role in ROLES}


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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("pull", "digests", "verify"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    args = parser.parse_args(argv)
    configured = configured_models(args.config)
    if args.command == "pull":
        return pull(configured)
    local = local_digests()
    if args.command == "digests":
        return digests(configured, local)
    return verify(configured, local)


if __name__ == "__main__":
    raise SystemExit(main())
