"""PreToolUse and Stop hook: nothing red gets committed or handed back.

Claude Code runs this for two events, told apart by ``hook_event_name``:

* ``PreToolUse`` on ``Bash``. When the command is a ``git commit``, ``just check``
  runs first. A failure exits 2, which blocks the commit and shows the output to
  the model. Any other command exits 0 at once.
* ``Stop``. When the working tree differs from ``HEAD``, ``just check`` runs and a
  failure exits 2 with the output, so the model keeps working instead of ending
  its turn on a red suite. ``stop_hook_active`` marks the retry and is honored:
  a failure the model cannot fix does not trap the turn in a loop.

The gate is ``just check`` (lint plus unit tests) in both cases, so what blocks a
commit and what ends a turn is the same command a person runs by hand.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

GIT_COMMIT = re.compile(r"\bgit\b(?:\s+(?:-C\s+\S+|--?[\w-]+(?:=\S+)?))*\s+commit\b")
GATE = ("just", "check")
MISSING_JUST = (
    "just: command not found. Install it with `brew install just` "
    "(or run `uvx --from rust-just just check` by hand)."
)
TAIL_LINES = 60


def project_dir(payload: dict) -> Path:
    """Return the project root: ``CLAUDE_PROJECT_DIR``, else the payload's cwd."""
    raw = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or Path.cwd()
    return Path(raw).resolve()


def is_git_commit(command: str) -> bool:
    """Tell whether a shell command line contains a ``git commit`` invocation."""
    return GIT_COMMIT.search(command) is not None


def has_changes(root: Path) -> bool:
    """Tell whether the working tree differs from HEAD, untracked files included."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return status.returncode == 0 and bool(status.stdout.strip())


def run_check(root: Path) -> tuple[int, str]:
    """Run ``just check`` and return its exit code with the tail of its output."""
    try:
        result = subprocess.run(
            list(GATE),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=280,
        )
    except FileNotFoundError:
        return 127, MISSING_JUST
    lines = result.stdout.splitlines()
    return result.returncode, "\n".join(lines[-TAIL_LINES:])


def gate(root: Path, instruction: str) -> int:
    """Run the gate; on failure hand the output and ``instruction`` to the model."""
    code, output = run_check(root)
    if code == 0:
        return 0
    print(f"{' '.join(GATE)} failed: {instruction}", file=sys.stderr)
    print(output, file=sys.stderr)
    return 2


def main() -> int:
    """Dispatch on the hook event and return the process exit code."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    root = project_dir(payload)
    event = payload.get("hook_event_name")

    if event == "PreToolUse":
        command = (payload.get("tool_input") or {}).get("command", "")
        if not is_git_commit(command):
            return 0
        return gate(root, "fix lint and tests before committing")

    if event == "Stop":
        if payload.get("stop_hook_active") or not has_changes(root):
            return 0
        return gate(root, "fix it before ending the turn, or say why you cannot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
