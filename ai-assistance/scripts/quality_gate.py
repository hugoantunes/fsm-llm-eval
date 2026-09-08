"""Commit and turn-end hook: nothing red gets committed or handed back.

The gate is always ``just check`` (lint plus unit tests), the same command a
person runs by hand. Four events reach this script, told apart by
``hook_event_name``; Claude Code and Cursor want the answer in different ways,
and Cursor's ``stop`` differs from Claude Code's ``Stop`` only in case:

* ``PreToolUse`` (Claude Code, on ``Bash``) and ``beforeShellExecution``
  (Cursor). When the command is a ``git commit``, the gate runs first. Claude
  Code blocks on exit 2 with the output on stderr; Cursor blocks on a
  ``permission: deny`` answer carrying the same output in ``agent_message``.
  Any other command is let through at once.
* ``Stop`` (Claude Code) and ``stop`` (Cursor). When the working tree differs
  from ``HEAD`` the gate runs, so a turn does not end on a red suite. Claude
  Code exits 2 with the output and the model keeps working; Cursor cannot block
  a turn, so the output comes back as a ``followup_message`` that restarts it.
  A failure the model cannot fix must not trap the turn in a loop. Claude Code
  marks its own retry in ``stop_hook_active`` and it is let through; Cursor
  counts the follow-ups itself and stops at the ``loop_limit`` of
  ``.cursor/hooks.json``, so counting again here would only disable the gate
  for the rest of the conversation. What Cursor does need is ``status``: a turn
  the user aborted is never followed up, or the hook would fight the interrupt.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

GIT_COMMIT = re.compile(r"\bgit\b(?:\s+(?:-C\s+\S+|--?[\w-]+(?:=\S+)?))*\s+commit\b")
GATE = ("just", "check")
GATE_LINE = " ".join(GATE)
COMMIT_FIX = "fix lint and tests before committing"
TURN_FIX = "fix it before ending the turn, or say why you cannot"
MISSING_JUST = (
    "just: command not found. Install it with `brew install just` "
    "(or run `uvx --from rust-just just check` by hand)."
)
TAIL_LINES = 60


def project_dir(payload: dict) -> Path:
    """Return the project root, from the environment or from either payload shape."""
    roots = payload.get("workspace_roots") or []
    raw = (
        os.environ.get("CLAUDE_PROJECT_DIR")
        or (roots[0] if roots else None)
        or payload.get("cwd")
        or Path.cwd()
    )
    return Path(raw).resolve()


def answer(**fields: str) -> int:
    """Print a Cursor hook answer as JSON on stdout and succeed."""
    print(json.dumps(fields))
    return 0


def failure(instruction: str, output: str) -> str:
    """Return the message a red gate hands to the model, whatever the harness."""
    return f"{GATE_LINE} failed: {instruction}\n{output}"


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
    print(failure(instruction, output), file=sys.stderr)
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
        return gate(root, COMMIT_FIX)

    if event == "beforeShellExecution":
        if not is_git_commit(payload.get("command", "")):
            return answer(permission="allow")
        code, output = run_check(root)
        if code == 0:
            return answer(permission="allow")
        return answer(
            permission="deny",
            user_message=f"{GATE_LINE} is red, so the commit was blocked.",
            agent_message=failure(COMMIT_FIX, output),
        )

    if event == "Stop":
        if payload.get("stop_hook_active") or not has_changes(root):
            return 0
        return gate(root, TURN_FIX)

    if event == "stop":
        if payload.get("status") != "completed" or not has_changes(root):
            return 0
        code, output = run_check(root)
        if code == 0:
            return 0
        return answer(followup_message=failure(TURN_FIX, output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
