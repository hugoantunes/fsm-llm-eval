"""Edit hook: format and lint-fix a Python file right after it is written.

Wired as ``PostToolUse`` in Claude Code and as ``afterFileEdit`` in Cursor; both
pipe the payload through stdin, and the two shapes are read here so the rule
lives in one script. For a ``.py`` file inside the project this runs
``ruff format`` and then ``ruff check --fix`` on that file. A clean result exits
0 without output: the harness itself tells the model when a file changed on
disk. A syntax error or a violation ruff cannot fix goes to stderr with exit 2,
which hands it to the model in Claude Code; Cursor only logs it, and there the
stop hook's ``just check`` is what catches the leftover.

Unused imports (F401) are reported but never removed here. Edits land one at a
time, and an import added in one edit is unused until the next edit uses it;
deleting it in between breaks the file the model is still writing. ``just format``
on a finished tree still removes them.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


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


def edited_python_file(payload: dict, root: Path) -> Path | None:
    """Return the edited file when it is an existing Python file inside ``root``."""
    raw = payload.get("file_path") or (payload.get("tool_input") or {}).get("file_path")
    if not raw:
        return None
    root = root.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.suffix != ".py" or not path.is_file() or not path.is_relative_to(root):
        return None
    return path


def ruff(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ruff from the project environment and capture its output."""
    return subprocess.run(
        ["uv", "run", "--no-sync", "-q", "ruff", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def main() -> int:
    """Format the edited file and report only what ruff could not fix."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    root = project_dir(payload)
    path = edited_python_file(payload, root)
    if path is None:
        return 0

    problems = []
    formatted = ruff(root, "format", str(path))
    if formatted.returncode != 0:
        problems.append(formatted.stderr.strip() or formatted.stdout.strip())
    checked = ruff(root, "check", "--fix", "--unfixable", "F401", str(path))
    if checked.returncode != 0:
        problems.append(checked.stdout.strip() or checked.stderr.strip())
    if not problems:
        return 0

    print(
        f"ruff could not fix everything in {path.relative_to(root)}:", file=sys.stderr
    )
    print("\n".join(problems), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
