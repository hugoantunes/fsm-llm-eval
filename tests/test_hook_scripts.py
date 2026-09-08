"""Tests for the hook scripts in ai-assistance/scripts, wired in .claude/settings.json.

Only the decisions that could silently disable a gate are tested here: which shell
commands count as a commit, which edited files get formatted, and what happens when
``just`` is missing.
"""

from pathlib import Path
from typing import NoReturn

import pytest

from helpers import load_script

quality_gate = load_script("ai-assistance/scripts/quality_gate.py")
format_on_edit = load_script("ai-assistance/scripts/format_on_edit.py")


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "T-07: add client"',
        'git add -A && git commit -m "x"',
        "git -C /tmp/repo commit --amend --no-edit",
        "cd src && git commit",
    ],
)
def test_git_commit_commands_are_gated(command: str) -> None:
    assert quality_gate.is_git_commit(command)


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        "git log --oneline -5",
        "git diff HEAD -- '*.py'",
        "just test",
        "git committee",
    ],
)
def test_other_commands_pass_through(command: str) -> None:
    assert not quality_gate.is_git_commit(command)


def test_python_file_inside_project_is_formatted(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "mod.py"
    target.parent.mkdir()
    target.write_text("x = 1\n")
    payload = {"tool_input": {"file_path": str(target)}}

    assert format_on_edit.edited_python_file(payload, tmp_path) == target.resolve()


def test_relative_path_resolves_against_project(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n")
    payload = {"tool_input": {"file_path": "mod.py"}}

    result = format_on_edit.edited_python_file(payload, tmp_path)

    assert result == (tmp_path / "mod.py").resolve()


def test_non_python_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# x\n")
    payload = {"tool_input": {"file_path": str(tmp_path / "notes.md")}}

    assert format_on_edit.edited_python_file(payload, tmp_path) is None


def test_python_file_outside_project_is_ignored(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "elsewhere.py"
    outside.write_text("x = 1\n")
    payload = {"tool_input": {"file_path": str(outside)}}

    assert format_on_edit.edited_python_file(payload, project) is None


def test_payload_without_file_path_is_ignored(tmp_path: Path) -> None:
    assert format_on_edit.edited_python_file({"tool_input": {}}, tmp_path) is None


def test_missing_just_is_reported_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> NoReturn:
        raise FileNotFoundError("just")

    monkeypatch.setattr(quality_gate.subprocess, "run", missing)

    code, output = quality_gate.run_check(Path())

    assert code == 127
    assert "brew install just" in output
