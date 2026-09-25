"""Tests for the hook scripts in ai-assistance/scripts.

The same two scripts are wired in ``.claude/settings.json`` and in
``.cursor/hooks.json``, which send different payloads and expect different
answers, so the dispatch on ``hook_event_name`` is tested for both harnesses.
Only the decisions that could silently disable a gate are tested here: which
shell commands count as a commit, which edited files get formatted, whether a
red gate reaches the model, and what happens when ``just`` is missing.
"""

import io
import json
from pathlib import Path
from typing import NoReturn

import pytest

from helpers import load_script

quality_gate = load_script("ai-assistance/scripts/quality_gate.py")
format_on_edit = load_script("ai-assistance/scripts/format_on_edit.py")


@pytest.mark.parametrize(
    "command",
    [
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
        "git committee",
    ],
)
def test_other_commands_pass_through(command: str) -> None:
    assert not quality_gate.is_git_commit(command)


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


def test_cursor_sends_the_file_path_at_the_top_level(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n")
    payload = {"hook_event_name": "afterFileEdit", "file_path": str(target)}

    assert format_on_edit.edited_python_file(payload, tmp_path) == target.resolve()


def test_cursor_workspace_root_is_the_project_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    payload = {"hook_event_name": "afterFileEdit", "workspace_roots": [str(tmp_path)]}

    assert format_on_edit.project_dir(payload) == tmp_path.resolve()


def run_gate(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    *,
    check: tuple[int, str] = (0, ""),
    changes: bool = True,
) -> int:
    """Run the gate's ``main`` on ``payload``, with ``just check`` and git faked."""
    monkeypatch.setattr(quality_gate.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(quality_gate, "run_check", lambda _root: check)
    monkeypatch.setattr(quality_gate, "has_changes", lambda _root: changes)
    return quality_gate.main()


def test_claude_code_blocks_a_commit_when_the_gate_is_red(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_input": {"command": 'git commit -m "x"'},
    }

    code = run_gate(monkeypatch, payload, check=(1, "2 failed"))

    assert code == 2
    assert "2 failed" in capsys.readouterr().err


def test_cursor_denies_a_commit_when_the_gate_is_red(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "hook_event_name": "beforeShellExecution",
        "command": 'git commit -m "x"',
    }

    code = run_gate(monkeypatch, payload, check=(1, "2 failed"))

    answer = json.loads(capsys.readouterr().out)
    assert code == 0
    assert answer["permission"] == "deny"
    assert "2 failed" in answer["agent_message"]


def test_cursor_allows_a_command_that_is_not_a_commit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"hook_event_name": "beforeShellExecution", "command": "just test"}

    code = run_gate(monkeypatch, payload, check=(1, "2 failed"))

    assert code == 0
    assert json.loads(capsys.readouterr().out)["permission"] == "allow"


def test_cursor_stop_asks_for_a_followup_when_the_gate_is_red(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"hook_event_name": "stop", "status": "completed", "loop_count": 0}

    code = run_gate(monkeypatch, payload, check=(1, "2 failed"))

    assert code == 0
    assert "2 failed" in json.loads(capsys.readouterr().out)["followup_message"]


def test_cursor_stop_stays_quiet_when_the_user_aborted_the_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"hook_event_name": "stop", "status": "aborted", "loop_count": 0}

    code = run_gate(monkeypatch, payload, check=(1, "2 failed"))

    assert code == 0
    assert capsys.readouterr().out == ""


def test_cursor_stop_stays_quiet_when_the_tree_is_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"hook_event_name": "stop", "status": "completed", "loop_count": 0}

    code = run_gate(monkeypatch, payload, check=(1, "2 failed"), changes=False)

    assert code == 0
    assert capsys.readouterr().out == ""


def test_missing_just_is_reported_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> NoReturn:
        raise FileNotFoundError("just")

    monkeypatch.setattr(quality_gate.subprocess, "run", missing)

    code, output = quality_gate.run_check(Path())

    assert code == 127
    assert "brew install just" in output
