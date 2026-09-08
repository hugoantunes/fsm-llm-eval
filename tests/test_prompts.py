"""Tests for the prompt loader of T-09: version line and safe rendering."""

from pathlib import Path

import pytest

from sim.prompts import PromptError, load_prompt


def write_prompt(directory: Path, name: str, text: str) -> Path:
    """Write a synthetic prompt file into ``directory`` and return its path."""
    path = directory / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_prompt_file_exposes_its_version_and_its_template(tmp_path: Path) -> None:
    write_prompt(tmp_path, "baseline", "Version: 3\n\nYou are $who, and you help.\n")

    prompt = load_prompt("baseline", directory=tmp_path)

    assert prompt.name == "baseline"
    assert prompt.version == 3
    assert prompt.template == "You are $who, and you help."


def test_rendering_fills_every_placeholder(tmp_path: Path) -> None:
    write_prompt(tmp_path, "baseline", "Version: 1\n\n$shared\n\n$knowledge_base\n")
    prompt = load_prompt("baseline", directory=tmp_path)

    rendered = prompt.render(shared="You are Nora.", knowledge_base="- F01 — a fact.")

    assert rendered == "You are Nora.\n\n- F01 — a fact."


def test_a_placeholder_left_unfilled_raises(tmp_path: Path) -> None:
    write_prompt(tmp_path, "baseline", "Version: 1\n\n$shared\n\n$knowledge_base\n")
    prompt = load_prompt("baseline", directory=tmp_path)

    with pytest.raises(PromptError, match="knowledge_base"):
        prompt.render(shared="You are Nora.")


def test_a_field_the_template_does_not_use_raises(tmp_path: Path) -> None:
    write_prompt(tmp_path, "baseline", "Version: 1\n\n$shared\n")
    prompt = load_prompt("baseline", directory=tmp_path)

    with pytest.raises(PromptError, match="knowledge_base"):
        prompt.render(shared="You are Nora.", knowledge_base="- F01 — a fact.")


def test_a_prompt_without_a_version_line_raises(tmp_path: Path) -> None:
    write_prompt(tmp_path, "baseline", "You are $who, and you help.\n")

    with pytest.raises(PromptError, match="Version:"):
        load_prompt("baseline", directory=tmp_path)
