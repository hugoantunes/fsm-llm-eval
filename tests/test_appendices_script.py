"""Tests for scripts/appendices.py (T-22)."""

from pathlib import Path

from helpers import load_script

appendices_script = load_script("scripts/appendices.py")


def test_appendices_script_writes_the_directory(tmp_path: Path) -> None:
    out = tmp_path / "appendices"

    assert appendices_script.main(["--out", str(out)]) == 0

    for letter in "abcd":
        assert (out / f"{letter}.md").is_file()
