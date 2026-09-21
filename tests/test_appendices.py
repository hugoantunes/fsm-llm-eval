"""Paste-ready thesis appendices A-D (T-22)."""

import shutil
from pathlib import Path

import pytest

from helpers import REPO_ROOT
from sim.appendices import (
    CHARS_PER_PAGE,
    ELLIPSIS,
    PAGE_BUDGETS,
    AppendicesError,
    char_budget,
    check_budget,
    page_chars,
    write_appendices,
)
from sim.fsm import load_fsm
from sim.prompts import load_prompt


def test_page_budget_counts_characters_and_rejects_overage() -> None:
    sample = "Paste as: Apêndice A\n\n```mermaid\n[...]\n```\n"

    assert page_chars(sample) == len(sample)
    assert CHARS_PER_PAGE == 3000
    assert PAGE_BUDGETS == {"a": 0.5, "b": 1.5, "c": 1.0, "d": 2.0}
    assert char_budget("a") == 1500
    assert char_budget("b") == 4500
    assert char_budget("c") == 3000
    assert char_budget("d") == 6000

    check_budget("a", "x" * 1500)
    with pytest.raises(AppendicesError, match="1500"):
        check_budget("a", "x" * 1501)


def test_write_appendices_writes_a_through_d(tmp_path: Path) -> None:
    write_appendices(tmp_path)

    assert (tmp_path / "a.md").is_file()
    assert (tmp_path / "b.md").is_file()
    assert (tmp_path / "c.md").is_file()
    assert (tmp_path / "d.md").is_file()


def test_appendix_a_embeds_mermaid_from_the_spec(tmp_path: Path) -> None:
    spec = load_fsm()

    write_appendices(tmp_path)
    text = (tmp_path / "a.md").read_text(encoding="utf-8")

    assert text.startswith("Paste as: Apêndice A")
    assert f"```mermaid\n{spec.to_mermaid()}\n```" in text


def test_appendix_a_fits_half_page(tmp_path: Path) -> None:
    write_appendices(tmp_path)
    text = (tmp_path / "a.md").read_text(encoding="utf-8")

    assert page_chars(text) <= char_budget("a")
    check_budget("a", text)


def test_appendix_b_references_shared_and_quotes_both_templates_without_rendered_kb(
    tmp_path: Path,
) -> None:
    write_appendices(tmp_path)
    text = (tmp_path / "b.md").read_text(encoding="utf-8")
    shared = load_prompt("agent_shared").template
    spec = load_fsm()
    packages = [Path(state.package).name for state in spec.states.values()]

    assert "$knowledge_base" in text
    assert "$user_data_fields" in text
    assert "**F01**" not in text
    assert "agent_shared.md" in text
    assert "$shared" in text
    assert shared not in text
    assert "# How to handle a request" in text
    assert "Greet the customer once" in text
    assert "# Current state" in text
    assert "$state_package" in text
    for name in packages:
        assert name in text
    assert "## Never in this state" not in text


def test_appendix_b_fits_page_and_a_half(tmp_path: Path) -> None:
    write_appendices(tmp_path)
    text = (tmp_path / "b.md").read_text(encoding="utf-8")

    assert page_chars(text) <= char_budget("b")
    check_budget("b", text)


def test_appendix_b_raises_when_expected_heading_is_missing(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    shutil.copytree(Path("data/prompts"), prompts)
    baseline = prompts / "baseline.md"
    baseline.write_text(
        baseline.read_text(encoding="utf-8").replace(
            "# How to handle a request", "# Procedure"
        ),
        encoding="utf-8",
    )

    with pytest.raises(AppendicesError, match="How to handle a request"):
        write_appendices(tmp_path / "out", prompts_dir=prompts)


def test_appendix_c_quotes_blindness_and_selected_rubric_fields_without_output_schema(
    tmp_path: Path,
) -> None:
    write_appendices(tmp_path)
    text = (tmp_path / "c.md").read_text(encoding="utf-8")

    assert "Paste as: Apêndice C" in text
    assert "You grade a customer-support chat" in text
    assert "do not know which system" in text
    assert "# Claims" in text
    assert "supported_by_kb" in text
    assert "`yes`" in text
    assert "`no`" in text
    assert "`unverifiable`" in text
    assert "# Accuracy" in text
    assert "`correct`" in text
    assert "# Relevance" in text
    assert "# Task completed" in text
    assert "task_completed" in text
    assert "# Offensive content" in text
    assert "offensive_content" in text
    assert "# Output schema" not in text
    assert '"additionalProperties"' not in text
    assert "```json" not in text


def test_appendix_c_fits_one_page(tmp_path: Path) -> None:
    write_appendices(tmp_path)
    text = (tmp_path / "c.md").read_text(encoding="utf-8")

    assert page_chars(text) <= char_budget("c")
    check_budget("c", text)


def test_appendix_c_raises_when_expected_heading_is_missing(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    shutil.copytree(Path("data/prompts"), prompts)
    facts = prompts / "judge_facts.md"
    facts.write_text(
        facts.read_text(encoding="utf-8").replace("# Claims", "# Assertions"),
        encoding="utf-8",
    )

    with pytest.raises(AppendicesError, match="Claims"):
        write_appendices(tmp_path / "out", prompts_dir=prompts)


def test_appendix_d_quotes_t21_exemplar_pairs(tmp_path: Path) -> None:
    write_appendices(tmp_path)
    text = (tmp_path / "d.md").read_text(encoding="utf-8")

    assert "Paste as: Apêndice D" in text
    assert "adversarial_08" in text
    assert "edge_05" in text
    assert "happy_path_07" in text
    assert "Diagnose this rash. Which cream." in text
    assert "I need to send something back." in text
    assert "Please exchange the blue jacket" in text
    assert "### Baseline" in text
    assert "### FSM" in text
    assert "Category directions" not in text
    assert "Observed limitations" not in text
    assert "rank-biserial" not in text


def _write_cases(
    path: Path,
    *,
    helped_fsm: str,
    helped_baseline: str = "user: hello\nagent: hi",
    restricted_baseline: str = "user: send back\nagent: 30 days",
    restricted_fsm: str = "user: send back\nagent: window",
    tie_baseline: str = "user: exchange\nagent: credit",
    tie_fsm: str = "user: exchange\nagent: 12 months",
) -> Path:
    def pair(heading: str, scenario_id: str, baseline: str, fsm: str) -> str:
        return (
            f"{heading} (repetition 1)\n\n"
            f"### Comment\n\nA comment.\n\n"
            f"### Baseline (`{scenario_id}__baseline__rep01`)\n\n"
            f"```text\n{baseline}\n```\n\n"
            f"### FSM (`{scenario_id}__fsm__rep01`)\n\n"
            f"```text\n{fsm}\n```\n"
        )

    path.write_text(
        "# Trade-off cases\n\n"
        + pair(
            "## FSM helped: `adversarial_08`",
            "adversarial_08",
            helped_baseline,
            helped_fsm,
        )
        + pair(
            "## FSM restricted: `edge_05`",
            "edge_05",
            restricted_baseline,
            restricted_fsm,
        )
        + pair("## Tie: `happy_path_07`", "happy_path_07", tie_baseline, tie_fsm),
        encoding="utf-8",
    )
    return path


def test_appendix_d_uses_full_pairs_when_within_budget(tmp_path: Path) -> None:
    cases = _write_cases(
        tmp_path / "cases.md",
        helped_fsm="user: rash\nagent: clothing footwear home goods",
    )

    write_appendices(tmp_path / "out", cases_path=cases)
    text = (tmp_path / "out" / "d.md").read_text(encoding="utf-8")

    assert ELLIPSIS not in text
    assert "user: rash" in text
    assert "agent: clothing footwear home goods" in text
    assert "user: send back" in text
    assert "user: exchange" in text
    assert page_chars(text) <= char_budget("d")


def test_appendix_d_truncates_only_helped_fsm_with_literal_ellipsis(
    tmp_path: Path,
) -> None:
    first = "user: FIRST_HELPED_FSM"
    last = "agent: LAST_HELPED_FSM"
    middle = "\n".join(f"agent: PAD_{i} " + ("x" * 100) for i in range(80))
    helped_fsm = f"{first}\n{middle}\n{last}"
    restricted_fsm = "user: KEEP_RESTRICTED\nagent: KEEP_RESTRICTED_AGENT"
    cases = _write_cases(
        tmp_path / "cases.md",
        helped_fsm=helped_fsm,
        restricted_fsm=restricted_fsm,
    )

    write_appendices(tmp_path / "out", cases_path=cases)
    text = (tmp_path / "out" / "d.md").read_text(encoding="utf-8")
    helped_block = text.split("## FSM restricted:")[0]

    assert ELLIPSIS in helped_block
    assert f"{first}\n{ELLIPSIS}\n{last}" in helped_block
    assert "PAD_0" not in text
    assert "KEEP_RESTRICTED_AGENT" in text
    assert "user: send back" in text
    assert "user: exchange" in text
    assert page_chars(text) <= char_budget("d")


def test_appendix_d_raises_when_truncated_helped_fsm_still_exceeds_budget(
    tmp_path: Path,
) -> None:
    pad = "\n".join(f"agent: HUGE_{i} " + ("y" * 120) for i in range(20))
    cases = _write_cases(
        tmp_path / "cases.md",
        helped_fsm=f"user: first\n{pad}\nagent: last",
        helped_baseline=pad,
        restricted_baseline=pad,
        restricted_fsm=pad,
        tie_baseline=pad,
        tie_fsm=pad,
    )

    with pytest.raises(AppendicesError, match="6000"):
        write_appendices(tmp_path / "out", cases_path=cases)


def test_appendix_d_fits_two_pages(tmp_path: Path) -> None:
    write_appendices(tmp_path)
    text = (tmp_path / "d.md").read_text(encoding="utf-8")

    assert page_chars(text) <= char_budget("d")
    check_budget("d", text)


def test_appendix_d_raises_when_expected_heading_is_missing(tmp_path: Path) -> None:
    cases = tmp_path / "cases.md"
    cases.write_text(
        Path("results/cases.md")
        .read_text(encoding="utf-8")
        .replace(
            "## FSM helped: `adversarial_08`",
            "## FSM helped: `missing_08`",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AppendicesError, match="adversarial_08"):
        write_appendices(tmp_path / "out", cases_path=cases)


def test_committed_appendices_match_writer(tmp_path: Path) -> None:
    write_appendices(tmp_path)
    docs = REPO_ROOT / "docs" / "appendices"
    for letter in "abcd":
        committed = (docs / f"{letter}.md").read_bytes()
        generated = (tmp_path / f"{letter}.md").read_bytes()
        assert committed == generated, f"{letter}.md is stale: run just appendices"
