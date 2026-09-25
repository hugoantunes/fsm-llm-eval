"""Tests for scripts/prompt_tokens.py: prompt size per agent condition (item 1.3)."""

import csv
from pathlib import Path

from helpers import load_script, make_llm_call_record, write_llm_calls

prompt_tokens = load_script("scripts/prompt_tokens.py")


def _call(caller: str, tokens: int, chars: int, **fields: object):
    return make_llm_call_record(
        caller=caller,
        prompt_tokens=tokens,
        messages=[{"role": "system", "content": "x" * chars}],
        **fields,
    )


def test_summarize_reports_size_and_tokens_per_char_per_condition() -> None:
    calls = [
        _call("baseline", 100, 400),
        _call("baseline", 200, 1000),
        _call("fsm", 90, 300),
    ]

    rows = prompt_tokens.summarize(calls, budget=150)

    baseline = rows["baseline"]
    assert baseline.n_calls == 2
    assert baseline.mean_prompt_tokens == 150
    assert baseline.max_prompt_tokens == 200
    assert baseline.min_tokens_per_char == 0.2
    assert baseline.max_tokens_per_char == 0.25
    assert baseline.over_budget == 1
    assert rows["fsm"].max_prompt_tokens == 90


def test_summarize_skips_failed_calls_and_other_roles() -> None:
    calls = [
        _call("baseline", 100, 400),
        _call("baseline", 0, 400, error="timed out"),
        make_llm_call_record(caller="classifier", role="classifier", prompt_tokens=9),
    ]

    rows = prompt_tokens.summarize(calls, budget=6553)

    assert set(rows) == {"baseline"}
    assert rows["baseline"].n_calls == 1


def test_main_writes_one_csv_row_per_condition(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp"
    run_dir.mkdir(parents=True)
    write_llm_calls(run_dir, [_call("baseline", 100, 400), _call("fsm", 80, 400)])
    out = tmp_path / "prompt_tokens.csv"

    assert prompt_tokens.main([str(run_dir), "--out", str(out)]) == 0

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [row["condition"] for row in rows] == ["baseline", "fsm"]
    assert rows[0]["num_ctx"] == "8192"
    assert rows[0]["budget"] == "6553"
