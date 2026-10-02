# Sensitivity: primary tests without the ambiguous-fact-ID dialogues (T-25)

Post-hoc and exploratory. The advisor's review asked whether the paired tests
hold without the census dialogues whose judge fact IDs are ambiguous. Nothing
here changes `results/exp_final/metrics.csv`, the frozen human sheets, the
confirmatory family or the thesis tables built from them.

## What the excluded dialogues are

`results/human_primary/ambiguous_fact_ids.csv`, written by
`just score-human-validation`, the same path that counts
`n_omitted_ambiguous_judge_calls` in `results/human_primary/agreement.json`.
Both judge configurations log to `runs/exp_final/llm_calls.jsonl`. In each listed
dialogue, the `gemma4:12b-mlx` call reproduces the frozen `metrics.csv` row, and
an extra `gemma4:12b` (GGUF) call reproduced the same score with other predicted
IDs. The score is not in doubt. Only which IDs the judge predicted is, so the
reconstruction omits the dialogue from the fact-ID agreement instead of guessing.

## Method

- Exclusion is per dialogue. A scenario keeps the mean of its remaining
  repetitions and leaves the pairing only if one agent has none left.
- The same exclusion applies to the judge rows (`results/exp_final/metrics.csv`)
  and to the human rows (`results/human_primary/metrics.csv`).
- T-19's `paired_test_table` on `task_completed`, `fact_f1` and `claim_support`,
  overall and per category. Seed 0, 10 000 resamples.
- Population `drop_ambiguous_fact_ids`, family `exploratory`. `wilcoxon_p_holm`
  on the overall rows is Holm over the three primaries of this population. It
  is a reference for comparison only, not a confirmatory test.

The reference rows are the overall primary rows of population
`semantic_primary` in `results/exp_final/tests.csv` (judge) and `human_primary`
in `results/human_primary/tests.csv` (human). With an empty list the output
reproduces them exactly (`tests/test_sensitivity.py`).

## Files

| File | Contents |
|---|---|
| `tests.csv` | `instrument` (`judge` / `human`) plus the T-19 test columns |
| `counts.csv` | dialogues and excluded dialogues per stratum and agent |

Regenerate: `just sensitivity-ambiguous`.
