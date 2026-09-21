# Human-primary confirmatory analysis

Scored from the frozen human sheets in
`results/human_validation/exp_final/frozen/`. Those CSVs were not edited.
Unblinding used `private/mapping.csv` only after that freeze.

Population: the same 348 `semantic_primary` dialogues. Statistics reuse T-19
(`pair_scenarios`, Wilcoxon, Holm on `task_completed` / `fact_f1` /
`claim_support`, permutation, bootstrap, seed 0, 10 000 resamples). Frozen-gate
was not re-annotated.

## Instrument comparison (human vs judge)

Dialogue grain. `uncertain` is excluded from binary `task_completed` agreement.
Atomic claims are not matched by `claim_id` or text: the judge and the human
segment sentences differently. `claim_support` agreement uses a three-way label
derived from `n_checkable_claims` and `claim_support` (`all_supported` /
`some_unsupported` / `none_checkable`).

`fact_ids_stated` is the union of response-level IDs vs the judge predicted set
recovered from `llm_calls.jsonl` (must reproduce the frozen metrics row). Calls
that reproduce the row but disagree on predicted IDs are omitted, not guessed.

## Outputs

| File | Contents |
|---|---|
| `metrics.csv` | 348 human-derived rows |
| `paired.csv` | scenario-level FSM − baseline |
| `descriptive.csv` / `tests.csv` | T-19 tables, population `human_primary` |
| `agreement.json` | human vs judge |
| `conclusions.csv` | primary-family direction and Holm significance, judge vs human |
| `environment.txt` | Python and analysis-package versions |
| `tables/` `figures/` | Portuguese plates (`just figures results/human_primary`) |

Regenerate scores: `just score-human-validation`.
Regenerate plates: `just figures results/human_primary`. Do not edit `frozen/`.
