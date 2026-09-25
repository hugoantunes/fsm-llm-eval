# Data audit (T-18)

> **Experiment guide** · step 9 of 11 · [All steps](README.md) ·
> [← Execution](execution.md) · [Next: Decomposing `fact_f1` →](decomposition.md)

Census of `runs/exp_final/` after T-17. The manifest defines the audit
universe. Dialogue logs define execution outcome; metric files define
inclusion. T-18 never rewrites execution history: contract false positives
remain failed executions but are included in the semantic primary only
through the adjudicated sidecar.

Eligibility was frozen in T-17 before judge evaluation. T-18 does not
redefine those populations. 340 and 348 are scored/exported subsets of the
previously frozen eligible populations, not the frozen experimental
populations.

Preserve judge provenance exactly as recorded. T-18 performs no
cross-backend aggregation or comparison; treatment of the MLX/GGUF split is
deferred to T-23.

Re-export the audited scored CSV copies with `just export-metrics`. Default
destination is `results/<exp_id>/` from the run manifest (`results/exp_final/`
here). T-18 validates census, inclusion, ID sets, and hashes, then performs
byte-identical copies of the already-scored run CSVs. It does not rerun
scoring or recompute metrics. Export fails closed on any unexplained,
duplicate, missing, or set-mismatched job.

## Frozen eligibility census

```text
frozen-gate eligible              342
semantic-primary eligible         350
```

```text
342 frozen-gate eligible
+ 8 contract false positives admitted via adjudicated sidecar
= 350 semantic-primary eligible
```

## Observed evaluation/export outputs

```text
frozen-gate scored                340
semantic-primary scored           348
locked eligible-but-unscored        2
semantic CFP rows                   8
```

```text
342 frozen eligible = 340 scored + 2 eligible-but-unscored
350 semantic eligible = 348 scored + 2 eligible-but-unscored
```

## Human census of the scored semantic-primary set

The 348 scored semantic-primary rows are the universe of the later blinded
human census (`results/human_validation/exp_final/`). The two
eligible-but-unscored dialogues were not annotated. Derived scores:
`results/human_primary/`. This does not change T-18 inclusion.

## Complete execution census (360 jobs)

```text
360
= 342 frozen-gate eligible
+ 8 CFP admitted only to semantic eligibility
+ 9 max_turns_with_incomplete_beat
+ 1 instrument_true_failure
```

## Exclusions

These IDs stay out of one or both exported scored CSVs. They are named
classes, not integrity anomalies.

### eligible-but-unscored (`excluded_unscored`)

Members of both T-17 eligible populations (342 and 350). Eval could not
score them: the stage labeler returned 5 labels for 4 turns. Not retried.
Not retroactively removed from the eligible populations.

| Dialogue | `execution_status` | `primary_status` | `frozen_status` |
|---|---|---|---|
| `edge_01__fsm__rep02` (`edge_01` / fsm / 2) | ok | excluded_unscored | excluded_unscored |
| `edge_13__fsm__rep02` (`edge_13` / fsm / 2) | ok | excluded_unscored | excluded_unscored |

### `instrument_contract_false_positive`

Failed executions (`status=failed` on disk). Included in the semantic
sidecar only. Frozen-gate: `excluded_failure`.

`adversarial_01__baseline__rep03`, `edge_16__baseline__rep02`,
`edge_16__baseline__rep03`, `happy_path_09__baseline__rep02`,
`adversarial_19__fsm__rep02`, `edge_16__fsm__rep02`,
`edge_16__fsm__rep03`, `happy_path_09__fsm__rep02`.

### `instrument_true_failure`

`adversarial_06__fsm__rep02`. `excluded_failure` on both exports.

### `max_turns_with_incomplete_beat`

Nine simulation failures, not recoded as agent failures, `excluded_failure`
on both exports: `adversarial_07` baseline rep01/rep03; `adversarial_12`
baseline all three reps and FSM rep02/rep03; `adversarial_19` baseline
rep01; `happy_path_18` baseline rep01.

## Exported SHA-256

Full 64-character digests (no prefixes). Destination equals source.

| File | SHA-256 |
|---|---|
| semantic primary `results/exp_final/metrics.csv` (sidecar copy) | `29fe9712d8e4eaa29676daa795c9663dc5955ef6237bed59a8e6eb6a880d6e2b` |
| frozen-gate `results/exp_final/metrics_frozen_gate.csv` | `968bb4ad04aca98ba1d23dcccdfd9aa1fa8db547c2c2850483f09dc937d38534` |

## LLM-call events

Timeouts, `PromptTooLongError`, and retries in `llm_calls.jsonl` are an
independent event census. A dialogue may time out on an earlier attempt and
still complete. They do not by themselves exclude a job.

Live `exp_final` log: **9** events (4 timeout + 5 retry + 0
`PromptTooLongError`), none used as an exclusion.

T-18 counts one JSONL line per event. Dialogue ids are not embedded in the
judge prompt; the timeout ids below were recovered by matching the first
agent turn in the prompt transcript to `runs/exp_final/dialogues/`. The
operational log's "three 300 s timeouts" on `happy_path_12` / fsm / 2 is one
failed evaluate episode whose JSONL record has `attempts=3` — those three
HTTP attempts are not three of this census of four.

```text
2 timeout records on happy_path_12 / fsm / 2
  (same prompt_hash 9324171e83c4fbae…; 2026-09-15 08:09 UTC and 14:07 UTC;
   the 14:07 record is the 15:52 local attempt in docs/execution.md)
+ 1 timeout on adversarial_19 / baseline / 3 (01:39 UTC; later recovered)
+ 1 timeout on adversarial_08 / fsm / 1 (03:57 UTC; later recovered)
= 4 timeout events
```

| n | Type | Caller | Dialogue | Notes |
|---:|---|---|---|---|
| 2 | timeout | `judge_facts` | `happy_path_12` / fsm / 2 | `gemma4:12b`; each record `attempts=3`; later scored after `timeout_s=600` |
| 1 | timeout | `judge_facts` | `adversarial_19` / baseline / 3 | `gemma4:12b`; `attempts=3`; same prompt later completed |
| 1 | timeout | `judge_facts` | `adversarial_08` / fsm / 1 | `gemma4:12b`; `attempts=3`; same prompt later completed |
| 5 | retry | `judge_facts` | not in the JSONL event | `attempts` 2 or 3; no error on the recorded line |

Zero `PromptTooLongError`. Destination hashes still match the sources.

## Latency outliers

Inclusive Tukey 1.5 IQR per agent on finite `llm_latency_s` of each scored
set (`statistics.quantiles(..., n=4, method="inclusive")`). If a group has
fewer than 4 finite values, do not compute the outlier rule: flag no
outliers and record `insufficient_n`. Report-only: never excluded, never a
column of `results/exp_final/metrics.csv`. Sidecar table:
`results/exp_final/latency_outliers.csv`.

Live run: every group had n ≥ 4 (`insufficient_n` unused). **20** flags
(10 semantic + 10 frozen; a dialogue can appear in both). None dropped.

| Population | Agent | n scored | Outliers |
|---|---|---:|---:|
| semantic | baseline | 173 | 7 |
| semantic | fsm | 175 | 3 |
| frozen | baseline | 169 | 6 |
| frozen | fsm | 171 | 4 |
