# Pilot v1 `fact_ids_stated` directional audit

Diagnostic only. Frame: T-16 Pilot v1 (`results/judge_validation/pilot_v1/`,
logical run `runs/exp_pilot`, manifest `exp_pilot_fixes_20260913_final_v1`).
Human `fact_ids_stated` is the reference. Do not pool with Pilot v2. Do not
read T-17 scores.

**Pilot v1 demonstrates pooled fact-ID under-detection relative to human
annotation, but provides no recoverable evidence that such under-detection
was differential by condition. Therefore it cannot establish a mechanism
that selectively depressed FSM `fact_f1`.**

## Sources

Used:

- `results/judge_validation/pilot_v1/response_annotations.csv`
- `results/judge_validation/pilot_v1/sample.json`
- `docs/judge_validation.md` (T-16 Results: exact-set 0.733, micro P 1.000,
  R 0.818, F1 0.900; 19 unmatched claims)
- `docs/pilot.md` (`runs/exp_pilot` deleted, not recoverable)

Not used: Pilot v2 sheets, Pilot v2 11/15 vs 9/15 exact-set split, T-17
`metrics.csv`, reconstructed judge labels.

Missing: `runs/exp_pilot/llm_calls.jsonl` (`judge_facts`) and
`dialogues/*.jsonl`. Judge `fact_ids_stated` is the set of `fact_id`s on
turn-attributed `supported_by_kb == "yes"` claims
(`scripts/validate_judge.py`). Without that log, per-response judge sets
cannot be listed.

Join of the frozen sample to the response sheet: 30 rows, 15 baseline, 15
FSM. Human ID sets exist for every `Axx`. Judge ID sets exist for none.

## Pooled constraint

Published exact-set agreement 0.733 on n = 30 is 22/30 agree, **8**
disagree.

Micro precision 1.000 means total false positives = 0: no judge-only fact
ID occurs in any of the 30 responses. An exact-set mismatch therefore
cannot be `judge_fp` or `mixed`. Each of the 8 mismatches has at least one
human-only ID and no judge-only ID, so all 8 are **`judge_fn`**.

```text
micro precision = 1.000  →  FP total = 0
exact-set mismatches = 8  →  each mismatch differs
no judge-only ID is possible  →  each mismatch is judge_fn
```

```text
8 pooled judge_fn  ≠  FSM under-detection
```

The identities and condition allocation of all 8 mismatches remain
unrecoverable because `runs/exp_pilot` is unavailable.

Turn attribution is a potential additional source of disagreement:
19 claims matched zero or several turns and were excluded from forced
turn-level labels. There is no evidence that those unmatched claims
explain any of the 8 specific mismatches. T-17 `fact_f1` uses the
dialogue-level `claims[]` list, not this turn-attributed set.

## 1. Per-disagreement table

The eight mismatch `Axx` IDs are unknown. No disagreement rows are listed.
Judge IDs are not inferred from transcripts.

| dialogue/scenario | condition | response / turn | human `fact_ids_stated` | judge `fact_ids_stated` | human-only | judge-only | classification | evidence |
|---|---|---|---|---|---|---|---|---|
| *(none recoverable)* | — | — | — | unavailable | unavailable | unavailable | — | mismatch identities unpublished; `runs/exp_pilot` gone |

Human ID sets for the 30 sampled responses (identity context, not a
disagreement list) are in the appendix. Classification is not assigned per
row.

## 2. Condition summary

Denominators: baseline 15 sampled responses; FSM 15 sampled responses.
No significance tests. n is too small and the condition split is missing.

| condition | disagreements | judge_fn | judge_fp | mixed | ambiguous |
| --------- | ------------: | -------: | -------: | ----: | --------: |
| baseline  |   unavailable | unavailable | 0 pooled only | 0 pooled only | unavailable |
| FSM       |   unavailable | unavailable | 0 pooled only | 0 pooled only | unavailable |
| pooled    |             8 |           8 |             0 |             0 |           0 |

Per-response judge sets are unavailable for all 30 sampled responses (15
baseline, 15 FSM), so disagreement identities and their condition
allocation cannot be recovered.

`ambiguous` is a disagreement class, not a missing-data bin. Pooled
`judge_fp` and `mixed` are 0 because micro precision is 1.000, not because
those rows were inspected. Per-condition `judge_fn` remains unavailable.

## 3. Interpretation

1. **Are FSM disagreements predominantly `judge_fn`?** Unknown. Pooled, all
   eight mismatches are `judge_fn`. That is not an FSM finding.
2. **Are `judge_fn` errors proportionally more common in FSM than
   baseline?** Not estimable. The 11/15 vs 9/15 exact-set counts belong to
   Pilot v2 and are out of scope here.
3. **Is there evidence that the judge systematically under-detects facts
   in FSM responses?** No recoverable condition-level evidence. Pooled
   recall 0.818 is consistent with under-detection versus the human
   overall; turn-attribution error is a potential additional source of
   disagreement because 19 claims were unmatched. That is not systematic
   under-detection against FSM.
4. **Could the observed disagreement pattern plausibly depress FSM
   `fact_f1`?** Not from this audit. Response-level `judge_fn` is
   turn-attributed; T-17 `fact_f1` is dialogue-level. Without the condition
   split, Pilot v1 does not establish a mechanism that would selectively
   lower FSM `fact_f1`.

`claim_support` is a different field. Pilot v1 has no published condition
split for it; it is not used here.

## Appendix: sampled responses (human IDs only)

Judge columns are unavailable for every row.

| ID | condition | dialogue | turn | human `fact_ids_stated` |
|---|---|---|---:|---|
| A01 | fsm | `adversarial_04__fsm__rep01` | 2 | F01; F03; F04 |
| A02 | baseline | `adversarial_04__baseline__rep02` | 1 | F01; F03; F04 |
| A03 | baseline | `edge_02__baseline__rep02` | 3 | F12 |
| A04 | fsm | `edge_02__fsm__rep01` | 2 | F03; F04; F12 |
| A05 | baseline | `edge_19__baseline__rep02` | 3 | F31 |
| A06 | fsm | `happy_path_06__fsm__rep02` | 2 | F21 |
| A07 | fsm | `adversarial_13__fsm__rep01` | 1 | *(empty)* |
| A08 | fsm | `adversarial_13__fsm__rep02` | 3 | F32 |
| A09 | baseline | `happy_path_06__baseline__rep01` | 2 | F21 |
| A10 | fsm | `edge_19__fsm__rep01` | 3 | F31 |
| A11 | baseline | `edge_19__baseline__rep02` | 2 | F05 |
| A12 | baseline | `adversarial_04__baseline__rep02` | 2 | F10 |
| A13 | baseline | `adversarial_04__baseline__rep01` | 2 | F04 |
| A14 | fsm | `adversarial_13__fsm__rep01` | 4 | F02 |
| A15 | fsm | `edge_02__fsm__rep02` | 1 | *(empty)* |
| A16 | fsm | `adversarial_04__fsm__rep02` | 2 | F01; F03; F04 |
| A17 | baseline | `happy_path_06__baseline__rep01` | 3 | F02 |
| A18 | fsm | `edge_19__fsm__rep02` | 3 | F01; F31 |
| A19 | baseline | `adversarial_13__baseline__rep01` | 3 | F32 |
| A20 | baseline | `edge_02__baseline__rep01` | 2 | F10; F12 |
| A21 | fsm | `edge_02__fsm__rep01` | 4 | F02 |
| A22 | fsm | `adversarial_04__fsm__rep02` | 3 | F01; F03; F04 |
| A23 | baseline | `edge_02__baseline__rep02` | 1 | F12; F26 |
| A24 | fsm | `happy_path_06__fsm__rep02` | 1 | F21 |
| A25 | baseline | `edge_19__baseline__rep01` | 1 | F05 |
| A26 | baseline | `happy_path_06__baseline__rep02` | 1 | *(empty)* |
| A27 | fsm | `edge_19__fsm__rep02` | 4 | F01; F28 |
| A28 | baseline | `adversarial_13__baseline__rep02` | 1 | F32; F33; F34 |
| A29 | baseline | `adversarial_13__baseline__rep02` | 3 | F32; F34; F35 |
| A30 | fsm | `happy_path_06__fsm__rep01` | 1 | *(empty)* |
