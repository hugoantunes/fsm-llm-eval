# Human evaluation sidecar (Final Experiment)

This is an additional human evaluation of the frozen final experiment. It does
not modify or replace the existing judge evaluation at this stage.

The packet covers the same 348 scored dialogues used by `semantic_primary`.
Blind IDs hide the experimental condition. Judge predictions, explanations,
and aggregate scores are not in these files.

## Workflow

1. Read `rubric.md`.
2. Annotate dialogues in blind-ID order (`D001` … `D348`).
3. Fill `dialogue_annotations.csv` (`task_completed`: `yes` / `no` / `uncertain`).
4. Fill `response_annotations.csv` (`fact_ids_stated` as `F03;F07`, or empty).
5. Add one row per atomic checkable claim to `claims_annotations.csv`.
   Do not fill `claim_support`; it will be derived later.
6. Do not inspect `private/mapping.csv` (or anything under `private/`).
7. Do not inspect old judge outputs, `metrics.csv` scores, or aggregate
   baseline/FSM results for individual cases while annotating.

## Post-annotation freeze

Filled sheets were frozen on 2026-09-20T21:11:58Z, before any look at judge
results or `private/mapping.csv`. Byte copies and SHA-256 live in `frozen/`:

| File | SHA-256 |
|---|---|
| `dialogue_annotations.csv` | `5c2c06d59a136f6396d35600df7fc17f15229b0ec7b98f28c1320cf8291a786d` |
| `response_annotations.csv` | `69fb6b85a64ec2d302e128e6fa7d8836b57f61290de2fdfe7c3d6625db7d5a3c` |
| `claims_annotations.csv` | `18b37afaac8931690e03e1209a6a68cdb186efe6a5f38050d40a11db541e0e59` |

Human vs judge comparison and unblinding happen only after this freeze, in
`results/human_primary/` (`just score-human-validation`). Do not edit `frozen/`.
The human census was chosen as the narrative reference for the thesis
conclusion after freeze and unblind; the judge remains the original
confirmatory instrument and is reported as comparison.
`just figures results/human_primary` writes the Portuguese plates.
