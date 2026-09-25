# Decomposition of the `fact_f1` effect (post hoc, exploratory)

> **Experiment guide** · step 10 of 11 · [All steps](README.md) ·
> [← Data audit](audit.md) · [Next: Decisions and limitations →](decisions_and_limitations.md)

`fact_f1` is the frozen confirmatory metric and nothing here changes how it is
computed. This file answers a separate question about the result already frozen:
**where** the difference came from. It is exploratory and post hoc, declared as
such, and it is not a fourth confirmatory test.

Regenerate with `just decompose`. Artifacts:
`results/exp_final/fact_f1_decomposition.csv` and its provenance sibling
`fact_f1_decomposition.json`. Populations and censuses are
[`docs/audit.md`](audit.md); metric definitions are
[`docs/metrics.md`](metrics.md).

## What the metric can and cannot say

`fact_f1` measures correspondence between the facts the agent stated and the set
one scenario expected. It is not general factuality and not a hallucination
rate. Two unrelated events lower it:

- a required ID the agent never stated (FN), and
- a predicted item outside the expected set (FP).

The FP total is itself two disjoint things, which `sim.metrics.fact_scores` adds
into one number:

- **`extra_supported_fact`** — a `yes` claim whose knowledge-base ID the scenario
  did not require. True in the knowledge base by construction; off the expected
  set for this scenario.
- **`unsupported_claim`** — a `no` claim: checkable, and the knowledge base does
  not support it.

Only the second is evidence about grounding, and `claim_support` already measures
exactly that on its own denominator. So a lower `fact_f1` does not imply a worse
`claim_support`, does not substitute for it, and does not license a statement
about the external world: the knowledge base is closed.

## Method: replay, never re-judge

No judge call was made for this analysis. The judge output is already in
`runs/exp_final/llm_calls.jsonl`, and `sim.decomposition` re-reads it.

The judge prompt carries no dialogue ID, so each dialogue is joined to its judge
call by the rendered transcript the prompt embeds (the technique
[`docs/audit.md`](audit.md) uses to attribute timeouts). That join is
one-to-many: the same transcript was judged under both judge backends of this run
and a cache replay logs a second line. The frozen row is therefore the arbiter —
a candidate is used only when `sim.metrics.fact_scores` over its claims
reproduces that dialogue's `fact_precision`, `fact_recall`, `claim_support` and
`n_checkable_claims` in the exported `results/exp_final/metrics.csv`. A dialogue
that no candidate reproduces, or whose surviving candidates disagree, raises
rather than being guessed at.

On the frozen run all **348** scored semantic-primary dialogues reconstructed and
verified, with 0 unmatched and 0 ambiguous.

Two limits of the replay, both recorded rather than worked around:

- **Unverifiable claims are not reported.** They are outside both fact scores, so
  the frozen row does not pin their number down. The two judge backends in fact
  disagree about it on `adversarial_01__fsm__rep03` (4 versus 7) while agreeing on
  every checkable count and on TP/FP/FN. Reporting a number the frozen artifacts
  cannot verify would be worse than omitting it.
- **Zero precision would be ambiguous.** Precision 0 means TP 0, which leaves the
  FP total free: the row fixes how many claims were checkable and how many were
  unsupported, but not how many distinct off-set IDs the supported ones named.
  No dialogue in this run hit that case with disagreeing candidates; the code
  raises if one ever does.

## The four metrics

Copied from the frozen `results/exp_final/descriptive.csv` and `tests.csv`
(`semantic_primary`, `overall`), not recomputed. `fact_f1` and `claim_support` are
confirmatory and Holm-adjusted; `fact_precision` and `fact_recall` are its
components and were declared secondary before the run.

| metric | baseline | FSM | diff | Wilcoxon p | Holm | rank-biserial | W/T/L |
|---|---:|---:|---:|---:|---:|---:|---|
| `fact_precision` | 0.3642 | 0.3367 | −0.0388 | 0.022 | — | −0.367 | 17/8/34 |
| `fact_recall` | 0.8305 | 0.7546 | −0.0800 | 0.012 | — | −0.558 | 6/33/20 |
| `fact_f1` | 0.4888 | 0.4390 | −0.0592 | 0.004 | 0.013 | −0.454 | 15/7/37 |
| `claim_support` | 0.9432 | 0.9481 | +0.0041 | 0.975 | 1.000 | +0.006 | 20/23/16 |

Read on the components alone the attribution looks like recall, whose difference
and effect size are the larger of the two. That reading is a trap. Recall sits
near its ceiling (baseline mean 0.83, median 1.0, 33 of 59 scenarios tied), while
precision sits near 0.35, where F1 is far more sensitive: with P ≈ 0.36 and
R ≈ 0.83, `∂F1/∂P ≈ 0.97` against `∂F1/∂R ≈ 0.18`. A Shapley split run per
scenario and then averaged — which sums to −0.0608 against the frozen −0.0592,
so it accounts for the gap it is splitting — accordingly puts ≈70% on precision
and ≈30% on recall. (Splitting the arm-level means instead gives 65/35; the
conclusion does not turn on the choice.) The scenario tally agrees: of the 37
scenarios where FSM lost F1, 33 had lower precision (17 precision-only, 16 both)
against 4 recall-only.

Neither reading is the mechanism, and the counts below are why: precision and
recall share the TP numerator, and TP is the only quantity that moved.

## Result: the counts

Scenario is the unit, repetitions averaged first, 59 paired scenarios
(`adversarial_12` has no baseline arm). `mean_diff` is FSM − baseline. Tallies are
by direction, not by winner: for `tp` a lower FSM count is worse and for every
other count it is better.

| count | baseline | FSM | diff | FSM lower / equal / higher |
|---|---:|---:|---:|---|
| `n_required` | 1.475 | 1.475 | 0.000 | 0 / 59 / 0 |
| `tp` | 1.167 | 1.059 | −0.107 | 20 / 33 / 6 |
| `fn` | 0.308 | 0.415 | +0.107 | 6 / 33 / 20 |
| `fp` | 2.201 | 2.195 | −0.006 | 20 / 10 / 29 |
| `extra_supported_fact` | 1.833 | 1.935 | +0.102 | 22 / 12 / 25 |
| `unsupported_claim` | 0.367 | 0.260 | −0.107 | 19 / 25 / 15 |
| `n_checkable_claims` | 5.805 | 5.186 | −0.619 | 40 / 4 / 15 |

The expected set is identical by construction (`n_required` diff exactly 0), so
`tp` and `fn` are one quantity with two signs.

**The FP total did not move: −0.006 per dialogue.** What moved is TP, down 0.107.
Because precision is `TP / (TP + FP)` and recall is `TP / (TP + FN)`, a fall in TP
lowers *both* components at once. Precision fell without the agent producing more
off-set items.

The sharpest form of this is the zero-TP dialogue. Where the judge credited the
agent with none of the expected facts, precision, recall and `fact_f1` are all 0
no matter how few off-set items it produced. The FSM has twice as many of those:

```text
fact_precision == 0  (that is, TP == 0)
  baseline   16 / 173  =  9.2%
  fsm        32 / 175  = 18.3%
recall is also 0 in every one of those 48 dialogues
```

So the same event — the FSM stating none of the facts a scenario expected — is
what lowers both components, and it is why the count-level ratio understates the
metric-level drop: precision as a ratio of the mean counts falls 0.021, while the
mean of the per-dialogue precisions falls 0.039. The gap between those two numbers
is the extra zero-TP dialogues, not extra false positives.

Inside the flat FP total the composition shifted away from the grounding-relevant
kind: `unsupported_claim` fell (0.367 → 0.260) and `extra_supported_fact` rose
(1.833 → 1.935). The FSM made **fewer** unsupported claims per dialogue, which is
the direction `claim_support` independently reports (0.943 → 0.948, Wilcoxon
p = 0.975, Holm 1.000).

By stratum, `edge` is where `fact_f1` fell hardest (−0.113) and the pattern is
starkest. The two `fp` means there coincide to floating-point error (2.1833 both
ways, 8 scenarios lower and 8 higher, which cancel), while `fn` rises 0.275. On
`edge` the whole net effect is coverage. The cancelling halves are a mean, not 20
identical scenarios: individual `edge` scenarios do differ in FP, in both
directions, and the exploratory per-stratum `fact_precision` test there is
negative (−0.074, p = 0.014). Precision on `edge` falls with the FP total held
flat, which is again TP.

## What this supports, and what it does not

Supported by the frozen artifacts:

- The FSM stated fewer of the facts its scenarios expected, and stated none of
  them twice as often. That is the entire moving part of the `fact_f1` difference,
  and it depresses precision and recall together because they share the TP
  numerator.
- It said less overall: 0.62 fewer checkable claims per dialogue (40 of 59
  scenarios lower), consistent with the terser state-package style.
- It did not produce more off-set factual items, and it produced fewer
  unsupported ones.

Not supported:

- That the FSM hallucinated more. FP did not rise, the unsupported-claim
  component fell, and `claim_support` did not degrade. There is no component of
  this decomposition pointing that way.
- That a lower `fact_f1` is a lower `claim_support`. They moved in opposite
  directions here.
- Any claim about truth outside the experiment's closed knowledge base.

Separate and not to be joined to this: the Pilot v1 `fact_ids_stated` audit
(`results/judge_validation/pilot_v1/fact_ids_directional_audit.md`) found pooled
fact-ID under-detection against human annotation with **no recoverable evidence**
that it was differential by condition. It therefore establishes no mechanism that
would selectively depress FSM `fact_f1`, and it is not an explanation for the
numbers above.

A later census of the 348 Final Experiment dialogues
([`docs/metrics.md`](metrics.md)) found that the judge's Holm-significant
`fact_f1` FSM deficit is not reproduced under human labels. That is a
separate instrument comparison. It does not enter this decomposition and
does not reopen Holm on the judge family.

Why the FSM covered fewer expected facts is a question about the FSM
implementation — its state packages, its transition guards, and when it leaves
`solution` — which this decomposition locates but does not answer. `edge_05` in
[`results/cases.md`](../results/cases.md) is one worked instance: the machine moved
into resolution without emitting the required fact.
