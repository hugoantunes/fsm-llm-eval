# Parity (T-16)

What is held equal between `baseline` and `fsm`, and what is allowed to differ, so a
score gap can be read as the instruction-base contrast rather than a confound. Quoted
from the corrected pilot at logical path `runs/exp_pilot` (manifest `exp_id`
`exp_pilot_fixes_20260913_final_v1`). Dataset hash
`0778a90110e6dd67685c1e6768934483cb4405213dad35ec172f3ddcf21d47c9` (`FROZEN_V1_HASH`).

The 11/09 sample is void. This directory after the adherence gate is the T-16 frame.
Dialogue-level judge validation is a census of all 20 ok dialogues. Response-level
validation is a blinded sample of 30/76 eligible agent responses. Frozen T-16
sheets: `results/judge_validation/pilot_v1/`. See
[`docs/judge_validation.md`](judge_validation.md).
Pilot v2 is a later 9B-simulated-user frame (`runs/pilot_v2`, 30/78, seed
`20260914`); agreement and disagreements are in the same file, under
`results/judge_validation/pilot_v2/`. Do not mix the two token tables.

A later census of the 348 scored T-17 dialogues is not this T-16 frame. See
[`docs/metrics.md`](metrics.md) (Human census of semantic-primary).

T-16 / Pilot v2 used `gemma4:12b` GGUF. T-17 frozen-gate evaluation used the
same `gemma4:12b` GGUF. T-17 semantic-primary sidecar used `gemma4:12b-mlx`
(MLX / nvfp4; different digest). The two backends are not pooled or treated as
equivalent. Rubrics are not cheapened to speed eval. `flow_adherence` stays out of
`PRIMARY_METRICS`.

## Held equal

- **Agent model and sampling.** Both agents call `qwen3.5:9b` from
  [`configs/models.yaml`](../configs/models.yaml): temperature `0.7`, base seed `42`,
  `think=False`, `num_ctx` 8192. The per-dialogue seed is derived from
  `(42, scenario_id, repetition)` and does not depend on which agent is speaking, so
  both meet the same customer.
- **Simulator (this T-16 frame).** Same `qwen3.5:4b` simulated user, same
  script-as-plan runtime, same `just adherence` gate (20/20 mandatory beats in
  order, 4/4 canary tokens verbatim, 4/4 injection beats as specified). After
  T-16 the simulated-user role was split from `classifier` / `state_labeler`
  and locked to `qwen3.5:9b` for pre-flight and Pilot v2; classifier and
  labeler stay `qwen3.5:4b` (`docs/decisions_and_limitations.md`). The token
  account below is this 4B frame and is not mixed with later 9B runs.
- **Scenarios.** Frozen `data/scenarios/v1/`, hash above. Never edited after T-06.
- **Knowledge base.** Both agents receive the full KB. The FSM does not see a subset.
- **Shared instruction block.** [`data/prompts/agent_shared.md`](../data/prompts/agent_shared.md)
  version 2 is included by both [`baseline.md`](../data/prompts/baseline.md) and
  [`fsm_template.md`](../data/prompts/fsm_template.md).
- **Judge.** Blind two-call judge, temperature 0. T-16 / Pilot v2 and T-17
  frozen-gate: `gemma4:12b` GGUF. T-17 semantic-primary sidecar:
  `gemma4:12b-mlx`. Same prompts; not the same weights. Not pooled.

## Allowed to differ

- **Instruction base.** Baseline: one system prompt with the full procedure.
  FSM: the instruction package of the current state, plus the user-event classifier
  that drives transitions.
- **Classifier cost.** Only the FSM pays a `classifier` call per turn. That is a
  deliberate part of the treatment, reported in the token account below, not a leak
  to be subtracted away.

## Token account (corrected pilot)

From `just stats runs/exp_pilot` on the corrected run (13/09). Means are over that
caller's lines in `llm_calls.jsonl`. Latency is uncached calls only.

| caller | n | mean prompt tokens | mean output tokens | mean latency (s) |
|---|---|---|---|---|
| `baseline` | 38 | 2276.7 | 54.6 | 18.32 |
| `fsm` | 38 | 2126.9 | 49.3 | 18.94 |
| `classifier` | 28 | 1198.2 | 30.1 | 11.90 |
| `simulated_user` | 89 | 1357.3 | 40.1 | 13.62 |
| `stage_labeler` | 20 | 608.4 | 35.5 | 10.68 |
| `judge_facts` | 20 | 2847.1 | 598.9 | 207.07 |
| `judge_global` | 20 | 1353.7 | 78.2 | 48.43 |

**Prompt tokens per agent turn.** Baseline: 2276.7 (the agent call). FSM: 2126.9
(agent) + 1198.2 (classifier) = 3325.1. The FSM's per-state package is smaller than
the baseline prompt; the classifier is the extra bill. These are sums of per-caller
means, not a paired per-turn join.

Run: 72.1 s/dialogue. Eval: 266.2 s/dialogue.

The 11/09 cost table in [`docs/pilot.md`](pilot.md) remains the machine-budget record
of the superseded instrument (three eval passes on an append-only log). Do not mix the
two tables.

## Token account (Pilot v2)

From `just stats runs/pilot_v2` (2026-09-14). Simulated user is `qwen3.5:9b`.
Means are over that caller's lines in `llm_calls.jsonl`. Latency is uncached
calls only.

| caller | n | mean prompt tokens | mean output tokens | mean latency (s) |
|---|---|---|---|---|
| `baseline` | 39 | 2276.5 | 51.8 | 17.19 |
| `fsm` | 39 | 2123.4 | 48.1 | 12.29 |
| `classifier` | 28 | 1195.9 | 31.8 | 9.24 |
| `simulated_user` | 78 | 1494.0 | 40.8 | 19.43 |
| `stage_labeler` | 20 | 610.5 | 36.0 | 9.39 |
| `judge_facts` | 20 | 2849.0 | 603.0 | 183.76 |
| `judge_global` | 20 | 1355.6 | 78.3 | 44.18 |

**Prompt tokens per agent turn (this frame).** Baseline: 2276.5. FSM: 2123.4
(agent) + 1195.9 (classifier) = 3319.3. Run: 72.9 s/dialogue. Eval: 237.3
s/dialogue.
