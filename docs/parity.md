# Parity (T-16)

What is held equal between `baseline` and `fsm`, and what is allowed to differ, so a
score gap can be read as the instruction-base contrast rather than a confound. Quoted
from the corrected pilot at logical path `runs/exp_pilot` (manifest `exp_id`
`exp_pilot_fixes_20260913_final_v1`). Dataset hash
`0778a90110e6dd67685c1e6768934483cb4405213dad35ec172f3ddcf21d47c9` (`FROZEN_V1_HASH`).

The 11/09 sample is void. This directory after the adherence gate is the T-16 frame.
Dialogue-level judge validation is a census of all 20 ok dialogues. Response-level
validation is a blinded sample of 30/76 eligible agent responses. See
[`docs/judge_validation.md`](judge_validation.md).

The same `gemma4:12b` judge is used in T-16 and T-17. Rubrics are not cheapened to speed
eval. `flow_adherence` stays out of `PRIMARY_METRICS`.

## Held equal

- **Agent model and sampling.** Both agents call `qwen3.5:9b` from
  [`configs/models.yaml`](../configs/models.yaml): temperature `0.7`, base seed `42`,
  `think=False`, `num_ctx` 8192. The per-dialogue seed is derived from
  `(42, scenario_id, repetition)` and does not depend on which agent is speaking, so
  both meet the same customer.
- **Simulator.** Same `qwen3.5:4b` simulated user, same script-as-plan runtime, same
  `just adherence` gate (20/20 mandatory beats in order, 4/4 canary tokens verbatim,
  4/4 injection beats as specified).
- **Scenarios.** Frozen `data/scenarios/v1/`, hash above. Never edited after T-06.
- **Knowledge base.** Both agents receive the full KB. The FSM does not see a subset.
- **Shared instruction block.** [`data/prompts/agent_shared.md`](../data/prompts/agent_shared.md)
  version 2 is included by both [`baseline.md`](../data/prompts/baseline.md) and
  [`fsm_template.md`](../data/prompts/fsm_template.md).
- **Judge.** Blind two-call judge, `gemma4:12b`, temperature 0. Same prompts in
  validation and in T-17.

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
