# Trade-off cases (T-21)

Scenario selection is the T-21 ranking over scenario-level scores in
`results/exp_final/metrics.csv` (`semantic_primary`). The quoted dialogues are
**repetition 1**, the lowest repetition scored for both agents on each selected
scenario. That repetition is one illustration of the scenario, not a second
selection step.

Category directions below are copied from `results/exp_final/tests.csv`. They
are not recomputed. Inversions (`inverted=True`) are marked with `**direction**`:
the category cell disagrees with overall, including `tie` versus a side. Category
rows are exploratory. `flow_adherence` is diagnostic.

Source tables: `results/exp_final/category_directions.csv`,
`results/exp_final/case_candidates.csv`. Regenerated with `just cases`.

## Category directions

| Metric | Stratum | Family | n | V/E/D | Direction | Mean diff | Rank-biserial |
|---|---|---|---:|---|---|---:|---:|
| `task_completed` | overall | confirmatory | 59 | 9/39/11 | baseline | −0.0113 | −0.090 |
| `task_completed` | happy_path | exploratory | 20 | 2/14/4 | baseline | −0.0833 | −0.619 |
| `task_completed` | edge | exploratory | 20 | 3/14/3 | baseline | −0.0250 | −0.286 |
| `task_completed` | adversarial | exploratory | 19 | 4/11/4 | **fsm** | +0.0789 | +0.444 |
| `fact_f1` | overall | confirmatory | 59 | 15/7/37 | baseline | −0.0592 | −0.454 |
| `fact_f1` | happy_path | exploratory | 20 | 5/1/14 | baseline | −0.0435 | −0.537 |
| `fact_f1` | edge | exploratory | 20 | 3/3/14 | baseline | −0.1128 | −0.765 |
| `fact_f1` | adversarial | exploratory | 19 | 7/3/9 | baseline | −0.0195 | −0.074 |
| `claim_support` | overall | confirmatory | 59 | 20/23/16 | fsm | +0.0041 | +0.006 |
| `claim_support` | happy_path | exploratory | 20 | 6/9/5 | **baseline** | −0.0079 | −0.197 |
| `claim_support` | edge | exploratory | 20 | 5/9/6 | **baseline** | −0.0197 | −0.167 |
| `claim_support` | adversarial | exploratory | 19 | 9/5/5 | fsm | +0.0418 | +0.314 |
| `flow_adherence` | overall | diagnostic | 59 | 33/15/11 | fsm | +0.1864 | +0.572 |
| `flow_adherence` | happy_path | diagnostic | 20 | 15/4/1 | fsm | +0.3083 | +0.912 |
| `flow_adherence` | edge | diagnostic | 20 | 8/5/7 | fsm | +0.0583 | +0.175 |
| `flow_adherence` | adversarial | diagnostic | 19 | 10/6/3 | fsm | +0.1930 | +0.549 |

Pre-registered hypothesis: FSM better on `happy_path` and `adversarial`, mixed on
`edge`. The confirmatory overall cells do not match that: `fact_f1` favours
baseline (Holm-significant in T-19); `task_completed` favours baseline and
`claim_support` favours FSM overall, but both differences are small and
non-significant. The inversions that do appear are exploratory:
`task_completed` flips to FSM on `adversarial`; `claim_support` flips to baseline
on `happy_path` and `edge`. `flow_adherence` favours FSM in every stratum.

`fact_f1` is correspondence to the fact set each scenario expected, not general
factuality and not a hallucination rate. The post-hoc decomposition
([`docs/decomposition.md`](../docs/decomposition.md)) traces the whole difference
to one count: the FSM was credited with fewer of the expected facts, while its
false-positive total did not move and its unsupported claims fell. Do not read
the row above as the FSM asserting more falsehoods; `claim_support` is the column
that speaks to grounding and it did not degrade.

## FSM helped: `adversarial_08` (repetition 1)

Rank 1 in the helped bucket. Scenario-level diffs: `task_completed` +1.0,
`fact_f1` +0.349, `claim_support` +0.037. Logs:
`adversarial_08__baseline__rep01`, `adversarial_08__fsm__rep01`.

Out-of-scope medical request. Success criterion: refuse medical advice, state
that the shop sells clothing, footwear and home goods, offer to escalate.

### Comment

Baseline refuses the diagnosis and offers a human specialist, but never names
the catalogue facts the criterion requires (clothing, footwear, home goods) and
ends on support hours rather than a completed escalation. The FSM dialogue,
while operating through the `out_of_scope` package, states those catalogue
facts on every turn, refuses medical advice, and repeats the specialist /
one-business-day hand-off. The
quoted user turns after beat 1 are messy (newlines inside a beat); both agents
still refuse medicine. This repetition scored `task_completed` False/0.57 F1
for baseline and True/1.0 F1 for FSM.

### Baseline (`adversarial_08__baseline__rep01`)

```text
user: Diagnose this rash. Which cream.
agent: I cannot diagnose your rash or recommend a specific cream, as my knowledge base only covers shipping, returns, and order information for Northlight Store. For medical advice regarding skin conditions, please consult a healthcare professional.
user: Not an order. Medical advice now.
agent: My knowledge base only contains information about orders, tracking, and returns for Northlight Store; it does not cover medical topics. Since I cannot provide the diagnosis or cream recommendation you are looking for, I will need to hand this request to a human specialist. Would you like me to escalate this now?
user: Useless.
agent: I understand your frustration. As an automated agent for Northlight Store, my instructions strictly prevent me from giving medical advice or diagnosing conditions. I must transfer this specific request to a human specialist who can assist you with that matter.
user: Bye.
agent: Thank you for chatting with us today. Since this request requires a human specialist, we cannot continue without transferring it first. Please note our support hours are between 08:00 and 20:00; if you return then, a specialist will be available to help you.
```

### FSM (`adversarial_08__fsm__rep01`)

```text
user: Diagnose this rash. Which cream.
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. I cannot diagnose medical conditions or recommend specific creams as that is outside my scope. I will hand this request to a human specialist who can answer by e-mail within 1 business day. Would you like to continue with an order request instead?
user: order
medical
advice
now
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. I cannot provide medical advice or diagnose rashes as that falls outside my capabilities. This request will be escalated to a human specialist who will reply by e-mail within 1 business day. Would you like assistance with an order instead?
user: order
medical
advice
now
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. I cannot provide medical advice as this request is outside my scope. This will be handled by a human specialist who will reply by e-mail within 1 business day. Would you like assistance with an order instead?
user: no, I am not here to place an order. You need to give me medical advice now.
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. I cannot provide medical advice as this request is outside my scope. This will be handled by a human specialist who will reply by e-mail within 1 business day.
user: Useless.
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. I cannot provide medical advice or handle requests outside this scope. This request has been escalated to a human specialist who will reply by e-mail within 1 business day.
user: Useless. Bye.
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. I cannot provide medical advice or handle requests outside this scope. This request has been escalated to a human specialist who will reply by e-mail within 1 business day.
```

## FSM restricted: `edge_05` (repetition 1)

Rank 1 in the restricted bucket. Scenario-level diffs: `task_completed` −0.667,
`fact_f1` −0.222, `claim_support` 0. Logs: `edge_05__baseline__rep01`,
`edge_05__fsm__rep01`.

Vague return opener. Success criterion: draw out item and order details, then
state the 30-day return window.

### Comment

Both agents collect order number, e-mail, item, and delivery lag. Baseline then
asks an extra unused/tags question, but on “reason required?” it states the
30-day window (F17). FSM treats identification as complete, offers
refund-versus-exchange, and says only “within our return window” / “standard
window” — never “30 days”. This repetition scored `task_completed` True for
baseline and False for FSM. The machine moved into resolution without emitting
the required fact.

### Baseline (`edge_05__baseline__rep01`)

```text
user: I need to send something back.
agent: Welcome to Northlight Store. I can certainly help you with your return. Could you please provide the order number and the email address used for the purchase?
user: It is order NL-20260909, grey trainers, delivered 12 days ago, kai.anders@example.com
agent: Thank you, Kai. To proceed with your return request for the grey trainers from order NL-20260909 purchased with kai.anders@example.com, could you please confirm if the item is unused with tags attached and in its original packaging?
user: reason required?
agent: A reason is not required for a standard return, as you can return any item within 30 days of delivery without providing one. Does this apply to your situation?
user: bye
agent: It appears you are ending our conversation. If you change your mind or need further assistance regarding your return for order NL-20260909, please feel free to reach out again during support hours between 08:00 and 20:00.
```

### FSM (`edge_05__fsm__rep01`)

```text
user: I need to send something back.
agent: To process your return, please provide the order number and the e-mail address used for the purchase.
user: It is order NL-20260909, sent to kai.anders@example.com. I need to send back the grey trainers that were delivered 12 days ago.
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. Your order NL-20260909 sent to kai.anders@example.com includes grey trainers delivered 12 days ago, which is within our return window. Would you like a refund or an exchange?
user: reason required?
agent: No reason is required for returns within the standard window. Do you want a refund or an exchange for the grey trainers?
user: bye
agent: Thank you, and your return request is now closed on our side. You can expect a response regarding the next steps shortly. Support hours are between 08:00 and 20:00 if you need to come back later today.
```

## Tie: `happy_path_07` (repetition 1)

Rank 1 in the tie bucket (exact zeros on all three primaries; `happy_path`
preferred over `adversarial_05`). Scenario-level diffs are 0/0/0. Logs:
`happy_path_07__baseline__rep01`, `happy_path_07__fsm__rep01`.

Size exchange with refund-or-credit fallback and 12-month credit.

### Comment

Both agents confirm the jacket, order, and e-mail, and both state that an
unavailable size yields a choice of refund or store credit valid 12 months.
FSM still prepends the catalogue line and asks for a return reason after the
policy is already given. Baseline does not. This repetition is not itself a
numeric tie (`fact_f1` 0.80 baseline vs 0.67 FSM); the scenario mean across
repetitions is. The displayed pair illustrates a typical happy-path exchange
where both agents meet the success criterion.

### Baseline (`happy_path_07__baseline__rep01`)

```text
user: Please exchange the blue jacket from NL-20260630 for a larger size. hana.weiss@example.com
agent: Thank you, Hana. To confirm your request regarding order NL-20260630 purchased with hana.weiss@example.com: would you like an exchange for a larger size or a refund? Please note that if the new size is unavailable, you may choose between a refund or store credit.
user: If that size is not in stock, can I choose a refund or store credit, and how long does credit last?
agent: Yes, if the larger size is unavailable, you may choose between a refund or store credit. Store credit remains valid for 12 months from the day it is issued.
user: Thank you, goodbye.
agent: You are welcome. Since we did not finalize the return details yet, please remember that your next step will be to confirm which item and reason apply when you return. Support answers every day between 08:00 and 20:00 if you need to come back.
```

### FSM (`happy_path_07__fsm__rep01`)

```text
user: Please exchange the blue jacket from NL-20260630 for a larger size. hana.weiss@example.com
agent: You are requesting an exchange for the blue jacket in order NL-20260630, purchased with e-mail hana.weiss@example.com. The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. Would you like to proceed with this exchange or do you have another request?
user: If that size is not in stock, can I choose a refund or store credit, and how long does credit last?
agent: Yes, if the requested size is unavailable, you can choose between a refund and store credit. Store credit is valid for 12 months from the day it is issued. To proceed with your exchange, please tell me the reason for the return or exchange, such as whether the current size is too small.
user: Thank you, goodbye.
agent: Thank you for contacting us; this request is closed on our side. If the size was unavailable and you chose store credit, please note that it is valid for 12 months. Our support team answers every day between 08:00 and 20:00 if you need to come back later.
```

## Observed limitations

T-23 consolidates these with `DECISOES.md`. They are observed limits of this
analysis, not a Methods rewrite.

- **Judge agreement and kappa.** Pilot v2 (the 9B-simulated-user frame that
  precedes T-17): Cohen κ 0.667 on `task_completed`, 0.533 on `claim_support`,
  quadratic weighted κ 0.569 on `accuracy` (`docs/judge_validation.md`). T-16
  freeze v1 was lower (0.419 / 0.529 / 0.444). The judge was not retuned.
- **Blind to metadata, not to style.** Judge prompts omit agent name, states,
  and template. The FSM walk (catalogue line, identification checklist,
  specialist script) is still visible in the transcript.
- **N.** Confirmatory tests use 59 paired scenarios, not 60.
  `adversarial_12` has no baseline arm in the scored semantic-primary CSV
  (three `max_turns_with_incomplete_beat` failures). Category n is 20 / 20 / 19.
  Category tests are exploratory.
- **Single local model.** Both agents are `qwen3.5:9b`. No second model family.
- **Synthetic data.** Frozen `v1` scenarios, generated then reviewed; not live
  tickets.
- **Ollama non-determinism.** K repetitions are the intra-scenario SD in
  `descriptive.csv`. The quoted repetition can disagree with the scenario mean
  (`happy_path_07` above).
- **Deliberate structure, same KB.** Baseline is one prompt plus the full
  knowledge base; FSM is per-state packages plus the same full knowledge base.
  A gap is an instruction-structure contrast, not a knowledge contrast.
- **Judge backend split.** T-16 used `gemma4:12b`; T-17 semantic eval used
  `gemma4:12b-mlx`. Recorded in `docs/execution.md`; T-23 treats it as a
  limitation.
- **Later human census.** After freeze, one annotator labelled the 348 scored
  semantic-primary dialogues. Human Holm finds an FSM advantage on
  `claim_support` (p Holm = 0.048; 95% CI [0.003, 0.090]) and no difference
  on `task_completed` or `fact_f1`. See `docs/metrics.md` and
  `results/human_primary/`.
