# Scenario taxonomy and quotas (T-05)

A scenario is one test case of the experiment: the brief the simulated user plays, the
answer key the judge grades against, and the condition that decides whether the task was
completed. Both agents run the same scenarios, so a scenario is also the unit of
analysis: the statistics of T-19 pair `baseline` and `fsm` by scenario ID and never by
dialogue.

The shape is fixed by [`src/sim/schemas.py`](../src/sim/schemas.py) and the two worked
examples are in [`data/scenarios/examples/`](../data/scenarios/examples/). The frozen
dataset of T-06 goes to `data/scenarios/v1/`, one JSONL per category, one scenario per
line, and never changes once its hash is recorded.

## The three categories

**`happy_path`** — one clear intent, a cooperative customer, the data given when asked.
This is the case any well-written prompt should handle, so it is where the FSM has the
least to gain and the most to lose to rigidity.

**`edge`** — the dialogue is legitimate but does not arrive tidily. Four shapes, one per
intent in the first block: a vague opening message, two intents mixed in one request, a
required datum the customer never supplies, and a change of subject mid-dialogue.

**`adversarial`** — the dialogue attacks the instructions or the knowledge base. Per
intent: an injection carrying a `canary` token, a plausible question the KB does not
answer, an attempt to bend a policy, and an aggressive or out-of-scope request. The
correct behaviour is always the same: state only what the KB contains, say plainly when
something is not covered, and escalate (F03, F04).

## The matrix

Rows are the intents of [`data/kb/knowledge_base.md`](../data/kb/knowledge_base.md);
`general` is not among them, because those facts are released in every state and belong
to no particular request. The first block is balanced by construction, four scenarios per
cell:

| Intent | `happy_path` | `edge` | `adversarial` | Total |
|---|---|---|---|---|
| `order_tracking` | 4 | 4 | 4 | 12 |
| `exchange_return` | 4 | 4 | 4 | 12 |
| `cancellation` | 4 | 4 | 4 | 12 |
| `payment_reissue` | 4 | 4 | 4 | 12 |
| **Total** | 16 | 16 | 16 | **48** |

The extras add a fifth scenario to every cell, which is the full dataset:

| Intent | `happy_path` | `edge` | `adversarial` | Total |
|---|---|---|---|---|
| `order_tracking` | 5 | 5 | 5 | 15 |
| `exchange_return` | 5 | 5 | 5 | 15 |
| `cancellation` | 5 | 5 | 5 | 15 |
| `payment_reissue` | 5 | 5 | 5 | 15 |
| **Total** | 20 | 20 | 20 | **60** |

## Writing order

The 48 of the first block are written and reviewed **before** any extra, cell by cell,
and the extras are appended last. That is what makes cut 1b free: if Friday runs out of
time, the dataset stops at 48 and is still balanced across every intent and category, one
whole pass ahead of the pre-declared floor of 45. Reducing N after the dataset is frozen
and run would be discarding data, which the *Plano de corte* forbids.

Within a category file the scenarios are in writing order, so the first sixteen lines of
each file are the first block. IDs run per category, `happy_path_01` to `happy_path_20`.

## What the quotas do not say

- Every needle of [`needles.json`](../data/kb/needles.json) is probed by at least one
  scenario with `is_needle` true, so the six specific facts are all exercised.
- Every question of [`unanswerable.json`](../data/kb/unanswerable.json) seeds at least one
  adversarial scenario. The two the KB files under `general`, U09 and U10, attach to a
  scenario of whichever intent opens the dialogue.
- Each intent has exactly one injection carrying a canary, so safety is measured on all
  four intents rather than on whichever one happened to get the attacks.
- No real brand, person or company appears anywhere, and a test enforces it.
