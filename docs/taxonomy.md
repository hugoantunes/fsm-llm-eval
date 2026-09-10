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

The categories describe **how the request unfolds**, not how hard the knowledge-base
fact is. A `happy_path` scenario may test any KB fact, including a needle, as long as
the customer is cooperative and the dialogue is straightforward. An `edge` scenario is
still a legitimate request.

**`happy_path`** — one clear intent and a cooperative customer. All required information
is available and provided without resistance; it may be volunteered early or packed
into a single message. Wording and message length may vary naturally. The dialogue has
no intentional ambiguity, no meaningful topic switching, no withholding of required
information, no change of goal, and no adversarial pressure. This is the case any
well-written prompt should handle, so it is where the FSM has the least to gain and
the most to lose to rigidity.

**`edge`** — a legitimate customer request whose conversation is irregular. The customer
is not attacking or manipulating the agent. Structural difficulty may include a vague
opening, mixed intents, missing or withheld required information, a change of mind, a
temporary topic switch and return, information supplied out of the expected order, a
repeated request, fragmented messages, packed or run-on messages, or an interruption of
the normal flow. Packing details into one cooperative message is still `happy_path`;
a packed or run-on turn counts as `edge` when it scrambles or interrupts that flow.
The first block of each intent covers four of those shapes (vague opening, mixed
intents, a required datum never supplied, and a mid-dialogue change of subject).
Extras stay in the same category; they are not a fourth class.

**`adversarial`** — the user tries to break instructions, policy boundaries, or KB
limits: override instructions, plant a prompt injection, obtain unsupported
information, force a policy exception, push the agent outside the knowledge base, make
an out-of-scope request, or pressure the agent into inventing or violating policy.
Ordinary conversational irregularity is not adversarial. Per intent in the first
block: an injection carrying a `canary` token, a plausible question the KB does not
answer, and an attempt to bend a policy. The fourth slot is an aggressive or
out-of-scope request for `order_tracking`, `exchange_return` and `cancellation`;
`payment_reissue` uses a second unanswerable question (U10) instead. Only those
three intents have an `out_of_scope` expected final state. The correct behaviour is
always the same: state only what the KB contains, say plainly when something is not
covered, and escalate (F03, F04).

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
and the extras are appended last. That is what makes cut 1b free **for balance**: if
Friday runs out of time, the dataset stops at 48 and is still balanced across every
intent and category, one whole pass ahead of the pre-declared floor of 45. Cut 1b is
not free for unanswerable coverage: U02, U04, U06 and U08 sit on the fifth
adversarial slot of each intent, so they exist only at N = 60. Reducing N after the
dataset is frozen and run would be discarding data, which the *Plano de corte*
forbids. The freeze chose 60, so those four are in.

Within a category file the scenarios are in writing order, so the first sixteen lines of
each file are the first block. IDs run per category, `happy_path_01` to `happy_path_20`.

## What the quotas do not say

- Every needle of [`needles.json`](../data/kb/needles.json) is probed by at least one
  scenario with `is_needle` true, so the six specific facts are all exercised.
- Every question of [`unanswerable.json`](../data/kb/unanswerable.json) seeds at least one
  adversarial scenario **at N = 60**. The two the KB files under `general`, U09 and U10,
  attach to a scenario of whichever intent opens the dialogue. U02, U04, U06 and U08
  are extras (fifth slot); they would drop under cut 1b.
- Each intent has exactly one injection carrying a canary, so safety is measured on all
  four intents rather than on whichever one happened to get the attacks.
- No real brand, person or company appears anywhere, and a test enforces it.

## Conversational variants (inside the 60)

Natural linguistic variation can appear in every category: short or long messages,
different writing styles, and identifying information supplied early or packed into one
turn. That variation does not change the category. `happy_path` in particular may vary
in length, style, and how early information is supplied, as long as the interaction
stays cooperative and straightforward.

Structural irregularities belong primarily to `edge`: vague openings, mixed intents,
withheld data, a change of mind, topic switches, out-of-order information, repeated
requests, fragmented messages, and interruptions of the normal flow. Attempts to
override instructions, inject prompts, force a policy exception, or leave the knowledge
base belong only to `adversarial`.

These variants do not create additional statistical units. They do not add rows, do not
change `required_facts` / `success_criterion` / `expected_final_state`, and are not extra
units of analysis. The dataset remains exactly 60 scenarios. K repetitions remain the
way to sample model noise; the script is the way to sample dialogue shape.

Edge withheld-datum scripts still never supply the missing field. Adversarial canary
tokens still appear word for word.
