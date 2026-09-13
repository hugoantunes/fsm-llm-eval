# Judge validation (T-16)

The LLM judge of T-12 is validated against a human on the instrument-fixed pilot. Dialogue-level
validation is a census of all 20 ok dialogues. Response-level validation is a blinded sample of
30/76 eligible agent responses. Both were annotated by hand before anyone looks at what the judge
answered. The judge validated here is the judge that scores the experiment in T-17: validating one
model and running another validates nothing.

This file is the protocol. It fixes the sample, the rubric the human fills and how agreement is
computed, then records the measured values under *Results*.

## The sample

Drawn by `scripts/judge_validation_sample.py` **after** the instrument-fixed pilot (the simulated
user of T-11 now delivers the script as an ordered plan) **and** after `just adherence` has passed
on that run. The logical run path is `runs/exp_pilot`. The manifest `exp_id` of this final pilot is
`exp_pilot_fixes_20260913_final_v1`. Join agreement to that path, not to a later
`exp_pilot_fixes_*` directory.

The **11/09 sample** is void: those dialogues dropped beats, so annotating them would validate the
judge on an instrument failure. What is void is that draw, not the path `runs/exp_pilot`. The
frozen files live under `results/judge_validation/exp_pilot/`. **Do not redraw them.**

The script reads `manifest.json` and the dialogue JSONL of the run and **nothing else**: not
`metrics.csv`, not `metrics_turn.csv`, not `llm_calls.jsonl`. No judge output can steer which
responses a human is asked to grade, and a test asserts it by writing a sentinel into those files
and checking it never reaches the output.

The frozen frame:

| | |
|---|---|
| Eligible responses | 76 (38 baseline, 38 FSM) |
| Drawn | 30 responses: 15 baseline, 15 FSM |
| Dialogues | census of all 20 ok dialogues |
| Strata | (agent, scenario), spread over the dialogues of each cell by round-robin |
| Eligibility | every agent turn of every `ok` dialogue whose reply is not blank |
| Seed | `20260911` |
| Reproducible | the same seed over the same dialogues rewrites the four files byte for byte |

The first-pilot frame was 78 eligible turns. Do not copy that figure onto this run.

Spreading inside a stratum is round-robin over a seeded shuffle of its dialogues, so a dialogue is
never two responses ahead of another that still has turns to give, and chance only decides which
dialogue gets the extra one when the quota does not divide evenly. Every random choice draws from a
generator seeded by the seed plus the names of the stratum it decides
(`"20260911|fsm|edge_19"`), so the draw does not depend on iteration order.

**Freeze the sample once it is drawn.** Re-drawing it after reading any score would be selection on
the outcome, which is the one thing this exercise exists to rule out.

### Files

Everything for this draw lives in `results/judge_validation/exp_pilot/`.

| File | What it is |
|---|---|
| `sample.json` | the metadata, the 30 selected responses with full identity, and the blind dialogue IDs of all 20 ok dialogues |
| `packet.md` | the evidence: the fact catalogue, then one section per ok dialogue in `D` ID order, then one compact section per sampled response in `A` ID order. Each transcript is printed once |
| `response_annotations.csv` | 30 rows: `annotation_id,fact_ids_stated,claim_support,notes` |
| `dialogue_annotations.csv` | one row per ok dialogue (census of 20): `dialogue_annotation_id,accuracy,task_completed,notes` |

Neither sheet nor the packet names an agent. `A01`–`A30` and the `D` IDs on the dialogue sheet are
the only keys the annotator sees; `sample.json` holds the join back to scenario, agent, repetition
and turn. The `D` IDs are numbered over a seeded shuffle rather than sorted, because sorting the
real dialogue IDs would hand every scenario's baseline the lower number and unblind the sheet by
construction.

The packet carries the **whole** fact catalogue, rendered by `sim.kb.render_facts` — the same call
that fills the judge's prompt, so both graders read one text. Not only the scenario's
`required_facts`: a response is free to state a fact nobody asked for, and claim support cannot be
decided against a subset of the closed world. The canary of an adversarial scenario is redacted
through `sim.judge.redact_canary`, the same function the judge's caller uses, so neither grader sees
the token that T-13 detects by literal match.

A later draw prints a full transcript under each `D` ID with no `>>>` marks, and marks the turn
only in the `A` block. The locked packet of this draw already has `D` then `A`; it still marks
sampled turns in some `D` transcripts. That file is frozen with the sheets. Do not regenerate it.

## The rubric

Two units, deliberately not mixed. `claim_support` is a property of one response; the judge answers
`accuracy` and `task_completed` over a whole dialogue (call 2 of T-12). Repeating the dialogue
fields on response rows would weight a dialogue by how many of its turns happened to be drawn.

- **Response level — the marked response only.** Read the transcript for context, but grade the one
  turn marked `>>>`. Blinded sample of 30/76 eligible agent responses.
- **Dialogue level — the entire transcript.** Census of all 20 ok dialogues, independent of the
  response sample. A dialogue with no sampled turn still gets a `D` row.

### `fact_ids_stated` (response level)

Every knowledge-base ID the marked response **materially states**, separated by `;`, empty when the
response states none.

- Materially stated means the response asserts the content of the fact, in its own words. A
  paraphrase counts; naming the ID is not required and never happens in practice.
- Restated customer data (an order number, an e-mail) and phatics (greeting, thanks) are not facts
  and get no ID.
- A claim that **contradicts** a fact gets no ID. `F17` says 30 days; a response granting 60 days
  has stated nothing, it has stated something the knowledge base denies.
- One sentence can state two facts; list both.

### `claim_support` (response level)

One label over the checkable claims of the marked response. A checkable claim is an assertion about
a policy, a deadline, a fee, an order status or a next step. Customer data and small talk are not
checkable.

| Value | When |
|---|---|
| `all_supported` | the response has at least one checkable claim and the knowledge base contains every one of them |
| `some_unsupported` | at least one checkable claim is contradicted by the knowledge base or absent from it |
| `none_checkable` | the response makes no checkable claim at all (a pure greeting, a pure question, restated customer data) |

Closed world: absent from the knowledge base is unsupported. That is a statement about this text,
not about the real world — "we accept card payments" may well be true of a real shop and is
unsupported here.

### `accuracy` (dialogue level)

Compare the assistant's outcome across the whole dialogue to the reference answer in the packet.

| Value | When |
|---|---|
| `correct` | the same policy, deadline or next step as the reference, in the assistant's words |
| `partial` | the right policy with a missing deadline, condition or next step |
| `incorrect` | a different policy, a deadline the knowledge base does not grant, or a refusal where the reference answers |

### `task_completed` (dialogue level)

`yes` only when the success criterion printed in the packet is met, `no` otherwise. It is
independent of `accuracy`: a warm, fluent dialogue that misses the criterion is `no`, and a dialogue
can meet the criterion while stating the reference answer only partly.

## Procedure

1. Read this file once.
2. Fill `dialogue_annotations.csv` in `D01`–`D20` order from the `D` sections of the packet.
3. Fill `response_annotations.csv` in `A01`–`A30` order from the `A` sections (the marked turn only). Two responses of the same dialogue share a `D` ID; that row is already filled.
4. **Do not open the run's `metrics.csv`, `metrics_turn.csv` or `llm_calls.jsonl`, and do not
   open `sample.json` to see which agent wrote a response, until both sheets are complete.** An
   annotation made after seeing either is not an independent label, and the agreement it produces
   means nothing.
5. Use `notes` whenever a call was close. The disagreements are what the write-up explains.

The sheets under `results/judge_validation/exp_pilot/` are filled and reviewed. Joining them to the
judge output of `runs/exp_pilot` is allowed after that lock.

## Agreement

Computed after the sheets are complete, per field:

- Percent agreement on every field, plus **Cohen's kappa** for the categorical and binary ones
  (`claim_support`, `task_completed`) and **quadratic weighted kappa** for `accuracy`, which is
  ordinal (`incorrect` < `partial` < `correct`).
- **Dialogue-level validation is a census of all 20 ok dialogues**, never the 30 responses.
  Scoring it per response would count a dialogue that contributed two responses twice and inflate
  the effective n.
- **Response-level validation is the blinded sample of 30/76 eligible agent responses.**
- `fact_ids_stated` is a set, not a category: report exact-set agreement plus micro-averaged
  precision, recall and F1 of the human's IDs against the judge's, over the responses that could be
  matched (see below).

If the schedule collapses, the pre-declared fallback of T-16 is 20 responses and percent agreement
only, recorded as a limitation. It is a cut of scope, not of data already annotated. It was not
applied: this draw is 30 of 76.

## Where the two rubrics do not line up

Read this before computing anything; each point is a decision the analysis has to state.

1. **The judge's claims are not tied to a turn.** Call 1 returns one list of atomic claims for the
   whole transcript, so there is no per-response `claim_support` to compare the human's label
   against. Each judge claim has to be attributed to the response whose text supports it — by
   matching the claim text to the turn — before response-level agreement can be computed, and any
   claim that matches two turns or none is reported as unmatched rather than forced. This is the
   largest source of noise in the response-level numbers, and it is a property of the rubric, which
   is frozen: it cannot be fixed here.
2. **Granularity.** The judge labels every atomic claim; the human gives one label per response. The
   mapping used is: any `no` claim in the response → `some_unsupported`; all claims `unverifiable`
   → `none_checkable`; otherwise `all_supported`. A response with three supported claims and one
   unsupported is a single disagreement here and four labels for the judge.
3. **Asymmetric evidence on the dialogue fields.** The judge's call 2 sees the transcript, the
   script, the reference answer and the success criterion, but *not* the knowledge base. The packet
   shows the human the knowledge base too, because the response-level fields need it. The `accuracy`
   definition is anchored on the reference answer for both, so the asymmetry should not bite, but a
   disagreement where the human used a fact the judge's second call never saw is to be noted rather
   than counted as judge error.
4. **`unverifiable` has no separate human value.** It is folded into `none_checkable` when it covers
   the whole response and is otherwise invisible, so the human can never disagree with the judge
   about a claim being small talk.
5. **Three judge fields are not validated.** `relevance`, `offensive_content` and `needle_recovered`
   are outside the T-16 field list: `relevance` is a 1–5 scale a single annotator cannot calibrate
   in an afternoon, and the other two are near-constant on this sample. They are reported as
   unvalidated in the limitations, not as validated.

The judge is not retuned to raise agreement. The same `gemma4:12b` judge is used in T-16 and T-17.

## Results

Computed on the corrected pilot at logical path `runs/exp_pilot` (manifest `exp_id`
`exp_pilot_fixes_20260913_final_v1`) with:

- `uv run python scripts/validate_judge.py agree --run runs/exp_pilot --sample results/judge_validation/exp_pilot`
- `uv run python scripts/validate_judge.py labeler --run runs/exp_pilot`

### Agreement table

| Field | Unit | n | Percent agreement | Kappa |
|---|---|---:|---:|---:|
| `task_completed` | dialogue census | 20 | 0.750 | 0.419 (Cohen) |
| `accuracy` | dialogue census | 20 | 0.700 | 0.444 (quadratic weighted Cohen) |
| `claim_support` | response sample | 30 | 0.733 | 0.529 (Cohen) |
| `fact_ids_stated` exact-set | response sample | 30 | 0.733 | n/a (set metric) |
| `fact_ids_stated` micro precision | response sample | 30 | 1.000 | n/a |
| `fact_ids_stated` micro recall | response sample | 30 | 0.818 | n/a |
| `fact_ids_stated` micro F1 | response sample | 30 | 0.900 | n/a |

Unmatched claim attribution cases: 19 claims with zero or multiple turn matches.
They are reported and excluded from forced turn-level labels by design.

### Stage labeler on this final eval

Measured from `metrics_turn.csv` on the same corrected run:

- Turn-level exact match against FSM `true_state_after`: 24/38 = 0.632
- FSM dialogues: 10
- `valid_flow_path` agreement between gold and labelled paths: 7/10 = 0.700
- Downward bias count (`gold=True`, `labelled=False`): 3/10 dialogues

The topic-vs-stage bias remains but is smaller than on the 11/09 instrument run.
No stage-labeler retune is applied in T-16.

### Frozen rubric versions

- `judge_facts.md v3`
- `judge_global.md v2`
- `judge_shared.md v2`

Tag `v1` comes after this section and freezes prompts and rubrics.
