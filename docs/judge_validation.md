# Judge validation (T-16)

> **Experiment guide** · step 7 of 11 · [All steps](README.md) ·
> [← Parity](parity.md) · [Next: Execution →](execution.md)

The LLM judge of T-12 is validated against a human on the instrument-fixed pilot. Dialogue-level
validation is a census of all 20 ok dialogues. Response-level validation is a blinded sample of
30/76 eligible agent responses. Both were annotated by hand before anyone looks at what the judge
answered. The rule was that the judge validated here is the judge that scores the experiment in
T-17: validating one model and running another validates nothing. T-17 kept that rule for its
frozen-gate population, scored by the same `gemma4:12b` GGUF blob, but scored the
semantic-primary population with `gemma4:12b-mlx` (DECISOES 2026-09-16). The Pilot v2 sample was
therefore re-scored with that configuration against the same human sheets (*Pilot v2 robustness
check*, below).

This file is the protocol, then the measurements in the order they were taken:

1. the protocol — sample, rubric, procedure, agreement — fixed on Pilot v1 (2026-09-13);
2. *Results* on Pilot v1, after which tag `v1` froze prompts and rubrics;
3. *Pilot v2 validation round* (2026-09-14), the T-16 closure and the agreement quoted
   everywhere else, with the stage-labeler decision (tag `v1.1`);
4. two later checks on the same Pilot v2 dialogues (2026-09-20), neither a retune: the judge
   backend used by T-17 (*Pilot v2 robustness check*) and the classifier/labeler model size
   (*Pilot v2 instrument sidecar*);
5. a pointer to the separate human census of the T-17 dialogues.

The pilot rounds are named in [`docs/pilot.md`](pilot.md).

## The sample

Drawn by `scripts/judge_validation_sample.py` **after** the instrument-fixed pilot (the simulated
user of T-11 now delivers the script as an ordered plan) **and** after `just adherence` has passed
on that run — Pilot v1. The logical run path is `runs/exp_pilot`. The manifest `exp_id` of this
final pilot is `exp_pilot_fixes_20260913_final_v1`. Join agreement to that path, not to a later
`exp_pilot_fixes_*` directory.

The **11/09 sample** is void: those dialogues dropped beats, so annotating them would validate the
judge on an instrument failure. What is void is that draw, not the path `runs/exp_pilot`. The
frozen files live under `results/judge_validation/pilot_v1/`. **Do not redraw them.**

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

Everything for this draw lives in `results/judge_validation/pilot_v1/`.

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

The sheets under `results/judge_validation/pilot_v1/` are filled and reviewed. Joining them to the
judge output of `runs/exp_pilot` is allowed after that lock. The raw `runs/exp_pilot` tree is
gone; the sheets and packet remain.

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

The judge stays at the T-16 rubric. T-16 and Pilot v2 used `gemma4:12b` GGUF.
T-17 frozen-gate evaluation used that same GGUF blob. T-17 semantic-primary
used `gemma4:12b-mlx` (different backend and digest). The Pilot v2 dialogues
were later re-scored with that `gemma4:12b-mlx` configuration as a robustness
check of the instrument used in the confirmatory eval; see *Pilot v2 robustness
check* below. Prompts stay frozen. GGUF and MLX remain two recorded backends.

## Results

Computed on Pilot v1, the corrected pilot at logical path `runs/exp_pilot` (manifest `exp_id`
`exp_pilot_fixes_20260913_final_v1`) with:

- `uv run python scripts/validate_judge.py agree --run runs/exp_pilot --sample results/judge_validation/pilot_v1`
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

### Stage labeler on Pilot v1

Measured from `metrics_turn.csv` on the same corrected run:

- Turn-level exact match against FSM `true_state_after`: 24/38 = 0.632
- FSM dialogues: 10
- `valid_flow_path` agreement between gold and labelled paths: 7/10 = 0.700
- Downward bias count (`gold=True`, `labelled=False`): 3/10 dialogues

The topic-vs-stage bias remains but is smaller than on the 11/09 instrument run.
At the `v1` freeze point, no stage-labeler retune was applied.

### Frozen rubric versions

- `judge_facts.md v3`
- `judge_global.md v2`
- `judge_shared.md v2`

Tag `v1` comes after this section and freezes prompts and rubrics.

## Pilot v2 validation round (T-16 closure)

Second round on the instrument-fixed, 9B-simulated-user pilot, run under the configuration that
T-17 then used, its judge backend aside. T-16 closed on it (tag `v1.1`). Same judge
(`gemma4:12b`), same rubrics (`judge_facts.md` v3, `judge_global.md` v2,
`judge_shared.md` v2), same protocol as above. Not pooled with the T-16
`v1` freeze. **Do not redraw these sheets.**

| | |
|---|---|
| Run | `runs/pilot_v2` (manifest `exp_id` `pilot_v2`) |
| Sheets | `results/judge_validation/pilot_v2/` |
| Eligible responses | 78 (39 baseline, 39 FSM) |
| Drawn | 30 responses: 15 baseline, 15 FSM |
| Dialogues | census of all 20 ok dialogues |
| Seed | `20260914` |
| Adherence | 20/20, 4/4 canary, 4/4 injection |

Computed with:

- `uv run python scripts/validate_judge.py agree --run runs/pilot_v2 --sample results/judge_validation/pilot_v2`
- `uv run python scripts/validate_judge.py labeler --run runs/pilot_v2`

### Agreement table (Pilot v2)

| Field | Unit | n | Percent agreement | Kappa |
|---|---|---:|---:|---:|
| `task_completed` | dialogue census | 20 | 0.850 | 0.667 (Cohen) |
| `accuracy` | dialogue census | 20 | 0.750 | 0.569 (quadratic weighted Cohen) |
| `claim_support` | response sample | 30 | 0.733 | 0.533 (Cohen) |
| `fact_ids_stated` exact-set | response sample | 30 | 0.667 | n/a (set metric) |
| `fact_ids_stated` micro precision | response sample | 30 | 0.966 | n/a |
| `fact_ids_stated` micro recall | response sample | 30 | 0.718 | n/a |
| `fact_ids_stated` micro F1 | response sample | 30 | 0.824 | n/a |

Unmatched claim attribution cases: 17 claims with zero or multiple turn matches.
They are reported and excluded from forced turn-level labels by design.

### Disagreement table (Pilot v2)

| Field | Unit | Count | IDs (human → judge) |
|---|---|---:|---|
| `task_completed` | dialogue census | 3 | `D01` no→yes; `D02` no→yes; `D04` no→yes |
| `accuracy` | dialogue census | 5 | `D01` partial→correct; `D02` partial→correct; `D04` partial→correct; `D08` partial→correct; `D13` partial→incorrect |
| `claim_support` | response sample | 8 | `A02` some_unsupported→none_checkable; `A12` some_unsupported→all_supported; `A15` all_supported→none_checkable; `A17` all_supported→none_checkable; `A18` some_unsupported→none_checkable; `A19` some_unsupported→all_supported; `A21` some_unsupported→none_checkable; `A22` some_unsupported→none_checkable |
| `fact_ids_stated` (exact-set mismatch) | response sample | 10 | `A03`, `A05`, `A08`, `A09`, `A12`, `A15`, `A17`, `A22`, `A24`, `A27` |

Join IDs to scenario/agent/turn in `results/judge_validation/pilot_v2/sample.json`.
The three `task_completed` disagreements are all `edge_19`.
The judge was not retuned using this validation sample.

### Stage labeler on Pilot v2

Measured from `metrics_turn.csv` on `runs/pilot_v2`:

- Turn-level exact match against FSM `true_state_after`: 25/39 = 0.641
- FSM dialogues: 10
- `valid_flow_path` agreement between gold and labelled paths: 8/10 = 0.800
- Downward bias count (`gold=True`, `labelled=False`): 2/10 dialogues

Mismatch analysis on all 14 FSM mismatches (`labelled_stage != true_state_after`)
confirms one narrow contract issue: revision 1 labels the assistant's speech act
more than the procedure stage after FSM auto-advance. The eight grouped patterns
are in `results/judge_validation/pilot_v2/labeler_v1/metrics_turn.csv` and in
the side-by-side notes of this round.

### Stage-labeler revision 2 candidate (development check only)

Revision 2 was written after inspecting those 14 Pilot v2 errors. Its comparison
on the same 39 turns is therefore post-hoc/resubstitution evidence, not
independent validation.

- Candidate relabel sidecar: `runs/pilot_v2_labeler_v2/`
- Source run left frozen: `runs/pilot_v2/`
- Command: `just relabel-stages runs/pilot_v2 runs/pilot_v2_labeler_v2`
- Validation command: `just validate-labeler runs/pilot_v2_labeler_v2`

Measured on the same 39 FSM turns:

- Turn-level exact match: 28/39 = 0.718 (up from 25/39)
- `valid_flow_path` agreement: 7/10 = 0.700 (down from 8/10)
- Downward bias (`gold=True`, `labelled=False`): 3/10 (up from 2/10)

The candidate fixed 4 revision-1 errors and introduced 1 new wrong label, but it
made the harmful path-level bias worse. Adoption rule is not met.

Decision: **KEEP STAGE-LABELER REVISION 1**.
Experimental instrument remains v1.

The stage-labeler revision-2 comparison is used as a regression/development check
only. No independent holdout was used to estimate revision-2 accuracy. This
limitation affects only the diagnostic stage-labeling and flow-adherence columns;
`flow_adherence` stays out of `PRIMARY_METRICS`.

## Pilot v2 robustness check (`gemma4:12b-mlx`)

Same 20 dialogues, same 30 human response labels, same prompts
(`judge_facts.md` v3, `judge_global.md` v2, `judge_shared.md` v2), same schemas,
and the same `agree()` path (including `attribute_claims()`). Dialogues were
copied to `runs/pilot_v2_mlx/` omitting the GGUF cache; `sim eval` scored that
sidecar with the configuration `gemma4:12b-mlx` served by Ollama. Human sheets
stay the frozen files under `results/judge_validation/pilot_v2/`. Prompts and
the judge stay frozen. The comparison checks that configuration against the
published validation, on the same sample.

Computed with:

- `just validate-judge runs/pilot_v2 results/judge_validation/pilot_v2`
  (must still match the published Pilot v2 table; it did)
- `just eval runs/pilot_v2_mlx 2`
- `just validate-judge-compare`

Artifacts: `results/judge_validation/pilot_v2_mlx/`. Observed judge identity
from that sidecar's `llm_calls.jsonl`: tag `gemma4:12b-mlx`, digest
`ded7a2735003…`. Configured sampling from `configs/models.yaml`: temperature 0,
seed 42, `num_ctx` 8192.

### Agreement table (Pilot v2 vs `gemma4:12b-mlx`)

| Metric | Original | `gemma4:12b-mlx` | Δ |
|---|---:|---:|---:|
| `task_completed` agreement | 0.850 | 0.900 | +0.050 |
| `task_completed` κ | 0.667 | 0.783 | +0.116 |
| `accuracy` agreement | 0.750 | 0.800 | +0.050 |
| `accuracy` QWK | 0.569 | 0.658 | +0.089 |
| `claim_support` agreement | 0.733 | 0.700 | −0.033 |
| `claim_support` κ | 0.533 | 0.500 | −0.033 |
| `fact_ids_stated` exact-set | 0.667 | 0.633 | −0.033 |
| precision | 0.966 | 0.897 | −0.069 |
| recall | 0.718 | 0.667 | −0.051 |
| F1 | 0.824 | 0.765 | −0.059 |

Complementary `fact_ids_stated` counts: original TP/FP/FN = 28/1/11; MLX =
26/3/13. Unmatched claim attributions: 17 → 27.

### Changed cases (original judge ≠ `gemma4:12b-mlx`)

Five units (`D04`, `A05`, `A08`, `A12`, `A29`). `changed_cases.csv` compares the two
`agree()` judge labels, not the human.

| unit | field | original | mlx |
|---|---|---|---|
| `D04` | `task_completed` | yes | no |
| `D04` | `accuracy` | correct | partial |
| `A05` | `claim_support` | all_supported | none_checkable |
| `A05` | `fact_ids_stated` | F06 | (empty) |
| `A08` | `fact_ids_stated` | F02 | F02;F04;F21 |
| `A12` | `claim_support` | all_supported | none_checkable |
| `A12` | `fact_ids_stated` | F01;F28 | (empty) |
| `A29` | `fact_ids_stated` | F10 | F04;F10 |

`D04` is `edge_19` / baseline / rep 2. The human label was `task_completed=no`,
`accuracy=partial`; the MLX labels match the human, which is why dialogue-level
agreement rose. `D01` and `D02` kept their original `task_completed` labels.

`A05` and `A12` are two turns of `edge_19` / fsm / rep 1. Both lost attributed
facts and became `none_checkable` (recall). `A08` (`happy_path_06` / baseline)
and `A29` (`adversarial_04` / baseline) each gained an extra `F04` (precision).
The two `claim_support` changes are empty attributions. The remaining 19
dialogues kept the same `task_completed` and `accuracy` labels.

The quoted Pilot v2 validation remains the GGUF table. Semantic-primary uses
the MLX configuration checked here. T-18 keeps GGUF and MLX as two exported
populations.

## Pilot v2 instrument sidecar (`qwen3.5:9b`)

Capacity check of the 4B event classifier and stage labeler. Not a retune.
Frozen Pilot v2 dialogues, prompts (`stage_labeler.md` v1, `event_classifier.md`
v4), schemas, temperature 0.7, seed policy, `think=False`, and `num_ctx` 8192
stay identical. Only the model tag/digest changes, copied from the already
configured agent `qwen3.5:9b` (digest `6488c96fa5fa…`). Human judge sheets were
not read. `configs/models.yaml` still has `classifier` and `state_labeler` on
`qwen3.5:4b`. 9B does not replace those instruments and does not enter T-17.

Labeler gold remains FSM `true_state_after` on the 10 FSM dialogues / 39 turns.
Classifier gold on the T-08 phrases is `tests/fixtures/user_events.jsonl`. The
39-turn classifier comparison is 9B versus the logged 4B `event`, not versus a
human event label.

Pre-declared bar, the same one that rejected labeler revision 2: **clear
improvement** only if turn match rises **and** path validity does not fall.

Computed with:

- `just validate-labeler runs/pilot_v2` (still 25/39 and 8/10)
- `just relabel-stages-9b`
- `just validate-labeler-compare`
- `just replay-classifier-pilot`
- `just replay-classifier-fixtures qwen3.5:4b`
- `just replay-classifier-fixtures qwen3.5:9b`

Artifacts: `results/judge_validation/pilot_v2_qwen9b/`. Sidecar runs:
`runs/pilot_v2_labeler_9b/`, `runs/pilot_v2_classifier_9b/`.

### Stage labeler (4B vs 9B)

| Metric | 4B (published) | 9B sidecar | Δ |
|---|---|---:|---:|
| turn match vs `true_state_after` | 25/39 = 0.641 | 27/39 = 0.692 | +2 |
| `valid_flow_path` agreement | 8/10 | 6/10 | −2 |
| downward bias | 2/10 | 4/10 | +2 |
| of the 14 4B mismatches, corrected | — | 5 | |
| remaining | — | 9 | |
| new errors on turns 4B had right | — | 3 | |

Turn match rose; path validity fell. The adoption bar is not met. **KEEP
STAGE-LABELER REVISION 1** and keep `qwen3.5:4b`.

### Event classifier

Pilot v2 FSM turns. The 11/28 split was verified against
`runs/pilot_v2/llm_calls.jsonl` (28 `classifier` calls) before quoting.

| | |
|---|---|
| `rule-fired` | 11/39 (no LLM; 9B matched the log on all 11) |
| `LLM-fallback` | 28/39 |
| event agreement 9B vs logged 4B, LLM-fallback only | **23/28** |
| event changes on those 28 | 5 |
| logged `intent_classified` | 5 |
| intent changes among those 5 | 0 |

Headline is 23/28, not 34/39. The 11 rule turns cannot inflate agreement.

The five LLM event changes (logged 4B → 9B):

| dialogue | turn | logged | 9B |
|---|---:|---|---|
| `adversarial_13` / 1 | 2 | `intent_classified` | `out_of_scope_request` |
| `adversarial_13` / 2 | 3 | `solution_accepted` | `out_of_scope_request` |
| `edge_19` / 1 | 1 | `request_received` | `none` |
| `edge_19` / 2 | 1 | `request_received` | `none` |
| `happy_path_06` / 2 | 2 | `none` | `intent_classified` |

T-08 fixtures, re-run in this session (not the historical 100%):

| model | n | correct | accuracy |
|---|---:|---:|---:|
| `qwen3.5:4b` | 21 | 21/21 | 1.000 |
| `qwen3.5:9b` | 21 | 19/21 | 0.905 |

9B misses, both `request_received` → `none`: greeting “Hi, I need help with an
order.”; `out_of_scope` “Actually I have a question about my order.”

A 9B event that disagrees with the Pilot v2 log is a counterfactual on a
4B-produced path. Dialogues were not re-simulated.

### Cost (secondary)

The Pilot v2 means in `docs/parity.md` (labeler 20 calls / 9.39 s; classifier
28 calls / 9.24 s) are a **historical reference**, not a controlled 4B vs 9B
latency benchmark. They may come from different machine load, residency, and
date. Quality metrics do not use them.

This sidecar, observed from `llm_calls.jsonl` plus operator wall clock:

- labeler 9B: 20 uncached calls, mean prompt 610.5, mean output 36.1, mean
  latency 5.93 s, wall ~119 s
- classifier 9B on 28 LLM-fallback turns: 28 unique prompt hashes, 34 uncached
  log lines, mean uncached latency 10.35 s, wall ~80 s
- T-08 fixtures: 14 LLM calls each (7 phrases are rules); 4B mean 5.43 s, 9B
  mean 8.80 s

### Conclusion

The pre-declared bar for a model-capacity effect is not met: turn match rose
and path validity fell, the same pattern that rejected labeler revision 2. On
the T-08 human gold phrases, 9B scored worse than 4B (19/21 vs 21/21). Little
movement that counts as a capacity win. The test provides no evidence that
model capacity is the main explanation for the 14 labeler errors and is
consistent with limitations in the stage contract or prompt. It cannot show
that the prompt or contract caused the errors. Experimental instruments stay
revision 1 and `qwen3.5:4b`.

## Later census of semantic-primary (not T-16)

T-16 stays the instrument-validation protocol on the pilot. After the Final
Experiment freeze, one annotator labelled the same 348 scored
`semantic_primary` dialogues (blind IDs D001–D348, freeze 2026-09-20T21:11:58Z,
then unblind). That sidecar does not retune the judge and does not edit T-16
sheets. The judge stays the pre-declared confirmatory instrument; the census re-ran the
same Holm family and was chosen as the narrative reference for the thesis conclusion after
unblind, a choice declared as such. Holm family, human vs judge agreement, the 107 omitted fact-ID
reconstructions and their post-hoc sensitivity: [`docs/metrics.md`](metrics.md) (Human census of
semantic-primary),
[`docs/decisions_and_limitations.md`](decisions_and_limitations.md),
[`results/human_primary/`](../results/human_primary/README.md),
[`results/sensitivity/ambiguous_fact_ids/`](../results/sensitivity/ambiguous_fact_ids/README.md).

