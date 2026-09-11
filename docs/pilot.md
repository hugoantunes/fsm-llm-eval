# Pilot (T-15)

5 scenarios × 2 agents × 2 repetitions = 20 dialogues, run and evaluated **entirely on the Air**, as
T-15 asks. All 20 dialogues were read. Directory: `runs/exp_pilot` (git-ignored).

Scenarios: `happy_path_06`, `edge_02`, `edge_19`, `adversarial_04`, `adversarial_13` — one per
category, one needle, one canary injection. Dataset `data/scenarios/v1`, hash
`0778a90110e6dd67685c1e6768934483cb4405213dad35ec172f3ddcf21d47c9`.

Result: 20 ok, 0 failed, 0 unscored.

The directory was renamed by hand after the run, so `manifest.json` still records
`exp_id: exp_pilot2`. Nothing reads it — `sim eval --run` takes the directory path — and rewriting a
run artifact after the fact would be worse than the mismatch. Noted here so the next reader is not
surprised.

**The numbers below are exploratory.** Five scenarios cannot separate two agents; they are here to
show the pipeline produces the columns and to size the machine budget. Nothing in this file is a
result of the experiment, and no cut was decided from it.

## What broke

**The user-event classifier could return `intent_classified` without an intent.** The FSM engine
then advanced out of `intent_classification` with nothing to collect data for, and the dialogue
derailed. Found on the first Air pilot. Two fixes, one in the contract and one in the prompt:

- `src/sim/events.py`: a pydantic `model_validator` on the event model makes
  `intent_classified ⇒ intent is not None` a parse error rather than a silent advance, raised as the
  new `ClassifierSchemaError` (split out of `EventError`) and retried `SCHEMA_RETRIES` times with
  `seed + attempt`. The JSON Schema Ollama enforces through `format=` cannot express the cross-field
  rule, so the same seed would replay the same invalid object; the retry changes the seed.
- `data/prompts/event_classifier.md` **v3 → v4**: `intent_classified` must name an intent, and a
  bare acknowledgement ("ok", "thanks") must not be classified as `intent_classified`.

That first pilot directory was removed during pilot iteration, after the fixes landed and the run
was replayed. It is not part of the record; `runs/exp_pilot` is the run this file reports.

**The judge could attach a `fact_id` to a claim it had just marked unsupported**, which inflated the
predicted-ID set and therefore false positives. `data/prompts/judge_facts.md` **v2 → v3**: `fact_id`
names only the one fact that supports the claim, and on a `no` or `unverifiable` claim `fact_id` is
always null. The rubric and the schema moved together — `JudgeClaim` became a discriminated union
(`SupportedClaim` / `UnsupportedClaim`), so `fact_id: None` is a type on the unsupported branch and
`format=` rules the shape out for Ollama exactly as pydantic does for a cached payload.

**`valid_flow_path` was constant `False`** — all 20 rows, both agents, so the column and
`flow_adherence` carried no information at all. Two independent defects in the definition, not in
the labeler:

1. it excluded the `from: "*"` edges, so every legal `out_of_scope` escape and every legal
   `farewell` scored invalid; and
2. it assumed one edge per turn, while `FsmEngine.step` fires the classified user event and then
   offers two auto-advance events — so a turn can land two states on.

Together these marked **8 of the FSM's own 10 recorded paths** invalid: the metric contradicted the
machine it was measuring. Now a step between two distinct labels is valid when at most
`MAX_FLOW_EDGES_PER_TURN` = 2 flow edges join them, or when it takes a `from: "*"` edge. Both
universal edges count: asking for something out of scope and saying goodbye are moves of the *user*,
which every state answers. `happy_path_06` is the case that settles it — an informational goal,
answered in turn 2, user says goodbye, `stop_reason: goal_reached`; the flow's
`data_collection → solution → confirmation` stages simply do not apply, and scoring it invalid would
punish the agent for succeeding efficiently.

The bound is what keeps the column from collapsing into "any legal path": it still rejects 11 of 20
labelled paths, against 7 of 20 for `ended_in_expected_state`. All 10 FSM gold paths now validate.
Re-evaluation cost nothing — the prompt-hash cache replayed every call, 0.23 s for the 20 dialogues.

**2 is the engine's ceiling, not a threshold fitted to those 10 paths.** `FsmEngine.step` fires
exactly one classified user event, then `_advance_when_ready` offers exactly two auto-advance
events, in a fixed order, each at most once, with no loop: `order_identified` and `data_provided`.
All three can never fire — `order_identified` lands on `intent_classification`, `data_provided` is
declared only out of `data_collection`, and the single edge between them is `intent_classified`,
which `_advance_when_ready` deliberately excludes so that a misread request is not made final before
the state built to settle it has spoken. Assume every guard passes, try every state against every
event, and the longest chain the machine admits is
`greeting --request_received--> identification --order_identified--> intent_classification`: two
edges. One label is one turn and consecutive labels are consecutive turns' `state_after`, so two is
exactly the gap a turn can produce and a wider one is a skipped stage.
`test_max_flow_edges_per_turn_is_the_ceiling_the_engine_can_reach` rederives this from the loaded
spec, so the constant fails the suite if `machine.yaml` changes under it. The gold paths are a check
on the derivation, not its justification — and had they disagreed, the derivation would have won.

## What changed

| Artifact | Change |
|---|---|
| `data/prompts/event_classifier.md` | v3 → v4: `intent_classified` must name an intent; a bare acknowledgement is not an intent |
| `data/prompts/judge_facts.md` | v2 → v3: `fact_id` only on a `yes` claim, null otherwise; schema became a discriminated union |
| `src/sim/events.py` | `ClassifierSchemaError` + cross-field validator + schema retries |
| `src/sim/evaluators.py` | `_is_valid_flow_path`: bounded walk k ≤ 2 over flow edges, `from: "*"` destinations reachable from anywhere |
| `src/sim/metrics.py` | `flow_adherence` dropped from `PRIMARY_METRICS` |
| `scripts/run_stats.py` | new: per-caller tokens and latency from `llm_calls.jsonl` |

State packages, the simulated user and the stage-labeler prompt were read and left unchanged.

## Cost per dialogue

From `llm_calls.jsonl` and the manifest (`uv run python scripts/run_stats.py runs/exp_pilot`).
Latency is averaged over uncached calls only. The three eval callers show **60 calls, not 20**:
the log is append-only, and this directory was evaluated three times (the later two from cache,
after the flow-rule change). Token and latency means match the first uncached pass.

| caller | calls | mean prompt tokens | mean output tokens | mean latency (s) |
|---|---|---|---|---|
| `baseline` | 40 | 2257.1 | 56.1 | 15.54 |
| `fsm` | 38 | 2023.9 | 43.6 | 17.76 |
| `classifier` | 25 | 1210.3 | 20.6 | 9.47 |
| `simulated_user` | 79 | 662.0 | 40.5 | 10.67 |
| `stage_labeler` | 60 | 648.7 | 36.5 | 6.12 |
| `judge_facts` | 60 | 2890.2 | 614.8 | 139.10 |
| `judge_global` | 60 | 1396.8 | 79.9 | 35.41 |

**Mean prompt tokens per agent: baseline 2257, FSM 2024.** The FSM's per-state package is smaller
than the baseline's single prompt, and both carry the same full knowledge base. The FSM pays a
separate classifier call per turn instead (1210 prompt tokens, 9.5 s), which is why its
`turn_latency_s` exceeds its `llm_latency_s`. T-16 carries this into the parity table.

**Run: 57.7 s/dialogue. Eval: 180.6 s/dialogue.**

Projected at N = 60, K = 3 (360 dialogues per phase):

| Phase | Planned | Measured | Measured on | T-17 runs it on |
|---|---|---|---|---|
| `run` | 6.6 h | **5.8 h** | Air | Air |
| `eval` | 9.4 h | **18.1 h** (serial) | Air | **Pro** |

The `eval` row is measured on one machine and planned for another, which is a gap and not a
detail: the pilot ran end to end on the Air, so 180.6 s per dialogue, the 18.1 h, and the
concurrency measurement below are all M4 / 24 GB numbers. The Pro is an M5 with 16 GB. Nothing
here has been measured on it.

Run is inside budget. **Eval is 1.9× over**, and the overrun is one caller: `judge_facts` at 139.1 s
a call. Against `judge_global` (79.9 output tokens, 35.4 s) on the same model, the fixed cost is
≈ 20 s and generation ≈ 119 s, so the 614.8 output tokens are the whole of it — the atomic-claims
list is long by construction. Raising an Ollama token limit does not help: `num_predict` is unset,
so nothing is being truncated, and `num_ctx` only sizes the KV cache (larger would risk the eviction
penalty measured on the Pro on 2026-09-08, and would change every prompt hash, discarding the
cache). The two real levers are fewer judge output tokens, which is a rubric change and so has to
happen before the `v1` tag, or concurrency in `sim eval`.

### Concurrency in `sim eval`

The 18.1 h above is the sum of uncached call latency, so it is the **serial** figure. `sim eval`
now takes `--parallel` (default 1), superseding the 2026-09-10 T-14b decision that left it out. It
is validity-neutral by construction: each dialogue is graded on its own, the judge and labeler seeds
come from `configs/models.yaml` and never from position in the queue, and the CSV rows are ordered
by the manifest, so the files are byte-identical to a sequential eval. A test asserts exactly that.

Two facts decide how much `--parallel 2` actually buys, and they cut the opposite way from the run
phase:

- **The eval load is one model.** Split by model, the 180.6 s per dialogue is `gemma4:12b` 174.5 s
  (96.6%) and `qwen3.5:4b` 6.1 s (3.4%). So overlapping the two models — the mechanism behind the
  run phase's gain — is worth at most 3.4% here. Everything depends on `gemma4:12b` serving two
  requests at once.
- **It does, and Qwen does not.** Across both Ollama server logs, `model architecture does not
  currently support parallel requests` fires 49 times and every one of them reads
  `architecture=qwen35`; `gemma4` never triggers it. So the judge gets its two slots, while the run
  phase's models never did — which is why the run phase's **1.4×** cannot be carried over. It is
  also why the slots cost nothing extra: `ollama ps` on the Air during the pilot shows `gemma4:12b`
  resident at 9.0 GB against the 8.6 GB single-slot figure, so the second slot's KV cache is already
  allocated at load time by `OLLAMA_NUM_PARALLEL=2` and is paid for whether `sim eval` uses it or not.

Judge time is generation-dominated (≈ 119 s of 139.1 s), which suggested batched decode would beat
the run phase's 1.4×. **Measured on the Air, it does not.** Two real `judge_facts` prompts replayed
straight against Ollama, one arm at a time and one arm two at a time:

| | prompt eval | generation | wall |
|---|---|---|---|
| sequential | 28.9 s + 27.0 s | 37.8 s + 57.1 s = 94.9 s serial | 156.1 s |
| two at once | 0.3 s + 0.2 s | 68.1 s and 89.8 s, overlapped | 93.9 s |

The raw ratio reads 1.66×, and **it is an artifact**. With `OLLAMA_NUM_PARALLEL=2` each slot still
held one of the two prompts from the first arm, so the second arm skipped prompt processing
entirely — 55.9 s, 36% of the sequential wall clock. A real eval never gets that: the sequential arm
is the proof, where the second call paid its full 27.0 s with the first call's prompt one slot away.
360 distinct transcripts through 2 slots reuse nothing.

What is left once the artifact is removed is the number that matters, and it is flat:

- **Generation-only throughput: 1.06×.** 94.9 s of serial generation against 89.8 s elapsed.
- **Per-stream token rate collapses**: 12.0 → 6.7 tok/s and 11.6 → 7.4 tok/s, i.e. 0.56× and 0.64×.
  Two streams at a bit over half speed is **1.19×** aggregate at best.
- **Honest wall speedup ≈ 1.10×**, charging the concurrent arm the prompt eval it dodged.

`gemma4:12b` accepts the second slot; the GPU has nothing left to give it. Decode on a 12B at Q4 is
memory-bandwidth-bound, and one stream already saturates the bandwidth, so a second stream splits it
rather than filling idle capacity. Accepting parallel requests and benefiting from them are
different things, and the architecture warning only rules out the first.

| Gain | Eval at N = 60, K = 3 | vs planned 9.4 h |
|---|---|---|
| 1× (serial) | 18.1 h | 1.9× over |
| **1.10× (measured)** | **16.5 h** | **1.8× over** |

So `--parallel 2` is worth roughly 1.6 h of the 18.1, not the 5–9 h the bracket assumed. It stays in
because it is free and validity-neutral, but **it is not the lever, and the eval budget should be
planned at ~17 h.** The levers that remain are fewer judge output tokens — ruled out until after
T-16, since it invalidates the hand-annotated judge validation and the `v1` freeze — or splitting
the 360 dialogues across the Air and the Pro, which are separate GPUs and so really do halve it.
That second one needs code before the 2026-09-14 freeze and is not costed here.

### The Pro is unmeasured, and that is the larger number

Everything above ran on the Air. `ollama ps` before and after the probe showed `gemma4:12b` and
`qwen3.5:4b` both resident with no eviction, which settles the memory question **for the Air and
only for the Air**. T-17 assigns `eval` to the Pro, and on the Pro nothing here has been checked:

- **Memory.** The eval phase needs both models resident, 9.0 + 3.4 = 12.4 GB, against the
  6.3–8.1 GiB of free system RAM measured on the Pro on 2026-09-08. If it evicts, the penalty
  recorded that day is **3.3×** — against which the 1.10× above is noise. It applies at
  `--parallel 1` too, so it is not created by concurrency and not avoided by dropping it.
- **The 1.10× itself.** The reason it is 1.10× and not 2× is that one decode stream already
  saturates memory bandwidth. Bandwidth is a property of the chip, and the Pro is an M5 while the
  measurement is an M4. The ceiling could sit anywhere on the Pro; assuming it is the same number
  repeats exactly the mistake this measurement just corrected.
- **The 18.1 h.** Same problem one level up: it is a sum of Air latencies. The Pro's per-call
  judge time is unmeasured, with or without eviction.

Re-running the probe on the Pro costs about 8 minutes and answers all three at once. Until it is
run, the defensible plan is the one the pilot actually demonstrated: **`eval` on the Air, ~17 h**,
with the Pro as the optimisation rather than the assumption. That reverses the machine split of
2026-09-08 for the `eval` phase only, and it is a scheduling choice, not a measurement one —
`run` and `eval` are separate phases and no metric mixes the two machines either way.

## The transfer path

T-15 asks for the SSH path to be exercised with real files while there is still time to fix it. Done
on the Air against `localhost`: the Air's own key was added to its `~/.ssh/authorized_keys`, then
`rsync -az --partial -e ssh runs/exp_pilot localhost:<dest>/` moved all 258 files with identical
checksums, and `sim eval --run <dest>/exp_pilot` scored the 20 dialogues **with no LLM call** —
`cache/` travels with the run, so re-evaluating on the second machine is free for anything already
judged. The network hop and the Pro's `authorized_keys` are still untested; `ssh-copy-id` to the Pro
is the one remaining step before T-17.

## The measurement risk T-16 has to close

The stage labeler is the weakest instrument in the pipeline, and the flow columns rest on it.

**Turn-level accuracy against the FSM's true `state_after`: 23/38 = 0.605.** Dominant confusions:

| n | true | labelled |
|---|---|---|
| 3 | `intent_classification` | `solution` |
| 3 | `intent_classification` | `confirmation` |
| 2 | `identification` | `data_collection` |
| 1 | `solution` | `data_collection` |
| 1 | `confirmation` | `out_of_scope` |
| 1 | `identification` | `intent_classification` |

The labeler reads the agent's reply and hears the *topic* the agent is discussing, not the stage it
is in: an agent that states a price while still classifying the request gets labelled `solution`.

**Path-level, this matters more than 0.605 suggests.** Exact path match is 2/10. On the derived
column, the labelled path and the gold path agree on `valid_flow_path` in only **5 of 10** FSM
dialogues — and all five disagreements run the same way, gold `True` and labelled `False`. The
labeler jumps ahead, which manufactures apparent multi-stage jumps that the k ≤ 2 bound then
rejects. This is a systematic downward bias, not noise.

The bias applies to both agents, since the labeler is the same instrument run over each agent's turn
identically, so the paired comparison is partly protected. But there is no gold for the baseline,
so the magnitude cannot be checked on that side. **`valid_flow_path` and `flow_adherence` therefore
carry no inferential weight**, which is why `flow_adherence` is out of `PRIMARY_METRICS`. T-16 has
to quantify this in `docs/judge_validation.md` and decide whether a labeler prompt change is worth
it before the `v1` tag freezes it. Re-labelling the full experiment is cheap if it is: 360 × 6.12 s
≈ 37 min, and nothing else has to re-run.

## Exploratory direction

Means over 10 dialogues per agent. **Five scenarios; no test, no claim.**

| metric | baseline | fsm |
|---|---|---|
| `fact_f1` | 0.524 | 0.635 |
| `fact_precision` | 0.413 | 0.517 |
| `fact_recall` | 0.867 | 0.933 |
| `claim_support` | 0.928 | 0.936 |
| `accuracy_score` | 0.900 | 0.750 |
| `relevance` | 5.00 | 4.30 |
| `task_completed` | 0.800 | 0.600 |
| `n_turns` | 4.0 | 3.8 |
| `llm_latency_s` | 62.2 | 67.5 |
| `turn_latency_s` | 62.2 | 91.2 |

The FSM is ahead on the factual columns and behind on the judged ones. Its dominant failure mode is
visible in the dialogues: the classifier sends an in-scope turn to `out_of_scope` and the state
package then instructs the agent to decline. In `edge_19/fsm/rep01` a partial-cancellation request
escaped that way, the needle was never recovered and the run ended `user_gave_up`; the baseline
reached `goal_reached` on the same scenario in 8 turns. This is measured FSM behaviour, not a bug to
patch — the classifier is part of the architecture under test, and `event_classifier.md` was not
tuned to rescue it.

One confound was checked and **not** found: `fact_f1` does not penalise verbosity asymmetrically.
Correlation between `fact_precision` and `n_checkable_claims` is 0.08, and the two agents produce
almost the same number of claims per dialogue (6.5 baseline, 6.7 FSM).
