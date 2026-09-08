# Setup: machines and Ollama (T-03)

Record of the execution environment. Each item becomes a sentence in *Material e Métodos* or a row
in the Decisões table of `TICKETS.md`.

## Machines

| Role | Machine | Chip | RAM | macOS | Ollama | Checked on |
|---|---|---|---|---|---|---|
| `run` (agents + simulator) and writing | MacBook Air (Mac16,12) | Apple M4, 10 cores (4P + 6E) | 24 GB | 26.6.2 | 0.33.3 (app) | 2026-09-08 |
| `eval` (judge) | MacBook Pro (Mac17,2) | Apple M5, 10 CPU cores (4 Super + 6 Efficiency), 10 GPU cores | 16 GB | 26.4.1 (25E253) | 0.33.3 (app) | 2026-09-08 |

**Roles swapped on 2026-09-08**, reversing the assignment of 2026-09-07. The `run` phase needs two
models resident at once and only the Air's 24 GB delivers that; the `eval` phase runs one model at a
time, which the Pro's 16 GB handles comfortably. Measurements and reasoning in *Memory on the Pro*
below. Consequences: `runs/` now moves Air to Pro by `rsync`, and the Air does both the run blocks
and the writing, so the run blocks want to be overnight.

The two machines run different macOS versions (Air 26.6.2, Pro 26.4.1). It does not affect the
comparison: each phase runs entirely on one machine, `run` on the Air and `eval` on the Pro, so no
metric is computed from numbers produced by both. What has to match across machines is the Ollama
version and the model digests, and both do.

`iogpu.wired_limit_mb` is `0` (automatic) on the Pro, and raising it would change nothing: the
binding constraint turned out to be free *system* RAM, not the GPU allowance. See *Memory on the
Pro*.

Toolchain on both machines: `uv` 0.11.6, Python 3.13.13, `ollama` Python package 0.6.2.

The repository lives in `~/projects/fsm-llm-eval` on both machines, outside iCloud. Code travels by
GitHub; `runs/` is git-ignored on purpose and moves Air to Pro by `rsync` over SSH on the local
network.

Transfer path, state on 2026-09-08:

- **Remote Login is on** on the Pro, port 22 accepting connections, access limited to
  Administrators. `systemsetup -setremotelogin on` fails from a terminal without Full Disk Access;
  the System Settings toggle under General, Sharing does not need it. The machine's own name and
  address are deliberately not recorded here: read them off that Sharing pane when you need them.
- **Key auth is not set up yet.** The Pro has no `~/.ssh/authorized_keys`, so `rsync` will prompt
  for a password on every block. Run `ssh-copy-id <user>@<pro-host>` from the Air once. Needed for
  anything unattended.
- **"Allow full disk access for remote users" is on and is not needed here.** `rsync` writes into
  `~/projects/fsm-llm-eval/runs/`, which is not a TCC-protected path. Turning it off keeps SSH
  sessions away from Mail, Messages, Photos, Safari data and Time Machine without affecting the
  transfer.
- Still untested end to end, because it needs both machines awake. Do that during the pilot (T-15),
  not on the day of the full run.

The transfer is an optimization worth about 4 h, not a dependency: if it fails, run and eval both
happen on the Air in sequence, 12.3 h instead of 8.3 h, with nothing lost, because the two phases
are already separate and cached by prompt hash.

## Ollama installation

macOS app (`/Applications/Ollama.app`), which keeps `ollama serve` on `http://localhost:11434`. The
version has to be the same on both machines:

```bash
ollama --version
curl -s localhost:11434/api/version
```

## Environment variables

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_MAX_LOADED_MODELS` | 2 | lets agent and simulator stay loaded together; otherwise Ollama unloads and reloads on every turn and latency triples, measured at 3.3x on the Pro. **It permits, it does not guarantee**: the scheduler still evicts when free system RAM is short, which is what happens on the Pro. See *Memory on the Pro* |
| `OLLAMA_KEEP_ALIVE` | -1 | never unload a model for inactivity |
| `OLLAMA_NUM_PARALLEL` | 2 | the runner runs 2 dialogues at once (`sim run --parallel 2`). Qwen 3.5 serves one request per model at a time, so the gain is cross-model overlap, not two agent calls; see *Parallelism* below |
| `OLLAMA_CONTEXT_LENGTH` | 8192 | server default equal to the experiment's `num_ctx`. The client (T-07) sets `num_ctx` on every call; this is the second defense against the silent truncation of the beginning of the prompt, which would cut the baseline's KB |

The app inherits its environment from `launchd`, so `export` in a shell has no effect: the way is
`launchctl setenv` followed by a restart of the app. `scripts/ollama_env.sh` does both, prints the
configuration the server actually loaded, and fails if any value did not match. **It does not
survive a reboot**: run it again after each restart of the machine, before a `run` or `eval` block.

Manual check (configuration line in the server's startup log):

```bash
grep -E 'OLLAMA_NUM_PARALLEL' ~/.ollama/logs/server.log | tail -1 | tr ' ' '\n' \
  | grep -E '^OLLAMA_(NUM_PARALLEL|MAX_LOADED_MODELS|KEEP_ALIVE|CONTEXT_LENGTH):'
```

Defaults observed in Ollama 0.33.3 before configuration: `NUM_PARALLEL:1`, `MAX_LOADED_MODELS:0`
(automatic), `KEEP_ALIVE:5m0s`, `CONTEXT_LENGTH:0` (server default, 4096 tokens),
`FLASH_ATTENTION:false`. `KEEP_ALIVE=-1` shows up in the log as the maximum duration
(`2562047h47m16s`), not as `-1`.

`OLLAMA_FLASH_ATTENTION=1` with `OLLAMA_KV_CACHE_TYPE=q8_0` halves the KV cache. Only with a
Decisões row, and on both machines. Two measured notes on where it is worth reaching for, both from
2026-09-08:

- **Not for the `run` phase.** On the Qwen pair it recovers about 0.2 GB, because their KV cache is
  only 306 MiB against 4717 MiB of weights, and it does not make the pair co-resident on 16 GB.
  That is one of the reasons `run` moved to the Air. See *Memory on the Pro*.
- **Possibly for the judge**, which is now the Pro's job. `gemma4:12b` carries 1216 MiB of KV, four
  times the Qwen figure, so quantizing it would actually free something. The judge runs alone in
  16 GB and is not tight today, so this stays unused until it is needed.

| Variables check | Air | Pro |
|---|---|---|
| `scripts/ollama_env.sh` ran without error on | 2026-09-08 (Ollama 0.33.3; the server loaded all 4 values) | 2026-09-08 (Ollama 0.33.3; the server loaded all 4 values) |

## Models

Decision history (Decisões, T-03), mirrored in `configs/models.yaml`:

1. **2026-09-08, option A**: `qwen2.5:7b` (agents), `qwen2.5:3b` (simulator, classifier, labeler),
   `gemma3:12b` (judge). Chosen for memory fit on the 16 GB Pro (small KV cache), JSON Schema
   output and a third family for the judge. Pulled and validated on the Air (schema `enum` output,
   keep-alive, language measurement below).
2. **2026-09-08, revised** after checking the Ollama library, which the assistant's training data
   did not cover: **Qwen 3.5** for agents and simulator, **Gemma 4 12B** for the judge. Qwen 3.8
   and 3.6 exist only from 27B up (17.7 GB and more), so they cannot run on either machine.
   DeepSeek's own models are cloud-only or 400 GB, and its small "distills" are Qwen and Llama
   weights. Qwen 3.5 has 2B, 4B and 9B dense variants (2.7, 3.4 and 6.6 GB) with linear attention
   in 3 of every 4 layers, so its KV cache is *smaller* than Qwen2.5's; the binding constraint on
   the Pro is the weights. Gemma 4 12B (7.6 GB) is in the same footprint class as Gemma 3 12B.
   Both families have a thinking mode, switched off per request. A current generation also removes
   the "outdated model" objection from the thesis.
3. **Sizes**: fixed by the footprint measurement (`ollama ps` at `num_ctx` 8192 with
   `OLLAMA_NUM_PARALLEL=2`, which is machine-independent). Rule, declared before measuring: the
   strongest agent + simulator pair whose loaded footprint stays under the Pro's GPU budget of
   roughly 11 GB. Result: `qwen3.5:9b` + `qwen3.5:4b`, 9.1 GB loaded. The 4B was preferred over
   the 2B for the simulator because the event classifier and the stage labeler need reliability
   more than the extra speed, and memory allows it.

   **This rule used the wrong budget.** Measured on the Pro on 2026-09-08, the pair does not stay
   co-resident: Ollama admits a model against free *system* RAM, not free GPU memory. The footprint
   numbers above are right, the constraint they were compared against was not. See *Memory on the
   Pro* below. The models did not change; the `run` phase moved to the Air instead, which is the
   machine whose free RAM the rule should have been written against.

| Role | Model | Parameters | Download | Loaded footprint | Notes |
|---|---|---|---|---|---|
| Agents (`baseline` and `fsm`, the same model, same parameters) | `qwen3.5:9b` | 9.7B, Q4_K_M | 6.6 GB | 5.7 GB | same loaded size as Qwen2.5-7B; the vision tower is not loaded for text |
| Simulator, event classifier and stage labeler | `qwen3.5:4b` | 4.7B, Q4_K_M | 3.4 GB | 3.4 GB | faster than Qwen2.5-7B on both prompt processing and generation |
| Judge (the same in T-16 and T-17) | `gemma4:12b` | 11.9B, Q4_K_M | 7.6 GB | 8.6 GB | third family, a condition of the blind judge; same footprint and speed as Gemma 3 12B |

Candidates measured on the Air on 2026-09-08 (`num_ctx` 8192, 2 parallel dialogues, thinking
switched off where the model has it; speeds are indicative, several runs overlapped with model
downloads). Every model returned a valid schema-constrained `enum` and leaked no thinking text:

| Model | Download | Loaded | Prompt eval | Generation |
|---|---|---|---|---|
| `qwen3.5:9b` | 6.6 GB | 5.7 GB | 182 tok/s | 17.6 tok/s |
| `qwen3.5:4b` | 3.4 GB | 3.4 GB | 315 tok/s | 28.5 tok/s |
| `qwen3.5:2b` | 2.7 GB | 2.5 GB | 823 tok/s | 40.0 tok/s |
| `gemma4:12b` | 7.6 GB | 8.6 GB | 113 tok/s | 12.9 tok/s |
| `qwen2.5:7b` | 4.7 GB | 5.6 GB | 200 tok/s | 21.5 tok/s |
| `qwen2.5:3b` | 1.9 GB | 2.7 GB | 439 tok/s | 46.8 tok/s |
| `gemma3:12b` | 8.1 GB | 8.7 GB | 108 tok/s | 12.9 tok/s |

Alternatives considered: `llama3.1:8b` + `llama3.2:3b` for agents and simulator (KV cache with 8
heads and 32 layers: about 12 GB in the `run` phase, at the limit of what macOS gives the GPU on
16 GB, would need `OLLAMA_FLASH_ATTENTION=1` and a quantized KV cache); `qwen2.5:14b` or `phi4:14b`
as judge (9 GB, same order of time as Gemma 3 12B); `llama3.1:8b` as a fast judge (about half the
time, if the Air becomes the bottleneck). Sizes checked in the Ollama registry
(`registry.ollama.ai`) on 2026-09-08, default q4 tags.

Pull the configured models on both machines, the judge included on the Pro, and check the digests:

```bash
just pull-models     # ollama pull for every model in configs/models.yaml
just digests         # role, name and local digest, to paste into configs/models.yaml
just verify-models   # fails unless the local digests match the recorded ones
```

### Digests

The `ID` in `ollama list` is the first 12 characters of the manifest digest; `just digests` prints
the full one. Procedure: on the first machine, `just digests`, paste the values into
`configs/models.yaml`, commit; on the other machine, `just pull-models && just verify-models`. If a
digest differs the experiment is not reproducible across machines: `ollama pull` again until
`verify-models` passes.

The full digests live in `configs/models.yaml`; the table records the 12-character IDs that
`ollama list` shows.

| Role | Model | ID (Air, 2026-09-08) | ID (Pro, 2026-09-08) | Same? |
|---|---|---|---|---|
| agents (baseline and fsm) | `qwen3.5:9b` | `6488c96fa5fa` | `6488c96fa5fa` | yes |
| simulator, event classifier, stage labeler | `qwen3.5:4b` | `2a654d98e6fb` | `2a654d98e6fb` | yes |
| judge | `gemma4:12b` | `4eb23ef187e2` | `4eb23ef187e2` | yes |

`just verify-models` passes on both machines, so all three digests also match the ones recorded in
`configs/models.yaml`.

## Language and sanity checks (Air, 2026-09-08)

The experiment is English only (Decisões, 2026-09-08). Measured with `qwen2.5:3b` (same tokenizer
as `qwen2.5:7b`), `raw=True`, the same twelve-sentence e-commerce policy text in both languages:

| Text | Words | Characters | Tokens |
|---|---|---|---|
| English | 164 | 944 | 199 |
| Portuguese | 176 | 1012 | 284 |

Portuguese costs **1.43×** the tokens of the equivalent English text with the Qwen2.5 tokenizer,
so English cuts prompt processing, KV cache and judge time by roughly 30% for the same content.
With the final models the gap is smaller: **1.22** for Qwen 3.5 (242 vs 199 tokens) and **1.26**
for Gemma 4 (249 vs 198), so English saves about 20% of the tokens for the same content, not 30%.
The reliability argument for the small models stands.

Also checked on the Air with `qwen2.5:3b`: schema-constrained output with an `enum` of events
(`format=<JSON Schema>` through the `ollama` Python package 0.6.2) returned a valid value
(`order_identified`) in 0.5 s, 14 output tokens; with `OLLAMA_KEEP_ALIVE=-1` the model stays loaded
(`ollama ps` shows `Forever`, context 8192, 100% GPU); model load time 1.6 s.

## Memory on the Pro: the two models do not stay co-resident (2026-09-08)

The plan assumed the `run` phase would hold the agent and the simulator in memory together on the
Pro, so that a dialogue never pays a model load. On the Pro it does not happen. In a single
measurement of 2 parallel dialogues of 8 turns, the server logged **16 evictions**: the 9B agent and
the 4B simulator take turns being unloaded, so almost every call pays a model load plus a full
prompt re-processing (`forcing full prompt re-processing due to lack of cache data`).

The cause is not the GPU. Ollama's scheduler admits a model against **free system RAM**, and the
footprint of an already-resident model counts against that budget:

```
msg="llama-server model predicted to exceed available memory, evicting"
  predicted="6.4 GiB" available="1.9 GiB" gpu_free="8.7 GiB" system_free="1.9 GiB" system_limited=true
```

`gpu_free` is 8.7 GiB and Metal reports 11.8 GiB total, yet the model is refused because
`system_limited=true` and `available` tracks `system_free`. With the 9B resident (6.4 GiB) and about
8.0 GiB of system RAM free, only 1.6 GiB is left, and the 4B needs 3.3 GiB. Holding both wants
roughly 9.7 GiB free at once; with the machine in normal working use (browser, editor) free system
RAM measured between 6.3 and 8.1 GiB. `iogpu.wired_limit_mb` is irrelevant here, and raising it
would not help.

**The chip is not the problem, and the proof is within the Pro.** Running the same measurement with
a single model, so nothing can ever be evicted (`--parallel 1 --turns 4`, the 9B in both roles), the
agent call takes **6.2 s** against **20.4 s** with the pair thrashing, on an identical 3230-token
prompt. Eviction costs this machine a factor of 3.3. No cross-machine comparison is needed for that
claim.

An earlier draft of this section argued the same point from the judge, 47.2 s on the Pro against
48.6 s on the Air. **That argument does not hold**: the judge probe is only two calls, and two runs
on the Pro with identical prompts gave 47.2 s and 58.5 s. A 24% spread within one machine cannot
distinguish two machines, so the judge is too noisy to compare hardware and is quoted here only as a
budget figure.

**The pre-declared fallback does not fix it.** `OLLAMA_FLASH_ATTENTION=1` with
`OLLAMA_KV_CACHE_TYPE=q8_0` cut the resident footprint only from 5.6 + 3.3 GB to 5.5 + 3.2 GB, left
15 evictions in the same measurement, and moved the agent call from 27.6 s to 25.2 s (run budget
12.1 h to 11.0 h for N = 45, K = 3). That confirms the reasoning already recorded above: on Qwen 3.5
the binding constraint is the weights, not the KV cache, because linear attention already makes the
cache small. Measured as a diagnostic and reverted; the committed configuration is unchanged.

| Configuration | Agent resident | Simulator resident | Evictions | Agent mean | Run, N = 45 K = 3 |
|---|---|---|---|---|---|
| Pro, committed (f16 KV) | 5.6 GB | 3.3 GB | 16 | 27.6 s | 12.1 h |
| Pro, flash attention + q8_0 KV | 5.5 GB | 3.2 GB | 15 | 25.2 s | 11.0 h |
| Air, both co-resident | 5.7 GB | 3.4 GB | not counted | 10.6 s | 5.0 h |

The Air's eviction count was never measured; what is recorded is that `ollama ps` showed both models
resident at 9.1 GB, and an agent mean of 10.6 s is consistent with no reload, so "none" is an
inference and not a count.

### Like-for-like, at the same turn count

The rows above compare the Air at `--turns 4` with the Pro at `--turns 8`, which is not the same
workload: more turns means a longer history, so more prompt and more generation. Re-measured on the
Pro at `--turns 4` on 2026-09-08, the prompt and output token counts now match the Air's exactly,
so the work is identical:

| Call | Pro, 4 turns | Air, 4 turns | Prompt tokens | Output tokens | Ratio |
|---|---|---|---|---|---|
| agent | 20.4 s | 10.6 s | 3232 both | 44 both | 1.9x |
| simulator | 8.2 s | 4.3 s | 540 both | 31 both | 1.9x |
| classifier | 2.1 s | 1.7 s | 61 both | 16 both | 1.2x |

So the honest gap is **1.9x, not the 2.6x** the mismatched tables implied. The Pro is still the
worse `run` machine as configured, and the reason is memory capacity: the Air's 24 GB holds both
models and the Pro's 16 GB does not. Nothing here says the M4 is a faster chip than the M5.
Un-thrashed, the Pro does the same agent call in 6.2 s.

### Where the footprint actually goes

Ollama prints a memory breakdown per model load, which settles which knobs are worth touching:

| Model | Weights | KV context | Compute | Total |
|---|---|---|---|---|
| `qwen3.5:9b` | 4717 MiB | 306 MiB | 104 MiB | 5127 MiB |
| `qwen3.5:4b` | 2513 MiB | 306 MiB | 176 MiB | 2995 MiB |
| `gemma4:12b` | 7024 MiB | 1216 MiB | 170 MiB | 8410 MiB |

For the Qwen pair the KV cache is **6% of the footprint**. Every context-side lever, `num_ctx`,
`OLLAMA_NUM_PARALLEL` and `OLLAMA_KV_CACHE_TYPE`, is therefore capped at roughly 0.3 GB, which is
consistent with the 0.2 GB the q8_0 experiment actually recovered. Only smaller weights or more free
RAM move this. The judge is the opposite case: 1216 MiB of KV, so cache quantization would matter
there if the Pro, its machine since the role swap, ever got tight. It is not tight today, since the
judge runs alone.

### Decision: swap the machine roles (2026-09-08)

**`run` moves to the Air, `eval` moves to the Pro.** The Air holds both models with no eviction, and
the judge on the Pro runs one model at a time inside 16 GB. It costs no code and reverses only the
assignment of 2026-09-07.

What it costs: the Air is passively cooled, so a long block may throttle and the real throughput has
to come from the pilot rather than from these probes. The Air is also the writing machine, so run
blocks want to be overnight. Both are cheaper than a 3.3x eviction penalty on a metric T-13 reports.

The three options not taken, for the record:

1. **Free RAM on the Pro** and keep the roles. Borderline, since the pair wants about 9.7 GiB free
   of 16 GB total, and fragile: anything that reopens mid-block silently reintroduces the thrash
   into the agent-latency metric.
2. **Shrink the simulator** to `qwen3.5:2b` (2.5 GB resident, 8.1 GB for the pair). Still above the
   free RAM measured here, and it weakens the event classifier and the stage labeler, which are
   validity gates in T-13 and T-16.
3. **Accept 12.1 h of run on the Pro.** It fits the weekend in blocks, but under eviction the agent
   latency measures the machine rather than the agent.

Still to do on the Air before T-17: re-measure with `--turns 8` to replace the indicative 5.0 h with
a real figure, at parallelism 1 and 2 both, for the reason in the next section.

## Parallelism: `--parallel 2` works, but not as a clean 2x (2026-09-08)

Ollama offers no advice about the memory problem, but it does emit one warning on every load of a
Qwen 3.5 model:

```
level=WARN source=sched.go:509 msg="model architecture does not currently support parallel requests"
  architecture=qwen35
```

Fifteen occurrences across the runs of 2026-09-08, and **only** for `qwen35`. The judge's `gemma4`
never triggers it. The linear attention that makes Qwen 3.5's KV cache small is the likely reason.

**This is narrower than it first looks, and the run phase is mostly fine.** The warning says one
model cannot serve two requests at once. It does not say the run phase has no parallelism, because
the run phase uses **two different models**: dialogue A's agent call on the 9B genuinely overlaps
dialogue B's simulator or classifier call on the 4B. That cross-model overlap is the parallelism
that matters here, and it is exactly what a machine holding both models resident buys, which is now
the Air.

Measured in the pessimistic case, the same model in both roles so there is no cross-model overlap at
all, on the Pro with a single model loaded so nothing can be evicted:

| Configuration | Sum of role means per turn | Dialogues in flight | Work per turn |
|---|---|---|---|
| `--parallel 1` | 14.8 s | 1 | 14.8 s |
| `--parallel 2` | 20.9 s | 2 | 10.5 s |

So even with one model and the architecture warning in the log, two dialogues in flight give a
**1.4x throughput gain**. The mechanism was not chased down; host-side work overlapping GPU work is
a plausible explanation. With two models resident it should do better than 1.4x, which is what the
Air has to confirm.

What still needs care:

- **The budget's division by 2 is optimistic.** The true factor is somewhere between the measured
  1.4x and 2x, so the 5.0 h figure for N = 45, K = 3 on the Air is a floor. Re-derive it there with
  `--turns 8` at parallelism 1 and 2 before T-17.
- **A per-call latency measured at parallelism 2 is not the model's latency**, it includes queueing.
  On the Pro the agent call reads 20.4 s under eviction and queueing, 6.6 s at parallelism 2 without
  eviction, and 6.2 s alone. T-13 reports agent LLM latency as an efficiency metric, so it has to
  come from the per-call timer inside the runner, and the comparison between agents has to hold
  parallelism fixed. Worth settling in T-13 before the pilot.
- **Raising `--parallel` past 2 probably buys little.** Two models serving one request each saturate
  at two requests in flight. Worth one measurement on the Air rather than an assumption.

`OLLAMA_NUM_PARALLEL=2` stays: it costs nothing, it is what the judge uses, and it is what any
future non-Qwen model would use.

## Latency and budget

What `scripts/measure_latency.py` does: P parallel dialogues of T turns, each turn
one simulator call (persona + history), one schema-constrained classifier call and one agent call
(a 2.5k-token KB-sized system prompt + history); `--judge` adds two judge-sized calls (3.5k-token
prompt, structured verdict). It prints mean and p95 per role with token counts, then the budget for
N in {45, 60} and K in {3, 5}, assuming 8 turns per dialogue (`--budget-turns`):

```bash
scripts/ollama_env.sh
uv run python scripts/measure_latency.py --parallel 2 --turns 8 --judge
```

Measured on the Pro on 2026-09-08 with the final models and the committed configuration
(`--parallel 2 --turns 8 --judge`, 16 calls per role, machine in normal working use). **These
numbers include the eviction described above**: the agent and simulator rows are the cost of a call
plus a model load, not the cost of a call. They are why `run` moved to the Air, and they are kept
here as the evidence for that decision rather than as a working budget:

| Call | n | Mean (s) | p95 (s) | Prompt tokens | Output tokens | Output tok/s |
|---|---|---|---|---|---|---|
| agent (`qwen3.5:9b`) | 16 | 27.6 | 42.4 | 3436 | 60 | 2.2 |
| simulator (`qwen3.5:4b`) | 16 | 10.4 | 17.6 | 740 | 34 | 3.3 |
| classifier (`qwen3.5:4b`, schema) | 16 | 2.5 | 3.5 | 64 | 17 | 6.9 |
| judge (`gemma4:12b`), per call | 2 | 47.2 | 64.1 | 4338 | 248 | 5.3 |

Budget from those means, 8 turns per dialogue, 2 dialogues in parallel:

| N | K | Dialogues | Run on the Pro, with eviction | Judge |
|---|---|---|---|---|
| 45 | 3 | 270 | 12.1 h | 7.1 h |
| 45 | 5 | 450 | 20.2 h | 11.8 h |
| 60 | 3 | 360 | 16.2 h | 9.4 h |
| 60 | 5 | 600 | 26.9 h | 15.7 h |

The run column is history now that `run` moved to the Air, and it was 2.4x the Air's figure purely
because of the eviction. The judge column is the one that stays useful, and the Pro is the judge
machine, so those hours are the ones T-17 will actually spend on eval.

Earlier smoke test on the Air on 2026-09-08 (2 turns, 2 dialogues, Qwen2.5 + Gemma 3) while four
model downloads were running: it works end to end; those numbers were not recorded because of the
contention.

First run with the final models on the Air, 2026-09-08 (`--parallel 2 --turns 4 --judge`, no
downloads running). Since the role swap these agent, simulator and classifier rows are the ones that
count, because the Air is the run machine, but they are only four turns and need redoing at eight:

| Call | n | Mean (s) | p95 (s) | Prompt tokens | Output tokens |
|---|---|---|---|---|---|
| judge (`gemma4:12b`), per call | 2 | 48.6 | 72.0 | 4338 | 248 |
| agent (`qwen3.5:9b`) | 8 | 10.6 | 25.7 | 3232 | 44 |
| simulator (`qwen3.5:4b`) | 8 | 4.3 | 7.7 | 540 | 31 |
| classifier (`qwen3.5:4b`, schema) | 8 | 1.7 | 2.3 | 61 | 16 |

Budget from those means, 8 turns per dialogue, 2 dialogues in parallel: N = 45, K = 3 (270
dialogues) gives about 5.0 h of run on the Air and 7.3 h of judge, one night; N = 60, K = 3 gives
6.6 h and 9.7 h. Those run figures assume both models stay co-resident, which is true on the Air and
was the reason `run` moved there. The judge figures here are the Air's own; since the role swap eval
runs on the Pro, which measured 7.1 h for N = 45, K = 3.

Every judge mean in this document is **per call**, and there are two calls per dialogue, so a
dialogue costs about 97 s of judge time. The budget rows already account for both calls.

## What the budget says about N and K

Formula: `N × 2 agents × K × ~8 turns × (1 agent + 1 simulator + 1 classifier) / parallelism` for
the run, plus `N × 2 × K × 2 calls` of the judge. Measured result, N = 45 and K = 3:

| Where the run happens | Run | Judge | Total machine time |
|---|---|---|---|
| **Air, chosen** (holds both models) | 5.0 h | 7.1 h | 12.1 h |
| Pro with eviction, the option not taken | 12.1 h | 7.1 h | 19.2 h |

Both figures carry the caveat from *Parallelism* above: the run column divides by a parallelism of 2
that the Qwen models only partly deliver, so treat 5.0 h as a floor and re-derive it on the Air with
`--turns 8` before T-17.

**N = 45, K = 3 survives either way**, so the floor declared in TICKETS.md is safe and the corte 1a
and 1b decisions do not have to move.

**N = 60, K = 3** costs 6.6 h of run on the Air against 5.0 h for N = 45, which the chosen
configuration absorbs. It was only the Pro that made this choice tight (16.2 h), so moving `run` to
the Air takes the machine budget out of the N decision and leaves it to the statistics, which is
where the plan wanted it. Still freeze N on Friday with the dataset hash.

K = 5 stays what the plan already says it is: 8.3 h of run on the Air, incremental repetitions, only
if there is machine time left over.
