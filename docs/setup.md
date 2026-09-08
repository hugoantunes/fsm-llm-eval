# Setup: machines and Ollama (T-03)

Record of the execution environment. Each item becomes a sentence in *Material e Métodos* or a row
in the Decisões table of `TICKETS.md`.

## Machines

| Role | Machine | Chip | RAM | macOS | Ollama | Checked on |
|---|---|---|---|---|---|---|
| `eval` (judge) and writing | MacBook Air (Mac16,12) | Apple M4, 10 cores (4P + 6E) | 24 GB | 26.6.2 | 0.33.3 (app) | 2026-09-08 |
| `run` (agents + simulator) | MacBook Pro M5 | to fill in | 16 GB | to fill in | to fill in | to fill in |

The repository lives in `~/projects/fsm-llm-eval` on both machines, outside iCloud. `runs/` moves
from one to the other by `rsync` over SSH on the local network.

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
| `OLLAMA_MAX_LOADED_MODELS` | 2 | agent and simulator stay loaded together; otherwise Ollama unloads and reloads on every turn and latency triples |
| `OLLAMA_KEEP_ALIVE` | -1 | never unload a model for inactivity |
| `OLLAMA_NUM_PARALLEL` | 2 | the runner runs 2 dialogues at once (`sim run --parallel 2`); generation is memory-bandwidth bound and two streams use the GPU better |
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

If memory gets tight on the Pro (16 GB) during `run`, `OLLAMA_FLASH_ATTENTION=1` with
`OLLAMA_KV_CACHE_TYPE=q8_0` halves the KV cache. Only with a Decisões row, and on both machines.

| Variables check | Air | Pro |
|---|---|---|
| `scripts/ollama_env.sh` ran without error on | 2026-09-08 (Ollama 0.33.3; the server loaded all 4 values) | to fill in |

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

| Role | Model | ID (Air, 2026-09-08) | ID (Pro) | Same? |
|---|---|---|---|---|
| agents (baseline and fsm) | `qwen3.5:9b` | `6488c96fa5fa` | to fill in | |
| simulator, event classifier, stage labeler | `qwen3.5:4b` | `2a654d98e6fb` | to fill in | |
| judge | `gemma4:12b` | `4eb23ef187e2` | to fill in | |

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

## Latency and budget

Measured on the Pro with `scripts/measure_latency.py`: P parallel dialogues of T turns, each turn
one simulator call (persona + history), one schema-constrained classifier call and one agent call
(a 2.5k-token KB-sized system prompt + history); `--judge` adds two judge-sized calls (3.5k-token
prompt, structured verdict). It prints mean and p95 per role with token counts, then the budget for
N in {45, 60} and K in {3, 5}, assuming 8 turns per dialogue (`--budget-turns`):

```bash
scripts/ollama_env.sh
uv run python scripts/measure_latency.py --parallel 2 --turns 8 --judge
```

Smoke-tested on the Air on 2026-09-08 (2 turns, 2 dialogues, Qwen2.5 + Gemma 3) while four model
downloads were running: it works end to end; those numbers are not recorded because of the
contention. The table below is filled from the Pro.

| Call | Mean (s) | p95 (s) | Prompt tokens |
|---|---|---|---|
| agent | to fill in | | |
| simulator | to fill in | | |
| classifier | to fill in | | |

First run with the final models on the Air, 2026-09-08 (`--parallel 2 --turns 4 --judge`, no
downloads running). The judge row is the eval machine's own number; the other rows are
indicative, the Pro is the run machine:

| Call | n | Mean (s) | p95 (s) | Prompt tokens | Output tokens |
|---|---|---|---|---|---|
| judge (`gemma4:12b`, two calls) | 2 | 48.6 | 72.0 | 4338 | 248 |
| agent (`qwen3.5:9b`) | 8 | 10.6 | 25.7 | 3232 | 44 |
| simulator (`qwen3.5:4b`) | 8 | 4.3 | 7.7 | 540 | 31 |
| classifier (`qwen3.5:4b`, schema) | 8 | 1.7 | 2.3 | 61 | 16 |

Budget from those means, 8 turns per dialogue, 2 dialogues in parallel: N = 45, K = 3 (270
dialogues) gives about 5.0 h of run on the Air and 7.3 h of judge, one night; N = 60, K = 3 gives
6.6 h and 9.7 h. The run figure will be replaced by the Pro's.

Budget: `N × 2 agents × K × ~8 turns × (1 agent + 1 simulator + 1 classifier) / parallelism` on the
Pro, plus `N × 2 × K × 2 calls` of the judge on the Air. Result: to fill in. This number decides N
and K (floor N = 45, K = 3).
