# fsm-llm-eval

Simulation and evaluation harness comparing two LLM customer-service agents: a **baseline**
(one well-written system prompt with the full knowledge base) and an **FSM** agent (a finite-state
machine as the instruction base: on every turn, the instruction package of the current state plus
the same full knowledge base the baseline carries). Same local model (Ollama), same simulated user, same scenarios; evaluation
by a blind LLM judge and by deterministic evaluators; paired statistics per scenario.

Experiment for an MBA thesis (TCC, written in Portuguese): *Sistema de Simulações e Avaliação para
Otimizar o Desempenho de LLMs usando Máquinas de Estados Finitas como Base de Instrução*. The work
plan, the ticket board (T-01 to T-24), the decisions table and the session log live in
`~/Documents/mba/projeto/` (`TICKETS.md`, `DECISOES.md`, `PROGRESSO.md`), outside this repository. The experiment, this repository and
everything the models read or write are in English; the thesis glosses names in Portuguese.

> Status: `sim run` / `sim eval` in place; v1 dataset frozen. Pilot v1 (T-15/T-16)
> is historical: logical path `runs/exp_pilot` (manifest
> `exp_pilot_fixes_20260913_final_v1`); sheets locked under
> `results/judge_validation/pilot_v1/` (do not redraw). The 11/09 sample is
> void. Full-scenario pre-flight (2026-09-14, `runs/full_scenario_preflight/`)
> passed as QA: 116/120 ok ([`docs/pilot.md`](docs/pilot.md)). Pilot v2
> (`runs/pilot_v2/`) is closed as instrument validation: 20/20 ok,
> `just adherence` 20/20, 4/4 canary, 4/4 injection; eval written; sheets
> locked under `results/judge_validation/pilot_v2/` (do not redraw). Agreement
> and disagreements are in [`docs/judge_validation.md`](docs/judge_validation.md)
> *Pilot v2 validation round*. Stage-labeler revision 2 was tested in a sidecar
> (`runs/pilot_v2_labeler_v2/`) as a post-hoc development check and rejected:
> revision 1 is kept (`docs/judge_validation.md`). Pilot v1 is not pooled with
> Pilot v2. **T-16 CLOSED — ready to freeze final configuration.** Do not start
> `runs/exp_final/` in this step. Metrics:
> [`docs/metrics.md`](docs/metrics.md).

## Requirements

- macOS on Apple Silicon. `run` phase and writing on the MacBook Air M4 24 GB; `eval` phase (judge)
  on the MacBook Pro M5 16 GB. The `run` phase needs the agent and the simulator resident at once,
  which 16 GB does not manage; `docs/setup.md` has the measurements.
- [uv](https://docs.astral.sh/uv/) for Python and dependencies (`brew install uv`).
- [just](https://just.systems) for the day-to-day recipes (`brew install just`).
- Python 3.13 (`.python-version`); uv installs it if missing.
- [Ollama](https://ollama.com) with the models in `configs/models.yaml`, **same version and same
  digests on both machines**.

## Installation

```bash
git clone git@github.com:hugoantunes/fsm-llm-eval.git ~/projects/fsm-llm-eval
cd ~/projects/fsm-llm-eval
just install   # uv sync: creates .venv from uv.lock
just test      # must pass before anything else
```

The repository lives in `~/projects/fsm-llm-eval` on both machines, **outside iCloud** (iCloud
evicts files and corrupts a `.git` synced between machines). Code travels through GitHub; `runs/`
travels by `rsync` over SSH on the local network.

### Ollama

Environment variables (agent and simulator stay loaded together; two dialogues in parallel; default
`num_ctx` of 8192). The macOS app inherits its environment from `launchd`, so `export` in a shell
has no effect: the script applies the variables with `launchctl setenv`, restarts the app and
verifies what the server loaded. It does not survive a reboot; run it again after each restart.

```bash
scripts/ollama_env.sh
```

Pull the models of `configs/models.yaml` (agent, simulator/classifier, judge) on both machines and
check that the digests match the ones recorded there:

```bash
ollama --version     # same on both machines; recorded in configs/models.yaml
just pull-models     # ollama pull for every configured model
just digests         # role, name and local digest, to paste into configs/models.yaml
just verify-models   # fails unless the local digests match the recorded ones
```

`num_ctx` is fixed explicitly (8192): Ollama's default silently truncates the beginning of a prompt
that exceeds it, which would cut the baseline's knowledge base. Values and rationale in
[docs/setup.md](docs/setup.md).

## How to run

After `just install`, `scripts/ollama_env.sh` (again after every reboot) and `just verify-models`.

The recipe is `just run [exp_id] [scenarios] [reps] [parallel] [args...]`. Defaults: `exp`,
`data/scenarios/examples`, 1 repetition, `--parallel 2`. Extra args go to `python -m sim run`.
Omit `--agent` to run both agents (baseline then FSM), which is what the experiment needs.

```bash
uv run python -m sim --help
uv run python -m sim run --help

# Smoke: 3 example scenarios x 2 agents x 1 rep → runs/exp/
just run exp

# Experiment dataset (T-06, frozen): N=60
just run exp data/scenarios/v1 3 2

# One agent only (debug or resume a failed side)
just run exp --agent fsm

# More repetitions, then resume after a stop (skips JSONL already written)
just run exp data/scenarios/examples 3 2
just run exp data/scenarios/examples 3 2 --resume
```

After a run, check that every dialogue delivered its script beats, then score it
without re-playing the dialogues. Judge and labeler calls go through the same
prompt-hash cache, so a second eval of the same directory is free when the
rubrics have not changed. The T-16 sheets under
`results/judge_validation/pilot_v1/` are frozen (census of all 20 ok
dialogues; blinded sample of 30/76 eligible agent responses). Pilot v2 sheets
under `results/judge_validation/pilot_v2/` are frozen (census of 20; blinded
sample of 30/78). Do not redraw either. Draw a sample only from a run that
passed `just adherence`, never from the 11/09 instrument:

```bash
just adherence runs/<exp_id>
just eval runs/<exp_id>
# uv run python -m sim eval --run runs/<exp_id>
# just judge-sample runs/<exp_id>   # frozen for this pilot; do not rerun
```
```

Equivalent without `just`:

```bash
uv run python -m sim run \
  --exp-id exp \
  --scenarios data/scenarios/examples \
  --reps 1 \
  --parallel 2
```

`--exp-id` is required. Output lands in `runs/<exp_id>/` (the `runs/` directory is in the
repo, empty until the first run; change the parent with `--runs-dir`):

| Path | What it is |
|---|---|
| `dialogues/{scenario}__{agent}__repNN.jsonl` | one dialogue: turns, stop/termination metadata, structured failure classification (`failure_kind`, `failure_reason`, `failure_metadata`), beat completion flag (`active_beat_complete`), goal-reached provenance (`goal_reached_seen`, `goal_reached_at_turn`, `script_complete_at_goal_reached`), per-turn beat delivery (`user_beat`) plus beat span provenance (`beat_started_at_turn`, `beat_completed_at_turn`), and the FSM states/user event/edges per turn (empty on the baseline) |
| `manifest.json` | config, dataset hash, FSM hash, model digests, `num_ctx`, prompt versions, job list, throughput, LLM call / cache-hit counts |
| `llm_calls.jsonl`, `cache/` | every LLM call (`baseline`, `fsm`, `simulated_user`, `classifier`, then `judge_facts`, `judge_global`, `stage_labeler` after eval), keyed by prompt hash. Compact JSON: `"cached":true` has no space after the colon |
| `metrics.csv` | one row per ok dialogue: identity columns plus every metric of T-04 |
| `metrics_turn.csv` | one row per agent turn: labelled stage and, on the FSM side, true `state_after` |

The schedule is repetition → scenario → agent, so after repetition K the paired dataset is
complete and extra reps are incremental. `--resume` skips a job iff its final JSONL exists; a
leftover `.tmp` is not complete and that job runs again. A failed dialogue is recorded
(`status=failed`) and does not abort the rest; delete its file to retry. Runtime failures are
classified explicitly (for example `invalid_candidate_retry_exhausted` as instrument failure, and
`max_turns_with_incomplete_beat` as failed simulation). The process exits 1 if any dialogue failed.
`--parallel` is a thread pool of dialogues, matched to `OLLAMA_NUM_PARALLEL`.

`sim eval` reads a `runs/<exp_id>/` and writes `metrics.csv` and `metrics_turn.csv`
there, without re-executing the dialogues. Failed logs are skipped. Dialogues are
shuffled before the judge (seed from `configs/models.yaml`). CSV rows follow the
manifest's job list, then any other ok logs on disk (a later `--agent fsm --resume`
must not drop the baseline half). Re-eval of a growing directory after `rsync` is
free via the cache: it does not need named blocks. The scenario files must still hash
to `manifest.dataset_hash`. The FSM files must still hash to `manifest.fsm_hash`.

Full execution (T-17), on the Air: `caffeinate -is uv run python -m sim run ... --parallel 2`, in
blocks with `--resume`; `just adherence` on the block before it is evaluated. The Pro pulls `runs/`
over SSH (`docs/setup.md`) and runs `just eval`. Splitting the phases across the two machines
saves about 4 h; running both on the Air in sequence works too and costs only wall clock.

## Development

```bash
just                    # list the recipes
just test               # pytest, without the integration tests
just test-all           # includes the tests marked `integration` (they need Ollama)
just lint               # ruff check + ruff format --check
just format             # ruff format + ruff check --fix
just check              # lint + test: the gate the hooks run
just generate-scenarios # expand plan.yaml into the next unused vN (never overwrite frozen v1)
just fsm-diagram        # Mermaid diagram of data/fsm/machine.yaml, for docs/fsm.md
just stats <run>        # per-caller tokens and latency (T-15)
just adherence [run]    # fail unless every dialogue delivered its script beats
just judge-sample <run> # draw the T-16 sample into results/judge_validation
just install-analysis   # pandas, scipy, matplotlib, jupyter (T-19, T-20)
```

Code conventions (TDD, DRY, PEP 8/20/257, naming) in
[ai-assistance/DEVELOPMENT.md](ai-assistance/DEVELOPMENT.md); project context and rules for AI
assistants in [ai-assistance/PREAMBLE.md](ai-assistance/PREAMBLE.md). The root `CLAUDE.md` only
points there. Hooks in `.claude/settings.json` run `ruff` on every edited file and `just check`
before any commit.

## Layout

Parts that do not exist yet are marked with the ticket that creates them.

| Path | What it is |
|---|---|
| `src/sim/` | Python package (`python -m sim`): LLM client, FSM engine, agents, simulated user (ordered script), judge, evaluators, runner, eval (T-07 to T-14) |
| `tests/` | pytest; anything that talks to Ollama is marked `integration` |
| `data/kb/` | knowledge base: numbered facts (F01...), needles, unanswerable questions (T-01) |
| `data/fsm/` | `machine.yaml` (states, events, transitions, guards) and `states/*.md` (instruction package per state) (T-02) |
| `data/prompts/` | versioned prompts: baseline, FSM template, simulated user, event classifier, stage labeler, judge shared/facts/global; `scenario_phrasing.md` (T-06, unused on the frozen `--phrasing seed` set) |
| `data/scenarios/` | golden dataset: `examples/` (T-05), `plan.yaml` (authoring source), and frozen `v1/` (T-06). Do not edit `v1/` |
| `configs/` | `models.yaml`: models, digests, `num_ctx`, fixed parameters |
| `runs/` | output of `sim run` / `sim eval`: dialogues, manifest, LLM cache, `metrics.csv`, `metrics_turn.csv`. Contents git-ignored; moved between machines by `rsync` |
| `results/` | `metrics.csv` (T-18, audited copy), `descriptive.csv`, `tests.csv`, `tables/`, `figures/` (T-19, T-20; CSVs git-ignored, regenerated from `runs/`); `judge_validation/pilot_v1/` and `judge_validation/pilot_v2/` (hand annotations are primary data) |
| `notebooks/` | `analysis.ipynb`: regenerates tables and figures from `metrics.csv` (T-20) |
| `scripts/` | `ollama_env.sh`, `models.py`, `measure_latency.py`, `run_stats.py`, `script_adherence.py` (T-11), `judge_validation_sample.py` (T-16), `generate_scenarios.py` (T-06; default out is the next unused `vN`, never overwrite `v1`) |
| `docs/` | `setup.md`, `metrics.md`, `taxonomy.md`, `fsm.md`, `decisions_and_limitations.md`, `pilot.md`, `judge_validation.md`, `parity.md`; later the appendices (T-16 to T-22) |
| `ai-assistance/` | instructions for AI assistants (`PREAMBLE.md`, `DEVELOPMENT.md`) and the hook scripts |
| `.claude/` | Claude Code config: hooks and permissions (`settings.json`), the `/ticket` skill, path-scoped rules |

## License

MIT. Hugo Seixas Antunes.
