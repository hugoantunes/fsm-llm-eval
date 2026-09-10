# fsm-llm-eval

Simulation and evaluation harness comparing two LLM customer-service agents: a **baseline**
(one well-written system prompt with the full knowledge base) and an **FSM** agent (a finite-state
machine as the instruction base: on every turn, only the instruction package and the facts released
for the current state). Same local model (Ollama), same simulated user, same scenarios; evaluation
by a blind LLM judge and by deterministic evaluators; paired statistics per scenario.

Experiment for an MBA thesis (TCC, written in Portuguese): *Sistema de Simulações e Avaliação para
Otimizar o Desempenho de LLMs usando Máquinas de Estados Finitas como Base de Instrução*. The work
plan, the ticket board (T-01 to T-24), the decisions table and the session log live in
`~/Documents/mba/projeto/` (`TICKETS.md`, `DECISOES.md`, `PROGRESSO.md`), outside this repository. The experiment, this repository and
everything the models read or write are in English; the thesis glosses names in Portuguese.

> Status: `sim run` (T-14a) is in place. The blind two-call judge (T-12) is in place. Metrics and rubrics: T-04, `docs/metrics.md`. `sim eval` is T-14b and is not implemented yet.

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

# One agent only (debug or resume a failed side)
just run exp --agent fsm

# More repetitions, then resume after a stop (skips JSONL already written)
just run exp data/scenarios/examples 3 2
just run exp data/scenarios/examples 3 2 --resume
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
| `dialogues/{scenario}__{agent}__repNN.jsonl` | one dialogue: turns, stop reason, and per turn the FSM states, the user event and every edge the turn walked (empty on the baseline) |
| `manifest.json` | config, dataset hash, model digests, `num_ctx`, prompt versions, job list, throughput, LLM call / cache-hit counts |
| `llm_calls.jsonl`, `cache/` | every LLM call (`baseline`, `fsm`, `simulated_user`, `classifier`), keyed by prompt hash. Compact JSON: `"cached":true` has no space after the colon |

The schedule is repetition → scenario → agent, so after repetition K the paired dataset is
complete and extra reps are incremental. `--resume` skips a job iff its final JSONL exists; a
leftover `.tmp` is not complete and that job runs again. A failed dialogue is recorded
(`status=failed`) and does not abort the rest; delete its file to retry. The process exits 1 if
any dialogue failed. `--parallel` is a thread pool of dialogues, matched to `OLLAMA_NUM_PARALLEL`.

`sim eval` (T-14b) will read a `runs/<exp_id>/` and write `metrics.csv` without re-executing; it is
not implemented yet.

Full execution (T-17), on the Air: `caffeinate -is uv run python -m sim run ... --parallel 2`, in
blocks with `--resume`; after each block, `rsync -av runs/ <pro>:~/projects/fsm-llm-eval/runs/` and
`just eval` on the Pro. Splitting the phases across the two machines saves about 4 h; running both
on the Air in sequence works too and costs only wall clock.

## Development

```bash
just                    # list the recipes
just test               # pytest, without the integration tests
just test-all           # includes the tests marked `integration` (they need Ollama)
just lint               # ruff check + ruff format --check
just format             # ruff format + ruff check --fix
just check              # lint + test: the gate the hooks run
just fsm-diagram        # Mermaid diagram of data/fsm/machine.yaml, for docs/fsm.md
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
| `src/sim/` | Python package (`python -m sim`): LLM client, FSM engine, agents, simulated user, judge, evaluators, runner (T-07 to T-14) |
| `tests/` | pytest; anything that talks to Ollama is marked `integration` |
| `data/kb/` | knowledge base: numbered facts (F01...), needles, unanswerable questions (T-01) |
| `data/fsm/` | `machine.yaml` (states, events, transitions, guards) and `states/*.md` (instruction package per state) (T-02) |
| `data/prompts/` | versioned prompts: baseline, FSM template, judge shared/facts/global (T-04, T-09, T-10, T-12) |
| `data/scenarios/` | golden dataset: `examples/` (T-05) and `v1/`, frozen by hash (T-06) |
| `configs/` | `models.yaml`: models, digests, `num_ctx`, fixed parameters |
| `runs/` | output of `sim run` and the LLM call cache. Contents git-ignored; moved between machines by `rsync` |
| `results/` | `metrics.csv`, `descriptive.csv`, `tests.csv`, `tables/`, `figures/` (T-14b, T-19, T-20; CSVs git-ignored, regenerated from `runs/`) |
| `notebooks/` | `analysis.ipynb`: regenerates tables and figures from `metrics.csv` (T-20) |
| `scripts/` | `ollama_env.sh`, `models.py`, `measure_latency.py`; `generate_scenarios.py` (T-06) |
| `docs/` | `setup.md`; `metrics.md`, `taxonomy.md`, `fsm.md`, `pilot.md`, `parity.md`, `judge_validation.md` and the appendices (T-02 to T-22) |
| `ai-assistance/` | instructions for AI assistants (`PREAMBLE.md`, `DEVELOPMENT.md`) and the hook scripts |
| `.claude/` | Claude Code config: hooks and permissions (`settings.json`), the `/ticket` skill, path-scoped rules |

## License

MIT. Hugo Seixas Antunes.
