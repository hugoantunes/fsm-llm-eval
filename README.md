# fsm-llm-eval

Simulation and evaluation harness comparing two LLM customer-service agents: a **baseline**
(one well-written system prompt with the full knowledge base) and an **FSM** agent (a finite-state
machine as the instruction base: on every turn, only the instruction package and the facts released
for the current state). Same local model (Ollama), same simulated user, same scenarios; evaluation
by a blind LLM judge and by deterministic evaluators; paired statistics per scenario.

Experiment for an MBA thesis (TCC, written in Portuguese): *Sistema de Simulações e Avaliação para
Otimizar o Desempenho de LLMs usando Máquinas de Estados Finitas como Base de Instrução*. The work
plan, the ticket board (T-01 to T-24) and the decisions table live in
`~/Documents/mba/projeto/TICKETS.md`, outside this repository. The experiment, this repository and
everything the models read or write are in English; the thesis glosses names in Portuguese.

> Status: initial structure (T-03). `sim run` and `sim eval` are implemented in T-14a and T-14b.

## Requirements

- macOS on Apple Silicon. `run` phase on the MacBook Pro M5 16 GB; `eval` phase (judge) and
  writing on the MacBook Air M4 24 GB.
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

## Usage

```bash
uv run python -m sim --help
just run baseline data/scenarios/examples 1 2   # sim run --agent ... --scenarios ... --reps ... --parallel ...
just eval runs/<exp_id>                         # sim eval --run ...
```

- `sim run` writes one JSONL per dialogue to `runs/<exp_id>/` plus a `manifest.json` (config,
  dataset hash, model digests, `num_ctx`, prompt and Ollama versions). The loop order is
  repetition → scenario → agent; `--resume` skips what already ran.
- `sim eval` reads a `runs/<exp_id>/`, runs the deterministic evaluators and the judge's two calls,
  and writes `metrics.csv` and `metrics_turn.csv`. It is separate from `run` so a run can be
  re-evaluated without re-executing (LLM calls are cached by prompt hash).

Full execution (T-17), on the Pro: `caffeinate -is uv run python -m sim run ... --parallel 2`, in
blocks with `--resume`; after each block, `rsync -av runs/ <air>:~/projects/fsm-llm-eval/runs/` and
`just eval` on the Air.

## Development

```bash
just                    # list the recipes
just test               # pytest, without the integration tests
just test-all           # includes the tests marked `integration` (they need Ollama)
just lint               # ruff check + ruff format --check
just format             # ruff format + ruff check --fix
just check              # lint + test: the gate the hooks run
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
| `data/prompts/` | versioned prompts: baseline, FSM template, judge (facts and global) (T-04, T-09, T-10) |
| `data/scenarios/` | golden dataset: `examples/` (T-05) and `v1/`, frozen by hash (T-06) |
| `configs/` | `models.yaml`: models, digests, `num_ctx`, fixed parameters |
| `runs/` | output of `sim run` and the LLM call cache. **Git-ignored**; moved between machines by `rsync` |
| `results/` | `metrics.csv`, `descriptive.csv`, `tests.csv`, `tables/`, `figures/` (T-14b, T-19, T-20; CSVs git-ignored, regenerated from `runs/`) |
| `notebooks/` | `analysis.ipynb`: regenerates tables and figures from `metrics.csv` (T-20) |
| `scripts/` | `ollama_env.sh`, `models.py`, `measure_latency.py`; `generate_scenarios.py` (T-06) |
| `docs/` | `setup.md`; `metrics.md`, `taxonomy.md`, `fsm.md`, `pilot.md`, `parity.md`, `judge_validation.md` and the appendices (T-02 to T-22) |
| `ai-assistance/` | instructions for AI assistants (`PREAMBLE.md`, `DEVELOPMENT.md`) and the hook scripts |
| `.claude/` | Claude Code config: hooks and permissions (`settings.json`), the `/ticket` skill, path-scoped rules |

## License

MIT. Hugo Seixas Antunes.
