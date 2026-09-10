# fsm-llm-eval: day-to-day recipes. Everything goes through `uv run`, so no venv
# activation is needed. `just` alone lists the recipes.

# List the recipes
default:
    @just --list --unsorted

# Create .venv from uv.lock (dev group included)
install:
    uv sync

# Add pandas, scipy, matplotlib and jupyter (T-19, T-20)
install-analysis:
    uv sync --group analysis

# Unit tests; skips tests marked `integration`. Extra args go to pytest
test *args:
    uv run pytest -m "not integration" {{args}}

# Every test, including the ones that need Ollama
test-all *args:
    uv run pytest {{args}}

# Unit tests with a coverage report
cov:
    uv run pytest -m "not integration" --cov --cov-report=term-missing

# ruff check + ruff format --check
lint:
    uv run ruff check .
    uv run ruff format --check .

# ruff format + ruff check --fix
format:
    uv run ruff format .
    uv run ruff check --fix .

# The quality gate the hooks run: lint, then unit tests
check: lint test

# Print the Mermaid diagram of data/fsm/machine.yaml; paste into docs/fsm.md
fsm-diagram:
    uv run python -c 'from sim.fsm import load_fsm; fsm = load_fsm(); print(fsm.to_mermaid())'

# Pull the models of configs/models.yaml (agent, simulator, judge)
pull-models:
    uv run python scripts/models.py pull

# Print role, name and local digest of the configured models
digests:
    uv run python scripts/models.py digests

# Fail unless the local Ollama holds exactly the digests recorded in configs/models.yaml
verify-models:
    uv run python scripts/models.py verify

# Expand plan.yaml into the next unused data/scenarios/vN (refuses a dir that already has JSONL; never overwrite frozen v1)
generate-scenarios *args:
    uv run python scripts/generate_scenarios.py --n 60 --phrasing seed {{args}}

# python -m sim run (T-14a). Example: just run exp data/scenarios/v1 3 2 --resume
run exp_id="exp" scenarios="data/scenarios/examples" reps="1" parallel="2" *args:
    uv run python -m sim run --exp-id {{exp_id}} --scenarios {{scenarios}} --reps {{reps}} --parallel {{parallel}} {{args}}

# python -m sim eval (T-14b). Example: just eval runs/exp_pilot
eval run_dir:
    uv run python -m sim eval --run {{run_dir}}

# Remove caches and coverage output
clean:
    rm -rf .pytest_cache .ruff_cache .coverage htmlcov
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
