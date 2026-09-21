# fsm-llm-eval: day-to-day recipes. Everything goes through `uv run`, so no venv
# activation is needed. `just` alone lists the recipes.

# List the recipes
default:
    @just --list --unsorted

# Create .venv from uv.lock (dev + analysis groups)
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

# Show the configured judge and `ollama ps`; warns, does not stop models
eval-preflight:
    uv run python scripts/models.py eval-preflight

# Load the judge with the experiment num_ctx, outside the run cache
warmup-judge:
    uv run python scripts/models.py warmup-judge

# Expand plan.yaml into the next unused data/scenarios/vN (refuses a dir that already has JSONL; never overwrite frozen v1)
generate-scenarios *args:
    uv run python scripts/generate_scenarios.py --n 60 --phrasing seed {{args}}

# python -m sim run (T-14a). Example: just run exp data/scenarios/v1 3 2 --resume
run exp_id="exp" scenarios="data/scenarios/examples" reps="1" parallel="2" *args:
    uv run python -m sim run --exp-id {{exp_id}} --scenarios {{scenarios}} --reps {{reps}} --parallel {{parallel}} {{args}}

# python -m sim eval (T-14b). Example: just eval runs/exp_pilot 2
eval run_dir parallel="2" *args:
    uv run python -m sim eval --run {{run_dir}} --parallel {{parallel}} {{args}}

# Sidecar eval: ok logs plus the run-local frozen inclusion artifact
eval-semantic run_dir parallel="2" *args:
    uv run python -m sim eval --run {{run_dir}} --parallel {{parallel}} --include-failed-from {{run_dir}}/adjudication/contract_false_positives.json --out {{run_dir}}_semantic {{args}}

# Discover instrument retry-exhausted failures (not an inclusion verdict)
instrument-failures run_dir *args:
    uv run python scripts/list_instrument_failures.py {{run_dir}} {{args}}

# Failed-dialogue audit: generated evidence only; never writes the inclusion JSON
audit-failed run_dir *args:
    uv run python scripts/audit_failed_dialogues.py --run {{run_dir}} {{args}}

# T-18: census exp_final and byte-copy scored CSVs into results/<exp_id>/
export-metrics run_dir="runs/exp_final" sidecar="runs/exp_final_semantic":
    uv run python scripts/export_audited_metrics.py --run {{run_dir}} --sidecar {{sidecar}}

# T-19: scenario-level paired statistics into results/<exp_id>/
analyze metrics="results/exp_final/metrics.csv" frozen="results/exp_final/metrics_frozen_gate.csv" out="results/exp_final" *args:
    uv run --group analysis python scripts/analyze.py --metrics {{metrics}} --frozen {{frozen}} --out {{out}} {{args}}

# T-21: category-direction table and exemplar shortlist (does not overwrite cases.md)
cases tests="results/exp_final/tests.csv" metrics="results/exp_final/metrics.csv" out="results/exp_final":
    uv run python scripts/cases.py --tests {{tests}} --metrics {{metrics}} --out {{out}}

# T-22: paste-ready appendices A–D into docs/appendices
appendices out="docs/appendices":
    uv run python scripts/appendices.py --out {{out}}

# T-24: copy the delivery snapshot (default ~/Documents/mba/entregas/simulacao_v1)
deliver *args:
    uv run python scripts/export_delivery.py {{args}}

# T-20: thesis tables and figures from frozen T-19 CSVs
figures results="results/exp_final" *args:
    uv run --group analysis python scripts/report.py --results-dir {{results}} {{args}}

# Post-hoc: split the frozen fact_f1 effect into its counts (replays judge output, never re-judges)
decompose run_dir="runs/exp_final" metrics="results/exp_final/metrics.csv" out="results/exp_final":
    uv run python scripts/decompose_fact_f1.py --run {{run_dir}} --metrics {{metrics}} --out {{out}}

# Blind human-evaluation sidecar for the frozen semantic_primary dialogues
human-validation-packet run_dir="runs/exp_final" metrics="results/exp_final/metrics.csv" out="results/human_validation/exp_final" *args:
    uv run python scripts/export_human_validation.py --run {{run_dir}} --metrics {{metrics}} --out {{out}} {{args}}

# Score frozen human sheets, compare to the judge, rerun confirmatory tests
score-human-validation packet="results/human_validation/exp_final" out="results/human_primary" *args:
    uv run --group analysis python scripts/score_human_validation.py --packet {{packet}} --out {{out}} {{args}}

# Materialize an explicit human selection of contract false positives
freeze-contract-false-positives run_dir *ids:
    uv run python scripts/freeze_contract_false_positives.py {{run_dir}} {{ids}}

# Per-caller token and latency averages plus seconds per dialogue (T-15)
stats run_dir:
    uv run python scripts/run_stats.py {{run_dir}}

# Fail unless every dialogue of a run delivered its script beats in order (T-11)
adherence run_dir="runs/exp_pilot":
    uv run python scripts/script_adherence.py --run {{run_dir}}

# Draw the stratified judge-validation sample into results/judge_validation (T-16)
judge-sample run_dir *args:
    uv run python scripts/judge_validation_sample.py --run {{run_dir}} {{args}}

# Compute T-16 agreement from frozen sheets and one run directory
# Default: Pilot v1. Pilot v2: just validate-judge runs/pilot_v2 results/judge_validation/pilot_v2
validate-judge run_dir="runs/exp_pilot" sample_dir="results/judge_validation/pilot_v1":
    uv run python scripts/validate_judge.py agree --run {{run_dir}} --sample {{sample_dir}}

# Copy dialogues into a sidecar for re-eval (never writes runs/pilot_v2)
prepare-eval-sidecar source="runs/pilot_v2" target="runs/pilot_v2_mlx":
    uv run python scripts/prepare_eval_sidecar.py --source {{source}} --target {{target}}

# Compare published Pilot v2 agree() against a gemma4:12b-mlx sidecar
validate-judge-compare original="runs/pilot_v2" mlx="runs/pilot_v2_mlx" sample="results/judge_validation/pilot_v2" out="results/judge_validation/pilot_v2_mlx":
    uv run python scripts/validate_judge.py compare --original {{original}} --mlx {{mlx}} --sample {{sample}} --out {{out}}

# Measure stage-labeler accuracy and flow-path agreement on one run
validate-labeler run_dir="runs/exp_pilot":
    uv run python scripts/validate_judge.py labeler --run {{run_dir}}

# Relabel one frozen run into a sidecar with stage-labeler only
relabel-stages source_run="runs/pilot_v2" target_run="runs/pilot_v2_labeler_v2":
    uv run python scripts/relabel_stages.py --source {{source_run}} --target {{target_run}}

# Relabel Pilot v2 with qwen3.5:9b (sidecar; never writes runs/pilot_v2)
relabel-stages-9b source_run="runs/pilot_v2" target_run="runs/pilot_v2_labeler_9b":
    uv run python scripts/relabel_stages.py --source {{source_run}} --target {{target_run}} --model qwen3.5:9b

# Compare two stage-labeler metrics_turn.csv files against FSM gold
validate-labeler-compare original="runs/pilot_v2" sidecar="runs/pilot_v2_labeler_9b" out="results/judge_validation/pilot_v2_qwen9b":
    uv run python scripts/validate_judge.py labeler-compare --original {{original}} --sidecar {{sidecar}} --out {{out}}

# Replay the event classifier on frozen Pilot v2 FSM turns with qwen3.5:9b
replay-classifier-pilot source="runs/pilot_v2" target="runs/pilot_v2_classifier_9b":
    uv run python scripts/replay_classifier.py pilot --source {{source}} --target {{target}} --model qwen3.5:9b

# Score T-08 gold phrases with one configured model (4B or 9B)
replay-classifier-fixtures model out="results/judge_validation/pilot_v2_qwen9b":
    uv run python scripts/replay_classifier.py fixtures --model {{model}} --out {{out}}

# Remove caches and coverage output
clean:
    rm -rf .pytest_cache .ruff_cache .coverage htmlcov
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
