# Development guide

How code gets written in this repo. Project context, layout, experiment invariants and the hooks that enforce this guide are in [PREAMBLE.md](PREAMBLE.md); the plan lives outside the repo in `~/Documents/mba/projeto/` (`TICKETS.md`, `DECISOES.md`, `PROGRESSO.md`).

The deadline is short and the result has to be *defensible*, not pretty. Every rule below exists because it makes the experiment easier to trust or faster to change. When two rules conflict, pick the one that keeps the experiment honest.

## 1. The Zen of Python, applied (PEP 20)

`python -c "import this"` is the style guide. How it maps here:

- **Explicit is better than implicit.** `num_ctx`, seeds, temperatures and model digests live in `configs/models.yaml` and in the run manifest, never as defaults hidden in code. Every function signature is type-hinted.
- **Errors should never pass silently.** A prompt over 80% of `num_ctx` raises. A malformed scenario is rejected by the validator. No bare `except:`, no `except Exception: pass`, no "try to parse the JSON and fall back". LLM output is constrained by JSON Schema (`format=<schema>`), so there is nothing to parse defensively.
- **Simple is better than complex. Flat is better than nested.** Small pure functions; `argparse`, not a CLI framework; pydantic models and dataclasses, not dicts of dicts; early returns, not deep `if` trees.
- **There should be one obvious way to do it.** One LLM client module, one config loader, one scenario schema, one FSM definition file. If you need the same thing twice, import it.
- **Now is better than never. Although never is often better than *right* now.** Build the thin vertical slice the ticket asks for. Do not implement the next ticket's feature while you are there.
- **Special cases aren't special enough to break the rules.** `baseline` and `fsm` share the same interface, persona, tone, general rules and evaluators. Only the structure of the instruction differs.
- **If the implementation is hard to explain, it's a bad idea.** Every module in `src/sim` should be explainable in one sentence of *Material e Métodos*.

## 2. TDD: red, green, refactor

Tests are written **before** the code they exercise.

1. **Red.** Write the smallest test that fails for the right reason (missing function, wrong value). Run it and watch it fail.
2. **Green.** Write the least code that makes it pass. Hard-coding is allowed at this step.
3. **Refactor.** Remove duplication, name things, keep the tests green. Then commit.

Start a ticket with `/ticket T-xx`. It turns the acceptance criteria into the test list and stops for review before any code is written.

Rules of thumb:

- One behaviour per test, named after the behaviour: `test_run_rejects_unknown_agent`, `test_prompt_over_80_percent_of_num_ctx_raises`. Arrange, act, assert, separated by blank lines.
- Test pure functions with synthetic data: a hand-written dialogue log, a 3-state FSM, a KB with 5 facts. The evaluators of T-13 must be testable with a synthetic JSONL and no model.
- Anything that calls Ollama is an **integration test**: mark it `@pytest.mark.integration`. `just test` skips them; `just test-all` runs them. Keep them few and fast, and assert on structure (schema-valid output), not on exact model text.
- Do not mock the `ollama` package. Mock **our** thin client, the seam T-07 creates. Prefer a fake (a `FakeLLM` returning canned schema-valid replies) over `MagicMock`.
- Fixed seeds wherever randomness appears; tests are deterministic. Use `tmp_path` for files.
- A bug is fixed by first writing the test that reproduces it.
- Negative paths matter as much as happy paths: invalid transitions (T-08), malformed scenarios (T-05), a judge prompt leaking metadata (T-12), a simulator prompt leaking the KB (T-11). The acceptance criteria in TICKETS.md are the test list.

## 3. DRY: one source of truth

Duplication in an experiment is not just ugly. It is how the two agents end up being evaluated differently without anyone noticing.

| Fact | Lives in | Everyone else |
|---|---|---|
| State and event names | `data/fsm/machine.yaml` | FSM engine, stage-labeler `enum`, Mermaid diagram: **load** them |
| KB fact IDs (`F01`, ...) | `data/kb/knowledge_base.md` | scenarios, judge and policy checks reference the IDs |
| Scenario shape | `src/sim/schemas.py` | generator, validator, runner and judge import it |
| Frozen v1 dataset hash | `FROZEN_V1_HASH` in `tests/helpers.py` | DECISOES.md, `docs/taxonomy.md`, `tests/test_dataset.py` |
| Scenario cell plan | `data/scenarios/plan.yaml` | `scripts/generate_scenarios.py`; not loaded by `sim run` |
| Models, digests, `num_ctx`, params | `configs/models.yaml` | LLM client and runner manifest read it |
| Prompts and rubrics | `data/prompts/*.md` | loaded at runtime, never embedded in Python strings |
| Beat contract of a scenario script | `sim.script` | simulated user, `TurnRecord.user_beat`, `scripts/script_adherence.py` |
| Persona, tone, general rules | one shared text block | both agents include the same block |
| Package version | `pyproject.toml` | `sim.__version__` reads it via `importlib.metadata` |
| The quality gate | `just check` | hooks and people run that one target |

In code: shared agent behaviour goes in the base class (T-09), shared test scaffolding in `tests/helpers.py` (plain helpers such as `load_script`) and `tests/conftest.py` (fixtures, once needed), constants are defined once and imported. Three similar lines are fine. A second copy of a *rule* is not.

## 4. Style: PEP 8, PEP 257, PEP 484

`ruff` enforces it. The edit hook formats every `.py` file you touch; `just lint` must be clean before a commit, and the commit hook checks.

- **Formatting:** `ruff format` (black-compatible, 88 columns, double quotes). Imports sorted by ruff; first-party is `sim`.
- **Naming (PEP 8):** `snake_case` functions and variables, `PascalCase` classes, `UPPER_CASE` constants, `_leading_underscore` for internals. No abbreviation that needs a comment.
- **Docstrings (PEP 257, Google style):** every public module, class and function gets one; the first line is a one-sentence summary. Not required for tests or trivial private helpers. Ruff checks the shape of the docstrings that exist.
- **Types (PEP 484):** all signatures annotated. Built-in generics (`list[str]`, `X | None`), `collections.abc` for abstract types, `pathlib.Path` for paths, never `os.path`.
- **Data:** pydantic models for anything that crosses a boundary (files, LLM output, CLI); dataclasses for internal values; dicts only as transient containers.
- **Logging:** `logging` in library code, never `print`. `print` is fine in `scripts/`, in hook scripts and for CLI output.
- **Errors:** specific exceptions with a message that says what to do, e.g. `"prompt uses 7100 of 8192 tokens (> 80%): shrink the KB or raise num_ctx"`.
- **Imports:** absolute (`from sim.schemas import ...`), no wildcards, no import-time side effects.

## 5. Language

The experiment and the repository are **English only** (decision of 2026-09-08): the knowledge base, prompts, rubrics, scenarios, dialogues, identifiers (states, events, schema fields, metric names, CLI flags, file names), code, comments, docs and README. Reasons: small models follow instructions and output schemas more reliably in English, and English text costs fewer tokens. Both matter with a 3B classifier and a 16 GB machine.

The thesis is written in Portuguese, outside this repo. It glosses identifiers when it first uses them ("greeting (saudação)") and its appendices reproduce prompts and dialogues in English. The control documents in `~/Documents/mba/projeto/` (`TICKETS.md`, `DECISOES.md`, `PROGRESSO.md`) stay in Portuguese and still use their original Portuguese artifact names; PREAMBLE.md maps them to the repository names.

Identifiers are ASCII `snake_case`: `scenario`, `greeting`, `data_collection`, `required_facts`, `expected_final_state`, `flow_adherence`, `--scenarios`. The state and event names in `data/fsm/machine.yaml` are the same strings the stage labeler's `enum` and the metrics use: define once, load everywhere.

## 6. Workflow

- One ticket at a time, in the order of *Ordem de execução*. Read its acceptance criteria first.
- Before committing: `just check` (lint plus unit tests). The hooks in `.claude/settings.json` and `.cursor/hooks.json` run ruff on every edited file, run `just check` before a `git commit`, and run it again when a turn ends with uncommitted changes. A red gate is fixed, never bypassed.
- Small commits, imperative English subject, ticket prefix: `T-08: load FSM from machine.yaml with transitions`.
- Any choice that changes the experiment (a model, a parameter, a metric, a cut) gets a dated row in `DECISOES.md` before anything is run with it.
- `runs/` is never committed; result CSVs are regenerated from `runs/`. The dataset in `data/scenarios/v1/` is frozen (T-06; hash in `DECISOES.md` and `FROZEN_V1_HASH`).
- Dependencies: `uv add <pkg>` or `uv add --group dev <pkg>`; commit `pyproject.toml` and `uv.lock` together. The lock is what makes both machines identical, so never edit it by hand.

## 7. Definition of done (per ticket)

- [ ] Every acceptance criterion in TICKETS.md is checked, with a test wherever one makes sense.
- [ ] `just check` is green; integration tests pass on the machine that has Ollama.
- [ ] Decisions logged; the docs the ticket lists are written.
- [ ] Ticket status updated in TICKETS.md; session note in PROGRESSO.md.
