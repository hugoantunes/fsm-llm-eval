# AI assistant instructions: fsm-llm-eval

Instructions for any AI assistant working in this repository. Claude Code imports this file from the root `CLAUDE.md`; point other tools here the same way.

## Project

MBA thesis (TCC) experiment. Question: does a finite-state machine used as the *instruction base* of an LLM customer-service agent beat one well-written system prompt? Two agents, `baseline` (one prompt, full knowledge base) and `fsm` (per-state instruction package, same full knowledge base), same local model via Ollama, same simulated user, same scenarios. Evaluation: blind LLM judge (two calls per dialogue) plus deterministic evaluators. Statistics paired by scenario.

The experiment, this repository and everything the models read or write are **English** (decision of 2026-09-08: small models follow instructions and schemas more reliably in English, and English costs fewer tokens). The thesis is written in Portuguese, outside the repo, and glosses names on first use. The control documents live in `~/Documents/mba/projeto/`, in Portuguese and with the original Portuguese artifact names (map them with the table under *Layout*):

- `TICKETS.md`: plan, tickets T-01 to T-24, acceptance criteria
- `DECISOES.md`: dated **Decisões** table
- `PROGRESSO.md`: session log

That folder is on iCloud: if a file is unreadable it is probably evicted, so ask instead of guessing. Deadlines: tag `v1` by 2026-09-11, code frozen 2026-09-14, thesis due 2026-09-22.

## Sections

- @DEVELOPMENT.md: how code gets written here (Zen of Python, TDD, DRY, PEP 8/257/484, language, workflow, definition of done). Imported, so it is always loaded.
- `.claude/skills/ticket/SKILL.md`: `/ticket T-07` works one ticket test-first, with a review stop after the plan and before any code. Cursor discovers this skill on its own.
- `.claude/rules/prompts.md`: judge-blindness and prompt-parity rules, loaded when a file under `data/prompts/` or `src/sim/` is read.
- `.cursor/`: what Claude Code gets from `CLAUDE.md` and `.claude/`, for Cursor, which reads neither the `@` imports nor `.claude/rules/`. `rules/project.mdc` sends the assistant to these two files at the start of a session, `rules/prompts.mdc` points at the rule above, and `hooks.json` wires the same scripts (see *Enforcement*). It adds no instruction of its own: a second copy of a rule is how the two agents end up judged differently.
- `ai-assistance/scripts/`: the hook scripts, wired in both `.claude/settings.json` and `.cursor/hooks.json` (see *Enforcement*).

## Commands

- `just` lists the recipes. `just install`: `uv sync` (Python 3.13, dev group). `just install-analysis` adds pandas, scipy, matplotlib, jupyter.
- `just test`: pytest without `integration` tests. `just test-all` includes them (needs Ollama with the models in `configs/models.yaml`). Extra arguments go to pytest: `just test tests/test_cli.py -k help`.
- `just lint` / `just format`: ruff check and ruff format. `just check` = lint + test; it is the gate the hooks run.
- `just run [exp_id] [scenarios] [reps] [parallel] [args...]` and `just eval runs/<exp_id>`: thin wrappers over `python -m sim run|eval` (T-14a, T-14b). Smoke defaults to `data/scenarios/examples`; the experiment is `data/scenarios/v1`.
- `just generate-scenarios`: expand `plan.yaml`. Default `--out` is the next unused `data/scenarios/vN`; a directory that already has JSONL is refused. Do not overwrite frozen `v1`.
- `just pull-models` / `just digests` / `just verify-models`: pull the models of `configs/models.yaml`, print their digests, fail on any digest mismatch (`scripts/models.py`).
- Always go through `uv run ...`; add dependencies with `uv add` (or `uv add --group dev`), never pip.

## Layout

| Path | Purpose |
|---|---|
| `src/sim/` | the package (`python -m sim`): LLM client, FSM engine, agents, simulated user, judge, evaluators, runner, eval |
| `tests/` | pytest; `@pytest.mark.integration` on anything that talks to Ollama |
| `data/` | KB (`kb/`), FSM (`fsm/machine.yaml`, `fsm/states/*.md`), prompts (`prompts/*.md`), scenarios (`scenarios/plan.yaml`, `scenarios/examples/`, frozen `scenarios/v1/`) |
| `configs/models.yaml` | models, digests, `num_ctx`, fixed params: the single place for these |
| `runs/` | everything produced by `sim run` / `sim eval`: dialogue JSONL, LLM call logs, prompt cache, `metrics.csv`, `metrics_turn.csv`. Git-ignored; moved between machines by rsync |
| `results/`, `notebooks/`, `docs/`, `scripts/` | analysis outputs, `analysis.ipynb`, project docs (English; the thesis translates what it lifts), utilities |
| `ai-assistance/` | these instructions and the hook scripts |
| `.claude/` | Claude Code config: `settings.json` (hooks, permissions), `skills/`, `rules/` |

### TICKETS.md names → repository names

TICKETS.md predates the English-only decision. When a ticket names an artifact, use the repository name:

| In TICKETS.md | In this repository |
|---|---|
| `data/kb/base_conhecimento.md`, `agulhas.json`, `sem_resposta.json` | `data/kb/knowledge_base.md`, `needles.json`, `unanswerable.json` |
| `data/fsm/maquina.yaml`, `data/fsm/estados/*.md` | `data/fsm/machine.yaml`, `data/fsm/states/*.md` |
| `data/prompts/juiz_fatos.md`, `juiz_global.md` | `data/prompts/judge_facts.md`, `judge_global.md`, plus `judge_shared.md` (blindness block, T-04) |
| `data/cenarios/exemplos/`, `data/cenarios/v1/` | `data/scenarios/examples/`, `data/scenarios/plan.yaml` (authoring; not loaded at runtime), frozen `data/scenarios/v1/` |
| `scripts/gerar_cenarios.py` | `scripts/generate_scenarios.py` |
| `metricas.csv`, `metricas_turno.csv`, `descritiva.csv`, `testes.csv`, `ambiente.txt`, `casos.md`, `tabelas/`, `figuras/` | `metrics.csv`, `metrics_turn.csv`, `descriptive.csv`, `tests.csv`, `environment.txt`, `cases.md`, `tables/`, `figures/` |
| `notebooks/analise.ipynb` | `notebooks/analysis.ipynb` |
| `docs/metricas.md`, `taxonomia.md`, `piloto.md`, `paridade.md`, `validacao_juiz.md`, `execucao.md`, `decisoes_e_limitacoes.md`, `apendices/` | `docs/metrics.md`, `taxonomy.md`, `pilot.md`, `parity.md`, `judge_validation.md`, `execution.md`, `decisions_and_limitations.md`, `appendices/` |
| `sim run --cenarios` | `sim run --scenarios` |
| states `saudacao → identificacao → classificacao_demanda → coleta_dados → solucao → confirmacao → encerramento`, `fora_de_escopo` | `greeting → identification → intent_classification → data_collection → solution → confirmation → closing`, `out_of_scope` |
| events `pedido_identificado`, `demanda_classificada`, `usuario_insatisfeito` | `order_identified`, `intent_classified`, `user_dissatisfied` |
| scenario fields `categoria`, `intencao`, `persona_usuario`, `objetivo_usuario`, `roteiro`, `resposta_referencia`, `fatos_obrigatorios`, `fatos_proibidos`, `estado_final_esperado`, `criterio_sucesso`, `max_turnos`, `eh_agulha`, `canario` | `category`, `intent`, `user_persona`, `user_goal`, `script`, `reference_answer`, `required_facts`, `forbidden_facts`, `expected_final_state`, `success_criterion`, `max_turns`, `is_needle`, `canary` |
| categories `caminho_feliz`, `borda`, `adversarial` | `happy_path`, `edge`, `adversarial` |
| intents rastreio, troca/devolução, cancelamento, 2ª via de pagamento | `order_tracking`, `exchange_return`, `cancellation`, `payment_reissue` |
| metrics `aderencia_fluxo`, `tarefa_concluida`, `conteudo_ofensivo`, `agulha_recuperada`, `acuracia` (correta/parcial/incorreta) | `flow_adherence`, `task_completed`, `offensive_content`, `needle_recovered`, `accuracy` (correct/partial/incorrect on the judge), `accuracy_score` (0 / 0.5 / 1), `fact_precision` / `fact_recall` / `fact_f1`, `claim_support` |

## Invariants: do not change without a dated row in Decisões

From the section *O que não deve mudar* of TICKETS.md:

- `run` and `eval` are separate phases; every LLM call is cached by prompt hash so re-evaluation never re-executes.
- Judge: a different model family, blind to agent metadata (no agent name, states or prompt template in its prompts), two calls per dialogue, randomized order, and the **same** judge in validation (T-16) and in the experiment (T-17).
- Stage labeler runs over the **agent's** turn, identically for both agents. The user-event classifier (T-08) is a different thing and is never used to infer the baseline's path.
- Unit of analysis is the scenario (repetitions aggregated); report wins/ties/losses per scenario.
- Parity checklist, dated decisions table, explicit out-of-scope list, frozen dataset hash.
- `num_ctx` explicit in config; fail loudly when a prompt exceeds 80% of it. Structured output via JSON Schema (`format=<schema>`), no defensive parsing.
- The simulated user sees only persona, goal and script, never the KB or the reference answer.
- Both agents get the full KB. The FSM differs by explicit states, transitions, guards, event classification and state-specific instructions: a deliberate difference, not a bug.

## Working rules

- Work one ticket at a time in the order of *Ordem de execução*, preferably through `/ticket T-xx`. Its acceptance criteria are the test list; a ticket is done when they are checked, `just check` is green, and any experiment-affecting choice is logged in `DECISOES.md` (`AAAA-MM-DD · Ticket · Decisão · Por quê`).
- Vertical slice first: make the thin end-to-end path work, then fatten it (full KB, 8 states, full dataset). Do not implement future tickets while passing by.
- Cuts (N, K, metrics) are decided before looking at results, per *Plano de corte*. Never drop data or scenarios after seeing numbers.
- Never commit `runs/`; never edit `data/scenarios/v1/` (T-06 freeze: hash in `DECISOES.md` and `FROZEN_V1_HASH` in `tests/helpers.py`; `just check` fails if the files move).
- Commit messages: imperative, English, ticket prefix (`T-07: add ollama client with prompt-hash cache`).

## Enforcement

Hooks run without asking, from `.claude/settings.json` in Claude Code and `.cursor/hooks.json` in Cursor. Both call the same two scripts, which read either payload shape, so the rule is written once:

- **After every Write or Edit of a `.py` file**: `ruff format` and `ruff check --fix` on that file (`ai-assistance/scripts/format_on_edit.py`). The file may change on disk; the harness tells you, so re-read it before the next edit. Anything ruff cannot fix comes back to you: fix it before moving on. Unused imports are reported, never removed, because edits land one at a time.
- **Before any `git commit`**: `just check` (`ai-assistance/scripts/quality_gate.py`). Red blocks the commit and shows the output. Fix the code; never bypass with `--no-verify`, by skipping or deleting tests, or by editing the hook.
- **When you end a turn with uncommitted changes**: `just check` again. Red hands you the output and you continue. If the failure is not yours to fix, say so plainly and stop; the second stop is not gated.

Two of these land differently in Cursor, which is worth knowing before trusting them. Its edit hook has no channel back to the model, so what ruff could not fix is only logged and the turn-end gate is what catches it. And it cannot block a turn: the gate comes back as a follow-up message that restarts the turn, capped by the `loop_limit` of `.cursor/hooks.json`, and a turn you interrupted is never followed up.

`just check` is the gate everywhere, so keep it fast: no Ollama call outside an `integration`-marked test.
