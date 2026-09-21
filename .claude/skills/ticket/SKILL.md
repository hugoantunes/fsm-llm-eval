---
name: ticket
description: Work one ticket (T-01 to T-24) from TICKETS.md test-first. Use when the user asks to start, implement or continue a ticket, e.g. "vamos fazer o T-07" or "/ticket T-08". Turns the acceptance criteria into a test list, stops for review, then runs red-green-refactor per test and drafts the Decisões rows.
argument-hint: "T-07"
---

# Ticket

Work ticket `$ARGUMENTS` end to end, test-first. The control documents live in `~/Documents/mba/projeto/`, split so a session loads only what it needs:

- `TICKETS.md`: quadro, ordem de execução, detalhes e critérios de aceite
- `DECISOES.md`: tabela datada
- `PROGRESSO.md`: registro de sessão

The conventions are `ai-assistance/DEVELOPMENT.md`, already loaded.

## Phase 1: Load

1. Read the ticket's detail block in TICKETS.md: the heading `#### $ARGUMENTS ·`, its description and its *Critérios de aceite*. If TICKETS.md cannot be read, the iCloud copy is probably evicted: stop and ask the user to open the file. TICKETS.md is in Portuguese and names artifacts in Portuguese; the repository is English only, so translate every path, field, state and metric name with the table *TICKETS.md names → repository names* in `ai-assistance/PREAMBLE.md`.
2. Read the ticket's row in the *Quadro* of TICKETS.md (Est., Depende de, Status) and every row in DECISOES.md that names `$ARGUMENTS`. Those decisions are already made; do not reopen them.
3. Check the dependencies (*Depende de*): confirm their deliverables exist in the repo. Report a missing one and stop, unless the user says to proceed anyway.
4. Read the code and data the ticket touches. Do not read unrelated modules. Do not load PROGRESSO.md unless you need to resume a session.

## Phase 2: Plan, then STOP

Turn each acceptance criterion into test cases, one behaviour per test, and present:

```
## $ARGUMENTS: <title>

### Tests, in the order they will be written
1. tests/test_<module>.py::test_<behaviour>   <- criterion N   [unit | integration]
...

### Files
- src/sim/<module>.py (new | modify): <what>
- data/...: <what>

### Decisions this ticket requires
- <choice>: proposed <value>, because <why>   (-> DECISOES.md row)

### Not covered by a test
- <criterion>: <how it will be verified instead>
```

Unit tests use synthetic data and no model. Anything that needs Ollama is `@pytest.mark.integration`. Prefer a fake over a mock.

**STOP HERE.** Wait for the user to approve or change the plan. Write no code before that.

## Phase 3: Red, green, refactor, one test at a time

For each test in the approved order:

1. **Red.** Write that single test. Run it: `uv run pytest <file>::<test>`. Confirm it fails for the right reason, a missing name or a wrong value, not a broken fixture.
2. **Green.** Write the least production code that passes it. Run it again.
3. **Refactor.** Remove duplication, fix names, keep the tests green. Run `just test`.

Never write production code without a failing test. Never write several tests ahead. If a test turns out to be wrong, fix the test first, then continue.

## Phase 4: Close

1. `just check` is green. If the ticket has integration tests, run `just test-all` when Ollama is reachable; otherwise say they were not run.
2. Walk the acceptance criteria and report each one: met (naming the test or artifact that proves it), not met, or manual step for the user.
3. Draft the *Decisões* rows in the table's format, `| AAAA-MM-DD | $ARGUMENTS | decisão | por quê |`, with today's date, one row per decision made, plus the *Registro de progresso* line and the new ticket status. Propose them; apply them only when the user approves: decisions to `DECISOES.md`, the session line to `PROGRESSO.md`, ticket status to `TICKETS.md`. Those files are the user's control documents and live outside the repo.
4. Propose the commit message, `$ARGUMENTS: <imperative summary>`. Commit if the user asked for it.
