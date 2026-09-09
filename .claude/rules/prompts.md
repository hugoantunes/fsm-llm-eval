---
paths:
  - "data/prompts/**"
  - "src/sim/**"
---

# Prompts: blindness and parity

Applies whenever you touch a prompt file or the code that assembles a prompt. These are the invariants a thesis committee will probe first.

- **The judge is blind to metadata, not to style.** Its two prompts (T-12) receive only: the clean full transcript (user and agent turns), the scenario script, the reference answer, the required and forbidden fact IDs, and the KB. Never the agent name, any `state_*` field, the transition history, a state package, or text from `data/prompts/baseline.md` or `fsm_template.md`. The T-12 test that scans the judge prompts for agent identifiers, `state_` and template fragments stays green.
- **The simulated user sees only its brief.** `user_persona`, `user_goal`, `script`. Never the KB, `reference_answer`, `required_facts`, `forbidden_facts` or `success_criterion`, or the reference leaks to the agent through the dialogue. The T-11 test for this stays green.
- **Parity.** Baseline and FSM share one identical block for persona, tone and general rules. Only the structure differs: full KB versus the state package plus the facts released for that state. Change the shared block in one place; never tune one agent's wording without the other's.
- **Prompts live in files.** `data/prompts/*.md`, English, loaded at runtime. No
  prompt text in Python strings. Each file carries a version string that the run
  manifest records. A renderer may fill a versioned row template (`user_data_field.md`)
  or reshape structured data into the source's own markdown (`render_facts`); it
  must not invent instructional English.
- **Schema and rubric move together.** Judge output is constrained by JSON Schema (`format=<schema>`); a change to the rubric's fields is a change to the schema, in the same commit.
- **Frozen at tag `v1`.** After T-16, prompts and rubrics change only with a dated Decisões row and a new tag. Before that, every change to a rubric is noted in `docs/pilot.md`.
- English only, and no real brands, people or companies anywhere in prompts or scenarios.
