# Decisions and limitations

> **Experiment guide** · step 11 of 11 · [All steps](README.md) ·
> [← Decomposing `fact_f1`](decomposition.md)

English synthesis for Material e Métodos and Discussão. Dated experiment-affecting
choices remain in `DECISOES.md` (outside this repository). This file does not copy
that table. Narrative that already has a home stays there:
[`docs/judge_validation.md`](judge_validation.md),
[`docs/execution.md`](execution.md),
[`docs/pilot.md`](pilot.md),
[`docs/parity.md`](parity.md),
[`docs/metrics.md`](metrics.md),
[`docs/audit.md`](audit.md),
[`docs/decomposition.md`](decomposition.md),
[`results/human_primary/README.md`](../results/human_primary/README.md).

## For Methods

The experiment asks whether a finite-state machine used as the *instruction base*
of an LLM customer-service agent outperforms one well-written system prompt. The
**treatment** is instruction **structure**: baseline receives a single prompt;
FSM receives per-state instruction packages plus the user-event classifier that
drives transitions. Both agents receive the same complete knowledge base (37
facts). Agent model and knowledge base are held fixed, so the design estimates
the effect of instruction structure under that pair of constants. It does not
estimate the effect of FSMs in general, of model choice, or of judge backend.

Agents and the simulated user call `qwen3.5:9b`. The event classifier and stage
labeler call `qwen3.5:4b`. Every call uses `think=False`. The frozen-gate judge
is the T-16-validated `gemma4:12b` GGUF blob. Semantic-primary, the confirmatory
population, uses `gemma4:12b-mlx`. The same Pilot v2 dialogues were re-scored
with that Ollama-served `gemma4:12b-mlx` configuration so the confirmatory
instrument can be compared with the human labels of the original validation
([`docs/judge_validation.md`](judge_validation.md)). T-18 keeps GGUF and MLX as
two exported populations. `run` and `eval` executed on the MacBook Air as the
documented fallback; the Pro only pulls `runs/` as backup
([`docs/execution.md`](execution.md), [`docs/parity.md`](parity.md)).

The frozen dataset is `data/scenarios/v1/` (planned N = 60, K = 3, hash
`0778a90110e6dd67685c1e6768934483cb4405213dad35ec172f3ddcf21d47c9`), English-only,
synthetic, authored as a cell plan (`data/scenarios/plan.yaml`) and expanded with
`--phrasing seed`, so no model rephrased it, then reviewed scenario by scenario. The simulated
user sees persona, goal, and script only. The script is a mandatory ordered plan;
`just adherence` is the execution gate. The first pilot does not validate the
judge ([`docs/pilot.md`](pilot.md)).

The judge makes two schema-constrained calls per dialogue, is metadata-blind, and
sees dialogues in randomized order. It is not guaranteed blind to stylistic
differences between conditions. It was not retuned after T-16
([`docs/judge_validation.md`](judge_validation.md)).

Frozen populations ([`docs/audit.md`](audit.md)): semantic-primary
350 eligible / 348 scored; frozen-gate 342 eligible / 340 scored.
8 CFP rows exist only in the semantic sidecar. 1 TIF excluded.
9 `max_turns` excluded. Two eligible-but-unscored rows remain unscored:
`edge_01/fsm/2` and `edge_13/fsm/2`.

The unit of analysis is the scenario, not the dialogue. The n of each confirmatory
test is the number of paired scenarios. Confirmatory paired n = 59:
`adversarial_12` is unpaired because baseline ended in `max_turns`. Category n =
20/20/19 (happy_path / edge / adversarial). Tests are Wilcoxon signed-rank and
paired permutation. Holm correction applies to `PRIMARY_METRICS`
(`task_completed`, `fact_f1`, `claim_support`) at `semantic_primary × overall`.
Category analyses are exploratory. V/E/D is reported. `flow_adherence` is
diagnostic, not a primary outcome. The planning power benchmark is planned
N = 60 → dz ≈ 0.37; that figure is not the achieved detectable effect of the
confirmatory n = 59 sample ([`docs/metrics.md`](metrics.md)).

A later census labelled the same 348 semantic-primary dialogues under blind IDs
D001–D348 (seed 20260920). Filled sheets were frozen on 2026-09-20T21:11:58Z
(SHA-256 of the three CSVs in `results/human_validation/exp_final/frozen/`)
before unblinding via `private/mapping.csv`. Human `task_completed`, `fact_f1`
and `claim_support` were derived after that freeze; the frozen CSVs were not
edited and disagreements were not used to correct gold. The T-19 confirmatory
family was rerun at `human_primary` × overall (paired n = 59, same
`PRIMARY_METRICS`, Holm). The judge was not retuned. Choosing the human census
as the narrative reference for the thesis conclusion happened after freeze and
unblind, once both Holm families were visible; it was not a pre-declared swap
of the confirmatory instrument. Artifacts:
[`results/human_primary/`](../results/human_primary/README.md).

`fact_f1` measures correspondence to one scenario's expected fact set. It is
not general factuality and not a hallucination rate: its false-positive total
adds `extra_supported_fact` (true in the closed knowledge base, outside this
scenario's expected set) to `unsupported_claim`, and only the second speaks to
grounding, which `claim_support` measures on its own denominator. The post-hoc,
exploratory decomposition of the frozen effect — declared as such, computed by
replaying judge output already in `llm_calls.jsonl` and verified against the
exported rows, never by re-judging — is
[`docs/decomposition.md`](decomposition.md). It is not a fourth confirmatory
test and it does not enter the Holm family.

Cut 0 is applied (no sentiment metric; no toxicity classifier). Cuts 1–4 are not
applied. `relevance` remains in judge call 2, so Cut 2 was not applied. Cut 0
fields are operational proxies, not independently validated constructs:
`task_completed` and `relevance` for task/interaction quality, `offensive_content`
for toxicity.

## For Discussion

### Single agent model configuration

The treatment comparison uses one agent model/configuration, `qwen3.5:9b`.
Classifier, labeler, simulated user, and judge are other models used as
instruments, so the study is not a single-model stack. Generalization to other
agent models is untested.

Mitigation: both agents share that model, sampling, seed derivation, and the
full knowledge base, so a gap is attributable to instruction structure rather
than model identity ([`docs/parity.md`](parity.md)). A second agent family is
out of scope.

### Judge blindness and agreement

The judge is blind to metadata and condition labels, not guaranteed blind to
style: the FSM walk remains visible in the transcript. Pilot v2 (9B simulated
user, the frame that precedes T-17) is the quoted agreement:
`task_completed` κ = 0.667, `claim_support` κ = 0.533, `accuracy` weighted κ =
0.569 ([`docs/judge_validation.md`](judge_validation.md)). The T-16 freeze
(Pilot v1) is historically lower (`task_completed` κ = 0.419). Pilot v2 does
not validate every judge field: `relevance`, `offensive_content`, and
`needle_recovered` were not independently validated.

Mitigation: two schema-constrained calls, metadata omitted from prompts,
randomized evaluation order, human validation before T-17, judge not retuned.
Unvalidated fields are not in `PRIMARY_METRICS`.

### Stage labeler accuracy

Revision 1 on Pilot v2: 25/39 turn match, `valid_flow_path` = 8/10. Revision 1
was retained. Revision 2 was a post-hoc sidecar after the instrument was frozen
and was not adopted. A later sidecar re-ran revision 1 with `qwen3.5:9b` on the
same 39 FSM turns (prompts and sampling unchanged): 27/39 turn match, path
validity 6/10, 5 of 14 4B errors corrected, 3 new errors. The pre-declared bar
(turn match up and path validity not down) was not met. On T-08 gold phrases the
9B classifier scored 19/21 against 4B 21/21. That sidecar is evidence about
capacity, not a replacement of the 4B instruments
([`docs/judge_validation.md`](judge_validation.md)). `flow_adherence` is scored
from revision-1 4B labels, which is why it is diagnostic rather than a primary
outcome.

Mitigation: KEEP STAGE-LABELER REVISION 1; classifier and labeler stay
`qwen3.5:4b`; `flow_adherence` stays outside `PRIMARY_METRICS` and outside Holm.

### Simulated user as instrument

The simulated user follows persona, goal, and an ordered script; delivery is
decided by `sim.script`, not by the model. The first pilot treated the script as
a hint, dropped beats, and does not validate the judge. Three frozen v1 rows
(`adversarial_04`, `adversarial_08`, `adversarial_12`) set
`expected_final_state: out_of_scope` but a farewell beat fires `farewell` from
`*` into `closing`, so that expected state is unreachable. Their relationship to
the primary outcomes follows the frozen pipeline semantics:
`expected_final_state` is consumed by `flow_scores` (`ended_in_expected_state` /
`flow_adherence`); `flow_adherence` is diagnostic and outside `PRIMARY_METRICS`.
The three rows still receive judge scores on the primaries
([`docs/metrics.md`](metrics.md), [`tests/test_dataset.py`](../tests/test_dataset.py)).

Mitigation: runtime beat contract, `just adherence` as the execution gate, and
first-pilot sheets discarded. The waiver is frozen as a dataset invariant, not
treated as an exclusion from confirmatory tests.

### Synthetic dataset generated by another model

Frozen `v1` is synthetic and reviewed, not production tickets. The plan of
2026-09-07 was to phrase scenarios with the judge model; the dataset that was frozen
on 2026-09-10 was instead written as a cell plan (`data/scenarios/plan.yaml`: persona,
goal, script and answer key per cell) and expanded with `--phrasing seed`, which copies
that text as written, so neither the judge nor the agent model rephrased it
([`docs/taxonomy.md`](taxonomy.md)). What remains is the risk of any authored set:
a narrower range of customer styles than real traffic. All four canary scenarios plant the same
injection sentence, varying only the token and the opening intent: one adversarial
template, not four independent probes.

Mitigation: no model wrote the frozen text, so there is no generator–evaluator
overlap to measure and agents are never scored on their own phrasing; every scenario
was reviewed; the dataset hash is frozen; `injection_succeeded` stays exploratory
([`docs/metrics.md`](metrics.md)).

### Sample size and power

Planned N = 60. Confirmatory paired n = 59. Category n = 20/20/19. The planning
benchmark dz ≈ 0.37 belongs to planned N = 60 at two-sided α = 0.05 and 80%
power; it is not the achieved detectable effect of the n = 59 confirmatory
sample ([`docs/metrics.md`](metrics.md)).

Mitigation: N and the floor of 45 were fixed before results; the unpaired
scenario is named (`adversarial_12`); category tests are declared exploratory.

### Scenario as the unit of analysis

The inferential unit is the scenario. Treating K dialogues as independent
observations would inflate n by K. The trade-off is the smaller effective
sample.

Mitigation: repetitions still appear as intra-scenario SD in
`descriptive.csv`; V/E/D is reported per scenario
([`docs/metrics.md`](metrics.md)).

### Ollama non-determinism

Local sampling is not deterministic. K = 3. The per-dialogue seed is derived
from `(42, scenario_id, repetition)` without the agent name, so both agents
meet the same customer on a given pair. Repeated runs reduce dependence on one
draw; they do not make execution deterministic.

Mitigation: identical derived seeds across agents, K = 3, and intra-scenario
SD reported rather than a single-shot transcript treated as the scenario.

### Small knowledge base

The knowledge base has 37 facts for a fictitious shop (Northlight Store). That
limits external validity to larger or more heterogeneous KBs.

Mitigation: facts are atomic IDs (F01–F37) with needles and an unanswerable
list; both agents receive the same complete knowledge base, so a gap is not a
knowledge-coverage contrast.

### Deliberate structure contrast with the same knowledge base

Both conditions receive the same full knowledge base. The intervention is
instruction structure. That improves attribution to structure and narrows the
claim: results do not generalize to systems where an FSM also changes
retrieval, tools, model, memory, or accessible knowledge.

Mitigation: parity checklist; shared `agent_shared.md`; identical
`render_facts` on both agents ([`docs/parity.md`](parity.md)).

### Cut 0 metrics and construct validity

Cut 0 dropped a sentiment score (the simulated user plays a script, so affect
would measure the dataset) and a dedicated toxicity classifier. What remains
are operational proxies, not validated UX or toxicity instruments:
`task_completed` and `relevance` for task/interaction quality, `offensive_content`
for toxicity. `relevance` remains in call 2, so Cut 2 was not applied. Those
judge fields were not independently validated (see Judge blindness and
agreement).

Mitigation: Cut 0 was pre-declared before results; confirmatory inference uses
`PRIMARY_METRICS` only; unvalidated fields are not treated as validated
constructs ([`docs/metrics.md`](metrics.md)).

### Judge backend split between GGUF and MLX

Frozen-gate uses the T-16-validated `gemma4:12b` GGUF blob. Semantic-primary
uses the sidecar `gemma4:12b-mlx`. They are two backends and two populations.
Frozen-gate versus semantic-primary differences are population and inclusion
differences ([`docs/execution.md`](execution.md), [`docs/audit.md`](audit.md)).
The sidecar expands semantic-primary; frozen-gate is the GGUF sensitivity
population.

The same 20 Pilot v2 dialogues and 30 human response labels were re-scored with
the configuration `gemma4:12b-mlx` served by Ollama, keeping the dialogues,
prompts, and judge frozen. Dialogue-level agreement stayed close and rose
slightly (`task_completed` κ 0.667 → 0.783; accuracy weighted κ 0.569 → 0.658)
because one census case (`D04`) moved toward the human. Response-level
`claim_support` κ moved 0.533 → 0.500. `fact_ids_stated` micro F1 moved
0.824 → 0.765, with unmatched claims 17 → 27, concentrated on four sampled
responses. Semantic-primary uses this MLX configuration; the GGUF Pilot v2
table remains the original validation; T-18 keeps both backends as two
exported populations ([`docs/judge_validation.md`](judge_validation.md)).

Mitigation: provenance stays on each exported row; T-18 copies both CSVs as two
populations; frozen-gate remains the sensitivity population; robustness-check
artifacts live under `results/judge_validation/pilot_v2_mlx/`.

### Human confirmatory evaluation of exp_final

The census is one annotator on 348 dialogues, with no second rater and no
adjudication. Style cues of the FSM walk remain in the transcript. Freeze
preceded unblind; that blocks peeking at judge scores during annotation, not
the later choice of which instrument to privilege in the conclusion.

Human vs judge agreement on this census: `task_completed` 89.4% / κ = 0.729
(n = 348, 0 uncertain). Claim support is a three-way dialogue label from
`n_checkable_claims` and `claim_support`, not atomic claim matching: 45.4% /
κ = 0.102. Fact-ID exact-set comparison kept 241 dialogues and omitted 107
because two `judge_facts` calls reproduce the frozen metrics row and disagree
on predicted IDs (GGUF vs MLX in `llm_calls.jsonl`). Those 107 stay in the
human confirmatory tests; they are missing only from the ID-set agreement.
A post-hoc, exploratory sensitivity (T-25) drops them from both instruments and
reruns the primary tests on 56 paired scenarios
(`results/sensitivity/ambiguous_fact_ids/`): the judge's `fact_f1` difference is
−0.075, 95% CI [−0.131, −0.022], reference Holm p = 0.060; the human
`claim_support` difference is +0.045, CI [−0.011, 0.100], reference Holm p = 0.180.

On the human Holm family, only `claim_support` is significant: mean paired
difference +0.047, 95% CI [0.003, 0.090], p Holm = 0.048 (Wilcoxon p = 0.016,
permutation p = 0.042, rank-biserial +0.361, 39 wins / 0 ties / 20 losses).
The interval excludes zero by a narrow margin. `task_completed` and `fact_f1`
do not detect a difference (p Holm = 1.000). Direction agrees with the judge
on all three metrics; Holm significance agrees only for `task_completed`. The
judge's `fact_f1` FSM deficit is not reproduced by the human labels.

Mitigation: report both instruments; do not retune the judge; do not edit
frozen sheets; name the 107 omissions; delimit the conclusion to claim
support, and treat p Holm = 0.048 / CI [0.003, 0.090] as a fragile result; its
interval includes zero in the T-25 sensitivity.

### English-only experiment

All experimental material (KB, prompts, scenarios, dialogues, identifiers) is
English. Generalization to Portuguese or other languages was not tested and
remains future work.

Mitigation: English was chosen for small-model instruction/schema reliability
and lower token cost (measured); the thesis is Portuguese and glosses
identifiers on first use.
