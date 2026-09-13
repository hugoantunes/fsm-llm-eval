# Run Cleanup Log (2026-09-13)

This note records run artifacts removed from `runs/` during cleanup.

Kept directory: `runs/full_scenario_preflight/` (fresh, empty, paused).

Cleanup result: removed 10 run directories, freed approximately 43.93 MB.

## Removed Run Directories

| run_dir | dialogues | ok | failed | manifest n_ok | manifest n_failed | manifest jobs | failed reasons | notes |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `denial_controls_repro` | 6 | 6 | 0 | 6 | 0 | 6 | - | - |
| `exp_final_pre_fix` | 120 | 117 | 3 | 3 | 3 | 10 | legacy-unclassified=3 | manifest totals differ from dialogue files |
| `full_scenario_preflight_after_goal_fix_fail_2026-09-13` | 120 | 114 | 6 | 114 | 6 | 120 | instrument/invalid_candidate_retry_exhausted=2, simulation/max_turns_with_incomplete_beat=4 | - |
| `full_scenario_preflight_pre_fix_fail_2026-09-13` | 44 | 41 | 3 | - | - | - | simulation/max_turns_with_incomplete_beat=3 | no manifest |
| `hp09_repro_after_denial_fix` | 2 | 2 | 0 | 2 | 0 | 2 | - | - |
| `model_selection_4b` | 20 | 15 | 5 | 15 | 5 | 20 | instrument/invalid_candidate_retry_exhausted=1, simulation/max_turns_with_incomplete_beat=4 | - |
| `model_selection_9b` | 20 | 18 | 2 | 18 | 2 | 20 | simulation/max_turns_with_incomplete_beat=2 | - |
| `pilot_v1` | 20 | 20 | 0 | 20 | 0 | 20 | - | - |
| `role_split_equivalence` | 10 | 7 | 3 | 7 | 3 | 10 | simulation/max_turns_with_incomplete_beat=3 | - |
| `termination_fix_repro` | 12 | 9 | 3 | 9 | 3 | 12 | simulation/max_turns_with_incomplete_beat=3 | - |

## Remarks

- `exp_final_pre_fix` has a manifest from a resumed subset, so its manifest totals differ from the full dialogue file census.
- `full_scenario_preflight_after_goal_fix_fail_2026-09-13` is preserved in this log as the failed 120-dialogue pre-flight snapshot before denial-detector follow-up reruns.
- `full_scenario_preflight_pre_fix_fail_2026-09-13` is preserved in this log as the earlier partial pre-flight snapshot (44 dialogues, no manifest).

Generated at: 2026-09-13T21:54:28.038791Z

Interpretation of these runs (Pilot v1 loss, pre-flight snapshots, 4B/9B
selection) is in [`docs/pilot.md`](pilot.md) and
[`docs/decisions_and_limitations.md`](decisions_and_limitations.md), not here.
