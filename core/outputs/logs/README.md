# Run logs — pipeline evidence

Console logs of the Step 6 (repeated nested CV from raw data) and Step 7
(feature-selection stability) runs. These are committed because the
manuscript cites the determinism check and the runtime of those runs.

## Canonical logs (match the committed artifacts)

| Log | Command | Artifacts |
|---|---|---|
| `nested_cv_raw_final.log` | `python -m src.nested_cv_raw` | `nested_cv_raw_results.json`, `nested_cv_raw_folds.csv`, `nested_cv_raw_oof_predictions.csv`, `nested_cv_raw_pso_selections.csv` |
| `stability_selection_final.log` | `python -m src.stability_selection` | `stability_summary.json`, `stability_genomic.csv`, `stability_clinical.csv`, `stability_fresh_runs.csv`, `figures/feature_stability*.png` |

Both canonical logs show the corrected branch masks
(`Column groups: 115 clinical, 18904 gene`), 5 folds x 3 repeats
(seeds 42/43/44) for Step 6, and 15 fold events + 20 fresh runs
(seeds 500-519) for Step 7.

## `superseded/` — retained evidence of the debugging path

Kept for transparency; their numbers are **not** cited anywhere.

| Log | Why superseded |
|---|---|
| `nested_cv_raw_pilot_pre_fix.log` | Single-repeat pilot run under the buggy branch masks. |
| `nested_cv_raw_pilot_post_fix.log` | Pilot after the branch-mask fix (0.7817) — first evidence the bug depressed the estimate. |
| `nested_cv_raw_full_pre_fix_run1.log`, `..._run2.log` | Full pre-fix runs (0.7551). The identical repeat AUCs across these two runs are the determinism evidence; the level itself is superseded by the mask correction. |
| `stability_selection_pre_fix_killed.log` | Stability run under the buggy masks; killed after the mask bug was found. |
| `stability_selection_pre_count_fix.log` | Stability run with correct masks but before the event-counting bug in the aggregation was fixed (fold frequencies were capped at 1/15). |
| `stability_selection_aborted_duplicate.log` | Accidental second concurrent stability process, terminated to avoid a write race. |
