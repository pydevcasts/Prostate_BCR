# Run doc — Prostate BCR worktree preview

## How to run the "server"

This is a Python data-science project (no Node/web server). The preview is a
**static standalone HTML file** — no process, port, or dependency install:

- Preview file: `.freebuff/preview.html` (registered via `htmlPath`).
- The Preview tab reloads it from disk; just edit the file and reload.

## How to reproduce the data artifacts shown in the preview

All commands run from `core/` with the project's Python environment:

```bash
# MSKCC external cohort (Step 1): builds core/data/external/mskcc_cohort.csv
# and core/outputs/tables/mskcc_cohort_summary.json
python -m src.mskcc_cohort

# Transferable features (Step 2): builds core/data/processed/
# tcga_transferable_features.csv + core/data/external/
# mskcc_transferable_features.csv (hard validation gates included)
python -m src.transferable_features

# Transferable late fusion (Step 3): retrains on TCGA, applies frozen to
# MSKCC -> core/outputs/tables/transferable_fusion_results.json
python -m src.transferable_fusion

# Rank-based genomic transfer (Step 4, frozen rule, single run):
# -> core/outputs/tables/rank_transfer_fusion_results.json
python -m src.rank_transfer_fusion

# Clinical-value diagnostics (Step 5): pure-clinical baseline, paired
# bootstrap deltas, calibration, DCA -> clinical_value_diagnostics.json
python -m src.clinical_value_diagnostics

# Repeated nested CV from raw TCGA (5 folds x 3 repeats, ~30 min):
# full pipeline refit inside every fold -> nested_cv_raw_results.json
# Use --pilot for a single-repeat smoke run (~10 min).
python -m src.nested_cv_raw

# Feature-selection stability (Step 7, ~45 min): 35 prespecified PSO events
# (15 fold selections + 20 fresh full-train runs, seeds 500-519).
# Each finished fresh run is checkpointed to stability_fresh_runs.csv, so an
# interrupted run resumes instead of restarting; --aggregate-only re-builds
# the tables/figure from the checkpoint without any PSO work.
python -m src.stability_selection
```

Artifacts written by the two commands above: `stability_summary.json`,
`stability_genomic.csv`, `stability_clinical.csv`, `stability_fresh_runs.csv`,
and `figures/feature_stability.png`.

Numbers quoted in `preview.html` come from the real pipeline outputs,
notably `core/outputs/tables/nested_cv_raw_results.json` (definitive
internal estimate), `nested_late_fusion_results.json` (legacy
artifact-based), `transferable_fusion_results.json`,
`rank_transfer_fusion_results.json`, `clinical_value_diagnostics.json`,
and `mskcc_cohort_summary.json`.
