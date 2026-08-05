

# Prostate Cancer Biochemical Recurrence (BCR) Prediction
**Multi-omics (Clinical + RNA-Seq) Machine-Learning Pipeline with Nested Cross-Validation and SHAP-Based Explainability — TCGA-PRAD**

## 1. Overview

This project predicts **biochemical recurrence (BCR)** after radical prostatectomy in prostate adenocarcinoma patients using TCGA (PRAD) data. It integrates:

- **Clinical features** (Gleason patterns/score, histology, stage, survival follow-up, …)
- **RNA-Seq gene expression** (~18.9k genes after QC)

into a supervised binary-classification pipeline consisting of:

1. Reproducible **data preparation** with an explicit **leakage audit**,
2. **Feature selection** down to a compact 30-feature signature,
3. **Nested cross-validation** model comparison (XGBoost selected as the final model),
4. **Explainability** (XGBoost importance + SHAP) producing a publication-ready biomarker ranking.

**Target variable:** `Biochemical Recurrence Indicator` (YES/True → 1, NO/False → 0); positive rate ≈ **13.5%** (imbalanced classification).

## 2. Repository Layout

```
Prostate_BCR_Q1/
└── Core/                        # project root (config.py lives here)
    ├── config.py                # central paths & constants (TARGET_COLUMN, dirs, …)
    ├── notebooks/
    │   ├── 01_Data_Preparation.ipynb
    │   ├── 05_Model_Training.ipynb
    │   └── 06_Explainability.ipynb
    ├── src/                     # all business logic (notebooks only orchestrate)
    │   ├── io.py                # logging, path validation, load/save helpers
    │   ├── clinical.py          # clinical preprocessing + target creation
    │   ├── genomics.py          # RNA-Seq preprocessing + QC report
    │   ├── merge.py             # clinical–genomics join by Patient Identifier
    │   ├── leakage.py           # leakage audit & report
    │   ├── feature_selection.py # feature selection / transform_selected
    │   ├── models.py            # XGBoost factory, safe-frame & name-map utils
    │   ├── pipeline.py          # nested-CV evaluation & model comparison
    │   ├── visualization.py     # plotting style & comparison charts
    │   └── explainability.py    # importance, biomarker ranking, SHAP plots
    ├── data/
    │   ├── raw/
    │   │   ├── data_clinical_patient.tsv
    │   │   └── data_mrna_seq_v2_rsem.txt
    │   └── processed/
    │       ├── clinical_data_complete.csv
    │       ├── X_features_final.csv
    │       └── y_target_final.csv
    └── outputs/
        ├── models/best_model_xgboost.joblib
        ├── figures/             # nested_cv_comparison.png, per_fold_auc_by_model.png,
        │                        # feature_importance_top20.png, SHAP plots, …
        └── tables/              # leakage_report.csv, feature_importance_full.csv,
                                 # biomarker_ranking_genes.csv, biomarker_ranking_clinical.csv
```

## 3. Data Sources

| Dataset | File | Raw shape | Notes |
|---|---|---|---|
| Clinical | `data_clinical_patient.tsv` | 504 × 69 | 4 cBioPortal metadata rows + 500 patients |
| RNA-Seq (RSEM) | `data_mrna_seq_v2_rsem.txt` | 20,531 genes × 498 samples | `Hugo_Symbol` + `Entrez_Gene_Id` columns |

## 4. Data Preparation (Notebook 01)

### 4.1 Clinical preprocessing (`src/clinical.py`)
- Drop 4 metadata rows → **500 patients**.
- Type detection: 12 numeric / 56 categorical columns.
- Drop 21 near-constant columns; remove 8 leakage-prone columns.
- Drop high-cardinality categoricals (> 15 levels), e.g. `#Other Patient ID`, `Form completion date`, `Days to bone scan performed`, `Days to CT scan`, `Days to MRI`, `Tissue Source Site`.
- One-hot encoding → 115 columns.
- Target creation from `Biochemical Recurrence Indicator` (YES/True → 1, NO/False → 0).

### 4.2 RNA-Seq preprocessing (`src/genomics.py`)
- Drop `Entrez_Gene_Id`; resolve 17 duplicated gene symbols → 20,514 genes.
- Low-expression filtering → **18,905 genes**.
- Remove 1 duplicated patient → **497 samples × 18,905 genes**.
- QC report: min 0.00 · max ≈ 1,573,524 · mean ≈ 1,047.7 · median ≈ 252.1 · %zeros ≈ 6.96 · %missing = 0.

### 4.3 Merge & leakage audit (`src/merge.py`, `src/leakage.py`)
- Inner join on `Patient Identifier`; leakage audit writes `leakage_report.csv` and drops flagged columns.

| Stage | Samples | Features |
|---|---|---|
| Merged | 429 | ≈ 20,664 |
| After leakage audit | 429 | ≈ 20,645 |
| Final positive rate | — | ≈ 13.5% |

Final artifacts: `X_features_final.csv`, `y_target_final.csv`.

## 5. Modeling (Notebook 05)

- **Nested cross-validation** (`evaluate_nested_cv`, `compare_models_nested_cv`) for unbiased comparison of candidate models (XGBoost, Logistic Regression, …).
- Aggregated nested-CV AUC statistics per pipeline configuration:

| Config | Mean AUC | Std | Min | Max |
|---|---|---|---|---|
| mi | 0.747 | 0.081 | 0.678 | 0.866 |
| pso | 0.680 | 0.108 | 0.529 | 0.806 |

- **Final model:** `XGBClassifier`, persisted to `outputs/models/best_model_xgboost.joblib`.
- Figures: `nested_cv_comparison.png`, `per_fold_auc_by_model.png`.

## 6. Explainability & Biomarker Discovery (Notebook 06)

- Inputs: final XGBoost model + selected feature set (**30 features**); held-out test set of **86 samples**.
- XGBoost importance → `feature_importance_top20.png` + `feature_importance_full.csv`.
- Biomarker ranking split into gene and clinical tables: `biomarker_ranking_genes.csv` (19 genes) and `biomarker_ranking_clinical.csv` (1 clinical feature).
- Example top-ranked genes (importance): **LRR1** (0.0275), **DHX34** (0.0265), **APEH** (0.0265), **ACSM2A** (0.0264), **TNIP3** (0.0262), **PLAC8** (0.0261), **ARMC4** (0.0257), **VPS13A** (0.0255).
- SHAP analysis: summary (beeswarm), waterfall (per-patient explanation), and dependence plots, saved under `outputs/figures/`.

## 7. Setup & Reproduction

1. Create a virtual environment (project runs on Python 3.14, `venv`).
2. Install dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `joblib`.
3. Place raw TCGA files under `data/raw/`.
4. Run notebooks **in order**: 01 → 05 → 06 (notebooks only orchestrate; all logic lives in `src/`).
5. Collect artifacts from `outputs/{models,figures,tables}`.

## 8. Engineering Conventions

- **Separation of concerns:** notebooks orchestrate; `src/` modules own the logic; `config.py` owns all paths/constants.
- **Leakage hygiene:** dedicated leakage-audit step; target built from a single source column; ID-like/high-cardinality columns dropped.
- **Reproducibility:** centralized logging (`prostate_bcr` logger), validated paths (`validate_path`), and all figures/tables written via `save_figure` / `save_table`.

## 9. Troubleshooting / Known Issues

| Symptom | Cause / Fix |
|---|---|
| `ValueError: Target source column 'Biochemical Recurrence Indicator' not found` | The source column was dropped before `create_target_column` (e.g., by leakage/cardinality filtering) or is absent from the raw file. Verify column presence and preprocessing order. |
| `FileNotFoundError: … clinical_data_complete.csv` | Notebook 01 did not complete the clinical save step; re-run Notebook 01 end-to-end before merging. |
| `FileNotFoundError: … best_model_xgboost.joblib` | Path mismatch between where Notebook 05 saves the model and where Notebook 06 loads it (e.g., `core/outputs` vs `Qwen/outputs`). Align paths in `config.py`. |
| sklearn `FutureWarning` on `penalty` (LogisticRegression) | scikit-learn ≥ 1.8 deprecation; use `l1_ratio` / `C` instead. |

