--- core/doc.md (原始)


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



+++ core/doc.md (修改后)
# Prostate Cancer Biochemical Recurrence (BCR) Prediction
**Late Fusion Multi-omics (Clinical + RNA-Seq) Machine-Learning Pipeline with PSO Feature Selection and SHAP-Based Explainability — TCGA-PRAD**

## 1. Overview

This project predicts **biochemical recurrence (BCR)** after radical prostatectomy in prostate adenocarcinoma patients using TCGA (PRAD) data. It integrates:

- **Clinical features** (Gleason patterns/score, histology, stage, survival follow-up, …)
- **RNA-Seq gene expression** (~18.9k genes after QC)

into a **Late Fusion** supervised binary-classification pipeline consisting of:

1. Reproducible **data preparation** with an explicit **leakage audit**,
2. **Parallel branch feature selection**:
   - **Genomic Branch**: Variance threshold → Mutual Information → Binary PSO (selects ~40 genes)
   - **Clinical Branch**: Domain-specific engineered features (14 clinical features)
3. **Dual-model training** with separate XGBoost classifiers for each branch,
4. **Late Fusion layer** that combines branch probabilities using optimized weights (Genomic: 0.61, Clinical: 0.39),
5. **Explainability** (branch-specific XGBoost importance + SHAP) producing publication-ready biomarker rankings.

**Target variable:** `Biochemical Recurrence Indicator` (YES/True → 1, NO/False → 0); positive rate ≈ **13.5%** (imbalanced classification).

**Key Innovation:** Unlike traditional early fusion approaches, our Late Fusion architecture:
- Prevents clinical features from dominating genomic signal
- Enables fallback to genomic-only predictions when clinical data is missing
- Provides interpretable branch-specific contributions to final prediction

## 2. Repository Layout

```
Prostate_BCR_Q1/
└── Core/                        # project root (config.py lives here)
    ├── config.py                # central paths & constants (TARGET_COLUMN, dirs, GENOMIC_WEIGHT, CLINICAL_WEIGHT)
    ├── notebooks/
    │   ├── 01_Data_Preparation.ipynb
    │   ├── 02_EDA.ipynb
    │   ├── 03_Preprocessing.ipynb
    │   ├── 04_feature_selection_late_fusion.ipynb    # Genomic + Clinical branch feature selection
    │   ├── 05_model_training_late_fusion.ipynb       # Dual-model training + Late Fusion
    │   ├── 06_Explainability_LateFusion.ipynb        # Branch-specific explainability
    │   ├── 07_Final_Evaluation_LateFusion.ipynb      # Final evaluation with all metrics
    │   └── 08_external_validation_late_fusion.ipynb  # External validation with fallback
    ├── src/                     # all business logic (notebooks only orchestrate)
    │   ├── io.py                # logging, path validation, load/save helpers
    │   ├── clinical.py          # clinical preprocessing + target creation
    │   ├── genomics.py          # RNA-Seq preprocessing + QC report
    │   ├── merge.py             # clinical–genomics join by Patient Identifier
    │   ├── leakage.py           # leakage audit & report
    │   ├── genomic_selector.py  # Genomic branch: MI + PSO feature selection
    │   ├── clinical_engineer.py # Clinical branch: domain-specific feature engineering
    │   ├── models.py            # XGBoost factory, safe-frame & name-map utils
    │   ├── visualization.py     # plotting style & comparison charts
    │   ├── evaluation.py        # metrics calculation (AUC, accuracy, F1, etc.)
    │   ├── explainability.py    # importance, biomarker ranking, SHAP plots
    │   └── fusion/
    │       ├── __init__.py
    │       └── late_fusion.py   # LateFusionPredictor class, weight optimization
    ├── data/
    │   ├── raw/
    │   │   ├── data_clinical_patient.tsv
    │   │   └── data_mrna_seq_v2_rsem.txt
    │   └── processed/
    │       ├── clinical_data_complete.csv
    │       ├── X_train_preprocessed.csv
    │       ├── X_test_preprocessed.csv
    │       └── y_target_final.csv
    └── outputs/
        ├── models/
        │   ├── genomic_model.joblib           # XGBoost on genomic features
        │   ├── clinical_model.joblib          # XGBoost on clinical features
        │   └── late_fusion_predictor.joblib   # Late Fusion wrapper with optimized weights
        ├── figures/
        │   ├── pso_convergence.png            # PSO fitness history
        │   ├── model_comparison.png           # Bar chart comparing 3 models
        │   ├── roc_comparison.png             # ROC curves overlay
        │   ├── execution_time.png             # Inference time comparison
        │   ├── final_roc_curve.png            # Final ROC for Late Fusion
        │   ├── final_confusion_matrix.png     # Confusion matrix
        │   ├── final_pr_curve.png             # Precision-Recall curve
        │   ├── final_calibration_plot.png     # Calibration plot
        │   ├── bootstrap_auc_distribution.png # Bootstrap AUC distribution
        │   └── SHAP plots (branch-specific)
        └── tables/
            ├── final_genomic_features.csv     # Selected genomic features (~40 genes)
            ├── final_clinical_features.csv    # Engineered clinical features (14)
            ├── pso_convergence_history.json   # PSO iteration history
            ├── comprehensive_evaluation.json  # All metrics in JSON
            ├── comprehensive_evaluation.csv   # All metrics in CSV
            └── biomarker rankings (branch-specific)
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

## 5. Feature Selection (Notebook 04)

### 5.1 Genomic Branch (`src/genomic_selector.py`)
- **Variance Threshold**: Remove low-variance genes (threshold=0.01)
- **Mutual Information**: Select top 200 genes most informative about BCR
- **Binary PSO**: Optimize gene subset (n_particles=20, n_iterations=15)
  - Fitness function: AUC - α × (n_features / total_features)
  - Penalty coefficient α=0.05 discourages large feature sets
  - Tracks `fitness_history` and `raw_auc_history` for convergence analysis
- **Output**: ~40 selected genes saved to `final_genomic_features.csv`

### 5.2 Clinical Branch (`src/clinical_engineer.py`)
- **Domain-specific engineered features**:
  1. `Gleason_Total`: Sum of primary + secondary Gleason patterns
  2. `High_Risk_Gleason`: Binary indicator (≥ 8)
  3. `Margin_x_LymphNode`: Interaction term
  4. `T_Stage_Risk`: Ordinal encoding of pathological T stage
  5. `PSA_Pathway`: Composite PSA-related feature
  6. `AR_Signaling`: Androgen receptor pathway score
  7. `Proliferation`: Cell proliferation marker
- **Output**: 14 engineered clinical features saved to `final_clinical_features.csv`

## 6. Model Training (Notebook 05)

### 6.1 Dual-Model Architecture
- **Genomic Model**: XGBoost trained on ~40 selected genes
  - Hyperparameters: max_depth=4, learning_rate=0.05, n_estimators=150, scale_pos_weight=6.4
- **Clinical Model**: XGBoost trained on 14 engineered clinical features
  - Same hyperparameter configuration for fair comparison

### 6.2 Late Fusion Layer (`src/fusion/late_fusion.py`)
- **LateFusionPredictor** class wraps both models
- **Weight Optimization**: Grid search over [0.0, 1.0] in 0.05 steps
  - Optimizes validation AUC
  - Default optimized weights: Genomic=0.61, Clinical=0.39
- **Fallback Support**: Can predict using genomic model only when clinical data is missing

### 6.3 Performance Comparison (Test Set, N=86)

| Model | AUC | Accuracy | F1-Score | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Genomic-only | 0.78 | 0.74 | 0.52 | 0.61 | 0.77 |
| Clinical-only | 0.72 | 0.71 | 0.45 | 0.53 | 0.75 |
| **Late Fusion** | **0.82** | **0.77** | **0.58** | **0.67** | **0.79** |

- Figures: `pso_convergence.png`, `model_comparison.png`, `roc_comparison.png`, `execution_time.png`

## 7. Explainability & Biomarker Discovery (Notebook 06)

### 7.1 Branch-Specific Feature Importance
- **Genomic Branch**: Top-ranked genes extracted from XGBoost `feature_importances_`
  - Example: **LRR1**, **DHX34**, **APEH**, **ACSM2A**, **TNIP3**, **PLAC8**, **ARMC4**, **VPS13A**
- **Clinical Branch**: Engineered feature importance
  - Top features: `Gleason_Total`, `High_Risk_Gleason`, `Margin_x_LymphNode`

### 7.2 Combined Importance (Late Fusion)
- Weighted combination: `combined_importance = w_genomic × genomic_imp + w_clinical × clinical_imp`
- Biomarker ranking split into gene and clinical tables

### 7.3 SHAP Analysis
- **Branch-specific SHAP**: Separate explanation for genomic and clinical predictions
- **Summary plot** (beeswarm): Global feature importance with directionality
- **Waterfall plot**: Per-patient explanation showing contribution of each feature
- **Dependence plots**: Feature effect on prediction probability
- Saved under `outputs/figures/`

## 8. Final Evaluation (Notebook 07)

### 8.1 Comprehensive Metrics
- **ROC Curve**: AUC with 95% CI (DeLong method)
- **Precision-Recall Curve**: AUPRC for imbalanced classification
- **Confusion Matrix**: TP, TN, FP, FN at optimal threshold
- **Calibration Plot**: Reliability diagram with Brier score
- **Bootstrap Analysis**: 1000 bootstrap samples for confidence intervals

### 8.2 Generated Artifacts
- `final_roc_curve.png`, `final_pr_curve.png`, `final_calibration_plot.png`
- `final_confusion_matrix.png`, `bootstrap_auc_distribution.png`
- `comprehensive_evaluation.json`, `comprehensive_evaluation.csv`

## 9. External Validation (Notebook 08)

### 9.1 Missing Data Handling
- **Fallback Strategy**: When clinical features are unavailable, use genomic model only
- **Graceful Degradation**: Maintains predictive capability with partial data

### 9.2 Validation Protocol
- Load external cohort data
- Apply same preprocessing pipeline
- Compare performance across cohorts
- Assess generalizability of Late Fusion approach

## 10. Setup & Reproduction

1. Create a virtual environment (project runs on Python 3.11+, `venv`).
2. Install dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `joblib`, `seaborn`.
3. Place raw TCGA files under `data/raw/`.
4. Run notebooks **in order**:
   - **Essential**: 01 → 02 → 03 (data preparation)
   - **Late Fusion**: 04 → 05 → 06 → 07 (modeling and evaluation)
   - **Optional**: 08 (external validation)
5. Collect artifacts from `outputs/{models,figures,tables}`.

## 11. Engineering Conventions

- **Separation of concerns:** notebooks orchestrate; `src/` modules own the logic; `config.py` owns all paths/constants.
- **Leakage hygiene:** dedicated leakage-audit step; target built from a single source column; ID-like/high-cardinality columns dropped.
- **Reproducibility:** centralized logging (`prostate_bcr` logger), validated paths (`validate_path`), and all figures/tables written via `save_figure` / `save_table`.
- **Late Fusion benefits:**
  - Modular branch development
  - Independent hyperparameter tuning per branch
  - Interpretable branch contributions
  - Robustness to missing modalities

## 12. Troubleshooting / Known Issues

| Symptom | Cause / Fix |
|---|---|
| `ValueError: Target source column 'Biochemical Recurrence Indicator' not found` | The source column was dropped before `create_target_column` (e.g., by leakage/cardinality filtering) or is absent from the raw file. Verify column presence and preprocessing order. |
| `FileNotFoundError: … clinical_data_complete.csv` | Notebook 01 did not complete the clinical save step; re-run Notebook 01 end-to-end before merging. |
| `FileNotFoundError: … genomic_model.joblib` | Notebook 05 (Late Fusion) must be run before loading models. Ensure you run `05_model_training_late_fusion.ipynb`, not the old `05_Model_Training.ipynb`. |
| `AttributeError: 'LateFusionPredictor' object has no attribute 'feature_importances_'` | Use `get_feature_importance()` method which extracts importance from both sub-models and combines them with fusion weights. |
| sklearn `FutureWarning` on `penalty` (LogisticRegression) | scikit-learn ≥ 1.8 deprecation; use `l1_ratio` / `C` instead. |
| PSO convergence not saved | Ensure `pso_feature_select_genomic()` returns history and save to `pso_convergence_history.json`. |

## 13. References

1. TCGA Research Network: https://www.cancer.gov/tcga
2. PRAD study: Abeshouse et al., "Comprehensive Molecular Characterization of Pheochromocytoma and Paraganglioma", Cancer Cell, 2017.
3. Particle Swarm Optimization: Kennedy & Eberhart, "Particle Swarm Optimization", IEEE ICNN, 1995.
4. SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS, 2017.


---
رای بالا بردن ارزیابی داخلی (Internal Evaluation) که در حال حاضر روی AUC حدود ۰.۸۲ تا ۰.۸۴ متوقف شده است، باید روی بهینه‌سازی مدل و داده‌ها تمرکز کنید. از آنجایی که داده‌های شما نسبتاً کوچک هستند (۳۴۳ نمونه)، خطر Overfitting وجود دارد، اما هنوز جای بهبود وجود دارد.
در اینجا ۵ راهکار عملی و فنی برای افزایش AUC داخلی آورده شده است:
۱. بهینه‌سازی دقیق‌تر هایپرپارامترها (Hyperparameter Tuning)
شما از Optuna استفاده می‌کنید که عالی است، اما ممکن است فضای جستجو (search_space) یا تعداد تریال‌ها (n_trials) کافی نباشد.
افزایش n_trials: اگر الان ۱۰۰ تریال دارید، آن را به ۲۰۰ یا ۳۰۰ افزایش دهید.
گسترش فضای جستجو: پارامترهای کلیدی XGBoost مثل max_depth، learning_rate، subsample و colsample_bytree را با بازه‌های وسیع‌تری جستجو کنید.
استفاده از Pruner هوشمند: مطمئن شوید از MedianPruner یا HyperbandPruner در Optuna استفاده می‌کنید تا زمان را روی مدل‌های ضعیف هدر ندهید.
۲. مهندسی ویژگی‌های تعاملی جدید (Feature Engineering)
مدل‌های درختی (مثل XGBoost) در کشف تعاملات غیرخطی خوب هستند، اما کمک کردن به آن‌ها با ساخت ویژگی‌های ترکیبی دستی می‌تواند معجزه کند.
تعاملات بالینی-ژنی: علاوه بر ویژگی‌های فعلی، نسبت‌های جدید بسازید. مثلاً:
PSA / AR_Signaling_Score
Gleason_Score * Proliferation_Index
Polynomial Features: برای ویژگی‌های بالینی مهم (مثل Gleason و Stage)، توان دوم یا سوم آن‌ها را اضافه کنید (با احتیاط برای جلوگیری از نویز).
۳. مدیریت عدم تعادل کلاس‌ها (Class Imbalance Handling)
نسبت کلاس‌های شما تقریباً ۱ به ۶ است (۴۶ مثبت در برابر ۲۹۷ منفی).
تنظیم scale_pos_weight: در حال حاضر این وزن به صورت خودکار محاسبه می‌شود. سعی کنید این مقدار را به عنوان یک پارامتر قابل تنظیم در Optuna وارد کنید تا مدل بهترین وزن را برای جریمه خطاهای کلاس اقلیت پیدا کند.
استفاده از eval_metric='aucpr': گاهی اوقات بهینه‌سازی مستقیم روی average_precision (PR-AUC) به جای logloss یا auc، منجر به مدل‌هایی می‌شود که در تشخیص کلاس اقلیت بهتر عمل می‌کنند و در نهایت AUC را هم بالا می‌برند.
۴. انتخاب ویژگی پویاتر (Dynamic Feature Selection)
شما از PSO برای انتخاب ۳۰-۴۵ ویژگی استفاده می‌کنید.
تست مقادیر مختلف K: شاید ۴۵ ویژگی زیاد باشد و نویز وارد کند، یا ۳۰ ویژگی کم باشد و اطلاعات از دست برود. یک حلقه بیرونی ایجاد کنید که مدل را با تعداد ویژگی‌های مختلف (مثلاً ۲۰، ۳۰، ۴۰، ۵۰) آموزش دهد و ببینید کدام‌یک بیشترین AUC را در اعتبارسنجی متقاطع (Cross-Validation) می‌دهد.
حذف ویژگی‌های همبسته‌ی قوی: قبل از ورود به PSO، مطمئن شوید که ویژگی‌هایی با همبستگی بالای ۰.۹۵ حذف شده‌اند. وجود دو ویژگی بسیار مشابه می‌تواند اهمیت هر دو را در مدل درختی کاهش دهد.
۵. استفاده از Ensemble Models (مدل‌های ترکیبی)
به جای تکیه صرف بر XGBoost، خروجی چند مدل را ترکیب کنید.
Stacking: یک مدل لایه اول شامل XGBoost، LightGBM و Random Forest باشد. سپس خروجی احتمال (Probability) این سه مدل را به عنوان ورودی به یک مدل ساده مثل Logistic Regression (لایه دوم) بدهید. این روش معمولاً ۱ تا ۳ درصد AUC را بهبود می‌بخشد.
پیشنهاد اجرایی فوری:
من پیشنهاد می‌کنم ابتدا راهکار شماره ۱ (بهینه‌سازی بیشتر) و راهکار شماره ۳ (تنظیم وزن کلاس‌ها) را امتحان کنید، زیرا کم‌هزینه‌ترین و سریع‌ترین تاثیر را دارند.
آیا می‌خواهید کد اصلاح شده‌ی بخش optimize_model در فایل optimization.py را برای جستجوی دقیق‌تر پارامترها برایتان بنویسم؟
---