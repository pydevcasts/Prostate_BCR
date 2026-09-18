# TCGA-PRAD Biochemical Recurrence Prediction 🧬

> Multi-omics machine learning pipeline for predicting biochemical recurrence (BCR) after radical prostatectomy using TCGA-PRAD clinical + RNA-Seq data, with a **Late Fusion architecture** (parallel genomic & clinical branches), MI + Binary PSO feature selection, and SHAP-based biomarker discovery.

## 🌍 Documentation

| Language | File |
|---|---|
| 🇮🇷 **فارسی (Persian) — سند کامل مستندات** | [`DOCUMENTATION.fa.md`](DOCUMENTATION.fa.md) |
| 🇬🇧 English — manuscript draft | [`manuscript_draft.md`](manuscript_draft.md) |

The Persian documentation covers the full pipeline: architecture, every `src/` module, all 8 notebooks, evaluation metrics, leakage-prevention layers, and developer notes — written so any developer can understand what the code does.

## Overview

This project predicts **biochemical recurrence (BCR)** after radical prostatectomy by integrating TCGA-PRAD clinical features and RNA-Seq gene expression (~18,900 genes). The current architecture implements **Late Fusion**:

1. **Rigorous preprocessing** with leakage auditing
2. **Two parallel branches**:
   - 🔬 **Genomic branch**: Variance filter → Mutual Information → Binary PSO (40 genes)
   - 🏥 **Clinical branch**: domain-informed engineered features (Gleason risk, stage risk, margin × lymph node) — the four engineered features are currently part of the 40 clinically selected features; a pure 4-feature branch is a pending simplification
3. **Late Fusion layer**: weighted average of both branch probabilities, with weights estimated from **out-of-fold (OOF)** predictions
4. **SHAP-based explainability** with publication-ready biomarker ranking

## Pipeline Architecture

```
Raw Data (Clinical + RNA-Seq)
        │
        ▼
┌─────────────────────────┐
│   Preprocessing         │
│  • Imputation           │
│  • Log1p / Winsorize    │
│  • Leakage Audit        │
└──────────┬──────────────┘
           ▼
┌─────────────────────────────────────────┐
│      Feature Selection (Late Fusion)    │
│  ┌────────────────┐  ┌────────────────┐ │
│  │ Genomic Branch │  │Clinical Branch │ │
│  │ Variance→MI→   │  │ Engineered     │ │
│  │ Binary PSO     │  │ Features (MI+  │ │
│  │ (40 genes)     │  │ PSO optional)  │ │
│  └───────┬────────┘  └───────┬────────┘ │
└──────────┼───────────────────┼──────────┘
           ▼                   ▼
┌─────────────────────────────────────────┐
│           Model Training                │
│   XGBoost (genomic) + XGBoost (clinical)│
│   p_fusion = w·p_gen + (1-w)·p_clin     │
│   Weights & threshold from OOF only     │
└──────────┬──────────────────────────────┘
           ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│   Explainability        │    │  External Validation    │
│  • SHAP Summary         │    │  GSE54460 · MSKCC 2010  │
│  • Biomarker Ranking    │    │  Genomic-only fallback  │
└─────────────────────────┘    └─────────────────────────┘
```

## Key Results (Late Fusion)

> ⚠️ These are current single-split results from the Late Fusion pipeline (see `manuscript_draft.md` and `core/outputs/tables/final_evaluation.json`). Repeated nested CV from raw data is pending.

| Metric | Genomic | Clinical | Late Fusion |
|--------|--------:|---------:|------------:|
| Test ROC-AUC (legacy train-optimized weights) | 0.686 | 0.721 | 0.753 |
| **OOF-fused test ROC-AUC (reference, leakage-free)** | — | — | **0.738** |
| Nested 5-fold OOF ROC-AUC | — | — | 0.861* |

The **OOF-fused** row is the leakage-free reference result: fusion weights (genomic 0.48 / clinical 0.52) and the Youden threshold (~0.312) are estimated exclusively from out-of-fold predictions. The legacy 0.753 comes from the old train-optimized `optimize_weights` method (kept for comparison only). The nested 5-fold evaluation yields balanced accuracy 0.795 with OOF weights genomic 0.62 / clinical 0.38.

`*` Computed on pre-selected feature artifacts; the definitive estimate must rerun selection from raw data inside every outer fold.

### External results on MSKCC 2010 (five-step transferable pipeline)

A prespecified, leakage-free transfer pipeline (built by `core/src/mskcc_cohort.py`, `transferable_features.py`, `transferable_fusion.py`, `rank_transfer_fusion.py`, `clinical_value_diagnostics.py`) was applied **frozen** to the MSKCC 2010 cohort (131 primary-tumor samples, 27 recurrences, 20.6% event rate):

| Model | External ROC-AUC (MSKCC) |
|---|---:|
| Pure-clinical baseline (logistic regression, 3 features) | 0.695 |
| Transferable late fusion — raw-scaler transfer (Step 3) | 0.711 (CI95 0.579–0.831) |
| **Transferable late fusion — rank-based transfer (Step 4)** | **0.717 (CI95 0.588–0.830)** |
| Genomic branch alone (rank transfer) | 0.586 |

**Honest headline:** the fusion is statistically indistinguishable from the parsimonious clinical baseline (paired bootstrap p ≥ 0.34); the external signal is carried by the clinical branch, and the genomic branch — strong within TCGA (0.836) — did not transfer (documented limitation; see `manuscript_draft.md` §6.9–6.11). The rank-transfer fusion had the best calibration (Brier 0.145).

### Selected Features

- 🔬 **Genomic**: 40 genes by Binary PSO (from ~18,905 after variance filtering) + 3 pathway scores (`PSA_Pathway_Score`, `AR_Signaling_Score`, `Proliferation_Score`)
- 🏥 **Clinical**: engineered features (`Gleason_Total`, `High_Risk_Gleason`, `Margin_x_LymphNode`, `T_Stage_Risk`) plus selected clinical variables (40 selected features in current artifacts)

## External Validation

- **GSE54460** (done): genomic-only fallback (no clinical data available), 106 samples (55 BCR-positive), ROC-AUC = **0.560** — highlights cross-cohort transferability challenges.
- **MSKCC 2010** (done): the gold-standard prostatectomy cohort, validated with the five-step transferable pipeline above — external fusion ROC-AUC **0.717**, on par with the pure-clinical baseline (0.695). Cohort files live in `core/data/external/` (`mskcc_cohort.csv`, `mskcc_transferable_features.csv`, `mskcc_rank_features.csv`, `prad_mskcc.tar.gz` + extracted `prad_mskcc/`).

## Project Structure

```
core/
├── config.py                  # Central configuration (single source of truth)
├── build_external_validation.py  # Builds the GSE54460 external cohort
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_feature_selection_late_fusion.ipynb
│   ├── 05_model_training_late_fusion.ipynb
│   ├── 06_Explainability_LateFusion.ipynb
│   ├── 07_Final_Evaluation_LateFusion.ipynb
│   └── 08_external_validation_late_fusion.ipynb
├── src/
│   ├── io.py                  # I/O utilities & logging
│   ├── clinical.py            # Clinical preprocessing
│   ├── genomics.py            # RNA-Seq preprocessing
│   ├── merge.py               # Clinical-genomics merge
│   ├── leakage.py             # Leakage audit
│   ├── preprocessing.py       # sklearn pipelines (log1p, winsorize, log2)
│   ├── feature_selection.py   # Backward-compatible re-exports
│   ├── genomic_selector.py    # MI + Binary PSO on genes
│   ├── genomic_engineer.py    # Pathway scores (PSA/AR/Proliferation)
│   ├── clinical_selector.py   # Selection for the clinical branch
│   ├── clinical_engineer.py   # Domain-informed clinical features
│   ├── features_config.py     # Engineered-feature definitions
│   ├── clinical_benchmark.py  # Clinical feature-selection strategies
│   ├── mskcc_cohort.py             # Step 1: audited MSKCC 2010 cohort
│   ├── transferable_features.py    # Step 2: 6 transferable features + gates
│   ├── transferable_fusion.py      # Step 3: frozen raw-scaler transfer
│   ├── rank_transfer_fusion.py     # Step 4: frozen rank-based transfer
│   ├── clinical_value_diagnostics.py # Step 5: baseline, calibration, DCA
│   ├── models.py              # Model factories & registry (6 classifiers)
│   ├── pipeline.py            # Nested CV & model comparison
│   ├── evaluation.py          # Metrics + bootstrap CI
│   ├── explainability.py      # SHAP & biomarker ranking
│   ├── visualization.py       # Publication-quality figures
│   ├── validation.py          # Feature alignment & consistency checks
│   └── fusion/
│       ├── late_fusion.py     # LateFusionPredictor + OOF fusion
│       └── nested_evaluation.py  # Leakage-aware nested fusion eval
├── data/raw/                  # Raw TCGA data (not in Git)
├── data/processed/            # Processed datasets (not in Git)
├── data/external/             # External cohorts (tracked: MSKCC + GSE54460 artifacts)
└── outputs/                   # Models, figures, tables (result tables/figures tracked)
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r ./requirements.txt

# Run pipeline
jupyter notebook notebooks/01_Data_Preparation.ipynb
```

### Notebook execution order

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

Each notebook reads the artifacts produced by the previous ones (`core/data/processed/` and `core/outputs/`).

## Leakage Prevention

All feature selection and preprocessing are fitted **exclusively on training folds** within nested cross-validation:

- ✅ Feature selection fitted only on training folds
- ✅ PSO fitness evaluated via inner CV on training data only
- ✅ Test data never accessed during training or selection
- ✅ All preprocessing fitted on training data only
- ✅ Fusion weights and the classification threshold estimated from **out-of-fold predictions** (never from training predictions or the test set)
- ✅ External missing genes are never imputed/fabricated (`align_common_features` keeps only truly measured features)

## Requirements

- Python ≥ 3.11
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- shap, matplotlib, seaborn, joblib

## License

MIT