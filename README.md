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
   - 🏥 **Clinical branch**: domain-informed engineered features (Gleason risk, stage risk, margin × lymph node)
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
│  • SHAP Summary         │    │  GSE54460 cohort        │
│  • Biomarker Ranking    │    │  Genomic-only fallback  │
└─────────────────────────┘    └─────────────────────────┘
```

## Key Results (Late Fusion)

> ⚠️ These are current single-split results from the Late Fusion pipeline (see `manuscript_draft.md` and `core/outputs/tables/final_evaluation.json`). Repeated nested CV from raw data is pending.

| Metric | Genomic | Clinical | Late Fusion |
|--------|--------:|---------:|------------:|
| Test ROC-AUC | 0.686 | 0.721 | **0.753** |
| OOF-fused test ROC-AUC (corrected) | — | — | **0.738** |
| Nested 5-fold OOF ROC-AUC | — | — | 0.861* |

`*` Computed on pre-selected feature artifacts; the definitive estimate must rerun selection from raw data inside every outer fold.

### Selected Features

- 🔬 **Genomic**: 40 genes by Binary PSO (from ~18,905 after variance filtering) + 3 pathway scores (`PSA_Pathway_Score`, `AR_Signaling_Score`, `Proliferation_Score`)
- 🏥 **Clinical**: engineered features (`Gleason_Total`, `High_Risk_Gleason`, `Margin_x_LymphNode`, `T_Stage_Risk`) plus selected clinical variables

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
├── data/external/             # External cohort GSE54460 (not in Git)
└── outputs/                   # Models, figures, tables (not in Git)
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