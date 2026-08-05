# TCGA-PRAD Biochemical Recurrence Prediction

> Multi-omics machine learning pipeline for predicting biochemical recurrence (BCR) after radical prostatectomy using TCGA-PRAD clinical + RNA-Seq data, with PSO-based feature selection and SHAP-based biomarker discovery.

## Overview

This project predicts **biochemical recurrence (BCR)** after radical prostatectomy by integrating TCGA-PRAD clinical features and RNA-Seq gene expression (~18,900 genes). The pipeline implements:

1. **Rigorous preprocessing** with leakage auditing
2. **Two-stage feature selection**: Mutual Information → Binary PSO
3. **Nested cross-validation** model comparison across 6 classifiers
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
┌─────────────────────────┐
│   Feature Selection     │
│  • Variance Threshold   │
│  • Mutual Information   │
│  • Binary PSO (30 feat) │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│   Model Training        │
│  • Nested 5-Fold CV     │
│  • 6-classifier compare │
│  • Best: XGBoost       │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│   Explainability        │
│  • Feature Importance   │
│  • SHAP Summary/Waterfall│
│  • Biomarker Ranking    │
└─────────────────────────┘
```

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | XGBoost |
| Nested CV AUC | 0.856 ± 0.053 |
| Selected Features | 30 (PSO-selected from 18,985) |
| Test ROC-AUC | 0.818 (CI: 0.70–0.92) |
| Test Sensitivity | 83.3% |
| Balanced Accuracy | 76.4% |

### Top Biomarkers Identified

| Rank | Gene | Importance |
|------|------|------------|
| 1 | DYNLT1 | 0.074 |
| 2 | POU2AF1 | 0.066 |
| 3 | SOCS2 | 0.058 |
| 4 | CNTRL | 0.040 |
| 5 | HSD11B1L | 0.035 |

## Project Structure

```
core/
├── config.py                  # Central configuration
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 05_Model_Training.ipynb
│   └── 06_Explainability.ipynb
├── src/
│   ├── io.py                  # I/O utilities & logging
│   ├── clinical.py            # Clinical preprocessing
│   ├── genomics.py            # RNA-Seq preprocessing
│   ├── merge.py               # Clinical-genomics merge
│   ├── leakage.py             # Leakage audit
│   ├── feature_selection.py   # MI + Binary PSO
│   ├── models.py              # Model factories & registry
│   ├── pipeline.py            # Nested CV & evaluation
│   ├── visualization.py       # Plotting utilities
│   └── explainability.py      # SHAP & biomarker ranking
├── data/raw/                  # Raw TCGA data (not in Git)
├── data/processed/            # Processed datasets (not in Git)
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

## Leakage Prevention

All feature selection and preprocessing are fitted **exclusively on training folds** within nested cross-validation:

- ✅ Feature selection fitted only on training folds
- ✅ PSO fitness evaluated via inner CV on training data only
- ✅ Test data never accessed during training or selection
- ✅ All preprocessing fitted on training data only

## Requirements

- Python ≥ 3.11
- pandas, numpy, scikit-learn ≥ 1.8
- xgboost, lightgbm, catboost
- shap, matplotlib, joblib

## License

MIT