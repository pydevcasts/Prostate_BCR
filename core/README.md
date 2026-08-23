# Interpretable Prediction of Biochemical Recurrence in Prostate Cancer using PSO-Optimized Gene Signatures and Hybrid Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository contains the complete codebase for our machine learning pipeline that predicts biochemical recurrence (BCR) in prostate cancer patients using gene expression data from TCGA-PRAD (training) and GSE70769 (external validation) cohorts.

### Key Features

- **Hybrid Feature Selection**: Variance Threshold → Mutual Information → Binary PSO
- **Multiple Classifiers**: XGBoost, Logistic Regression, Random Forest, SVM, LightGBM, CatBoost
- **Explainability**: SHAP-based feature importance interpretation
- **Clinical Utility**: Decision Curve Analysis, Risk Stratification
- **Survival Analysis**: Kaplan-Meier curves, Log-rank tests, Time-dependent ROC
- **Reproducible Research**: Fixed random seeds, leakage-free pipeline

### Performance Metrics

| Cohort | AUC | 95% CI |
|--------|-----|--------|
| Internal Test (TCGA-PRAD) | ~0.82 | [0.75-0.89] |
| External Validation (GSE70769) | ~0.61 | [0.48-0.74] |

## Repository Structure

```
prostate_bcr_prediction/
├── config.py                    # Global configuration (paths, seeds, hyperparameters)
├── src/
│   ├── __init__.py
│   ├── clinical_utility.py      # Decision Curve Analysis, confusion matrix with CI
│   ├── evaluation.py            # Model evaluation metrics (AUC, F1, MCC, etc.)
│   ├── explainability.py        # SHAP analysis
│   ├── feature_selection.py     # Variance, MI, Binary PSO feature selection
│   ├── features_config.py       # Gene sets for pathway scores
│   ├── genomics.py              # Genomic data processing
│   ├── io.py                    # I/O utilities
│   ├── leakage.py               # Data leakage detection and prevention
│   ├── merge.py                 # Data merging utilities
│   ├── models.py                # Model factories and hyperparameter tuning
│   ├── pipeline.py              # Main ML pipeline orchestration
│   ├── preprocessing.py         # Data preprocessing and normalization
│   ├── survival_analysis.py     # Kaplan-Meier, log-rank test, C-index
│   └── visualization.py         # Plotting utilities
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Explainability.ipynb
│   ├── 07_Final_Evaluation.ipynb
│   └── 08_External_Evaluation.ipynb
├── data/
│   ├── raw/                     # Raw data files (not included)
│   ├── interim/                 # Intermediate processed data
│   └── processed/               # Final processed datasets
├── outputs/
│   ├── figures/                 # Generated plots
│   ├── tables/                  # Results tables
│   └── models/                  # Saved model artifacts
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/prostate_bcr_prediction.git
cd prostate_bcr_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install as editable package
pip install -e .
```

## Usage

### Quick Start

The analysis pipeline is organized into sequential Jupyter notebooks:

```bash
# Navigate to notebooks directory
cd notebooks

# Run notebooks in order:
# 1. Data preparation
# 2. Exploratory data analysis
# 3. Preprocessing
# 4. Feature selection
# 5. Model training
# 6. Explainability (SHAP)
# 7. Final evaluation
# 8. External validation
```

### Programmatic Usage

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import config
from src.pipeline import evaluate_final_model
from src.feature_selection import run_feature_selection
from src.models import build_model
from src.clinical_utility import decision_curve_analysis
from src.survival_analysis import kaplan_meier_by_risk_group

# Load preprocessed data
X_train = pd.read_csv(config.PROCESSED_DIR / "X_train_preprocessed.csv")
y_train = pd.read_csv(config.PROCESSED_DIR / "y_train.csv").iloc[:, 0]

# Run feature selection
selector, selected_features = run_feature_selection(
    X_train, y_train,
    variance_threshold=config.VARIANCE_THRESHOLD,
    mi_top_k=config.MI_TOP_K,
    pso_final_k=config.PSO_FINAL_K,
    run_pso=True,
    random_state=config.RANDOM_STATE,
)

# Build and train model
model = build_model("XGBoost", y_train=y_train)
X_selected = X_train[selected_features]
model.fit(X_selected, y_train)

# Evaluate
results = evaluate_final_model(model, X_test, y_test, selected_features)

# Clinical utility analysis
dca_results = decision_curve_analysis(y_test, y_prob_test)

# Survival analysis (if time-to-event data available)
km_results = kaplan_meier_by_risk_group(
    event_times, event_observed, risk_scores, strategy="median"
)
```

## Configuration

All hyperparameters and paths are defined in `config.py`:

```python
# Feature selection
VARIANCE_THRESHOLD = 0.01
MI_TOP_K = 200
PSO_FINAL_K = 40

# PSO parameters
PSO_N_PARTICLES = 12
PSO_N_ITERATIONS = 10
PSO_PENALTY_ALPHA = 0.001

# Cross-validation
OUTER_SPLITS = 5
INNER_SPLITS = 3

# Reproducibility
RANDOM_STATE = 42
```

## Methodology

### Feature Selection Pipeline

1. **Variance Threshold**: Remove near-constant features (threshold=0.01)
2. **Mutual Information**: Rank features by relevance to target (top-k=200)
3. **Feature Engineering**: Create pathway scores (PSA, AR, Proliferation)
4. **Binary PSO**: Wrapper selection with inner CV fitness evaluation

### Classification Models

- XGBoost (primary model with hyperparameter tuning)
- Logistic Regression (baseline)
- Random Forest
- Support Vector Machine (RBF kernel)
- LightGBM
- CatBoost

### Explainability

- SHAP (SHapley Additive exPlanations) values for global and local interpretability
- Feature importance ranking
- Dependence plots

### Clinical Utility

- **Decision Curve Analysis (DCA)**: Net benefit across threshold probabilities
- **Confusion Matrix with Confidence Intervals**: Bootstrap-based uncertainty estimation
- **Risk Stratification**: Median/tercile/quartile-based patient grouping

### External Validation

- Probe-to-gene mapping for microarray data (GSE70769)
- Common gene intersection approach
- Batch effect considerations

## Results

### Internal Validation (TCGA-PRAD)

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.82 | [0.75-0.89] |
| PR-AUC | 0.54 | [0.42-0.66] |
| Sensitivity | 0.71 | [0.58-0.82] |
| Specificity | 0.79 | [0.73-0.84] |
| F1 Score | 0.38 | [0.28-0.48] |
| MCC | 0.35 | [0.24-0.46] |

### External Validation (GSE70769)

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC (common genes) | 0.61 | [0.48-0.74] |
| Number of common genes | 31 | - |

## Reproducibility

To ensure reproducibility:

1. All random seeds are fixed (`RANDOM_STATE = 42`)
2. Feature selection is performed inside cross-validation folds
3. No data leakage from test set during preprocessing
4. Complete dependency list in `requirements.txt`
5. Version control for all code changes

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2024,
  title={Interpretable Prediction of Biochemical Recurrence in Prostate Cancer using PSO-Optimized Gene Signatures and Hybrid Machine Learning},
  author={Your Name and Collaborators},
  journal={Bioinformatics},
  year={2024},
  volume={},
  number={},
  pages={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please contact:
- Email: your.email@institution.edu
- GitHub Issues: [Open an issue](https://github.com/yourusername/prostate_bcr_prediction/issues)

## Acknowledgments

- TCGA Research Network: https://www.cancer.gov/tcga
- GEO Database: https://www.ncbi.nlm.nih.gov/geo/
- SHAP Library: https://github.com/slundberg/shap
- scikit-learn: https://scikit-learn.org/
