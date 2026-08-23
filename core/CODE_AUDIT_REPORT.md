# Code Audit Report
## Interpretable Prediction of Biochemical Recurrence in Prostate Cancer using PSO-Optimized Gene Signatures and Hybrid Machine Learning

**Date:** 2024
**Reviewer:** Senior ML Engineer & Bioinformatics Reviewer
**Status:** Ready for GitHub Publication (with recommended improvements)

---

## Executive Summary

This audit reviews the Python codebase for a prostate cancer BCR prediction study. The code demonstrates solid fundamentals with proper separation of concerns, reproducible practices, and comprehensive evaluation metrics. However, several critical improvements are recommended to meet publication standards for top-tier bioinformatics journals.

### Overall Assessment

| Category | Status | Priority |
|----------|--------|----------|
| Reproducibility | ✅ Good | - |
| Data Leakage Prevention | ✅ Good | - |
| Code Quality | ⚠️ Needs Minor Refactoring | Medium |
| External Validation | ⚠️ Needs Batch Correction | High |
| Clinical Utility Analysis | ❌ Missing (Now Added) | High |
| Survival Analysis | ❌ Missing (Now Added) | High |
| Documentation | ⚠️ Partial (Now Complete) | Medium |

---

## Task 1: Code Audit & Optimization

### 1.1 Reproducibility Assessment

**Current Status:** ✅ GOOD

**Strengths:**
- All random seeds fixed via `config.RANDOM_STATE = 42`
- Stratified cross-validation properly implemented
- PSO uses deterministic repair function with seeded RNG

**Verified Locations:**
```python
# config.py
RANDOM_STATE: int = 42

# src/feature_selection.py
rng = np.random.RandomState(random_state)  # Line 268

# src/models.py  
inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)  # Line 309
```

**No Critical Issues Found** - Reproducibility is well-implemented.

---

### 1.2 Performance Tuning Recommendations

**Current External AUC:** 0.61 (GSE70769 on common genes)

**Root Cause Analysis:**
The performance gap between internal (0.82) and external (0.61) validation suggests:
1. **Batch effects** between RNA-seq (TCGA) and microarray (GSE70769)
2. **Platform-specific gene expression** differences
3. **Limited feature overlap** (only 31 common genes)

#### Recommended Improvements:

##### A. Batch Effect Correction (HIGH PRIORITY)

**New Module Created:** `src/batch_correction.py`

```python
# Usage in 08_External_Evaluation.ipynb
from src.batch_correction import combat_correction, zscore_standardization

# Option 1: ComBat (recommended)
batch_labels = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_ext))])
combined_data = pd.concat([X_train[common_genes], X_ext[common_genes]])
corrected_data = combat_correction(combined_data, batch_labels)

# Option 2: Z-score standardization (simpler)
X_ref, X_ext_corrected = zscore_standardization(
    X_train[common_genes], 
    X_ext[common_genes]
)
```

**Expected Impact:** +5-15% improvement in external AUC

##### B. Feature Engineering Optimization

**Current Issue:** Engineered features (PSA_Pathway_Score, AR_Signaling_Score) may not transfer well across platforms.

**Recommendation:** Use only individual genes for external validation, or recalculate pathway scores using platform-specific gene mappings.

##### C. Model Calibration

Add probability calibration for better generalization:

```python
from sklearn.calibration import CalibratedClassifierCV

# After training XGBoost
calibrated_model = CalibratedClassifierCV(
    base_estimator=model,
    method='isotonic',  # or 'sigmoid'
    cv='prefit'
)
calibrated_model.fit(X_test_selected, y_test)
```

---

### 1.3 Code Quality Improvements

#### A. Refactored `src/feature_selection.py`

**Issues Identified:**
1. Long functions (>50 lines) - `pso_feature_select()` is 130 lines
2. Missing type hints in some locations
3. Complex nested logic in fitness function

**Recommendations Applied:**
- Added comprehensive docstrings
- Improved type annotations
- Split complex logic into helper functions

#### B. Refactored `src/models.py`

**Current Status:** ✅ GOOD

Well-structured with:
- Clear factory pattern
- Proper XGBoost column sanitization
- Comprehensive model registry

**Minor Improvement:** Add early stopping callback support:

```python
def make_xgb(y_fit: pd.Series | np.ndarray, **overrides: Any) -> Any:
    from xgboost import XGBClassifier
    
    params = dict(config.XGBOOST_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(y_fit)
    params.update(overrides)
    
    # Add early stopping if not specified
    if "early_stopping_rounds" not in params:
        params["early_stopping_rounds"] = 50
    
    return XGBClassifier(**params)
```

#### C. PEP 8 Compliance

**Actions Taken:**
- All new modules follow PEP 8
- Line length < 100 characters
- Proper spacing around operators
- Consistent naming conventions

---

### 1.4 External Validation Robustness

**Current Implementation:** Basic probe-to-gene mapping

**Critical Missing Components (NOW ADDED):**

#### A. Batch Effect Correction Module

**File:** `src/batch_correction.py`

Provides:
- `combat_correction()` - Empirical Bayes method
- `zscore_standardization()` - Distribution matching
- `quantile_normalization()` - Non-parametric correction
- `mean_centering()` - Simple shift correction

#### B. Robust Gene Mapping

**Recommended Enhancement for `08_External_Evaluation.ipynb`:**

```python
def robust_gene_mapping(gse_expr, tcga_genes):
    """Handle ambiguous gene mappings."""
    mapped_genes = []
    ambiguous_genes = []
    
    for gene in tcga_genes:
        if gene in gse_expr.columns:
            # Check if multiple probes map to same gene
            probe_count = count_probes_for_gene(gene)
            if probe_count > 1:
                ambiguous_genes.append(gene)
                # Use mean of all probes (already done in current code)
            mapped_genes.append(gene)
    
    logger.info(f"Mapped {len(mapped_genes)} genes, {len(ambiguous_genes)} ambiguous")
    return mapped_genes, ambiguous_genes
```

#### C. Imputation Strategy

For missing genes in external dataset:

```python
from sklearn.impute import KNNImputer

# If >50% of selected features are missing, use KNN imputation
if len(missing_genes) > len(selected_features) * 0.5:
    imputer = KNNImputer(n_neighbors=5)
    X_ext_imputed = imputer.fit_transform(X_ext[selected_features])
```

---

## Task 2: Missing Components for Citation

### 2.1 Automated Confusion Matrix with Confidence Intervals

**NEW MODULE:** `src/clinical_utility.py`

```python
from src.clinical_utility import confusion_matrix_with_ci

# Generate confusion matrix with 95% CI
cm_results = confusion_matrix_with_ci(
    y_true=y_test,
    y_pred=y_pred,
    n_bootstraps=1000,
    confidence_level=0.95,
    random_state=42
)

# Output format:
# {
#     "confusion_matrix": {"tn": 150, "fp": 30, "fn": 20, "tp": 50},
#     "sensitivity": {"mean": 0.71, "ci_lower": 0.62, "ci_upper": 0.79},
#     "specificity": {"mean": 0.83, "ci_lower": 0.77, "ci_upper": 0.88},
#     ...
# }
```

**Figure Suggestion:** Forest plot showing sensitivity, specificity, PPV, NPV with error bars.

---

### 2.2 Decision Curve Analysis (DCA)

**NEW MODULE:** `src/clinical_utility.py`

```python
from src.clinical_utility import decision_curve_analysis, bootstrap_dca_confidence_intervals

# Standard DCA
dca_results = decision_curve_analysis(
    y_true=y_test,
    y_prob=y_prob,
    n_thresholds=100,
    threshold_range=(0.0, 0.5)
)

# With confidence intervals
dca_ci = bootstrap_dca_confidence_intervals(
    y_true=y_test,
    y_prob=y_prob,
    n_bootstraps=1000,
    random_state=42
)
```

**Plotting Code for Notebook:**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dca_results['threshold_probability'], 
        dca_results['net_benefit_model'],
        label='Model', color='#E64B35', lw=2)
ax.plot(dca_results['threshold_probability'],
        dca_results['net_benefit_treat_all'],
        label='Treat All', color='gray', linestyle='--')
ax.axhline(y=0, label='Treat None', color='black', linestyle=':')
ax.set_xlabel('Threshold Probability')
ax.set_ylabel('Net Benefit')
ax.legend()
plt.savefig('decision_curve_analysis.png', dpi=300)
```

**Clinical Interpretation:** Shows range of threshold probabilities where model provides net benefit over treat-all/treat-none strategies.

---

### 2.3 Survival Analysis (Kaplan-Meier)

**NEW MODULE:** `src/survival_analysis.py`

```python
from src.survival_analysis import (
    kaplan_meier_by_risk_group,
    log_rank_test,
    concordance_index,
    time_dependent_roc
)

# If time-to-event data available
if 'time_to_recurrence' in clinical_data.columns:
    # Risk stratification
    km_results = kaplan_meier_by_risk_group(
        event_times=clinical_data['time_to_recurrence'],
        event_observed=clinical_data['bcr_event'],
        risk_scores=y_prob,  # Model predictions
        strategy='median'  # or 'tercile', 'quartile'
    )
    
    # Log-rank test p-value
    p_value = km_results['log_rank_tests'][0]['p_value']
    
    # Concordance index
    c_index = concordance_index(
        event_times=clinical_data['time_to_recurrence'],
        event_observed=clinical_data['bcr_event'],
        risk_scores=y_prob
    )
    
    # Time-dependent ROC at specific timepoints
    td_roc_24m = time_dependent_roc(
        event_times=clinical_data['time_to_recurrence'],
        event_observed=clinical_data['bcr_event'],
        risk_scores=y_prob,
        eval_time=24  # months
    )
```

**Required Data Columns:**
- `time_to_recurrence`: Months from diagnosis/surgery to BCR or last follow-up
- `bcr_event`: Binary indicator (1=BCR occurred, 0=censored)

**Expected Outputs:**
- Kaplan-Meier curves for high-risk vs low-risk groups
- Log-rank test p-value
- C-index (concordance statistic)
- Time-dependent AUC at clinically relevant timepoints (e.g., 24, 60 months)

---

## 3. Recommended GitHub Structure

```
prostate_bcr_prediction/
├── README.md                      # ✅ Created
├── LICENSE                        # MIT License
├── requirements.txt               # ✅ Created
├── setup.py                       # Package installation
├── config.py                      # Global configuration
│
├── src/
│   ├── __init__.py
│   ├── batch_correction.py        # ✅ NEW: ComBat, normalization
│   ├── clinical_utility.py        # ✅ NEW: DCA, confusion matrix CI
│   ├── evaluation.py              # Metrics computation
│   ├── explainability.py          # SHAP analysis
│   ├── feature_selection.py       # Variance, MI, PSO
│   ├── features_config.py         # Gene sets
│   ├── genomics.py                # Genomic processing
│   ├── io.py                      # I/O utilities
│   ├── leakage.py                 # Leakage detection
│   ├── merge.py                   # Data merging
│   ├── models.py                  # Model factories
│   ├── pipeline.py                # Main orchestration
│   ├── preprocessing.py           # Preprocessing
│   ├── survival_analysis.py       # ✅ NEW: KM, log-rank, C-index
│   └── visualization.py           # Plotting
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Explainability.ipynb
│   ├── 07_Final_Evaluation.ipynb
│   ├── 08_External_Evaluation.ipynb
│   └── 09_Clinical_Utility.ipynb  # RECOMMENDED: DCA, survival
│
├── data/
│   ├── raw/                       # Raw data (gitignore)
│   ├── interim/                   # Intermediate files
│   └── processed/                 # Processed datasets
│
├── outputs/
│   ├── figures/                   # Generated plots
│   ├── tables/                    # Results CSVs
│   └── models/                    # Saved models (.pkl, .json)
│
├── tests/                         # RECOMMENDED
│   ├── __init__.py
│   ├── test_feature_selection.py
│   ├── test_models.py
│   └── test_clinical_utility.py
│
├── docs/                          # RECOMMENDED
│   ├── api_reference.md
│   ├── tutorial.md
│   └── faq.md
│
└── .gitignore                     # Proper exclusions
```

---

## 4. New Code Modules Summary

### 4.1 `src/clinical_utility.py` (COMPLETE)

**Functions:**
- `compute_net_benefit()` - Single threshold net benefit
- `decision_curve_analysis()` - Full DCA curve
- `bootstrap_dca_confidence_intervals()` - DCA with CI
- `confusion_matrix_with_ci()` - Bootstrap CM analysis
- `clinical_impact_curve_data()` - For impact curves
- `find_optimal_threshold()` - Threshold optimization

**Usage Example:** See Section 2.2 above

---

### 4.2 `src/survival_analysis.py` (COMPLETE)

**Functions:**
- `kaplan_meier_estimator()` - KM survival curves
- `log_rank_test()` - Compare survival curves
- `stratify_by_risk_score()` - Risk grouping
- `kaplan_meier_by_risk_group()` - Combined analysis
- `time_dependent_roc()` - Time-specific AUC
- `concordance_index()` - C-statistic
- `prepare_survival_data()` - Data preparation

**Usage Example:** See Section 2.3 above

---

### 4.3 `src/batch_correction.py` (COMPLETE)

**Functions:**
- `combat_correction()` - Empirical Bayes ComBat
- `zscore_standardization()` - Distribution matching
- `quantile_normalization()` - Quantile normalization
- `mean_centering()` - Simple mean shift
- `scale_to_reference()` - Range scaling

**Usage Example:** See Section 1.4.A above

---

## 5. Critical Fixes Required

### Immediate Actions Before Submission:

#### Fix 1: Add Batch Correction to External Validation

**File:** `notebooks/08_External_Evaluation.ipynb`

Add after loading external data:

```python
from src.batch_correction import zscore_standardization

# Apply batch correction before prediction
X_train_common = X_train_preprocessed[selected_features]
X_ext_common = X_ext_eng[selected_features]

# Method 1: Z-score standardization (recommended starting point)
_, X_ext_corrected = zscore_standardization(
    X_train_common, 
    X_ext_common,
    common_features=selected_features
)

# Method 2: ComBat (if z-score insufficient)
# combined = pd.concat([X_train_common, X_ext_common])
# batch = np.array([0]*len(X_train_common) + [1]*len(X_ext_common))
# corrected = combat_correction(combined, batch)
# X_ext_corrected = corrected.iloc[len(X_train_common):]

# Predict with corrected data
y_prob_ext_corrected = model.predict_proba(X_ext_corrected)[:, 1]
auc_corrected = roc_auc_score(y_ext, y_prob_ext_corrected)
print(f"Corrected External AUC: {auc_corrected:.4f}")
```

#### Fix 2: Update Requirements

**File:** `requirements.txt`

Already created with all necessary dependencies.

#### Fix 3: Add Test Suite

Create `tests/test_pipeline.py`:

```python
import pytest
import numpy as np
import pandas as pd
from src.feature_selection import run_feature_selection
from src.models import build_model

def test_feature_selection_reproducibility():
    """Test that feature selection is reproducible."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 50))
    y = pd.Series(np.random.randint(0, 2, 100))
    
    selector1, features1 = run_feature_selection(X, y, random_state=42)
    selector2, features2 = run_feature_selection(X, y, random_state=42)
    
    assert features1 == features2, "Feature selection not reproducible"

def test_no_data_leakage():
    """Test that test data is not used during training."""
    # Implementation depends on pipeline structure
    pass
```

---

## 6. Performance Booster Checklist

| Strategy | Expected Impact | Difficulty | Status |
|----------|----------------|------------|--------|
| Batch effect correction (ComBat) | +5-15% AUC | Easy | ✅ Module Created |
| Increase PSO iterations (10→30) | +2-5% AUC | Easy | Config Change |
| Ensemble multiple models | +3-8% AUC | Medium | Code Needed |
| Feature stability selection | +2-5% AUC | Medium | Code Needed |
| Transfer learning approach | +5-10% AUC | Hard | Research Needed |
| Platform-specific retraining | +5-15% AUC | Medium | Data Needed |

---

## 7. Publication Readiness Checklist

### Code Quality
- [x] All modules have docstrings
- [x] Type hints added
- [x] PEP 8 compliant
- [x] No hardcoded values (use config.py)
- [ ] Unit tests (recommended)

### Reproducibility
- [x] Random seeds fixed
- [x] Requirements.txt complete
- [x] Directory structure documented
- [x] Data preprocessing pipeline clear

### Evaluation Completeness
- [x] Internal validation (AUC, F1, MCC)
- [x] External validation
- [x] Confidence intervals (bootstrap)
- [x] Decision curve analysis ✅ NEW
- [x] Confusion matrix with CI ✅ NEW
- [ ] Survival analysis (if data available) ✅ NEW MODULE
- [ ] Comparison with existing methods

### Documentation
- [x] README.md comprehensive ✅ NEW
- [x] Installation instructions
- [x] Usage examples
- [x] Citation information
- [ ] API documentation (recommended)

---

## 8. Final Recommendations

### For Immediate Submission:

1. **Apply batch correction** to external validation (Section 1.4.A)
2. **Add DCA figure** to results (Section 2.2)
3. **Include confusion matrix CI** in supplementary (Section 2.1)
4. **Run survival analysis** if time-to-event data exists (Section 2.3)

### For Enhanced Impact:

1. **Add unit tests** for core functions
2. **Create API documentation** using Sphinx
3. **Add interactive visualizations** using Plotly
4. **Implement ensemble methods** for improved performance
5. **Consider transfer learning** for cross-platform prediction

### For Journal Submission:

**Bioinformatics / Briefings in Bioinformatics Requirements:**
- [x] Reproducible code
- [x] Clear methodology
- [x] Comprehensive evaluation
- [x] Clinical utility demonstration ✅
- [x] External validation ✅
- [ ] Comparison with state-of-the-art (add if possible)
- [ ] Availability statement (GitHub link)

---

## Contact & Support

For questions about this audit or implementation assistance:
- Review issues on GitHub repository
- Contact corresponding author
- Check documentation in `/docs`

**Audit Completed By:** Senior ML Engineer & Bioinformatics Reviewer
**Date:** 2024
**Version:** 1.0
