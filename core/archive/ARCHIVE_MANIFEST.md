
# Archive Manifest

## Archive Date: 2024-09-11

## Summary
This archive contains obsolete files that have been moved to preserve project history while maintaining a clean working directory.

**Important**: Notebooks 01, 02, and 03 are **ESSENTIAL** and have been PRESERVED:
- `01_Data_Preparation.ipynb` → Prepares raw TCGA data
- `02_EDA.ipynb` → Exploratory data analysis
- `03_Preprocessing.ipynb` → Produces preprocessed train/test data

## Archived Files

### Directories Removed
| Original Path | Reason |
|--------------|--------|
| `/workspace/core/__pycache__/` | Python bytecode cache - auto-generated |
| `/workspace/core/src/__pycache__/` | Python bytecode cache - auto-generated |
| `/workspace/core/src/fusion/__pycache__/` | Python bytecode cache - auto-generated |

### Files Preserved (NOT Archived)
The following files remain in their original locations as they are essential:

#### Essential Notebooks (DO NOT DELETE)
- `notebooks/01_Data_Preparation.ipynb`
- `notebooks/02_EDA.ipynb`
- `notebooks/03_Preprocessing.ipynb`
- `notebooks/04_feature_selection.ipynb`
- `notebooks/05_Model_Training.ipynb`
- `notebooks/06_Explainability.ipynb`
- `notebooks/07_Final_Evaluation.ipynb`
- `notebooks/08_External _Evaluation.ipynb`

#### Late Fusion Notebooks (New)
- `notebooks/06_Explainability_LateFusion.ipynb`
- `notebooks/07_Final_Evaluation_LateFusion.ipynb`

#### Active Models
- `outputs/models/genomic_model.joblib`
- `outputs/models/clinical_model.joblib`
- `outputs/models/late_fusion_predictor.joblib`

#### Active Tables
- `outputs/tables/final_genomic_features.csv`
- `outputs/tables/final_clinical_features.csv`
- All other existing tables

## Archive Statistics
- **Files archived**: 0 (no truly obsolete files found)
- **Directories removed**: 3 (`__pycache__` directories)
- **Notebooks preserved**: All notebooks 01-08 plus new Late Fusion notebooks

## Notes
- No files with `_old`, `_backup`, or `_deprecated` suffixes were found
- No duplicate notebooks requiring archival were identified
- All model files are current and in use
- The archive directories have been created for future use if needed
