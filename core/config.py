# config.py
"""
Project configuration for Prostate BCR Prediction (Q1 publication).

Single source of truth for paths, seeds, and hyperparameters.
Every module imports from here — no magic numbers elsewhere.

NOTE (reproducibility): The values below are the STABLE configuration
that produced the reported results (Test ROC-AUC ≈ 0.84). Hyperparameter
tuning is handled automatically by RandomizedSearchCV via
XGBOOST_SEARCH_SPACE — do not hand-edit defaults to chase results.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

# Ensure output directories exist
for _dir in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
             FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw data filenames (adjust if your files are named differently)
# ---------------------------------------------------------------------------
CLINICAL_RAW = RAW_DIR / "data_clinical_patient.tsv"
RNA_SEQ_RAW = RAW_DIR / "data_mrna_seq_v2_rsem.txt"
CLINICAL_PROCESSED = PROCESSED_DIR / "clinical_data_complete.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
N_JOBS: int = -1

# ---------------------------------------------------------------------------
# Train / Test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
OUTER_SPLITS: int = 5
INNER_SPLITS: int = 3
PSO_INNER_SPLITS: int = 3

# ---------------------------------------------------------------------------
# Feature selection — STABLE values (tested; do not chase results)
# ---------------------------------------------------------------------------
VARIANCE_THRESHOLD: float = 0.01
MI_TOP_K: int = 200
BORUTA_PERCENTILE: int = 95
PSO_FINAL_K: int = 30
PSO_N_PARTICLES: int = 12
PSO_N_ITERATIONS: int = 10
PSO_W: float = 0.7        # inertia weight
PSO_C1: float = 1.5       # cognitive
PSO_C2: float = 1.5       # social

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MISSING_THRESHOLD: float = 0.30   # drop column if >30 % missing
LOW_VARIANCE_THRESHOLD: float = 0.01

# ---------------------------------------------------------------------------
# Leakage audit — columns known to leak target information
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS: list[str] = [
    # --- From clinic.ipynb Phase 5: post-recurrence / outcome proxies ---
    "New Neoplasm Event Post Initial Therapy Indicator",
    "Disease Free Status",
    "Disease Free (Months)",
    "Overall Survival Status",
    "Overall Survival (Months)",
    "Days to biochemical recurrence first",
    "Days_to_biochemical_recurrence_first",
    # --- From FINAL_v2 pipeline: PSA / temporal leakage + raw target ---
    "Psa most recent results",
    "Days to PSA",
    "Psa_most_recent_results",
    "Days_to_PSA",
    "Biochemical Recurrence Indicator",      # preserved by src/clinical.py
    "Biochemical_Recurrence_Indicator",      # until target is created
]

# ---------------------------------------------------------------------------
# Target / ID columns
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "Biochemical_Recurrence_Code"
TARGET_SOURCE_COLUMN: str = "Biochemical Recurrence Indicator"
PATIENT_ID_COLUMN: str = "Patient Identifier"

# ---------------------------------------------------------------------------
# Peenalty PSO alpha
# -----------------------------------------------------------
PSO_PENALTY_ALPHA: float = 0.001
# ---------------------------------------------------------------------------
# Model hyperparameter defaults (baseline; overridden by tuning)
# ---------------------------------------------------------------------------
XGBOOST_PARAMS: dict = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

LIGHTGBM_PARAMS: dict = {
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "objective": "binary",
    "metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
    "verbose": -1,
}

CATBOOST_PARAMS: dict = {
    "iterations": 250,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "objective": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "train_dir": str(OUTPUT_DIR / "catboost_info")
}

LOGISTIC_PARAMS: dict = {
    "max_iter": 2000,
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

SVM_PARAMS: dict = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Hyperparameter search spaces (RandomizedSearchCV)
# ---------------------------------------------------------------------------
XGBOOST_SEARCH_SPACE: dict = {
    "n_estimators": [100, 150, 250, 350, 500],
    "max_depth": [1, 2, 3, 4, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "subsample": [0.65, 0.75, 0.85, 1.0],
    "colsample_bytree": [0.65, 0.75, 0.85, 1.0],
    "min_child_weight": [3, 8, 12, 20, 30],
    "reg_alpha": [0, 0.1, 0.5, 1, 2],
    "reg_lambda": [1, 2, 5, 10, 20],
    "gamma": [0, 0.5, 1, 2, 5],
}

RANDOM_SEARCH_N_ITER: int = 20      # random parameter settings to evaluate
RANDOM_SEARCH_CV_SPLITS: int = 5    # inner stratified CV folds for tuning

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
BOOTSTRAP_N: int = 3000
CONFIDENCE_LEVEL: float = 0.95

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 300
FIGURE_STYLE: str = "seaborn-v0_8-whitegrid"