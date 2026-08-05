"""
Preprocessing pipelines for clinical and RNA-Seq features.

All transformers are built as sklearn Pipeline objects so they can be
embedded directly inside cross-validation loops without data leakage.
Every transform is fitted on training data only and then applied to
validation/test data.

Clinical pipeline:  Missing values → Encoding (already done) → Scaling
RNA pipeline:       Filtering → Normalization → Log transform → Scaling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as config
from src.io import logger


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------
class Log1pSkewTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transform to columns with skewness above a threshold.

    Fit computes which columns are skewed on training data only.
    """

    def __init__(self, skew_threshold: float = 2.0):
        self.skew_threshold = skew_threshold
        self.skew_cols_: list[int] = []

    def fit(self, X: np.ndarray, y: Any = None) -> "Log1pSkewTransformer":
        X = np.asarray(X, dtype=np.float64)
        self.skew_cols_ = [
            i for i in range(X.shape[1])
            if abs(skew(X[:, i][~np.isnan(X[:, i])])) > self.skew_threshold
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        for col in self.skew_cols_:
            X[:, col] = np.log1p(np.clip(X[:, col], 0, None))
        return X


class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    """Clip values to [lower, upper] percentiles computed on training data.

    Parameters
    ----------
    lower : Lower percentile (e.g. 0.01 for 1%).
    upper : Upper percentile (e.g. 0.99 for 99%).
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.limits_: list[tuple[float, float]] = []

    def fit(self, X: np.ndarray, y: Any = None) -> "WinsorizeTransformer":
        X = np.asarray(X, dtype=np.float64)
        self.limits_ = []
        for i in range(X.shape[1]):
            col = X[:, i][~np.isnan(X[:, i])]
            if len(col) > 0:
                lo = float(np.percentile(col, self.lower * 100))
                hi = float(np.percentile(col, self.upper * 100))
            else:
                lo, hi = 0.0, 0.0
            self.limits_.append((lo, hi))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        for i, (lo, hi) in enumerate(self.limits_):
            X[:, i] = np.clip(X[:, i], lo, hi)
        return X


class LogNormalizeTransformer(BaseEstimator, TransformerMixin):
    """Apply log2(x + 1) normalization to RNA-Seq counts.

    Suitable for RSEM FPKM/TPM values that are already non-negative.
    """

    def __init__(self, pseudocount: float = 1.0):
        self.pseudocount = pseudocount

    def fit(self, X: np.ndarray, y: Any = None) -> "LogNormalizeTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.log2(X + self.pseudocount)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def build_clinical_pipeline(
    *,
    apply_log_skew: bool = True,
    apply_winsorize: bool = True,
    skew_threshold: float = 2.0,
    winsorize_limits: tuple[float, float] = (0.01, 0.99),
) -> Pipeline:
    """Build the clinical preprocessing pipeline.

    Steps:
        1. Median imputation for missing values
        2. Log1p on skewed columns (optional)
        3. Winsorization (optional)
        4. Standard scaling

    Returns a fitted-on-train Pipeline object.
    """
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if apply_log_skew:
        steps.append(("log_skew", Log1pSkewTransformer(skew_threshold=skew_threshold)))

    if apply_winsorize:
        steps.append(("winsorize", WinsorizeTransformer(
            lower=winsorize_limits[0],
            upper=winsorize_limits[1],
        )))

    steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def build_rna_pipeline() -> Pipeline:
    """Build the RNA-Seq preprocessing pipeline.

    Steps:
        1. Median imputation for any remaining NaN/Inf
        2. Log2(x+1) normalization
        3. Standard scaling

    Returns a Pipeline object.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log_norm", LogNormalizeTransformer(pseudocount=1.0)),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Feature-type aware combined pipeline
# ---------------------------------------------------------------------------
def identify_column_groups(
    X: pd.DataFrame,
    clinical_prefixes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split column names into clinical and gene groups.

    Clinical columns are typically non-numeric names or one-hot encoded
    names from clinical preprocessing. Gene columns are numeric-looking
    identifiers from RNA-Seq (e.g., 'BRCA1', 'TP53').

    Parameters
    ----------
    X : Merged feature DataFrame.
    clinical_prefixes : Optional list of prefixes to identify clinical columns.

    Returns
    -------
    Tuple of (clinical_cols, gene_cols).
    """
    if clinical_prefixes is None:
        clinical_prefixes = [
            "Gleason", "Lymph", "Days", "Diagnosis", "Age",
            "Patient", "Tumor", "Primary", "Bone", "Ct",
            "Mri", "Surgical", "Person", "Did", "Neoplasm",
            "International", "American", "Race", "Ethnicity",
            "Adjuvant", "Tissue", "Form", "Year", "Overall",
            "Disease", "Stage", "Sex", "Informed", "ICD",
        ]

    clinical_cols = []
    gene_cols = []

    for col in X.columns:
        is_clinical = any(
            col.lower().startswith(p.lower()) or col.lower().startswith(p.lower().replace(" ", "_"))
            for p in clinical_prefixes
        )
        if is_clinical:
            clinical_cols.append(col)
        else:
            gene_cols.append(col)

    logger.info("Column groups: %d clinical, %d gene", len(clinical_cols), len(gene_cols))
    return clinical_cols, gene_cols


def build_combined_pipeline(
    X: pd.DataFrame,
    *,
    apply_log_skew: bool = True,
    apply_winsorize: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer that applies different pipelines to
    clinical and gene features.

    Parameters
    ----------
    X : Merged feature DataFrame (used only to identify column groups).
    apply_log_skew : Whether to apply log1p to skewed clinical columns.
    apply_winsorize : Whether to apply winsorization to clinical columns.

    Returns
    -------
    ColumnTransformer ready to be fitted on training data.
    """
    clinical_cols, gene_cols = identify_column_groups(X)

    clinical_pipeline = build_clinical_pipeline(
        apply_log_skew=apply_log_skew,
        apply_winsorize=apply_winsorize,
    )
    rna_pipeline = build_rna_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("clinical", clinical_pipeline, clinical_cols),
            ("genes", rna_pipeline, gene_cols),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Fit / transform helpers for use inside CV
# ---------------------------------------------------------------------------
def fit_preprocessing(
    preprocessor: ColumnTransformer | Pipeline,
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray | None = None,
) -> ColumnTransformer | Pipeline:
    """Fit a preprocessor on training data only.

    Parameters
    ----------
    preprocessor : Unfitted pipeline or ColumnTransformer.
    X_train : Training features.
    y_train : Training target (unused by most transformers).

    Returns
    -------
    Fitted preprocessor.
    """
    preprocessor.fit(X_train, y_train)
    logger.info("Preprocessing fitted on %d training samples", len(X_train))
    return preprocessor


def transform_data(
    preprocessor: ColumnTransformer | Pipeline,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Apply a fitted preprocessor to new data.

    Parameters
    ----------
    preprocessor : Already fitted pipeline.
    X : Data to transform (validation or test).

    Returns
    -------
    Transformed numpy array.
    """
    return preprocessor.transform(X)


def get_feature_names(
    preprocessor: ColumnTransformer,
    clinical_cols: list[str],
    gene_cols: list[str],
) -> list[str]:
    """Extract feature names after ColumnTransformer application.

    Parameters
    ----------
    preprocessor : Fitted ColumnTransformer.
    clinical_cols : Original clinical column names.
    gene_cols : Original gene column names.

    Returns
    -------
    List of feature names in transformed order.
    """
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if trans != "drop":
            names.extend(cols)
    return names