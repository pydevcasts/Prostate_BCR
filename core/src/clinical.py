"""
Clinical data preprocessing for TCGA-PRAD BCR prediction.

Handles cleaning of raw TCGA clinical files: metadata removal, type
detection, missing-value handling, low-variance filtering, leakage
removal, categorical encoding, and target creation.

All heavy numerical transforms (scaling, log, winsorize) belong in
src.preprocessing — this module only produces a clean, analysis-ready
DataFrame that still contains Patient Identifier and target column.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config
from src.io import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_METADATA_ROWS = 4  # rows 0-3: description / type / priority / name
_NUMERIC_RATIO = 0.90
_DOMINANT_RATIO = 0.95
_MAX_CATEGORIES = 15
_TARGET_SOURCE_COL = "Biochemical Recurrence Indicator"
_INVALID_VALUES = frozenset({
    "[Not Available]", "[Discrepancy]", "[Unknown]",
    " [Unknown]", "[Not Applicable]",
})
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------
def drop_metadata_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the four TCGA header rows (description, type, priority, name)."""
    cleaned = df.iloc[_METADATA_ROWS:].reset_index(drop=True)
    logger.info("Dropped %d metadata rows → %d patients", _METADATA_ROWS, len(cleaned))
    return cleaned


def detect_numeric_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Auto-detect numeric columns via regex on sampled values.

    Returns (df, numeric_cols, categorical_cols). Numeric columns are
    cast to float in-place on a copy.
    """
    df = df.copy()
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col == config.PATIENT_ID_COLUMN:
            continue
        sample = df[col].dropna().astype(str).head(100)
        if sample.empty:
            categorical_cols.append(col)
            continue
        ratio = sample.str.strip().map(lambda x: bool(_NUMBER_RE.match(x))).mean()
        if ratio > _NUMERIC_RATIO:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    logger.info("Detected %d numeric / %d categorical columns", len(numeric_cols), len(categorical_cols))
    return df, numeric_cols, categorical_cols


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float = config.MISSING_THRESHOLD,
) -> pd.DataFrame:
    """Replace TCGA sentinel strings with NaN; impute numerics with median.

    Numeric columns with NaN fraction > *threshold* are dropped entirely.
    """
    df = df.copy()

    # sentinel → NaN (object columns only)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(list(_INVALID_VALUES), pd.NA)

    # numeric imputation / removal
    to_drop: list[str] = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        frac = df[col].isna().mean()
        if frac > threshold:
            to_drop.append(col)
        elif frac > 0:
            df[col] = df[col].fillna(df[col].median())

    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info("Dropped %d columns exceeding %.0f%% missing", len(to_drop), threshold * 100)

    return df


def drop_low_variance_columns(
    df: pd.DataFrame,
    dominant_ratio: float = _DOMINANT_RATIO,
) -> pd.DataFrame:
    """Drop columns where a single value dominates (near-constant)."""
    to_drop: list[str] = []

    for col in df.columns:
        if col in (config.PATIENT_ID_COLUMN, _TARGET_SOURCE_COL):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            to_drop.append(col)
            continue
        counts = non_null.value_counts()
        if counts.iloc[0] / len(non_null) > dominant_ratio:
            to_drop.append(col)

    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info("Dropped %d near-constant columns", len(to_drop))
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that leak target information.

    IMPORTANT: The target source column is PRESERVED here because it is
    needed later to create the binary target. It will be dropped after
    target creation.
    """
    present = [
        c for c in config.LEAKAGE_COLUMNS
        if c in df.columns and c != _TARGET_SOURCE_COL
    ]
    if present:
        df = df.drop(columns=present)
        logger.info("Removed %d leakage columns", len(present))
    return df


def one_hot_encode(
    df: pd.DataFrame,
    max_categories: int = _MAX_CATEGORIES,
) -> pd.DataFrame:
    """One-hot encode categorical columns; drop high-cardinality ones.

    Patient Identifier and target source column are protected throughout.
    """
    protected: dict[str, pd.Series] = {}
    for col in (config.PATIENT_ID_COLUMN, _TARGET_SOURCE_COL):
        if col in df.columns:
            protected[col] = df[col].copy()
            df = df.drop(columns=col)

    categorical = df.select_dtypes(include="object").columns.tolist()
    dummies: list[pd.DataFrame] = []

    for col in categorical:
        n_levels = df[col].nunique(dropna=False)
        if n_levels <= max_categories:
            dummies.append(pd.get_dummies(df[col], prefix=col, dtype=int))
        else:
            logger.info("Dropped %s (%d levels > %d)", col, n_levels, max_categories)
        df = df.drop(columns=col)

    if dummies:
        df = pd.concat([df, *dummies], axis=1)

    for col, series in protected.items():
        df[col] = series

    logger.info("One-hot encoding done → %d total columns", df.shape[1])
    return df


def create_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'Biochemical Recurrence Indicator' to a binary integer target.

    After mapping, the raw indicator column is dropped to prevent leakage.
    """
    if _TARGET_SOURCE_COL not in df.columns:
        raise ValueError(f"Target source column '{_TARGET_SOURCE_COL}' not found")

    df = df.copy()
    mapping = {"YES": 1, "NO": 0, True: 1, False: 0}
    df[config.TARGET_COLUMN] = df[_TARGET_SOURCE_COL].map(mapping)
    df = df.dropna(subset=[config.TARGET_COLUMN])
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)

    # Drop the raw indicator after target creation to prevent leakage
    df = df.drop(columns=[_TARGET_SOURCE_COL])

    n_pos = int(df[config.TARGET_COLUMN].sum())
    n_neg = len(df) - n_pos
    logger.info(
        "Target created: %d positive / %d negative (%.1f%% positive rate)",
        n_pos, n_neg, n_pos / len(df) * 100,
    )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def preprocess_clinical(
    df: pd.DataFrame,
    *,
    missing_threshold: float = config.MISSING_THRESHOLD,
    dominant_ratio: float = _DOMINANT_RATIO,
    max_categories: int = _MAX_CATEGORIES,
) -> pd.DataFrame:
    """Run the full clinical preprocessing pipeline end-to-end.

    Steps:
        1. Drop TCGA metadata rows
        2. Auto-detect numeric columns
        3. Handle missing values (impute / drop)
        4. Drop near-constant columns
        5. Drop leakage columns (target source preserved)
        6. One-hot encode categoricals
        7. Create binary target column and drop raw indicator

    Returns a DataFrame with Patient Identifier, all features, and target.
    """
    df = drop_metadata_rows(df)
    df, numeric_cols, _ = detect_numeric_columns(df)
    df = handle_missing_values(df, numeric_cols, missing_threshold)
    df = drop_low_variance_columns(df, dominant_ratio)
    df = drop_leakage_columns(df)
    df = one_hot_encode(df, max_categories)
    df = create_target_column(df)
    return df