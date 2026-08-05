"""
Dataset merging for TCGA-PRAD BCR prediction.

Handles merging clinical and RNA-Seq data by Patient Identifier,
validating the merge, detecting duplicates, and ensuring X and y
remain aligned throughout the pipeline.

Merge ONLY by PATIENT_ID — never assume row order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as config
from src.io import logger


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    dataset_name: str,
) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: missing required columns {missing}. "
            f"Available: {list(df.columns[:10])}..."
        )


def find_common_patients(
    clinical_ids: pd.Series,
    rna_ids: pd.Series,
) -> list[str]:
    """Find sorted intersection of patient IDs between two datasets."""
    common = sorted(set(clinical_ids).intersection(rna_ids))
    return common


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------
def merge_datasets(
    clinical: pd.DataFrame,
    rna_seq: pd.DataFrame,
    *,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Merge clinical and RNA-Seq data by Patient Identifier.

    Parameters
    ----------
    clinical : Clinical DataFrame with Patient Identifier and target.
    rna_seq : Expression matrix with Patient Identifier as index.
    validate : If True, perform validation checks.

    Returns
    -------
    Tuple of (X_merged, y) where X_merged contains all features
    and y is the binary target column.

    Raises
    ------
    ValueError : If required columns are missing or no common patients found.
    """
    # Validate required columns
    patient_col = config.PATIENT_ID_COLUMN
    target_col = config.TARGET_COLUMN

    validate_required_columns(
        clinical,
        [patient_col, target_col],
        "Clinical data",
    )

    # Ensure rna_seq has Patient Identifier as index
    if rna_seq.index.name != patient_col:
        raise ValueError(
            f"RNA-Seq index must be named '{patient_col}', "
            f"got '{rna_seq.index.name}'"
        )

    # Drop rows with missing target
    n_before = len(clinical)
    clinical = clinical.dropna(subset=[target_col]).copy()
    n_dropped = n_before - len(clinical)
    if n_dropped > 0:
        logger.warning("Dropped %d clinical rows with missing target", n_dropped)

    # Convert target to int
    clinical[target_col] = pd.to_numeric(clinical[target_col], errors="coerce")
    clinical = clinical.dropna(subset=[target_col]).copy()
    clinical[target_col] = clinical[target_col].astype(int)

    # Find common patients
    common_patients = find_common_patients(
        clinical[patient_col],
        rna_seq.index,
    )

    if not common_patients:
        raise ValueError(
            "No common patients found between clinical and RNA-Seq data. "
            f"Clinical has {len(clinical)} patients, RNA-Seq has {len(rna_seq)} samples."
        )

    logger.info("Found %d common patients", len(common_patients))

    # Filter and sort both datasets by common patients
    clinical_common = (
        clinical[clinical[patient_col].isin(common_patients)]
        .sort_values(patient_col)
        .reset_index(drop=True)
    )

    rna_common = (
        rna_seq.loc[common_patients]
        .reset_index()
        .rename(columns={"index": patient_col})
        .sort_values(patient_col)
        .reset_index(drop=True)
    )

    # Validate alignment
    if validate:
        validate_merge(clinical_common, rna_common)

    # Extract target and features
    y = clinical_common[target_col].astype(int).copy()

    clinical_features = clinical_common.drop(
        columns=[patient_col, target_col],
        errors="ignore",
    )

    gene_features = rna_common.drop(columns=[patient_col], errors="ignore")

    # Merge features
    X_merged = pd.concat(
        [clinical_features.reset_index(drop=True),
         gene_features.reset_index(drop=True)],
        axis=1,
    )

    logger.info(
        "Merged dataset: %d samples × %d features (%d clinical, %d genes)",
        len(X_merged),
        X_merged.shape[1],
        clinical_features.shape[1],
        gene_features.shape[1],
    )

    return X_merged, y


def validate_merge(
    clinical_common: pd.DataFrame,
    rna_common: pd.DataFrame,
) -> None:
    """Validate that merged datasets are properly aligned.

    Parameters
    ----------
    clinical_common : Clinical data filtered to common patients.
    rna_common : RNA-Seq data filtered to common patients.

    Raises
    ------
    AssertionError : If patient IDs don't match or lengths differ.
    """
    patient_col = config.PATIENT_ID_COLUMN

    # Check same number of samples
    assert len(clinical_common) == len(rna_common), (
        f"Length mismatch: clinical={len(clinical_common)}, "
        f"rna_seq={len(rna_common)}"
    )

    # Check patient IDs match exactly
    clinical_ids = clinical_common[patient_col].tolist()
    rna_ids = rna_common[patient_col].tolist()

    assert clinical_ids == rna_ids, (
        "Patient IDs do not match between clinical and RNA-Seq data. "
        f"First mismatch: clinical={clinical_ids[0]}, rna={rna_ids[0]}"
    )

    logger.info("Merge validation passed: %d samples aligned", len(clinical_common))


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
def check_duplicates(
    df: pd.DataFrame,
    id_column: str = config.PATIENT_ID_COLUMN,
) -> pd.DataFrame:
    """Detect and report duplicate patient IDs.

    Parameters
    ----------
    df : DataFrame to check for duplicates.
    id_column : Column name containing patient IDs.

    Returns
    -------
    DataFrame with duplicates removed (keeping first occurrence).
    """
    if id_column not in df.columns and df.index.name != id_column:
        raise ValueError(f"Column '{id_column}' not found in DataFrame")

    # Check if ID is in index or columns
    if df.index.name == id_column:
        ids = df.index
    else:
        ids = df[id_column]

    n_duplicates = ids.duplicated().sum()
    if n_duplicates > 0:
        logger.warning("Found %d duplicate patient IDs, keeping first occurrence", n_duplicates)
        if df.index.name == id_column:
            df = df[~df.index.duplicated(keep="first")]
        else:
            df = df[~df[id_column].duplicated(keep="first")]
    else:
        logger.info("No duplicate patient IDs found")

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_merge_summary(
    X: pd.DataFrame,
    y: pd.Series,
    clinical_only: pd.DataFrame | None = None,
    rna_only: pd.DataFrame | None = None,
) -> dict:
    """Generate a summary report of the merged dataset.

    Parameters
    ----------
    X : Merged feature matrix.
    y : Target variable.
    clinical_only : Optional clinical data before merge.
    rna_only : Optional RNA-Seq data before merge.

    Returns
    -------
    Dictionary with summary statistics.
    """
    summary = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": float(y.mean()),
    }

    if clinical_only is not None:
        summary["n_clinical_samples"] = len(clinical_only)

    if rna_only is not None:
        summary["n_rna_samples"] = len(rna_only)

    logger.info("Merge Summary:")
    logger.info("  Samples: %d", summary["n_samples"])
    logger.info("  Features: %d", summary["n_features"])
    logger.info("  Positive: %d (%.1f%%)", summary["n_positive"], summary["positive_rate"] * 100)
    logger.info("  Negative: %d", summary["n_negative"])

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def load_and_merge(
    clinical_path: str | None = None,
    rna_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and merge clinical and RNA-Seq data.

    This is the main entry point that orchestrates the full merge pipeline:
        1. Load processed clinical data
        2. Load preprocessed RNA-Seq data
        3. Check for duplicates
        4. Merge by Patient Identifier
        5. Validate alignment
        6. Generate summary report

    Parameters
    ----------
    clinical_path : Optional path to clinical CSV. Uses config default if None.
    rna_path : Optional path to RNA-Seq TSV. Uses config default if None.

    Returns
    -------
    Tuple of (X_merged, y).
    """
    from src.io import load_processed_clinical
    from src.genomics import preprocess_rna_seq

    # Load data
    clinical = load_processed_clinical()
    rna_seq = preprocess_rna_seq(rna_path)

    # Check duplicates
    clinical = check_duplicates(clinical)
    rna_seq = check_duplicates(rna_seq)

    # Merge
    X_merged, y = merge_datasets(clinical, rna_seq)

    # Report
    report_merge_summary(X_merged, y, clinical, rna_seq)

    return X_merged, y