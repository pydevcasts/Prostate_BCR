"""
RNA-Seq data preprocessing for TCGA-PRAD BCR prediction.

Handles loading, cleaning, and standardizing RNA-Seq expression data:
patient ID standardization, duplicate gene handling, low-expression
filtering, and quality control.

Normalization and log transformation belong in src.preprocessing —
this module only produces a clean, analysis-ready expression matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as config
from src.io import logger


# ---------------------------------------------------------------------------
# Patient ID standardization
# ---------------------------------------------------------------------------
def clean_patient_id(patient_id: str) -> str:
    """Extract the first three parts of a TCGA patient ID.

    TCGA sample IDs follow the pattern TCGA-XX-XXXX-YY where the last
    part indicates sample type (e.g., -01 for tumor, -11 for normal).
    This function keeps only the patient-level identifier.

    Examples
    --------
    >>> clean_patient_id("TCGA-2A-A8VL-01")
    'TCGA-2A-A8VL'
    >>> clean_patient_id("TCGA-2A-A8VL")
    'TCGA-2A-A8VL'
    """
    patient_id = str(patient_id).strip()
    parts = patient_id.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else patient_id


def standardize_patient_ids(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """Standardize patient IDs in the index or columns of a DataFrame.

    Parameters
    ----------
    df : Expression matrix (genes × samples or samples × genes).
    axis : 0 to standardize index, 1 to standardize columns.

    Returns
    -------
    DataFrame with standardized patient IDs and duplicates removed.
    """
    df = df.copy()

    if axis == 0:
        df.index = [clean_patient_id(i) for i in df.index]
        df.index.name = config.PATIENT_ID_COLUMN
        # Remove duplicate patients, keeping the first occurrence
        n_before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info("Removed %d duplicate patients from index", n_removed)
    else:
        df.columns = [clean_patient_id(c) for c in df.columns]
        n_before = len(df.columns)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        n_removed = n_before - len(df.columns)
        if n_removed > 0:
            logger.info("Removed %d duplicate patients from columns", n_removed)

    return df


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def load_rna_seq(path: str | None = None) -> pd.DataFrame:
    """Load raw RNA-Seq expression matrix from TSV.

    The file is expected to have genes as rows and samples as columns,
    with gene identifiers in the first column.

    Returns
    -------
    DataFrame with genes as index and samples as columns.
    """
    path = path or config.RNA_SEQ_RAW
    logger.info("Loading RNA-Seq data from %s", path)

    df = pd.read_csv(path, sep="\t", index_col=0)

    # Drop metadata columns if present
    metadata_cols = [c for c in ["Entrez_Gene_Id", "Hugo_Symbol"] if c in df.columns]
    if metadata_cols:
        df = df.drop(columns=metadata_cols)
        logger.info("Dropped metadata columns: %s", metadata_cols)

    logger.info("Loaded RNA-Seq matrix: %d genes × %d samples", *df.shape)
    return df


def handle_duplicate_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicate gene names by keeping the one with highest mean expression.

    Parameters
    ----------
    df : Expression matrix with genes as index.

    Returns
    -------
    DataFrame with unique gene names.
    """
    if not df.index.duplicated().any():
        logger.info("No duplicate genes found")
        return df

    n_duplicates = df.index.duplicated().sum()
    logger.info("Found %d duplicate gene names", n_duplicates)

    # For each gene, keep the row with the highest mean expression
    df = df.copy()
    df["_mean_expr"] = df.mean(axis=1)
    df = df.sort_values("_mean_expr", ascending=False)
    df = df[~df.index.duplicated(keep="first")]
    df = df.drop(columns="_mean_expr")

    logger.info("Resolved duplicates → %d unique genes", len(df))
    return df


def filter_low_expression_genes(
    df: pd.DataFrame,
    min_samples: int = 10,
    min_expression: float = 1.0,
) -> pd.DataFrame:
    """Filter out genes with low expression across samples.

    A gene is kept if it has expression >= min_expression in at least
    min_samples samples.

    Parameters
    ----------
    df : Expression matrix with genes as index.
    min_samples : Minimum number of samples with expression above threshold.
    min_expression : Minimum expression value to count a sample.

    Returns
    -------
    Filtered expression matrix.
    """
    n_before = len(df)

    # Count samples with expression above threshold for each gene
    expressed_counts = (df >= min_expression).sum(axis=1)
    mask = expressed_counts >= min_samples

    df = df[mask].copy()
    n_removed = n_before - len(df)

    logger.info(
        "Filtered low-expression genes: %d → %d (removed %d)",
        n_before, len(df), n_removed,
    )
    return df


def transpose_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose matrix from genes×samples to samples×genes and standardize IDs.

    Parameters
    ----------
    df : Expression matrix with genes as index and samples as columns.

    Returns
    -------
    DataFrame with samples as index and genes as columns.
    """
    df = df.T
    df = standardize_patient_ids(df, axis=0)
    logger.info("Transposed and standardized: %d samples × %d genes", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
def quality_control_report(df: pd.DataFrame) -> pd.Series:
    """Generate a quality control report for the expression matrix.

    Parameters
    ----------
    df : Expression matrix with samples as index and genes as columns.

    Returns
    -------
    Series with QC metrics.
    """
    report = pd.Series({
        "n_samples": len(df),
        "n_genes": df.shape[1],
        "min_expression": float(df.min().min()),
        "max_expression": float(df.max().max()),
        "mean_expression": float(df.mean().mean()),
        "median_expression": float(df.median().median()),
        "pct_zero": float((df == 0).mean().mean() * 100),
        "pct_missing": float(df.isna().mean().mean() * 100),
    })

    logger.info("QC Report:")
    for metric, value in report.items():
        logger.info("  %s: %.2f", metric, value)

    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def preprocess_rna_seq(
    path: str | None = None,
    *,
    min_samples: int = 10,
    min_expression: float = 1.0,
    run_qc: bool = True,
) -> pd.DataFrame:
    """Run the full RNA-Seq preprocessing pipeline.

    Steps:
        1. Load raw expression matrix
        2. Handle duplicate genes
        3. Filter low-expression genes
        4. Transpose to samples × genes
        5. Standardize patient IDs
        6. Run quality control (optional)

    Returns
    -------
    Cleaned expression matrix with samples as index and genes as columns.
    """
    df = load_rna_seq(path)
    df = handle_duplicate_genes(df)
    df = filter_low_expression_genes(df, min_samples, min_expression)
    df = transpose_and_standardize(df)

    if run_qc:
        quality_control_report(df)

    return df