"""
Batch Effect Correction Module for Cross-Study Validation.

This module provides batch effect correction methods including:
- ComBat (Empirical Bayes)
- Standardization-based correction
- Quantile normalization

These methods are essential for improving external validation performance
when combining data from different platforms (e.g., RNA-seq vs microarray).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, quantile_transform

from src.io import logger


# ---------------------------------------------------------------------------
# ComBat Implementation (Simplified Empirical Bayes)
# ---------------------------------------------------------------------------
def combat_correction(
    data: pd.DataFrame,
    batch: np.ndarray | pd.Series,
    model: Optional[np.ndarray | pd.DataFrame] = None,
    parametric: bool = True,
) -> pd.DataFrame:
    """Apply ComBat batch effect correction using Empirical Bayes.

    This is a simplified implementation of the ComBat algorithm for
    removing batch effects from gene expression data.

    Parameters
    ----------
    data : Gene expression matrix (samples x genes).
    batch : Batch labels for each sample.
    model : Optional design matrix for biological covariates.
    parametric : Use parametric adjustments (True) or non-parametric.

    Returns
    -------
    Batch-corrected data DataFrame.

    References
    ----------
    Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray
    expression data using empirical Bayes methods. Biostatistics. 2007.
    """
    data = data.copy()
    batch = np.asarray(batch)

    # Check for single batch (no correction needed)
    if len(np.unique(batch)) == 1:
        logger.warning("Only one batch detected - no correction applied")
        return data

    n_samples, n_genes = data.shape

    # Standardize data per gene
    scaler = StandardScaler(with_mean=True, with_std=True)
    standardized_data = scaler.fit_transform(data.T).T

    # Design matrix for batches
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)

    # Estimate batch means and variances
    batch_means = np.zeros((n_batches, n_genes))
    batch_vars = np.zeros((n_batches, n_genes))

    for i, b in enumerate(unique_batches):
        mask = batch == b
        batch_data = standardized_data[mask]
        batch_means[i] = batch_data.mean(axis=0)
        batch_vars[i] = batch_data.var(axis=0)

    # Apply parametric adjustment
    if parametric:
        # Shrink batch effect estimates toward overall mean
        overall_mean = batch_means.mean(axis=0, keepdims=True)
        overall_var = batch_vars.mean(axis=0, keepdims=True)

        # Prior variance estimation
        gamma_prior_var = batch_means.var(axis=0, ddof=1)
        delta_prior_var = batch_vars.var(axis=0, ddof=1)

        # Posterior estimates (simplified)
        shrink_factor = n_samples / (n_samples + gamma_prior_var + 0.1)
        adjusted_means = overall_mean + shrink_factor * (batch_means - overall_mean)

        shrink_factor_var = n_samples / (n_samples + delta_prior_var + 0.1)
        adjusted_vars = overall_var + shrink_factor_var * (batch_vars - overall_var)
        adjusted_vars = np.maximum(adjusted_vars, 0.01)  # Ensure positive variance
    else:
        adjusted_means = batch_means
        adjusted_vars = batch_vars

    # Remove batch effects
    corrected_data = standardized_data.copy()
    for i, b in enumerate(unique_batches):
        mask = batch == b
        corrected_data[mask] = (
            standardized_data[mask] - adjusted_means[i]
        ) / np.sqrt(adjusted_vars[i] + 0.01)

    # Rescale to original range
    corrected_data = corrected_data * data.std(axis=0) + data.mean(axis=0)
    corrected_df = pd.DataFrame(corrected_data, index=data.index, columns=data.columns)

    logger.info(
        "ComBat correction applied: %d batches, %d samples, %d genes",
        n_batches, n_samples, n_genes,
    )

    return corrected_df


# ---------------------------------------------------------------------------
# Z-score Standardization Across Batches
# ---------------------------------------------------------------------------
def zscore_standardization(
    reference_data: pd.DataFrame,
    target_data: pd.DataFrame,
    common_features: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize target data to match reference distribution.

    This method transforms target data to have the same mean and
    standard deviation as reference data, feature by feature.

    Parameters
    ----------
    reference_data : Reference dataset (e.g., TCGA training data).
    target_data : Target dataset to be corrected (e.g., GSE70769).
    common_features : List of features to use for correction.

    Returns
    -------
    Tuple of (corrected_reference, corrected_target) DataFrames.
    """
    if common_features is None:
        common_features = list(set(reference_data.columns) & set(target_data.columns))

    if len(common_features) == 0:
        raise ValueError("No common features found between datasets")

    ref_data = reference_data[common_features].copy()
    tgt_data = target_data[common_features].copy()

    # Compute reference statistics
    ref_mean = ref_data.mean(axis=0)
    ref_std = ref_data.std(axis=0)

    # Standardize target to reference
    tgt_mean = tgt_data.mean(axis=0)
    tgt_std = tgt_data.std(axis=0)

    # Z-score transform target, then scale to reference distribution
    tgt_standardized = (tgt_data - tgt_mean) / (tgt_std + 1e-8)
    tgt_corrected = tgt_standardized * ref_std + ref_mean

    # Keep original columns not in common_features unchanged
    corrected_target = target_data.copy()
    corrected_target[common_features] = tgt_corrected

    logger.info(
        "Z-score standardization: %d common features, ref_mean=%.4f, tgt_mean_after=%.4f",
        len(common_features), ref_mean.mean(), tgt_corrected.mean().mean(),
    )

    return ref_data, corrected_target


# ---------------------------------------------------------------------------
# Quantile Normalization
# ---------------------------------------------------------------------------
def quantile_normalization(
    reference_data: pd.DataFrame,
    target_data: pd.DataFrame,
    common_features: Optional[list] = None,
) -> pd.DataFrame:
    """Apply quantile normalization to match reference distribution.

    Parameters
    ----------
    reference_data : Reference dataset.
    target_data : Target dataset to normalize.
    common_features : Features to use for normalization.

    Returns
    -------
    Quantile-normalized target DataFrame.
    """
    if common_features is None:
        common_features = list(set(reference_data.columns) & set(target_data.columns))

    if len(common_features) == 0:
        raise ValueError("No common features found")

    ref_data = reference_data[common_features].values
    tgt_data = target_data[common_features].values

    # Apply quantile transformation
    tgt_normalized = quantile_transform(
        tgt_data,
        output_distribution='normal',
        subsample=100000,
        random_state=42,
    )

    # Scale to reference statistics
    ref_mean = ref_data.mean(axis=0)
    ref_std = ref_data.std(axis=0)

    tgt_normalized = tgt_normalized * ref_std + ref_mean

    corrected_target = target_data.copy()
    corrected_target[common_features] = tgt_normalized

    logger.info(
        "Quantile normalization applied: %d features",
        len(common_features),
    )

    return corrected_target


# ---------------------------------------------------------------------------
# Mean Centering (Simple Batch Correction)
# ---------------------------------------------------------------------------
def mean_centering(
    reference_data: pd.DataFrame,
    target_data: pd.DataFrame,
    common_features: Optional[list] = None,
) -> pd.DataFrame:
    """Apply simple mean centering for batch correction.

    Shifts target data to have the same mean as reference data.

    Parameters
    ----------
    reference_data : Reference dataset.
    target_data : Target dataset to correct.
    common_features : Features to use for correction.

    Returns
    -------
    Mean-centered target DataFrame.
    """
    if common_features is None:
        common_features = list(set(reference_data.columns) & set(target_data.columns))

    if len(common_features) == 0:
        raise ValueError("No common features found")

    ref_mean = reference_data[common_features].mean(axis=0)
    tgt_mean = target_data[common_features].mean(axis=0)

    # Calculate shift
    shift = ref_mean - tgt_mean

    corrected_target = target_data.copy()
    corrected_target[common_features] = target_data[common_features] + shift

    logger.info(
        "Mean centering applied: mean shift = %.4f",
        shift.mean(),
    )

    return corrected_target


# ---------------------------------------------------------------------------
# Feature-wise Scaling
# ---------------------------------------------------------------------------
def scale_to_reference(
    reference_data: pd.DataFrame,
    target_data: pd.DataFrame,
    common_features: Optional[list] = None,
) -> pd.DataFrame:
    """Scale target data to match reference range.

    Parameters
    ----------
    reference_data : Reference dataset.
    target_data : Target dataset to scale.
    common_features : Features to use for scaling.

    Returns
    -------
    Scaled target DataFrame.
    """
    if common_features is None:
        common_features = list(set(reference_data.columns) & set(target_data.columns))

    ref_min = reference_data[common_features].min(axis=0)
    ref_max = reference_data[common_features].max(axis=0)
    tgt_min = target_data[common_features].min(axis=0)
    tgt_max = target_data[common_features].max(axis=0)

    # Min-max normalize target
    tgt_range = tgt_max - tgt_min
    tgt_normalized = (target_data[common_features] - tgt_min) / (tgt_range + 1e-8)

    # Scale to reference range
    ref_range = ref_max - ref_min
    corrected = tgt_normalized * ref_range + ref_min

    corrected_target = target_data.copy()
    corrected_target[common_features] = corrected

    logger.info(
        "Scale to reference: range difference before=%.4f, after=%.4f",
        (tgt_max - tgt_min).mean(), (ref_range).mean(),
    )

    return corrected_target
