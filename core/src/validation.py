"""
Validation utilities for the Prostate BCR prediction pipeline.
Provides functions to validate feature consistency, model inputs, and pipeline integrity.
Also includes cross-platform normalization for external validation.
"""

from typing import Any, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, quantile_transform
from src.io import logger


def validate_features_available(
    X_data: pd.DataFrame,
    required_features: List[str],
    context: str = "",
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """Validate that all required features are present in DataFrame.
    
    Args:
        X_data: Input DataFrame to validate
        required_features: List of feature names that must be present
        context: Optional context string for error messages (e.g., "Notebook 07:")
        strict: If True, raise ValueError on missing features; if False, log warning
        
    Returns:
        Tuple of (valid_features, missing_features)
        
    Raises:
        ValueError: If strict=True and features are missing
    """
    available = set(X_data.columns)
    required = set(required_features)
    
    valid = [f for f in required_features if f in available]
    missing = list(required - available)
    extra = list(available - required)
    
    if missing:
        msg = f"{context} Missing {len(missing)} required features: {missing[:10]}"
        if len(missing) > 10:
            msg += "..."
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    
    if extra:
        logger.info(f"{context} Found {len(extra)} extra columns (ignored): {extra[:5]}")
    
    return valid, missing


def validate_model_input(
    model: Any,
    X_data: pd.DataFrame,
    context: str = "",
) -> None:
    """Validate that X_data matches model's expected input.
    
    Args:
        model: Trained model object
        X_data: Feature DataFrame to validate
        context: Optional context string for error messages
        
    Raises:
        ValueError: If feature count or names don't match
    """
    expected_count = model.n_features_in_
    actual_count = X_data.shape[1]
    
    if actual_count != expected_count:
        raise ValueError(
            f"{context} Model expects {expected_count} features, got {actual_count}"
        )
    
    # For XGBoost, check feature names match exactly
    if hasattr(model, 'get_booster'):
        expected_names = model.get_booster().feature_names
        actual_names = list(X_data.columns)
        if expected_names != actual_names:
            raise ValueError(
                f"{context} Feature names don't match.\n"
                f"Model expects: {expected_names[:5]}...\n"
                f"Data has: {actual_names[:5]}..."
            )
    
    logger.info(f"{context} Model input validation passed: {actual_count} features")


def validate_feature_consistency(
    X_data: pd.DataFrame,
    expected_features: List[str],
    context: str = "",
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """Validate feature consistency between datasets.
    
    This is an alias for validate_features_available for backward compatibility.
    
    Args:
        X_data: Input DataFrame to validate
        expected_features: List of expected feature names
        context: Optional context string for error messages
        strict: If True, raise ValueError on missing features
        
    Returns:
        Tuple of (valid_features, missing_features)
    """
    return validate_features_available(
        X_data, expected_features, context=context, strict=strict
    )


# ---------------------------------------------------------------------------
# Cross-Platform Normalization Functions
# ---------------------------------------------------------------------------
def normalize_cross_platform(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    method: str = "quantile",
    common_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize train and test datasets to align distributions across platforms.
    
    This function addresses batch effects between different platforms (e.g., RNA-seq
    vs microarray) by applying distribution-matching transformations.
    
    Parameters
    ----------
    df_train : Reference/training dataset (e.g., TCGA).
    df_test : Target/test dataset (e.g., GSE70769).
    method : Normalization method ("quantile", "zscore", "rank", "combat").
    common_features : List of features to use for normalization.
                     If None, uses intersection of columns.
    
    Returns
    -------
    Tuple of (normalized_train, normalized_test) DataFrames.
    
    Examples
    --------
    >>> X_train_norm, X_ext_norm = normalize_cross_platform(
    ...     X_train, X_external, method="quantile"
    ... )
    """
    # Find common features
    if common_features is None:
        common_features = list(set(df_train.columns) & set(df_test.columns))
    
    if len(common_features) == 0:
        raise ValueError("No common features found between datasets")
    
    logger.info(
        "Cross-platform normalization: %d common features, method=%s",
        len(common_features), method
    )
    
    # Work on copies
    train_norm = df_train.copy()
    test_norm = df_test.copy()
    
    if method == "quantile":
        # Quantile normalization to match distributions
        train_norm[common_features], test_norm[common_features] = _quantile_normalize(
            train_norm[common_features], test_norm[common_features]
        )
    
    elif method == "zscore":
        # Z-score standardization per gene
        train_norm[common_features], test_norm[common_features] = _zscore_normalize(
            train_norm[common_features], test_norm[common_features]
        )
    
    elif method == "rank":
        # Rank-based transformation (robust to platform differences)
        train_norm[common_features], test_norm[common_features] = _rank_normalize(
            train_norm[common_features], test_norm[common_features]
        )
    
    elif method == "combat":
        # ComBat-style batch correction (requires both datasets combined)
        from src.batch_correction import combat_correction
        
        combined = pd.concat([train_norm[common_features], test_norm[common_features]])
        batch = np.array([0] * len(train_norm) + [1] * len(test_norm))
        corrected = combat_correction(combined, batch)
        
        train_norm[common_features] = corrected.iloc[:len(train_norm)]
        test_norm[common_features] = corrected.iloc[len(train_norm):]
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    logger.info(
        "Normalization complete: train_mean=%.4f, test_mean_after=%.4f",
        train_norm[common_features].mean().mean(),
        test_norm[common_features].mean().mean()
    )
    
    return train_norm, test_norm


def _quantile_normalize(
    ref_data: pd.DataFrame,
    target_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply quantile normalization to match reference distribution."""
    # Transform both to normal distribution based on combined ranks
    ref_transformed = quantile_transform(
        ref_data.values,
        output_distribution='normal',
        subsample=100000,
        random_state=42,
    )
    
    target_transformed = quantile_transform(
        target_data.values,
        output_distribution='normal',
        subsample=100000,
        random_state=42,
    )
    
    # Scale target to match reference statistics
    ref_mean = ref_data.mean(axis=0)
    ref_std = ref_data.std(axis=0)
    
    target_scaled = target_transformed * ref_std.values + ref_mean.values
    
    return (
        pd.DataFrame(ref_transformed, index=ref_data.index, columns=ref_data.columns),
        pd.DataFrame(target_scaled, index=target_data.index, columns=target_data.columns)
    )


def _zscore_normalize(
    ref_data: pd.DataFrame,
    target_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score normalize per gene using reference statistics."""
    # Compute reference statistics
    ref_mean = ref_data.mean(axis=0)
    ref_std = ref_data.std(axis=0)
    
    # Keep reference as-is (already normalized in training)
    # Standardize target to reference distribution
    tgt_mean = target_data.mean(axis=0)
    tgt_std = target_data.std(axis=0)
    
    # Z-score transform target, then scale to reference
    target_standardized = (target_data - tgt_mean) / (tgt_std + 1e-8)
    target_normalized = target_standardized * ref_std + ref_mean
    
    return ref_data.copy(), target_normalized


def _rank_normalize(
    ref_data: pd.DataFrame,
    target_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert expression values to ranks (robust to platform differences)."""
    def rank_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Transform each column to ranks."""
        ranked = df.copy()
        for col in df.columns:
            # Rank values (average for ties)
            ranked[col] = df[col].rank(method='average')
            # Normalize to [0, 1] range
            ranked[col] = (ranked[col] - ranked[col].min()) / (ranked[col].max() - ranked[col].min() + 1e-8)
        return ranked
    
    ref_ranked = rank_transform(ref_data)
    target_ranked = rank_transform(target_data)
    
    logger.info("Rank transformation applied: values normalized to [0, 1]")
    
    return ref_ranked, target_ranked


def get_common_features(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Get list of common features between two datasets.
    
    Parameters
    ----------
    df1 : First DataFrame.
    df2 : Second DataFrame.
    exclude_patterns : Optional list of column name patterns to exclude
                       (e.g., engineered clinical features).
    
    Returns
    -------
    List of common feature names.
    """
    common = set(df1.columns) & set(df2.columns)
    
    if exclude_patterns:
        for pattern in exclude_patterns:
            common = {c for c in common if pattern not in c}
    
    return sorted(list(common))


def prepare_external_validation(
    X_train: pd.DataFrame,
    X_external: pd.DataFrame,
    y_external: pd.Series | np.ndarray,
    selected_features: List[str],
    exclude_clinical: bool = True,
    normalization_method: str = "quantile",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series | np.ndarray, List[str]]:
    """Prepare external validation dataset with proper normalization.
    
    This function handles:
    1. Feature intersection (only genes present in both datasets)
    2. Exclusion of clinical features if not available in external data
    3. Cross-platform normalization
    4. Proper alignment of features
    
    Parameters
    ----------
    X_train : Training features (TCGA).
    X_external : External validation features (e.g., GSE70769).
    y_external : External validation target.
    selected_features : Original selected features from training.
    exclude_clinical : Whether to exclude clinical/engineered features.
    normalization_method : Method for cross-platform normalization.
    
    Returns
    -------
    Tuple of (X_train_aligned, X_external_aligned, y_external, used_features).
    
    Examples
    --------
    >>> X_train_aligned, X_ext_aligned, y_ext, features = prepare_external_validation(
    ...     X_train, X_GSE70769, y_GSE70769, selected_features,
    ...     exclude_clinical=True, normalization_method="quantile"
    ... )
    >>> # Then use aligned data for prediction
    >>> y_prob = model.predict_proba(X_ext_aligned[features])[:, 1]
    """
    # Define clinical/engineered feature patterns to potentially exclude
    clinical_patterns = [
        "Gleason", "PSA_Pathway", "AR_Signaling", "Proliferation",
        "Margin", "Lymph", "T_Stage", "High_Risk"
    ] if exclude_clinical else []
    
    # Get common features
    common_features = get_common_features(X_train, X_external, exclude_patterns=None)
    
    # Filter to only selected features that are common
    gene_features = [f for f in selected_features if f in common_features]
    
    # Check if we should exclude clinical features
    if exclude_clinical:
        gene_only_features = []
        for feat in gene_features:
            is_clinical = any(pat in feat for pat in clinical_patterns)
            if not is_clinical:
                gene_only_features.append(feat)
        
        if len(gene_only_features) > 0:
            logger.info(
                "Excluding %d clinical features, using %d gene features only",
                len(gene_features) - len(gene_only_features),
                len(gene_only_features)
            )
            gene_features = gene_only_features
    
    if len(gene_features) < len(selected_features):
        logger.warning(
            "Only %d/%d selected features available in external dataset",
            len(gene_features), len(selected_features)
        )
    
    if len(gene_features) == 0:
        raise ValueError("No usable features found for external validation")
    
    # Subset to common features
    X_train_subset = X_train[gene_features].copy()
    X_ext_subset = X_external[gene_features].copy()
    
    # Apply cross-platform normalization
    X_train_norm, X_ext_norm = normalize_cross_platform(
        X_train_subset, X_ext_subset, method=normalization_method
    )
    
    logger.info(
        "External validation prepared: %d features, normalization=%s",
        len(gene_features), normalization_method
    )
    
    return X_train_norm, X_ext_norm, y_external, gene_features


def verify_pipeline_consistency(config: Any) -> dict:
    """Verify consistency across all pipeline artifacts.
    
    Args:
        config: Configuration object with TABLES_DIR, PROCESSED_DIR, MODELS_DIR
        
    Returns:
        Dictionary with keys:
            - passed: bool indicating if all checks passed
            - issues: list of issue descriptions
            - artifacts: dictionary of artifact metadata
            - skipped: list of checks that were skipped due to missing prerequisites
            
    Raises:
        FileNotFoundError: If critical artifacts are missing
    """
    import json
    from pathlib import Path
    
    results = {
        "passed": True,
        "issues": [],
        "artifacts": {},
        "skipped": []
    }
    
    # Check 1: Feature list consistency
    final_set = None
    try:
        final_features_path = config.TABLES_DIR / "selected_features_final.csv"
        pso_features_path = config.TABLES_DIR / "selected_features.csv"
        
        if not final_features_path.exists():
            results["issues"].append(
                f"Missing final features file: {final_features_path}. "
                f"Run Notebook 05 (Model Training) first to generate artifacts."
            )
            results["passed"] = False
            results["skipped"].append("Feature consistency check")
        else:
            final_features = pd.read_csv(final_features_path)
            final_set = set(final_features["feature"])
            results["artifacts"]["final_feature_count"] = len(final_set)
        
        if pso_features_path.exists():
            pso_features = pd.read_csv(pso_features_path)
            pso_set = set(pso_features["feature"])
            results["artifacts"]["pso_feature_count"] = len(pso_set)
            
            if final_set is not None and not pso_set.issubset(final_set):
                results["issues"].append(
                    "PSO features not subset of final features"
                )
                results["passed"] = False
                
    except Exception as e:
        results["issues"].append(f"Error checking feature files: {e}")
        results["passed"] = False
    
    # Check 2: Test data exists and matches
    try:
        test_data_path = config.PROCESSED_DIR / "X_test_selected.csv"
        if not test_data_path.exists():
            results["issues"].append(
                f"Missing test data file: {test_data_path}. "
                f"Run Notebook 05 (Model Training) first to generate artifacts."
            )
            results["passed"] = False
            results["skipped"].append("Test data consistency check")
        else:
            X_test = pd.read_csv(test_data_path)
            results["artifacts"]["test_sample_count"] = len(X_test)
            results["artifacts"]["test_feature_count"] = X_test.shape[1]
            
            if final_set is not None:
                test_cols_set = set(X_test.columns)
                if test_cols_set != final_set:
                    # Provide detailed diagnostic information
                    missing_in_test = final_set - test_cols_set
                    extra_in_test = test_cols_set - final_set
                    results["issues"].append(
                        f"Test data columns ({len(test_cols_set)}) don't match "
                        f"final features ({len(final_set)}). "
                        f"Missing in test: {list(missing_in_test)[:5]}{'...' if len(missing_in_test) > 5 else ''}. "
                        f"Extra in test: {list(extra_in_test)[:5]}{'...' if len(extra_in_test) > 5 else ''}."
                    )
                    results["passed"] = False
                    
    except Exception as e:
        results["issues"].append(f"Error checking test data: {e}")
        results["passed"] = False
    
    # Check 3: Model exists
    try:
        model_path = config.MODELS_DIR / "best_model_xgboost.joblib"
        if not model_path.exists():
            results["issues"].append(
                f"Model file not found: {model_path}. "
                f"Run Notebook 05 (Model Training) first to generate artifacts."
            )
            results["passed"] = False
            results["skipped"].append("Model existence check")
        else:
            results["artifacts"]["model_exists"] = True
    except Exception as e:
        results["issues"].append(f"Error checking model: {e}")
        results["passed"] = False
    
    return results