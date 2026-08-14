"""
Validation utilities for the Prostate BCR prediction pipeline.

Provides functions to validate feature consistency, model inputs, and pipeline integrity.
"""

from typing import Any, List, Set, Tuple
import pandas as pd
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


def verify_pipeline_consistency(config: Any) -> dict:
    """Verify consistency across all pipeline artifacts.
    
    Args:
        config: Configuration object with TABLES_DIR, PROCESSED_DIR, MODELS_DIR
        
    Returns:
        Dictionary with keys:
            - passed: bool indicating if all checks passed
            - issues: list of issue descriptions
            - artifacts: dictionary of artifact metadata
            
    Raises:
        FileNotFoundError: If critical artifacts are missing
    """
    import json
    from pathlib import Path
    
    results = {
        "passed": True,
        "issues": [],
        "artifacts": {}
    }
    
    # Check 1: Feature list consistency
    try:
        final_features_path = config.TABLES_DIR / "selected_features_final.csv"
        pso_features_path = config.TABLES_DIR / "selected_features.csv"
        
        if not final_features_path.exists():
            results["issues"].append(
                f"Missing final features file: {final_features_path}"
            )
            results["passed"] = False
        else:
            final_features = pd.read_csv(final_features_path)
            final_set = set(final_features["feature"])
            results["artifacts"]["final_feature_count"] = len(final_set)
        
        if not pso_features_path.exists():
            logger.warning(f"PSO features file not found: {pso_features_path}")
        else:
            pso_features = pd.read_csv(pso_features_path)
            pso_set = set(pso_features["feature"])
            results["artifacts"]["pso_feature_count"] = len(pso_set)
            
            if 'final_set' in locals() and not pso_set.issubset(final_set):
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
            results["issues"].append(f"Missing test data file: {test_data_path}")
            results["passed"] = False
        else:
            X_test = pd.read_csv(test_data_path)
            results["artifacts"]["test_sample_count"] = len(X_test)
            results["artifacts"]["test_feature_count"] = X_test.shape[1]
            
            if 'final_set' in locals():
                if set(X_test.columns) != final_set:
                    results["issues"].append(
                        f"Test data columns ({len(X_test.columns)}) don't match "
                        f"final features ({len(final_set)})"
                    )
                    results["passed"] = False
                    
    except Exception as e:
        results["issues"].append(f"Error checking test data: {e}")
        results["passed"] = False
    
    # Check 3: Model exists
    try:
        model_path = config.MODELS_DIR / "best_model_xgboost.joblib"
        if not model_path.exists():
            results["issues"].append("Model file not found")
            results["passed"] = False
        else:
            results["artifacts"]["model_exists"] = True
    except Exception as e:
        results["issues"].append(f"Error checking model: {e}")
        results["passed"] = False
    
    return results
