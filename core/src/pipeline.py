

"""
Pipeline utilities for Prostate BCR prediction.
Provides nested cross-validation, model comparison, and evaluation helpers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score

from src.io import logger
from src.models import build_model, xgb_safe_frame, requires_xgb_safe


def compare_models_nested_cv(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model_names: Optional[List[str]] = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare multiple models using honest nested cross-validation.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_names: List of model names to compare
        outer_splits: Number of outer CV folds
        inner_splits: Number of inner CV folds for tuning
        scoring: Scoring metric
        random_state: Random seed
    
    Returns:
        DataFrame with mean/std scores for each model
    """
    if model_names is None:
        model_names = ["XGBoost", "RandomForest", "LogisticRegression"]
    
    results = []
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    
    for model_name in model_names:
        logger.info(f"Evaluating {model_name} with {outer_splits}-fold nested CV...")
        fold_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                # Build model with default/hyper-tuned params
                model = build_model(model_name, y_train=y_tr if model_name in ["XGBoost", "LightGBM"] else None)
                
                # Apply safe frame if needed
                if requires_xgb_safe(model_name):
                    X_tr_safe = xgb_safe_frame(X_tr)
                    X_te_safe = xgb_safe_frame(X_te)
                else:
                    X_tr_safe = X_tr
                    X_te_safe = X_te
                
                model.fit(X_tr_safe, y_tr)
                y_prob = model.predict_proba(X_te_safe)[:, 1]
                
                if scoring == "roc_auc":
                    score = roc_auc_score(y_te, y_prob)
                elif scoring == "average_precision":
                    score = average_precision_score(y_te, y_prob)
                else:
                    score = roc_auc_score(y_te, y_prob)
                    
                fold_scores.append(score)
                logger.debug(f"  Fold {fold_idx+1}: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"  Fold {fold_idx+1} failed for {model_name}: {e}")
                continue
        
        if fold_scores:
            results.append({
                "model": model_name,
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores)),
                "n_folds": len(fold_scores),
                "scores": fold_scores
            })
        else:
            logger.error(f"All folds failed for {model_name}")
    
    return pd.DataFrame(results).sort_values("mean_score", ascending=False).reset_index(drop=True)


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model_name: str = "Unknown",
) -> Dict[str, float]:
    """
    Evaluate a trained model on given data.
    
    Returns dict with ROC-AUC, PR-AUC, Accuracy, etc.
    """
    if requires_xgb_safe(model_name):
        X_safe = xgb_safe_frame(X)
    else:
        X_safe = X
    
    y_prob = model.predict_proba(X_safe)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "accuracy": float((y_pred == y).mean()),
        "sensitivity": float(((y_pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "specificity": float(((y_pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
    }
    
    logger.info(f"{model_name} Evaluation: AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}, Acc={metrics['accuracy']:.4f}")
    return metrics