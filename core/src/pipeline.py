# """
# Pipeline orchestration for TCGA-PRAD BCR prediction.

# Implements nested cross-validation, model comparison, and full
# end-to-end pipeline execution. All feature selection and preprocessing
# happen inside CV folds to prevent data leakage.
# """

# from __future__ import annotations
# import sys
# from pathlib import Path
# project_root = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(project_root))

# import config as config
# from src.io import logger

# from typing import Any

# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold

# import config as config
# from src.io import logger
# from src.feature_selection import fit_filter_selector, transform_selected, pso_feature_select
# from src.models import (
#     MODEL_REGISTRY,
#     build_model,
#     make_xgb,
#     requires_xgb_safe,
#     xgb_safe_frame,
# )


# # ---------------------------------------------------------------------------
# # Nested CV: MI-only vs MI→PSO→XGBoost
# # ---------------------------------------------------------------------------
# def evaluate_nested_cv(
#     X_train: pd.DataFrame,
#     y_train: pd.Series | np.ndarray,
#     *,
#     outer_splits: int = config.OUTER_SPLITS,
#     variance_threshold: float = config.VARIANCE_THRESHOLD,
#     mi_top_k: int = config.MI_TOP_K,
#     pso_final_k: int = config.PSO_FINAL_K,
#     run_pso: bool = True,
#     random_state: int = config.RANDOM_STATE,
# ) -> pd.DataFrame:
#     """Run nested CV comparing MI-only baseline vs MI→PSO→XGBoost.

#     Each outer fold:
#         1. Fits Variance + MI selectors on the training portion only
#         2. Trains XGBoost on MI-selected features (baseline)
#         3. Runs Binary PSO inside the training portion (wrapper)
#         4. Trains XGBoost on PSO-selected features
#         5. Evaluates both on the held-out validation portion

#     Parameters
#     ----------
#     X_train : Training features (full training set).
#     y_train : Training target.
#     outer_splits : Number of outer CV folds.
#     variance_threshold : Minimum variance for feature filtering.
#     mi_top_k : Number of top MI features to keep.
#     pso_final_k : Target number of features after PSO.
#     run_pso : Whether to run the PSO step.
#     random_state : Random seed.

#     Returns
#     -------
#     DataFrame with per-fold results.
#     """
#     outer_cv = StratifiedKFold(
#         n_splits=outer_splits, shuffle=True, random_state=random_state,
#     )
#     records: list[dict[str, Any]] = []

#     for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_train, y_train), start=1):
#         Xtr = X_train.iloc[tr_idx].reset_index(drop=True)
#         Xva = X_train.iloc[va_idx].reset_index(drop=True)
#         ytr = y_train.iloc[tr_idx].reset_index(drop=True)
#         yva = y_train.iloc[va_idx].reset_index(drop=True)

#         # Step 1: Filter selection on training fold only
#         fs = fit_filter_selector(
#             Xtr, ytr,
#             variance_threshold=variance_threshold,
#             mi_top_k=mi_top_k,
#             random_state=random_state,
#         )
#         mi_feats = fs["mi_features"]
#         Xtr_mi = transform_selected(Xtr, fs, mi_feats)
#         Xva_mi = transform_selected(Xva, fs, mi_feats)

#         # Step 2: MI-only baseline
#         mi_model = make_xgb(ytr)
#         mi_model.fit(xgb_safe_frame(Xtr_mi), ytr)
#         p_mi = mi_model.predict_proba(xgb_safe_frame(Xva_mi))[:, 1]
#         auc_mi = float(roc_auc_score(yva, p_mi))

#         # Step 3: PSO wrapper
#         if run_pso:
#             pso_feats, pso_fit = pso_feature_select(
#                 Xtr, ytr, mi_feats,
#                 n_features=min(pso_final_k, len(mi_feats)),
#                 random_state=random_state + fold,
#             )
#             Xtr_pso = transform_selected(Xtr, fs, pso_feats)
#             Xva_pso = transform_selected(Xva, fs, pso_feats)
#             pso_model = make_xgb(ytr)
#             pso_model.fit(xgb_safe_frame(Xtr_pso), ytr)
#             p_pso = pso_model.predict_proba(xgb_safe_frame(Xva_pso))[:, 1]
#             auc_pso = float(roc_auc_score(yva, p_pso))
#         else:
#             pso_feats, pso_fit, auc_pso = [], np.nan, np.nan

#         records.append({
#             "fold": fold,
#             "n_var_features": len(fs["variance_features"]),
#             "n_mi_features": len(mi_feats),
#             "n_pso_features": len(pso_feats),
#             "mi_auc": auc_mi,
#             "pso_inner_fitness": pso_fit,
#             "pso_auc": auc_pso,
#         })
#         logger.info(
#             "Fold %d: MI AUC=%.4f | PSO AUC=%.4f | PSO features=%d",
#             fold, auc_mi, auc_pso, len(pso_feats),
#         )

#     results = pd.DataFrame(records)
#     logger.info(
#         "Nested CV summary: MI mean=%.4f±%.4f | PSO mean=%.4f±%.4f",
#         results["mi_auc"].mean(), results["mi_auc"].std(ddof=0),
#         results["pso_auc"].mean(), results["pso_auc"].std(ddof=0),
#     )
#     return results


# # ---------------------------------------------------------------------------
# # Multi-model comparison via nested CV
# # ---------------------------------------------------------------------------
# def compare_models_nested_cv(
#     X_train: pd.DataFrame,
#     y_train: pd.Series | np.ndarray,
#     selected_features: list[str],
#     *,
#     outer_splits: int = config.OUTER_SPLITS,
#     random_state: int = config.RANDOM_STATE,
# ) -> pd.DataFrame:
#     """Compare all registered models using outer CV on selected features.

#     Parameters
#     ----------
#     X_train : Training features (full training set).
#     y_train : Training target.
#     selected_features : Pre-selected feature names.
#     outer_splits : Number of outer CV folds.
#     random_state : Random seed.

#     Returns
#     -------
#     DataFrame with columns [fold, model, auc].
#     """
#     outer_cv = StratifiedKFold(
#         n_splits=outer_splits, shuffle=True, random_state=random_state,
#     )
#     model_names = list(MODEL_REGISTRY.keys())
#     records: list[dict[str, Any]] = []

#     for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_train, y_train), start=1):
#         Xtr = X_train.iloc[tr_idx][selected_features].reset_index(drop=True)
#         Xva = X_train.iloc[va_idx][selected_features].reset_index(drop=True)
#         ytr = y_train.iloc[tr_idx].reset_index(drop=True)
#         yva = y_train.iloc[va_idx].reset_index(drop=True)

#         for name in model_names:
#             try:
#                 model = build_model(name, y_train=ytr)
#                 if requires_xgb_safe(name):
#                     model.fit(xgb_safe_frame(Xtr), ytr)
#                     p = model.predict_proba(xgb_safe_frame(Xva))[:, 1]
#                 else:
#                     model.fit(Xtr, ytr)
#                     p = model.predict_proba(Xva)[:, 1]
#                 auc = float(roc_auc_score(yva, p))
#             except Exception as e:
#                 logger.warning("Model '%s' failed on fold %d: %s", name, fold, e)
#                 auc = np.nan

#             records.append({"fold": fold, "model": name, "auc": auc})
#             logger.info("  Fold %d | %s: AUC=%.4f", fold, name, auc)

#     results = pd.DataFrame(records)

#     summary = results.groupby("model")["auc"].agg(["mean", "std"]).sort_values("mean", ascending=False)
#     logger.info("Model comparison summary:\n%s", summary.to_string())

#     return results


# # ---------------------------------------------------------------------------
# # Final evaluation on held-out test
# # ---------------------------------------------------------------------------
# def evaluate_final_model(
#     model: Any,
#     X_test: pd.DataFrame,
#     y_test: pd.Series | np.ndarray,
#     model_name: str = "Final Model",
#     needs_xgb_safe: bool = False,
# ) -> dict[str, float]:
#     """Evaluate a fitted model on the held-out test set.

#     Parameters
#     ----------
#     model : Fitted model.
#     X_test : Test features (already preprocessed and feature-selected).
#     y_test : Test target.
#     model_name : Name for logging.
#     needs_xgb_safe : Whether to apply XGBoost-safe column names.

#     Returns
#     -------
#     Dictionary with all evaluation metrics.
#     """
#     from src.evaluation import compute_metrics

#     if needs_xgb_safe:
#         X_eval = xgb_safe_frame(X_test)
#     else:
#         X_eval = X_test

#     y_prob = model.predict_proba(X_eval)[:, 1]
#     y_pred = (y_prob >= 0.5).astype(int)

#     metrics = compute_metrics(y_test, y_pred, y_prob)
#     metrics["model"] = model_name

#     logger.info("=" * 60)
#     logger.info("FINAL TEST EVALUATION: %s", model_name)
#     logger.info("=" * 60)
#     for key, value in metrics.items():
#         if key != "model" and isinstance(value, float):
#             logger.info("  %s: %.4f", key, value)
#     logger.info("=" * 60)

#     return metrics

# ==================================




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