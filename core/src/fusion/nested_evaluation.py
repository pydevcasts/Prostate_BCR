"""Leakage-aware nested feature selection and late-fusion evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

import config
from src.clinical_selector import fit_clinical_selector, transform_clinical
from src.genomic_selector import fit_genomic_selector, pso_feature_select_genomic, transform_genomic
from src.io import logger
from src.models import xgb_safe_frame


def _select_branch_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    branch: str,
    mi_top_k: int,
    pso_final_k: int,
    random_state: int,
    run_pso: bool,
) -> tuple[Any, list[str]]:
    if branch == "genomic":
        selector = fit_genomic_selector(
            X_train,
            y_train,
            mi_top_k=mi_top_k,
            random_state=random_state,
        )
        candidate_features = selector["mi_features"]
        selector_transform = transform_genomic
    elif branch == "clinical":
        selector = fit_clinical_selector(
            X_train,
            y_train,
            mi_top_k=mi_top_k,
            random_state=random_state,
        )
        candidate_features = selector["mi_features"]
        selector_transform = transform_clinical
    else:
        raise ValueError("branch must be 'genomic' or 'clinical'")

    if run_pso:
        selected_features, score, history, raw_history = pso_feature_select_genomic(
            X_train,
            y_train,
            candidate_features,
            n_features=min(pso_final_k, len(candidate_features)),
            random_state=random_state + 1000,
            branch_name=f"Nested {branch}",
        )
        selector.update({
            "pso_score": score,
            "pso_fitness_history": history,
            "pso_raw_auc_history": raw_history,
        })
    else:
        selected_features = candidate_features[:pso_final_k]

    return selector, selected_features


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(xgb_safe_frame(X))[:, 1]


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def evaluate_nested_late_fusion(
    genomic_model: Any,
    clinical_model: Any,
    X_genomic: pd.DataFrame,
    X_clinical: pd.DataFrame,
    y: pd.Series,
    *,
    outer_splits: int = config.OUTER_SPLITS,
    mi_top_k: int = config.MI_TOP_K,
    pso_final_k: int = config.PSO_FINAL_K,
    run_pso: bool = True,
    random_state: int = config.RANDOM_STATE,
    n_weight_steps: int = 100,
) -> dict[str, Any]:
    """Evaluate late fusion with feature selection performed inside each fold.

    The supplied models are unfitted templates containing fixed hyperparameters.
    Feature selection and model fitting are repeated independently in every outer
    fold. Fusion weights and the operating threshold are estimated only after
    collecting held-out predictions for every row.
    """
    if len(X_genomic) != len(X_clinical) or len(X_genomic) != len(y):
        raise ValueError("Genomic, clinical, and target lengths must match")
    if n_weight_steps < 1:
        raise ValueError("n_weight_steps must be at least 1")

    y_array = np.asarray(y)
    oof_genomic = np.full(len(y_array), np.nan, dtype=float)
    oof_clinical = np.full(len(y_array), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_genomic, y_array), start=1):
        X_genomic_train = X_genomic.iloc[train_idx]
        X_genomic_valid = X_genomic.iloc[valid_idx]
        X_clinical_train = X_clinical.iloc[train_idx]
        X_clinical_valid = X_clinical.iloc[valid_idx]
        y_train = y_array[train_idx]
        y_valid = y_array[valid_idx]

        genomic_selector, genomic_features = _select_branch_features(
            X_genomic_train,
            y_train,
            branch="genomic",
            mi_top_k=mi_top_k,
            pso_final_k=pso_final_k,
            random_state=random_state + fold,
            run_pso=run_pso,
        )
        clinical_selector, clinical_features = _select_branch_features(
            X_clinical_train,
            y_train,
            branch="clinical",
            mi_top_k=mi_top_k,
            pso_final_k=pso_final_k,
            random_state=random_state + fold,
            run_pso=run_pso,
        )
        X_genomic_train_selected = transform_genomic(
            X_genomic_train, genomic_selector, genomic_features,
        )
        X_genomic_valid_selected = transform_genomic(
            X_genomic_valid, genomic_selector, genomic_features,
        )
        X_clinical_train_selected = transform_clinical(
            X_clinical_train, clinical_selector, clinical_features,
        )
        X_clinical_valid_selected = transform_clinical(
            X_clinical_valid, clinical_selector, clinical_features,
        )

        fold_genomic_model = clone(genomic_model)
        fold_clinical_model = clone(clinical_model)
        fold_genomic_model.fit(xgb_safe_frame(X_genomic_train_selected), y_train)
        fold_clinical_model.fit(xgb_safe_frame(X_clinical_train_selected), y_train)
        genomic_probabilities = _predict(fold_genomic_model, X_genomic_valid_selected)
        clinical_probabilities = _predict(fold_clinical_model, X_clinical_valid_selected)
        oof_genomic[valid_idx] = genomic_probabilities
        oof_clinical[valid_idx] = clinical_probabilities
        fold_rows.append({
            "fold": fold,
            "n_genomic_features": len(genomic_features),
            "n_clinical_features": len(clinical_features),
            "genomic_auc": roc_auc_score(y_valid, genomic_probabilities),
            "clinical_auc": roc_auc_score(y_valid, clinical_probabilities),
        })
        logger.info(
            "Nested fold %d: genomic AUC=%.4f, clinical AUC=%.4f",
            fold,
            fold_rows[-1]["genomic_auc"],
            fold_rows[-1]["clinical_auc"],
        )

    if np.isnan(oof_genomic).any() or np.isnan(oof_clinical).any():
        raise RuntimeError("Nested evaluation left unfilled OOF predictions")

    weights = np.linspace(0.0, 1.0, n_weight_steps + 1)
    fusion_auc = [
        roc_auc_score(y_array, weight * oof_genomic + (1.0 - weight) * oof_clinical)
        for weight in weights
    ]
    best_index = int(np.argmax(fusion_auc))
    genomic_weight = float(weights[best_index])
    clinical_weight = 1.0 - genomic_weight
    oof_fusion = genomic_weight * oof_genomic + clinical_weight * oof_clinical
    _, true_positive_rate, thresholds = roc_curve(y_array, oof_fusion)
    false_positive_rate, _, _ = roc_curve(y_array, oof_fusion)
    threshold_index = int(np.argmax(true_positive_rate - false_positive_rate))
    threshold = float(thresholds[threshold_index])

    return {
        "fold_results": pd.DataFrame(fold_rows),
        "oof_genomic": oof_genomic,
        "oof_clinical": oof_clinical,
        "oof_fusion": oof_fusion,
        "genomic_weight": genomic_weight,
        "clinical_weight": clinical_weight,
        "oof_auc": float(fusion_auc[best_index]),
        "threshold": threshold,
        "metrics": _classification_metrics(y_array, oof_fusion, threshold),
    }
