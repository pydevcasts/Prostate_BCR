"""Leakage-aware benchmark for clinical feature-selection strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold

import config
from src.clinical_selector import fit_clinical_selector
from src.genomic_selector import pso_feature_select_genomic
from src.models import make_xgb, xgb_safe_frame


def _impute_train_valid(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(X_train)
    valid_values = imputer.transform(X_valid)
    return (
        pd.DataFrame(train_values, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(valid_values, columns=X_valid.columns, index=X_valid.index),
        imputer,
    )


def _select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    pso_k: int,
    random_state: int,
    pso_n_particles: int,
    pso_n_iterations: int,
) -> list[str]:
    if strategy.startswith("all_"):
        return X_train.columns.tolist()

    X_imputed, _, _ = _impute_train_valid(X_train, X_train)
    variance_selector = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
    X_variance = variance_selector.fit_transform(X_imputed)
    variance_features = X_train.columns[variance_selector.get_support()].tolist()

    if strategy == "variance_pso":
        candidate_features = variance_features
    elif strategy == "mi_pso":
        fitted_selector = fit_clinical_selector(
            X_train,
            y_train,
            variance_threshold=config.VARIANCE_THRESHOLD,
            mi_top_k=config.MI_TOP_K,
            random_state=random_state,
        )
        candidate_features = fitted_selector["mi_features"]
    else:
        raise ValueError(f"Unknown clinical benchmark strategy: {strategy}")

    selected, _, _, _ = pso_feature_select_genomic(
        X_train,
        y_train,
        candidate_features,
        n_features=min(pso_k, len(candidate_features)),
        n_particles=pso_n_particles,
        n_iterations=pso_n_iterations,
        random_state=random_state + 2000,
        branch_name="Clinical benchmark",
    )
    return selected


def _make_model(model_name: str, y_train: pd.Series) -> Any:
    if model_name == "xgboost":
        return make_xgb(y_train)
    if model_name == "logistic":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
    raise ValueError(f"Unknown clinical benchmark model: {model_name}")


def benchmark_clinical_strategies(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 3,
    pso_k_values: tuple[int, ...] = (10, 20, 30),
    random_state: int = config.RANDOM_STATE,
    pso_n_particles: int = config.PSO_N_PARTICLES,
    pso_n_iterations: int = config.PSO_N_ITERATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare clinical strategies with feature selection inside each CV fold."""
    strategies = [
        ("all_features", "xgboost", None),
        ("all_features", "logistic", None),
        *[("variance_pso", "xgboost", k) for k in pso_k_values],
        ("mi_pso", "xgboost", config.PSO_FINAL_K),
    ]
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []

    for strategy, model_name, pso_k in strategies:
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]
            selected = _select_features(
                X_train,
                y_train,
                strategy,
                pso_k or len(X_train.columns),
                random_state + fold,
                pso_n_particles,
                pso_n_iterations,
            )
            X_train_selected, X_valid_selected, _ = _impute_train_valid(
                X_train[selected], X_valid[selected]
            )
            model = _make_model(model_name, y_train)
            if model_name == "xgboost":
                model.fit(xgb_safe_frame(X_train_selected), y_train)
                probabilities = model.predict_proba(
                    xgb_safe_frame(X_valid_selected)
                )[:, 1]
            else:
                model.fit(X_train_selected, y_train)
                probabilities = model.predict_proba(X_valid_selected)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            rows.append({
                "strategy": strategy if pso_k is None else f"{strategy}_k{pso_k}",
                "model": model_name,
                "fold": fold,
                "n_features": len(selected),
                "roc_auc": roc_auc_score(y_valid, probabilities),
                "pr_auc": average_precision_score(y_valid, probabilities),
                "precision": precision_score(y_valid, predictions, zero_division=0),
                "recall": recall_score(y_valid, predictions, zero_division=0),
                "f1": f1_score(y_valid, predictions, zero_division=0),
            })

    fold_results = pd.DataFrame(rows)
    metric_columns = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    summary = (
        fold_results.groupby(["strategy", "model"], as_index=False)
        .agg(
            n_features=("n_features", "mean"),
            **{f"{metric}_mean": (metric, "mean") for metric in metric_columns},
            **{f"{metric}_std": (metric, "std") for metric in metric_columns},
        )
        .sort_values("roc_auc_mean", ascending=False)
    )
    return summary, fold_results