"""Feature selection for the clinical branch of Late Fusion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer

import config
from src.genomic_selector import pso_feature_select_genomic
from src.io import logger


def fit_clinical_selector(
    X_clinical: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    """Fit imputation, variance and mutual-information filters on clinical data."""
    imputer = SimpleImputer(strategy="median")
    X_imputed = np.asarray(imputer.fit_transform(X_clinical), dtype=np.float64)

    variance_selector = VarianceThreshold(threshold=variance_threshold)
    X_variance = variance_selector.fit_transform(X_imputed)
    variance_features = X_clinical.columns[variance_selector.get_support()].tolist()
    if not variance_features:
        raise RuntimeError("VarianceThreshold removed all clinical features")

    mi_scores = mutual_info_classif(
        X_variance,
        y,
        random_state=random_state,
    )
    mi_scores = pd.Series(mi_scores, index=variance_features).sort_values(ascending=False)
    mi_features = mi_scores.head(min(mi_top_k, len(mi_scores))).index.tolist()

    logger.info(
        "Clinical selector: %d variance -> %d MI features",
        len(variance_features),
        len(mi_features),
    )
    return {
        "imputer": imputer,
        "variance_selector": variance_selector,
        "variance_features": variance_features,
        "mi_features": mi_features,
        "mi_scores": mi_scores,
    }


def transform_clinical(
    X_clinical: pd.DataFrame,
    fitted_selector: dict[str, Any],
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Apply a fitted clinical selector and return selected columns."""
    X_imputed = fitted_selector["imputer"].transform(X_clinical)
    X_imputed = pd.DataFrame(
        X_imputed,
        columns=X_clinical.columns,
        index=X_clinical.index,
    )
    selected = fitted_selector["mi_features"] if features is None else features
    return X_imputed[[feature for feature in selected if feature in X_imputed.columns]].copy()


def run_clinical_feature_selection(
    X_train_clinical: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    pso_final_k: int = config.PSO_FINAL_K,
    run_pso: bool = True,
    random_state: int = config.RANDOM_STATE,
) -> tuple[dict[str, Any], list[str]]:
    """Run clinical Variance -> MI -> PSO selection on training data only."""
    fitted_selector = fit_clinical_selector(
        X_train_clinical,
        y_train,
        variance_threshold=variance_threshold,
        mi_top_k=mi_top_k,
        random_state=random_state,
    )
    mi_features = fitted_selector["mi_features"]

    if run_pso:
        (
            final_features,
            pso_score,
            fitness_history,
            raw_auc_history,
        ) = pso_feature_select_genomic(
            X_train_clinical,
            y_train,
            mi_features,
            n_features=min(pso_final_k, len(mi_features)),
            random_state=random_state + 2000,
            branch_name="Clinical",
        )
    else:
        final_features = mi_features[:pso_final_k]
        pso_score = np.nan
        fitness_history = []
        raw_auc_history = []

    fitted_selector.update({
        "pso_score": pso_score,
        "pso_fitness_history": fitness_history,
        "pso_raw_auc_history": raw_auc_history,
    })
    logger.info(
        "Clinical feature selection complete: %d -> %d -> %d features",
        len(fitted_selector["variance_features"]),
        len(mi_features),
        len(final_features),
    )
    return fitted_selector, final_features
