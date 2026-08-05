"""
Model definitions for TCGA-PRAD BCR prediction.

Implements six classifiers: Logistic Regression, Random Forest, SVM,
XGBoost, LightGBM, and CatBoost. Each factory returns a configured
estimator with hyperparameters sourced from config.py.

XGBoost column-name sanitization is handled here because XGBoost rejects
certain characters ([, ], <, >) in feature names.

Hyperparameter tuning is provided via tune_model() using
RandomizedSearchCV, fitted exclusively on training data.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import config
from src.io import logger


# ---------------------------------------------------------------------------
# XGBoost column-name sanitization
# ---------------------------------------------------------------------------
def _sanitize_xgb_name(name: str) -> str:
    """Replace characters that XGBoost rejects in feature names."""
    name = str(name)
    name = re.sub(r"[\[\]<>]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def xgb_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with XGBoost-safe, unique column names.

    Collisions are resolved by appending a numeric suffix.
    """
    out = df.copy()
    used: dict[str, str] = {}
    safe_names: list[str] = []

    for original in out.columns:
        base = _sanitize_xgb_name(original) or "feature"
        candidate = base
        counter = 1
        while candidate in used:
            counter += 1
            candidate = f"{base}_{counter}"
        used[candidate] = original
        safe_names.append(candidate)

    out.columns = safe_names
    return out


def xgb_feature_name_map(columns: pd.Index | list[str]) -> pd.DataFrame:
    """Build a mapping from original feature names to XGBoost-safe names.

    Useful for translating SHAP / feature-importance output back to
    the original column names for publication figures.
    """
    rows: list[dict[str, str]] = []
    used: dict[str, str] = {}

    for original in columns:
        base = _sanitize_xgb_name(original) or "feature"
        candidate = base
        counter = 1
        while candidate in used:
            counter += 1
            candidate = f"{base}_{counter}"
        used[candidate] = original
        rows.append({"original_feature": str(original), "xgb_feature": candidate})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------
def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Compute the negative/positive ratio for class-weight balancing."""
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return n_neg / n_pos


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_logistic(**overrides: Any) -> LogisticRegression:
    """Create a Logistic Regression classifier.

    Note: sklearn >= 1.8 deprecates penalty='l2' (which is the default);
    we drop it silently to avoid FutureWarning noise in the notebooks.
    """
    params = dict(config.LOGISTIC_PARAMS)
    params.update(overrides)
    if params.get("penalty") == "l2":
        params.pop("penalty")
    return LogisticRegression(**params)


def make_random_forest(**overrides: Any) -> RandomForestClassifier:
    """Create a Random Forest classifier."""
    params = dict(config.RANDOM_FOREST_PARAMS)
    params.update(overrides)
    return RandomForestClassifier(**params)


def make_svm(**overrides: Any) -> SVC:
    """Create a Support Vector Machine classifier with probability support."""
    params = dict(config.SVM_PARAMS)
    params.update(overrides)
    return SVC(**params)


def make_xgb(y_fit: pd.Series | np.ndarray, **overrides: Any) -> Any:
    """Create an XGBoost classifier with automatic class-weight balancing.

    Parameters
    ----------
    y_fit : Training target used to compute scale_pos_weight.
    **overrides : Any XGBClassifier parameter to override defaults.
    """
    from xgboost import XGBClassifier

    params = dict(config.XGBOOST_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(y_fit)
    params.update(overrides)
    return XGBClassifier(**params)


def make_lightgbm(y_fit: pd.Series | np.ndarray, **overrides: Any) -> Any:
    """Create a LightGBM classifier with automatic class-weight balancing.

    Parameters
    ----------
    y_fit : Training target used to compute scale_pos_weight.
    **overrides : Any LGBMClassifier parameter to override defaults.
    """
    from lightgbm import LGBMClassifier

    params = dict(config.LIGHTGBM_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(y_fit)
    params.update(overrides)
    return LGBMClassifier(**params)


def make_catboost(**overrides: Any) -> Any:
    """Create a CatBoost classifier.

    Parameters
    ----------
    **overrides : Any CatBoostClassifier parameter to override defaults.
    """
    from catboost import CatBoostClassifier

    params = dict(config.CATBOOST_PARAMS)
    params.update(overrides)
    return CatBoostClassifier(**params)


# ---------------------------------------------------------------------------
# Model registry — unified access point
# ---------------------------------------------------------------------------
ModelFactory = Callable[..., Any]

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "Logistic Regression": {
        "factory": make_logistic,
        "needs_y": False,
        "needs_xgb_safe": False,
    },
    "Random Forest": {
        "factory": make_random_forest,
        "needs_y": False,
        "needs_xgb_safe": False,
    },
    "SVM": {
        "factory": make_svm,
        "needs_y": False,
        "needs_xgb_safe": False,
    },
    "XGBoost": {
        "factory": make_xgb,
        "needs_y": True,
        "needs_xgb_safe": True,
    },
    "LightGBM": {
        "factory": make_lightgbm,
        "needs_y": True,
        "needs_xgb_safe": False,
    },
    "CatBoost": {
        "factory": make_catboost,
        "needs_y": False,
        "needs_xgb_safe": False,
    },
}


def build_model(
    name: str,
    y_train: pd.Series | np.ndarray | None = None,
    **overrides: Any,
) -> Any:
    """Build a model by name from the registry.

    Parameters
    ----------
    name : Model name (must exist in MODEL_REGISTRY).
    y_train : Training target (required for XGBoost and LightGBM).
    **overrides : Hyperparameter overrides passed to the factory.

    Returns
    -------
    Configured model instance, ready for .fit().

    Raises
    ------
    ValueError : If model name is unknown or y_train is missing.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[name]

    if entry["needs_y"] and y_train is None:
        raise ValueError(f"Model '{name}' requires y_train to compute class weights")

    if entry["needs_y"]:
        model = entry["factory"](y_train, **overrides)
    else:
        model = entry["factory"](**overrides)

    logger.info("Built model: %s", name)
    return model


def get_all_model_names() -> list[str]:
    """Return the list of all registered model names."""
    return list(MODEL_REGISTRY.keys())


def requires_xgb_safe(name: str) -> bool:
    """Check whether a model requires XGBoost-safe column names."""
    if name not in MODEL_REGISTRY:
        return False
    return MODEL_REGISTRY[name]["needs_xgb_safe"]


# ---------------------------------------------------------------------------
# Hyperparameter tuning (RandomizedSearchCV)
# ---------------------------------------------------------------------------
def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    search_space: dict,
    *,
    n_iter: int = config.RANDOM_SEARCH_N_ITER,
    cv_splits: int = config.RANDOM_SEARCH_CV_SPLITS,
    scoring: str = "roc_auc",
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = config.N_JOBS,
) -> tuple[Any, Any]:
    """Tune hyperparameters via RandomizedSearchCV on training data ONLY.

    The search uses stratified inner CV and ROC-AUC scoring. The returned
    estimator is refitted on the FULL training set with the best params
    (refit=True), so it is ready for final test evaluation.

    Parameters
    ----------
    name : Model name from MODEL_REGISTRY (e.g. "XGBoost").
    X_train : Training features.
    y_train : Training target.
    search_space : Dict of parameter lists to sample from
        (e.g. config.XGBOOST_SEARCH_SPACE).
    n_iter : Number of random parameter settings to evaluate.
    cv_splits : Number of inner stratified CV folds.
    scoring : Scoring metric for the search.
    random_state : Seed for reproducibility.
    n_jobs : Parallel jobs.

    Returns
    -------
    Tuple of (best_estimator refitted on full train, fitted search object).
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    model = build_model(name, y_train=y_train)

    if requires_xgb_safe(name):
        X_train = xgb_safe_frame(X_train)

    inner_cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=inner_cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info("Tuned '%s': best CV %s = %.4f", name, scoring, search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    return search.best_estimator_, search