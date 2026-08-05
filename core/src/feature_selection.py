"""
Feature selection for TCGA-PRAD BCR prediction.

Implements the full feature selection pipeline:
    1. Variance Threshold — remove near-constant features
    2. Mutual Information — rank features by relevance to target
    3. Binary PSO — wrapper selection with inner CV fitness AND penalty

All selectors must be fitted on training data only. This module provides
functions that can be embedded directly inside cross-validation loops
without data leakage.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from src.io import logger


# ---------------------------------------------------------------------------
# Filter-based selection: Variance + Mutual Information
# ---------------------------------------------------------------------------
def fit_filter_selector(
    X_fit: pd.DataFrame,
    y_fit: pd.Series | np.ndarray,
    *,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    """Fit Variance Threshold + Mutual Information selectors on training data."""
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_fit)
    X_imp = np.asarray(X_imp, dtype=np.float64)

    vt = VarianceThreshold(threshold=variance_threshold)
    X_var = vt.fit_transform(X_imp)
    var_features = X_fit.columns[vt.get_support()].tolist()

    if len(var_features) == 0:
        raise RuntimeError("VarianceThreshold removed all features")

    mi_scores = mutual_info_classif(X_var, y_fit, random_state=random_state)
    mi_scores = pd.Series(mi_scores, index=var_features).sort_values(ascending=False)
    k = min(mi_top_k, len(mi_scores))
    mi_features = mi_scores.head(k).index.tolist()

    logger.info(
        "Filter selector: %d variance → %d MI features",
        len(var_features), len(mi_features),
    )

    return {
        "imputer": imputer,
        "variance_selector": vt,
        "variance_features": var_features,
        "mi_features": mi_features,
        "mi_scores": mi_scores,
    }


def transform_selected(
    X_data: pd.DataFrame,
    fitted_selector: dict[str, Any],
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Transform data using a fitted selector (imputer + feature subset)."""
    imputer = fitted_selector["imputer"]
    X_imp = imputer.transform(X_data)
    X_imp = pd.DataFrame(X_imp, columns=X_data.columns, index=X_data.index)
    selected = fitted_selector["mi_features"] if features is None else features
    return X_imp[selected].copy()


# ---------------------------------------------------------------------------
# Binary PSO helpers
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def repair_exact_k(
    mask: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Repair a binary mask to have exactly k selected features."""
    idx = np.flatnonzero(mask)

    if len(idx) > k:
        keep = rng.choice(idx, size=k, replace=False)
        out = np.zeros_like(mask, dtype=int)
        out[keep] = 1
        return out

    if len(idx) < k:
        zero_idx = np.flatnonzero(mask == 0)
        add_n = min(k - len(idx), len(zero_idx))
        if add_n > 0:
            add = rng.choice(zero_idx, size=add_n, replace=False)
            mask = mask.copy()
            mask[add] = 1

    return mask


# ---------------------------------------------------------------------------
# Binary PSO feature selection (WITH PENALTY)
# ---------------------------------------------------------------------------
def pso_feature_select(
    X_fit: pd.DataFrame,
    y_fit: pd.Series | np.ndarray,
    candidate_features: list[str],
    *,
    n_features: int = config.PSO_FINAL_K,
    n_particles: int = config.PSO_N_PARTICLES,
    n_iterations: int = config.PSO_N_ITERATIONS,
    inner_splits: int = config.PSO_INNER_SPLITS,
    fitness_fn: Callable[[np.ndarray, pd.DataFrame, pd.Series], float] | None = None,
    w: float = config.PSO_W,
    c1: float = config.PSO_C1,
    c2: float = config.PSO_C2,
    penalty_alpha: float = config.PSO_PENALTY_ALPHA, # 👈 دریافت ضریب جریمه از کانفیگ
    random_state: int = config.RANDOM_STATE,
) -> tuple[list[str], float]:
    """Fixed-cardinality Binary PSO for feature selection WITH PENALTY.

    Fitness = Mean_CV_AUC - (penalty_alpha * number_of_selected_features)
    """
    from src.models import make_xgb, xgb_safe_frame

    candidate_features = list(candidate_features)

    if len(candidate_features) <= n_features:
        logger.info(
            "PSO skipped: %d candidates ≤ %d target features",
            len(candidate_features), n_features,
        )
        return candidate_features, np.nan

    Xc = X_fit[candidate_features].copy()
    imputer = SimpleImputer(strategy="median")
    Xc = pd.DataFrame(imputer.fit_transform(Xc), columns=candidate_features)

    n_dim = len(candidate_features)
    rng = np.random.RandomState(random_state)
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state,
    )

    # Cache to avoid re-evaluating identical feature subsets
    cache: dict[tuple[int, ...], float] = {}

    def default_fitness(mask: np.ndarray) -> float:
        """Default fitness: mean inner-CV ROC-AUC MINUS penalty."""
        repaired_mask = repair_exact_k(mask.astype(int), n_features, rng)
        key = tuple(np.flatnonzero(repaired_mask).tolist())

        if key in cache:
            return cache[key]

        cols = [candidate_features[i] for i in key]
        fold_scores = []

        for tr_idx, va_idx in inner_cv.split(Xc, y_fit):
            xtr = Xc.iloc[tr_idx][cols]
            xva = Xc.iloc[va_idx][cols]
            ytr = y_fit.iloc[tr_idx] if hasattr(y_fit, "iloc") else y_fit[tr_idx]
            yva = y_fit.iloc[va_idx] if hasattr(y_fit, "iloc") else y_fit[va_idx]

            model = make_xgb(ytr)
            model.fit(xgb_safe_frame(xtr), ytr)
            p = model.predict_proba(xgb_safe_frame(xva))[:, 1]
            fold_scores.append(roc_auc_score(yva, p))

        mean_auc = float(np.mean(fold_scores))
        
        # Penalty 
        num_selected = len(key)
        penalty = penalty_alpha * num_selected
        
        final_fitness = mean_auc - penalty
        
        cache[key] = final_fitness
        return final_fitness

    fitness = fitness_fn if fitness_fn is not None else default_fitness

    # Initialize particles
    position = rng.uniform(-1, 1, size=(n_particles, n_dim))
    velocity = rng.uniform(-0.1, 0.1, size=(n_particles, n_dim))
    binary = (sigmoid(position) > 0.5).astype(int)

    for i in range(n_particles):
        binary[i] = repair_exact_k(binary[i], n_features, rng)

    # Evaluate initial fitness
    pbest_pos = binary.copy()
    pbest_score = np.array([fitness(m) for m in binary])
    gbest_idx = int(np.argmax(pbest_score))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = float(pbest_score[gbest_idx])

    logger.info(
        "PSO: %d particles, %d iterations, target %d features, alpha=%.4f",
        n_particles, n_iterations, n_features, penalty_alpha,
    )

    # PSO main loop
    for iteration in range(n_iterations):
        r1 = rng.rand(n_particles, n_dim)
        r2 = rng.rand(n_particles, n_dim)

        velocity = w * velocity + c1 * r1 * (pbest_pos - binary) + c2 * r2 * (gbest_pos - binary)
        velocity = np.clip(velocity, -4, 4)

        prob = sigmoid(velocity)
        binary = (rng.rand(n_particles, n_dim) < prob).astype(int)

        for i in range(n_particles):
            binary[i] = repair_exact_k(binary[i], n_features, rng)

        scores = np.array([fitness(m) for m in binary])
        improved = scores > pbest_score
        pbest_pos[improved] = binary[improved]
        pbest_score[improved] = scores[improved]

        best_idx = int(np.argmax(pbest_score))
        if pbest_score[best_idx] > gbest_score:
            gbest_pos = pbest_pos[best_idx].copy()
            gbest_score = float(pbest_score[best_idx])

        if (iteration + 1) % 5 == 0 or iteration == n_iterations - 1:
            # محاسبه AUC خالص برای لاگ (بدون جریمه) جهت مقایسه بهتر
            raw_auc = gbest_score + (penalty_alpha * n_features)
            logger.info(
                "  PSO iter %d/%d: best Fitness=%.4f (Raw AUC ≈ %.4f)",
                iteration + 1, n_iterations, gbest_score, raw_auc,
            )

    selected = [candidate_features[i] for i in np.flatnonzero(gbest_pos)]
    logger.info("PSO selected %d features with final fitness=%.4f", len(selected), gbest_score)

    return selected, gbest_score


# ---------------------------------------------------------------------------
# Full pipeline: Variance → MI → PSO
# ---------------------------------------------------------------------------
def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    pso_final_k: int = config.PSO_FINAL_K,
    run_pso: bool = True,
    random_state: int = config.RANDOM_STATE,
) -> tuple[dict[str, Any], list[str]]:
    """Run the full feature selection pipeline on training data."""
    fitted_selector = fit_filter_selector(
        X_train, y_train,
        variance_threshold=variance_threshold,
        mi_top_k=mi_top_k,
        random_state=random_state,
    )

    mi_features = fitted_selector["mi_features"]

    if run_pso:
        final_features, pso_score = pso_feature_select(
            X_train, y_train,
            mi_features,
            n_features=min(pso_final_k, len(mi_features)),
            random_state=random_state + 1000,
        )
    else:
        final_features = mi_features
        pso_score = np.nan

    logger.info(
        "Feature selection complete: %d → %d → %d features",
        len(fitted_selector["variance_features"]),
        len(mi_features),
        len(final_features),
    )

    return fitted_selector, final_features