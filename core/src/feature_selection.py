

"""
Feature selection for TCGA-PRAD BCR prediction.

Implements the full feature selection pipeline:
    1. Variance Threshold — remove near-constant features
    2. Mutual Information — rank features by relevance to target
    3. Feature Engineering — create interaction & pathway features
    4. Binary PSO — wrapper selection with inner CV fitness & penalty

All selectors must be fitted on training data only.
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
# Feature Engineering: Interaction & Pathway Features
# ---------------------------------------------------------------------------
def create_engineered_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create clinically meaningful engineered features.
    
    Compensates for removed post-operative PSA (leakage) by capturing
    similar biological information through pre-operative clinical variables
    and gene expression pathways.
    
    Returns:
        Tuple of (DataFrame with new features, list of new feature names)
    """
    X = X.copy()
    created_features = []
    
    # ── 1. Gleason-based features ──
    if 'Gleason pattern primary' in X.columns and 'Gleason pattern secondary' in X.columns:
        X['Gleason_Total'] = X['Gleason pattern primary'] + X['Gleason pattern secondary']
        X['High_Risk_Gleason'] = (
            (X['Gleason pattern primary'] >= 4) | 
            (X['Gleason pattern secondary'] >= 4)
        ).astype(int)
        created_features.extend(['Gleason_Total', 'High_Risk_Gleason'])
        logger.info("Engineered: Gleason_Total, High_Risk_Gleason")

    # ── 2. Margin × Lymph Node interaction ──
    margin_col = 'Surgical Margin Resection Status_R1'
    lymph_col = 'Primary Lymph Node Presentation Assessment Ind-3_YES'
    if margin_col in X.columns and lymph_col in X.columns:
        X['Margin_x_LymphNode'] = (
            X[margin_col].astype(float) * X[lymph_col].astype(float)
        )
        created_features.append('Margin_x_LymphNode')
        logger.info("Engineered: Margin_x_LymphNode")

    # ── 3. T-Stage risk score ──
    t_stage_cols = [c for c in X.columns if 'Tumor Stage Code_T3' in c or 'Tumor Stage Code_T4' in c]
    if len(t_stage_cols) >= 2:
        X['T_Stage_Risk'] = X[t_stage_cols].sum(axis=1)
        created_features.append('T_Stage_Risk')
        logger.info("Engineered: T_Stage_Risk")

    # ── 4. PSA Pathway Score (gene expression) ──
    psa_genes = ['KLK3', 'KLK2', 'ACPP', 'TMPRSS2', 'AR', 'NKX3-1', 'STEAP2']
    available_psa = [g for g in psa_genes if g in X.columns]
    if len(available_psa) >= 3:
        X['PSA_Pathway_Score'] = X[available_psa].mean(axis=1)
        created_features.append('PSA_Pathway_Score')
        logger.info(f"Engineered: PSA_Pathway_Score (from {len(available_psa)} genes)")

    # ── 5. AR Signaling Score ──
    ar_genes = ['AR', 'FKBP5', 'KLK3', 'KLK2', 'TMPRSS2', 'NKX3-1', 'STEAP2', 'CAMKK2']
    available_ar = [g for g in ar_genes if g in X.columns]
    if len(available_ar) >= 3:
        X['AR_Signaling_Score'] = X[available_ar].mean(axis=1)
        created_features.append('AR_Signaling_Score')
        logger.info(f"Engineered: AR_Signaling_Score (from {len(available_ar)} genes)")

    # ── 6. Proliferation Score ──
    prolif_genes = ['MKI67', 'TOP2A', 'CCNB1', 'CCNE1', 'CDK1', 'AURKA', 'BIRC5']
    available_prolif = [g for g in prolif_genes if g in X.columns]
    if len(available_prolif) >= 3:
        X['Proliferation_Score'] = X[available_prolif].mean(axis=1)
        created_features.append('Proliferation_Score')
        logger.info(f"Engineered: Proliferation_Score (from {len(available_prolif)} genes)")

    return X, created_features


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
    
    # Only select features that exist in the transformed data
    valid_features = [f for f in selected if f in X_imp.columns]
    return X_imp[valid_features].copy()


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
    penalty_alpha: float = config.PSO_PENALTY_ALPHA,
    random_state: int = config.RANDOM_STATE,
) -> tuple[list[str], float]:
    """Fixed-cardinality Binary PSO for feature selection WITH PENALTY."""
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
        
        # Apply penalty
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
            raw_auc = gbest_score + (penalty_alpha * n_features)
            logger.info(
                "  PSO iter %d/%d: best Fitness=%.4f (Raw AUC ≈ %.4f)",
                iteration + 1, n_iterations, gbest_score, raw_auc,
            )

    selected = [candidate_features[i] for i in np.flatnonzero(gbest_pos)]
    logger.info("PSO selected %d features with final fitness=%.4f", len(selected), gbest_score)

    return selected, gbest_score


# ---------------------------------------------------------------------------
# Full pipeline: Variance → MI → Engineering → PSO
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
    """Run the full feature selection pipeline on training data.

    Steps:
        1. Variance Threshold
        2. Mutual Information (top-k)
        3. Feature Engineering (interaction/pathway features)
        4. Binary PSO (optional)
    """
    # Step 1 & 2: Filter selection
    fitted_selector = fit_filter_selector(
        X_train, y_train,
        variance_threshold=variance_threshold,
        mi_top_k=mi_top_k,
        random_state=random_state,
    )

    mi_features = fitted_selector["mi_features"]
    
    # Step 3: Feature Engineering
    # Create engineered features on the FULL training data first
    X_train_eng, engineered_feature_names = create_engineered_features(X_train)
    
    # Add engineered features to the candidate pool for PSO
    # (They will compete with MI-selected features)
    candidate_pool = list(set(mi_features + engineered_feature_names))
    
    # Ensure all candidate features exist in the engineered dataframe
    candidate_pool = [f for f in candidate_pool if f in X_train_eng.columns]
    
    logger.info(
        "Candidate pool: %d MI features + %d engineered features = %d total candidates",
        len(mi_features), len(engineered_feature_names), len(candidate_pool)
    )

    # Step 4: PSO Selection
    if run_pso:
        final_features, pso_score = pso_feature_select(
            X_train_eng,  # Use engineered dataframe
            y_train,
            candidate_pool,
            n_features=min(pso_final_k, len(candidate_pool)),
            random_state=random_state + 1000,
        )
    else:
        # If PSO is disabled, use top MI features + all engineered features
        final_features = mi_features[:pso_final_k] + engineered_feature_names
        pso_score = np.nan

    logger.info(
        "Feature selection complete: %d → %d → %d features (including engineered)",
        len(fitted_selector["variance_features"]),
        len(mi_features),
        len(final_features),
    )

    # Update fitted_selector to include engineered info
    fitted_selector["engineered_features"] = engineered_feature_names
    fitted_selector["X_train_engineered"] = X_train_eng  # Store for transform
    
    return fitted_selector, final_features