

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

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from src.features_config import (
    AR_GENES,
    GLEASON_PRIMARY_COL,
    GLEASON_SECONDARY_COL,
    LYMPH_NODE_COL,
    MARGIN_COL,
    MIN_GENES_FOR_PATHWAY,
    PROLIF_GENES,
    PSA_GENES,
)
from src.io import logger


# =============================================================================
# 3-LAYER FEATURE ENGINEERING STRATEGY
# =============================================================================
# Layer 1: Raw Gene Filtering & Selection (Pre-Engineering)
# Layer 2: Domain-Specific Feature Engineering (Post-Selection)
# Layer 3: Final Feature Assembly & Alignment
# =============================================================================


# ---------------------------------------------------------------------------
# Layer 1: Raw Gene Filtering & Selection
# ---------------------------------------------------------------------------
def fit_layer1_gene_selector(
    X_raw: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    """Fit Layer 1 gene selector on training data ONLY.
    
    This layer operates ONLY on raw gene expression columns to select
    the most relevant genes before any feature engineering.
    
    Args:
        X_raw: Raw gene expression DataFrame (genes only, no clinical features)
        y_train: Target variable for MI scoring
        variance_threshold: Threshold for variance filtering
        mi_top_k: Number of top genes to select via MI
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing fitted selector components and selected genes
    """
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    X_imp = np.asarray(X_imp, dtype=np.float64)

    vt = VarianceThreshold(threshold=variance_threshold)
    X_var = vt.fit_transform(X_imp)
    var_features = X_raw.columns[vt.get_support()].tolist()

    if len(var_features) == 0:
        raise RuntimeError("VarianceThreshold removed all features")

    mi_scores = mutual_info_classif(X_var, y_train, random_state=random_state)
    mi_scores = pd.Series(mi_scores, index=var_features).sort_values(ascending=False)
    k = min(mi_top_k, len(mi_scores))
    mi_features = mi_scores.head(k).index.tolist()

    logger.info(
        "Layer 1 - Gene Selection: %d raw genes → %d after variance → %d MI-selected genes",
        len(X_raw.columns), len(var_features), len(mi_features),
    )

    return {
        "imputer": imputer,
        "variance_selector": vt,
        "variance_features": var_features,
        "mi_features": mi_features,
        "mi_scores": mi_scores,
        "is_fitted": True,
    }


def transform_layer1_genes(
    X_raw: pd.DataFrame,
    fitted_selector: dict[str, Any],
) -> pd.DataFrame:
    """Transform data using fitted Layer 1 gene selector.
    
    For external/test data: applies the same imputation and gene selection
    learned from training data. No re-fitting occurs.
    
    Args:
        X_raw: Raw gene expression DataFrame
        fitted_selector: Fitted selector dictionary from training
    
    Returns:
        DataFrame with selected genes only
    """
    if not fitted_selector.get("is_fitted", False):
        raise ValueError("Layer 1 selector must be fitted before transform")
    
    imputer = fitted_selector["imputer"]
    
    # Handle missing features in external data by adding them with NaN
    # This allows the imputer to handle them properly
    fit_feature_names = fitted_selector["variance_features"]
    missing_features = set(fit_feature_names) - set(X_raw.columns)
    
    if missing_features:
        logger.info(
            "Layer 1 - Transform: Adding %d missing features (will be imputed)",
            len(missing_features),
        )
        # Add missing columns filled with NaN (will be imputed)
        # Use .copy() to avoid SettingWithCopyWarning
        X_raw = X_raw.copy()
        for feat in missing_features:
            X_raw[feat] = np.nan
    
    # Reorder columns to match fit order
    X_raw_ordered = X_raw[fit_feature_names]
    
    X_imp = imputer.transform(X_raw_ordered)
    X_imp = pd.DataFrame(X_imp, columns=fit_feature_names, index=X_raw.index)
    
    selected_genes = fitted_selector["mi_features"]
    # Only select genes that exist in the transformed data
    valid_genes = [g for g in selected_genes if g in X_imp.columns]
    
    logger.info(
        "Layer 1 - Transform: %d genes → %d selected genes (%d missing in data)",
        len(X_raw.columns), len(valid_genes), len(selected_genes) - len(valid_genes),
    )
    
    return X_imp[valid_genes].copy()


# ---------------------------------------------------------------------------
# Layer 2: Domain-Specific Feature Engineering
# ---------------------------------------------------------------------------
def create_engineered_features(
    X: pd.DataFrame,
    selected_genes: Optional[List[str]] = None,
    clinical_cols: Optional[List[str]] = None,
    strict_mode: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create clinically meaningful engineered features (Layer 2).
    
    This function creates composite features using:
    1. Selected genes from Layer 1 (for pathway scores)
    2. Available clinical columns
    
    Handles missing genes gracefully for external validation datasets.
    
    Args:
        X: Input DataFrame (can contain both genes and clinical features)
        selected_genes: List of genes selected from Layer 1 (optional, used for pathway scores)
        clinical_cols: List of clinical column names to use for interactions
        strict_mode: If True, only create pathway scores if ALL required genes present
    
    Returns:
        Tuple of (DataFrame with new features, list of new feature names)
    """
    X = X.copy()
    created_features = []
    
    # Use selected_genes if provided, otherwise use default gene sets
    # This allows Layer 2 to work with the filtered gene set from Layer 1
    
    # ── 1. Gleason-based features ──
    if GLEASON_PRIMARY_COL in X.columns and GLEASON_SECONDARY_COL in X.columns:
        X['Gleason_Total'] = X[GLEASON_PRIMARY_COL] + X[GLEASON_SECONDARY_COL]
        X['High_Risk_Gleason'] = (
            (X[GLEASON_PRIMARY_COL] >= 4) | 
            (X[GLEASON_SECONDARY_COL] >= 4)
        ).astype(int)
        created_features.extend(['Gleason_Total', 'High_Risk_Gleason'])
        logger.info("Layer 2 - Engineered: Gleason_Total, High_Risk_Gleason")

    # ── 2. Margin × Lymph Node interaction ──
    if MARGIN_COL in X.columns and LYMPH_NODE_COL in X.columns:
        X['Margin_x_LymphNode'] = (
            X[MARGIN_COL].astype(float) * X[LYMPH_NODE_COL].astype(float)
        )
        created_features.append('Margin_x_LymphNode')
        logger.info("Layer 2 - Engineered: Margin_x_LymphNode")

    # ── 3. T-Stage risk score ──
    t_stage_cols = [c for c in X.columns if 'Tumor Stage Code_T3' in c or 'Tumor Stage Code_T4' in c]
    if len(t_stage_cols) >= 2:
        X['T_Stage_Risk'] = X[t_stage_cols].sum(axis=1)
        created_features.append('T_Stage_Risk')
        logger.info("Layer 2 - Engineered: T_Stage_Risk")

    # ── 4. PSA Pathway Score (gene expression) ──
    # Use selected_genes if provided to check availability
    psa_genes = list(PSA_GENES)
    if selected_genes is not None:
        # Only consider genes that passed Layer 1 selection
        psa_genes = [g for g in psa_genes if g in selected_genes]
    
    available_psa = [g for g in psa_genes if g in X.columns]
    
    if strict_mode:
        # STRICT MODE: Only create if ALL original genes are available
        if set(PSA_GENES).issubset(set(X.columns)):
            X['PSA_Pathway_Score'] = X[list(PSA_GENES)].mean(axis=1)
            created_features.append('PSA_Pathway_Score')
            logger.info(f"Layer 2 - Engineered: PSA_Pathway_Score (strict mode, {len(PSA_GENES)} genes)")
    else:
        # LENIENT MODE: Create if at least MIN_GENES_FOR_PATHWAY genes available
        if len(available_psa) >= MIN_GENES_FOR_PATHWAY:
            X['PSA_Pathway_Score'] = X[available_psa].mean(axis=1)
            created_features.append('PSA_Pathway_Score')
            logger.info(f"Layer 2 - Engineered: PSA_Pathway_Score (from {len(available_psa)} genes)")

    # ── 5. AR Signaling Score ──
    ar_genes = list(AR_GENES)
    if selected_genes is not None:
        ar_genes = [g for g in ar_genes if g in selected_genes]
    
    available_ar = [g for g in ar_genes if g in X.columns]
    
    if strict_mode:
        if set(AR_GENES).issubset(set(X.columns)):
            X['AR_Signaling_Score'] = X[list(AR_GENES)].mean(axis=1)
            created_features.append('AR_Signaling_Score')
            logger.info(f"Layer 2 - Engineered: AR_Signaling_Score (strict mode, {len(AR_GENES)} genes)")
    else:
        if len(available_ar) >= MIN_GENES_FOR_PATHWAY:
            X['AR_Signaling_Score'] = X[available_ar].mean(axis=1)
            created_features.append('AR_Signaling_Score')
            logger.info(f"Layer 2 - Engineered: AR_Signaling_Score (from {len(available_ar)} genes)")

    # ── 6. Proliferation Score ──
    prolif_genes = list(PROLIF_GENES)
    if selected_genes is not None:
        prolif_genes = [g for g in prolif_genes if g in selected_genes]
    
    available_prolif = [g for g in prolif_genes if g in X.columns]
    
    if strict_mode:
        if set(PROLIF_GENES).issubset(set(X.columns)):
            X['Proliferation_Score'] = X[list(PROLIF_GENES)].mean(axis=1)
            created_features.append('Proliferation_Score')
            logger.info(f"Layer 2 - Engineered: Proliferation_Score (strict mode, {len(PROLIF_GENES)} genes)")
    else:
        if len(available_prolif) >= MIN_GENES_FOR_PATHWAY:
            X['Proliferation_Score'] = X[available_prolif].mean(axis=1)
            created_features.append('Proliferation_Score')
            logger.info(f"Layer 2 - Engineered: Proliferation_Score (from {len(available_prolif)} genes)")

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
    
# ---------------------------------------------------------------------------
# Layer 3: Final Feature Assembly & Alignment
# ---------------------------------------------------------------------------
def assemble_layer3_features(
    X_selected_genes: pd.DataFrame,
    X_engineered: pd.DataFrame,
    engineered_feature_names: List[str],
    final_feature_order: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Layer 3: Assemble final feature set from selected genes and engineered features.
    
    This layer combines:
    1. Selected raw genes from Layer 1
    2. Engineered features from Layer 2
    
    And ensures consistent column ordering and handles missing values.
    
    Args:
        X_selected_genes: DataFrame with genes selected from Layer 1
        X_engineered: DataFrame with engineered features from Layer 2
        engineered_feature_names: List of engineered feature names to extract
        final_feature_order: Optional list specifying desired column order
    
    Returns:
        Tuple of (final DataFrame, list of all feature names)
    """
    # Extract only the engineered features we created
    available_engineered = [f for f in engineered_feature_names if f in X_engineered.columns]
    X_eng_subset = X_engineered[available_engineered].copy()
    
    logger.info(
        "Layer 3 - Assembly: %d selected genes + %d engineered features",
        len(X_selected_genes.columns), len(available_engineered),
    )
    
    # Concatenate selected genes and engineered features
    # Ensure same index
    assert X_selected_genes.index.equals(X_eng_subset.index), \
        "Index mismatch between gene and engineered feature DataFrames"
    
    X_final = pd.concat([X_selected_genes, X_eng_subset], axis=1)
    
    # Determine final feature order
    if final_feature_order is not None:
        # Use provided order, only including available features
        final_features = [f for f in final_feature_order if f in X_final.columns]
        X_final = X_final[final_features]
    else:
        # Default: genes first, then engineered features
        final_features = list(X_selected_genes.columns) + available_engineered
        X_final = X_final[final_features]
    
    # Handle any missing values in engineered features (fill with 0)
    if X_final.isnull().any().any():
        n_missing = X_final.isnull().sum().sum()
        logger.warning("Layer 3 - Found %d missing values, filling with 0", n_missing)
        X_final = X_final.fillna(0)
    
    return X_final, final_features


# =============================================================================
# Unified 3-Layer Strategy Function
# =============================================================================
def apply_3_layer_feature_engineering(
    X_raw: pd.DataFrame,
    y_train: Optional[pd.Series | np.ndarray] = None,
    fitted_layer1_selector: Optional[dict[str, Any]] = None,
    clinical_cols: Optional[List[str]] = None,
    *,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    random_state: int = config.RANDOM_STATE,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, List[str], dict[str, Any]]:
    """Apply the complete 3-Layer Feature Engineering strategy.
    
    This unified function implements the full 3-layer pipeline:
    - Layer 1: Raw Gene Filtering & Selection
    - Layer 2: Domain-Specific Feature Engineering  
    - Layer 3: Final Feature Assembly & Alignment
    
    For TRAINING data (is_training=True):
        - Fits Layer 1 selector using y_train
        - Creates engineered features using all available data
        - Returns fitted selector for later use on test/external data
    
    For TEST/EXTERNAL data (is_training=False):
        - Uses pre-fitted Layer 1 selector (fitted_layer1_selector required)
        - Creates engineered features using only genes that passed Layer 1
        - Aligns features to match training set structure
    
    Args:
        X_raw: Raw input DataFrame (genes + optional clinical columns)
        y_train: Target variable (required for training data)
        fitted_layer1_selector: Pre-fitted selector from training (required for test data)
        clinical_cols: List of clinical column names to preserve for engineering
        variance_threshold: Variance threshold for Layer 1 (training only)
        mi_top_k: Number of top genes for Layer 1 (training only)
        random_state: Random seed (training only)
        is_training: If True, fit Layer 1; if False, use fitted_layer1_selector
    
    Returns:
        Tuple of:
            - X_final: Transformed DataFrame with final features
            - final_features: List of final feature names
            - fitted_layer1_selector: Fitted selector (same as input if is_training=False)
    
    Raises:
        ValueError: If y_train is missing during training or fitted_layer1_selector
                   is missing during inference
    """
    # -------------------------------------------------------------------------
    # LAYER 1: Raw Gene Filtering & Selection
    # -------------------------------------------------------------------------
    # Identify gene columns (exclude known clinical columns)
    if clinical_cols is not None:
        gene_cols = [c for c in X_raw.columns if c not in clinical_cols]
    else:
        # Assume all columns are genes if no clinical cols specified
        gene_cols = list(X_raw.columns)
    
    X_genes = X_raw[gene_cols]
    
    if is_training:
        if y_train is None:
            raise ValueError("y_train is required for training data")
        
        # Fit Layer 1 selector on training data ONLY
        fitted_layer1_selector = fit_layer1_gene_selector(
            X_genes, y_train,
            variance_threshold=variance_threshold,
            mi_top_k=mi_top_k,
            random_state=random_state,
        )
        selected_genes = fitted_layer1_selector["mi_features"]
    else:
        if fitted_layer1_selector is None or not fitted_layer1_selector.get("is_fitted", False):
            raise ValueError("fitted_layer1_selector is required for non-training data")
        
        # Transform using pre-fitted selector (no re-fitting!)
        selected_genes = fitted_layer1_selector["mi_features"]
    
    # Apply Layer 1 transformation (gene selection)
    X_selected_genes = transform_layer1_genes(X_genes, fitted_layer1_selector)
    
    # -------------------------------------------------------------------------
    # LAYER 2: Domain-Specific Feature Engineering
    # -------------------------------------------------------------------------
    # Combine selected genes with clinical columns for engineering
    if clinical_cols is not None:
        X_for_engineering = pd.concat([X_selected_genes, X_raw[clinical_cols]], axis=1)
    else:
        X_for_engineering = X_selected_genes.copy()
    
    # Create engineered features using only selected genes
    X_engineered, engineered_feature_names = create_engineered_features(
        X_for_engineering,
        selected_genes=selected_genes,  # Pass selected genes for pathway scoring
        clinical_cols=clinical_cols,
        strict_mode=False,  # Lenient mode for external validation
    )
    
    # -------------------------------------------------------------------------
    # LAYER 3: Final Feature Assembly & Alignment
    # -------------------------------------------------------------------------
    X_final, final_features = assemble_layer3_features(
        X_selected_genes,
        X_engineered,
        engineered_feature_names,
        final_feature_order=None,  # Use default ordering
    )
    
    logger.info(
        "3-Layer Feature Engineering Complete: %d raw features → %d final features",
        len(X_raw.columns), len(final_features),
    )
    
    return X_final, final_features, fitted_layer1_selector


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