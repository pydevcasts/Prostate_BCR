

# """
# Feature selection for TCGA-PRAD BCR prediction.

# Implements the full feature selection pipeline:
#     1. Variance Threshold — remove near-constant features
#     2. Mutual Information — rank features by relevance to target
#     3. Feature Engineering — create interaction & pathway features
#     4. Binary PSO — wrapper selection with inner CV fitness & penalty

# All selectors must be fitted on training data only.
# """

# from __future__ import annotations

# from typing import Any, Callable, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold

# import config
# from src.features_config import (
#     AR_GENES,
#     GLEASON_PRIMARY_COL,
#     GLEASON_SECONDARY_COL,
#     LYMPH_NODE_COL,
#     MARGIN_COL,
#     MIN_GENES_FOR_PATHWAY,
#     PROLIF_GENES,
#     PSA_GENES,
# )
# from src.io import logger


# # ---------------------------------------------------------------------------
# # Feature Engineering: Interaction & Pathway Features
# # ---------------------------------------------------------------------------
# def create_engineered_features(
#     X: pd.DataFrame,
#     strict_mode: bool = False,
#     required_genes: Optional[Dict[str, List[str]]] = None,
# ) -> Tuple[pd.DataFrame, List[str]]:
#     """Create clinically meaningful engineered features.
    
#     Compensates for removed post-operative PSA (leakage) by capturing
#     similar biological information through pre-operative clinical variables
#     and gene expression pathways.
    
#     Args:
#         X: Input DataFrame
#         strict_mode: If True, only create pathway scores if ALL required genes present
#         required_genes: Optional dict mapping score name to required gene list.
#                        If provided, overrides default gene sets.
    
#     Returns:
#         Tuple of (DataFrame with new features, list of new feature names)
#     """
#     X = X.copy()
#     created_features = []
    
#     # ── 1. Gleason-based features ──
#     if GLEASON_PRIMARY_COL in X.columns and GLEASON_SECONDARY_COL in X.columns:
#         X['Gleason_Total'] = X[GLEASON_PRIMARY_COL] + X[GLEASON_SECONDARY_COL]
#         X['High_Risk_Gleason'] = (
#             (X[GLEASON_PRIMARY_COL] >= 4) | 
#             (X[GLEASON_SECONDARY_COL] >= 4)
#         ).astype(int)
#         created_features.extend(['Gleason_Total', 'High_Risk_Gleason'])
#         logger.info("Engineered: Gleason_Total, High_Risk_Gleason")

#     # ── 2. Margin × Lymph Node interaction ──
#     if MARGIN_COL in X.columns and LYMPH_NODE_COL in X.columns:
#         X['Margin_x_LymphNode'] = (
#             X[MARGIN_COL].astype(float) * X[LYMPH_NODE_COL].astype(float)
#         )
#         created_features.append('Margin_x_LymphNode')
#         logger.info("Engineered: Margin_x_LymphNode")

#     # ── 3. T-Stage risk score ──
#     t_stage_cols = [c for c in X.columns if 'Tumor Stage Code_T3' in c or 'Tumor Stage Code_T4' in c]
#     if len(t_stage_cols) >= 2:
#         X['T_Stage_Risk'] = X[t_stage_cols].sum(axis=1)
#         created_features.append('T_Stage_Risk')
#         logger.info("Engineered: T_Stage_Risk")

#     # ── 4. PSA Pathway Score (gene expression) ──
#     psa_genes = list(required_genes.get('PSA', PSA_GENES) if required_genes else PSA_GENES)
#     available_psa = [g for g in psa_genes if g in X.columns]
    
#     if strict_mode:
#         # STRICT MODE: Only create if ALL genes are available
#         if set(psa_genes).issubset(set(X.columns)):
#             X['PSA_Pathway_Score'] = X[psa_genes].mean(axis=1)
#             created_features.append('PSA_Pathway_Score')
#             logger.info(f"Engineered: PSA_Pathway_Score (strict mode, {len(psa_genes)} genes)")
#     else:
#         # LENIENT MODE: Create if at least MIN_GENES_FOR_PATHWAY genes available
#         if len(available_psa) >= MIN_GENES_FOR_PATHWAY:
#             X['PSA_Pathway_Score'] = X[available_psa].mean(axis=1)
#             created_features.append('PSA_Pathway_Score')
#             logger.info(f"Engineered: PSA_Pathway_Score (from {len(available_psa)} genes)")

#     # ── 5. AR Signaling Score ──
#     ar_genes = list(required_genes.get('AR', AR_GENES) if required_genes else AR_GENES)
#     available_ar = [g for g in ar_genes if g in X.columns]
    
#     if strict_mode:
#         if set(ar_genes).issubset(set(X.columns)):
#             X['AR_Signaling_Score'] = X[ar_genes].mean(axis=1)
#             created_features.append('AR_Signaling_Score')
#             logger.info(f"Engineered: AR_Signaling_Score (strict mode, {len(ar_genes)} genes)")
#     else:
#         if len(available_ar) >= MIN_GENES_FOR_PATHWAY:
#             X['AR_Signaling_Score'] = X[available_ar].mean(axis=1)
#             created_features.append('AR_Signaling_Score')
#             logger.info(f"Engineered: AR_Signaling_Score (from {len(available_ar)} genes)")

#     # ── 6. Proliferation Score ──
#     prolif_genes = list(required_genes.get('PROLIF', PROLIF_GENES) if required_genes else PROLIF_GENES)
#     available_prolif = [g for g in prolif_genes if g in X.columns]
    
#     if strict_mode:
#         if set(prolif_genes).issubset(set(X.columns)):
#             X['Proliferation_Score'] = X[prolif_genes].mean(axis=1)
#             created_features.append('Proliferation_Score')
#             logger.info(f"Engineered: Proliferation_Score (strict mode, {len(prolif_genes)} genes)")
#     else:
#         if len(available_prolif) >= MIN_GENES_FOR_PATHWAY:
#             X['Proliferation_Score'] = X[available_prolif].mean(axis=1)
#             created_features.append('Proliferation_Score')
#             logger.info(f"Engineered: Proliferation_Score (from {len(available_prolif)} genes)")

#     return X, created_features


# # ---------------------------------------------------------------------------
# # Filter-based selection: Variance + Mutual Information
# # ---------------------------------------------------------------------------
# def fit_filter_selector(
#     X_fit: pd.DataFrame,
#     y_fit: pd.Series | np.ndarray,
#     *,
#     variance_threshold: float = config.VARIANCE_THRESHOLD,
#     mi_top_k: int = config.MI_TOP_K,
#     random_state: int = config.RANDOM_STATE,
# ) -> dict[str, Any]:
#     """Fit Variance Threshold + Mutual Information selectors on training data."""
#     imputer = SimpleImputer(strategy="median")
#     X_imp = imputer.fit_transform(X_fit)
#     X_imp = np.asarray(X_imp, dtype=np.float64)

#     vt = VarianceThreshold(threshold=variance_threshold)
#     X_var = vt.fit_transform(X_imp)
#     var_features = X_fit.columns[vt.get_support()].tolist()

#     if len(var_features) == 0:
#         raise RuntimeError("VarianceThreshold removed all features")

#     mi_scores = mutual_info_classif(X_var, y_fit, random_state=random_state)
#     mi_scores = pd.Series(mi_scores, index=var_features).sort_values(ascending=False)
#     k = min(mi_top_k, len(mi_scores))
#     mi_features = mi_scores.head(k).index.tolist()

#     logger.info(
#         "Filter selector: %d variance → %d MI features",
#         len(var_features), len(mi_features),
#     )

#     return {
#         "imputer": imputer,
#         "variance_selector": vt,
#         "variance_features": var_features,
#         "mi_features": mi_features,
#         "mi_scores": mi_scores,
#     }


# def transform_selected(
#     X_data: pd.DataFrame,
#     fitted_selector: dict[str, Any],
#     features: list[str] | None = None,
# ) -> pd.DataFrame:
#     """Transform data using a fitted selector (imputer + feature subset)."""
#     imputer = fitted_selector["imputer"]
#     X_imp = imputer.transform(X_data)
#     X_imp = pd.DataFrame(X_imp, columns=X_data.columns, index=X_data.index)
#     selected = fitted_selector["mi_features"] if features is None else features
    
#     # Only select features that exist in the transformed data
#     valid_features = [f for f in selected if f in X_imp.columns]
#     return X_imp[valid_features].copy()


# # ---------------------------------------------------------------------------
# # Binary PSO helpers
# # ---------------------------------------------------------------------------
# def sigmoid(x: np.ndarray) -> np.ndarray:
#     """Numerically stable sigmoid function."""
#     return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# def repair_exact_k(
#     mask: np.ndarray,
#     k: int,
#     rng: np.random.RandomState,
# ) -> np.ndarray:
#     """Repair a binary mask to have exactly k selected features."""
#     idx = np.flatnonzero(mask)

#     if len(idx) > k:
#         keep = rng.choice(idx, size=k, replace=False)
#         out = np.zeros_like(mask, dtype=int)
#         out[keep] = 1
#         return out

#     if len(idx) < k:
#         zero_idx = np.flatnonzero(mask == 0)
#         add_n = min(k - len(idx), len(zero_idx))
#         if add_n > 0:
#             add = rng.choice(zero_idx, size=add_n, replace=False)
#             mask = mask.copy()
#             mask[add] = 1

#     return mask


# # ---------------------------------------------------------------------------
# # Binary PSO feature selection (WITH PENALTY)
# # ---------------------------------------------------------------------------
# def pso_feature_select(
#     X_fit: pd.DataFrame,
#     y_fit: pd.Series | np.ndarray,
#     candidate_features: list[str],
#     *,
#     n_features: int = config.PSO_FINAL_K,
#     n_particles: int = config.PSO_N_PARTICLES,
#     n_iterations: int = config.PSO_N_ITERATIONS,
#     inner_splits: int = config.PSO_INNER_SPLITS,
#     fitness_fn: Callable[[np.ndarray, pd.DataFrame, pd.Series], float] | None = None,
#     w: float = config.PSO_W,
#     c1: float = config.PSO_C1,
#     c2: float = config.PSO_C2,
#     penalty_alpha: float = config.PSO_PENALTY_ALPHA,
#     random_state: int = config.RANDOM_STATE,
# ) -> tuple[list[str], float]:
#     """Fixed-cardinality Binary PSO for feature selection WITH PENALTY."""
#     from src.models import make_xgb, xgb_safe_frame

#     candidate_features = list(candidate_features)

#     if len(candidate_features) <= n_features:
#         logger.info(
#             "PSO skipped: %d candidates ≤ %d target features",
#             len(candidate_features), n_features,
#         )
#         return candidate_features, np.nan

#     Xc = X_fit[candidate_features].copy()
#     imputer = SimpleImputer(strategy="median")
#     Xc = pd.DataFrame(imputer.fit_transform(Xc), columns=candidate_features)

#     n_dim = len(candidate_features)
#     rng = np.random.RandomState(random_state)
#     inner_cv = StratifiedKFold(
#         n_splits=inner_splits, shuffle=True, random_state=random_state,
#     )

#     cache: dict[tuple[int, ...], float] = {}

#     def default_fitness(mask: np.ndarray) -> float:
#         """Default fitness: mean inner-CV ROC-AUC MINUS penalty."""
#         repaired_mask = repair_exact_k(mask.astype(int), n_features, rng)
#         key = tuple(np.flatnonzero(repaired_mask).tolist())

#         if key in cache:
#             return cache[key]

#         cols = [candidate_features[i] for i in key]
#         fold_scores = []

#         for tr_idx, va_idx in inner_cv.split(Xc, y_fit):
#             xtr = Xc.iloc[tr_idx][cols]
#             xva = Xc.iloc[va_idx][cols]
#             ytr = y_fit.iloc[tr_idx] if hasattr(y_fit, "iloc") else y_fit[tr_idx]
#             yva = y_fit.iloc[va_idx] if hasattr(y_fit, "iloc") else y_fit[va_idx]

#             model = make_xgb(ytr)
#             model.fit(xgb_safe_frame(xtr), ytr)
#             p = model.predict_proba(xgb_safe_frame(xva))[:, 1]
#             fold_scores.append(roc_auc_score(yva, p))

#         mean_auc = float(np.mean(fold_scores))
        
#         # Apply penalty
#         num_selected = len(key)
#         penalty = penalty_alpha * num_selected
#         final_fitness = mean_auc - penalty
        
#         cache[key] = final_fitness
#         return final_fitness

#     fitness = fitness_fn if fitness_fn is not None else default_fitness

#     # Initialize particles
#     position = rng.uniform(-1, 1, size=(n_particles, n_dim))
#     velocity = rng.uniform(-0.1, 0.1, size=(n_particles, n_dim))
#     binary = (sigmoid(position) > 0.5).astype(int)

#     for i in range(n_particles):
#         binary[i] = repair_exact_k(binary[i], n_features, rng)

#     # Evaluate initial fitness
#     pbest_pos = binary.copy()
#     pbest_score = np.array([fitness(m) for m in binary])
#     gbest_idx = int(np.argmax(pbest_score))
#     gbest_pos = pbest_pos[gbest_idx].copy()
#     gbest_score = float(pbest_score[gbest_idx])

#     logger.info(
#         "PSO: %d particles, %d iterations, target %d features, alpha=%.4f",
#         n_particles, n_iterations, n_features, penalty_alpha,
#     )

#     # PSO main loop
#     for iteration in range(n_iterations):
#         r1 = rng.rand(n_particles, n_dim)
#         r2 = rng.rand(n_particles, n_dim)

#         velocity = w * velocity + c1 * r1 * (pbest_pos - binary) + c2 * r2 * (gbest_pos - binary)
#         velocity = np.clip(velocity, -4, 4)

#         prob = sigmoid(velocity)
#         binary = (rng.rand(n_particles, n_dim) < prob).astype(int)

#         for i in range(n_particles):
#             binary[i] = repair_exact_k(binary[i], n_features, rng)

#         scores = np.array([fitness(m) for m in binary])
#         improved = scores > pbest_score
#         pbest_pos[improved] = binary[improved]
#         pbest_score[improved] = scores[improved]

#         best_idx = int(np.argmax(pbest_score))
#         if pbest_score[best_idx] > gbest_score:
#             gbest_pos = pbest_pos[best_idx].copy()
#             gbest_score = float(pbest_score[best_idx])

#         if (iteration + 1) % 5 == 0 or iteration == n_iterations - 1:
#             raw_auc = gbest_score + (penalty_alpha * n_features)
#             logger.info(
#                 "  PSO iter %d/%d: best Fitness=%.4f (Raw AUC ≈ %.4f)",
#                 iteration + 1, n_iterations, gbest_score, raw_auc,
#             )

#     selected = [candidate_features[i] for i in np.flatnonzero(gbest_pos)]
#     logger.info("PSO selected %d features with final fitness=%.4f", len(selected), gbest_score)

#     return selected, gbest_score


# # ---------------------------------------------------------------------------
# # Full pipeline: Variance → MI → Engineering → PSO
# # ---------------------------------------------------------------------------
# def run_feature_selection(
#     X_train: pd.DataFrame,
#     y_train: pd.Series | np.ndarray,
#     *,
#     variance_threshold: float = config.VARIANCE_THRESHOLD,
#     mi_top_k: int = config.MI_TOP_K,
#     pso_final_k: int = config.PSO_FINAL_K,
#     run_pso: bool = True,
#     random_state: int = config.RANDOM_STATE,
# ) -> tuple[dict[str, Any], list[str]]:
#     """Run the full feature selection pipeline on training data.

#     Steps:
#         1. Variance Threshold
#         2. Mutual Information (top-k)
#         3. Feature Engineering (interaction/pathway features)
#         4. Binary PSO (optional)
#     """
#     # Step 1 & 2: Filter selection
#     fitted_selector = fit_filter_selector(
#         X_train, y_train,
#         variance_threshold=variance_threshold,
#         mi_top_k=mi_top_k,
#         random_state=random_state,
#     )

#     mi_features = fitted_selector["mi_features"]
    
#     # Step 3: Feature Engineering
#     # Create engineered features on the FULL training data first
#     X_train_eng, engineered_feature_names = create_engineered_features(X_train)
    
#     # Add engineered features to the candidate pool for PSO
#     # (They will compete with MI-selected features)
#     candidate_pool = list(set(mi_features + engineered_feature_names))
    
#     # Ensure all candidate features exist in the engineered dataframe
#     candidate_pool = [f for f in candidate_pool if f in X_train_eng.columns]
    
#     logger.info(
#         "Candidate pool: %d MI features + %d engineered features = %d total candidates",
#         len(mi_features), len(engineered_feature_names), len(candidate_pool)
#     )

#     # Step 4: PSO Selection
#     if run_pso:
#         final_features, pso_score = pso_feature_select(
#             X_train_eng,  # Use engineered dataframe
#             y_train,
#             candidate_pool,
#             n_features=min(pso_final_k, len(candidate_pool)),
#             random_state=random_state + 1000,
#         )
#     else:
#         # If PSO is disabled, use top MI features + all engineered features
#         final_features = mi_features[:pso_final_k] + engineered_feature_names
#         pso_score = np.nan

#     logger.info(
#         "Feature selection complete: %d → %d → %d features (including engineered)",
#         len(fitted_selector["variance_features"]),
#         len(mi_features),
#         len(final_features),
#     )

#     # Update fitted_selector to include engineered info
#     fitted_selector["engineered_features"] = engineered_feature_names
#     fitted_selector["X_train_engineered"] = X_train_eng  # Store for transform
    
#     return fitted_selector, final_features



# ===================================







"""
Feature selection for TCGA-PRAD BCR prediction using 3-Layer Strategy.

Implements:
    Layer 1: Raw Gene Filtering & Selection (Variance + MI)
    Layer 2: Extended Domain-Specific Feature Engineering (~20 features)
             + Automatic High-Correlation Removal (|r| > 0.9)
    Layer 3: Final Assembly (PSO Genes + Clean Engineered Features)
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from src.features_config import (
    AR_GENES, PROLIF_GENES, PSA_GENES,
    GLEASON_PRIMARY_COL, GLEASON_SECONDARY_COL,
    MARGIN_COL, LYMPH_NODE_COL, MIN_GENES_FOR_PATHWAY
)
from src.io import logger


# =============================================================================
# LAYER 2: EXTENDED FEATURE ENGINEERING + CORRELATION FILTER
# =============================================================================

def create_extended_engineered_features(
    X: pd.DataFrame,
    selected_genes: Optional[List[str]] = None,
    correlation_threshold: float = 0.90
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create ~20 engineered features across 3 sub-layers and remove multicollinearity.
    
    Args:
        X: Input DataFrame with genes and clinical columns
        selected_genes: Genes from Layer 1 (used to filter pathway genes)
        correlation_threshold: Max allowed pairwise correlation
        
    Returns:
        Tuple of (DataFrame with clean engineered features, list of feature names)
    """
    X = X.copy()
    created_features: List[str] = []

    # --- SUB-LAYER 2A: Base Clinical & Pathway Scores (7 features) ---
    if GLEASON_PRIMARY_COL in X.columns and GLEASON_SECONDARY_COL in X.columns:
        X['Gleason_Total'] = X[GLEASON_PRIMARY_COL] + X[GLEASON_SECONDARY_COL]
        X['High_Risk_Gleason'] = ((X[GLEASON_PRIMARY_COL] >= 4) | 
                                   (X[GLEASON_SECONDARY_COL] >= 4)).astype(int)
        created_features.extend(['Gleason_Total', 'High_Risk_Gleason'])

    if MARGIN_COL in X.columns and LYMPH_NODE_COL in X.columns:
        X['Margin_x_LymphNode'] = X[MARGIN_COL].astype(float) * X[LYMPH_NODE_COL].astype(float)
        created_features.append('Margin_x_LymphNode')

    t_stage_cols = [c for c in X.columns if 'Tumor Stage Code_T3' in c or 'Tumor Stage Code_T4' in c]
    if len(t_stage_cols) >= 2:
        X['T_Stage_Risk'] = X[t_stage_cols].sum(axis=1)
        created_features.append('T_Stage_Risk')

    # Pathway scores (lenient mode for external validation)
    for name, gene_set in [('PSA_Pathway_Score', PSA_GENES), 
                           ('AR_Signaling_Score', AR_GENES), 
                           ('Proliferation_Score', PROLIF_GENES)]:
        available = [g for g in gene_set if g in X.columns]
        if selected_genes is not None:
            available = [g for g in available if g in selected_genes]
        
        if len(available) >= MIN_GENES_FOR_PATHWAY:
            X[name] = X[available].mean(axis=1)
            created_features.append(name)

    # --- SUB-LAYER 2B: Gene-Clinical Interactions (~8 features) ---
    interactions = [
        ('AR_x_Gleason', 'AR_Signaling_Score', 'Gleason_Total'),
        ('Prolif_x_Margin', 'Proliferation_Score', 'Margin_x_LymphNode'),
        ('PSA_x_TStage', 'PSA_Pathway_Score', 'T_Stage_Risk'),
        ('AR_x_Prolif', 'AR_Signaling_Score', 'Proliferation_Score'),
        ('HighRisk_x_Prolif', 'High_Risk_Gleason', 'Proliferation_Score'),
        ('Gleason_x_AR', 'Gleason_Total', 'AR_Signaling_Score'),
        ('TStage_x_Margin', 'T_Stage_Risk', MARGIN_COL),
        ('PSA_x_Gleason', 'PSA_Pathway_Score', 'Gleason_Total')
    ]
    
    for new_col, col1, col2 in interactions:
        if col1 in X.columns and col2 in X.columns:
            X[new_col] = X[col1] * X[col2]
            created_features.append(new_col)

    # --- SUB-LAYER 2C: Multi-Pathway Ratios & Composites (~5 features) ---
    ratios = {
        'AR_to_Prolif_Ratio': lambda df: df['AR_Signaling_Score'] / (df['Proliferation_Score'] + 1e-6),
        'PSA_to_AR_Ratio': lambda df: df['PSA_Pathway_Score'] / (df['AR_Signaling_Score'] + 1e-6),
        'Pathway_Balance': lambda df: df['PSA_Pathway_Score'] + df['AR_Signaling_Score'] - df['Proliferation_Score'],
        'Combined_Risk': lambda df: df['Gleason_Total'] * 0.4 + df.get('T_Stage_Risk', pd.Series(0, index=df.index)) * 0.3 + df['Proliferation_Score'] * 0.3,
        'Gene_Variability': lambda df: df[[g for g in (selected_genes or []) if g in df.columns]].std(axis=1) 
                                          if len([g for g in (selected_genes or []) if g in df.columns]) > 1 
                                          else pd.Series(0.0, index=df.index)
    }

    req_map = {
        'AR_to_Prolif_Ratio': ['AR_Signaling_Score', 'Proliferation_Score'],
        'PSA_to_AR_Ratio': ['PSA_Pathway_Score', 'AR_Signaling_Score'],
        'Pathway_Balance': ['PSA_Pathway_Score', 'AR_Signaling_Score', 'Proliferation_Score'],
        'Combined_Risk': ['Gleason_Total', 'Proliferation_Score'],
        'Gene_Variability': []
    }

    for feat_name, calc_fn in ratios.items():
        req = req_map.get(feat_name, [])
        if all(c in X.columns for c in req):
            try:
                X[feat_name] = calc_fn(X)
                created_features.append(feat_name)
            except Exception as e:
                logger.warning(f"Failed to create {feat_name}: {e}")

    # --- AUTOMATIC CORRELATION FILTER ---
    if len(created_features) > 1:
        corr_matrix = X[created_features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > correlation_threshold)]
        
        if to_drop:
            logger.info(f"Layer 2 - Removed {len(to_drop)} highly correlated features (|r|>{correlation_threshold}): {to_drop}")
            created_features = [f for f in created_features if f not in to_drop]
            X = X.drop(columns=to_drop, errors='ignore')

    logger.info(f"Layer 2 - Created {len(created_features)} clean engineered features")
    return X, created_features


# =============================================================================
# LAYER 1: RAW GENE SELECTION
# =============================================================================

def fit_layer1_selector(
    X_genes: pd.DataFrame, y_train: pd.Series,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    random_state: int = config.RANDOM_STATE
) -> dict[str, Any]:
    """Fit Variance + MI selector on training genes ONLY."""
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_genes), columns=X_genes.columns, index=X_genes.index)
    
    vt = VarianceThreshold(threshold=variance_threshold)
    X_var = vt.fit_transform(X_imp)
    var_features = X_genes.columns[vt.get_support()].tolist()
    
    mi_scores = mutual_info_classif(X_var, y_train, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=var_features).sort_values(ascending=False)
    mi_features = mi_series.head(min(mi_top_k, len(mi_series))).index.tolist()
    
    logger.info(f"Layer 1 - Selected {len(mi_features)} genes from {len(X_genes.columns)} raw genes")
    return {"imputer": imputer, "vt": vt, "mi_features": mi_features, "is_fitted": True}


def transform_layer1(X_genes: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Transform test/external genes using fitted Layer 1 selector."""
    fit_cols = fitted["vt"].get_feature_names_out().tolist() if hasattr(fitted["vt"], 'get_feature_names_out') else fitted["vt"].get_support(indices=True).tolist()
    # Reorder and add missing as NaN for imputation
    for c in fit_cols:
        if c not in X_genes.columns:
            X_genes = X_genes.copy()
            X_genes[c] = np.nan
    
    X_imp = pd.DataFrame(fitted["imputer"].transform(X_genes[fit_cols]), columns=fit_cols, index=X_genes.index)
    valid_mi = [g for g in fitted["mi_features"] if g in X_imp.columns]
    logger.info(f"Layer 1 - Transform: {len(valid_mi)}/{len(fitted['mi_features'])} genes available")
    return X_imp[valid_mi]


# =============================================================================
# PSO HELPER FUNCTIONS
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def repair_exact_k(mask: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) > k:
        keep = rng.choice(idx, size=k, replace=False)
        out = np.zeros_like(mask, dtype=int); out[keep] = 1; return out
    if len(idx) < k:
        zero_idx = np.flatnonzero(mask == 0)
        add_n = min(k - len(idx), len(zero_idx))
        if add_n > 0:
            mask = mask.copy(); mask[rng.choice(zero_idx, size=add_n, replace=False)] = 1
    return mask


def pso_feature_select(
    X_fit: pd.DataFrame, y_fit: pd.Series, candidate_features: List[str],
    n_features: int = config.PSO_FINAL_K,
    n_particles: int = config.PSO_N_PARTICLES,
    n_iterations: int = config.PSO_N_ITERATIONS,
    penalty_alpha: float = config.PSO_PENALTY_ALPHA,
    random_state: int = config.RANDOM_STATE
) -> Tuple[List[str], float]:
    """Binary PSO with penalty for fixed-cardinality feature selection."""
    from src.models import make_xgb, xgb_safe_frame
    
    candidates = list(candidate_features)
    if len(candidates) <= n_features:
        return candidates, np.nan

    Xc = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_fit[candidates]), columns=candidates)
    n_dim = len(candidates)
    rng = np.random.RandomState(random_state)
    inner_cv = StratifiedKFold(n_splits=config.PSO_INNER_SPLITS, shuffle=True, random_state=random_state)
    cache: Dict[tuple, float] = {}

    def fitness(mask: np.ndarray) -> float:
        repaired = repair_exact_k(mask.astype(int), n_features, rng)
        key = tuple(np.flatnonzero(repaired).tolist())
        if key in cache: return cache[key]
        
        cols = [candidates[i] for i in key]
        scores = []
        for tr_idx, va_idx in inner_cv.split(Xc, y_fit):
            model = make_xgb(y_fit.iloc[tr_idx])
            model.fit(xgb_safe_frame(Xc.iloc[tr_idx][cols]), y_fit.iloc[tr_idx])
            p = model.predict_proba(xgb_safe_frame(Xc.iloc[va_idx][cols]))[:, 1]
            scores.append(roc_auc_score(y_fit.iloc[va_idx], p))
        
        mean_auc = float(np.mean(scores))
        final = mean_auc - (penalty_alpha * n_features)
        cache[key] = final
        return final

    # Initialize PSO
    pos = rng.uniform(-1, 1, (n_particles, n_dim))
    vel = rng.uniform(-0.1, 0.1, (n_particles, n_dim))
    binary = np.array([repair_exact_k((sigmoid(pos[i]) > 0.5).astype(int), n_features, rng) for i in range(n_particles)])
    
    pbest_pos = binary.copy()
    pbest_score = np.array([fitness(m) for m in binary])
    gbest_idx = int(np.argmax(pbest_score))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = float(pbest_score[gbest_idx])

    for it in range(n_iterations):
        r1, r2 = rng.rand(n_particles, n_dim), rng.rand(n_particles, n_dim)
        vel = config.PSO_W * vel + config.PSO_C1 * r1 * (pbest_pos - binary) + config.PSO_C2 * r2 * (gbest_pos - binary)
        vel = np.clip(vel, -4, 4)
        binary = np.array([repair_exact_k((rng.rand(n_dim) < sigmoid(vel[i])).astype(int), n_features, rng) for i in range(n_particles)])
        
        scores = np.array([fitness(m) for m in binary])
        improved = scores > pbest_score
        pbest_pos[improved] = binary[improved]; pbest_score[improved] = scores[improved]
        
        best_idx = int(np.argmax(pbest_score))
        if pbest_score[best_idx] > gbest_score:
            gbest_pos = pbest_pos[best_idx].copy(); gbest_score = float(pbest_score[best_idx])
            
        if (it + 1) % 5 == 0:
            logger.info(f"  PSO iter {it+1}/{n_iterations}: Fitness={gbest_score:.4f} (Raw AUC≈{gbest_score + penalty_alpha*n_features:.4f})")

    selected = [candidates[i] for i in np.flatnonzero(gbest_pos)]
    logger.info(f"PSO selected {len(selected)} features")
    return selected, gbest_score


# =============================================================================
# MAIN PIPELINE: 3-LAYER STRATEGY
# =============================================================================

def run_3layer_feature_selection(
    X_train: pd.DataFrame, y_train: pd.Series,
    clinical_cols: Optional[List[str]] = None,
    run_pso: bool = True,
    random_state: int = config.RANDOM_STATE
) -> Tuple[dict, List[str]]:
    """
    Execute the complete 3-Layer Feature Selection Pipeline.
    
    Returns:
        fitted_layer1: Selector dict for transforming external data
        final_features: List of ~40-45 final feature names
    """
    # Identify gene vs clinical columns
    if clinical_cols is None:
        clinical_cols = [c for c in X_train.columns if any(kw in c for kw in 
                         ['Gleason', 'Margin', 'Lymph', 'Tumor Stage', 'PSA'])]
    gene_cols = [c for c in X_train.columns if c not in clinical_cols]
    
    # === LAYER 1 ===
    fitted_l1 = fit_layer1_selector(X_train[gene_cols], y_train, random_state=random_state)
    selected_genes = fitted_l1["mi_features"]
    X_selected_genes = transform_layer1(X_train[gene_cols], fitted_l1)
    
    # === LAYER 2 ===
    X_for_eng = pd.concat([X_selected_genes, X_train[clinical_cols]], axis=1)
    X_eng, eng_features = create_extended_engineered_features(X_for_eng, selected_genes=selected_genes)
    
    # === LAYER 3: ASSEMBLY & PSO ===
    # Combine selected genes + clean engineered features as candidate pool
    candidate_pool = list(set(selected_genes + eng_features))
    candidate_pool = [f for f in candidate_pool if f in X_eng.columns]
    
    if run_pso:
        # PSO selects from BOTH genes and engineered features
        final_features, _ = pso_feature_select(
            X_eng, y_train, candidate_pool,
            n_features=min(config.PSO_FINAL_K, len(candidate_pool)),
            random_state=random_state + 1000
        )
    else:
        final_features = selected_genes[:config.PSO_FINAL_K] + eng_features
    
    logger.info(f"✅ 3-Layer Pipeline Complete: {len(final_features)} final features")
    return fitted_l1, final_features


def apply_3layer_to_external(
    X_ext_raw: pd.DataFrame,
    fitted_l1: dict,
    final_feature_list: List[str],
    clinical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply trained 3-layer pipeline to external/test data.
    Handles missing genes gracefully via imputation and lenient engineering.
    """
    if clinical_cols is None:
        clinical_cols = [c for c in X_ext_raw.columns if any(kw in c for kw in 
                         ['Gleason', 'Margin', 'Lymph', 'Tumor Stage', 'PSA'])]
    gene_cols = [c for c in X_ext_raw.columns if c not in clinical_cols]
    
    # Layer 1 Transform
    X_sel_genes = transform_layer1(X_ext_raw[gene_cols], fitted_l1)
    
    # Layer 2 Engineering (lenient)
    X_for_eng = pd.concat([X_sel_genes, X_ext_raw[clinical_cols]], axis=1)
    X_eng, _ = create_extended_engineered_features(X_for_eng, selected_genes=fitted_l1["mi_features"])
    
    # Layer 3: Align to final feature list
    available = [f for f in final_feature_list if f in X_eng.columns]
    missing = [f for f in final_feature_list if f not in X_eng.columns]
    
    if missing:
        logger.warning(f"External data missing {len(missing)} features: {missing[:5]}...")
        for m in missing:
            X_eng[m] = 0.0  # Safe default for missing engineered/clinical features
    
    X_final = X_eng[final_feature_list].fillna(0.0)
    logger.info(f"External validation: {len(available)}/{len(final_feature_list)} features available")
    return X_final