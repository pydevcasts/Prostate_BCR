"""
Feature selection for TCGA-PRAD BCR prediction using 3-Layer Strategy.
Implements:
Layer 1: Raw Gene Filtering & Selection (Variance + MI)
Layer 2: Extended Domain-Specific Feature Engineering (~20 features)
         + Automatic High-Correlation Removal (|r| > 0.9)
Layer 3: Final Assembly (PSO Genes + Clean Engineered Features)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
        X: Input DataFrame (genes + clinical columns)
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


# Backward compatibility alias
def create_engineered_features(X, selected_genes=None, strict_mode=False):
    """Alias for backward compatibility with old notebooks."""
    return create_extended_engineered_features(
        X=X,
        selected_genes=selected_genes,
        correlation_threshold=0.90
    )


# =============================================================================
# LAYER 1: RAW GENE SELECTION
# =============================================================================

def fit_layer1_selector(
    X_genes: pd.DataFrame, y_train: pd.Series,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    random_state: int = config.RANDOM_STATE
) -> Dict[str, Any]:
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
            logger.info(f"  PSO iter {it+1}/{n_iterations}: Fitness={gbest_score:.4f} (Raw AUC~={gbest_score + penalty_alpha*n_features:.4f})")

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
    """Execute the complete 3-Layer Feature Selection Pipeline."""
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
    candidate_pool = list(set(selected_genes + eng_features))
    candidate_pool = [f for f in candidate_pool if f in X_eng.columns]

    if run_pso:
        final_features, _ = pso_feature_select(
            X_eng, y_train, candidate_pool,
            n_features=min(config.PSO_FINAL_K, len(candidate_pool)),
            random_state=random_state + 1000
        )
    else:
        final_features = selected_genes[:config.PSO_FINAL_K] + eng_features

    logger.info(f"3-Layer Pipeline Complete: {len(final_features)} final features")
    return fitted_l1, final_features


def apply_3layer_to_external(
    X_ext_raw: pd.DataFrame,
    fitted_l1: dict,
    final_feature_list: List[str],
    clinical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Apply trained 3-layer pipeline to external/test data."""
    if clinical_cols is None:
        clinical_cols = [c for c in X_ext_raw.columns if any(kw in c for kw in
                         ['Gleason', 'Margin', 'Lymph', 'Tumor Stage', 'PSA'])]
    gene_cols = [c for c in X_ext_raw.columns if c not in clinical_cols]

    X_sel_genes = transform_layer1(X_ext_raw[gene_cols], fitted_l1)
    X_for_eng = pd.concat([X_sel_genes, X_ext_raw[clinical_cols]], axis=1)
    X_eng, _ = create_extended_engineered_features(X_for_eng, selected_genes=fitted_l1["mi_features"])

    available = [f for f in final_feature_list if f in X_eng.columns]
    missing = [f for f in final_feature_list if f not in X_eng.columns]

    if missing:
        logger.warning(f"External data missing {len(missing)} features: {missing[:5]}...")
        for m in missing:
            X_eng[m] = 0.0

    X_final = X_eng[final_feature_list].fillna(0.0)
    logger.info(f"External validation: {len(available)}/{len(final_feature_list)} features available")
    return X_final
