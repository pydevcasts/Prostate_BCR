"""
Feature selection for TCGA-PRAD BCR prediction using simple MI-based strategy.
Implements:
1. Variance Threshold + Mutual Information for gene selection
2. Domain-specific feature engineering (7 features)
3. Simple transformation for test/external data
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
# FEATURE ENGINEERING (7 Core Features)
# =============================================================================

def create_engineered_features(
    X: pd.DataFrame,
    selected_genes: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create 7 domain-specific engineered features.
    
    Args:
        X: Input DataFrame (genes + clinical columns)
        selected_genes: Genes from feature selection (used to filter pathway genes)

    Returns:
        Tuple of (DataFrame with engineered features, list of feature names)
    """
    X = X.copy()
    created_features: List[str] = []

    # 1. Gleason Total
    if GLEASON_PRIMARY_COL in X.columns and GLEASON_SECONDARY_COL in X.columns:
        X['Gleason_Total'] = X[GLEASON_PRIMARY_COL] + X[GLEASON_SECONDARY_COL]
        X['High_Risk_Gleason'] = ((X[GLEASON_PRIMARY_COL] >= 4) |
                                   (X[GLEASON_SECONDARY_COL] >= 4)).astype(int)
        created_features.extend(['Gleason_Total', 'High_Risk_Gleason'])

    # 2. Margin x LymphNode interaction
    if MARGIN_COL in X.columns and LYMPH_NODE_COL in X.columns:
        X['Margin_x_LymphNode'] = X[MARGIN_COL].astype(float) * X[LYMPH_NODE_COL].astype(float)
        created_features.append('Margin_x_LymphNode')

    # 3. T-Stage Risk
    t_stage_cols = [c for c in X.columns if 'Tumor Stage Code_T3' in c or 'Tumor Stage Code_T4' in c]
    if len(t_stage_cols) >= 2:
        X['T_Stage_Risk'] = X[t_stage_cols].sum(axis=1)
        created_features.append('T_Stage_Risk')

    # 4-6. Pathway scores
    for name, gene_set in [('PSA_Pathway_Score', PSA_GENES),
                           ('AR_Signaling_Score', AR_GENES),
                           ('Proliferation_Score', PROLIF_GENES)]:
        available = [g for g in gene_set if g in X.columns]
        if selected_genes is not None:
            available = [g for g in available if g in selected_genes]

        if len(available) >= MIN_GENES_FOR_PATHWAY:
            X[name] = X[available].mean(axis=1)
            created_features.append(name)

    logger.info(f"Created {len(created_features)} engineered features: {created_features}")
    return X, created_features


# =============================================================================
# FEATURE SELECTION (Variance + MI)
# =============================================================================

def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    variance_threshold: float = config.VARIANCE_THRESHOLD,
    mi_top_k: int = config.MI_TOP_K,
    random_state: int = config.RANDOM_STATE
) -> Tuple[Any, List[str]]:
    """
    Perform feature selection using Variance Threshold + Mutual Information.
    
    Args:
        X: Input DataFrame
        y: Target Series
        variance_threshold: Minimum variance threshold
        mi_top_k: Number of top features to select via MI
        random_state: Random seed
        
    Returns:
        Tuple of (fitted_selector dict, list of selected feature names)
    """
    # Separate clinical and gene columns
    clinical_cols = [c for c in X.columns if any(kw in c for kw in
                     ['Gleason', 'Margin', 'Lymph', 'Tumor Stage', 'PSA'])]
    gene_cols = [c for c in X.columns if c not in clinical_cols]
    
    logger.info(f"Separating {len(gene_cols)} genes and {len(clinical_cols)} clinical features")
    
    # Impute missing values in genes
    imputer = SimpleImputer(strategy="median")
    X_genes_imp = pd.DataFrame(
        imputer.fit_transform(X[gene_cols]), 
        columns=gene_cols, 
        index=X.index
    )
    
    # Variance Threshold
    vt = VarianceThreshold(threshold=variance_threshold)
    X_var = vt.fit_transform(X_genes_imp)
    var_features = X_genes_imp.columns[vt.get_support()].tolist()
    logger.info(f"After variance threshold: {len(var_features)} features")
    
    # Mutual Information
    mi_scores = mutual_info_classif(X_var, y, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=var_features).sort_values(ascending=False)
    selected_genes = mi_series.head(min(mi_top_k, len(mi_series))).index.tolist()
    
    logger.info(f"Selected {len(selected_genes)} genes via MI (top {mi_top_k})")
    
    # Create fitted selector object
    fitted_selector = {
        "imputer": imputer,
        "variance_threshold": vt,
        "selected_genes": selected_genes,
        "clinical_cols": clinical_cols,
        "is_fitted": True
    }
    
    return fitted_selector, selected_genes


def transform_selected(
    X: pd.DataFrame,
    fitted_selector: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply fitted feature selector to new data (test/external).
    
    Args:
        X: New DataFrame to transform
        fitted_selector: Fitted selector from run_feature_selection
        
    Returns:
        DataFrame with selected features
    """
    X = X.copy()
    
    # Get original column lists
    clinical_cols = fitted_selector["clinical_cols"]
    selected_genes = fitted_selector["selected_genes"]
    imputer = fitted_selector["imputer"]
    vt = fitted_selector["variance_threshold"]
    
    # Identify gene columns in new data
    all_cols = set(X.columns)
    gene_cols = [c for c in all_cols if c not in clinical_cols]
    
    # Ensure all variance-filtered columns exist
    var_cols = (vt.get_feature_names_out().tolist() 
                if hasattr(vt, 'get_feature_names_out') 
                else gene_cols)
    
    for c in var_cols:
        if c not in X.columns:
            X[c] = np.nan
    
    # Impute and transform
    X_genes_imp = pd.DataFrame(
        imputer.transform(X[var_cols]),
        columns=var_cols,
        index=X.index
    )
    
    # Select MI features that are available
    available_genes = [g for g in selected_genes if g in X_genes_imp.columns]
    logger.info(f"Transform: {len(available_genes)}/{len(selected_genes)} genes available")
    
    # Combine selected genes + clinical features
    final_features = available_genes + clinical_cols
    X_final = X[final_features].copy()
    
    return X_final
