"""
Clinical feature engineering module for Late Fusion architecture.

This module handles domain-specific clinical feature engineering ONLY.
It is one branch of the Late Fusion architecture, parallel to genomic features.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from src.features_config import (
    GLEASON_PRIMARY_COL,
    GLEASON_SECONDARY_COL,
    LYMPH_NODE_COL,
    MARGIN_COL,
)
from src.io import logger


def create_clinical_features(
    X_clinical: pd.DataFrame,
    strict_mode: bool = False,
    required_genes: Optional[dict[str, list[str]]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create clinically meaningful engineered features.
    
    This is the Clinical Branch of Late Fusion - it creates features from
    clinical variables ONLY (Gleason scores, tumor stage, surgical margins, etc.).
    
    Args:
        X_clinical: Input DataFrame with clinical features
        strict_mode: If True, only create pathway scores if ALL required genes present
        required_genes: Optional dict mapping score name to required gene list.
                       If provided, overrides default gene sets.
    
    Returns:
        Tuple of (DataFrame with new features, list of new feature names)
    """
    X = X_clinical.copy()
    created_features = []
    
    # ── 1. Gleason-based features ──
    if GLEASON_PRIMARY_COL in X.columns and GLEASON_SECONDARY_COL in X.columns:
        X['Gleason_Total'] = X[GLEASON_PRIMARY_COL] + X[GLEASON_SECONDARY_COL]
        X['High_Risk_Gleason'] = (
            (X[GLEASON_PRIMARY_COL] >= 4) | 
            (X[GLEASON_SECONDARY_COL] >= 4)
        ).astype(int)
        created_features.extend(['Gleason_Total', 'High_Risk_Gleason'])
        logger.info("Clinical: Gleason_Total, High_Risk_Gleason")

    # ── 2. Margin × Lymph Node interaction ──
    if MARGIN_COL in X.columns and LYMPH_NODE_COL in X.columns:
        X['Margin_x_LymphNode'] = (
            X[MARGIN_COL].astype(float) * X[LYMPH_NODE_COL].astype(float)
        )
        created_features.append('Margin_x_LymphNode')
        logger.info("Clinical: Margin_x_LymphNode")

    # ── 3. T-Stage risk score ──
    t_stage_cols = [c for c in X.columns if 'Tumor Stage Code_T3' in c or 'Tumor Stage Code_T4' in c]
    if len(t_stage_cols) >= 2:
        X['T_Stage_Risk'] = X[t_stage_cols].sum(axis=1)
        created_features.append('T_Stage_Risk')
        logger.info("Clinical: T_Stage_Risk")

    return X, created_features


def get_clinical_feature_names() -> list[str]:
    """Return the standard list of clinical engineered feature names.
    
    Returns:
        List of clinical feature names in order of creation
    """
    return [
        'Gleason_Total',
        'High_Risk_Gleason',
        'Margin_x_LymphNode',
        'T_Stage_Risk',
    ]


def prepare_clinical_branch(
    X_train_clinical: pd.DataFrame,
    X_test_clinical: pd.DataFrame | None = None,
    strict_mode: bool = False,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame | None]:
    """Prepare clinical features for the Late Fusion Clinical Branch.
    
    Args:
        X_train_clinical: Training clinical data
        X_test_clinical: Optional test clinical data
        strict_mode: Whether to use strict mode for pathway scores
        
    Returns:
        Tuple of (X_train_engineered, clinical_feature_names, X_test_engineered or None)
    """
    X_train_eng, clinical_features = create_clinical_features(
        X_train_clinical, strict_mode=strict_mode
    )
    
    X_test_eng = None
    if X_test_clinical is not None:
        X_test_eng, _ = create_clinical_features(
            X_test_clinical, strict_mode=strict_mode
        )
    
    return X_train_eng, clinical_features, X_test_eng
