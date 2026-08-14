"""
Configuration for engineered features in the Prostate BCR prediction pipeline.

This module provides a single source of truth for all engineered feature definitions,
ensuring consistency across training, explainability, and evaluation notebooks.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EngineeredFeaturesConfig:
    """Single source of truth for engineered feature definitions."""
    
    # Clinical column names
    GLEASON_PRIMARY_COL: str = 'Gleason pattern primary'
    GLEASON_SECONDARY_COL: str = 'Gleason pattern secondary'
    MARGIN_COL: str = 'Surgical Margin Resection Status_R1'
    LYMPH_NODE_COL: str = 'Primary Lymph Node Presentation Assessment Ind-3_YES'
    
    # Gene sets for pathway scores
    PSA_GENES: Tuple[str, ...] = (
        'KLK3', 'KLK2', 'ACPP', 'TMPRSS2', 'AR', 'NKX3-1', 'STEAP2'
    )
    AR_GENES: Tuple[str, ...] = (
        'AR', 'FKBP5', 'KLK3', 'KLK2', 'TMPRSS2', 'NKX3-1', 'STEAP2', 'CAMKK2'
    )
    PROLIF_GENES: Tuple[str, ...] = (
        'MKI67', 'TOP2A', 'CCNB1', 'CCNE1', 'CDK1', 'AURKA', 'BIRC5'
    )
    
    # Engineered feature names (in order of creation)
    ENGINEERED_FEATURE_NAMES: Tuple[str, ...] = (
        'Gleason_Total',
        'High_Risk_Gleason',
        'Margin_x_LymphNode',
        'T_Stage_Risk',
        'PSA_Pathway_Score',
        'AR_Signaling_Score',
        'Proliferation_Score',
    )
    
    # Minimum number of genes required to create pathway score
    MIN_GENES_FOR_PATHWAY: int = 3
    
    # T-stage column pattern
    T_STAGE_PATTERN: str = 'Tumor Stage Code_T'


# Convenience exports
GLEASON_PRIMARY_COL = EngineeredFeaturesConfig.GLEASON_PRIMARY_COL
GLEASON_SECONDARY_COL = EngineeredFeaturesConfig.GLEASON_SECONDARY_COL
MARGIN_COL = EngineeredFeaturesConfig.MARGIN_COL
LYMPH_NODE_COL = EngineeredFeaturesConfig.LYMPH_NODE_COL

PSA_GENES = EngineeredFeaturesConfig.PSA_GENES
AR_GENES = EngineeredFeaturesConfig.AR_GENES
PROLIF_GENES = EngineeredFeaturesConfig.PROLIF_GENES

ENGINEERED_FEATURE_NAMES = EngineeredFeaturesConfig.ENGINEERED_FEATURE_NAMES
MIN_GENES_FOR_PATHWAY = EngineeredFeaturesConfig.MIN_GENES_FOR_PATHWAY
