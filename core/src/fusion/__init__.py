"""
Late Fusion architecture module for Prostate BCR Prediction.

This package implements the Late Fusion approach where:
- Genomic Branch: Uses MI + PSO for feature selection on gene expression data ONLY
- Clinical Branch: Uses domain-specific engineered features (Gleason, Stage, etc.) ONLY
- Fusion Layer: Combines probabilities from both branches using weighted average
"""

from src.fusion.late_fusion import (
    LateFusionPredictor,
    train_late_fusion_models,
    evaluate_late_fusion,
    fallback_to_genomic_only,
)

__all__ = [
    "LateFusionPredictor",
    "train_late_fusion_models",
    "evaluate_late_fusion",
    "fallback_to_genomic_only",
]
