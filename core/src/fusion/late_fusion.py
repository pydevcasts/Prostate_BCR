"""
Late Fusion module for Prostate BCR Prediction.

This module implements the Late Fusion architecture where:
- Genomic Branch: Uses MI + PSO for feature selection on gene expression data ONLY
- Clinical Branch: Uses domain-specific engineered features (Gleason, Stage, etc.) ONLY
- Fusion Layer: Combines probabilities from both branches using weighted average
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from src.io import logger


class LateFusionPredictor:
    """Late Fusion Predictor that combines genomic and clinical model probabilities.

    This class implements the fusion layer of the Late Fusion architecture.
    It takes two trained models (genomic and clinical) and combines their
    predict_proba outputs using a weighted average.

    Attributes:
        genomic_model: Trained model for genomic branch
        clinical_model: Trained model for clinical branch
        genomic_weight: Weight for genomic model predictions (0.0 to 1.0)
        clinical_weight: Weight for clinical model predictions (0.0 to 1.0)
        optimized: Whether weights have been optimized
    """

    def __init__(
        self,
        genomic_model: Any,
        clinical_model: Any,
        genomic_weight: float = 0.5,
        clinical_weight: float = 0.5,
    ):
        """Initialize the Late Fusion Predictor.

        Args:
            genomic_model: Trained model for genomic branch
            clinical_model: Trained model for clinical branch
            genomic_weight: Initial weight for genomic model (default 0.5)
            clinical_weight: Initial weight for clinical model (default 0.5)
        """
        self.genomic_model = genomic_model
        self.clinical_model = clinical_model
        self.genomic_weight = genomic_weight
        self.clinical_weight = clinical_weight
        self.optimized = False

        # Normalize weights to sum to 1.0
        total = self.genomic_weight + self.clinical_weight
        if total > 0:
            self.genomic_weight /= total
            self.clinical_weight /= total

    def predict_proba(
        self,
        X_genomic: pd.DataFrame,
        X_clinical: pd.DataFrame,
    ) -> np.ndarray:
        """Predict probabilities using late fusion.

        Args:
            X_genomic: Genomic features DataFrame
            X_clinical: Clinical features DataFrame

        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        # Get probabilities from each branch
        proba_genomic = self.genomic_model.predict_proba(X_genomic)
        proba_clinical = self.clinical_model.predict_proba(X_clinical)

        # Weighted average fusion
        fused_proba = (
            self.genomic_weight * proba_genomic +
            self.clinical_weight * proba_clinical
        )

        return fused_proba

    def predict(
        self,
        X_genomic: pd.DataFrame,
        X_clinical: pd.DataFrame,
    ) -> np.ndarray:
        """Predict class labels using late fusion.

        Args:
            X_genomic: Genomic features DataFrame
            X_clinical: Clinical features DataFrame

        Returns:
            Array of predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X_genomic, X_clinical)
        return (proba[:, 1] >= 0.5).astype(int)

    def optimize_weights(
        self,
        X_genomic_val: pd.DataFrame,
        X_clinical_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
        n_steps: int = 100,
    ) -> Tuple[float, float, float]:
        """Optimize fusion weights based on validation AUC.

        This method searches for the optimal weights that maximize
        the ROC-AUC score on validation data.

        Args:
            X_genomic_val: Validation genomic features
            X_clinical_val: Validation clinical features
            y_val: Validation target
            n_steps: Number of weight combinations to try

        Returns:
            Tuple of (best_genomic_weight, best_clinical_weight, best_auc)
        """
        best_auc = -np.inf
        best_genomic_weight = 0.5
        best_clinical_weight = 0.5

        # Get individual model predictions
        proba_genomic = self.genomic_model.predict_proba(X_genomic_val)[:, 1]
        proba_clinical = self.clinical_model.predict_proba(X_clinical_val)[:, 1]

        # Search over weight combinations
        for i in range(n_steps + 1):
            genomic_w = i / n_steps
            clinical_w = 1.0 - genomic_w

            fused_proba = genomic_w * proba_genomic + clinical_w * proba_clinical
            auc = roc_auc_score(y_val, fused_proba)

            if auc > best_auc:
                best_auc = auc
                best_genomic_weight = genomic_w
                best_clinical_weight = clinical_w

        self.genomic_weight = best_genomic_weight
        self.clinical_weight = best_clinical_weight
        self.optimized = True

        logger.info(
            "Late Fusion weights optimized: genomic=%.3f, clinical=%.3f, val_AUC=%.4f",
            best_genomic_weight, best_clinical_weight, best_auc,
        )

        return best_genomic_weight, best_clinical_weight, best_auc

    def get_feature_importance(
        self,
        genomic_features: list[str],
        clinical_features: list[str],
    ) -> pd.DataFrame:
        """Get combined feature importance from both branches.

        Args:
            genomic_features: List of genomic feature names
            clinical_features: List of clinical feature names

        Returns:
            DataFrame with feature names and importance scores
        """
        # Get importance from each model
        genomic_importance = getattr(self.genomic_model, 'feature_importances_', None)
        clinical_importance = getattr(self.clinical_model, 'feature_importances_', None)

        importance_data = []

        if genomic_importance is not None:
            for feat, imp in zip(genomic_features, genomic_importance):
                importance_data.append({
                    'feature': feat,
                    'importance': imp * self.genomic_weight,
                    'branch': 'genomic',
                })

        if clinical_importance is not None:
            for feat, imp in zip(clinical_features, clinical_importance):
                importance_data.append({
                    'feature': feat,
                    'importance': imp * self.clinical_weight,
                    'branch': 'clinical',
                })

        df = pd.DataFrame(importance_data)
        return df.sort_values('importance', ascending=False)


def train_late_fusion_models(
    X_train_genomic: pd.DataFrame,
    y_train: pd.Series,
    X_train_clinical: pd.DataFrame,
    genomic_model: Any,
    clinical_model: Any,
    X_val_genomic: pd.DataFrame | None = None,
    X_val_clinical: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    optimize_weights: bool = True,
) -> LateFusionPredictor:
    """Train both branch models and create a LateFusionPredictor.

    Args:
        X_train_genomic: Training genomic features
        y_train: Training target
        X_train_clinical: Training clinical features
        genomic_model: Untrained genomic model (will be fitted)
        clinical_model: Untrained clinical model (will be fitted)
        X_val_genomic: Optional validation genomic features for weight optimization
        X_val_clinical: Optional validation clinical features for weight optimization
        y_val: Optional validation target for weight optimization
        optimize_weights: Whether to optimize fusion weights (default True)

    Returns:
        Trained LateFusionPredictor instance
    """
    # Train genomic branch
    logger.info("Training genomic branch model...")
    genomic_model.fit(X_train_genomic, y_train)

    # Train clinical branch
    logger.info("Training clinical branch model...")
    clinical_model.fit(X_train_clinical, y_train)

    # Create fusion predictor
    fusion_predictor = LateFusionPredictor(
        genomic_model=genomic_model,
        clinical_model=clinical_model,
        genomic_weight=0.5,
        clinical_weight=0.5,
    )

    # Optimize weights if validation data provided
    if optimize_weights and X_val_genomic is not None and y_val is not None:
        fusion_predictor.optimize_weights(
            X_val_genomic, X_val_clinical, y_val,
        )

    return fusion_predictor


def evaluate_late_fusion(
    fusion_predictor: LateFusionPredictor,
    X_test_genomic: pd.DataFrame,
    X_test_clinical: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate the Late Fusion model on test data.

    Args:
        fusion_predictor: Trained LateFusionPredictor
        X_test_genomic: Test genomic features
        X_test_clinical: Test clinical features
        y_test: Test target

    Returns:
        Dictionary with evaluation metrics
    """
    # Get fused predictions
    proba = fusion_predictor.predict_proba(X_test_genomic, X_test_clinical)[:, 1]
    predictions = (proba >= 0.5).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_test, proba)
    accuracy = (predictions == y_test).mean()

    # Get individual branch performance
    proba_genomic = fusion_predictor.genomic_model.predict_proba(X_test_genomic)[:, 1]
    proba_clinical = fusion_predictor.clinical_model.predict_proba(X_test_clinical)[:, 1]

    auc_genomic = roc_auc_score(y_test, proba_genomic)
    auc_clinical = roc_auc_score(y_test, proba_clinical)

    results = {
        'fusion_auc': auc,
        'fusion_accuracy': accuracy,
        'genomic_auc': auc_genomic,
        'clinical_auc': auc_clinical,
        'genomic_weight': fusion_predictor.genomic_weight,
        'clinical_weight': fusion_predictor.clinical_weight,
    }

    logger.info(
        "Late Fusion Evaluation: Fusion AUC=%.4f, Genomic AUC=%.4f, Clinical AUC=%.4f",
        auc, auc_genomic, auc_clinical,
    )

    return results


def fallback_to_genomic_only(
    genomic_model: Any,
    X_genomic: pd.DataFrame,
    X_clinical: pd.DataFrame | None = None,
    has_clinical: bool = True,
) -> np.ndarray:
    """Fallback prediction using only genomic model when clinical data is missing.

    This is useful for external validation where clinical data may be incomplete.

    Args:
        genomic_model: Trained genomic model
        X_genomic: Genomic features
        X_clinical: Optional clinical features (ignored if has_clinical=False)
        has_clinical: Whether clinical data is available

    Returns:
        Predicted probabilities array
    """
    if has_clinical and X_clinical is not None:
        # Use genomic model only (this is the fallback)
        return genomic_model.predict_proba(X_genomic)
    else:
        # Clinical data missing - use genomic only
        logger.warning("Clinical data missing - using genomic model only")
        return genomic_model.predict_proba(X_genomic)
