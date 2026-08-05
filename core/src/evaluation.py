"""
Model evaluation metrics for TCGA-PRAD BCR prediction.

Implements comprehensive evaluation metrics including ROC-AUC, PR-AUC,
accuracy, precision, recall, specificity, sensitivity, F1, MCC,
balanced accuracy, calibration, and bootstrap confidence intervals.

All metrics are computed on held-out test data only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config as config
from src.io import logger


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_pred : Predicted binary labels.
    y_prob : Predicted probabilities (optional, for AUC metrics).
    threshold : Threshold used to convert probabilities to labels.

    Returns
    -------
    Dictionary with all computed metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    # Compute specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)

    # AUC metrics if probabilities are provided
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute confusion matrix as a DataFrame.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_pred : Predicted binary labels.

    Returns
    -------
    DataFrame with confusion matrix values.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def stratified_bootstrap_auc(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_boot: int = config.BOOTSTRAP_N,
    random_state: int = config.RANDOM_STATE,
) -> tuple[tuple[float, float], np.ndarray]:
    """Compute stratified bootstrap confidence interval for ROC-AUC.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.
    n_boot : Number of bootstrap iterations.
    random_state : Random seed for reproducibility.

    Returns
    -------
    Tuple of ((lower_ci, upper_ci), bootstrap_scores).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(random_state)

    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)

    if len(pos) == 0 or len(neg) == 0:
        logger.warning("Cannot compute bootstrap CI: only one class present")
        return (np.nan, np.nan), np.array([])

    scores = []
    for _ in range(n_boot):
        idx_pos = rng.choice(pos, size=len(pos), replace=True)
        idx_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([idx_pos, idx_neg])
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))

    scores = np.asarray(scores)
    alpha = 1 - config.CONFIDENCE_LEVEL
    ci = np.percentile(scores, [alpha / 2 * 100, (1 - alpha / 2) * 100])

    logger.info(
        "Bootstrap AUC CI (%.0f%%): [%.4f, %.4f]",
        config.CONFIDENCE_LEVEL * 100, ci[0], ci[1],
    )

    return (float(ci[0]), float(ci[1])), scores


# ---------------------------------------------------------------------------
# ROC and PR curve data
# ---------------------------------------------------------------------------
def get_roc_curve_data(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Get ROC curve data for plotting.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.

    Returns
    -------
    DataFrame with FPR, TPR, and thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    })


def get_pr_curve_data(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Get Precision-Recall curve data for plotting.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.

    Returns
    -------
    DataFrame with precision, recall, and thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": np.append(thresholds, np.nan),
    })


def get_calibration_curve_data(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Get calibration curve data for plotting.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.
    n_bins : Number of bins for calibration.

    Returns
    -------
    DataFrame with mean predicted probability and fraction of positives.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return pd.DataFrame({
        "mean_predicted": prob_pred,
        "fraction_positive": prob_true,
    })


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def generate_evaluation_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None = None,
    model_name: str = "Model",
) -> pd.DataFrame:
    """Generate a comprehensive evaluation report as DataFrame.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_pred : Predicted binary labels.
    y_prob : Predicted probabilities (optional).
    model_name : Name of the model for labeling.

    Returns
    -------
    DataFrame with all metrics.
    """
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["model"] = model_name

    report = pd.DataFrame([metrics])

    logger.info("=" * 60)
    logger.info("EVALUATION REPORT: %s", model_name)
    logger.info("=" * 60)
    for key, value in metrics.items():
        if key != "model":
            logger.info("  %s: %.4f", key, value)
    logger.info("=" * 60)

    return report