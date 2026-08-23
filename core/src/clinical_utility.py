"""
Clinical Utility Module for Prostate BCR Prediction Model.

This module provides clinical utility analysis tools including:
- Decision Curve Analysis (DCA)
- Confusion Matrix with Confidence Intervals
- Clinical Impact Curves
- Net Benefit Calculations

These analyses are essential for demonstrating the clinical applicability
of prediction models in biomedical research publications.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix

from src.io import logger


# ---------------------------------------------------------------------------
# Decision Curve Analysis (DCA)
# ---------------------------------------------------------------------------
def compute_net_benefit(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold_prob: float,
) -> float:
    """Compute net benefit at a specific threshold probability.

    Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))

    where pt is the threshold probability.

    Parameters
    ----------
    y_true : Ground truth binary labels (0 or 1).
    y_prob : Predicted probabilities.
    threshold_prob : Threshold probability for classification.

    Returns
    -------
    Net benefit value at the specified threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    if n == 0:
        return np.nan

    # Convert probabilities to predictions using threshold
    y_pred = (y_prob >= threshold_prob).astype(int)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Avoid division by zero
    if threshold_prob == 1.0:
        threshold_prob = 0.999

    # Calculate net benefit
    weight = threshold_prob / (1 - threshold_prob)
    net_benefit = (tp / n) - (fp / n) * weight

    return net_benefit


def decision_curve_analysis(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_thresholds: int = 100,
    threshold_range: Tuple[float, float] = (0.0, 0.5),
) -> pd.DataFrame:
    """Perform Decision Curve Analysis across a range of thresholds.

    DCA evaluates the clinical usefulness of a prediction model by
    calculating net benefit across different threshold probabilities.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.
    n_thresholds : Number of threshold points to evaluate.
    threshold_range : Tuple of (min_threshold, max_threshold).

    Returns
    -------
    DataFrame with threshold probabilities and corresponding net benefits.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Generate threshold probabilities
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    # Compute net benefit for each threshold
    net_benefits = []
    for thresh in thresholds:
        nb = compute_net_benefit(y_true, y_prob, thresh)
        net_benefits.append(nb)

    # Also compute "treat all" and "treat none" strategies
    treat_all_nb = []
    for thresh in thresholds:
        # Treat all: everyone is predicted positive
        weight = thresh / (1 - thresh) if thresh < 1.0 else 999
        prevalence = y_true.mean()
        nb_all = prevalence - (1 - prevalence) * weight
        treat_all_nb.append(nb_all)

    results = pd.DataFrame({
        "threshold_probability": thresholds,
        "net_benefit_model": net_benefits,
        "net_benefit_treat_all": treat_all_nb,
        "net_benefit_treat_none": [0.0] * len(thresholds),
    })

    logger.info(
        "DCA completed: max net benefit = %.4f at threshold = %.3f",
        results["net_benefit_model"].max(),
        results.loc[results["net_benefit_model"].idxmax(), "threshold_probability"],
    )

    return results


def bootstrap_dca_confidence_intervals(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bootstraps: int = 1000,
    random_state: int = 42,
    n_thresholds: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute bootstrap confidence intervals for DCA.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.
    n_bootstraps : Number of bootstrap iterations.
    random_state : Random seed for reproducibility.
    n_thresholds : Number of threshold points.

    Returns
    -------
    Dictionary with threshold probabilities and CI bounds.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(random_state)

    thresholds = np.linspace(0.0, 0.5, n_thresholds)
    bootstrap_nbs = np.zeros((n_bootstraps, len(thresholds)))

    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    for i in range(n_bootstraps):
        # Stratified bootstrap sampling
        boot_pos = rng.choice(pos_indices, size=len(pos_indices), replace=True)
        boot_neg = rng.choice(neg_indices, size=len(neg_indices), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        y_true_boot = y_true[boot_idx]
        y_prob_boot = y_prob[boot_idx]

        for j, thresh in enumerate(thresholds):
            bootstrap_nbs[i, j] = compute_net_benefit(y_true_boot, y_prob_boot, thresh)

    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_nbs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_nbs, 97.5, axis=0)
    ci_mean = np.mean(bootstrap_nbs, axis=0)

    return {
        "thresholds": thresholds,
        "net_benefit_mean": ci_mean,
        "net_benefit_ci_lower": ci_lower,
        "net_benefit_ci_upper": ci_upper,
    }


# ---------------------------------------------------------------------------
# Confusion Matrix with Confidence Intervals
# ---------------------------------------------------------------------------
def confusion_matrix_with_ci(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute confusion matrix with bootstrap confidence intervals.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_pred : Predicted binary labels.
    n_bootstraps : Number of bootstrap iterations.
    confidence_level : Confidence level for intervals (e.g., 0.95).
    random_state : Random seed for reproducibility.

    Returns
    -------
    Dictionary with confusion matrix values and their CIs.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(random_state)

    # Base confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Bootstrap sampling for CIs
    bootstrap_metrics = {
        "tp": [],
        "fp": [],
        "tn": [],
        "fn": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
    }

    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    for _ in range(n_bootstraps):
        boot_pos = rng.choice(pos_indices, size=len(pos_indices), replace=True)
        boot_neg = rng.choice(neg_indices, size=len(neg_indices), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        y_true_boot = y_true[boot_idx]
        y_pred_boot = y_pred[boot_idx]

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(
            y_true_boot, y_pred_boot, labels=[0, 1]
        ).ravel()

        bootstrap_metrics["tp"].append(tp_b)
        bootstrap_metrics["fp"].append(fp_b)
        bootstrap_metrics["tn"].append(tn_b)
        bootstrap_metrics["fn"].append(fn_b)

        # Derived metrics
        sensitivity = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        specificity = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
        precision = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0
        )

        bootstrap_metrics["sensitivity"].append(sensitivity)
        bootstrap_metrics["specificity"].append(specificity)
        bootstrap_metrics["precision"].append(precision)
        bootstrap_metrics["f1"].append(f1)

    # Calculate CIs
    alpha = 1 - confidence_level
    ci_lower_pct = alpha / 2 * 100
    ci_upper_pct = (1 - alpha / 2) * 100

    results = {"confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}

    for metric, values in bootstrap_metrics.items():
        values_arr = np.array(values)
        results[metric] = {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
            "ci_lower": float(np.percentile(values_arr, ci_lower_pct)),
            "ci_upper": float(np.percentile(values_arr, ci_upper_pct)),
            "median": float(np.median(values_arr)),
        }

    logger.info("Confusion matrix with CI computed (%d bootstraps)", n_bootstraps)

    return results


# ---------------------------------------------------------------------------
# Clinical Impact Curve Data
# ---------------------------------------------------------------------------
def clinical_impact_curve_data(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """Generate data for clinical impact curve plotting.

    Shows how many patients would be classified as high-risk at each
    threshold, and how many of those would actually experience the event.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.
    n_thresholds : Number of threshold points.

    Returns
    -------
    DataFrame with threshold, total high-risk, and true positives.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    total_high_risk = []
    true_positives = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        total_high_risk.append(y_pred.sum())

        # True positives at this threshold
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        true_positives.append(tp)

    results = pd.DataFrame({
        "threshold": thresholds,
        "total_high_risk": total_high_risk,
        "true_positives": true_positives,
        "false_positives": np.array(total_high_risk) - np.array(true_positives),
    })

    return results


# ---------------------------------------------------------------------------
# Standardized Net Benefit for Comparison
# ---------------------------------------------------------------------------
def standardized_net_benefit(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold_prob: float,
) -> float:
    """Compute standardized net benefit (scaled 0-1).

    Standardizes net benefit relative to 'treat all' and 'treat none'
    strategies for easier interpretation.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.
    threshold_prob : Threshold probability.

    Returns
    -------
    Standardized net benefit (0 = no benefit, 1 = perfect prediction).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    nb_model = compute_net_benefit(y_true, y_prob, threshold_prob)

    # Net benefit of treating all
    prevalence = y_true.mean()
    weight = threshold_prob / (1 - threshold_prob) if threshold_prob < 1.0 else 999
    nb_treat_all = prevalence - (1 - prevalence) * weight

    # Standardize
    if nb_treat_all <= 0:
        return 0.0

    standardized_nb = nb_model / nb_treat_all
    return max(0.0, min(1.0, standardized_nb))


# ---------------------------------------------------------------------------
# Optimal Threshold Selection
# ---------------------------------------------------------------------------
def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    criterion: str = "youden",
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
) -> Dict[str, Any]:
    """Find optimal classification threshold based on various criteria.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.
    criterion : Method for optimization:
        - 'youden': Maximize Youden's J statistic (sensitivity + specificity - 1)
        - 'f1': Maximize F1 score
        - 'cost': Minimize weighted cost (requires cost_fn and cost_fp)
        - 'net_benefit': Maximize net benefit
    cost_fn : Cost of false negative (for 'cost' criterion).
    cost_fp : Cost of false positive (for 'cost' criterion).

    Returns
    -------
    Dictionary with optimal threshold and associated metrics.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thresholds = np.unique(y_prob)
    if len(thresholds) < 2:
        thresholds = np.linspace(0.01, 0.99, 100)

    best_threshold = 0.5
    best_score = -np.inf

    metrics_at_thresholds = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        youden_j = sensitivity + specificity - 1
        net_benefit = compute_net_benefit(y_true, y_prob, thresh)

        # Cost-based score (lower is better, so negate)
        cost_score = -(fn * cost_fn + fp * cost_fp)

        metrics_at_thresholds.append({
            "threshold": thresh,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "youden_j": youden_j,
            "net_benefit": net_benefit,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

        # Select scoring based on criterion
        if criterion == "youden":
            score = youden_j
        elif criterion == "f1":
            score = f1
        elif criterion == "cost":
            score = cost_score
        elif criterion == "net_benefit":
            score = net_benefit
        else:
            score = youden_j

        if score > best_score:
            best_score = score
            best_threshold = thresh

    metrics_df = pd.DataFrame(metrics_at_thresholds)

    result = {
        "optimal_threshold": float(best_threshold),
        "criterion": criterion,
        "score": float(best_score),
        "metrics_at_optimal": metrics_df[metrics_df["threshold"] == best_threshold].iloc[0].to_dict(),
        "all_thresholds": metrics_df,
    }

    logger.info(
        "Optimal threshold: %.4f (criterion=%s, score=%.4f)",
        best_threshold, criterion, best_score,
    )

    return result
