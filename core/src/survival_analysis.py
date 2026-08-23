"""
Survival Analysis Module for Prostate BCR Prediction Model.

This module provides survival analysis tools including:
- Kaplan-Meier Estimator
- Log-Rank Test
- Risk Stratification based on predicted scores
- Time-dependent ROC Analysis

These analyses are essential for demonstrating prognostic value
of prediction models in oncology research publications.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve
from src.io import logger


# ---------------------------------------------------------------------------
# Kaplan-Meier Estimator
# ---------------------------------------------------------------------------
def kaplan_meier_estimator(
    event_times: np.ndarray | pd.Series,
    event_observed: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute Kaplan-Meier survival estimates.

    Parameters
    ----------
    event_times : Array of times to event or censoring.
    event_observed : Binary array indicating if event was observed (1)
                     or censored (0).

    Returns
    -------
    DataFrame with time points, survival probability, and confidence intervals.
    """
    event_times = np.asarray(event_times)
    event_observed = np.asarray(event_observed)

    # Sort by event times
    sorted_idx = np.argsort(event_times)
    times = event_times[sorted_idx]
    events = event_observed[sorted_idx]

    n = len(times)
    at_risk = np.arange(n, 0, -1)

    # Calculate survival probability at each time point
    survival_prob = np.ones(n + 1)
    variance = np.zeros(n + 1)

    unique_times = np.unique(times)
    km_results = []

    for t in unique_times:
        mask = times == t
        d = events[mask].sum()  # Number of events at time t
        n_at_risk = at_risk[times >= t].sum()  # Number at risk at time t

        if n_at_risk > 0:
            # Survival probability
            s_t = 1 - d / n_at_risk
            last_surv = survival_prob[-1]
            new_surv = last_surv * s_t

            # Greenwood's formula for variance
            if n_at_risk > d:
                var_increment = d / (n_at_risk * (n_at_risk - d))
            else:
                var_increment = 0

            new_var = variance[-1] + var_increment

            survival_prob = np.append(survival_prob, new_surv)
            variance = np.append(variance, new_var)

            # 95% confidence interval using log-log transformation
            se = np.sqrt(new_var) if new_var > 0 else 0
            if new_surv > 0 and new_surv < 1:
                log_log = np.log(-np.log(new_surv))
                se_log_log = se / (new_surv * np.abs(np.log(new_surv))) if new_surv not in [0, 1] else 0
                ci_lower = np.exp(-np.exp(log_log + 1.96 * se_log_log))
                ci_upper = np.exp(-np.exp(log_log - 1.96 * se_log_log))
                ci_lower = max(0, min(1, ci_lower))
                ci_upper = max(0, min(1, ci_upper))
            else:
                ci_lower = new_surv
                ci_upper = new_surv

            km_results.append({
                "time": t,
                "n_at_risk": int(n_at_risk),
                "n_events": int(d),
                "survival_probability": float(new_surv),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            })

    return pd.DataFrame(km_results)


def log_rank_test(
    event_times_1: np.ndarray | pd.Series,
    event_observed_1: np.ndarray | pd.Series,
    event_times_2: np.ndarray | pd.Series,
    event_observed_2: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """Perform log-rank test to compare two survival curves.

    Parameters
    ----------
    event_times_1 : Event times for group 1.
    event_observed_1 : Event indicators for group 1.
    event_times_2 : Event times for group 2.
    event_observed_2 : Event indicators for group 2.

    Returns
    -------
    Dictionary with test statistic, p-value, and degrees of freedom.
    """
    event_times_1 = np.asarray(event_times_1)
    event_observed_1 = np.asarray(event_observed_1)
    event_times_2 = np.asarray(event_times_2)
    event_observed_2 = np.asarray(event_observed_2)

    # Combine data
    all_times = np.concatenate([event_times_1, event_times_2])
    all_events = np.concatenate([event_observed_1, event_observed_2])
    all_groups = np.concatenate([np.zeros(len(event_times_1)), np.ones(len(event_times_2))])

    # Get unique event times
    unique_times = np.unique(all_times[all_events == 1])

    observed_1 = 0
    expected_1 = 0
    variance = 0

    for t in unique_times:
        # Group 1
        at_risk_1 = ((event_times_1 >= t)).sum()
        events_1 = ((event_times_1 == t) & (event_observed_1 == 1)).sum()

        # Group 2
        at_risk_2 = ((event_times_2 >= t)).sum()
        events_2 = ((event_times_2 == t) & (event_observed_2 == 1)).sum()

        # Total
        n = at_risk_1 + at_risk_2
        d = events_1 + events_2

        if n > 0:
            # Expected events in group 1
            e_1 = at_risk_1 * d / n
            expected_1 += e_1
            observed_1 += events_1

            # Variance (hypergeometric)
            if n > 1:
                v = (at_risk_1 * at_risk_2 * d * (n - d)) / (n * n * (n - 1))
                variance += v

    # Chi-square statistic
    if variance > 0:
        chi_square = (observed_1 - expected_1) ** 2 / variance
        p_value = 1 - stats.chi2.cdf(chi_square, df=1)
    else:
        chi_square = 0
        p_value = 1.0

    result = {
        "chi_square": float(chi_square),
        "p_value": float(p_value),
        "degrees_of_freedom": 1,
        "observed_group1": float(observed_1),
        "expected_group1": float(expected_1),
    }

    logger.info(
        "Log-rank test: χ² = %.4f, p = %.4e",
        chi_square, p_value,
    )

    return result


# ---------------------------------------------------------------------------
# Risk Stratification
# ---------------------------------------------------------------------------
def stratify_by_risk_score(
    risk_scores: np.ndarray | pd.Series,
    strategy: str = "median",
    percentiles: Optional[List[float]] = None,
) -> np.ndarray:
    """Stratify patients into risk groups based on predicted scores.

    Parameters
    ----------
    risk_scores : Continuous risk scores from model predictions.
    strategy : Stratification method:
        - 'median': Split into high/low risk at median
        - 'tercile': Split into three equal groups
        - 'quartile': Split into four equal groups
        - 'percentile': Use custom percentiles (requires percentiles parameter)
    percentiles : List of percentile thresholds (for 'percentile' strategy).

    Returns
    -------
    Array of risk group assignments (0, 1, 2, ...).
    """
    risk_scores = np.asarray(risk_scores)

    if strategy == "median":
        threshold = np.median(risk_scores)
        groups = (risk_scores >= threshold).astype(int)

    elif strategy == "tercile":
        thresholds = np.percentile(risk_scores, [33.33, 66.67])
        groups = np.digitize(risk_scores, thresholds)

    elif strategy == "quartile":
        thresholds = np.percentile(risk_scores, [25, 50, 75])
        groups = np.digitize(risk_scores, thresholds)

    elif strategy == "percentile":
        if percentiles is None:
            raise ValueError("percentiles must be provided for 'percentile' strategy")
        thresholds = np.percentile(risk_scores, percentiles)
        groups = np.digitize(risk_scores, thresholds)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return groups


def kaplan_meier_by_risk_group(
    event_times: np.ndarray | pd.Series,
    event_observed: np.ndarray | pd.Series,
    risk_scores: np.ndarray | pd.Series,
    strategy: str = "median",
) -> Dict[str, Any]:
    """Compute Kaplan-Meier curves stratified by risk score.

    Parameters
    ----------
    event_times : Times to event or censoring.
    event_observed : Event indicators (1=event, 0=censored).
    risk_scores : Continuous risk scores from model.
    strategy : Risk stratification strategy.

    Returns
    -------
    Dictionary with KM curves for each group and log-rank test results.
    """
    event_times = np.asarray(event_times)
    event_observed = np.asarray(event_observed)
    risk_scores = np.asarray(risk_scores)

    # Stratify into risk groups
    risk_groups = stratify_by_risk_score(risk_scores, strategy)
    n_groups = len(np.unique(risk_groups))

    # Compute KM curve for each group
    km_curves = {}
    for g in range(n_groups):
        mask = risk_groups == g
        if mask.sum() > 0:
            km_curves[f"group_{g}"] = kaplan_meier_estimator(
                event_times[mask], event_observed[mask]
            )

    # Perform log-rank tests between groups
    log_rank_results = []
    if n_groups == 2:
        mask_0 = risk_groups == 0
        mask_1 = risk_groups == 1
        lr_test = log_rank_test(
            event_times[mask_0], event_observed[mask_0],
            event_times[mask_1], event_observed[mask_1],
        )
        lr_test["comparison"] = "group_0 vs group_1"
        log_rank_results.append(lr_test)

    result = {
        "km_curves": km_curves,
        "log_rank_tests": log_rank_results,
        "n_groups": n_groups,
        "group_sizes": [int((risk_groups == g).sum()) for g in range(n_groups)],
        "risk_groups": risk_groups,
    }

    logger.info(
        "KM analysis by risk group: %d groups, sizes = %s",
        n_groups, result["group_sizes"],
    )

    return result


# ---------------------------------------------------------------------------
# Time-Dependent ROC Analysis
# ---------------------------------------------------------------------------
def time_dependent_roc(
    event_times: np.ndarray | pd.Series,
    event_observed: np.ndarray | pd.Series,
    risk_scores: np.ndarray | pd.Series,
    eval_time: float,
) -> Dict[str, Any]:
    """Compute time-dependent ROC AUC at a specific time point.

    Uses the cumulative/dynamic approach where:
    - Cases: subjects who experienced event before eval_time
    - Controls: subjects who were event-free at eval_time

    Parameters
    ----------
    event_times : Times to event or censoring.
    event_observed : Event indicators (1=event, 0=censored).
    risk_scores : Continuous risk scores from model.
    eval_time : Time point at which to evaluate ROC.

    Returns
    -------
    Dictionary with AUC, sensitivity, specificity, and ROC curve data.
    """
    event_times = np.asarray(event_times)
    event_observed = np.asarray(event_observed)
    risk_scores = np.asarray(risk_scores)

    # Identify cases and controls at eval_time
    # Cases: event occurred before or at eval_time
    cases_mask = (event_times <= eval_time) & (event_observed == 1)
    # Controls: still at risk (event-free) at eval_time
    controls_mask = event_times > eval_time

    y_true = np.zeros(len(event_times))
    y_true[cases_mask] = 1

    # Only use cases and controls
    valid_mask = cases_mask | controls_mask
    y_true_valid = y_true[valid_mask]
    risk_scores_valid = risk_scores[valid_mask]

    if y_true_valid.sum() == 0 or y_true_valid.sum() == len(y_true_valid):
        logger.warning("Time-dependent ROC: only one class present at time %.2f", eval_time)
        return {
            "auc": np.nan,
            "fpr": np.array([]),
            "tpr": np.array([]),
            "thresholds": np.array([]),
            "n_cases": 0,
            "n_controls": 0,
        }

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_valid, risk_scores_valid)
    auc = stats.auc(fpr, tpr) if len(fpr) > 1 else np.nan

    result = {
        "auc": float(auc),
        "eval_time": float(eval_time),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "n_cases": int(cases_mask.sum()),
        "n_controls": int(controls_mask.sum()),
    }

    logger.info(
        "Time-dependent ROC at t=%.2f: AUC = %.4f (n_cases=%d, n_controls=%d)",
        eval_time, auc, result["n_cases"], result["n_controls"],
    )

    return result


def concordance_index(
    event_times: np.ndarray | pd.Series,
    event_observed: np.ndarray | pd.Series,
    risk_scores: np.ndarray | pd.Series,
) -> float:
    """Compute Harrell's Concordance Index (C-index).

    The C-index measures the discriminative ability of the risk scores.
    It represents the probability that, for a random pair of subjects,
    the subject with the higher risk score experiences the event first.

    Parameters
    ----------
    event_times : Times to event or censoring.
    event_observed : Event indicators (1=event, 0=censored).
    risk_scores : Continuous risk scores from model.

    Returns
    -------
    Concordance index (0.5 = random, 1.0 = perfect discrimination).
    """
    event_times = np.asarray(event_times)
    event_observed = np.asarray(event_observed)
    risk_scores = np.asarray(risk_scores)

    n = len(event_times)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Check if comparison is valid (at least one event observed)
            if event_observed[i] == 0 and event_observed[j] == 0:
                continue

            # Determine comparable pair
            if event_times[i] < event_times[j]:
                if event_observed[i] == 1:
                    # Subject i had event before j
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied += 1

            elif event_times[j] < event_times[i]:
                if event_observed[j] == 1:
                    # Subject j had event before i
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += 1
                    else:
                        tied += 1

    total = concordant + discordant + tied

    if total == 0:
        return 0.5

    c_index = (concordant + 0.5 * tied) / total

    logger.info("Concordance index: %.4f (concordant=%d, discordant=%d, tied=%d)", 
                c_index, concordant, discordant, tied)

    return float(c_index)


# ---------------------------------------------------------------------------
# Survival Data Preparation Helper
# ---------------------------------------------------------------------------
def prepare_survival_data(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    risk_score_col: Optional[str] = None,
    follow_up_max: Optional[float] = None,
) -> pd.DataFrame:
    """Prepare survival data from clinical dataframe.

    Parameters
    ----------
    df : Clinical dataframe.
    time_col : Column name for time-to-event.
    event_col : Column name for event indicator.
    risk_score_col : Optional column for risk scores.
    follow_up_max : Maximum follow-up time for truncation.

    Returns
    -------
    DataFrame prepared for survival analysis.
    """
    survival_data = df[[time_col, event_col]].copy()

    if risk_score_col and risk_score_col in df.columns:
        survival_data["risk_score"] = df[risk_score_col]

    # Truncate at maximum follow-up if specified
    if follow_up_max is not None:
        mask = survival_data[time_col] > follow_up_max
        survival_data.loc[mask, time_col] = follow_up_max
        survival_data.loc[mask, event_col] = 0  # Censor at truncation

    # Remove invalid entries
    survival_data = survival_data.dropna()
    survival_data = survival_data[survival_data[time_col] >= 0]

    return survival_data
