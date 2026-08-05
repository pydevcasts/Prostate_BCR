"""
Publication-quality visualization for TCGA-PRAD BCR prediction.

Generates all figures for the manuscript: ROC curves, PR curves,
calibration plots, confusion matrices, feature importance bar charts,
and SHAP-based visualizations. All outputs are saved at 300 DPI.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as config
from src.io import logger, save_figure


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
def setup_style() -> None:
    """Apply publication-quality matplotlib style."""
    try:
        plt.style.use(config.FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": config.FIGURE_DPI,
    })


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------
def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    filename: str | None = None,
    figsize: tuple[int, int] = (7, 6),
) -> plt.Figure:
    """Plot ROC curve with AUC annotation.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities for the positive class.
    model_name : Name for the legend label.
    filename : If provided, save figure to outputs/figures/.
    figsize : Figure dimensions.

    Returns
    -------
    Matplotlib Figure object.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    setup_style()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, linewidth=2, color="#E64B35",
            label=f"{model_name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="grey",
            label="Random Classifier")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — Held-out Test Set")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    if filename:
        save_figure(fig, filename)
    plt.close(fig)
    logger.info("ROC curve plotted (AUC=%.4f)", auc_val)
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall Curve
# ---------------------------------------------------------------------------
def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    filename: str | None = None,
    figsize: tuple[int, int] = (7, 6),
) -> plt.Figure:
    """Plot Precision-Recall curve with PR-AUC annotation.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities for the positive class.
    model_name : Name for the legend label.
    filename : If provided, save figure to outputs/figures/.
    figsize : Figure dimensions.

    Returns
    -------
    Matplotlib Figure object.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    setup_style()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, linewidth=2, color="#4DBBD5",
            label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve — Held-out Test Set")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    if filename:
        save_figure(fig, filename)
    plt.close(fig)
    logger.info("PR curve plotted (PR-AUC=%.4f)", pr_auc)
    return fig


# ---------------------------------------------------------------------------
# Calibration Plot
# ---------------------------------------------------------------------------
def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    filename: str | None = None,
    figsize: tuple[int, int] = (7, 6),
) -> plt.Figure:
    """Plot calibration curve (reliability diagram).

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_prob : Predicted probabilities.
    model_name : Name for the legend label.
    n_bins : Number of calibration bins.
    filename : If provided, save figure to outputs/figures/.
    figsize : Figure dimensions.

    Returns
    -------
    Matplotlib Figure object.
    """
    from sklearn.calibration import calibration_curve

    setup_style()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(prob_pred, prob_true, "o-", linewidth=2, color="#00A087",
            label=model_name)
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="grey",
            label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve — Held-out Test Set")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    if filename:
        save_figure(fig, filename)
    plt.close(fig)
    logger.info("Calibration plot generated")
    return fig


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str | None = None,
    figsize: tuple[int, int] = (6, 5),
) -> plt.Figure:
    """Plot annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true : Ground truth binary labels.
    y_pred : Predicted binary labels.
    filename : If provided, save figure to outputs/figures/.
    figsize : Figure dimensions.

    Returns
    -------
    Matplotlib Figure object.
    """
    from sklearn.metrics import confusion_matrix

    setup_style()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Held-out Test Set")

    if filename:
        save_figure(fig, filename)
    plt.close(fig)
    logger.info("Confusion matrix plotted")
    return fig


# ---------------------------------------------------------------------------
# Feature Importance Bar Chart
# ---------------------------------------------------------------------------
def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 20,
    title: str = "Top Feature Importance",
    filename: str | None = None,
    figsize: tuple[int, int] = (8, 7),
) -> plt.Figure:
    """Plot horizontal bar chart of top features by importance.

    Parameters
    ----------
    importance_df : DataFrame with 'feature' and 'importance' columns.
    top_k : Number of top features to display.
    title : Plot title.
    filename : If provided, save figure to outputs/figures/.
    figsize : Figure dimensions.

    Returns
    -------
    Matplotlib Figure object.
    """
    setup_style()
    top = importance_df.head(top_k).sort_values("importance")

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["feature"], top["importance"], color="#3C5488")
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)

    if filename:
        save_figure(fig, filename)
    plt.close(fig)
    logger.info("Feature importance plot generated (%d features)", len(top))
    return fig


# ---------------------------------------------------------------------------
# Multi-model ROC comparison
# ---------------------------------------------------------------------------
def plot_multi_model_roc(
    model_results: dict[str, tuple[np.ndarray, np.ndarray]],
    filename: str | None = None,
    figsize: tuple[int, int] = (8, 7),
) -> plt.Figure:
    """Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    model_results : Dict mapping model name to (y_true, y_prob) tuple.
    filename : If provided, save figure to outputs/figures/.
    figsize : Figure dimensions.

    Returns
    -------
    Matplotlib Figure object.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    setup_style()
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))

    fig, ax = plt.subplots(figsize=figsize)
    for (name, (y_true, y_prob)), color in zip(model_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Model Comparison — ROC Curves")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    if filename:
        save_figure(fig, filename)
    plt.close(fig)
    logger.info("Multi-model ROC comparison plotted (%d models)", len(model_results))
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison_bar(
    metrics_df: pd.DataFrame,
    metrics_to_show: list[str] | None = None,
    filename: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot grouped bar chart comparing models across metrics.

    Parameters
    ----------
    metrics_df : DataFrame with 'model' column and metric columns.
    metrics_to_show : List of metric column names. Defaults to common ones.
    filename : If provided, save figure to outputs/figures/.
    figsize : Figure dimensions.

    Returns
    -------
    Matplotlib Figure object.
    """
    setup_style()
    if metrics_to_show is None:
        metrics_to_show = ["roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy"]
        metrics_to_show = [m for m in metrics_to_show if m in metrics_df.columns]

    plot_df = metrics_df.set_index("model")[metrics_to_show]

    fig, ax = plt.subplots(figsize=figsize)
    plot_df.plot(kind="bar", ax=ax, rot=30, colormap="tab10", width=0.8)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_ylim([0, 1.05])
    ax.legend(title="Metric", framealpha=0.9)
    ax.tick_params(axis="x", labelsize=9)

    if filename:
        save_figure(fig, filename)
    plt.close(fig)
    logger.info("Model comparison bar chart generated")
    return fig


# ---------------------------------------------------------------------------
# Gene expression heatmap
# ---------------------------------------------------------------------------
def plot_gene_heatmap(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    top_genes: list[str],
    filename: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot heatmap of top gene expressions grouped by BCR status.

    Parameters
    ----------
    X : Feature matrix (samples × genes).
    y : Target labels.
    top_genes : List of gene names to display.
    filename : If provided, save figure to outputs/figures/.
    figsize : Figure dimensions.

    Returns
    -------
    Matplotlib Figure object.
    """
    setup_style()
    subset = X[top_genes].copy()
    subset["_label"] = np.asarray(y)
    subset = subset.sort_values("_label")
    labels = subset.pop("_label")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(subset.T, cmap="RdBu_r", center=0, ax=ax,
                xticklabels=False, yticklabels=True,
                cbar_kws={"label": "Expression (z-score)"})
    ax.set_title("Top Gene Expression by BCR Status")
    ax.set_xlabel("Samples (sorted by BCR status)")
    ax.set_ylabel("Gene")

    if filename:
        save_figure(fig, filename)
    plt.close(fig)
    logger.info("Gene heatmap generated (%d genes)", len(top_genes))
    return fig