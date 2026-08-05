"""
Model explainability for TCGA-PRAD BCR prediction.

Implements SHAP-based interpretation, feature importance extraction,
and biomarker ranking for publication-quality figures.

All functions accept a fitted model and return structured outputs
(DataFrames or matplotlib figures) ready for saving.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config as config
from src.io import logger


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def compute_feature_importance(
    model: Any,
    feature_names: list[str],
    top_k: int = 20,
) -> pd.DataFrame:
    """Extract feature importance from a fitted model.

    Parameters
    ----------
    model : Fitted model with feature_importances_ attribute.
    feature_names : List of feature names corresponding to model input.
    top_k : Number of top features to return.

    Returns
    -------
    DataFrame with feature names and importance scores, sorted descending.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Extracted feature importance: %d features", len(importance))
    return importance.head(top_k)


def get_biomarker_ranking(
    importance_df: pd.DataFrame,
    feature_type: str = "gene",
) -> pd.DataFrame:
    """Rank biomarkers from feature importance DataFrame.

    Parameters
    ----------
    importance_df : Output of compute_feature_importance.
    feature_type : Type of features ('gene' or 'clinical').

    Returns
    -------
    DataFrame with ranked biomarkers.
    """
    ranking = importance_df.copy()
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking["feature_type"] = feature_type

    logger.info("Biomarker ranking: %d %s features", len(ranking), feature_type)
    return ranking[["rank", "feature", "importance", "feature_type"]]


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------
def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for a fitted model.

    Parameters
    ----------
    model : Fitted model.
    X : Data to compute SHAP values for.
    feature_names : Optional list of feature names.

    Returns
    -------
    Tuple of (shap_values, base_value).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]

    logger.info("Computed SHAP values: %d samples × %d features", *shap_values.shape)
    return shap_values, base_value


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    figsize: tuple[int, int] = (10, 8),
) -> Any:
    """Create SHAP summary plot.

    Parameters
    ----------
    shap_values : SHAP values array.
    X : Feature matrix.
    max_display : Maximum number of features to display.
    figsize : Figure size.

    Returns
    -------
    Matplotlib figure.
    """
    import shap
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()

    logger.info("Created SHAP summary plot")
    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    base_value: float,
    X: pd.DataFrame,
    sample_index: int = 0,
    figsize: tuple[int, int] = (10, 8),
) -> Any:
    """Create SHAP waterfall plot for a single sample.

    Parameters
    ----------
    shap_values : SHAP values array.
    base_value : Base value from explainer.
    X : Feature matrix.
    sample_index : Index of sample to explain.
    figsize : Figure size.

    Returns
    -------
    Matplotlib figure.
    """
    import shap
    import matplotlib.pyplot as plt

    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=base_value,
        data=X.iloc[sample_index].values,
        feature_names=X.columns.tolist(),
    )

    plt.figure(figsize=figsize)
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()

    logger.info("Created SHAP waterfall plot for sample %d", sample_index)
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_name: str,
    interaction_feature: str | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> Any:
    """Create SHAP dependence plot for a specific feature.

    Parameters
    ----------
    shap_values : SHAP values array.
    X : Feature matrix.
    feature_name : Name of feature to plot.
    interaction_feature : Optional interaction feature for coloring.
    figsize : Figure size.

    Returns
    -------
    Matplotlib figure.
    """
    import shap
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    shap.dependence_plot(
        feature_name,
        shap_values,
        X,
        interaction_index=interaction_feature,
        show=False,
    )
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()

    logger.info("Created SHAP dependence plot for feature: %s", feature_name)
    return fig


# ---------------------------------------------------------------------------
# Full explainability pipeline
# ---------------------------------------------------------------------------
def run_explainability(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 20,
    run_shap: bool = True,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Run the full explainability pipeline.

    Parameters
    ----------
    model : Fitted model.
    X_test : Test data for SHAP analysis.
    feature_names : List of feature names.
    top_k : Number of top features to extract.
    run_shap : Whether to run SHAP analysis.
    sample_index : Sample index for waterfall plot.

    Returns
    -------
    Dictionary with importance DataFrame and optional SHAP figures.
    """
    results: dict[str, Any] = {}

    # Feature importance
    importance_df = compute_feature_importance(model, feature_names, top_k)
    results["feature_importance"] = importance_df
    results["biomarker_ranking"] = get_biomarker_ranking(importance_df)

    # SHAP analysis
    if run_shap:
        try:
            shap_values, base_value = compute_shap_values(model, X_test, feature_names)
            results["shap_values"] = shap_values
            results["base_value"] = base_value
            results["summary_plot"] = plot_shap_summary(shap_values, X_test)
            results["waterfall_plot"] = plot_shap_waterfall(
                shap_values, base_value, X_test, sample_index
            )
        except Exception as e:
            logger.warning("SHAP analysis failed: %s", e)
            results["shap_error"] = str(e)

    logger.info("Explainability pipeline complete")
    return results