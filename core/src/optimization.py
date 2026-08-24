"""
Hyperparameter Optimization using Optuna for Prostate BCR Prediction.

This module provides Optuna-based hyperparameter optimization for XGBoost,
LightGBM, and CatBoost models. It supports:
- Bayesian optimization with pruning
- Cross-validation based evaluation
- Feature importance extraction
- Ensemble model creation
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Please install optuna: pip install optuna")

from src.io import logger
from src.models import (
    build_model,
    get_all_model_names,
    requires_xgb_safe,
    xgb_safe_frame,
)


# ---------------------------------------------------------------------------
# Optuna Study Configuration
# ---------------------------------------------------------------------------
def create_optuna_study(
    study_name: str = "bcr_optimization",
    direction: str = "maximize",
    sampler: Optional[Any] = None,
    pruner: Optional[Any] = None,
) -> optuna.Study:
    """Create an Optuna study with recommended settings.
    
    Parameters
    ----------
    study_name : Name of the study.
    direction : Optimization direction ("maximize" or "minimize").
    sampler : Optuna sampler (default: TPESampler).
    pruner : Optuna pruner (default: MedianPruner).
    
    Returns
    -------
    Configured Optuna Study object.
    """
    if sampler is None:
        sampler = TPESampler(seed=42, multivariate=True)
    
    if pruner is None:
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )
    
    logger.info("Created Optuna study: %s (direction=%s)", study_name, direction)
    return study


# ---------------------------------------------------------------------------
# Search Space Definitions
# ---------------------------------------------------------------------------
def suggest_xgboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest XGBoost hyperparameters for a trial.
    
    Parameters
    ----------
    trial : Optuna trial object.
    
    Returns
    -------
    Dictionary of suggested parameters.
    """
    params = {
        # Tree structure
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        
        # Learning rate and estimators
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        
        # Subsampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        
        # Regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        
        # Other
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "random_state": 42,
        "n_jobs": -1,
    }
    
    return params


def suggest_lightgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest LightGBM hyperparameters for a trial.
    
    Parameters
    ----------
    trial : Optuna trial object.
    
    Returns
    -------
    Dictionary of suggested parameters.
    """
    params = {
        # Tree structure
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        
        # Learning rate and estimators
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        
        # Subsampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        
        # Regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        
        # Other
        "random_state": 42,
        "n_jobs": -1,
    }
    
    return params


def suggest_catboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest CatBoost hyperparameters for a trial.
    
    Parameters
    ----------
    trial : Optuna trial object.
    
    Returns
    -------
    Dictionary of suggested parameters.
    """
    params = {
        # Tree structure
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        
        # Learning rate and estimators
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        
        # Subsampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        
        # Other
        "random_state": 42,
        "verbose": 0,
    }
    
    return params


# ---------------------------------------------------------------------------
# Objective Functions
# ---------------------------------------------------------------------------
def objective_xgboost(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
) -> float:
    """Optuna objective function for XGBoost optimization.
    
    Parameters
    ----------
    trial : Optuna trial object.
    X_train : Training features.
    y_train : Training target.
    cv_splits : Number of CV folds.
    scoring : Scoring metric.
    
    Returns
    -------
    Mean CV score (to maximize).
    """
    # Suggest parameters
    params = suggest_xgboost_params(trial)
    
    # Build model
    try:
        from xgboost import XGBClassifier
        
        # Handle scale_pos_weight calculation
        y_array = np.asarray(y_train)
        n_neg = (y_array == 0).sum()
        n_pos = (y_array == 1).sum()
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        
        model = XGBClassifier(**params)
        
        # Apply XGBoost-safe column names if needed
        if requires_xgb_safe("XGBoost"):
            X_train_safe = xgb_safe_frame(X_train)
        else:
            X_train_safe = X_train
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_safe, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        mean_score = float(np.mean(scores))
        
        # Report intermediate result for pruning
        trial.report(mean_score, trial.number)
        
        # Check for pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_score
        
    except Exception as e:
        logger.warning("Trial failed: %s", str(e))
        return 0.0


def objective_lightgbm(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
) -> float:
    """Optuna objective function for LightGBM optimization."""
    from lightgbm import LGBMClassifier
    
    params = suggest_lightgbm_params(trial)
    
    # Handle class imbalance
    y_array = np.asarray(y_train)
    n_neg = (y_array == 0).sum()
    n_pos = (y_array == 1).sum()
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)
    
    model = LGBMClassifier(**params)
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    mean_score = float(np.mean(scores))
    trial.report(mean_score, trial.number)
    
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return mean_score


def objective_catboost(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
) -> float:
    """Optuna objective function for CatBoost optimization."""
    from catboost import CatBoostClassifier
    
    params = suggest_catboost_params(trial)
    model = CatBoostClassifier(**params)
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    mean_score = float(np.mean(scores))
    trial.report(mean_score, trial.number)
    
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return mean_score


# ---------------------------------------------------------------------------
# Main Optimization Function
# ---------------------------------------------------------------------------
def optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_trials: int = 100,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
) -> Tuple[Any, optuna.Study, Dict[str, Any]]:
    """Run Optuna hyperparameter optimization for a specified model.
    
    Parameters
    ----------
    model_name : Model name ("XGBoost", "LightGBM", or "CatBoost").
    X_train : Training features.
    y_train : Training target.
    n_trials : Maximum number of trials.
    cv_splits : Number of CV folds.
    scoring : Scoring metric.
    timeout : Timeout in seconds (optional).
    study_name : Custom study name (optional).
    
    Returns
    -------
    Tuple of (best_estimator, study, best_params).
    """
    # Select objective function
    if model_name == "XGBoost":
        objective = lambda trial: objective_xgboost(
            trial, X_train, y_train, cv_splits, scoring
        )
    elif model_name == "LightGBM":
        objective = lambda trial: objective_lightgbm(
            trial, X_train, y_train, cv_splits, scoring
        )
    elif model_name == "CatBoost":
        objective = lambda trial: objective_catboost(
            trial, X_train, y_train, cv_splits, scoring
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create study
    if study_name is None:
        study_name = f"{model_name}_BCR_Optimization"
    
    study = create_optuna_study(study_name=study_name)
    
    # Run optimization
    logger.info(
        "Starting Optuna optimization for %s: %d trials, %d CV splits",
        model_name, n_trials, cv_splits
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(
        "Optimization complete! Best %s = %.4f", scoring, best_score
    )
    logger.info("Best parameters: %s", best_params)
    
    # Build final model with best parameters
    # Add missing parameters that weren't in search space
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        
        # Calculate scale_pos_weight
        y_array = np.asarray(y_train)
        n_neg = (y_array == 0).sum()
        n_pos = (y_array == 1).sum()
        best_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1
        
        best_model = XGBClassifier(**best_params)
        
        # Fit on full training data
        if requires_xgb_safe("XGBoost"):
            X_train_safe = xgb_safe_frame(X_train)
        else:
            X_train_safe = X_train
        
        best_model.fit(X_train_safe, y_train)
        
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        
        y_array = np.asarray(y_train)
        n_neg = (y_array == 0).sum()
        n_pos = (y_array == 1).sum()
        best_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1
        
        best_model = LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        
        best_params["random_state"] = 42
        best_params["verbose"] = 0
        
        best_model = CatBoostClassifier(**best_params)
        best_model.fit(X_train, y_train)
    
    return best_model, study, best_params


# ---------------------------------------------------------------------------
# Ensemble Learning
# ---------------------------------------------------------------------------
class VotingEnsemble:
    """Soft voting ensemble classifier for BCR prediction.
    
    Combines predictions from XGBoost, LightGBM, and CatBoost
    using weighted averaging of predicted probabilities.
    """
    
    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        weights: Optional[List[float]] = None,
    ):
        """Initialize the ensemble.
        
        Parameters
        ----------
        models : Dictionary of {name: fitted_model}.
        weights : Optional list of weights for each model (default: equal).
        """
        self.models = models or {}
        self.weights = weights
        self.model_names: List[str] = []
        
        if self.models:
            self.model_names = list(self.models.keys())
            if self.weights is None:
                self.weights = [1.0 / len(self.models)] * len(self.models)
            else:
                # Normalize weights
                total = sum(self.weights)
                self.weights = [w / total for w in self.weights]
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        model_configs: Optional[Dict[str, Dict]] = None,
    ) -> "VotingEnsemble":
        """Fit all models in the ensemble.
        
        Parameters
        ----------
        X_train : Training features.
        y_train : Training target.
        model_configs : Optional dict of model-specific configurations.
        
        Returns
        -------
        Self.
        """
        from src.models import build_model, xgb_safe_frame
        
        if model_configs is None:
            model_configs = {}
        
        # Default models
        default_models = ["XGBoost", "LightGBM", "CatBoost"]
        
        for model_name in default_models:
            logger.info("Training %s for ensemble...", model_name)
            
            config = model_configs.get(model_name, {})
            
            # Build and train model
            if model_name in ["XGBoost", "LightGBM"]:
                model = build_model(model_name, y_train=y_train, **config)
            else:
                model = build_model(model_name, **config)
            
            # Handle XGBoost column names
            if model_name == "XGBoost":
                X_safe = xgb_safe_frame(X_train)
                model.fit(X_safe, y_train)
            else:
                model.fit(X_train, y_train)
            
            self.models[model_name] = model
        
        self.model_names = list(self.models.keys())
        
        # Set equal weights if not provided
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        logger.info("Ensemble trained with %d models", len(self.models))
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using soft voting.
        
        Parameters
        ----------
        X : Features DataFrame.
        
        Returns
        -------
        Array of shape (n_samples, 2) with class probabilities.
        """
        from src.models import xgb_safe_frame
        
        all_probs = []
        
        for i, (name, model) in enumerate(self.models.items()):
            # Handle XGBoost column names
            if name == "XGBoost":
                X_input = xgb_safe_frame(X)
            else:
                X_input = X
            
            # Get probabilities
            probs = model.predict_proba(X_input)[:, 1]
            all_probs.append(probs * self.weights[i])
        
        # Weighted average
        avg_probs = np.sum(all_probs, axis=0)
        
        # Return as 2D array
        return np.column_stack([1 - avg_probs, avg_probs])
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels.
        
        Parameters
        ----------
        X : Features DataFrame.
        threshold : Classification threshold.
        
        Returns
        -------
        Array of predicted class labels.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance from all models.
        
        Returns
        -------
        DataFrame with feature names and importance scores.
        """
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                
                # Get feature names
                if name == "XGBoost" and hasattr(model, 'get_booster'):
                    feature_names = model.get_booster().feature_names
                elif hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                else:
                    feature_names = [f"feature_{i}" for i in range(len(imp))]
                
                for fname, val in zip(feature_names, imp):
                    if fname not in importances:
                        importances[fname] = []
                    importances[fname].append(val)
        
        # Average across models
        avg_importance = {k: np.mean(v) for k, v in importances.items()}
        
        df = pd.DataFrame({
            "feature": list(avg_importance.keys()),
            "importance": list(avg_importance.values())
        })
        
        return df.sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# Feature Stability Analysis
# ---------------------------------------------------------------------------
def analyze_feature_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_folds: int = 5,
    n_iterations: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Analyze feature selection stability across CV folds.
    
    Parameters
    ----------
    X_train : Training features.
    y_train : Training target.
    n_folds : Number of CV folds.
    n_iterations : Number of iterations per fold.
    random_state : Random seed.
    
    Returns
    -------
    DataFrame with feature stability scores.
    """
    from src.feature_selection import run_feature_selection
    
    rng = np.random.RandomState(random_state)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    feature_counts: Dict[str, int] = {}
    total_selections = 0
    
    logger.info(
        "Analyzing feature stability: %d folds × %d iterations = %d total runs",
        n_folds, n_iterations, n_folds * n_iterations
    )
    
    for fold_idx, (train_idx, _) in enumerate(cv.split(X_train, y_train)):
        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
        
        for iter_idx in range(n_iterations):
            try:
                _, selected_features = run_feature_selection(
                    X_fold, y_fold,
                    random_state=random_state + fold_idx * n_iterations + iter_idx,
                )
                
                for feat in selected_features:
                    if feat not in feature_counts:
                        feature_counts[feat] = 0
                    feature_counts[feat] += 1
                
                total_selections += 1
                
            except Exception as e:
                logger.warning("Feature selection failed: %s", str(e))
                continue
    
    # Calculate stability scores
    stability_df = pd.DataFrame({
        "feature": list(feature_counts.keys()),
        "selection_count": list(feature_counts.values()),
        "stability_score": [c / total_selections for c in feature_counts.values()]
    })
    
    stability_df = stability_df.sort_values("stability_score", ascending=False)
    
    logger.info(
        "Stability analysis complete: %d features, top stability = %.3f",
        len(stability_df), stability_df["stability_score"].iloc[0] if len(stability_df) > 0 else 0
    )
    
    return stability_df
