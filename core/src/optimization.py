"""
Hyperparameter Optimization using Optuna for Prostate BCR Prediction.
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

# =============================================================================
# Parameter Suggestion Functions
# =============================================================================
def suggest_xgboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss"
    }

def suggest_lightgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

def suggest_catboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "random_state": 42,
        "verbose": 0
    }

# =============================================================================
# Objective Functions
# =============================================================================
def objective_xgboost(trial, X_train, y_train, cv_splits=5, scoring="roc_auc"):
    from xgboost import XGBClassifier
    params = suggest_xgboost_params(trial)
    y_array = np.asarray(y_train)
    n_neg = (y_array == 0).sum()
    n_pos = (y_array == 1).sum()
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)
    
    model = XGBClassifier(**params)
    X_train_safe = xgb_safe_frame(X_train) if requires_xgb_safe("XGBoost") else X_train
        
    try:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_safe, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = float(np.mean(scores))
        trial.report(mean_score, trial.number)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return mean_score
    except Exception as e:
        logger.warning(f"XGBoost Trial {trial.number} failed: {str(e)}")
        return 0.0

def objective_lightgbm(trial, X_train, y_train, cv_splits=5, scoring="roc_auc"):
    from lightgbm import LGBMClassifier
    params = suggest_lightgbm_params(trial)
    y_array = np.asarray(y_train)
    n_neg = (y_array == 0).sum()
    n_pos = (y_array == 1).sum()
    params["is_unbalance"] = True
    
    model = LGBMClassifier(**params)
    try:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = float(np.mean(scores))
        trial.report(mean_score, trial.number)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return mean_score
    except Exception as e:
        logger.warning(f"LightGBM Trial {trial.number} failed: {str(e)}")
        return 0.0

def objective_catboost(trial, X_train, y_train, cv_splits=5, scoring="roc_auc"):
    from catboost import CatBoostClassifier
    params = suggest_catboost_params(trial)
    model = CatBoostClassifier(**params)
    try:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = float(np.mean(scores))
        trial.report(mean_score, trial.number)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return mean_score
    except Exception as e:
        logger.warning(f"CatBoost Trial {trial.number} failed: {str(e)}")
        return 0.0

# =============================================================================
# Main Optimization Function
# =============================================================================
def optimize_model(model_name, X_train, y_train, n_trials=100, cv_splits=5, scoring="roc_auc", timeout=None, study_name=None):
    if model_name == "XGBoost":
        objective = lambda trial: objective_xgboost(trial, X_train, y_train, cv_splits, scoring)
    elif model_name == "LightGBM":
        objective = lambda trial: objective_lightgbm(trial, X_train, y_train, cv_splits, scoring)
    elif model_name == "CatBoost":
        objective = lambda trial: objective_catboost(trial, X_train, y_train, cv_splits, scoring)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if study_name is None:
        study_name = f"{model_name}_BCR_Optimization"
        
    study = create_optuna_study(study_name=study_name)
    logger.info(f"Starting Optuna optimization for {model_name}: {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"Optimization complete! Best {scoring} = {best_score:.4f}")
    
    # Build final model
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        y_array = np.asarray(y_train)
        n_neg = (y_array == 0).sum()
        n_pos = (y_array == 1).sum()
        best_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1
        best_params["eval_metric"] = "logloss"
        best_model = XGBClassifier(**best_params)
        X_train_safe = xgb_safe_frame(X_train) if requires_xgb_safe("XGBoost") else X_train
        best_model.fit(X_train_safe, y_train)
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1
        best_model = LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train)
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        best_params["random_state"] = 42
        best_params["verbose"] = 0
        best_model = CatBoostClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
    return best_model, study, best_params

class VotingEnsemble:
    """Soft voting ensemble classifier."""
    def __init__(self, models=None, weights=None):
        self.models = models or {}
        self.weights = weights
        self.model_names = list(self.models.keys()) if self.models else []
        if self.models and self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
            
    def fit(self, X_train, y_train, model_configs=None):
        if model_configs is None: model_configs = {}
        default_models = ["XGBoost", "LightGBM", "CatBoost"]
        for model_name in default_models:
            if model_name not in self.models:
                try:
                    model = build_model(model_name, y_train=y_train if model_name in ["XGBoost", "LightGBM"] else None, **model_configs.get(model_name, {}))
                    if model_name == "XGBoost":
                        model.fit(xgb_safe_frame(X_train), y_train)
                    else:
                        model.fit(X_train, y_train)
                    self.models[model_name] = model
                except Exception as e:
                    logger.warning(f"Failed to train {model_name}: {e}")
        self.model_names = list(self.models.keys())
        if self.weights is None and self.model_names:
            self.weights = [1.0 / len(self.model_names)] * len(self.model_names)
        return self
        
    def predict_proba(self, X):
        all_probs = []
        valid_weights = []
        for i, name in enumerate(self.model_names):
            try:
                X_input = xgb_safe_frame(X) if name == "XGBoost" else X
                probs = self.models[name].predict_proba(X_input)[:, 1]
                all_probs.append(probs * self.weights[i])
                valid_weights.append(self.weights[i])
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
        if not all_probs: raise RuntimeError("No models could generate predictions.")
        weighted_sum = np.sum(all_probs, axis=0)
        weight_sum = np.sum(valid_weights)
        return np.column_stack([1 - weighted_sum/weight_sum, weighted_sum/weight_sum])
        
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

def create_optuna_study(study_name, direction="maximize", sampler=None, pruner=None):
    if sampler is None: sampler = TPESampler(seed=42, multivariate=True)
    if pruner is None: pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(study_name=study_name, direction=direction, sampler=sampler, pruner=pruner)
    logger.info(f"Created Optuna study: {study_name}")
    return study

def analyze_feature_stability(X_train, y_train, n_folds=5, n_iterations=10, random_state=42):
    from src.feature_selection import run_3layer_feature_selection
    feature_counts = {}
    total_selections = 0
    rng = np.random.RandomState(random_state)
    for fold_idx in range(n_folds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold_idx)
        for tr_idx, _ in skf.split(X_train, y_train):
            for iter_idx in range(n_iterations):
                try:
                    _, final_features = run_3layer_feature_selection(X_train.iloc[tr_idx], y_train.iloc[tr_idx], random_state=rng.randint(0, 10000))
                    for feat in final_features:
                        feature_counts[feat] = feature_counts.get(feat, 0) + 1
                    total_selections += 1
                except Exception as e:
                    continue
    stability_df = pd.DataFrame({
        "feature": list(feature_counts.keys()),
        "selection_count": list(feature_counts.values()),
        "stability_score": [c / total_selections for c in feature_counts.values()]
    })
    return stability_df.sort_values("stability_score", ascending=False).reset_index(drop=True)