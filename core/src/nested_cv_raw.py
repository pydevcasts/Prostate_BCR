"""
Repeated nested CV from raw TCGA data (Pipeline Step 6).

The definitive internal estimate: inside EVERY outer fold, the complete
pipeline is refitted from the raw merged table (X_features_final):
preprocessing (log1p on skewed clinical + winsorization + z-score), then
Variance -> MI -> PSO feature selection per branch, then branch models, with
OOF fusion weights and the Youden threshold estimated only afterwards from
the pooled out-of-fold predictions. Nothing outside the fold's training data
is used at any point.

Prespecification (declared before any result of this step was seen):
    - 5 outer folds x 3 repeats, StratifiedKFold(shuffle=True), seeds 42, 43, 44.
    - Identical recipe to the earlier single-repeat nested evaluation
      (evaluate_nested_late_fusion) PLUS fold-local preprocessing:
      ColumnTransformer built by build_combined_pipeline, fitted on the fold's
      training rows only. PSO stays enabled (12 particles x 10 iterations),
      inner CV 3 folds for fitness.
    - Per-fold PSO selections from every repeat are recorded for the
      Step 7 stability analysis (no separate PSO reruns needed).
    - Runtime control: per-fold timing is logged; a pilot run (1 repeat)
      validates timing before the full run.

Outputs
-------
- core/outputs/tables/nested_cv_raw_results.json       (per-repeat OOF metrics)
- core/outputs/tables/nested_cv_raw_folds.csv          (per-fold rows, all repeats)
- core/outputs/tables/nested_cv_raw_pso_selections.csv (stability input, Step 7)
- core/outputs/tables/nested_cv_raw_oof_predictions.csv

Run from the ``core`` directory:

    python -m src.nested_cv_raw            # full run (3 repeats)
    python -m src.nested_cv_raw --pilot    # pilot: 1 repeat
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import config
from src.clinical_selector import fit_clinical_selector, transform_clinical
from src.genomic_selector import (
    fit_genomic_selector,
    pso_feature_select_genomic,
    transform_genomic,
)
from src.io import logger
from src.models import make_xgb, xgb_safe_frame
from src.preprocessing import (
    build_combined_pipeline,
    fit_preprocessing,
    transform_data,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
X_FINAL_CSV = config.PROCESSED_DIR / "X_features_final.csv"
Y_FINAL_CSV = config.PROCESSED_DIR / "y_target_final.csv"
RESULTS_JSON = config.TABLES_DIR / "nested_cv_raw_results.json"
FOLDS_CSV = config.TABLES_DIR / "nested_cv_raw_folds.csv"
PSO_CSV = config.TABLES_DIR / "nested_cv_raw_pso_selections.csv"
OOF_CSV = config.TABLES_DIR / "nested_cv_raw_oof_predictions.csv"

REPEAT_SEEDS = (42, 43, 44)          # full run: 3 repeats
REPEAT_SEEDS_PILOT = (42,)           # pilot: 1 repeat
OUTER_SPLITS = config.OUTER_SPLITS   # 5

# Clinical (non-gene) columns of the merged table, verified against the file.
# Numeric clinical columns carry these base names; one-hot clinical columns
# are prefixed by them. Everything else is a gene symbol.
CLINICAL_BASE_NAMES = [
    "Gleason pattern primary",
    "Gleason pattern secondary",
    "Gleason pattern tertiary",
    "Radical Prostatectomy Gleason Score for Prostate Cancer",
    "American Joint Committee on Cancer Tumor Stage Code",
    "Primary Lymph Node Presentation Assessment Ind-3",
    "Surgical Margin Resection Status",
    "Positive Finding Lymph Node Hematoxylin and Eosin Staining Microscopy Count",
    "Neopl Disease Lymph Node Stage American Joint Committee on Cancer Code",
    "Person Neopl Status",
    "Primary Therapy Outcome Success Type",
    "Did patient start adjuvant postoperative radiotherapy?",
    "International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code",
    "Neopl American Joint Committee on Cancer Clinical Primary Tumor T Stage",
    "Ct scan ab pelvis indicator",
    "Year Cancer Initial Diagnosis",
    "Tissue Retrospective Collection Indicator",
    "Tissue Prospective Collection Indicator",
    "Mri results",
    "Ct scan ab pelvis results",
    "Patient Primary Tumor Site",
    "Diagonstic MRI Result",
    "Tumor Level",
    "Cause of death source",
    "Prior Cancer Diagnosis Occurence",
]


def _is_clinical_column(col: str) -> bool:
    """True if the merged-table column belongs to the clinical branch."""
    return any(col == base or col.startswith(base + "_") for base in CLINICAL_BASE_NAMES)


def load_raw_merged() -> tuple[pd.DataFrame, pd.Series]:
    """Load the raw merged clinical + expression table and the target."""
    X = pd.read_csv(X_FINAL_CSV)
    y = pd.read_csv(Y_FINAL_CSV).iloc[:, 0]
    if len(X) != len(y):
        raise RuntimeError("X_features_final and y_target_final misaligned")
    logger.info("Raw merged table: %d samples x %d features", *X.shape)
    return X, y


# ---------------------------------------------------------------------------
# Per-fold pipeline (canonical recipe, refit from scratch every fold)
# ---------------------------------------------------------------------------
def run_single_fold(
    X_raw: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    *,
    fold_seed: int,
    run_pso: bool = True,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Fit the complete fold-local pipeline; return predictions and PSO picks."""
    t0 = time.time()
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]
    X_train = X_raw.iloc[train_idx].reset_index(drop=True)
    X_valid = X_raw.iloc[valid_idx].reset_index(drop=True)

    clinical_mask = [c for c in X_raw.columns if _is_clinical_column(c)]
    gene_mask = [c for c in X_raw.columns if not _is_clinical_column(c)]
    Xc_train_raw = X_train[clinical_mask]
    Xc_valid_raw = X_valid[clinical_mask]
    Xg_train_raw = X_train[gene_mask]
    Xg_valid_raw = X_valid[gene_mask]

    # -- 1. Fold-local preprocessing ------------------------------------------
    preprocessor = fit_preprocessing(
        build_combined_pipeline(X_train), X_train
    )
    train_pp = transform_data(preprocessor, X_train)
    valid_pp = transform_data(preprocessor, X_valid)

    # ColumnTransformer drops column names; rebuild branch frames positionally:
    # clinical columns are transformed first, then gene columns (the order in
    # build_combined_pipeline).
    n_clinical = len(clinical_mask)
    Xc_train = pd.DataFrame(train_pp[:, :n_clinical], columns=clinical_mask)
    Xc_valid = pd.DataFrame(valid_pp[:, :n_clinical], columns=clinical_mask)
    Xg_train = pd.DataFrame(train_pp[:, n_clinical:], columns=gene_mask)
    Xg_valid = pd.DataFrame(valid_pp[:, n_clinical:], columns=gene_mask)
    t_pre = time.time() - t0

    # -- 2. Fold-local feature selection per branch ---------------------------
    genomic_selector = fit_genomic_selector(
        Xg_train, y_train, random_state=fold_seed
    )
    clinical_selector = fit_clinical_selector(
        Xc_train, y_train, random_state=fold_seed
    )

    genomic_candidates = genomic_selector["mi_features"]
    clinical_candidates = clinical_selector["mi_features"]

    if run_pso:
        genomic_features, g_score, _, _ = pso_feature_select_genomic(
            Xg_train, y_train, genomic_candidates,
            n_features=min(config.PSO_FINAL_K, len(genomic_candidates)),
            random_state=fold_seed + 1000,
            branch_name="NestedCV genomic",
        )
        clinical_features, c_score, _, _ = pso_feature_select_genomic(
            Xc_train, y_train, clinical_candidates,
            n_features=min(config.PSO_FINAL_K, len(clinical_candidates)),
            random_state=fold_seed + 2000,
            branch_name="NestedCV clinical",
        )
    else:
        genomic_features = genomic_candidates[: config.PSO_FINAL_K]
        clinical_features = clinical_candidates[: config.PSO_FINAL_K]

    Xg_train_sel = transform_genomic(Xg_train, genomic_selector, genomic_features)
    Xg_valid_sel = transform_genomic(Xg_valid, genomic_selector, genomic_features)
    Xc_train_sel = transform_clinical(Xc_train, clinical_selector, clinical_features)
    Xc_valid_sel = transform_clinical(Xc_valid, clinical_selector, clinical_features)
    t_select = time.time() - t0 - t_pre

    # -- 3. Fold-local branch models -------------------------------------------
    genomic_model = make_xgb(y_train)
    clinical_model = make_xgb(y_train)
    genomic_model.fit(xgb_safe_frame(Xg_train_sel), y_train)
    clinical_model.fit(xgb_safe_frame(Xc_train_sel), y_train)
    g_proba = genomic_model.predict_proba(xgb_safe_frame(Xg_valid_sel))[:, 1]
    c_proba = clinical_model.predict_proba(xgb_safe_frame(Xc_valid_sel))[:, 1]
    t_fit = time.time() - t0 - t_pre - t_select

    fold_info = {
        "n_genomic_features": len(genomic_features),
        "n_clinical_features": len(clinical_features),
        "genomic_auc": float(roc_auc_score(y_valid, g_proba)),
        "clinical_auc": float(roc_auc_score(y_valid, c_proba)),
        "seconds_preprocess": round(t_pre, 1),
        "seconds_selection": round(t_select, 1),
        "seconds_fit": round(t_fit, 1),
        "seconds_total": round(time.time() - t0, 1),
    }
    pso_picks = {"genomic": genomic_features, "clinical": clinical_features}
    return fold_info, g_proba, c_proba, pso_picks


def run_repeat(
    X_raw: pd.DataFrame,
    y: pd.Series,
    seed: int,
    run_pso: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Run one 5-fold repeat; return results, fold table, PSO picks, OOF arrays."""
    splitter = StratifiedKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=seed
    )
    y_array = y.values
    oof_g = np.full(len(y_array), np.nan)
    oof_c = np.full(len(y_array), np.nan)
    fold_rows: list[dict[str, Any]] = []
    pso_rows: list[dict[str, Any]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_raw, y_array), start=1):
        logger.info("Repeat seed=%d fold %d/%d ...", seed, fold, OUTER_SPLITS)
        info, g_proba, c_proba, pso_picks = run_single_fold(
            X_raw, y, train_idx, valid_idx,
            fold_seed=seed + fold, run_pso=run_pso,
        )
        oof_g[valid_idx] = g_proba
        oof_c[valid_idx] = c_proba
        fold_rows.append({"repeat_seed": seed, "fold": fold, **info})
        for branch, feats in pso_picks.items():
            for rank, feat in enumerate(feats, start=1):
                pso_rows.append({
                    "repeat_seed": seed, "fold": fold,
                    "branch": branch, "feature": feat, "rank": rank,
                })

    if np.isnan(oof_g).any() or np.isnan(oof_c).any():
        raise RuntimeError("Unfilled OOF predictions after the repeat")

    # -- OOF fusion weights + threshold (pooled across folds) ------------------
    weights = np.linspace(0.0, 1.0, 101)
    fusion_aucs = [
        roc_auc_score(y_array, w * oof_g + (1 - w) * oof_c) for w in weights
    ]
    best = int(np.argmax(fusion_aucs))
    w_gen = float(weights[best])
    oof_fusion = w_gen * oof_g + (1 - w_gen) * oof_c
    fpr, tpr, thresholds = roc_curve(y_array, oof_fusion)
    threshold = float(thresholds[int(np.argmax(tpr - fpr))])

    fold_df = pd.DataFrame(fold_rows)
    pso_df = pd.DataFrame(pso_rows)
    results = {
        "repeat_seed": seed,
        "oof_auc": float(fusion_aucs[best]),
        "genomic_weight": w_gen,
        "clinical_weight": 1.0 - w_gen,
        "threshold": threshold,
        "fold_mean_genomic_auc": float(fold_df["genomic_auc"].mean()),
        "fold_mean_clinical_auc": float(fold_df["clinical_auc"].mean()),
        "total_seconds": float(fold_df["seconds_total"].sum()),
    }
    return results, fold_df, pso_df, oof_g, oof_c, oof_fusion


def main() -> None:
    """Pilot or full repeated nested CV from raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="run 1 repeat only")
    args = parser.parse_args()

    seeds = REPEAT_SEEDS_PILOT if args.pilot else REPEAT_SEEDS
    X_raw, y = load_raw_merged()

    all_results, all_folds, all_pso = [], [], []
    oof_frames = []
    run_start = time.time()
    for seed in seeds:
        results, fold_df, pso_df, oof_g, oof_c, oof_f = run_repeat(X_raw, y, seed)
        all_results.append(results)
        all_folds.append(fold_df)
        all_pso.append(pso_df)
        oof_frames.append(pd.DataFrame({
            "repeat_seed": seed,
            "row_index": np.arange(len(y)),
            "y": y.values,
            "oof_genomic": oof_g,
            "oof_clinical": oof_c,
            "oof_fusion": oof_f,
        }))
        logger.info(
            "Repeat seed=%d done: OOF AUC=%.4f (weights %.2f/%.2f), %.1f min",
            seed, results["oof_auc"], results["genomic_weight"],
            results["clinical_weight"], results["total_seconds"] / 60,
        )

    folds = pd.concat(all_folds, ignore_index=True)
    pso = pd.concat(all_pso, ignore_index=True)
    oofs = pd.concat(oof_frames, ignore_index=True)

    # -- Summary across repeats -------------------------------------------------
    aucs = [r["oof_auc"] for r in all_results]
    summary = {
        "n_repeats": len(seeds),
        "repeat_seeds": list(seeds),
        "outer_splits": OUTER_SPLITS,
        "mean_oof_auc": float(np.mean(aucs)),
        "std_oof_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "min_oof_auc": float(np.min(aucs)),
        "max_oof_auc": float(np.max(aucs)),
        "mean_genomic_weight": float(np.mean([r["genomic_weight"] for r in all_results])),
        "total_runtime_minutes": round((time.time() - run_start) / 60, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_repeat": all_results,
    }

    FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_CSV, index=False)
    pso.to_csv(PSO_CSV, index=False)
    oofs.to_csv(OOF_CSV, index=False)
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Step 6: repeated nested CV from raw TCGA ===")
    print(f"Repeats: {len(seeds)} x {OUTER_SPLITS} folds | runtime {summary['total_runtime_minutes']} min")
    for r in all_results:
        print(f"  seed {r['repeat_seed']}: OOF AUC={r['oof_auc']:.4f}, "
              f"weights={r['genomic_weight']:.2f}/{r['clinical_weight']:.2f}, "
              f"threshold={r['threshold']:.4f}")
    print(f"Mean OOF AUC = {summary['mean_oof_auc']:.4f} ± {summary['std_oof_auc']:.4f}")


if __name__ == "__main__":
    main()
