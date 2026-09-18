"""
Transferable Late Fusion retraining and external application (Pipeline Step 3).

Retrains both fusion branches on TCGA using ONLY the six transferable features
built in Step 2, estimates fusion weights and the operating threshold from
out-of-fold predictions (estimate_oof_fusion — no leakage), then applies the
completely frozen pipeline to the MSKCC 2010 cohort.

Prespecified transfer rule (recorded before looking at any external result):
    1. StandardScaler is FIT on the TCGA TRAIN rows only (343 rows) over the
       six transferable features and applied unchanged to TCGA train, TCGA
       test, and MSKCC. No scaler is ever fitted on MSKCC data.
    2. Branch models are XGBoost classifiers with the project's default
       hyperparameters (config.XGBOOST_PARAMS + automatic scale_pos_weight).
    3. Fusion weights and the operating threshold come from TCGA OOF
       predictions only. The SAME threshold is applied to MSKCC — no
       threshold tuning, calibration, or any other fitting on external data.
    4. The single MSKCC sample with missing Gleason fields is median-imputed
       from TCGA TRAIN values (deterministic, no external peeking) and the
       imputed rows are listed in the audit.

Outputs
-------
- core/outputs/tables/transferable_fusion_results.json
- core/outputs/tables/mskcc_transferable_predictions.csv
- core/outputs/tables/tcga_transferable_test_predictions.csv
- core/outputs/models/transferable_scaler.joblib
- core/outputs/models/transferable_genomic_model.joblib
- core/outputs/models/transferable_clinical_model.joblib

Run from the ``core`` directory:

    python -m src.transferable_fusion
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import config
from src.evaluation import stratified_bootstrap_auc
from src.fusion.late_fusion import estimate_oof_fusion
from src.io import logger
from src.models import make_xgb, xgb_safe_frame
from src.transferable_features import FEATURE_ORDER

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
TCGA_CSV = config.PROCESSED_DIR / "tcga_transferable_features.csv"
MSKCC_CSV = config.DATA_DIR / "external" / "mskcc_transferable_features.csv"
RESULTS_JSON = config.TABLES_DIR / "transferable_fusion_results.json"
MSKCC_PREDS_CSV = config.TABLES_DIR / "mskcc_transferable_predictions.csv"
TCGA_TEST_PREDS_CSV = config.TABLES_DIR / "tcga_transferable_test_predictions.csv"

GENOMIC_FEATURES = FEATURE_ORDER[:3]
CLINICAL_FEATURES = FEATURE_ORDER[3:]


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------
def _classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Compute the standard metric battery at a fixed threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "ppv": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "n": int(len(y_true)),
        "n_events": int(y_true.sum()),
        "predicted_positive": int(pred.sum()),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_transferable_fusion() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run the full Step 3 pipeline and return (results, mskcc_preds, tcga_test_preds)."""
    audit: dict[str, Any] = {}
    tcga = pd.read_csv(TCGA_CSV)
    mskcc = pd.read_csv(MSKCC_CSV)
    train_mask = tcga["split"] == "train"

    X_tcga = tcga[FEATURE_ORDER].astype(float)
    y_tcga = tcga["y"].astype(int).values
    X_train = X_tcga[train_mask].reset_index(drop=True)
    y_train = y_tcga[train_mask.values]
    X_test = X_tcga[~train_mask].reset_index(drop=True)
    y_test = y_tcga[(~train_mask).values]
    audit["tcga_train_rows"], audit["tcga_test_rows"] = len(X_train), len(X_test)

    # -- 1. Prespecified scaler: fit on TCGA train only, apply unchanged -----
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, config.MODELS_DIR / "transferable_scaler.joblib")
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=FEATURE_ORDER)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_ORDER)

    # MSKCC: impute the missing Gleason fields with TCGA-train medians FIRST
    # (imputation statistics come from training data only), then scale with
    # the frozen scaler.
    X_mskcc = mskcc[FEATURE_ORDER].astype(float).copy()
    imputed_rows = X_mskcc.isna().any(axis=1)
    audit["mskcc_imputed_rows"] = mskcc.loc[imputed_rows.values, "SAMPLE_ID"].tolist()
    audit["mskcc_imputation"] = {
        col: {"n_missing": int(X_mskcc[col].isna().sum()), "fill_value": float(X_train[col].median())}
        for col in FEATURE_ORDER if X_mskcc[col].isna().any()
    }
    for col in FEATURE_ORDER:
        if X_mskcc[col].isna().any():
            X_mskcc[col] = X_mskcc[col].fillna(float(X_train[col].median()))
    X_mskcc_s = pd.DataFrame(scaler.transform(X_mskcc), columns=FEATURE_ORDER)

    # Record the standardized feature distributions: MSKCC samples should be
    # visibly below the TCGA train mean for the raw-scale pathway features —
    # this quantifies the cross-platform shift the frozen scaler inherits.
    audit["z_score_shift"] = {
        cohort: {
            f: {"mean": float(vals[f].mean()), "std": float(vals[f].std())}
            for f in GENOMIC_FEATURES
        }
        for cohort, vals in {"tcga_train": X_train_s, "mskcc": X_mskcc_s}.items()
    }

    # -- 2. OOF fusion weights and threshold on TCGA train --------------------
    logger.info("Estimating OOF fusion weights and threshold on TCGA train...")
    oof = estimate_oof_fusion(
        genomic_model=make_xgb(y_train),
        clinical_model=make_xgb(y_train),
        X_genomic=X_train_s[GENOMIC_FEATURES],
        X_clinical=X_train_s[CLINICAL_FEATURES],
        y=y_train,
    )
    w_gen, w_clin = oof["genomic_weight"], oof["clinical_weight"]
    threshold = oof["threshold"]
    audit["oof"] = {
        "genomic_weight": w_gen,
        "clinical_weight": w_clin,
        "oof_fusion_auc": oof["oof_auc"],
        "threshold": threshold,
    }
    logger.info(
        "OOF fusion: weights=%.2f/%.2f, OOF AUC=%.4f, threshold=%.4f",
        w_gen, w_clin, oof["oof_auc"], threshold,
    )

    # -- 3. Final branch models fitted on all TCGA train rows -----------------
    genomic_model = make_xgb(y_train)
    clinical_model = make_xgb(y_train)
    genomic_model.fit(xgb_safe_frame(X_train_s[GENOMIC_FEATURES]), y_train)
    clinical_model.fit(xgb_safe_frame(X_train_s[CLINICAL_FEATURES]), y_train)

    def _branch_probas(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        g = genomic_model.predict_proba(xgb_safe_frame(X[GENOMIC_FEATURES]))[:, 1]
        c = clinical_model.predict_proba(xgb_safe_frame(X[CLINICAL_FEATURES]))[:, 1]
        return g, c

    def _fused(g: np.ndarray, c: np.ndarray) -> np.ndarray:
        return w_gen * g + w_clin * c

    # -- 4. Internal sanity check on the untouched TCGA test rows -------------
    g_test, c_test = _branch_probas(X_test_s)
    f_test = _fused(g_test, c_test)
    test_metrics = _classification_metrics(y_test, f_test, threshold)
    test_metrics["fusion_auc"] = float(roc_auc_score(y_test, f_test))
    test_metrics["genomic_auc"] = float(roc_auc_score(y_test, g_test))
    test_metrics["clinical_auc"] = float(roc_auc_score(y_test, c_test))
    test_ci, _ = stratified_bootstrap_auc(y_test, f_test)
    test_metrics["fusion_auc_ci95"] = [float(test_ci[0]), float(test_ci[1])]

    tcga_test_preds = pd.DataFrame({
        "ROW_INDEX": tcga.loc[~train_mask, "ROW_INDEX"].values,
        "y": y_test,
        "genomic_proba": g_test,
        "clinical_proba": c_test,
        "fusion_proba": f_test,
        "prediction": (f_test >= threshold).astype(int),
    })

    # -- 5. Frozen application to MSKCC ---------------------------------------
    g_ext, c_ext = _branch_probas(X_mskcc_s)
    f_ext = _fused(g_ext, c_ext)
    y_ext = mskcc["y"].astype(int).values
    external_metrics = _classification_metrics(y_ext, f_ext, threshold)
    external_metrics["fusion_auc"] = float(roc_auc_score(y_ext, f_ext))
    external_metrics["genomic_auc"] = float(roc_auc_score(y_ext, g_ext))
    external_metrics["clinical_auc"] = float(roc_auc_score(y_ext, c_ext))
    ext_ci, _ = stratified_bootstrap_auc(y_ext, f_ext)
    external_metrics["fusion_auc_ci95"] = [float(ext_ci[0]), float(ext_ci[1])]
    # Post-hoc diagnostic ONLY (not a prespecified operating point): the
    # Youden-optimal threshold recomputed ON the external cohort.
    fpr, tpr, thr = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(
        y_ext, f_ext
    )
    posthoc_threshold = float(thr[int(np.argmax(tpr - fpr))])
    external_metrics["posthoc_youden_threshold"] = posthoc_threshold
    external_metrics["posthoc_youden_metrics"] = _classification_metrics(
        y_ext, f_ext, posthoc_threshold
    )

    mskcc_preds = pd.DataFrame({
        "SAMPLE_ID": mskcc["SAMPLE_ID"],
        "PATIENT_ID": mskcc["PATIENT_ID"],
        "y": y_ext,
        "genomic_proba": g_ext,
        "clinical_proba": c_ext,
        "fusion_proba": f_ext,
        "prediction": (f_ext >= threshold).astype(int),
    })

    joblib.dump(genomic_model, config.MODELS_DIR / "transferable_genomic_model.joblib")
    joblib.dump(clinical_model, config.MODELS_DIR / "transferable_clinical_model.joblib")

    results = {
        "step": 3,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transfer_rule": {
            "scaler": "StandardScaler fit on TCGA train rows only, applied unchanged to TCGA test and MSKCC",
            "models": "XGBoost (config defaults + scale_pos_weight), one per branch",
            "weights_source": "estimate_oof_fusion on TCGA train (5-fold)",
            "threshold_source": "Youden on TCGA OOF fusion predictions",
            "external_tuning": "none — frozen pipeline applied to MSKCC",
        },
        "features": {"genomic": GENOMIC_FEATURES, "clinical": CLINICAL_FEATURES},
        "audit": audit,
        "tcga_oof": audit["oof"],
        "tcga_test": test_metrics,
        "mskcc_external": external_metrics,
    }
    return results, mskcc_preds, tcga_test_preds


def main() -> None:
    """Run Step 3, save all artifacts, and print the summary report."""
    results, mskcc_preds, tcga_test_preds = run_transferable_fusion()

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    mskcc_preds.to_csv(MSKCC_PREDS_CSV, index=False)
    tcga_test_preds.to_csv(TCGA_TEST_PREDS_CSV, index=False)
    logger.info("Saved %s, %s, %s", RESULTS_JSON, MSKCC_PREDS_CSV, TCGA_TEST_PREDS_CSV)

    oof, test, ext = results["tcga_oof"], results["tcga_test"], results["mskcc_external"]
    print("=== Step 3: transferable late fusion ===")
    print(f"OOF fusion (TCGA train): AUC={oof['oof_fusion_auc']:.4f}, "
          f"weights={oof['genomic_weight']:.2f}/{oof['clinical_weight']:.2f}, "
          f"threshold={oof['threshold']:.4f}")
    print(f"TCGA test (sanity):      AUC={test['fusion_auc']:.4f} "
          f"CI95=[{test['fusion_auc_ci95'][0]:.3f}, {test['fusion_auc_ci95'][1]:.3f}], "
          f"bal-acc={test['balanced_accuracy']:.3f}, "
          f"sens={test['sensitivity']:.3f}, spec={test['specificity']:.3f}")
    print(f"Branch AUCs (test):      genomic={test['genomic_auc']:.4f}, "
          f"clinical={test['clinical_auc']:.4f}")
    print("=== MSKCC external (frozen) ===")
    print(f"Fusion AUC={ext['fusion_auc']:.4f} "
          f"CI95=[{ext['fusion_auc_ci95'][0]:.3f}, {ext['fusion_auc_ci95'][1]:.3f}]")
    print(f"Branch AUCs: genomic={ext['genomic_auc']:.4f}, clinical={ext['clinical_auc']:.4f}")
    print(f"At transferred threshold {ext['threshold']:.4f}: "
          f"sens={ext['sensitivity']:.3f}, spec={ext['specificity']:.3f}, "
          f"bal-acc={ext['balanced_accuracy']:.3f}, PPV={ext['ppv']:.3f}, NPV={ext['npv']:.3f}")
    print(f"(post-hoc Youden threshold on MSKCC: {ext['posthoc_youden_threshold']:.4f} — diagnostic only, "
          f"sens={ext['posthoc_youden_metrics']['sensitivity']:.3f}, "
          f"spec={ext['posthoc_youden_metrics']['specificity']:.3f})")
    print(f"Imputed MSKCC rows: {results['audit']['mskcc_imputed_rows']}")


if __name__ == "__main__":
    main()
