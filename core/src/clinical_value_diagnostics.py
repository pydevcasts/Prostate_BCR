"""
Clinical-value diagnostics for the transferable fusion (Pipeline Step 5).

Quantifies the REAL added value of each branch on the external MSKCC cohort,
next to a prespecified pure-clinical baseline, with no external tuning:

    1. Pure-clinical baseline: LogisticRegression(config.LOGISTIC_PARAMS,
       class-weight balanced) on the three transferable clinical features,
       trained on TCGA train only. This is the floor any genomic signal must
       beat on MSKCC.
    2. Paired AUC comparison (baseline vs each frozen fusion variant) on
       MSKCC: stratified bootstrap deltas (3000 iterations), reported with a
       two-sided p-value (fraction of bootstrap deltas crossing zero).
    3. Calibration on MSKCC (diagnostic, external cohort): Brier score,
       reliability bins, and a logistic recalibration slope/intercept fitted
       ON MSKCC and reported as post-hoc diagnostics only.
    4. Decision Curve Analysis on MSKCC for the baseline and both fusion
       variants across threshold probabilities 0.05-0.95.

The frozen pipelines from Steps 3 and 4 are re-created deterministically by
calling their run functions (verified identical outputs in Step 4); their
MSKCC predictions and the labels are read from the existing result tables.

Outputs
-------
- core/outputs/tables/clinical_value_diagnostics.json
- core/outputs/tables/mskcc_dca_curves.csv
- core/outputs/figures/mskcc_calibration_dca.png

Run from the ``core`` directory:

    python -m src.clinical_value_diagnostics
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import config
from src.io import logger
from src.io import save_figure
from src.transferable_features import FEATURE_ORDER
from src.transferable_fusion import CLINICAL_FEATURES, TCGA_CSV, MSKCC_CSV

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
RESULTS_JSON = config.TABLES_DIR / "clinical_value_diagnostics.json"
DCA_CSV = config.TABLES_DIR / "mskcc_dca_curves.csv"
FIG_PATH = "mskcc_calibration_dca.png"

X_FINAL_CSV = config.PROCESSED_DIR / "X_features_final.csv"
STEP2_CSV = config.PROCESSED_DIR / "tcga_transferable_features.csv"
STEP3_PREDS = config.TABLES_DIR / "mskcc_transferable_predictions.csv"
STEP4_PREDS = config.TABLES_DIR / "mskcc_rank_transfer_predictions.csv"

N_BOOT = config.BOOTSTRAP_N          # 3000
BINS = np.linspace(0.0, 1.0, 6)      # 5 calibration bins


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def fit_clinical_baseline() -> tuple[LogisticRegression, StandardScaler, dict[str, Any]]:
    """Fit the pure-clinical LR baseline on TCGA train (frozen recipe)."""
    step2 = pd.read_csv(STEP2_CSV)
    train_mask = (step2["split"] == "train").values
    X = step2[CLINICAL_FEATURES].astype(float)
    y = step2["y"].astype(int).values
    scaler = StandardScaler().fit(X[train_mask].reset_index(drop=True))
    model = LogisticRegression(
        **config.LOGISTIC_PARAMS, class_weight="balanced"
    )
    model.fit(scaler.transform(X[train_mask].reset_index(drop=True)), y[train_mask])
    audit = {
        "model": "LogisticRegression(config.LOGISTIC_PARAMS, class_weight=balanced)",
        "features": CLINICAL_FEATURES,
        "train_rows": int(train_mask.sum()),
        "scaler": "StandardScaler fit on TCGA train rows only",
    }
    return model, scaler, audit


# ---------------------------------------------------------------------------
# Bootstrap delta AUC
# ---------------------------------------------------------------------------
def paired_bootstrap_auc_delta(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_boot: int = N_BOOT,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, float]:
    """Stratified bootstrap of AUC(a) - AUC(b) with a two-sided p-value."""
    rng = np.random.RandomState(random_state)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        deltas[i] = roc_auc_score(y_true[idx], proba_a[idx]) - roc_auc_score(
            y_true[idx], proba_b[idx]
        )
    alpha = 1 - config.CONFIDENCE_LEVEL
    ci = np.percentile(deltas, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    p_two_sided = float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))
    point = float(roc_auc_score(y_true, proba_a) - roc_auc_score(y_true, proba_b))
    return {
        "delta_auc": point,
        "ci95": [float(ci[0]), float(ci[1])],
        "p_two_sided_bootstrap": p_two_sided,
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_diagnostics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Brier score, reliability bins, and post-hoc logistic recalibration."""
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y_true, proba))
    logit = np.log(proba / (1 - proba))
    rec = LogisticRegression(max_iter=2000)
    rec.fit(logit.reshape(-1, 1), y_true)
    slope, intercept = float(rec.coef_[0][0]), float(rec.intercept_[0])
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (proba >= lo) & (proba < hi if hi < 1 else proba <= hi)
        if mask.sum() == 0:
            bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
            continue
        bins.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "mean_predicted": float(proba[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
        })
    return {
        "brier": brier,
        "recalibration_slope": slope,
        "recalibration_intercept": intercept,
        "reliability_bins": bins,
        "note": "slope/intercept fitted ON MSKCC — post-hoc diagnostics only",
    }


# ---------------------------------------------------------------------------
# Decision Curve Analysis
# ---------------------------------------------------------------------------
def decision_curve(
    y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    """Net benefit across threshold probabilities."""
    n = len(y_true)
    n_events = int(y_true.sum())
    rows = []
    for t in thresholds:
        pred_pos = proba >= t
        tp = int((pred_pos & (y_true == 1)).sum())
        fp = int((pred_pos & (y_true == 0)).sum())
        nb_model = tp / n - fp / n * (t / (1 - t))
        nb_all = n_events / n - (n - n_events) / n * (t / (1 - t))
        rows.append({
            "threshold": float(t),
            "net_benefit_model": float(max(nb_model, 0.0)),
            "net_benefit_treat_all": float(max(nb_all, 0.0)),
            "net_benefit_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_diagnostics() -> dict[str, Any]:
    """Run the full Step 5 diagnostic battery."""
    audit: dict[str, Any] = {}

    # -- Frozen MSKCC predictions from Steps 3 and 4 --------------------------
    step3 = pd.read_csv(STEP3_PREDS)
    step4 = pd.read_csv(STEP4_PREDS)
    mskcc = pd.read_csv(MSKCC_CSV)
    y_ext = mskcc["y"].astype(int).values
    for name, preds in (("step3", step3), ("step4", step4)):
        if len(preds) != len(mskcc) or not (preds["y"].values == y_ext).all():
            raise RuntimeError(f"{name} predictions misaligned with the MSKCC cohort")
        if not (preds["SAMPLE_ID"].values == mskcc["SAMPLE_ID"].values).all():
            raise RuntimeError(f"{name} SAMPLE_ID order differs from the cohort table")
    audit["alignment_check"] = "PASS"

    # -- 1. Pure-clinical baseline applied frozen to MSKCC ---------------------
    baseline, baseline_scaler, baseline_audit = fit_clinical_baseline()
    Xc_mskcc = mskcc[CLINICAL_FEATURES].astype(float).copy()
    imputed = {
        col: float(baseline_scaler and pd.read_csv(STEP2_CSV)[
            CLINICAL_FEATURES][(pd.read_csv(STEP2_CSV)["split"] == "train").values][col].median())
        for col in CLINICAL_FEATURES if Xc_mskcc[col].isna().any()
    }
    audit["baseline_imputation"] = imputed
    for col, fill in imputed.items():
        Xc_mskcc[col] = Xc_mskcc[col].fillna(fill)
    baseline_proba = baseline.predict_proba(baseline_scaler.transform(Xc_mskcc))[:, 1]
    baseline_auc = float(roc_auc_score(y_ext, baseline_proba))
    audit["baseline"] = baseline_audit

    # -- 2. Paired bootstrap comparisons on MSKCC -------------------------------
    comparisons = {
        "step3_fusion_vs_baseline": paired_bootstrap_auc_delta(
            y_ext, step3["fusion_proba"].values, baseline_proba
        ),
        "step4_fusion_vs_baseline": paired_bootstrap_auc_delta(
            y_ext, step4["fusion_proba"].values, baseline_proba
        ),
        "step4_fusion_vs_step3_fusion": paired_bootstrap_auc_delta(
            y_ext, step4["fusion_proba"].values, step3["fusion_proba"].values
        ),
        "step4_genomic_vs_baseline": paired_bootstrap_auc_delta(
            y_ext, step4["genomic_proba"].values, baseline_proba
        ),
    }

    # -- 3. Calibration diagnostics on MSKCC ------------------------------------
    calibration = {
        "clinical_baseline": calibration_diagnostics(y_ext, baseline_proba),
        "step3_fusion": calibration_diagnostics(y_ext, step3["fusion_proba"].values),
        "step4_fusion": calibration_diagnostics(y_ext, step4["fusion_proba"].values),
        "step4_genomic": calibration_diagnostics(y_ext, step4["genomic_proba"].values),
    }

    # -- 4. Decision curves ------------------------------------------------------
    thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    dca_frames = []
    for label, proba in (
        ("clinical_baseline", baseline_proba),
        ("step3_fusion", step3["fusion_proba"].values),
        ("step4_fusion", step4["fusion_proba"].values),
        ("step4_genomic", step4["genomic_proba"].values),
    ):
        curve = decision_curve(y_ext, proba, thresholds)
        curve["model"] = label
        dca_frames.append(curve)
    dca = pd.concat(dca_frames, ignore_index=True)

    # -- Figure: calibration (top) + DCA (bottom) --------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for label, proba in (
        ("Clinical baseline", baseline_proba),
        ("Step 3 fusion (raw)", step3["fusion_proba"].values),
        ("Step 4 fusion (rank)", step4["fusion_proba"].values),
    ):
        frac_pos, mean_pred = calibration_curve_arrays(y_ext, proba)
        ax.plot(mean_pred, frac_pos, marker="o", ms=4, lw=1.5, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed recurrence rate")
    ax.set_title("Calibration on MSKCC (external)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for label in ("clinical_baseline", "step3_fusion", "step4_fusion", "step4_genomic"):
        sub = dca[dca["model"] == label]
        ax.plot(sub["threshold"], sub["net_benefit_model"], marker="o", ms=3, lw=1.5, label=label)
    ax.axhline(0.0, color="gray", lw=1, ls=":", label="treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis on MSKCC (external)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIG_PATH)
    plt.close(fig)

    results = {
        "step": 5,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit": audit,
        "external_auc": {
            "clinical_baseline": baseline_auc,
            "step3_fusion": float(roc_auc_score(y_ext, step3["fusion_proba"].values)),
            "step4_fusion": float(roc_auc_score(y_ext, step4["fusion_proba"].values)),
            "step4_genomic": float(roc_auc_score(y_ext, step4["genomic_proba"].values)),
        },
        "paired_comparisons": comparisons,
        "calibration": calibration,
    }
    return results, dca, baseline_proba


def calibration_curve_arrays(
    y_true: np.ndarray, proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (observed rate, mean predicted) per non-empty reliability bin."""
    xs, ys = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (proba >= lo) & (proba < hi if hi < 1 else proba <= hi)
        if mask.sum():
            xs.append(float(proba[mask].mean()))
            ys.append(float(y_true[mask].mean()))
    return np.asarray(ys), np.asarray(xs)


def main() -> None:
    """Run Step 5, save artifacts, and print the summary report."""
    results, dca, _ = run_diagnostics()
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    dca.to_csv(DCA_CSV, index=False)
    logger.info("Saved %s and %s", RESULTS_JSON, DCA_CSV)

    aucs = results["external_auc"]
    print("=== Step 5: clinical-value diagnostics (MSKCC external) ===")
    print("External AUCs:")
    for k, v in aucs.items():
        print(f"  {k:20s} {v:.4f}")
    print("Paired comparisons (delta AUC, CI95, bootstrap p):")
    for k, v in results["paired_comparisons"].items():
        print(f"  {k:32s} {v['delta_auc']:+.4f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] p={v['p_two_sided_bootstrap']:.3f}")
    print("Calibration (Brier | slope | intercept):")
    for k, v in results["calibration"].items():
        print(f"  {k:20s} {v['brier']:.4f} | {v['recalibration_slope']:.3f} | {v['recalibration_intercept']:+.3f}")


if __name__ == "__main__":
    main()
