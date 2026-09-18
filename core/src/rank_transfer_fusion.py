"""
Rank-based genomic transfer for the Late Fusion (Pipeline Step 4).

FROZEN TRANSFER RULE — declared and written down BEFORE any external metric
was computed or inspected for this variant (no MSKCC label or prediction was
looked at while designing it):

    1. Per cohort, independently and label-free: every pathway gene's
       expression column is replaced by its PERCENTILE RANK among that
       cohort's own samples (pandas rank(pct=True), average ties).
    2. Pathway score = mean of the gene ranks in each of the three gene sets
       (PSA, AR signaling, proliferation; 16/16 genes available in both
       cohorts).
    3. Clinical branch is unchanged from Step 3 (the six transferable
       features; clinical values are platform-independent).
    4. StandardScaler is FIT on the TCGA TRAIN rows only and applied
       unchanged to TCGA test and MSKCC — same frozen-scaler invariant as
       Step 3, now operating on rank scores whose marginal distribution is
       comparable across cohorts BY CONSTRUCTION (uniform in both cohorts),
       which removes the raw-counts-versus-log-scale platform shift without
       sharing any cross-cohort statistic and without touching any label.
    5. Branch models, OOF fusion weights, and the Youden threshold follow
       the identical leakage-free recipe as Step 3 (estimate_oof_fusion on
       TCGA train). The frozen pipeline is applied to MSKCC exactly once.

Rationale: Step 3 showed the frozen linear scaler collapses the genomic
branch on MSKCC (AUC 0.500, constant prediction) because raw TCGA counts and
log-scale MSKCC values differ by ~five orders of magnitude. Rank
normalization is the prespecified remedy; it is a single rule, run once, and
its result is reported honestly whether it improves the genomic branch or
not.

Outputs
-------
- core/data/processed/tcga_rank_features.csv            (rank pathway scores)
- core/data/external/mskcc_rank_features.csv            (rank pathway scores)
- core/outputs/tables/rank_transfer_fusion_results.json
- core/outputs/tables/mskcc_rank_transfer_predictions.csv
- core/outputs/tables/tcga_rank_transfer_test_predictions.csv
- core/outputs/models/rank_transfer_{scaler,genomic_model,clinical_model}.joblib

Run from the ``core`` directory:

    python -m src.rank_transfer_fusion
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import config
from src.fusion.late_fusion import estimate_oof_fusion
from src.io import logger
from src.models import make_xgb, xgb_safe_frame
from src.transferable_features import FEATURE_ORDER, PATHWAY_ENTREZ
from src.transferable_fusion import TCGA_CSV, MSKCC_CSV
from src.mskcc_cohort import EXPRESSION_FILE, MSKCC_DIR
from src.transferable_fusion import (
    CLINICAL_FEATURES,
    GENOMIC_FEATURES,
    _classification_metrics,
)
from src.evaluation import stratified_bootstrap_auc

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
TCGA_RANK_CSV = config.PROCESSED_DIR / "tcga_rank_features.csv"
MSKCC_RANK_CSV = config.DATA_DIR / "external" / "mskcc_rank_features.csv"
RESULTS_JSON = config.TABLES_DIR / "rank_transfer_fusion_results.json"
MSKCC_PREDS_CSV = config.TABLES_DIR / "mskcc_rank_transfer_predictions.csv"
TCGA_TEST_PREDS_CSV = config.TABLES_DIR / "tcga_rank_transfer_test_predictions.csv"

X_FINAL_CSV = config.PROCESSED_DIR / "X_features_final.csv"


# ---------------------------------------------------------------------------
# Rank feature construction
# ---------------------------------------------------------------------------
def _rank_scores_from_matrix(matrix: pd.DataFrame, label: str) -> pd.DataFrame:
    """Convert pathway gene columns to within-cohort percentile ranks and average."""
    out = pd.DataFrame(index=matrix.index)
    for score_name in GENOMIC_FEATURES:
        from src.features_config import AR_GENES, PROLIF_GENES, PSA_GENES
        genes = {"PSA_Pathway_Score": PSA_GENES, "AR_Signaling_Score": AR_GENES,
                 "Proliferation_Score": PROLIF_GENES}[score_name]
        available = [g for g in genes if g in matrix.columns]
        if len(available) < 3:
            raise RuntimeError(f"{label}: {score_name} has {len(available)} genes")
        ranked = matrix[available].rank(pct=True)
        out[score_name] = ranked.mean(axis=1)
        logger.info(
            "%s %s: %d genes ranked (score mean=%.4f, std=%.4f)",
            label, score_name, len(available),
            out[score_name].mean(), out[score_name].std(),
        )
    return out


def build_tcga_rank_features() -> pd.DataFrame:
    """Rank-normalize TCGA pathway genes and attach the validated split labels."""
    X = pd.read_csv(X_FINAL_CSV)
    step2 = pd.read_csv(TCGA_CSV)
    if len(X) != len(step2):
        raise RuntimeError("X_features_final and Step-2 table row counts differ")
    y_now = pd.read_csv(config.PROCESSED_DIR / "y_target_final.csv").iloc[:, 0].values
    if not (y_now == step2["y"].values).all():
        raise RuntimeError("y_target_final no longer matches the Step-2 labels")

    scores = _rank_scores_from_matrix(X, "TCGA")
    out = pd.DataFrame({
        "ROW_INDEX": step2["ROW_INDEX"].values,
        "split": step2["split"].values,
        "y": step2["y"].values,
        **{c: scores[c].values for c in GENOMIC_FEATURES},
    })
    return out


def build_mskcc_rank_features() -> pd.DataFrame:
    """Rank-normalize MSKCC pathway genes for the clean cohort."""
    mskcc = pd.read_csv(MSKCC_CSV)
    expression = pd.read_csv(MSKCC_DIR / EXPRESSION_FILE, sep="\t")
    expression["Entrez_Gene_Id"] = expression["Entrez_Gene_Id"].astype(str)
    rows = expression[expression["Entrez_Gene_Id"].isin(PATHWAY_ENTREZ.values())]
    found = {sym: ent for sym, ent in PATHWAY_ENTREZ.items() if ent in set(rows["Entrez_Gene_Id"])}
    if len(found) != 16:
        raise RuntimeError(f"Expected 16/16 pathway genes, found {len(found)}")
    matrix = rows.set_index("Entrez_Gene_Id").loc[list(found.values())].T
    matrix.columns = list(found.keys())
    matrix = matrix.reindex(mskcc["SAMPLE_ID"].astype(str))
    if matrix.isna().any().any():
        raise RuntimeError("Cohort samples missing from expression matrix")

    scores = _rank_scores_from_matrix(matrix, "MSKCC")
    out = pd.DataFrame({
        "SAMPLE_ID": mskcc["SAMPLE_ID"].values,
        "PATIENT_ID": mskcc["PATIENT_ID"].values,
        "y": mskcc["y"].values,
        **{c: scores[c].values for c in GENOMIC_FEATURES},
    })
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_rank_transfer_fusion(
    tcga_ranks: pd.DataFrame, mskcc_ranks: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run the frozen rank-based fusion pipeline once and report honestly."""
    audit: dict[str, Any] = {}
    train_mask = (tcga_ranks["split"] == "train").values

    Xg_tcga = tcga_ranks[GENOMIC_FEATURES].astype(float)
    y_tcga = tcga_ranks["y"].astype(int).values

    Xg_train = Xg_tcga[train_mask].reset_index(drop=True)
    y_train = y_tcga[train_mask]

    # -- Frozen scaler on TCGA train rank scores (genomic) --------------------
    scaler = StandardScaler().fit(Xg_train)
    joblib.dump(scaler, config.MODELS_DIR / "rank_transfer_scaler.joblib")

    def _scale_genomic(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(scaler.transform(df), columns=GENOMIC_FEATURES)

    Xg_train_s = _scale_genomic(Xg_train)
    Xg_test_s = _scale_genomic(Xg_tcga[~train_mask].reset_index(drop=True))
    Xg_mskcc_s = _scale_genomic(mskcc_ranks[GENOMIC_FEATURES].astype(float))

    audit["rank_score_shift_check"] = {
        "mskcc_z_mean": {c: float(Xg_mskcc_s[c].mean()) for c in GENOMIC_FEATURES},
        "mskcc_z_std": {c: float(Xg_mskcc_s[c].std()) for c in GENOMIC_FEATURES},
        "note": "rank scores should keep MSKCC near the TCGA scale by construction",
    }

    # -- Clinical branch: identical to Step 3 (raw transferable features) ----
    clinical_step3 = pd.read_csv(TCGA_CSV)[CLINICAL_FEATURES].astype(float)
    mskcc_step3 = pd.read_csv(MSKCC_CSV)[CLINICAL_FEATURES].astype(float)
    clin_train_median = clinical_step3[train_mask].median()
    audit["clinical_imputation"] = {
        col: {"n_missing": int(mskcc_step3[col].isna().sum()), "fill": float(v)}
        for col, v in clin_train_median.items() if mskcc_step3[col].isna().any()
    }
    for col in CLINICAL_FEATURES:
        mskcc_step3[col] = mskcc_step3[col].fillna(float(clin_train_median[col]))
    clin_scaler = StandardScaler().fit(clinical_step3[train_mask].reset_index(drop=True))
    joblib.dump(clin_scaler, config.MODELS_DIR / "rank_transfer_clinical_scaler.joblib")
    Xc_train_s = pd.DataFrame(clin_scaler.transform(clinical_step3[train_mask].reset_index(drop=True)), columns=CLINICAL_FEATURES)
    Xc_test_s = pd.DataFrame(clin_scaler.transform(clinical_step3[~train_mask].reset_index(drop=True)), columns=CLINICAL_FEATURES)
    Xc_mskcc_s = pd.DataFrame(clin_scaler.transform(mskcc_step3), columns=CLINICAL_FEATURES)

    # -- OOF fusion weights and threshold (TCGA train only) -------------------
    oof = estimate_oof_fusion(
        genomic_model=make_xgb(y_train),
        clinical_model=make_xgb(y_train),
        X_genomic=Xg_train_s,
        X_clinical=Xc_train_s,
        y=y_train,
    )
    w_gen, w_clin, threshold = oof["genomic_weight"], oof["clinical_weight"], oof["threshold"]
    audit["oof"] = {
        "genomic_weight": w_gen, "clinical_weight": w_clin,
        "oof_fusion_auc": oof["oof_auc"], "threshold": threshold,
    }

    # -- Final frozen models ---------------------------------------------------
    genomic_model = make_xgb(y_train)
    clinical_model = make_xgb(y_train)
    genomic_model.fit(xgb_safe_frame(Xg_train_s), y_train)
    clinical_model.fit(xgb_safe_frame(Xc_train_s), y_train)
    joblib.dump(genomic_model, config.MODELS_DIR / "rank_transfer_genomic_model.joblib")
    joblib.dump(clinical_model, config.MODELS_DIR / "rank_transfer_clinical_model.joblib")

    def _evaluate(Xg_s: pd.DataFrame, Xc_s: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
        g = genomic_model.predict_proba(xgb_safe_frame(Xg_s))[:, 1]
        c = clinical_model.predict_proba(xgb_safe_frame(Xc_s))[:, 1]
        f = w_gen * g + w_clin * c
        metrics = _classification_metrics(y, f, threshold)
        metrics["fusion_auc"] = float(roc_auc_score(y, f))
        metrics["genomic_auc"] = float(roc_auc_score(y, g))
        metrics["clinical_auc"] = float(roc_auc_score(y, c))
        ci, _ = stratified_bootstrap_auc(y, f)
        metrics["fusion_auc_ci95"] = [float(ci[0]), float(ci[1])]
        return metrics, g, c, f

    test_metrics, g_test, c_test, f_test = _evaluate(
        Xg_test_s, Xc_test_s, y_tcga[~train_mask]
    )
    ext_metrics, g_ext, c_ext, f_ext = _evaluate(
        Xg_mskcc_s, Xc_mskcc_s, mskcc_ranks["y"].astype(int).values
    )

    # Post-hoc Youden on MSKCC: diagnostic only, never a prespecified point.
    fpr, tpr, thr = roc_curve(mskcc_ranks["y"].astype(int).values, f_ext)
    posthoc_threshold = float(thr[int(np.argmax(tpr - fpr))])
    ext_metrics["posthoc_youden_threshold"] = posthoc_threshold
    ext_metrics["posthoc_youden_metrics"] = _classification_metrics(
        mskcc_ranks["y"].astype(int).values, f_ext, posthoc_threshold
    )

    tcga_test_preds = pd.DataFrame({
        "ROW_INDEX": tcga_ranks.loc[~train_mask, "ROW_INDEX"].values,
        "y": y_tcga[~train_mask],
        "genomic_proba": g_test, "clinical_proba": c_test,
        "fusion_proba": f_test,
        "prediction": (f_test >= threshold).astype(int),
    })
    mskcc_preds = pd.DataFrame({
        "SAMPLE_ID": mskcc_ranks["SAMPLE_ID"].values,
        "PATIENT_ID": mskcc_ranks["PATIENT_ID"].values,
        "y": mskcc_ranks["y"].values,
        "genomic_proba": g_ext, "clinical_proba": c_ext,
        "fusion_proba": f_ext,
        "prediction": (f_ext >= threshold).astype(int),
    })

    results = {
        "step": 4,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transfer_rule": (
            "within-cohort per-gene percentile ranks -> pathway mean -> "
            "StandardScaler fit on TCGA train only -> frozen application; "
            "clinical branch unchanged from Step 3; declared before any "
            "external metric was computed; executed exactly once"
        ),
        "features": {"genomic": GENOMIC_FEATURES, "clinical": CLINICAL_FEATURES},
        "audit": audit,
        "tcga_oof": audit["oof"],
        "tcga_test": test_metrics,
        "mskcc_external": ext_metrics,
    }
    return results, mskcc_preds, tcga_test_preds


def main() -> None:
    """Build rank features, run the frozen pipeline once, save, and report."""
    tcga_ranks = build_tcga_rank_features()
    mskcc_ranks = build_mskcc_rank_features()
    results, mskcc_preds, tcga_test_preds = run_rank_transfer_fusion(tcga_ranks, mskcc_ranks)

    tcga_ranks.to_csv(TCGA_RANK_CSV, index=False)
    mskcc_ranks.to_csv(MSKCC_RANK_CSV, index=False)
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    mskcc_preds.to_csv(MSKCC_PREDS_CSV, index=False)
    tcga_test_preds.to_csv(TCGA_TEST_PREDS_CSV, index=False)
    logger.info(
        "Saved %s, %s, %s", RESULTS_JSON, MSKCC_PREDS_CSV, TCGA_TEST_PREDS_CSV
    )

    oof, test, ext = results["tcga_oof"], results["tcga_test"], results["mskcc_external"]
    shift = results["audit"]["rank_score_shift_check"]["mskcc_z_mean"]
    print("=== Step 4: rank-based genomic transfer (frozen rule, single run) ===")
    print(f"OOF fusion: AUC={oof['oof_fusion_auc']:.4f}, weights={oof['genomic_weight']:.2f}/{oof['clinical_weight']:.2f}, threshold={oof['threshold']:.4f}")
    print(f"TCGA test:  AUC={test['fusion_auc']:.4f} CI95=[{test['fusion_auc_ci95'][0]:.3f}, {test['fusion_auc_ci95'][1]:.3f}], genomic={test['genomic_auc']:.4f}, clinical={test['clinical_auc']:.4f}")
    print(f"MSKCC ext:  AUC={ext['fusion_auc']:.4f} CI95=[{ext['fusion_auc_ci95'][0]:.3f}, {ext['fusion_auc_ci95'][1]:.3f}], genomic={ext['genomic_auc']:.4f}, clinical={ext['clinical_auc']:.4f}")
    print(f"  at threshold {ext['threshold']:.4f}: sens={ext['sensitivity']:.3f}, spec={ext['specificity']:.3f}, bal-acc={ext['balanced_accuracy']:.3f}")
    print(f"  MSKCC z-means (rank scores): {json.dumps({k: round(v, 3) for k, v in shift.items()})}")
    print(f"  post-hoc Youden: {ext['posthoc_youden_threshold']:.4f} (sens={ext['posthoc_youden_metrics']['sensitivity']:.3f}, spec={ext['posthoc_youden_metrics']['specificity']:.3f}) — diagnostic only")


if __name__ == "__main__":
    main()
