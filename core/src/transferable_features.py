"""
Transferable feature construction (Pipeline Step 2).

Builds the six transferable features on BOTH cohorts so a fusion model can be
retrained on TCGA (Step 3) and applied to MSKCC (Step 4):

    Genomic branch  : PSA_Pathway_Score, AR_Signaling_Score, Proliferation_Score
                      (mean normalized expression of the pathway gene sets in
                      src.features_config; raw unscaled values on both sides so
                      both cohorts carry the same semantic quantity)
    Clinical branch : Gleason_Total, High_Risk_Gleason (Gleason pattern primary
                      + secondary), T_Stage_Risk (T3/T4 indicator; TCGA from the
                      one-hot T3a/T3b/T4 columns, MSKCC from PATH_T_STAGE)

Margin_x_LymphNode is intentionally NOT built: MSKCC has no margin or lymph
node columns (contract decision).

TCGA reconstruction: the canonical pipeline (notebook 03) split
``X_features_final.csv`` (the merged raw clinical + expression table, 429 ×
19019, unscaled) with train_test_split(test_size=0.20, stratify=y,
random_state=42). The split is deterministic, so this module reproduces it and
HARD-VALIDATES the result against the saved y_train / y_test / engineered
artifacts before writing anything. Any mismatch aborts with an error — no
silent approximations.

Scale note (recorded in the audit for Step 3): TCGA expression in
X_features_final is on a raw-count-like scale, while the MSKCC Agilent matrix
is log-scale. The RAW pathway scores written here are therefore NOT directly
comparable across cohorts; Step 3 must standardize within cohort (or fit the
scaler on TCGA train and choose a defensible transfer rule) before training.

Outputs
-------
- core/data/processed/tcga_transferable_features.csv        (429 rows)
- core/outputs/tables/tcga_transferable_features_summary.json
- core/data/external/mskcc_transferable_features.csv        (131 rows)
- core/outputs/tables/mskcc_transferable_features_summary.json

Run from the ``core`` directory:

    python -m src.transferable_features
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.features_config import (
    AR_GENES,
    MIN_GENES_FOR_PATHWAY,
    PROLIF_GENES,
    PSA_GENES,
)
from src.io import logger
from src.mskcc_cohort import MSKCC_COHORT_CSV, MSKCC_DIR, EXPRESSION_FILE

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
TCGA_OUT_CSV = config.PROCESSED_DIR / "tcga_transferable_features.csv"
TCGA_SUMMARY_JSON = config.TABLES_DIR / "tcga_transferable_features_summary.json"
MSKCC_OUT_CSV = config.DATA_DIR / "external" / "mskcc_transferable_features.csv"
MSKCC_SUMMARY_JSON = config.TABLES_DIR / "mskcc_transferable_features_summary.json"

PATHWAY_SETS: dict[str, tuple[str, ...]] = {
    "PSA_Pathway_Score": PSA_GENES,
    "AR_Signaling_Score": AR_GENES,
    "Proliferation_Score": PROLIF_GENES,
}
FEATURE_ORDER = [
    "PSA_Pathway_Score",
    "AR_Signaling_Score",
    "Proliferation_Score",
    "Gleason_Total",
    "High_Risk_Gleason",
    "T_Stage_Risk",
]

# Verified static Entrez-ID mapping for the 16 pathway genes (each entry was
# confirmed present in the MSKCC Agilent matrix on 2026-09-18).
PATHWAY_ENTREZ: dict[str, str] = {
    "KLK3": "354", "KLK2": "5655", "ACPP": "55", "TMPRSS2": "7113",
    "AR": "367", "NKX3-1": "4824", "STEAP2": "261729", "FKBP5": "2289",
    "CAMKK2": "54890", "MKI67": "4288", "TOP2A": "7153", "CCNB1": "891",
    "CCNE1": "898", "CDK1": "983", "AURKA": "6790", "BIRC5": "332",
}

# MSKCC T-stage mapper (contract): T3-family and T4 map to risk 1.
MSKCC_T3_T4 = {"T3", "T3A", "T3B", "T3C", "T4"}


def _pathway_scores_from_matrix(
    matrix: pd.DataFrame,
    cohort_label: str,
    audit: dict[str, Any],
) -> pd.DataFrame:
    """Compute the three pathway scores as row means of available pathway genes."""
    scores = pd.DataFrame(index=matrix.index)
    for score_name, genes in PATHWAY_SETS.items():
        available = [g for g in genes if g in matrix.columns]
        audit.setdefault("pathway_gene_availability", {})[score_name] = {
            "available": len(available),
            "required": len(genes),
        }
        if len(available) < MIN_GENES_FOR_PATHWAY:
            raise RuntimeError(
                f"{cohort_label}: {score_name} has only {len(available)}/{len(genes)} "
                f"genes (minimum {MIN_GENES_FOR_PATHWAY})"
            )
        scores[score_name] = matrix[available].mean(axis=1)
    return scores


# ---------------------------------------------------------------------------
# TCGA
# ---------------------------------------------------------------------------
def build_tcga_transferable_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reproduce the canonical TCGA split and build the feature table."""
    audit: dict[str, Any] = {}

    # -- 1. Load the canonical merged table (same input as notebook 03) -------
    X = pd.read_csv(config.PROCESSED_DIR / "X_features_final.csv")
    y = pd.read_csv(config.PROCESSED_DIR / "y_target_final.csv").iloc[:, 0]
    if len(X) != len(y):
        raise RuntimeError(
            f"X_features_final ({len(X)}) and y_target_final ({len(y)}) misaligned"
        )
    y = y.reset_index(drop=True)
    audit["merged_samples"] = len(X)
    logger.info("Canonical merged table: %d samples × %d features", *X.shape)

    # -- 2. HARD GATE: reproduce the split indices ----------------------------
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    y_train_saved = pd.read_csv(config.PROCESSED_DIR / "y_train.csv")[
        "Biochemical_Recurrence_Code"
    ].reset_index(drop=True)
    if not y.iloc[train_idx].reset_index(drop=True).equals(y_train_saved):
        raise RuntimeError("Split reproduction failed: train targets do not match")
    y_test_saved = pd.read_csv(config.PROCESSED_DIR / "y_test.csv")[
        "Biochemical_Recurrence_Code"
    ].reset_index(drop=True)
    if not y.iloc[test_idx].reset_index(drop=True).equals(y_test_saved):
        raise RuntimeError("Split reproduction failed: test targets do not match")
    audit["split_reproduction_check"] = "PASS"
    audit["n_train"] = len(train_idx)
    audit["n_test"] = len(test_idx)

    # -- 3. Pathway scores from the UNSCALED expression columns ---------------
    scores = _pathway_scores_from_matrix(X, "TCGA", audit)

    # -- 4. Clinical transferable features ------------------------------------
    patterns = X[["Gleason pattern primary", "Gleason pattern secondary"]].apply(
        pd.to_numeric, errors="coerce"
    )
    gleason_total = patterns["Gleason pattern primary"] + patterns["Gleason pattern secondary"]
    high_risk = ((patterns["Gleason pattern primary"] >= 4) | (patterns["Gleason pattern secondary"] >= 4)).astype(int)

    t_cols = [
        c for c in X.columns
        if "Tumor Stage Code_T3" in c or "Tumor Stage Code_T4" in c
    ]
    t_risk = X[t_cols].sum(axis=1).astype(int) if t_cols else pd.Series(0, index=X.index)

    # -- 5. HARD GATE: engineered-column agreement on the train rows ----------
    # The historical artifact encodes T_Stage_Risk as the SUM OF STANDARDIZED
    # T3a/T3b/T4 one-hot columns (engineering ran after scaling in the old
    # pipeline), so it has 4 distinct real values instead of {0, 1}. The
    # transferable table intentionally stores the RAW 0/1 risk indicator,
    # which is the quantity that maps onto the MSKCC PATH_T_STAGE mapper.
    # The gate therefore checks BINARY agreement: every train row the
    # artifact marks as risk-free must be 0 here, and every risk-carrying
    # row must be 1 (i.e. the artifact's zero-group is exactly our zeros).
    engineered = pd.read_csv(config.PROCESSED_DIR / "X_train_clinical_engineered.csv")
    eng = engineered["T_Stage_Risk"].values
    mine = t_risk.iloc[train_idx].values
    zero_group = eng[mine == 0]
    one_group = eng[mine == 1]
    if not (len(set(zero_group)) == 1 and not (set(zero_group) & set(one_group))):
        raise RuntimeError(
            "T_Stage_Risk binary disagreement with X_train_clinical_engineered"
        )
    audit["t_stage_risk_check"] = (
        "PASS (binary agreement; artifact stores scaled one-hot sums, "
        "transferable table stores raw 0/1)"
    )

    selected = pd.read_csv(config.PROCESSED_DIR / "X_train_genomic_selected.csv")
    scaled = pd.read_csv(config.PROCESSED_DIR / "X_train_preprocessed.csv")
    scaled_psa = scaled[[g for g in PSA_GENES if g in scaled.columns]].mean(axis=1)
    if not np.allclose(scaled_psa.values, selected["PSA_Pathway_Score"].values, atol=1e-6):
        raise RuntimeError("Scaled PSA score recompute does not match X_train_genomic_selected")
    audit["gene_identity_check"] = "PASS (scaled PSA recompute matches selected artifact)"

    # Internal consistency: primary + secondary should equal the reported total
    reported = pd.to_numeric(
        X.get("Radical Prostatectomy Gleason Score for Prostate Cancer"), errors="coerce"
    )
    mismatches = int((reported.notna() & (gleason_total != reported)).sum())
    audit["gleason_total_mismatches_vs_reported"] = mismatches
    if mismatches:
        logger.warning(
            "%d samples where primary+secondary != reported total Gleason", mismatches
        )

    # -- 6. Assemble -----------------------------------------------------------
    out = scores.copy()
    out["Gleason_Total"] = gleason_total.values
    out["High_Risk_Gleason"] = high_risk.values
    out["T_Stage_Risk"] = t_risk.values
    out = out[FEATURE_ORDER]
    out.insert(0, "y", y.values)
    split = np.full(len(out), "test", dtype=object)
    split[train_idx] = "train"
    out.insert(0, "split", split)
    out.insert(0, "ROW_INDEX", np.arange(len(out)))

    audit["gleason_total_missing"] = int(out["Gleason_Total"].isna().sum())
    audit["gleason_total_range"] = [
        float(np.nanmin(out["Gleason_Total"])), float(np.nanmax(out["Gleason_Total"])),
    ]
    audit["feature_summary"] = {
        f: {"mean": float(out[f].mean()), "std": float(out[f].std())}
        for f in FEATURE_ORDER
    }
    audit["n_events"] = int(out["y"].sum())
    return out, audit


# ---------------------------------------------------------------------------
# MSKCC
# ---------------------------------------------------------------------------
def build_mskcc_transferable_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the transferable feature table for the clean MSKCC cohort."""
    audit: dict[str, Any] = {}
    cohort = pd.read_csv(MSKCC_COHORT_CSV)
    audit["cohort_samples"] = len(cohort)

    expression = pd.read_csv(MSKCC_DIR / EXPRESSION_FILE, sep="\t")
    expression["Entrez_Gene_Id"] = expression["Entrez_Gene_Id"].astype(str)
    gene_rows = expression[expression["Entrez_Gene_Id"].isin(PATHWAY_ENTREZ.values())].set_index("Entrez_Gene_Id")
    symbol_to_entrez = {sym: ent for sym, ent in PATHWAY_ENTREZ.items() if ent in gene_rows.index}
    audit["pathway_genes_found"] = f"{len(symbol_to_entrez)}/16"
    gene_matrix = gene_rows.loc[list(symbol_to_entrez.values())].T
    gene_matrix.columns = list(symbol_to_entrez.keys())
    gene_matrix.index.name = "SAMPLE_ID"

    # Align expression rows to the cohort's sample order (the expression file
    # stores samples in its own column order, not the cohort table order).
    gene_matrix = gene_matrix.reindex(cohort["SAMPLE_ID"].astype(str))
    if gene_matrix.isna().any().any():
        missing = gene_matrix.index[gene_matrix.isna().any(axis=1)].tolist()
        raise RuntimeError(f"Cohort samples missing from expression matrix: {missing}")

    scores = _pathway_scores_from_matrix(gene_matrix, "MSKCC", audit)

    p1 = pd.to_numeric(cohort["GLEASON_SCORE_1"], errors="coerce")
    p2 = pd.to_numeric(cohort["GLEASON_SCORE_2"], errors="coerce")
    gleason_total = p1 + p2
    high_risk = ((p1 >= 4) | (p2 >= 4))
    high_risk = high_risk.where(~(p1.isna() & p2.isna())).astype("Float64")

    path_t = cohort["PATH_T_STAGE"].astype("string").str.upper().str.strip()
    t_risk = path_t.isin(MSKCC_T3_T4).astype("float").where(path_t.notna())

    out = pd.DataFrame({
        "SAMPLE_ID": cohort["SAMPLE_ID"],
        "PATIENT_ID": cohort["PATIENT_ID"],
        "y": cohort["BCR_STATUS"].astype(int),
        **{name: scores[name].values for name in PATHWAY_SETS},
        "Gleason_Total": gleason_total.values,
        "High_Risk_Gleason": high_risk.values,
        "T_Stage_Risk": t_risk.values,
    })[["SAMPLE_ID", "PATIENT_ID", "y", *FEATURE_ORDER]]

    audit["samples"] = len(out)
    audit["n_events"] = int(out["y"].sum())
    audit["gleason_total_missing"] = int(out["Gleason_Total"].isna().sum())
    audit["gleason_total_range"] = [
        float(np.nanmin(out["Gleason_Total"])), float(np.nanmax(out["Gleason_Total"])),
    ]
    audit["t_stage_risk"] = {
        "risk1": int((out["T_Stage_Risk"] == 1).sum()),
        "risk0": int((out["T_Stage_Risk"] == 0).sum()),
        "missing": int(out["T_Stage_Risk"].isna().sum()),
    }
    audit["feature_summary"] = {
        f: {"mean": float(out[f].astype(float).mean()), "std": float(out[f].astype(float).std())}
        for f in FEATURE_ORDER
    }
    return out, audit


def main() -> None:
    """Build both transferable tables, validate, save, and report."""
    tcga, tcga_audit = build_tcga_transferable_features()
    mskcc, mskcc_audit = build_mskcc_transferable_features()

    TCGA_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tcga.to_csv(TCGA_OUT_CSV, index=False)
    MSKCC_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mskcc.to_csv(MSKCC_OUT_CSV, index=False)
    for path, payload in (
        (TCGA_SUMMARY_JSON, tcga_audit),
        (MSKCC_SUMMARY_JSON, mskcc_audit),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    logger.info("Saved %s and %s", TCGA_OUT_CSV, MSKCC_OUT_CSV)

    print("=== Step 2: transferable features ===")
    print(f"TCGA : {len(tcga)} rows ({tcga_audit['n_train']} train / {tcga_audit['n_test']} test), "
          f"events={tcga_audit['n_events']}, Gleason NaN={tcga_audit['gleason_total_missing']}")
    print(f"MSKCC: {len(mskcc)} rows, events={mskcc_audit['n_events']}, "
          f"Gleason NaN={mskcc_audit['gleason_total_missing']}, "
          f"genes found={mskcc_audit['pathway_genes_found']}")
    print("Gates:", {k: v for k, v in tcga_audit.items() if "check" in k})


if __name__ == "__main__":
    main()
