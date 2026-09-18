"""
MSKCC cohort construction for external validation (Pipeline Step 1).

Reads the raw cBioPortal PRAD-MSKCC 2010 export and builds a clean
per-sample cohort table with clinical labels and transferable raw fields:

- one row per expression sample with valid clinical mapping
- BCR_STATUS (0 = DiseaseFree, 1 = Recurred) derived from DFS_STATUS
- Gleason pattern primary / secondary / total (numeric-coerced)
- PATH_T_STAGE and CLIN_T_STAGE (raw strings; mapper applied in Step 2)
- DFS_MONTHS kept for context only (never used as a feature)

Filters (each one is logged with before/after counts so the cohort
definition is fully auditable):
- only SAMPLE_CLASS == "Tumor" (drops cell lines and xenografts)
- only SAMPLE_TYPE == "Primary" (drops metastases and NaN)
- patient must have a valid DFS_STATUS (drops NaN outcome rows)
- expression sample must exist in the Agilent microarray matrix
- one sample per patient (keeps the first primary tumor sample)

Run from the ``core`` directory:

    python -m src.mskcc_cohort
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

import config
from src.io import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MSKCC_DIR = config.PROJECT_ROOT / "data" / "external" / "prad_mskcc"
MSKCC_COHORT_CSV = config.PROJECT_ROOT / "data" / "external" / "mskcc_cohort.csv"
MSKCC_SUMMARY_JSON = config.TABLES_DIR / "mskcc_cohort_summary.json"

EXPRESSION_FILE = "data_mrna_agilent_microarray.txt"
PATIENT_FILE = "data_clinical_patient.txt"
SAMPLE_FILE = "data_clinical_sample.txt"

# cBioPortal encoded values for the disease-free survival status.
DFS_STATUS_MAP = {
    "0:DiseaseFree": 0,
    "1:Recurred": 1,
}

COHORT_COLUMNS = [
    "SAMPLE_ID",
    "PATIENT_ID",
    "BCR_STATUS",
    "DFS_MONTHS",
    "GLEASON_SCORE_1",
    "GLEASON_SCORE_2",
    "GLEASON_SCORE",
    "CLIN_T_STAGE",
    "PATH_T_STAGE",
    "ERG_FUSION_GEX",
    "SAMPLE_TYPE",
    "SAMPLE_CLASS",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_clinical_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MSKCC patient- and sample-level clinical tables."""
    patient_path = MSKCC_DIR / PATIENT_FILE
    sample_path = MSKCC_DIR / SAMPLE_FILE
    if not patient_path.exists():
        raise FileNotFoundError(f"MSKCC patient clinical file not found: {patient_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"MSKCC sample clinical file not found: {sample_path}")

    patient = pd.read_csv(patient_path, sep="\t", skiprows=4)
    sample = pd.read_csv(sample_path, sep="\t", skiprows=4)
    logger.info(
        "Loaded MSKCC clinical tables: %d patients, %d samples",
        len(patient), len(sample),
    )
    return patient, sample


def _load_expression_sample_ids() -> set[str]:
    """Return the set of sample IDs present in the Agilent expression matrix.

    Only the header row is read; the 49 MB expression body is not needed
    for cohort construction.
    """
    expression_path = MSKCC_DIR / EXPRESSION_FILE
    if not expression_path.exists():
        raise FileNotFoundError(f"MSKCC expression matrix not found: {expression_path}")
    columns = pd.read_csv(expression_path, sep="\t", nrows=0).columns.tolist()
    sample_ids = {str(c) for c in columns if c != "Entrez_Gene_Id"}
    logger.info("Expression matrix contains %d samples", len(sample_ids))
    return sample_ids


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------
def build_mskcc_cohort() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the clean MSKCC cohort and an audit summary of every filter.

    Returns:
        Tuple of (cohort DataFrame, audit dict with per-filter counts).
    """
    audit: dict[str, Any] = {}
    patient, sample = _load_clinical_tables()
    expression_ids = _load_expression_sample_ids()

    # -- 1. Patient level: derive binary BCR status from DFS_STATUS ---------
    patient = patient.copy()
    patient["BCR_STATUS"] = patient["DFS_STATUS"].map(DFS_STATUS_MAP)
    n_patients_raw = len(patient)
    patient = patient[patient["BCR_STATUS"].notna()].copy()
    patient["BCR_STATUS"] = patient["BCR_STATUS"].astype(int)
    audit["patients_raw"] = n_patients_raw
    audit["patients_with_dfs_status"] = len(patient)
    logger.info(
        "Patients with valid DFS_STATUS: %d / %d", len(patient), n_patients_raw
    )

    # -- 2. Sample level: keep primary tumors only --------------------------
    sample = sample.copy()
    audit["samples_raw"] = len(sample)
    tumor_mask = sample["SAMPLE_CLASS"] == "Tumor"
    audit["samples_after_tumor_filter"] = int(tumor_mask.sum())
    sample = sample[tumor_mask]
    primary_mask = sample["SAMPLE_TYPE"] == "Primary"
    audit["samples_after_primary_filter"] = int(primary_mask.sum())
    sample = sample[primary_mask]
    logger.info(
        "Primary tumor samples: %d / %d raw",
        audit["samples_after_primary_filter"], audit["samples_raw"],
    )

    # -- 3. Merge sample -> patient outcome + patient-level T stage ----------
    # CLIN_T_STAGE / PATH_T_STAGE live on the PATIENT table, so they must be
    # carried through the merge explicitly (they are 1:1 here because step 5
    # enforces one sample per patient).
    merged = sample.merge(
        patient[
            [
                "PATIENT_ID",
                "BCR_STATUS",
                "DFS_MONTHS",
                "CLIN_T_STAGE",
                "PATH_T_STAGE",
            ]
        ],
        on="PATIENT_ID",
        how="inner",
    )
    audit["samples_matched_to_dfs_patient"] = len(merged)
    logger.info(
        "Primary tumor samples matched to patients with DFS: %d",
        len(merged),
    )

    # -- 4. Keep only samples present in the expression matrix --------------
    merged = merged[merged["SAMPLE_ID"].astype(str).isin(expression_ids)]
    audit["samples_present_in_expression"] = len(merged)
    logger.info(
        "Samples present in expression matrix: %d", len(merged)
    )

    # -- 5. One sample per patient (keep first primary tumor) ---------------
    duplicates = int(merged["PATIENT_ID"].duplicated().sum())
    if duplicates:
        logger.warning("Dropping %d duplicate patient rows (keep=first)", duplicates)
    merged = merged.drop_duplicates(subset="PATIENT_ID", keep="first")
    audit["duplicate_patient_rows_dropped"] = duplicates
    audit["cohort_samples_final"] = len(merged)

    # -- 6. Column selection and numeric coercion ---------------------------
    cohort = merged.reindex(columns=COHORT_COLUMNS).copy()
    for col in ("DFS_MONTHS", "GLEASON_SCORE_1", "GLEASON_SCORE_2", "GLEASON_SCORE"):
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce")
        audit[f"{col}_missing"] = int(cohort[col].isna().sum())

    # -- 7. Acceptance checks (fail loudly, never silently) -----------------
    if cohort["BCR_STATUS"].isna().any():
        raise RuntimeError("Cohort contains rows without a valid BCR_STATUS")
    if not set(cohort["BCR_STATUS"].unique()).issubset({0, 1}):
        raise RuntimeError("BCR_STATUS must be binary {0, 1}")
    bad_type = cohort[
        (cohort["SAMPLE_CLASS"] != "Tumor") | (cohort["SAMPLE_TYPE"] != "Primary")
    ]
    if len(bad_type):
        raise RuntimeError(f"Cohort contains non-primary tumor rows: {len(bad_type)}")
    if len(cohort) == 0:
        raise RuntimeError("Cohort is empty after filtering")

    audit["n_events"] = int(cohort["BCR_STATUS"].sum())
    audit["n_non_events"] = int((cohort["BCR_STATUS"] == 0).sum())
    audit["event_rate"] = float(cohort["BCR_STATUS"].mean())
    audit["path_t_stage_missing"] = int(cohort["PATH_T_STAGE"].isna().sum())
    audit["path_t_stage_t3t4"] = int(
        cohort["PATH_T_STAGE"]
        .astype("string")
        .str.upper()
        .str.strip()
        .isin(["T3", "T3A", "T3B", "T3C", "T4"])
        .sum()
    )
    if audit["path_t_stage_missing"] == len(cohort):
        raise RuntimeError("PATH_T_STAGE is empty — patient T-stage merge failed")
    return cohort, audit


def summarize(cohort: pd.DataFrame, audit: dict[str, Any]) -> str:
    """Render a human-readable cohort report (also captured in the JSON)."""
    lines = [
        "=== MSKCC cohort (Step 1) ===",
        f"Samples (patients):            {len(cohort)}",
        f"Events (BCR / Recurred):       {audit['n_events']}",
        f"Non-events (DiseaseFree):      {audit['n_non_events']}",
        f"Event rate:                    {audit['event_rate']:.1%}",
        f"Patients raw / with DFS:       {audit['patients_raw']} / {audit['patients_with_dfs_status']}",
        f"Samples raw / tumor / primary: {audit['samples_raw']} / {audit['samples_after_tumor_filter']} / {audit['samples_after_primary_filter']}",
        f"Matched to DFS / in expression:{audit['samples_matched_to_dfs_patient']} / {audit['samples_present_in_expression']}",
        f"Duplicates dropped:            {audit['duplicate_patient_rows_dropped']}",
        f"Gleason total missing:         {audit['GLEASON_SCORE_missing']}",
        f"Gleason pattern 1 missing:     {audit['GLEASON_SCORE_1_missing']}",
        f"Gleason pattern 2 missing:     {audit['GLEASON_SCORE_2_missing']}",
        f"PATH_T_STAGE missing / T3-T4: {audit['path_t_stage_missing']} / {audit['path_t_stage_t3t4']}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Build, validate, save, and report the MSKCC cohort."""
    cohort, audit = build_mskcc_cohort()

    MSKCC_COHORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(MSKCC_COHORT_CSV, index=False)
    logger.info("Saved cohort -> %s", MSKCC_COHORT_CSV)

    MSKCC_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MSKCC_SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    logger.info("Saved audit summary -> %s", MSKCC_SUMMARY_JSON)

    print(summarize(cohort, audit))


if __name__ == "__main__":
    main()
