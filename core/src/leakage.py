"""
Data leakage detection and auditing for TCGA-PRAD BCR prediction.

Handles detection of target leakage, temporal leakage, preprocessing
leakage, and duplicate patients. Generates structured reports for
publication transparency.

All leakage audits should run BEFORE train/test split to ensure
no information from test data contaminates the training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as config
from src.io import logger


# ---------------------------------------------------------------------------
# Column-level leakage
# ---------------------------------------------------------------------------
def detect_leakage_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that leak target information.

    Parameters
    ----------
    df : DataFrame to scan for leakage columns.

    Returns
    -------
    List of column names that match known leakage patterns.
    """
    present = [c for c in config.LEAKAGE_COLUMNS if c in df.columns]
    if present:
        logger.warning("Detected %d leakage columns: %s", len(present), present)
    else:
        logger.info("No known leakage columns detected")
    return present


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that leak target information.

    Parameters
    ----------
    df : DataFrame to clean.

    Returns
    -------
    DataFrame with leakage columns removed.
    """
    to_drop = detect_leakage_columns(df)
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
        logger.info("Dropped %d leakage columns", len(to_drop))
    return df


# ---------------------------------------------------------------------------
# Temporal leakage (PSA audit)
# ---------------------------------------------------------------------------
def temporal_psa_audit(
    raw_clinical_path: Path | str,
    output_path: Path | str | None = None,
) -> dict:
    """Audit temporal ordering of PSA measurements relative to BCR events.

    This checks whether PSA measurements were taken at or after the
    documented recurrence date, which would indicate temporal leakage.

    Parameters
    ----------
    raw_clinical_path : Path to raw TCGA clinical TSV file.
    output_path : Optional path to save the audit CSV.

    Returns
    -------
    Dictionary with audit results.
    """
    raw_clinical_path = Path(raw_clinical_path)
    if not raw_clinical_path.exists():
        logger.warning("Raw clinical file not found: %s — temporal audit skipped", raw_clinical_path)
        return {"skipped": True, "reason": "file_not_found"}

    logger.info("Running temporal PSA audit on %s", raw_clinical_path)

    # Load raw clinical data (skip metadata rows)
    raw = pd.read_csv(raw_clinical_path, sep="\t", skiprows=[1, 2, 3], header=0)

    required_cols = [
        "Days to PSA",
        "Psa most recent results",
        "Days to biochemical recurrence first",
        "Biochemical Recurrence Indicator",
    ]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        logger.warning("Temporal audit skipped; missing columns: %s", missing)
        return {"skipped": True, "reason": "missing_columns", "missing": missing}

    audit = raw[required_cols].copy()
    for col in ["Days to PSA", "Psa most recent results", "Days to biochemical recurrence first"]:
        audit[col] = pd.to_numeric(audit[col], errors="coerce")

    # Filter to BCR-positive cases with valid dates
    yes_cases = audit[
        audit["Biochemical Recurrence Indicator"].eq("YES")
    ].dropna(subset=["Days to PSA", "Days to biochemical recurrence first"])

    results = {"skipped": False, "n_yes_cases": len(yes_cases)}

    if len(yes_cases) > 0:
        n_after = int(
            (yes_cases["Days to PSA"] >= yes_cases["Days to biochemical recurrence first"]).sum()
        )
        results["n_psa_at_or_after_recurrence"] = n_after
        logger.warning(
            "YES cases with PSA at/after recurrence: %d/%d",
            n_after, len(yes_cases),
        )
    else:
        results["n_psa_at_or_after_recurrence"] = 0
        logger.info("No BCR-positive cases with valid PSA dates found")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(output_path, index=False)
        logger.info("Saved temporal audit to %s", output_path)

    return results


# ---------------------------------------------------------------------------
# Preprocessing leakage
# ---------------------------------------------------------------------------
def detect_preprocessing_leakage(
    fitted_on_train: bool,
    transform_applied_to_test: bool,
) -> dict:
    """Check for preprocessing leakage patterns.

    This is a conceptual check to be called by pipeline orchestration
    code to verify that preprocessing steps are fitted only on training
    data before being applied to test data.

    Parameters
    ----------
    fitted_on_train : Whether the transformer was fitted on training data.
    transform_applied_to_test : Whether the transform is being applied to test data.

    Returns
    -------
    Dictionary with leakage status.
    """
    leakage_detected = False
    reasons = []

    if not fitted_on_train:
        leakage_detected = True
        reasons.append("Transformer not fitted on training data")

    if transform_applied_to_test and not fitted_on_train:
        leakage_detected = True
        reasons.append("Transform applied to test without fitting on train")

    if leakage_detected:
        logger.warning("Preprocessing leakage detected: %s", "; ".join(reasons))
    else:
        logger.info("No preprocessing leakage detected")

    return {
        "leakage_detected": leakage_detected,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
def detect_duplicate_patients(
    df: pd.DataFrame,
    id_column: str = config.PATIENT_ID_COLUMN,
) -> pd.DataFrame:
    """Detect and report duplicate patient IDs.

    Parameters
    ----------
    df : DataFrame to check.
    id_column : Column containing patient IDs.

    Returns
    -------
    DataFrame of duplicate records (empty if none found).
    """
    if id_column not in df.columns:
        logger.warning("Column '%s' not found; skipping duplicate check", id_column)
        return pd.DataFrame()

    duplicates = df[df[id_column].duplicated(keep=False)]
    if len(duplicates) > 0:
        logger.warning("Found %d duplicate patient IDs", len(duplicates))
    else:
        logger.info("No duplicate patient IDs found")

    return duplicates


# ---------------------------------------------------------------------------
# Comprehensive leakage report
# ---------------------------------------------------------------------------
def generate_leakage_report(
    df: pd.DataFrame,
    *,
    id_column: str = config.PATIENT_ID_COLUMN,
    target_column: str = config.TARGET_COLUMN,
    raw_clinical_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict:
    """Generate a comprehensive leakage audit report.

    Parameters
    ----------
    df : Merged DataFrame to audit.
    id_column : Column containing patient IDs.
    target_column : Column containing the target variable.
    raw_clinical_path : Optional path to raw clinical TSV for temporal audit.
    output_path : Optional path to save the report CSV.

    Returns
    -------
    Dictionary with all audit results.
    """
    logger.info("=" * 60)
    logger.info("LEAKAGE AUDIT REPORT")
    logger.info("=" * 60)

    report = {
        "n_samples": len(df),
        "n_features": df.shape[1],
    }

    # 1. Column-level leakage
    leakage_cols = detect_leakage_columns(df)
    report["n_leakage_columns"] = len(leakage_cols)
    report["leakage_columns"] = leakage_cols

    # 2. Duplicate patients
    duplicates = detect_duplicate_patients(df, id_column)
    report["n_duplicate_patients"] = len(duplicates)

    # 3. Target column validation
    if target_column in df.columns:
        n_target_missing = int(df[target_column].isna().sum())
        report["n_target_missing"] = n_target_missing
        if n_target_missing > 0:
            logger.warning("Found %d rows with missing target", n_target_missing)
    else:
        report["n_target_missing"] = None
        logger.warning("Target column '%s' not found", target_column)

    # 4. Temporal PSA audit (optional)
    if raw_clinical_path is not None:
        temporal_results = temporal_psa_audit(raw_clinical_path)
        report["temporal_audit"] = temporal_results
    else:
        report["temporal_audit"] = {"skipped": True, "reason": "no_raw_file_provided"}

    # Summary
    logger.info("-" * 60)
    logger.info("Summary:")
    logger.info("  Samples: %d", report["n_samples"])
    logger.info("  Features: %d", report["n_features"])
    logger.info("  Leakage columns: %d", report["n_leakage_columns"])
    logger.info("  Duplicate patients: %d", report["n_duplicate_patients"])
    logger.info("  Missing target: %s", report["n_target_missing"])
    logger.info("=" * 60)

    # Save report
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten for CSV
        flat_report = {}
        for key, value in report.items():
            if isinstance(value, (list, dict)):
                if key == "leakage_columns":
                    flat_report[key] = "; ".join(value)
                elif key == "temporal_audit":
                    for sub_key, sub_value in value.items():
                        flat_report[f"temporal_{sub_key}"] = sub_value
            else:
                flat_report[key] = value

        pd.Series(flat_report).to_csv(output_path, header=["value"])
        logger.info("Saved leakage report to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def audit_leakage(
    df: pd.DataFrame,
    *,
    drop_columns: bool = True,
    id_column: str = config.PATIENT_ID_COLUMN,
    target_column: str = config.TARGET_COLUMN,
    raw_clinical_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the full leakage audit and optionally clean the DataFrame.

    Parameters
    ----------
    df : Merged DataFrame to audit.
    drop_columns : If True, remove leakage columns from df.
    id_column : Column containing patient IDs.
    target_column : Column containing the target variable.
    raw_clinical_path : Optional path to raw clinical TSV for temporal audit.
    output_path : Optional path to save the report CSV.

    Returns
    -------
    Tuple of (cleaned_df, report_dict).
    """
    report = generate_leakage_report(
        df,
        id_column=id_column,
        target_column=target_column,
        raw_clinical_path=raw_clinical_path,
        output_path=output_path,
    )

    cleaned = df.copy()
    if drop_columns:
        cleaned = drop_leakage_columns(cleaned)

    return cleaned, report