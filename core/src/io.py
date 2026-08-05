"""
I/O utilities for the Prostate BCR project.

Centralizes all file operations: reading raw data, saving processed outputs,
path validation, and logging configuration. Every module that touches the
filesystem imports from here — never reads/writes directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

import config as config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str = "prostate_bcr", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project-wide logger (singleton per name)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = setup_logger()


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------
def validate_path(path: str | Path, description: str = "file") -> Path:
    """Return Path object, raising FileNotFoundError if it does not exist."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{description} not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------
def load_clinical_raw() -> pd.DataFrame:
    """Load raw TCGA clinical data from TSV."""
    path = validate_path(config.CLINICAL_RAW, "Raw clinical data")
    logger.info("Loading raw clinical data: %s", path)
    df = pd.read_csv(path, sep="\t")
    logger.info("Loaded clinical data: %d rows × %d columns", *df.shape)
    return df


def load_rna_seq_raw() -> pd.DataFrame:
    """Load raw RNA-Seq expression matrix (genes × samples)."""
    path = validate_path(config.RNA_SEQ_RAW, "Raw RNA-Seq data")
    logger.info("Loading raw RNA-Seq data: %s", path)
    df = pd.read_csv(path, sep="\t", index_col=0)
    logger.info("Loaded RNA-Seq matrix: %d genes × %d samples", *df.shape)
    return df


def load_processed_clinical() -> pd.DataFrame:
    """Load the fully processed clinical CSV (post src.clinical pipeline)."""
    path = validate_path(config.CLINICAL_PROCESSED, "Processed clinical data")
    logger.info("Loading processed clinical data: %s", path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Savers
# ---------------------------------------------------------------------------
def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    directory: Path | None = None,
    index: bool = False,
) -> Path:
    """Persist a DataFrame to CSV."""
    directory = Path(directory or config.PROCESSED_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    df.to_csv(path, index=index)
    logger.info("Saved %d rows → %s", len(df), path)
    return path


def save_figure(fig: Any, filename: str, dpi: int = config.FIGURE_DPI) -> Path:
    """Save a matplotlib figure to the figures directory."""
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure → %s", path)
    return path


def save_table(df: pd.DataFrame, filename: str, index: bool = False) -> Path:
    """Save a summary table to the tables directory."""
    return save_dataframe(df, filename, directory=config.TABLES_DIR, index=index)


def save_model(model: Any, filename: str) -> Path:
    """Serialize a fitted model with joblib."""
    path = config.MODELS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model → %s", path)
    return path


def load_model(filename: str) -> Any:
    """Deserialize a model from the models directory."""
    path = validate_path(config.MODELS_DIR / filename, "Saved model")
    logger.info("Loading model from %s", path)
    return joblib.load(path)