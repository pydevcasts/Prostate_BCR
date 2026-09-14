"""Domain-informed feature engineering for the genomic branch."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.features_config import AR_GENES, MIN_GENES_FOR_PATHWAY, PSA_GENES, PROLIF_GENES
from src.io import logger


def create_genomic_pathway_features(
    X_genomic: pd.DataFrame,
    strict_mode: bool = False,
    required_genes: Optional[dict[str, list[str]]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add biologically informed pathway scores to genomic expression data."""
    X = X_genomic.copy()
    created_features: list[str] = []
    gene_sets = {
        "PSA_Pathway_Score": list(required_genes.get("PSA", PSA_GENES) if required_genes else PSA_GENES),
        "AR_Signaling_Score": list(required_genes.get("AR", AR_GENES) if required_genes else AR_GENES),
        "Proliferation_Score": list(required_genes.get("PROLIF", PROLIF_GENES) if required_genes else PROLIF_GENES),
    }

    for feature_name, genes in gene_sets.items():
        available_genes = [gene for gene in genes if gene in X.columns]
        if strict_mode and len(available_genes) != len(genes):
            continue
        if len(available_genes) < MIN_GENES_FOR_PATHWAY:
            continue
        X[feature_name] = X[available_genes].mean(axis=1)
        created_features.append(feature_name)
        logger.info("Genomic: %s (from %d genes)", feature_name, len(available_genes))

    return X, created_features
