"""
Feature selection module refactored for Late Fusion architecture.

This module now provides backward-compatible imports from the new
split architecture modules:
- src/genomic_selector.py: Genomic branch (MI + PSO on genes only)
- src/clinical_engineer.py: Clinical branch (domain features only)

The old 3-Layer/Early Fusion logic has been REMOVED.
"""

from __future__ import annotations

# Import from new genomic selector module (Genomic Branch)
from src.genomic_selector import (
    fit_genomic_selector,
    transform_genomic,
    pso_feature_select_genomic,
    run_genomic_feature_selection,
    sigmoid,
    repair_exact_k,
)

# Import from new clinical engineer module (Clinical Branch)
from src.clinical_engineer import (
    create_clinical_features,
    get_clinical_feature_names,
    prepare_clinical_branch,
)

# Re-export for backward compatibility where needed
fit_filter_selector = fit_genomic_selector
transform_selected = transform_genomic
create_engineered_features = create_clinical_features
