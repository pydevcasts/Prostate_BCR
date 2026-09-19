"""
Feature-selection stability analysis (Pipeline Step 7).

PRESPECIFIED RULES (declared and frozen before any stability number or
figure was computed; nothing below was tuned after seeing results):

    R1. A "selection event" is one fitted PSO selection for a branch.
        The analysis pools 35 independent selection events per branch:
          - 15 fold-local selections from the repeated nested CV
            (`nested_cv_raw_pso_selections.csv`, 3 repeats x 5 folds,
            different outer splits, fold-local training subsets), and
          - 20 fresh independent runs on the FULL canonical TCGA train
            split (343 rows, train_test_split seed 42 reproduced and
            hard-validated), seeds 500..519, with the identical
            fold-local recipe: build_combined_pipeline preprocessing ->
            Variance -> MI top-200 -> binary PSO (target 40,
            random_state = run_seed + 1000 genomic, + 2000 clinical).
    R2. Stability(feature) = (# events selecting the feature) / 35.
    R3. A feature is "stable" iff Stability >= 0.70 (>= 25/35 events).
        This threshold was fixed before running; it is NOT recalibrated
        after seeing the distribution.
    R4. Both branches are analyzed with the same rules; the genomic
        branch is the headline for the manuscript, the clinical branch
        is reported as a supplementary table.
    R5. No result of this module feeds back into any model, threshold,
        or feature set used elsewhere (read-only diagnostic).

Outputs
-------
- core/outputs/tables/stability_summary.json      (rules, counts, top features)
- core/outputs/tables/stability_genomic.csv       (35-event frequency table)
- core/outputs/tables/stability_clinical.csv
- core/outputs/figures/feature_stability.png      (publication bar chart)Resumability: every completed fresh run is appended atomically to
``core/outputs/tables/stability_fresh_runs.csv`` before the next run
starts, so an interrupted execution loses at most one run.

Branch masks come from the canonical ``identify_column_groups`` (the same
function ``build_combined_pipeline`` uses). An earlier hand-rolled prefix
list missed 45 one-hot clinical columns and leaked them into the genomic
branch; that bug is fixed and the corrected Step 6 artifact is used here.

Run from the ``core`` directory:

    python -m src.stability_selection                 # folds + 20 fresh runs
    python -m src.stability_selection --aggregate-only  # re-aggregate checkpoint
    python -m src.stability_selection --folds-only      # reuse the 15 folds only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    identify_column_groups,
    transform_data,
)
from src.nested_cv_raw import load_raw_merged

# ---------------------------------------------------------------------------
# Paths / prespecified constants
# ---------------------------------------------------------------------------
PSO_FOLDS_CSV = config.TABLES_DIR / "nested_cv_raw_pso_selections.csv"
SUMMARY_JSON = config.TABLES_DIR / "stability_summary.json"
GENOMIC_CSV = config.TABLES_DIR / "stability_genomic.csv"
CLINICAL_CSV = config.TABLES_DIR / "stability_clinical.csv"
FIGURE_PNG = config.FIGURES_DIR / "feature_stability.png"

FRESH_RUN_SEEDS = tuple(range(500, 520))  # R1: 20 independent full-train runs
STABILITY_THRESHOLD = 0.70                # R3: frozen before running
N_EVENTS_EXPECTED = 35                    # 15 folds + 20 fresh runs
TOP_N_PLOT = 30
CHECKPOINT_CSV = config.TABLES_DIR / "stability_fresh_runs.csv"  # resumability


def load_fold_selections() -> pd.DataFrame:
    """Load the 15 nested-CV-fold PSO selections (Step 6 artifact)."""
    df = pd.read_csv(PSO_FOLDS_CSV)
    if df[["repeat_seed", "fold", "branch", "feature"]].isna().any().any():
        raise RuntimeError("Malformed PSO selections artifact")
    n_events = df.groupby("branch").apply(
        lambda g: len(g.groupby(["repeat_seed", "fold"])), include_groups=False
    )
    if not (n_events == 15).all():
        raise RuntimeError(f"Expected 15 fold events per branch, got {n_events.to_dict()}")
    logger.info("Fold selections loaded: %d rows, 15 events per branch", len(df))
    return df


def load_checkpoint() -> dict[int, dict[str, list[str]]]:
    """Completed fresh runs from the resumable checkpoint (empty if none)."""
    if not CHECKPOINT_CSV.exists():
        return {}
    df = pd.read_csv(CHECKPOINT_CSV)
    done: dict[int, dict[str, list[str]]] = {}
    for (seed, branch), grp in df.groupby(["run_seed", "branch"]):
        done.setdefault(int(seed), {})[branch] = (
            grp.sort_values("rank")["feature"].tolist()
        )
    return done


def append_checkpoint(run_seed: int, picks: dict[str, list[str]]) -> None:
    """Append one finished fresh run to the checkpoint atomically."""
    CHECKPOINT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"run_seed": run_seed, "branch": branch, "feature": feat, "rank": rank}
        for branch, feats in picks.items()
        for rank, feat in enumerate(feats, start=1)
    ]
    updated = pd.DataFrame(rows)
    if CHECKPOINT_CSV.exists():
        old = pd.read_csv(CHECKPOINT_CSV)
        updated = pd.concat(
            [old[old["run_seed"] != run_seed], updated], ignore_index=True
        )
    tmp = CHECKPOINT_CSV.with_suffix(".tmp")
    updated.to_csv(tmp, index=False)
    os.replace(tmp, CHECKPOINT_CSV)


# ---------------------------------------------------------------------------
# One independent full-train selection event (canonical recipe)
# ---------------------------------------------------------------------------
def run_full_train_selection(
    X_raw: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    *,
    run_seed: int,
) -> dict[str, list[str]]:
    """Fit preprocess -> variance -> MI -> PSO on the full train split."""
    X_train = X_raw.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    # Canonical grouping (same fix as nested_cv_raw — see the note there).
    clinical_mask, gene_mask = identify_column_groups(X_raw)

    preprocessor = fit_preprocessing(build_combined_pipeline(X_train), X_train)
    train_pp = transform_data(preprocessor, X_train)
    n_clinical = len(clinical_mask)
    Xc_train = pd.DataFrame(train_pp[:, :n_clinical], columns=clinical_mask)
    Xg_train = pd.DataFrame(train_pp[:, n_clinical:], columns=gene_mask)

    picks: dict[str, list[str]] = {}

    genomic_selector = fit_genomic_selector(Xg_train, y_train, random_state=run_seed)
    genomic_features, _, _, _ = pso_feature_select_genomic(
        Xg_train, y_train, genomic_selector["mi_features"],
        n_features=min(config.PSO_FINAL_K, len(genomic_selector["mi_features"])),
        random_state=run_seed + 1000,
        branch_name=f"Stability genomic s{run_seed}",
    )
    picks["genomic"] = genomic_features

    clinical_selector = fit_clinical_selector(Xc_train, y_train, random_state=run_seed)
    clinical_features, _, _, _ = pso_feature_select_genomic(
        Xc_train, y_train, clinical_selector["mi_features"],
        n_features=min(config.PSO_FINAL_K, len(clinical_selector["mi_features"])),
        random_state=run_seed + 2000,
        branch_name=f"Stability clinical s{run_seed}",
    )
    picks["clinical"] = clinical_features
    return picks


def build_frequency_table(
    fold_sel: pd.DataFrame,
    fresh_sel: list[dict[str, list[str]]],
    branch: str,
    total_events: int,
) -> pd.DataFrame:
    """Selection-frequency table for one branch over all events (R2)."""
    counts: dict[str, dict[str, Any]] = {}

    fold_b = fold_sel[fold_sel["branch"] == branch]
    for feat, grp in fold_b.groupby("feature"):
        entry = counts.setdefault(feat, {"fold_events": 0, "fresh_events": 0, "ranks": []})
        # Count EVENTS (distinct repeat_seed x fold), not group rows: an
        # earlier version incremented by 1 per feature, which capped the
        # fold component at a single event and under-reported stability.
        entry["fold_events"] += int(
            grp[["repeat_seed", "fold"]].drop_duplicates().shape[0]
        )
        entry["ranks"].extend(grp["rank"].tolist())

    for picks in fresh_sel:
        for rank, feat in enumerate(picks[branch], start=1):
            entry = counts.setdefault(feat, {"fold_events": 0, "fresh_events": 0, "ranks": []})
            entry["fresh_events"] += 1
            entry["ranks"].append(rank)

    rows = []
    for feat, entry in counts.items():
        n_sel = entry["fold_events"] + entry["fresh_events"]
        rows.append({
            "feature": feat,
            "fold_events": entry["fold_events"],
            "fresh_events": entry["fresh_events"],
            "n_selected": n_sel,
            "n_events": total_events,
            "stability": round(n_sel / total_events, 4),
            "stable_at_0.70": bool(n_sel / total_events >= STABILITY_THRESHOLD),
            "mean_rank_when_selected": round(float(np.mean(entry["ranks"])), 1),
        })
    table = pd.DataFrame(rows).sort_values(
        ["stability", "fold_events", "feature"], ascending=[False, False, True]
    ).reset_index(drop=True)
    return table


def plot_stability(table: pd.DataFrame, path, branch_label: str) -> None:
    """Publication-style bar chart of the top features with the 0.70 line."""
    top = table.head(TOP_N_PLOT).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = ["#3fb950" if s >= STABILITY_THRESHOLD else "#58a6ff" for s in top["stability"]]
    ax.barh(top["feature"], top["stability"], color=colors)
    ax.axvline(STABILITY_THRESHOLD, color="#f85149", linestyle="--", linewidth=1.2)
    ax.text(STABILITY_THRESHOLD + 0.005, 0.2, "stability = 0.70", color="#f85149", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"Selection frequency over {N_EVENTS_EXPECTED} PSO events")
    ax.set_ylabel("Feature")
    ax.set_title(f"Feature-selection stability — {branch_label} branch")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Frequency analysis over 15 folds + 20 fresh runs (or folds only)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds-only", action="store_true",
        help="analyze only the 15 nested-CV fold events (no fresh runs)",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="skip PSO runs and re-aggregate the existing fresh-run checkpoint",
    )
    args = parser.parse_args()

    t_start = time.time()
    fold_sel = load_fold_selections()
    done = load_checkpoint()

    if not args.aggregate_only and not args.folds_only:
        X_raw, y = load_raw_merged()

        # Canonical split reproduction gate (same recipe as Step 2).
        from sklearn.model_selection import train_test_split

        train_idx, _ = train_test_split(
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
        logger.info("Canonical train split reproduced: %d rows", len(train_idx))

        remaining = [s for s in FRESH_RUN_SEEDS if s not in done]
        logger.info(
            "Fresh runs: %d/%d already checkpointed, %d remaining",
            len(done), len(FRESH_RUN_SEEDS), len(remaining),
        )
        for run_seed in remaining:
            t0 = time.time()
            picks = run_full_train_selection(X_raw, y, train_idx, run_seed=run_seed)
            append_checkpoint(run_seed, picks)
            done[run_seed] = picks
            logger.info(
                "Fresh run seed=%d done in %.1f min (genomic head: %s)",
                run_seed, (time.time() - t0) / 60, ", ".join(picks["genomic"][:5]),
            )

    if args.folds_only:
        total_events = 15
        fresh_sel = []
    else:
        missing = [s for s in FRESH_RUN_SEEDS if s not in done]
        if missing:
            raise RuntimeError(
                f"Fresh-run checkpoint incomplete (missing seeds {missing}); "
                "re-run without --aggregate-only"
            )
        fresh_sel = [done[s] for s in FRESH_RUN_SEEDS]
        total_events = 15 + len(fresh_sel)
    if not args.folds_only and total_events != N_EVENTS_EXPECTED:
        raise RuntimeError("Event count deviates from the prespecified 35")

    tables = {}
    for branch, label in (("genomic", "Genomic"), ("clinical", "Clinical")):
        table = build_frequency_table(fold_sel, fresh_sel, branch, total_events)
        table.to_csv(GENOMIC_CSV if branch == "genomic" else CLINICAL_CSV, index=False)
        tables[branch] = table
        stable = int(table["stable_at_0.70"].sum())
        logger.info(
            "%s stability: %d unique features, %d stable at %.2f (max %.2f)",
            branch, len(table), stable, STABILITY_THRESHOLD, table["stability"].max(),
        )
        plot_stability(
            table if branch == "genomic" else table,
            FIGURE_PNG if branch == "genomic" else FIGURE_PNG.with_name(
                f"feature_stability_{branch}.png"
            ),
            label,
        )

    summary = {
        "rules_prespecified": {
            "events": "15 nested-CV fold selections + 20 full-train runs (seeds 500-519)",
            "fresh_run_seeds": list(FRESH_RUN_SEEDS),
            "stability_definition": "n_selected / n_events",
            "n_events": total_events,
            "stable_threshold": STABILITY_THRESHOLD,
            "frozen_before_running": True,
        },
        "genomic": {
            "n_unique_features": int(len(tables["genomic"])),
            "n_stable_at_0.70": int(tables["genomic"]["stable_at_0.70"].sum()),
            "top10": tables["genomic"].head(10)[
                ["feature", "stability", "mean_rank_when_selected"]
            ].to_dict(orient="records"),
        },
        "clinical": {
            "n_unique_features": int(len(tables["clinical"])),
            "n_stable_at_0.70": int(tables["clinical"]["stable_at_0.70"].sum()),
            "top10": tables["clinical"].head(10)[
                ["feature", "stability", "mean_rank_when_selected"]
            ].to_dict(orient="records"),
        },
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Step 7: feature-selection stability ===")
    print(f"Events: {total_events} per branch | threshold {STABILITY_THRESHOLD}")
    print(f"Genomic:  {summary['genomic']['n_unique_features']} unique, "
          f"{summary['genomic']['n_stable_at_0.70']} stable")
    print(f"Clinical: {summary['clinical']['n_unique_features']} unique, "
          f"{summary['clinical']['n_stable_at_0.70']} stable")
    print("Top-10 genomic:")
    for row in summary["genomic"]["top10"]:
        print(f"  {row['feature']:<12} stability={row['stability']:.2f} "
              f"(mean rank {row['mean_rank_when_selected']})")
    print(f"Runtime: {summary['runtime_minutes']} min")


if __name__ == "__main__":
    main()
