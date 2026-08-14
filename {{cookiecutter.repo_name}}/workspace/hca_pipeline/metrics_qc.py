"""Quality-control metrics for Live Cell Painting profiles.

Ports the logic of ``05_quality_metrics.ipynb`` (per-plate QC, cross-plate
batch assessment, dose-response consistency, treatment-vs-control separation,
optional time-course analysis, and the Go/No-Go dashboard) into reusable,
notebook-agnostic functions.

Two independent implementations of the "replicate consistency" question are
provided, matching the source notebook's dual-implementation design:

- **From-scratch** (Pearson correlation): :func:`percent_replicating`,
  :func:`percent_matching`, :func:`mean_average_precision`. Transparent and
  dependency-free (pandas/numpy only).
- **copairs-based** (cosine distance, FDR-corrected p-values):
  :func:`copairs_compute_map` and friends. Requires the optional ``copairs``
  dependency, imported lazily inside each function so importing this module
  never requires copairs to be installed.

Design notes
------------
This module intentionally accepts every column name (treatment, plate,
control, dose, time) as an explicit parameter rather than relying on
module-level constants or hardcoded literals such as ``"Metadata_Treatment"``.
The source notebook hardcoded the literal string ``"Metadata_Treatment"`` in
several places when indexing into a ``copairs`` mAP result, even though the
actual column name used for grouping (``pos_sameby``) was a caller-supplied
variable that is not always literally named ``Metadata_Treatment``. Because
``copairs`` names its output metadata columns after whatever ``pos_sameby``
columns were passed to it, that hardcoding silently breaks for any dataset
whose treatment column has a different name. Every such lookup here uses the
resolved ``treatment_col`` argument instead.

Two dead-code reconciliations relative to the source notebook:

- ``percent_matching`` (PM) was defined but never called in the source
  notebook, despite the notebook's own docs framing PR/PM/mAP as "the three
  metrics". It is kept here as a first-class public function and is now
  actually wired into :func:`run_per_plate_qc` (see its ``pm_results``
  output), since it is cheap to compute from the same correlation matrix
  already built for PR.
- ``copairs_per_plate_map`` (a standalone "loop over plates, call
  copairs_compute_map" helper) was defined but never called — the source
  notebook's ``run_per_plate_qc`` reimplements an equivalent per-plate loop
  inline because it needs to interleave the from-scratch and copairs mAP
  computations within the same per-plate iteration. Re-adding a separate
  ``copairs_per_plate_map`` on top of that would just be a second way to do
  the same loop, so it was dropped rather than ported.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu  # noqa: F401  (kept for API parity / future use)

__all__ = [
    "detect_time_column",
    "resolve_negcon_values",
    "compute_pairwise_correlations",
    "null_distribution_batch_aware",
    "percent_replicating",
    "percent_matching",
    "average_precision_for_query",
    "mean_average_precision",
    "copairs_compute_map",
    "copairs_treatment_vs_control_map",
    "copairs_dose_response_map",
    "detect_dose_response_treatments",
    "crossvalidate_scratch_vs_copairs_map",
    "run_per_plate_qc",
    "plot_per_plate_qc",
    "run_cross_plate_batch",
    "plot_cross_plate_batch",
    "dose_response_monotonicity_spearman",
    "run_dose_response",
    "plot_dose_response",
    "EFFECT_DETECTABILITY_QUADRANTS",
    "classify_effect_detectability",
    "run_treatment_vs_control",
    "plot_treatment_vs_control",
    "run_time_course",
    "plot_time_course",
    "generate_go_nogo_dashboard",
    "plot_go_nogo_dashboard",
]


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Column / vocabulary auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_time_column(
    df: pd.DataFrame,
    time_keywords: Sequence[str] = ("time", "hour", "day", "exposure", "duration", "hr", "hpi"),
) -> tuple[str | None, int]:
    """Auto-detect a time/exposure column among ``Metadata_*`` columns.

    Returns ``(column_name, n_unique)``, or ``(None, 0)`` if no keyword-matching
    column with more than one unique value is found.
    """
    for col in df.columns:
        if col.startswith("Metadata_"):
            col_lower = col.lower()
            if any(kw in col_lower for kw in time_keywords):
                n_unique = df[col].nunique()
                if n_unique > 1:
                    return col, n_unique
    return None, 0


def resolve_negcon_values(
    df: pd.DataFrame,
    control_col: str | None,
    negcon_values: Sequence[str] | None,
    poscon_value: str,
    trt_value: str,
) -> list[str]:
    """Resolve the negative-control vocabulary.

    If ``negcon_values`` is ``None``, auto-detect it as "all control types
    that are neither poscon nor trt", matching the source notebook's
    auto-detection convention. Falls back to ``["negcon"]`` if no control
    column is available.
    """
    if negcon_values is not None:
        return list(negcon_values)
    if control_col and control_col in df.columns:
        all_ctrl_types = set(df[control_col].dropna().unique())
        return sorted(all_ctrl_types - {poscon_value, trt_value})
    return ["negcon"]


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — From-scratch metric implementations
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_correlations(profiles: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation matrix; diagonal set to ``-inf``."""
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = centered / norms
    corr = normalized @ normalized.T
    np.fill_diagonal(corr, -np.inf)
    return corr


def null_distribution_batch_aware(
    profiles: np.ndarray,
    labels: np.ndarray,
    plate_labels: np.ndarray | None = None,
    n_pairs: int = 10000,
    seed: int = 42,
    same_plate_null: bool = False,
) -> np.ndarray:
    """Build a null distribution from random cross-treatment pairs.

    ``same_plate_null=True`` restricts null pairs to the same plate (for a
    within-plate null); ``False`` allows cross-plate pairs (global/cross-plate
    null).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels = np.asarray(labels)

    null_corrs: list[float] = []
    attempts = 0
    max_attempts = n_pairs * 20

    while len(null_corrs) < n_pairs and attempts < max_attempts:
        i, j = rng.randint(0, n, size=2)
        attempts += 1
        if i == j or labels[i] == labels[j]:
            continue
        if plate_labels is not None:
            if same_plate_null and plate_labels[i] != plate_labels[j]:
                continue
            if not same_plate_null and plate_labels[i] == plate_labels[j]:
                continue

        v1 = profiles[i] - profiles[i].mean()
        v2 = profiles[j] - profiles[j].mean()
        if np.std(v1) > 0 and np.std(v2) > 0:
            null_corrs.append(np.corrcoef(v1, v2)[0, 1])

    return np.array(null_corrs)


def percent_replicating(
    profiles: np.ndarray,
    labels: np.ndarray,
    plate_labels: np.ndarray | None = None,
    null_percentile: float = 95,
    n_null: int = 10000,
    seed: int = 42,
    batch_mode: str = "global",
) -> tuple[float, dict[Any, float], np.ndarray, float]:
    """Compute Percent Replicating (PR).

    ``batch_mode``:
      - ``"global"`` — all replicate pairs are positives, null is cross-treatment.
      - ``"within"`` — per-plate analysis (call separately per plate).
      - ``"cross"``  — only cross-plate replicate pairs are positives.

    Returns ``(pr_value, per_treatment_medians, null_distribution, threshold)``.
    """
    labels = np.asarray(labels)
    corr = compute_pairwise_correlations(profiles)

    null_dist = null_distribution_batch_aware(
        profiles, labels, plate_labels, n_null, seed, same_plate_null=False,
    )
    threshold = np.percentile(null_dist, null_percentile)

    unique_labels = np.unique(labels)
    per_treatment: dict[Any, float] = {}

    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]

        if len(indices) < 2:
            per_treatment[label] = np.nan
            continue

        if batch_mode == "cross" and plate_labels is not None:
            rep_corrs = []
            for idx_i in range(len(indices)):
                for idx_j in range(idx_i + 1, len(indices)):
                    i, j = indices[idx_i], indices[idx_j]
                    if plate_labels[i] != plate_labels[j]:
                        rep_corrs.append(corr[i, j])
        else:
            sub_corr = corr[np.ix_(indices, indices)]
            upper_tri = sub_corr[np.triu_indices(len(indices), k=1)]
            rep_corrs = upper_tri[upper_tri != -np.inf]

        per_treatment[label] = np.median(rep_corrs) if len(rep_corrs) > 0 else np.nan

    valid = [v for v in per_treatment.values() if not np.isnan(v)]
    pr_value = np.mean([v > threshold for v in valid]) if valid else 0.0

    return pr_value, per_treatment, null_dist, threshold


def percent_matching(
    profiles: np.ndarray,
    labels: np.ndarray,
    plate_labels: np.ndarray | None = None,
    exclude_same_plate: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute Percent Matching (PM) — top-1 nearest-neighbor retrieval.

    Kept as a first-class public function (the source notebook defines it but
    never calls it, despite framing PR/PM/mAP as its "three metrics"). It is
    cheap to compute from the same correlation matrix as PR/mAP and is wired
    into :func:`run_per_plate_qc` below.

    Returns ``(pm_value, match_details)``.
    """
    labels = np.asarray(labels)
    corr = compute_pairwise_correlations(profiles)
    n = len(labels)

    matches = []
    for i in range(n):
        if exclude_same_plate and plate_labels is not None:
            mask = (np.arange(n) != i) & (plate_labels != plate_labels[i])
        else:
            mask = np.arange(n) != i

        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            continue

        best_idx = candidates[np.argmax(corr[i, candidates])]
        matches.append({
            "query_idx": i,
            "query_label": labels[i],
            "match_idx": best_idx,
            "match_label": labels[best_idx],
            "correlation": corr[i, best_idx],
            "is_correct": labels[i] == labels[best_idx],
        })

    pm_value = np.mean([m["is_correct"] for m in matches]) if matches else 0.0
    return pm_value, matches


def average_precision_for_query(query_idx: int, corr_row: np.ndarray, labels: np.ndarray) -> float:
    """Average Precision for a single query, given a precomputed correlation row."""
    query_label = labels[query_idx]
    n = len(labels)

    mask = np.arange(n) != query_idx
    candidates = np.where(mask)[0]
    candidate_labels = labels[candidates]
    candidate_sims = corr_row[candidates]

    sorted_idx = np.argsort(-candidate_sims)
    sorted_labels = candidate_labels[sorted_idx]

    n_pos = (sorted_labels == query_label).sum()
    if n_pos == 0:
        return 0.0

    precision_at_k = []
    n_correct = 0
    for k, label in enumerate(sorted_labels, 1):
        if label == query_label:
            n_correct += 1
            precision_at_k.append(n_correct / k)

    return float(np.mean(precision_at_k)) if precision_at_k else 0.0


def mean_average_precision(
    profiles: np.ndarray,
    labels: np.ndarray,
    plate_labels: np.ndarray | None = None,
    exclude_same_plate: bool = False,
) -> tuple[float, dict[Any, float], list[float]]:
    """Phenotypic-consistency mAP (PC mAP) — full-ranking replicate retrieval.

    Returns ``(overall_map, per_treatment_dict, per_profile_list)``.
    Note: ``overall_map`` is micro-averaged (mean of all per-profile APs);
    copairs reports a macro-average (mean of per-treatment means).
    """
    labels = np.asarray(labels)
    corr = compute_pairwise_correlations(profiles)
    n = len(labels)

    per_treatment_ap: dict[Any, list[float]] = {}
    all_aps: list[float] = []

    for i in range(n):
        query_label = labels[i]

        if exclude_same_plate and plate_labels is not None:
            mask = (np.arange(n) != i) & (plate_labels != plate_labels[i])
        else:
            mask = np.arange(n) != i

        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            continue

        ap = average_precision_for_query(i, corr[i], labels)
        all_aps.append(ap)
        per_treatment_ap.setdefault(query_label, []).append(ap)

    per_treatment_mean = {k: float(np.mean(v)) for k, v in per_treatment_ap.items()}
    overall_map = float(np.mean(all_aps)) if all_aps else 0.0

    return overall_map, per_treatment_mean, all_aps


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — copairs library wrappers
# ─────────────────────────────────────────────────────────────────────────────

def copairs_compute_map(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    pos_sameby: Sequence[str],
    pos_diffby: Sequence[str] | None = None,
    neg_sameby: Sequence[str] | None = None,
    neg_diffby: Sequence[str] | None = None,
    null_size: int = 10000,
    seed: int = 42,
    distance: str = "cosine",
    batch_size: int = 20000,
) -> pd.DataFrame:
    """Compute mAP using copairs with configurable pair definitions.

    The caller must qualify the result from those definitions: same-treatment
    positives measure phenotypic consistency (PC), treatment-vs-control
    reference retrieval measures phenotypic activity, and same-MoA positives
    would measure phenotypic distinctiveness.

    Returns a DataFrame with ``mean_average_precision``, ``p_value``, and
    ``corrected_p_value``, plus one metadata column per entry in
    ``pos_sameby`` (named after the actual grouping column, e.g. whatever the
    caller's ``treatment_col`` is — never a hardcoded ``"Metadata_Treatment"``).
    """
    from copairs import map as copairs_map

    pos_diffby = list(pos_diffby) if pos_diffby else []
    neg_sameby = list(neg_sameby) if neg_sameby else []
    neg_diffby = list(neg_diffby) if neg_diffby else []
    pos_sameby = list(pos_sameby)

    meta_cols = list(set(
        [c for c in df.columns if c.startswith("Metadata_")]
        + pos_sameby + pos_diffby + neg_sameby + neg_diffby
    ))
    meta = df[meta_cols].copy()
    feats = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values

    ap_scores = copairs_map.average_precision(
        meta=meta, feats=feats,
        pos_sameby=pos_sameby, pos_diffby=pos_diffby,
        neg_sameby=neg_sameby, neg_diffby=neg_diffby,
        batch_size=batch_size, distance=distance,
        progress_bar=False,
    )
    map_scores = copairs_map.mean_average_precision(
        ap_scores=ap_scores, sameby=pos_sameby,
        null_size=null_size, threshold=0.05, seed=seed,
        cache_dir=os.environ.get("COPAIRS_CACHE_DIR"),
        progress_bar=False,
    )
    return copairs_map.apply_fdr_correction(map_scores)


def copairs_treatment_vs_control_map(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    treatment_col: str,
    control_col: str,
    negcon_values: str | Sequence[str],
    plate_col: str | None = None,
    null_size: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phenotypic-activity mAP using copairs treatment-vs-control indexing.

    Each negcon well gets a unique reference index; treatments are paired
    against their matched negcon reference.
    """
    from copairs import map as copairs_map
    from copairs.matching import assign_reference_index

    if isinstance(negcon_values, str):
        negcon_values = [negcon_values]

    df_work = df.copy()
    cond_parts = [f"{control_col} == '{v}'" for v in negcon_values]
    condition = " | ".join(cond_parts)

    df_work = assign_reference_index(
        df_work, condition=condition,
        reference_col="Metadata_Reference_Index",
        default_value=-1,
    )

    meta_cols = list(set(
        [c for c in df_work.columns if c.startswith("Metadata_")]
        + [treatment_col, control_col, "Metadata_Reference_Index"]
        + ([plate_col] if plate_col else [])
    ))
    meta = df_work[meta_cols].copy()
    feats = df_work[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values

    ap_scores = copairs_map.average_precision(
        meta=meta, feats=feats,
        pos_sameby=[treatment_col],
        pos_diffby=[],
        neg_sameby=["Metadata_Reference_Index"],
        neg_diffby=[treatment_col],
        batch_size=20000, distance="cosine",
        progress_bar=False,
    )
    map_scores = copairs_map.mean_average_precision(
        ap_scores=ap_scores, sameby=[treatment_col],
        null_size=null_size, threshold=0.05, seed=seed,
        cache_dir=os.environ.get("COPAIRS_CACHE_DIR"),
        progress_bar=False,
    )
    map_scores = copairs_map.apply_fdr_correction(map_scores)
    return map_scores, ap_scores


def detect_dose_response_treatments(
    df: pd.DataFrame, treatment_col: str, dose_col: str, min_doses: int = 2,
) -> tuple[list[Any], pd.Series]:
    """Auto-detect treatments with at least ``min_doses`` distinct doses."""
    dose_counts = df.groupby(treatment_col)[dose_col].nunique()
    multi_dose = dose_counts[dose_counts >= min_doses].index.tolist()
    return multi_dose, dose_counts


def copairs_dose_response_map(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    treatment_col: str,
    dose_col: str,
    null_size: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame | None, list]:
    """Dose-level phenotypic-consistency mAP (PC mAP).

    Groups by a composite ``treatment_dose`` label (not the bare treatment
    column), so the resulting mAP table's grouping column is intentionally
    a synthetic ``"Metadata_Treatment_Dose"`` column rather than
    ``treatment_col`` itself — this is not an instance of the
    ``'Metadata_Treatment'``-literal bug since it is genuinely the column
    that was grouped by.
    """
    from copairs import map as copairs_map

    multi_dose, _ = detect_dose_response_treatments(df, treatment_col, dose_col)
    if not multi_dose:
        return None, []

    df_dose = df[df[treatment_col].isin(multi_dose)].copy()
    df_dose["Metadata_Treatment_Dose"] = (
        df_dose[treatment_col].astype(str) + "_" + df_dose[dose_col].astype(str)
    )

    meta_cols = list(set(
        [c for c in df_dose.columns if c.startswith("Metadata_")]
        + [treatment_col, dose_col, "Metadata_Treatment_Dose"]
    ))
    meta = df_dose[meta_cols].copy()
    feats = df_dose[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values

    ap_scores = copairs_map.average_precision(
        meta=meta, feats=feats,
        pos_sameby=["Metadata_Treatment_Dose"],
        pos_diffby=[],
        neg_sameby=[treatment_col],
        neg_diffby=[dose_col],
        batch_size=20000, distance="cosine",
        progress_bar=False,
    )
    map_scores = copairs_map.mean_average_precision(
        ap_scores=ap_scores, sameby=["Metadata_Treatment_Dose"],
        null_size=null_size, threshold=0.05, seed=seed,
        cache_dir=os.environ.get("COPAIRS_CACHE_DIR"),
        progress_bar=False,
    )
    map_scores = copairs_map.apply_fdr_correction(map_scores)
    return map_scores, ap_scores


def crossvalidate_scratch_vs_copairs_map(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    treatment_col: str,
    null_size: int = 10000,
    seed: int = 42,
    agreement_threshold: float = 0.02,
) -> dict[str, Any]:
    """Compare the from-scratch (Pearson) global mAP against two copairs runs.

    This intentionally separates two different questions that the source
    notebook's single Pearson-vs-cosine comparison conflated:

    1. **Implementation check** (``distance="correlation"``, i.e. Pearson —
       the same similarity metric as the from-scratch implementation). Since
       both sides rank neighbors identically, a disagreement beyond
       ``agreement_threshold`` here is a genuine signal that one of the two
       implementations has a bug — this is the only one of the two with a
       pass/fail threshold.
    2. **Sensitivity analysis** (``distance="cosine"``, copairs' default). This
       uses a *different* similarity metric than the from-scratch implementation
       by design, so Pearson-correlation-based and cosine-based neighbor
       rankings are expected to disagree even when both implementations are
       correct — there is no pass/fail threshold for this comparison, since a
       fixed agreement threshold would conflate "different similarity metric,
       as designed" with "implementation bug."

    Both copairs runs also differ from the from-scratch result in aggregation
    (copairs reports a macro-average across treatments; from-scratch reports a
    micro-average across all profiles) — a small, expected difference in the
    summary statistic, not in the per-treatment AP values themselves.

    Ports the source notebook's Section 3 cross-validation cell. That cell
    keyed into the copairs result with the literal string
    ``'Metadata_Treatment'`` regardless of the actual treatment column name;
    here we key with the resolved ``treatment_col`` instead.
    """
    profiles = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    labels = df[treatment_col].values

    map_fs, per_treat_fs, _ = mean_average_precision(profiles, labels, plate_labels=None)

    def _copairs_map_by_treatment(distance: str) -> tuple[dict[Any, float], float]:
        result = copairs_compute_map(
            df, feat_cols,
            pos_sameby=[treatment_col],
            neg_diffby=[treatment_col],
            null_size=null_size, seed=seed, distance=distance,
        )
        return (
            dict(zip(result[treatment_col], result["mean_average_precision"])),
            float(result["mean_average_precision"].mean()),
        )

    def _per_treatment_diff(cp_dict: dict[Any, float]) -> tuple[dict[Any, dict[str, float]], float]:
        per_treatment = {}
        max_diff = 0.0
        for trt, fs in per_treat_fs.items():
            cp = cp_dict.get(trt, np.nan)
            diff = cp - fs
            per_treatment[trt] = {"from_scratch": fs, "copairs": cp, "diff": diff}
            if np.isfinite(diff):
                max_diff = max(max_diff, abs(diff))
        return per_treatment, max_diff

    cp_pearson_dict, map_cp_pearson_macro = _copairs_map_by_treatment("correlation")
    cp_cosine_dict, map_cp_cosine_macro = _copairs_map_by_treatment("cosine")

    implementation_per_treatment, implementation_max_diff = _per_treatment_diff(cp_pearson_dict)
    sensitivity_per_treatment, sensitivity_max_diff = _per_treatment_diff(cp_cosine_dict)

    return {
        "map_from_scratch_micro": map_fs,
        "implementation_check": {
            "distance": "correlation",
            "map_copairs_macro": map_cp_pearson_macro,
            "per_treatment": implementation_per_treatment,
            "max_abs_diff": implementation_max_diff,
            "agrees": implementation_max_diff <= agreement_threshold,
            "agreement_threshold": agreement_threshold,
        },
        "sensitivity_analysis": {
            "distance": "cosine",
            "map_copairs_macro": map_cp_cosine_macro,
            "per_treatment": sensitivity_per_treatment,
            "max_abs_diff": sensitivity_max_diff,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Application A: per-plate QC
# ─────────────────────────────────────────────────────────────────────────────

def run_per_plate_qc(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    treatment_col: str,
    plate_col: str,
    control_col: str,
    negcon_values: str | Sequence[str],
    poscon_values: str | Sequence[str],
    null_percentile: float = 95,
    n_null: int = 10000,
    seed: int = 42,
    null_size_copairs: int = 5000,
    negcon_pr_threshold: float = 0.10,
    poscon_pr_threshold: float = 0.50,
    pr_fraction_threshold: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Application A: does each plate independently show replicate consistency?

    Computes within-plate PR (from-scratch), within-plate PM, and within-plate
    mAP (from-scratch + copairs, cross-validated) per treatment, plus SC-19
    (negcon/poscon PR check) and a PR-fraction QC flag, per plate.
    """
    if isinstance(negcon_values, str):
        negcon_values = [negcon_values]
    if isinstance(poscon_values, str):
        poscon_values = [poscon_values]

    plates = sorted(df[plate_col].unique())
    all_pr, all_map, all_pm, all_nulls, qc_flags, sc19_results = [], [], [], {}, [], []

    for plate in plates:
        df_plate = df[df[plate_col] == plate].copy()
        profiles = df_plate[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        labels = df_plate[treatment_col].values
        trt_to_ctrl = df_plate.groupby(treatment_col)[control_col].first().to_dict()

        # Within-plate PR
        pr_value, pr_medians, null_dist, threshold = percent_replicating(
            profiles, labels, plate_labels=None,
            null_percentile=null_percentile, n_null=n_null, seed=seed,
            batch_mode="global",
        )

        for trt, median_corr in pr_medians.items():
            all_pr.append({
                plate_col: plate, "treatment": trt,
                "control_type": trt_to_ctrl.get(trt, "unknown"),
                "median_corr": median_corr, "null_threshold": threshold,
                "passes_pr": median_corr > threshold,
                "n_replicates": int((labels == trt).sum()),
            })
        all_nulls[plate] = {"threshold": threshold, "null_dist": null_dist}

        # Within-plate PM
        pm_value, _ = percent_matching(profiles, labels, plate_labels=None)
        all_pm.append({plate_col: plate, "pm_value": pm_value})

        # Within-plate mAP (from-scratch + copairs, cross-validated)
        map_fs, per_treat_fs, _ = mean_average_precision(profiles, labels, plate_labels=None)
        map_cp = copairs_compute_map(
            df_plate, feat_cols, pos_sameby=[treatment_col],
            neg_diffby=[treatment_col], null_size=null_size_copairs, seed=seed,
        )
        # BUG FIX: source notebook keyed with literal 'Metadata_Treatment';
        # copairs names this column after `pos_sameby`, i.e. `treatment_col`.
        cp_dict = dict(zip(map_cp[treatment_col], map_cp["mean_average_precision"]))
        cp_pval = dict(zip(map_cp[treatment_col], map_cp["corrected_p_value"]))

        for trt in sorted(per_treat_fs.keys()):
            all_map.append({
                plate_col: plate, "treatment": trt,
                "control_type": trt_to_ctrl.get(trt, "unknown"),
                "mAP_from_scratch": per_treat_fs[trt],
                "mAP_copairs": cp_dict.get(trt, np.nan),
                "mAP_diff": cp_dict.get(trt, np.nan) - per_treat_fs[trt],
                "copairs_fdr_significant": cp_pval.get(trt, np.nan) < 0.05,
            })

        # QC flags
        n_treatments = len(pr_medians)
        n_passing = sum(1 for m in pr_medians.values() if m > threshold)
        negcon_prs = [m for t, m in pr_medians.items() if trt_to_ctrl.get(t) in negcon_values]
        poscon_prs = [m for t, m in pr_medians.items() if trt_to_ctrl.get(t) in poscon_values]

        # Missing controls are unavailable evidence, never an automatic PASS.
        # Treating an empty list as True (Python's ``all([])`` behavior) can
        # incorrectly approve a plate whose platemap/control vocabulary is
        # incomplete.
        negcon_pass = bool(negcon_prs) and all(
            pr <= negcon_pr_threshold for pr in negcon_prs
        )
        poscon_pass = bool(poscon_prs) and all(
            pr > poscon_pr_threshold for pr in poscon_prs
        )

        sc19_results.append({
            plate_col: plate,
            "negcon_pr_max": max(negcon_prs) if negcon_prs else None,
            "negcon_available": bool(negcon_prs),
            "negcon_pass": negcon_pass,
            "poscon_pr_min": min(poscon_prs) if poscon_prs else None,
            "poscon_available": bool(poscon_prs),
            "poscon_pass": poscon_pass,
            "sc19_pass": negcon_pass and poscon_pass,
        })
        qc_flags.append({
            plate_col: plate, "n_treatments": n_treatments,
            "n_passing_pr": n_passing,
            "pr_fraction": n_passing / n_treatments if n_treatments > 0 else 0,
            "null_threshold": threshold,
            "plate_pass": (n_passing / n_treatments if n_treatments > 0 else 0) >= pr_fraction_threshold,
        })

    return {
        "pr_results": pd.DataFrame(all_pr),
        "pm_results": pd.DataFrame(all_pm),
        "map_results": pd.DataFrame(all_map),
        "null_data": all_nulls,
        "qc_flags": pd.DataFrame(qc_flags),
        "sc19_status": pd.DataFrame(sc19_results),
    }


def plot_per_plate_qc(qc_result: dict, plate_col: str, output_path: str | None = None):
    """Visualize Application A results: per-plate PR bars, null distributions, mAP cross-validation."""
    import os

    import matplotlib.pyplot as plt

    pr_df = qc_result["pr_results"]
    map_df = qc_result["map_results"]
    plates = sorted(pr_df[plate_col].unique())
    n_plates = len(plates)
    colors = ["#0279EE", "#FF9400", "#75A025", "#FD9BED"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    treatments = sorted(pr_df["treatment"].unique())
    x = np.arange(len(treatments))
    bar_width = 0.8 / max(n_plates, 1)
    for i, plate in enumerate(plates):
        pd_plate = pr_df[pr_df[plate_col] == plate].set_index("treatment")
        vals = [pd_plate.loc[t, "median_corr"] if t in pd_plate.index else 0 for t in treatments]
        label = plate.split("_Plate_")[0] if "_Plate_" in str(plate) else plate
        ax.bar(x + i * bar_width, vals, bar_width, label=label, color=colors[i % len(colors)], alpha=0.8)
    for i, plate in enumerate(plates):
        thr = qc_result["qc_flags"][qc_result["qc_flags"][plate_col] == plate]["null_threshold"].iloc[0]
        ax.axhline(y=thr, color=colors[i % len(colors)], linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(x + bar_width * (n_plates - 1) / 2)
    ax.set_xticklabels(treatments, rotation=45, ha="right")
    ax.set_ylabel("Median pairwise correlation")
    ax.set_title("Per-plate Percent Replicating (PR)")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax = axes[1]
    for i, plate in enumerate(plates):
        null_dist = qc_result["null_data"][plate]["null_dist"]
        threshold = qc_result["null_data"][plate]["threshold"]
        label = plate.split("_Plate_")[0] if "_Plate_" in str(plate) else plate
        ax.hist(null_dist, bins=50, alpha=0.3, color=colors[i % len(colors)], label=f"{label} (thr={threshold:.3f})")
        ax.axvline(x=threshold, color=colors[i % len(colors)], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Null correlation")
    ax.set_ylabel("Count")
    ax.set_title("Null Distributions (95th percentile threshold)")
    ax.legend(fontsize=8)

    ax = axes[2]
    sig_colors = ["#0279EE" if p else "#FF9400" for p in map_df["copairs_fdr_significant"]]
    ax.scatter(map_df["mAP_from_scratch"], map_df["mAP_copairs"], c=sig_colors, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("PC mAP (from-scratch, Pearson)")
    ax.set_ylabel("PC mAP (copairs, cosine)")
    ax.set_title("Phenotypic-consistency mAP cross-validation\n(blue=FDR sig)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Application B: cross-plate batch assessment
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_plate_batch(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    treatment_col: str,
    plate_col: str,
    control_col: str,
    negcon_values: str | Sequence[str],
    poscon_values: str | Sequence[str],
    null_percentile: float = 95,
    n_null: int = 10000,
    seed: int = 42,
    null_size_copairs: int = 5000,
    batch_drop_threshold: float = 0.30,
    min_within_pr_for_flag: float = 0.30,
) -> dict[str, Any]:
    """Application B: do replicates agree across plates? Is there a batch effect?

    Returns ``{"message": ..., "n_plates": 1}`` when fewer than 2 plates are
    present (cross-plate comparison is undefined for a single plate).
    """
    plates = sorted(df[plate_col].unique())
    if len(plates) < 2:
        return {"message": "Cross-plate analysis skipped: only 1 plate present", "n_plates": 1}

    if isinstance(negcon_values, str):
        negcon_values = [negcon_values]
    if isinstance(poscon_values, str):
        poscon_values = [poscon_values]

    profiles = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    labels = df[treatment_col].values
    plate_labels = df[plate_col].values
    trt_to_ctrl = df.groupby(treatment_col)[control_col].first().to_dict()

    # Within-plate PR (average across plates)
    within_plate_medians: dict[Any, list[float]] = {}
    for plate in plates:
        df_p = df[df[plate_col] == plate]
        prof_p = df_p[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        _, pr_meds, _, _ = percent_replicating(
            prof_p, df_p[treatment_col].values,
            plate_labels=None, null_percentile=null_percentile, n_null=n_null, seed=seed, batch_mode="global",
        )
        for trt, med in pr_meds.items():
            within_plate_medians.setdefault(trt, []).append(med)
    within_plate_avg = {t: float(np.mean(v)) for t, v in within_plate_medians.items()}

    # Cross-plate PR
    pr_cross, pr_cross_meds, null_cross, thr_cross = percent_replicating(
        profiles, labels, plate_labels=plate_labels,
        null_percentile=null_percentile, n_null=n_null, seed=seed, batch_mode="cross",
    )

    # Cross-plate mAP (copairs)
    map_cross = copairs_compute_map(
        df, feat_cols, pos_sameby=[treatment_col],
        pos_diffby=[plate_col], neg_diffby=[treatment_col], null_size=null_size_copairs, seed=seed,
    )

    # Within-plate mAP (copairs, averaged across plates)
    within_map: dict[Any, list[float]] = {}
    for plate in plates:
        df_p = df[df[plate_col] == plate]
        m = copairs_compute_map(
            df_p, feat_cols, pos_sameby=[treatment_col],
            neg_diffby=[treatment_col], null_size=null_size_copairs, seed=seed,
        )
        # BUG FIX: source notebook keyed with literal 'Metadata_Treatment'.
        for _, row in m.iterrows():
            within_map.setdefault(row[treatment_col], []).append(row["mean_average_precision"])
    within_map_avg = {t: float(np.mean(v)) for t, v in within_map.items()}

    # BUG FIX: source notebook keyed with literal 'Metadata_Treatment'.
    cp_map_dict = dict(zip(map_cross[treatment_col], map_cross["mean_average_precision"]))

    batch_results = []
    for trt in sorted(within_plate_avg.keys()):
        wp = within_plate_avg[trt]
        cp = pr_cross_meds.get(trt, np.nan)
        pr_drop = (wp - cp) / wp if wp > 0 else 0.0
        wm = within_map_avg.get(trt, np.nan)
        cm = cp_map_dict.get(trt, np.nan)
        map_drop = (wm - cm) / wm if wm > 0 else 0.0
        batch_results.append({
            "treatment": trt, "control_type": trt_to_ctrl.get(trt, "unknown"),
            "within_plate_pr": wp, "cross_plate_pr": cp, "pr_drop_pct": pr_drop * 100,
            "within_plate_map": wm, "cross_plate_map": cm, "map_drop_pct": map_drop * 100,
            "batch_effect": pr_drop > batch_drop_threshold,
            "batch_effect_meaningful": pr_drop > batch_drop_threshold and wp > min_within_pr_for_flag,
        })

    batch_df = pd.DataFrame(batch_results)
    n_meaningful = int(batch_df["batch_effect_meaningful"].sum())

    return {
        "n_plates": len(plates), "within_plate_pr": within_plate_avg,
        "cross_plate_pr": pr_cross_meds, "cross_plate_pr_value": pr_cross,
        "cross_plate_threshold": thr_cross, "batch_assessment": batch_df,
        "cross_plate_map": map_cross,
        "sc20_status": {
            "n_batch_affected": n_meaningful,
            "batch_affected_treatments": batch_df[batch_df["batch_effect_meaningful"]]["treatment"].tolist(),
            "sc20_pass": n_meaningful == 0,
            "threshold_pct": batch_drop_threshold * 100,
            "note": f"Only flags treatments with within-plate PR > {min_within_pr_for_flag}",
        },
    }


def plot_cross_plate_batch(batch_result: dict, output_path: str | None = None):
    """Visualize Application B results. No-op (returns ``None``) when skipped."""
    import os

    import matplotlib.pyplot as plt

    if "message" in batch_result:
        return None

    batch_df = batch_result["batch_assessment"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    treatments = batch_df["treatment"].tolist()
    x = np.arange(len(treatments))

    ax = axes[0]
    width = 0.35
    ax.bar(x - width / 2, batch_df["within_plate_pr"], width, label="Within-plate", color="#0279EE", alpha=0.8)
    ax.bar(x + width / 2, batch_df["cross_plate_pr"], width, label="Cross-plate", color="#FF9400", alpha=0.8)
    for i, row in batch_df.iterrows():
        if row["batch_effect_meaningful"]:
            ax.annotate("*", (x[i] + width / 2, row["cross_plate_pr"] + 0.02), ha="center", fontsize=14, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=45, ha="right")
    ax.set_ylabel("Median pairwise correlation")
    ax.set_title("Within vs Cross-plate PR\n(* = meaningful batch effect)")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax = axes[1]
    colors = ["#FF9400" if b else "#75A025" for b in batch_df["batch_effect_meaningful"]]
    ax.barh(x, batch_df["pr_drop_pct"], color=colors, alpha=0.8)
    ax.axvline(x=30, color="red", linestyle="--", linewidth=1.5, label="30% threshold")
    ax.set_yticks(x)
    ax.set_yticklabels(treatments)
    ax.set_xlabel("PR drop (%)")
    ax.set_title("Cross-plate PR Drop")
    ax.legend()

    ax = axes[2]
    ax.scatter(
        batch_df["within_plate_map"], batch_df["cross_plate_map"],
        c=["#FF9400" if b else "#0279EE" for b in batch_df["batch_effect_meaningful"]],
        s=100, alpha=0.7, edgecolors="black", linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    for _, row in batch_df.iterrows():
        ax.annotate(str(row["treatment"])[:8], (row["within_plate_map"], row["cross_plate_map"]), fontsize=7, alpha=0.8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Within-plate PC mAP")
    ax.set_ylabel("Cross-plate PC mAP")
    ax.set_title("Phenotypic consistency: within vs cross-plate")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Application C: dose-response consistency
# ─────────────────────────────────────────────────────────────────────────────

def dose_response_monotonicity_spearman(
    df_trt: pd.DataFrame,
    feat_cols: Sequence[str],
    dose_col: str,
    doses: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    rho_no_structure_band: float = 0.1,
) -> dict[str, Any]:
    """Continuous complement to SC-21's binary adjacent-vs-non-adjacent test.

    SC-21 asks a binary question (is mean adjacent-dose similarity greater
    than mean non-adjacent-dose similarity?) that can flip PASS/FAIL from a
    small numerical wobble when only a few dose levels are available. This
    computes the Spearman correlation between dose separation
    (``|dose_i - dose_j|``) and phenotypic distance (``1 - Pearson
    correlation`` between the two doses' mean profiles) across every pair of
    dose levels for one treatment — a continuous monotonicity descriptor,
    not a threshold.

    ``rho > 0`` means profiles grow more dissimilar as dose separation grows
    (consistent with a monotonic dose-response trajectory); ``rho < 0``
    means the opposite; values within ``rho_no_structure_band`` of zero are
    reported as no clear ordered structure. That banding is a reporting
    convenience, not a statistical test.

    With as few as 2-4 dose levels (the common case), there are only
    ``C(n_doses, 2)`` data points, which is too few for a meaningful
    permutation test over dose orderings (e.g. only 3! = 6 possible
    orderings for 3 doses). Instead, uncertainty on rho is estimated by
    bootstrapping over *replicate wells within each dose* — resampling which
    wells contribute to each dose's mean profile — which uses actual
    within-dose replicate variability rather than the degenerate space of
    dose-level permutations.

    Returns a dict with ``rho``, ``p_value`` (asymptotic, from
    ``scipy.stats.spearmanr`` — treat cautiously for n_dose_pairs < ~5),
    ``rho_ci_low``/``rho_ci_high`` (bootstrap 95% CI, or ``None`` if too few
    valid bootstrap draws), ``n_doses``, ``n_dose_pairs``, ``interpretation``,
    and a ``note`` flagging the trajectory-descriptor-only caveat when
    ``n_doses`` is small.
    """
    from scipy.stats import spearmanr

    n_doses = len(doses)
    if n_doses < 3:
        return {
            "n_doses": n_doses,
            "n_dose_pairs": 0,
            "rho": None,
            "p_value": None,
            "rho_ci_low": None,
            "rho_ci_high": None,
            "n_bootstrap_valid": 0,
            "interpretation": None,
            "note": "Fewer than 3 dose levels — a monotonicity trend is undefined.",
        }

    dose_wells = {d: df_trt.loc[df_trt[dose_col] == d, feat_cols].fillna(0).to_numpy() for d in doses}

    def _dose_diffs_and_distances(mean_profiles: dict[float, np.ndarray]) -> tuple[list[float], list[float]]:
        dose_diffs, distances = [], []
        for i, d1 in enumerate(doses):
            for d2 in doses[i + 1:]:
                v1 = mean_profiles[d1] - mean_profiles[d1].mean()
                v2 = mean_profiles[d2] - mean_profiles[d2].mean()
                corr = np.corrcoef(v1, v2)[0, 1] if (np.std(v1) > 0 and np.std(v2) > 0) else 0.0
                dose_diffs.append(abs(float(d2) - float(d1)))
                distances.append(1.0 - corr)
        return dose_diffs, distances

    observed_means = {d: dose_wells[d].mean(axis=0) for d in doses}
    dose_diffs, distances = _dose_diffs_and_distances(observed_means)
    n_pairs = len(dose_diffs)

    if len(set(dose_diffs)) < 2 or len(set(distances)) < 2:
        rho, p_value = 0.0, 1.0
    else:
        rho, p_value = spearmanr(dose_diffs, distances)
        rho, p_value = float(rho), float(p_value)

    rng = np.random.RandomState(seed)
    boot_rhos = []
    for _ in range(n_bootstrap):
        resampled_means = {}
        for d in doses:
            wells = dose_wells[d]
            if len(wells) == 0:
                continue
            sample_idx = rng.randint(0, len(wells), size=len(wells))
            resampled_means[d] = wells[sample_idx].mean(axis=0)
        if len(resampled_means) < n_doses:
            continue
        b_diffs, b_distances = _dose_diffs_and_distances(resampled_means)
        if len(set(b_diffs)) < 2 or len(set(b_distances)) < 2:
            continue
        b_rho, _ = spearmanr(b_diffs, b_distances)
        if np.isfinite(b_rho):
            boot_rhos.append(float(b_rho))

    if len(boot_rhos) >= 100:
        rho_ci_low, rho_ci_high = (float(v) for v in np.percentile(boot_rhos, [2.5, 97.5]))
    else:
        rho_ci_low, rho_ci_high = None, None

    if rho > rho_no_structure_band:
        interpretation = "increasing phenotypic separation with dose difference"
    elif rho < -rho_no_structure_band:
        interpretation = "potentially non-monotonic trajectory"
    else:
        interpretation = "no clear ordered dose-response structure"

    return {
        "n_doses": n_doses,
        "n_dose_pairs": n_pairs,
        "rho": rho,
        "p_value": p_value,
        "rho_ci_low": rho_ci_low,
        "rho_ci_high": rho_ci_high,
        "n_bootstrap_valid": len(boot_rhos),
        "interpretation": interpretation,
        "note": (
            f"Only {n_pairs} dose-level pair(s) from {n_doses} dose levels — interpret as a "
            "trajectory descriptor, not a well-powered significance test, unless more dose "
            "levels or replicates are available."
            if n_doses <= 4 else ""
        ),
    }


def run_dose_response(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    treatment_col: str,
    dose_col: str | None,
    null_size_copairs: int = 5000,
    seed: int = 42,
    min_doses: int = 2,
) -> dict[str, Any]:
    """Application C: do increasing doses show consistent ordering?

    Returns a ``{"message": ...}`` sentinel if ``dose_col`` is absent/``None``
    or no multi-dose treatments are detected.
    """
    if dose_col is None or dose_col not in df.columns:
        return {
            "message": "No concentration column detected — dose-response analysis skipped",
            "multi_dose_treatments": [], "spearman_monotonicity": pd.DataFrame(),
        }

    multi_dose, dose_counts = detect_dose_response_treatments(df, treatment_col, dose_col, min_doses)

    if not multi_dose:
        return {
            "message": "No multi-dose treatments detected — dose-response analysis skipped",
            "multi_dose_treatments": [], "dose_counts": dose_counts.to_dict(),
            "spearman_monotonicity": pd.DataFrame(),
        }

    results, sc21_results, spearman_results = [], [], []
    for trt in multi_dose:
        df_trt = df[df[treatment_col] == trt].copy()
        doses = sorted(df_trt[dose_col].unique())
        n_doses = len(doses)

        dose_profiles = {}
        for d in doses:
            mask = df_trt[dose_col] == d
            dose_profiles[d] = df_trt.loc[mask, feat_cols].fillna(0).mean(axis=0).values

        corr_matrix = np.zeros((n_doses, n_doses))
        for i, d1 in enumerate(doses):
            for j, d2 in enumerate(doses):
                v1 = dose_profiles[d1] - dose_profiles[d1].mean()
                v2 = dose_profiles[d2] - dose_profiles[d2].mean()
                corr_matrix[i, j] = np.corrcoef(v1, v2)[0, 1] if (np.std(v1) > 0 and np.std(v2) > 0) else 0.0

        adj_corrs, nonadj_corrs = [], []
        for i in range(n_doses):
            for j in range(i + 1, n_doses):
                if j == i + 1:
                    adj_corrs.append(corr_matrix[i, j])
                else:
                    nonadj_corrs.append(corr_matrix[i, j])

        adj_mean = float(np.mean(adj_corrs)) if adj_corrs else None
        nonadj_mean = float(np.mean(nonadj_corrs)) if nonadj_corrs else None

        sc21_pass, sc21_note = True, ""
        if n_doses >= 3 and adj_mean is not None and nonadj_mean is not None:
            if adj_mean > nonadj_mean:
                sc21_note = f"Adjacent ({adj_mean:.3f}) > non-adjacent ({nonadj_mean:.3f}): monotonic"
            else:
                sc21_pass = False
                sc21_note = f"Adjacent ({adj_mean:.3f}) <= non-adjacent ({nonadj_mean:.3f}): not monotonic"
        elif n_doses == 2:
            sc21_note = "Only 2 doses — cannot test adjacent vs non-adjacent"
        else:
            sc21_note = "Insufficient doses"

        dose_counts_trt = df_trt.groupby(dose_col).size().to_dict()
        results.append({
            "treatment": trt, "n_doses": n_doses, "doses": doses,
            "dose_replicate_counts": dose_counts_trt, "corr_matrix": corr_matrix,
            "adjacent_corr_mean": adj_mean, "nonadjacent_corr_mean": nonadj_mean,
            "mean_pairwise_corr": float(corr_matrix[np.triu_indices(n_doses, k=1)].mean()) if n_doses > 1 else None,
        })
        sc21_results.append({
            "treatment": trt, "n_doses": n_doses,
            "adjacent_corr": adj_mean, "nonadjacent_corr": nonadj_mean,
            "sc21_pass": sc21_pass, "note": sc21_note,
        })

        # Continuous complement to SC-21 (see dose_response_monotonicity_spearman's
        # docstring) -- descriptive only, does not gate SC-21's own pass/fail.
        spearman_result = dose_response_monotonicity_spearman(df_trt, feat_cols, dose_col, doses, seed=seed)
        spearman_results.append({"treatment": trt, **spearman_result})

    map_dose, _ = copairs_dose_response_map(df, feat_cols, treatment_col, dose_col, null_size=null_size_copairs, seed=seed)

    return {
        "multi_dose_treatments": multi_dose, "dose_counts": dose_counts.to_dict(),
        "dose_correlations": results,
        "dose_map": map_dose,
        "sc21_status": pd.DataFrame(sc21_results),
        "spearman_monotonicity": pd.DataFrame(spearman_results),
    }


def plot_dose_response(dose_result: dict, output_path: str | None = None):
    """Visualize Application C results. No-op (returns ``None``) when skipped."""
    import os

    import matplotlib.pyplot as plt

    if "message" in dose_result:
        return None
    dose_corrs = dose_result["dose_correlations"]
    if not dose_corrs:
        return None

    n_panels = min(len(dose_corrs), 3)
    fig, axes = plt.subplots(1, n_panels + 1, figsize=(6 * (n_panels + 1), 5))
    if n_panels + 1 == 1:
        axes = [axes]

    for idx, r in enumerate(dose_corrs[:n_panels]):
        ax = axes[idx]
        n_doses = r["n_doses"]
        doses = r["doses"]
        corr = r["corr_matrix"]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_doses))
        ax.set_yticks(range(n_doses))
        ax.set_xticklabels([f"{d}" for d in doses], fontsize=9)
        ax.set_yticklabels([f"{d}" for d in doses], fontsize=9)
        ax.set_title(f"{r['treatment']}\nDose-level correlations")
        for i in range(n_doses):
            for j in range(n_doses):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(corr[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[-1]
    if dose_result["dose_map"] is not None:
        dm = dose_result["dose_map"]
        y_pos = np.arange(len(dm))
        bar_colors = ["#0279EE" if p else "#FF9400" for p in dm["corrected_p_value"] < 0.05]
        ax.barh(y_pos, dm["mean_average_precision"], color=bar_colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(dm["Metadata_Treatment_Dose"], fontsize=9)
        ax.set_xlabel("Dose-level PC mAP")
        ax.set_title("Phenotypic consistency within treatment–dose\n(blue = FDR sig)")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Application D: treatment vs control separation
# ─────────────────────────────────────────────────────────────────────────────

EFFECT_DETECTABILITY_QUADRANTS: dict[tuple[bool, bool], dict[str, str]] = {
    (True, True): {
        "label": "Strong, consistent phenotype",
        "interpretation": "Robust phenotypic signature with large and reproducible changes.",
    },
    (True, False): {
        "label": "Strong but heterogeneous phenotype",
        "interpretation": (
            "Large effect size but heterogeneous response. The phenotype exists "
            "but is not consistently recovered across samples."
        ),
    },
    (False, True): {
        "label": "Subtle but reproducible phenotype",
        "interpretation": "Small but highly reproducible phenotype. Subtle biological changes that are consistently detected.",
    },
    (False, False): {
        "label": "No detectable (or weak) phenotype",
        "interpretation": "No detectable phenotypic signature, or changes indistinguishable from the negative control.",
    },
}


def classify_effect_detectability(
    cohens_d: float,
    map_value: float,
    *,
    cohens_d_threshold: float = 0.5,
    map_threshold: float = 0.5,
) -> dict[str, Any]:
    """Classify a treatment into one of four effect-size x detectability quadrants.

    Cohen's d (effect *magnitude*) and phenotypic-activity mAP (retrieval
    *consistency* relative to negative-control references, i.e.
    detectability) answer different questions -- a small but highly
    reproducible phenotype can score higher on activity mAP than a large but
    heterogeneous one. Looking at either metric alone can't distinguish
    these cases; the quadrant of (high/low d, high/low activity mAP) can.

    ``map_threshold=0.5`` matches the poscon-separation threshold already
    used elsewhere in this module (SC-22's "poscon mAP > 0.5" check);
    ``cohens_d_threshold=0.5`` matches the "medium effect" convention already
    used as the fingerprint "altered feature" threshold elsewhere in this
    pipeline (NB04/NB05), so neither threshold is a new, arbitrary choice.

    Returns ``{"high_d", "high_map", "label", "interpretation"}``.
    """
    high_d = bool(np.isfinite(cohens_d) and abs(cohens_d) >= cohens_d_threshold)
    high_map = bool(np.isfinite(map_value) and map_value >= map_threshold)
    quadrant = EFFECT_DETECTABILITY_QUADRANTS[(high_d, high_map)]
    return {"high_d": high_d, "high_map": high_map, **quadrant}


def run_treatment_vs_control(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    treatment_col: str,
    control_col: str,
    negcon_values: str | Sequence[str],
    poscon_values: str | Sequence[str] | None = None,
    plate_col: str | None = None,
    null_size_copairs: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Application D: can treatment profiles be distinguished from negative controls?"""
    if isinstance(negcon_values, str):
        negcon_values = [negcon_values]
    if isinstance(poscon_values, str):
        poscon_values = [poscon_values]

    trt_to_ctrl = df.groupby(treatment_col)[control_col].first().to_dict()

    map_tvc, _ = copairs_treatment_vs_control_map(
        df, feat_cols, treatment_col, control_col, negcon_values, plate_col,
        null_size=null_size_copairs, seed=seed,
    )

    negcon_mask = df[control_col].isin(negcon_values)
    negcon_profiles = df.loc[negcon_mask, feat_cols].fillna(0).values

    cohens_d = {}
    for trt in df[treatment_col].unique():
        trt_profiles = df.loc[df[treatment_col] == trt, feat_cols].fillna(0).values
        trt_mean = trt_profiles.mean(axis=0)
        neg_mean = negcon_profiles.mean(axis=0)
        trt_std = trt_profiles.std(axis=0)
        neg_std = negcon_profiles.std(axis=0)
        pooled_std = np.sqrt((trt_std ** 2 + neg_std ** 2) / 2)
        pooled_std[pooled_std == 0] = 1
        d_per_feature = (trt_mean - neg_mean) / pooled_std
        cohens_d[trt] = float(np.sqrt(np.mean(d_per_feature ** 2)))

    # BUG FIX: source notebook keyed with literal 'Metadata_Treatment'.
    cp_map = dict(zip(map_tvc[treatment_col], map_tvc["mean_average_precision"]))
    cp_norm = dict(zip(map_tvc[treatment_col], map_tvc["mean_normalized_average_precision"]))
    cp_pval = dict(zip(map_tvc[treatment_col], map_tvc["corrected_p_value"]))

    results = []
    for trt in sorted(cp_map.keys()):
        ctrl = trt_to_ctrl.get(trt, "unknown")
        _quadrant = classify_effect_detectability(cohens_d.get(trt, np.nan), cp_map[trt])
        results.append({
            "treatment": trt, "control_type": ctrl,
            "mAP_vs_negcon": cp_map[trt], "normalized_mAP": cp_norm[trt],
            "fdr_p_value": cp_pval[trt], "fdr_significant": cp_pval[trt] < 0.05,
            "cohens_d": cohens_d.get(trt, np.nan),
            "is_negcon": ctrl in negcon_values,
            "is_poscon": bool(poscon_values) and ctrl in poscon_values,
            "effect_detectability_quadrant": _quadrant["label"],
            "effect_detectability_interpretation": _quadrant["interpretation"],
        })
    results_df = pd.DataFrame(results)

    poscon_rows = results_df[results_df["is_poscon"]] if poscon_values else pd.DataFrame()
    trt_rows = results_df[~results_df["is_negcon"] & ~results_df["is_poscon"]]
    sc22_checks = []

    if len(poscon_rows) > 0:
        poscon_pass = poscon_rows["mAP_vs_negcon"].min() > 0.5
        sc22_checks.append({
            "check": "poscon_separation", "description": "Positive-control activity mAP > 0.5",
            "value": float(poscon_rows["mAP_vs_negcon"].min()), "threshold": 0.5,
            "pass": poscon_pass, "critical": True,
        })
    else:
        sc22_checks.append({
            "check": "poscon_separation",
            "description": "Positive-control activity could not be evaluated because no positive-control profile was found",
            "value": None,
            "threshold": 0.5,
            "pass": False,
            "critical": True,
        })

    unexpected_negcon_activity = results_df[
        results_df["is_negcon"] & (results_df["normalized_mAP"].abs() >= 0.05)
    ]
    sc22_checks.append({
        "check": "negcon_baseline_activity",
        "description": (
            "Negative-control profiles remain near the pooled negative-control baseline"
            if unexpected_negcon_activity.empty
            else f"{len(unexpected_negcon_activity)} negative-control treatment(s) show |normalized activity mAP| >= 0.05"
        ),
        "value": unexpected_negcon_activity["treatment"].tolist(),
        "threshold": 0.05,
        "pass": unexpected_negcon_activity.empty,
        "critical": True,
    })

    no_phenotype = trt_rows[trt_rows["normalized_mAP"].abs() < 0.05]
    if len(no_phenotype) > 0:
        sc22_checks.append({
            "check": "no_phenotype_treatments",
            "description": f"{len(no_phenotype)} treatment(s) with |normalized_mAP| < 0.05",
            "value": no_phenotype["treatment"].tolist(), "threshold": 0.05,
            "pass": True, "critical": False,
        })
    else:
        sc22_checks.append({
            "check": "no_phenotype_treatments",
            "description": "All treatments show detectable phenotype", "value": 0,
            "threshold": 0.05, "pass": True, "critical": False,
        })

    return {
        "results": results_df, "map_tvc": map_tvc,
        "sc22_status": {
            "checks": sc22_checks,
            "sc22_pass": all(c["pass"] for c in sc22_checks if c.get("critical", True)),
        },
    }


def plot_treatment_vs_control(tvc_result: dict, output_path: str | None = None):
    """Visualize Application D results."""
    import os

    import matplotlib.pyplot as plt

    results = tvc_result["results"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    plot_df = results[~results["is_negcon"]].sort_values("mAP_vs_negcon", ascending=True)
    colors = ["#75A025" if p else "#0279EE" if s else "#FF9400"
              for p, s in zip(plot_df["is_poscon"], plot_df["fdr_significant"])]
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["mAP_vs_negcon"], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["treatment"], fontsize=9)
    ax.set_xlabel("Phenotypic-activity mAP vs negative control")
    ax.set_title("Phenotypic activity: treatment vs control\n(green=poscon, blue=FDR sig, orange=not sig)")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    for i, val in enumerate(plot_df["mAP_vs_negcon"]):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=8)

    ax = axes[1]
    non_negcon = results[~results["is_negcon"]]
    scatter_colors = ["#75A025" if p else "#0279EE" for p in non_negcon["is_poscon"]]
    ax.scatter(non_negcon["cohens_d"], non_negcon["mAP_vs_negcon"], c=scatter_colors, s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
    for _i, (_, row) in enumerate(non_negcon.iterrows()):
        # Alternate the label above/below its point -- with several
        # treatments clustered close together (e.g. a dose series), stacking
        # every label at a fixed (5, 5) point offset makes them overlap.
        _y_offset = 6 if _i % 2 == 0 else -12
        ax.annotate(
            str(row["treatment"]), (row["cohens_d"], row["mAP_vs_negcon"]),
            fontsize=7, alpha=0.8, xytext=(5, _y_offset), textcoords="offset points",
        )
    ax.set_xlabel("Cohen's d (RMS) — effect magnitude")
    ax.set_ylabel("Phenotypic-activity mAP vs negcon")
    ax.set_title("Effect magnitude vs. detectability\n(green=poscon)")

    # Quadrant guide: Cohen's d (magnitude) x mAP (detectability) answer
    # different questions -- see classify_effect_detectability's docstring.
    # Thresholds match ones already used elsewhere in this pipeline (mAP=0.5
    # is SC-22's poscon-separation threshold; d=0.5 is the fingerprint
    # "altered feature" threshold used in NB04/NB05), not new arbitrary cutoffs.
    _d_threshold, _map_threshold = 0.5, 0.5
    ax.axvline(x=_d_threshold, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=_map_threshold, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    _x_max = max(float(non_negcon["cohens_d"].max(skipna=True) or 0.0) * 1.15, _d_threshold * 2.0, 0.1)
    # mAP is a similarity-derived score and can legitimately go negative, so
    # anchor the lower bound below the data (not at 0) -- fixed at the
    # threshold*0.05 label positions below assumed a y-axis starting near 0,
    # which broke as soon as autoscaled data didn't start near 0 (the actual
    # bug this fixes: labels rendered below the visible axes entirely).
    _y_min = min(float(non_negcon["mAP_vs_negcon"].min(skipna=True) or 0.0), 0.0) - 0.05
    _y_max = max(float(non_negcon["mAP_vs_negcon"].max(skipna=True) or 0.0) * 1.15, _map_threshold * 1.3)
    ax.set_xlim(0, _x_max)
    ax.set_ylim(_y_min, _y_max)
    _quadrant_label_kwargs = {"fontsize": 7, "color": "gray", "alpha": 0.9, "ha": "left", "va": "bottom"}
    ax.text(_x_max * 0.02, _map_threshold + (_y_max - _map_threshold) * 0.03, "subtle but\nreproducible", **_quadrant_label_kwargs)
    ax.text(_d_threshold + (_x_max - _d_threshold) * 0.05, _map_threshold + (_y_max - _map_threshold) * 0.03, "strong, consistent", **_quadrant_label_kwargs)
    ax.text(_x_max * 0.02, _y_min + (_map_threshold - _y_min) * 0.05, "no detectable\nphenotype", **_quadrant_label_kwargs)
    ax.text(_d_threshold + (_x_max - _d_threshold) * 0.05, _y_min + (_map_threshold - _y_min) * 0.05, "strong but\nheterogeneous", **_quadrant_label_kwargs)

    ax = axes[2]
    plot_df2 = results.sort_values("normalized_mAP", ascending=True)
    norm_colors = ["#FF9400" if abs(n) < 0.05 else "#0279EE" for n in plot_df2["normalized_mAP"]]
    y_pos2 = np.arange(len(plot_df2))
    ax.barh(y_pos2, plot_df2["normalized_mAP"], color=norm_colors, alpha=0.8)
    ax.set_yticks(y_pos2)
    ax.set_yticklabels(plot_df2["treatment"], fontsize=9)
    ax.set_xlabel("Normalized phenotypic-activity mAP")
    ax.set_title("Phenotypic activity / detectability\n(orange = no detectable phenotype)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=0.05, color="red", linestyle="--", alpha=0.3)
    ax.axvline(x=-0.05, color="red", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Time-course analysis (conditional)
# ─────────────────────────────────────────────────────────────────────────────

def run_time_course(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    treatment_col: str,
    time_col: str | None,
    control_col: str,
    negcon_values: str | Sequence[str],
    poscon_values: str | Sequence[str],
    plate_col: str | None = None,
    null_percentile: float = 95,
    n_null: int = 10000,
    seed: int = 42,
    null_size_copairs: int = 5000,
) -> dict[str, Any]:
    """Section 8: PR/mAP evolution over time, only if a time axis with >1 value exists."""
    if time_col is None or time_col not in df.columns:
        return {"message": "Time-course analysis skipped: no time column detected", "n_time_points": 0}

    n_unique = df[time_col].nunique()
    if n_unique < 2:
        return {"message": f"Time-course analysis skipped: only {n_unique} time point", "n_time_points": n_unique}

    if isinstance(negcon_values, str):
        negcon_values = [negcon_values]
    if isinstance(poscon_values, str):
        poscon_values = [poscon_values]

    time_values = sorted(df[time_col].unique())
    results = []

    for time_val in time_values:
        df_time = df[df[time_col] == time_val].copy()
        profiles = df_time[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        labels = df_time[treatment_col].values
        trt_to_ctrl = df_time.groupby(treatment_col)[control_col].first().to_dict()

        pr_value, pr_medians, _, threshold = percent_replicating(
            profiles, labels, plate_labels=None,
            null_percentile=null_percentile, n_null=n_null, seed=seed, batch_mode="global",
        )

        map_cp = copairs_compute_map(
            df_time, feat_cols, pos_sameby=[treatment_col],
            neg_diffby=[treatment_col], null_size=null_size_copairs, seed=seed,
        )
        # BUG FIX: source notebook keyed with literal 'Metadata_Treatment'.
        cp_dict = dict(zip(map_cp[treatment_col], map_cp["mean_average_precision"]))

        for trt in sorted(pr_medians.keys()):
            results.append({
                time_col: time_val, "treatment": trt,
                "control_type": trt_to_ctrl.get(trt, "unknown"),
                "median_corr": pr_medians[trt], "null_threshold": threshold,
                "passes_pr": pr_medians[trt] > threshold, "mAP_copairs": cp_dict.get(trt, np.nan),
            })

        n_treatments = len(pr_medians)
        n_passing = sum(1 for m in pr_medians.values() if m > threshold)
        trt_map_vals = [cp_dict.get(t, np.nan) for t in pr_medians if trt_to_ctrl.get(t) not in negcon_values]
        results.append({
            time_col: time_val, "treatment": "__AGGREGATE__", "control_type": "summary",
            "pr_fraction": n_passing / n_treatments if n_treatments > 0 else 0,
            "n_treatments": n_treatments, "n_passing": n_passing,
            "mAP_copairs": float(np.nanmean(trt_map_vals)) if trt_map_vals else np.nan,
        })

    results_df = pd.DataFrame(results)
    summary = results_df[results_df["treatment"] == "__AGGREGATE__"].sort_values(time_col)
    optimal_time = summary.loc[summary["mAP_copairs"].idxmax(), time_col] if len(summary) > 0 else None

    return {
        "n_time_points": len(time_values), "time_values": time_values, "time_col": time_col,
        "per_treatment_per_time": results_df[results_df["treatment"] != "__AGGREGATE__"],
        "time_course_summary": summary, "optimal_time": optimal_time,
    }


def plot_time_course(time_result: dict, output_path: str | None = None):
    """Visualize Section 8 results. No-op (returns ``None``) when skipped."""
    import os

    import matplotlib.pyplot as plt

    if "message" in time_result:
        return None

    per_treat = time_result["per_treatment_per_time"]
    summary = time_result["time_course_summary"]
    time_col = time_result["time_col"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    treatments = sorted(per_treat["treatment"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(treatments)))

    for i, trt in enumerate(treatments):
        td = per_treat[per_treat["treatment"] == trt].sort_values(time_col)
        axes[0].plot(td[time_col], td["median_corr"], marker="o", label=trt, color=colors[i], linewidth=2)
        axes[1].plot(td[time_col], td["mAP_copairs"], marker="s", label=trt, color=colors[i], linewidth=2)

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("PR median corr")
    axes[0].set_title("PR Over Time")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("PC mAP")
    axes[1].set_title("Phenotypic consistency over time")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 1.05)

    ax2 = axes[2]
    ax2_twin = ax2.twinx()
    ax2.plot(summary[time_col], summary["pr_fraction"], marker="o", color="#0279EE", linewidth=2, label="PR fraction")
    ax2_twin.plot(summary[time_col], summary["mAP_copairs"], marker="s", color="#FF9400", linewidth=2, label="Mean PC mAP")
    ax2.axvline(x=time_result["optimal_time"], color="red", linestyle="--", alpha=0.5, label=f"Optimal ({time_result['optimal_time']})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("PR fraction", color="#0279EE")
    ax2_twin.set_ylabel("Mean PC mAP", color="#FF9400")
    ax2.set_title("Aggregate Quality Over Time")
    ax2.legend(fontsize=8, loc="upper left")
    ax2_twin.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Go/No-Go dashboard
# ─────────────────────────────────────────────────────────────────────────────

def generate_go_nogo_dashboard(
    qc_result: dict,
    batch_result: dict,
    dose_result: dict,
    tvc_result: dict,
    time_result: dict | None = None,
) -> dict[str, Any]:
    """Aggregate all quality-metric applications into a Go/No-Go decision.

    Decision rules (unchanged from the source notebook):
      - **NO-GO** if any critical check fails.
      - **GO WITH CAUTION** if all critical checks pass but a non-critical
        check fails.
      - **GO** if all checks pass.

    Critical checks: SC-19 (control validation), PR-25% (PR fraction per
    plate), SC-22 (treatment phenotype detectability).
    Non-critical checks: SC-20 (cross-plate batch effect), SC-21
    (dose-response monotonicity), TIME (time-course, informational).
    """
    checks = []

    if "sc19_status" in qc_result:
        sc19 = qc_result["sc19_status"]
        n_pass = int(sc19["sc19_pass"].sum())
        checks.append({
            "check_id": "SC-19", "name": "Control validation (negcon PR ≤ 0.10, poscon PR > 0.50)",
            "n_pass": n_pass, "n_total": len(sc19), "pass": n_pass == len(sc19), "critical": True,
            "details": f"{n_pass}/{len(sc19)} plates pass",
        })

    qc_flags = qc_result["qc_flags"]
    n_pp = int(qc_flags["plate_pass"].sum())
    checks.append({
        "check_id": "PR-25%", "name": "PR fraction ≥ 25% (per plate)",
        "n_pass": n_pp, "n_total": len(qc_flags), "pass": n_pp == len(qc_flags), "critical": True,
        "details": f"{n_pp}/{len(qc_flags)} plates pass (mean: {qc_flags['pr_fraction'].mean():.1%})",
    })

    if "sc20_status" in batch_result:
        sc20 = batch_result["sc20_status"]
        checks.append({
            "check_id": "SC-20", "name": "Cross-plate batch effect (drop < 30%)",
            "n_pass": 0 if sc20["n_batch_affected"] > 0 else 1, "n_total": 1,
            "pass": sc20["sc20_pass"], "critical": False,
            "details": f"{sc20['n_batch_affected']} affected: {sc20['batch_affected_treatments']}",
        })

    if "sc22_status" in tvc_result:
        sc22 = tvc_result["sc22_status"]
        _critical_sc22 = [c for c in sc22["checks"] if c.get("critical", True)]
        n_p = sum(1 for c in _critical_sc22 if c["pass"])
        checks.append({
            "check_id": "SC-22", "name": "Control activity and phenotype detectability",
            "n_pass": n_p, "n_total": len(_critical_sc22), "pass": sc22["sc22_pass"], "critical": True,
            "details": "; ".join(
                f"{c['check']}: {'pass' if c['pass'] else 'fail'}" for c in _critical_sc22
            ),
        })

    if "sc21_status" in dose_result:
        sc21 = dose_result["sc21_status"]
        if "sc21_pass" in sc21.columns:
            n_dp = int(sc21["sc21_pass"].sum())
            n_dt = len(sc21)
        else:
            n_dp = 0
            n_dt = 0
        checks.append({
            "check_id": "SC-21", "name": "Dose-response monotonicity",
            "n_pass": n_dp, "n_total": n_dt, "pass": n_dp == n_dt if n_dt > 0 else True, "critical": False,
            "details": f"{n_dp}/{n_dt} pass" if n_dt > 0 else "No multi-dose treatments",
        })

    if time_result and "n_time_points" in time_result:
        checks.append({
            "check_id": "TIME", "name": "Time-course analysis",
            "n_pass": 1 if time_result["n_time_points"] > 1 else 0, "n_total": 1,
            "pass": True, "critical": False,
            "details": f"{time_result['n_time_points']} time point(s)" if time_result["n_time_points"] > 1 else time_result.get("message", "N/A"),
        })

    checks_df = pd.DataFrame(checks)
    critical = checks_df[checks_df["critical"]]
    n_crit_pass = int(critical["pass"].sum())
    non_crit = checks_df[~checks_df["critical"]]
    n_noncrit_fail = int((~non_crit["pass"]).sum())

    if n_crit_pass == len(critical):
        decision = "GO" if n_noncrit_fail == 0 else "GO WITH CAUTION"
        rationale = ("All critical checks pass" if n_noncrit_fail == 0
                     else f"All critical pass. Warnings: {non_crit[~non_crit['pass']]['check_id'].tolist()}")
    else:
        decision = "NO-GO"
        rationale = f"Critical checks failed: {critical[~critical['pass']]['check_id'].tolist()}"

    tvc_res = tvc_result["results"]
    non_neg = ~tvc_res["is_negcon"]
    sig = tvc_res["fdr_significant"] & non_neg
    poscon = tvc_res["is_poscon"]
    cross_pr = batch_result.get("cross_plate_pr_value", np.nan) if "cross_plate_pr_value" in batch_result else np.nan

    metric_summary = {
        "n_plates": len(qc_flags), "n_treatments": int(qc_result["pr_results"]["treatment"].nunique()),
        "mean_pr_fraction": float(qc_flags["pr_fraction"].mean()),
        "cross_plate_pr": float(cross_pr) if not np.isnan(cross_pr) else "N/A",
        "mean_treatment_map": float(tvc_res.loc[non_neg, "mAP_vs_negcon"].mean()),
        "n_fdr_significant": int(sig.sum()),
        "poscon_map": float(tvc_res.loc[poscon, "mAP_vs_negcon"].iloc[0]) if poscon.any() else "N/A",
        "decision": decision, "rationale": rationale,
    }

    return {"checks": checks_df, "metric_summary": metric_summary, "decision": decision, "rationale": rationale}


def plot_go_nogo_dashboard(dashboard: dict, output_path: str | None = None):
    """Visualize the Go/No-Go dashboard: check-status table + key metrics."""
    import os

    import matplotlib.pyplot as plt

    checks = dashboard["checks"]
    metrics = dashboard["metric_summary"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.axis("off")
    table_data = []
    for _, row in checks.iterrows():
        _status = "PASS" if row["pass"] else ("FAIL" if row["critical"] else "WARN")
        table_data.append([row["check_id"], _status,
                           "CRITICAL" if row["critical"] else "info", f"{row['n_pass']}/{row['n_total']}"])
    table_data.append(["", "", "", ""])
    table_data.append(["DECISION", dashboard["decision"], "", ""])

    colors = []
    for _, row in checks.iterrows():
        c = "#75A025" if row["pass"] else ("#FF4444" if row["critical"] else "#FF9400")
        colors.append([c] * 4)
    colors.append(["white"] * 4)
    dc = "#75A025" if dashboard["decision"] == "GO" else ("#FF4444" if dashboard["decision"] == "NO-GO" else "#FF9400")
    colors.append([dc] * 4)

    table = ax.table(cellText=table_data, colLabels=["Check", "Status", "Type", "Pass/Total"],
                      cellColours=colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title("Go/No-Go Decision Dashboard", fontsize=14, fontweight="bold", pad=20)

    ax = axes[1]
    plot_metrics = {}
    _metric_labels = {
        "mean_pr_fraction": "mean PR fraction",
        "cross_plate_pr": "cross-plate PR",
        "mean_treatment_map": "mean treatment activity mAP",
        "poscon_map": "poscon activity mAP",
    }
    for k in ["mean_pr_fraction", "cross_plate_pr", "mean_treatment_map", "poscon_map"]:
        v = metrics.get(k, np.nan)
        if isinstance(v, (int, float)) and not np.isnan(v):
            plot_metrics[_metric_labels[k]] = v

    if plot_metrics:
        y_pos = np.arange(len(plot_metrics))
        bars = ax.barh(y_pos, list(plot_metrics.values()), color="#0279EE", alpha=0.8)
        thresholds = {
            "mean PR fraction": 0.25,
            "cross-plate PR": 0.25,
            "mean treatment activity mAP": 0.5,
            "poscon activity mAP": 0.5,
        }
        for i, (label, val) in enumerate(plot_metrics.items()):
            bars[i].set_color("#75A025" if val >= thresholds.get(label, 0.5) else "#FF9400")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(plot_metrics.keys()), fontsize=9)
        ax.set_xlabel("Value")
        ax.set_title("Key Metrics\n(green = above threshold)")
        ax.set_xlim(0, 1.05)
        for i, (label, val) in enumerate(plot_metrics.items()):
            ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
