"""Phenotypic-profiling modelling pipeline: PCA-space QC, UMAP, LDA, KMeans.

Ports ``03_phenotypic_profiling_parallel_spaces.ipynb``, principally its
``run_modelling_space`` function (source cell ~25) — the core routine that
runs the same downstream modelling (multivariate QC, UMAP, treatment-level
LDA + leave-one-plate-out CV, dose-level LDA + leave-one-plate-out CV +
confusion matrix, KMeans clustering, model serialization) in one latent space
(e.g. uncorrected PCA vs. Harmony-batch-corrected PCA), so the two spaces can
be compared on equal footing.

Also ports ``get_qc_metric``, ``safe_relative_change`` and ``absolute_change``
(source cells ~32-33, used to compare an uncorrected space against a
batch-corrected one), and ``balanced_sample`` from
``06_single_cell_analysis.ipynb`` (source cell ~21), which is conceptually a
subsampling helper for single-cell modelling rather than a normalization step.

Design decisions relative to the source notebook
-------------------------------------------------
- Every column name (treatment, plate, dose) and every threshold/random seed
  is an explicit parameter — the source notebook read them off module-level
  globals (``TREATMENT_COL``, ``PLATE_COL``, ``RANDOM_STATE``, ...).
- The dose axis is optional. The source notebook assumed a dose column
  (``"_dose_label"``) always existed; here, if ``dose_col`` is ``None``, not
  present, or has fewer than 2 unique values, the dose-LDA/CV/confusion-matrix
  step is skipped and reported via a ``{"message": ...}`` sentinel, and KMeans
  clustering falls back to clustering the treatment-LDA representation
  instead of the dose-LDA representation (this fallback is a mild
  generalization beyond the source, needed because the source's KMeans step
  clusters exactly the dose-LDA coordinates into ``len(dose_classes)``
  clusters, which has no meaning without a dose axis).
- Leave-one-plate-out CV (``LeaveOneGroupOut`` grouped by the plate column)
  is guarded: if there is only one plate group, the CV step is skipped with a
  ``{"message": ...}`` sentinel rather than being silently degenerate (a
  single-group ``LeaveOneGroupOut`` split trains and tests on disjoint empty
  folds and produces meaningless — but not obviously invalid-looking —
  metrics).
- Matplotlib plotting (UMAP/LDA scatter panels) is intentionally **not**
  ported into this module. Those figures are presentation-only and belong in
  the sibling ``plotting.py``; this module returns all coordinates, models,
  and CV tables a caller needs to plot them itself.
- Disk I/O (CSV/model serialization) is optional: pass a ``directories``
  mapping to write outputs (mirroring the source notebook's per-space output
  directories); pass ``None`` to keep everything in memory (used by the
  smoke test and any caller that wants to defer persistence).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneGroupOut

try:
    from umap import UMAP
    _UMAP_OK = True
except ImportError:  # pragma: no cover - exercised only when umap-learn is absent
    _UMAP_OK = False

__all__ = [
    "validate_output_path",
    "balanced_sample",
    "run_modelling_space",
    "get_qc_metric",
    "safe_relative_change",
    "absolute_change",
]


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def validate_output_path(path: Path, overwrite: bool = True) -> Path:
    """Create parent directories and optionally protect an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    return path


def balanced_sample(
    df: pd.DataFrame,
    X: np.ndarray,
    label_col: str,
    n_per_class: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sample up to ``n_per_class`` rows per class in ``label_col``, stratified.

    ``X`` must be row-aligned with ``df`` (same length, same order); the
    returned ``X_sampled`` is re-indexed to match the sampled rows of ``df``.

    Returns ``(df_sampled, X_sampled)``.
    """
    rng = np.random.default_rng(random_state)
    keep_idx = []
    for _, sub in df.groupby(label_col):
        n = min(n_per_class, len(sub))
        chosen = rng.choice(sub.index, size=n, replace=False)
        keep_idx.extend(chosen)
    keep_idx = sorted(keep_idx)

    pos_map = {orig: pos for pos, orig in enumerate(df.index)}
    pos_idx = [pos_map[i] for i in keep_idx]
    return df.loc[keep_idx].copy(), X[pos_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Multivariate QC helpers (PERMANOVA / PERMDISP / silhouette)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.any(counts < 2) or len(labels) <= len(unique):
        return float("nan")
    return float(silhouette_score(matrix, labels, metric="euclidean"))


def _one_factor_permanova(
    matrix: np.ndarray, labels: np.ndarray, permutations: int = 999, random_state: int = 42,
) -> dict[str, float]:
    """One-factor PERMANOVA using Euclidean distances.

    Returns pseudo-F, permutation p-value, and R2 (between-group SS / total SS).
    """
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    n = matrix.shape[0]
    groups = np.unique(labels)
    g = len(groups)

    if n < 3 or g < 2 or n <= g:
        return {"pseudo_F": np.nan, "p_value": np.nan, "R2": np.nan}

    distances = pairwise_distances(matrix, metric="euclidean")
    squared = distances ** 2
    ss_total = squared[np.triu_indices(n, k=1)].sum() / n

    def statistic(current_labels):
        ss_within = 0.0
        for group in np.unique(current_labels):
            idx = np.flatnonzero(current_labels == group)
            ng = len(idx)
            if ng < 2:
                continue
            sub = squared[np.ix_(idx, idx)]
            ss_within += sub[np.triu_indices(ng, k=1)].sum() / ng
        ss_between = max(ss_total - ss_within, 0.0)
        df_between = g - 1
        df_within = n - g
        pseudo_f = (
            (ss_between / df_between) / (ss_within / df_within)
            if ss_within > 0 and df_within > 0
            else np.nan
        )
        r2 = ss_between / ss_total if ss_total > 0 else np.nan
        return pseudo_f, r2

    observed_f, observed_r2 = statistic(labels)
    rng = np.random.default_rng(random_state)
    permuted_f = []
    for _ in range(permutations):
        perm_f, _ = statistic(rng.permutation(labels))
        permuted_f.append(perm_f)

    permuted_f = np.asarray(permuted_f, dtype=float)
    p_value = (
        (1 + np.sum(permuted_f >= observed_f)) / (permutations + 1)
        if np.isfinite(observed_f)
        else np.nan
    )
    return {"pseudo_F": observed_f, "p_value": p_value, "R2": observed_r2}


def _one_factor_permdisp(
    matrix: np.ndarray, labels: np.ndarray, permutations: int = 999, random_state: int = 42,
) -> dict[str, float]:
    """Permutation test for equality of multivariate dispersion."""
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = len(labels)
    g = len(unique)

    if n < 3 or g < 2 or n <= g:
        return {"pseudo_F": np.nan, "p_value": np.nan}

    def distances_to_group_centroids(current_labels):
        d = np.empty(n, dtype=float)
        for group in np.unique(current_labels):
            idx = np.flatnonzero(current_labels == group)
            centroid = matrix[idx].mean(axis=0)
            d[idx] = np.linalg.norm(matrix[idx] - centroid, axis=1)
        return d

    def anova_f(values, current_labels):
        grand = values.mean()
        ss_between = 0.0
        ss_within = 0.0
        for group in np.unique(current_labels):
            vals = values[current_labels == group]
            ss_between += len(vals) * (vals.mean() - grand) ** 2
            ss_within += ((vals - vals.mean()) ** 2).sum()
        return (
            (ss_between / (g - 1)) / (ss_within / (n - g))
            if ss_within > 0
            else np.nan
        )

    observed_distances = distances_to_group_centroids(labels)
    observed_f = anova_f(observed_distances, labels)
    rng = np.random.default_rng(random_state)
    permuted_f = []
    for _ in range(permutations):
        perm_labels = rng.permutation(labels)
        perm_distances = distances_to_group_centroids(perm_labels)
        permuted_f.append(anova_f(perm_distances, perm_labels))

    permuted_f = np.asarray(permuted_f, dtype=float)
    p_value = (
        (1 + np.sum(permuted_f >= observed_f)) / (permutations + 1)
        if np.isfinite(observed_f)
        else np.nan
    )
    return {"pseudo_F": observed_f, "p_value": p_value}


def _qc_record(
    scope: str, factor: str, matrix: np.ndarray, labels: np.ndarray,
    permutations: int = 999, random_state: int = 42,
) -> dict[str, Any]:
    permanova = _one_factor_permanova(matrix, labels, permutations=permutations, random_state=random_state)
    permdisp = _one_factor_permdisp(matrix, labels, permutations=permutations, random_state=random_state)
    return {
        "scope": scope,
        "factor": factor,
        "n_wells": int(len(labels)),
        "n_groups": int(len(np.unique(labels))),
        "silhouette": _safe_silhouette(matrix, labels),
        "permanova_pseudo_F": permanova["pseudo_F"],
        "permanova_p": permanova["p_value"],
        "permanova_R2": permanova["R2"],
        "permdisp_pseudo_F": permdisp["pseudo_F"],
        "permdisp_p": permdisp["p_value"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core modelling-space routine
# ─────────────────────────────────────────────────────────────────────────────

def run_modelling_space(
    *,
    space_name: str,
    space_label: str,
    X_latent: np.ndarray,
    metadata_df: pd.DataFrame,
    treatment_col: str,
    plate_col: str,
    dose_col: str | None = None,
    negcon_mask: np.ndarray | pd.Series | None = None,
    directories: Mapping[str, Path] | None = None,
    random_state: int = 42,
    n_qc_permutations: int = 999,
    overwrite_existing_outputs: bool = True,
    run_umap: bool = True,
    umap_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full comparable-space modelling pipeline in one latent space.

    Parameters
    ----------
    X_latent:
        The latent coordinate matrix for this space (e.g. PCA or
        Harmony-corrected PCA coordinates), row-aligned with ``metadata_df``.
    metadata_df:
        Metadata rows aligned with ``X_latent``.
    treatment_col, plate_col:
        Resolved column names for treatment and plate grouping.
    dose_col:
        Resolved column name for the dose/concentration axis, or ``None`` if
        no such axis exists in this experiment. Dose-level LDA/CV/confusion
        matrix are skipped when absent or single-valued.
    negcon_mask:
        Boolean mask (aligned with ``metadata_df``/``X_latent``) selecting
        negative-control wells, used for an additional "negative controls
        only, by plate" QC record. Skipped if ``None`` or too few negcon
        wells/plates are available.
    directories:
        Optional mapping with keys ``"results"`` and ``"models"`` (at least)
        pointing at output directories; when provided, QC/CV tables are
        written as CSV and fitted models are serialized with joblib. When
        ``None``, everything stays in memory.
    """
    result: dict[str, Any] = {
        "space_name": space_name,
        "space_label": space_label,
        "n_wells": int(X_latent.shape[0]),
        "n_dimensions": int(X_latent.shape[1]),
    }

    y_treatment = metadata_df[treatment_col].astype(str).to_numpy()
    groups = metadata_df[plate_col].astype(str).to_numpy()

    has_dose = (
        dose_col is not None
        and dose_col in metadata_df.columns
        and metadata_df[dose_col].nunique() > 1
    )
    # Matches the source notebook's "_dose_label": a treatment x concentration
    # composite class, not the bare concentration value — this is what makes
    # the "dose LDA" / confusion matrix / clustering meaningfully finer-grained
    # than the treatment-level LDA (e.g. distinguishes trt_A|0.1 from trt_A|1.0).
    dose_label = (
        metadata_df[treatment_col].astype(str) + " | "
        + metadata_df[dose_col].map(lambda v: f"{float(v):g}")
    ) if has_dose else None
    y_dose = dose_label.to_numpy() if has_dose else None

    def _save_csv(frame: pd.DataFrame, relative_name: str) -> None:
        if directories is None:
            return
        path = validate_output_path(Path(directories["results"]) / relative_name, overwrite_existing_outputs)
        frame.to_csv(path, index=False)

    # ── Multivariate QC ──────────────────────────────────────────────────
    qc_rows = [
        _qc_record("all_wells", "plate", X_latent, groups, n_qc_permutations, random_state),
        _qc_record("all_wells", "treatment", X_latent, y_treatment, n_qc_permutations, random_state),
    ]
    if negcon_mask is not None:
        negcon_mask = np.asarray(negcon_mask)
        if negcon_mask.sum() >= 6 and metadata_df.loc[negcon_mask, plate_col].nunique() >= 2:
            qc_rows.append(
                _qc_record(
                    "negative_controls", "plate",
                    X_latent[negcon_mask],
                    metadata_df.loc[negcon_mask, plate_col].astype(str).to_numpy(),
                    n_qc_permutations, random_state,
                )
            )
    qc_df = pd.DataFrame(qc_rows)
    _save_csv(qc_df, "multivariate_qc.csv")
    result["qc"] = qc_df

    # ── UMAP (optional dependency) ───────────────────────────────────────
    umap_coordinates = None
    if run_umap:
        if _UMAP_OK:
            reducer = UMAP(
                n_neighbors=min(15, max(2, X_latent.shape[0] - 1)),
                min_dist=0.1,
                metric="euclidean",
                random_state=random_state,
                n_jobs=1,
                **(dict(umap_kwargs) if umap_kwargs else {}),
            )
            umap_coordinates = reducer.fit_transform(X_latent)
            if directories is not None:
                umap_df = metadata_df[[treatment_col, plate_col]].copy()
                umap_df["UMAP1"] = umap_coordinates[:, 0]
                umap_df["UMAP2"] = umap_coordinates[:, 1]
                _save_csv(umap_df, "umap_coordinates.csv")
        else:
            warnings.warn("umap-learn not installed — UMAP step skipped.")
    result["umap_coordinates"] = umap_coordinates

    # ── Treatment-level LDA + leave-one-plate-out CV ────────────────────
    logo = LeaveOneGroupOut()
    n_groups = len(np.unique(groups))
    n_classes_t = len(np.unique(y_treatment))

    if n_classes_t < 2:
        result["treatment_lda"] = {"message": "Treatment LDA skipped: fewer than 2 treatment classes present"}
    else:
        n_lda_t = min(n_classes_t - 1, X_latent.shape[1])
        lda_t = LinearDiscriminantAnalysis(n_components=n_lda_t, solver="svd")
        X_lda_t = lda_t.fit_transform(X_latent, y_treatment)

        if n_groups < 2:
            treatment_cv: pd.DataFrame | dict = {
                "message": "Leave-one-plate-out CV skipped: only 1 plate group present",
                "n_groups": n_groups,
            }
        else:
            treatment_cv_records = []
            for fold, (train_idx, test_idx) in enumerate(logo.split(X_latent, y_treatment, groups), start=1):
                model = LinearDiscriminantAnalysis(n_components=n_lda_t, solver="svd")
                model.fit(X_latent[train_idx], y_treatment[train_idx])
                prediction = model.predict(X_latent[test_idx])
                treatment_cv_records.append({
                    "fold": fold,
                    "held_out_plate": groups[test_idx[0]],
                    "accuracy": accuracy_score(y_treatment[test_idx], prediction),
                    "balanced_accuracy": balanced_accuracy_score(y_treatment[test_idx], prediction),
                    "macro_f1": f1_score(y_treatment[test_idx], prediction, average="macro", zero_division=0),
                })
            treatment_cv = pd.DataFrame(treatment_cv_records)
            _save_csv(treatment_cv, "treatment_lda_cv.csv")

        if directories is not None:
            import joblib
            joblib.dump(lda_t, validate_output_path(Path(directories["models"]) / "lda_treatment.pkl", overwrite_existing_outputs))

        result["treatment_lda"] = {"model": lda_t, "X_lda": X_lda_t, "cv": treatment_cv}

    # ── Dose-level LDA + leave-one-plate-out CV + confusion matrix ──────
    if not has_dose:
        result["dose_lda"] = {"message": "Dose-response LDA skipped: no dose/concentration axis with >1 value"}
        lda_d = None
        X_lda_d = None
        dose_classes: list[Any] = []
    else:
        dose_classes = sorted(np.unique(y_dose))
        n_lda_d = min(len(dose_classes) - 1, X_latent.shape[1])
        lda_d = LinearDiscriminantAnalysis(n_components=n_lda_d, solver="svd")
        X_lda_d = lda_d.fit_transform(X_latent, y_dose)

        lda_coordinates_df = pd.DataFrame(
            X_lda_d, columns=[f"LDA{i + 1}" for i in range(X_lda_d.shape[1])], index=metadata_df.index,
        )
        lda_coordinates_df = pd.concat(
            [metadata_df[[treatment_col, plate_col, dose_col]], lda_coordinates_df], axis=1,
        )
        lda_coordinates_df["dose_label"] = dose_label.to_numpy()
        _save_csv(lda_coordinates_df, "lda_coordinates.csv")

        if n_groups < 2:
            dose_cv: pd.DataFrame | dict = {
                "message": "Leave-one-plate-out CV skipped: only 1 plate group present",
                "n_groups": n_groups,
            }
            cm_df = None
        else:
            dose_cv_records = []
            cm_sum = np.zeros((len(dose_classes), len(dose_classes)), dtype=int)
            for fold, (train_idx, test_idx) in enumerate(logo.split(X_latent, y_dose, groups), start=1):
                model = LinearDiscriminantAnalysis(n_components=n_lda_d, solver="svd")
                model.fit(X_latent[train_idx], y_dose[train_idx])
                prediction = model.predict(X_latent[test_idx])
                dose_cv_records.append({
                    "fold": fold,
                    "held_out_plate": groups[test_idx[0]],
                    "accuracy": accuracy_score(y_dose[test_idx], prediction),
                    "balanced_accuracy": balanced_accuracy_score(y_dose[test_idx], prediction),
                    "macro_f1": f1_score(y_dose[test_idx], prediction, average="macro", zero_division=0),
                })
                cm_sum += confusion_matrix(y_dose[test_idx], prediction, labels=dose_classes)
            dose_cv = pd.DataFrame(dose_cv_records)
            _save_csv(dose_cv, "dose_lda_cv.csv")
            cm_df = pd.DataFrame(cm_sum, index=dose_classes, columns=dose_classes)
            if directories is not None:
                cm_df.to_csv(
                    validate_output_path(Path(directories["results"]) / "confusion_matrix.csv", overwrite_existing_outputs)
                )

        if directories is not None:
            import joblib
            joblib.dump(lda_d, validate_output_path(Path(directories["models"]) / "lda_dose.pkl", overwrite_existing_outputs))

        result["dose_lda"] = {
            "model": lda_d, "X_lda": X_lda_d, "cv": dose_cv,
            "confusion_matrix": cm_df, "dose_classes": dose_classes,
        }

    # ── KMeans clustering ────────────────────────────────────────────────
    # Cluster on the dose-LDA representation when a dose axis is available
    # (matching the source notebook, which validates cluster purity against
    # dose identity); otherwise fall back to the treatment-LDA
    # representation so clustering still runs in dose-agnostic experiments.
    if has_dose and X_lda_d is not None:
        cluster_source = X_lda_d
        k_clusters = len(dose_classes)
        cluster_basis = "dose"
    elif n_classes_t >= 2:
        cluster_source = result["treatment_lda"]["X_lda"]
        k_clusters = n_classes_t
        cluster_basis = "treatment"
    else:
        cluster_source = None
        k_clusters = 0
        cluster_basis = None

    if cluster_source is None or k_clusters < 2 or cluster_source.shape[0] <= k_clusters:
        result["clustering"] = {"message": "KMeans clustering skipped: fewer than 2 usable clusters/classes"}
    else:
        kmeans = KMeans(n_clusters=k_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_source)
        cluster_df = metadata_df[[treatment_col, plate_col]].copy()
        if has_dose:
            cluster_df[dose_col] = metadata_df[dose_col]
        cluster_df["cluster"] = cluster_labels
        _save_csv(cluster_df, "clustering_results.csv")

        if directories is not None:
            import joblib
            joblib.dump(kmeans, validate_output_path(Path(directories["models"]) / "kmeans.pkl", overwrite_existing_outputs))

        result["clustering"] = {
            "model": kmeans, "labels": cluster_labels,
            "silhouette": _safe_silhouette(cluster_source, cluster_labels),
            "basis": cluster_basis,
        }

    # ── Summary (headline numbers for cross-space comparison) ───────────
    def _cv_mean(cv_obj, column):
        if isinstance(cv_obj, pd.DataFrame) and column in cv_obj:
            return float(cv_obj[column].mean())
        return np.nan

    treatment_cv_obj = result["treatment_lda"].get("cv") if isinstance(result.get("treatment_lda"), dict) else None
    dose_cv_obj = result["dose_lda"].get("cv") if isinstance(result.get("dose_lda"), dict) else None
    clustering_obj = result.get("clustering", {})

    summary = {
        "space": space_name,
        "n_wells": int(X_latent.shape[0]),
        "n_dimensions": int(X_latent.shape[1]),
        "treatment_cv_accuracy": _cv_mean(treatment_cv_obj, "accuracy"),
        "treatment_cv_balanced_accuracy": _cv_mean(treatment_cv_obj, "balanced_accuracy"),
        "treatment_cv_macro_f1": _cv_mean(treatment_cv_obj, "macro_f1"),
        "dose_cv_accuracy": _cv_mean(dose_cv_obj, "accuracy"),
        "dose_cv_balanced_accuracy": _cv_mean(dose_cv_obj, "balanced_accuracy"),
        "dose_cv_macro_f1": _cv_mean(dose_cv_obj, "macro_f1"),
        "cluster_silhouette": clustering_obj.get("silhouette", np.nan) if isinstance(clustering_obj, dict) else np.nan,
    }
    _save_csv(pd.DataFrame([summary]), "analysis_summary.csv")
    result["summary"] = summary

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cross-space comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_qc_metric(
    analysis_results: Mapping[str, Mapping[str, Any]],
    space_name: str,
    scope: str,
    factor: str,
    metric: str,
) -> float:
    """Look up one QC metric from a ``run_modelling_space`` result.

    ``analysis_results`` maps ``space_name -> run_modelling_space(...)``
    result dict (each of which has a ``"qc"`` DataFrame with ``scope`` and
    ``factor`` columns). Returns NaN if the requested row is unavailable.
    """
    table = analysis_results[space_name]["qc"]
    row = table.loc[(table["scope"] == scope) & (table["factor"] == factor)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][metric])


def safe_relative_change(
    new_value: float,
    reference_value: float,
    *,
    minimum_denominator: float = 1e-8,
) -> float:
    """Return ``(new - reference) / abs(reference)``.

    Returns NaN when either value is non-finite or the reference is too
    close to zero for a meaningful relative comparison.
    """
    if (
        not np.isfinite(new_value)
        or not np.isfinite(reference_value)
        or abs(reference_value) < minimum_denominator
    ):
        return float("nan")
    return (new_value - reference_value) / abs(reference_value)


def absolute_change(new_value: float, reference_value: float) -> float:
    """Return ``new_value - reference_value``, or NaN when either is non-finite."""
    if not np.isfinite(new_value) or not np.isfinite(reference_value):
        return float("nan")
    return new_value - reference_value
