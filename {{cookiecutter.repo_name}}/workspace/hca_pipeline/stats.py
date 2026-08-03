"""Generic statistics helpers shared across LCP analysis notebooks.

Ports (and de-duplicates) the effect-size and multivariate-QC helpers that
were independently defined in ``03_phenotypic_profiling_parallel_spaces.ipynb``
and ``04_phenotypic_fingerprints.ipynb``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score

# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohens_d(
    group_values: Sequence[float] | np.ndarray | pd.Series,
    reference_values: Sequence[float] | np.ndarray | pd.Series,
    *,
    min_samples: int = 2,
) -> float:
    """Cohen's d with pooled standard deviation.

    NB03 and NB04 each defined their own copy. They agree on the pooled-
    variance formula but diverge in two ways:

    1. Non-finite handling: NB03 only dropped NaNs (``pd.Series.dropna``),
       leaving ``inf``/``-inf`` in the data to silently corrupt the variance
       calculation. NB04 filtered with ``np.isfinite`` before computing
       anything. NB04's approach is strictly safer and is used here.
    2. Degenerate pooled variance: when the pooled variance is zero or
       non-finite, NB03 returned ``0.0`` (implying "no effect"), while NB04
       returned ``NaN`` (implying "undefined"). A zero-variance pooled
       denominator means the true effect size is undefined (or unbounded, if
       the means differ) — reporting it as exactly ``0.0`` is misleading, so
       NB04's ``NaN`` behavior is kept as the more correct choice.
    """
    group_array = np.asarray(group_values, dtype=float)
    reference_array = np.asarray(reference_values, dtype=float)

    group_array = group_array[np.isfinite(group_array)]
    reference_array = reference_array[np.isfinite(reference_array)]

    n_group = group_array.size
    n_reference = reference_array.size

    if n_group < min_samples or n_reference < min_samples:
        return np.nan

    group_variance = np.var(group_array, ddof=1)
    reference_variance = np.var(reference_array, ddof=1)

    pooled_denominator = n_group + n_reference - 2
    if pooled_denominator <= 0:
        return np.nan

    pooled_variance = (
        (n_group - 1) * group_variance + (n_reference - 1) * reference_variance
    ) / pooled_denominator

    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_variance)
    return float((np.mean(group_array) - np.mean(reference_array)) / pooled_sd)


def effective_sample_size(n_treatment: int, n_control: int) -> float:
    """Weight for combining plate-specific effect sizes (harmonic-mean-like)."""
    total = n_treatment + n_control
    if total <= 0:
        return 0.0
    return (n_treatment * n_control) / total


def calculate_global_effects(
    data: pd.DataFrame,
    features: Sequence[str],
    *,
    condition_table: pd.DataFrame,
    negative_control_mask: pd.Series,
    treatment_column: str,
    concentration_column: str,
    condition_column: str = "Metadata_Condition",
    concentration_label_column: str = "Metadata_Concentration_Label",
    plate_column: str,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Effect of each condition against all negative controls (pooled).

    ``condition_table`` must have one row per condition with, at minimum,
    ``condition_column``, ``treatment_column``, ``concentration_column``, and
    ``concentration_label_column``.
    """
    control_df = data.loc[negative_control_mask, features]

    records = []
    for _, condition_row in condition_table.iterrows():
        condition_label = condition_row[condition_column]

        condition_mask = data[condition_column].eq(condition_label) & ~negative_control_mask
        condition_df = data.loc[condition_mask, features]

        for feature in features:
            effect = cohens_d(
                condition_df[feature].to_numpy(),
                control_df[feature].to_numpy(),
                min_samples=min_samples,
            )
            records.append(
                {
                    "condition": condition_label,
                    "treatment": condition_row[treatment_column],
                    "concentration": condition_row[concentration_column],
                    "concentration_label": condition_row[concentration_label_column],
                    "feature": feature,
                    "effect_size": effect,
                    "n_plates": int(data.loc[condition_mask, plate_column].nunique()),
                    "estimation_mode": "global",
                }
            )

    return pd.DataFrame(records)


def calculate_within_plate_effects(
    data: pd.DataFrame,
    features: Sequence[str],
    *,
    condition_table: pd.DataFrame,
    negative_control_mask: pd.Series,
    treatment_column: str,
    concentration_column: str,
    plate_column: str,
    condition_column: str = "Metadata_Condition",
    concentration_label_column: str = "Metadata_Concentration_Label",
    min_samples: int = 2,
) -> pd.DataFrame:
    """Per-plate Cohen's d, combined across plates via sample-size weights."""
    records = []
    for _, condition_row in condition_table.iterrows():
        condition_label = condition_row[condition_column]

        condition_mask = data[condition_column].eq(condition_label) & ~negative_control_mask
        condition_plates = data.loc[condition_mask, plate_column].dropna().unique()

        for feature in features:
            plate_effects: list[float] = []
            plate_weights: list[float] = []

            for plate in condition_plates:
                plate_condition = data.loc[
                    condition_mask & data[plate_column].eq(plate), feature
                ]
                plate_control = data.loc[
                    negative_control_mask & data[plate_column].eq(plate), feature
                ]

                effect = cohens_d(
                    plate_condition.to_numpy(),
                    plate_control.to_numpy(),
                    min_samples=min_samples,
                )

                if np.isfinite(effect):
                    plate_effects.append(effect)
                    plate_weights.append(
                        effective_sample_size(len(plate_condition), len(plate_control))
                    )

            if plate_effects:
                weights = np.asarray(plate_weights, dtype=float)
                if np.isfinite(weights).all() and weights.sum() > 0:
                    combined_effect = float(np.average(plate_effects, weights=weights))
                else:
                    combined_effect = float(np.mean(plate_effects))
            else:
                combined_effect = np.nan

            records.append(
                {
                    "condition": condition_label,
                    "treatment": condition_row[treatment_column],
                    "concentration": condition_row[concentration_column],
                    "concentration_label": condition_row[concentration_label_column],
                    "feature": feature,
                    "effect_size": combined_effect,
                    "n_plates": len(plate_effects),
                    "estimation_mode": "within_plate_weighted",
                }
            )

    return pd.DataFrame(records)


def build_fingerprint_matrix(
    effects: pd.DataFrame,
    *,
    grouping_level: str,
    metric_column: str,
    aggregation: str | Callable,
    condition_column: str = "condition",
    condition_order: Sequence[str] | None = None,
    drop_other: bool = True,
) -> pd.DataFrame:
    """Aggregate a per-feature effect-size table into a category x condition matrix.

    Placed here (rather than in ``taxonomy.py``) because it is fundamentally
    a statistical aggregation over an effects table, not a taxonomy lookup —
    it just happens to group by one of the taxonomy columns.
    """
    working = effects.dropna(subset=[grouping_level, metric_column]).copy()

    matrix = (
        working.groupby([grouping_level, condition_column], dropna=False)[metric_column]
        .agg(aggregation)
        .unstack(condition_column)
    )

    if condition_order is not None:
        ordered_columns = [c for c in condition_order if c in matrix.columns]
        matrix = matrix.reindex(columns=ordered_columns)

    if drop_other and "Other" in matrix.index:
        matrix = matrix.drop(index="Other", errors="ignore")

    return matrix


# ---------------------------------------------------------------------------
# Multivariate QC: silhouette, PERMANOVA, PERMDISP
# ---------------------------------------------------------------------------


def _safe_silhouette(matrix: np.ndarray, labels: Sequence) -> float:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.any(counts < 2) or len(labels) <= len(unique):
        return np.nan
    return float(silhouette_score(matrix, labels, metric="euclidean"))


def _one_factor_permanova(
    matrix: np.ndarray,
    labels: Sequence,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, float]:
    """One-factor PERMANOVA using Euclidean distances.

    Returns pseudo-F, permutation p-value, and R^2 (between-group SS / total SS).
    """
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    n = matrix.shape[0]
    groups = np.unique(labels)
    g = len(groups)

    if n < 3 or g < 2 or n <= g:
        return {"pseudo_F": np.nan, "p_value": np.nan, "R2": np.nan}

    distances = pairwise_distances(matrix, metric="euclidean")
    squared = distances**2
    ss_total = squared[np.triu_indices(n, k=1)].sum() / n

    def statistic(current_labels: np.ndarray) -> tuple[float, float]:
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
    matrix: np.ndarray,
    labels: Sequence,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, float]:
    """Permutation test for equality of multivariate dispersion (PERMDISP)."""
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = len(labels)
    g = len(unique)

    if n < 3 or g < 2 or n <= g:
        return {"pseudo_F": np.nan, "p_value": np.nan}

    def distances_to_group_centroids(current_labels: np.ndarray) -> np.ndarray:
        d = np.empty(n, dtype=float)
        for group in np.unique(current_labels):
            idx = np.flatnonzero(current_labels == group)
            centroid = matrix[idx].mean(axis=0)
            d[idx] = np.linalg.norm(matrix[idx] - centroid, axis=1)
        return d

    def anova_f(values: np.ndarray, current_labels: np.ndarray) -> float:
        grand = values.mean()
        ss_between = 0.0
        ss_within = 0.0
        for group in np.unique(current_labels):
            vals = values[current_labels == group]
            ss_between += len(vals) * (vals.mean() - grand) ** 2
            ss_within += ((vals - vals.mean()) ** 2).sum()
        return (ss_between / (g - 1)) / (ss_within / (n - g)) if ss_within > 0 else np.nan

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
    scope: str,
    factor: str,
    matrix: np.ndarray,
    labels: Sequence,
    *,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, object]:
    """Bundle silhouette, PERMANOVA, and PERMDISP results for one QC factor."""
    permanova = _one_factor_permanova(
        matrix, labels, permutations=permutations, random_state=random_state
    )
    permdisp = _one_factor_permdisp(
        matrix, labels, permutations=permutations, random_state=random_state
    )
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
