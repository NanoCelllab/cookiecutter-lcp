"""Per-plate feature normalization against negative controls.

Ports the normalization-specific logic of Section 6 of
``02_aggregate_normalize_featureselect_cv.ipynb`` (cells ~15-23, excluding the
unrelated scratch/image-QC cells and the feature-selection step, which lives
in the sibling ``feature_select.py``):

1. :func:`filter_high_missingness_features` and :func:`impute_missing_median`
   — the missingness filter + median-imputation step that precedes
   normalization (source cell ~18, first half).
2. :func:`resolve_normalization_control` and
   :func:`normalize_per_plate_mad_robustize` — the pycytominer-based
   per-plate ``mad_robustize`` normalization against a resolved negative
   control, including the control-selection fallback logic (source cell
   ~18, second half).

The source notebook hardcoded ``NEGCON_VALUES = ["negcon", "con"]`` as a
module-level constant. Here it is a required parameter on every function that
needs it, since a unified ``hca_pipeline.config.ExperimentConfig`` (built
elsewhere) will hold the real per-experiment vocabulary.

PascalCase metadata-column enforcement and plate-effect diagnostic plotting
that also lived in the source cells are intentionally not ported here — they
are metadata-naming and visualization concerns that belong to
``metadata.py`` / ``plotting.py`` respectively, not to normalization math.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "filter_high_missingness_features",
    "impute_missing_median",
    "clean_features_before_normalization",
    "drop_near_zero_mad_features",
    "resolve_normalization_control",
    "normalize_per_plate_mad_robustize",
    "drop_extreme_magnitude_features",
]


def filter_high_missingness_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    max_missing_fraction: float = 0.20,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop features whose fraction of missing (NaN/inf) values exceeds a threshold.

    Ports the source notebook's ``MAX_MISSING_FRACTION`` filter (cell ~18).

    Returns ``(df_without_dropped_columns, remaining_feature_cols, removed_feature_cols)``.
    """
    feature_values = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    missing_fraction = feature_values.isna().mean()
    removed = missing_fraction[missing_fraction > max_missing_fraction].index.tolist()

    remaining = [c for c in feature_cols if c not in removed]
    df_out = df.drop(columns=removed) if removed else df.copy()
    return df_out, remaining, removed


def impute_missing_median(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace ±inf with NaN, then median-impute remaining NaNs per feature.

    Falls back to filling with ``0`` for a feature that is entirely NaN
    (its median would itself be NaN), matching the source notebook's guard.

    Returns ``(df_imputed, summary)`` where ``summary`` reports
    ``n_missing_before`` and ``n_missing_after``.
    """
    df_out = df.copy()
    for c in feature_cols:
        df_out[c] = df_out[c].replace([np.inf, -np.inf], np.nan)

    n_before = int(df_out[feature_cols].isna().sum().sum())
    if n_before > 0:
        for c in feature_cols:
            med = df_out[c].median()
            if pd.isna(med):
                df_out[c] = df_out[c].fillna(0)
            else:
                df_out[c] = df_out[c].fillna(med)
    n_after = int(df_out[feature_cols].isna().sum().sum())

    return df_out, {"n_missing_before": n_before, "n_missing_after": n_after}


def clean_features_before_normalization(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    max_missing_fraction: float = 0.20,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Convenience wrapper combining the missingness filter and median imputation.

    Returns ``(cleaned_df, remaining_feature_cols, summary)`` where
    ``summary`` combines the removed-features list with the imputation counts.
    """
    df_filtered, remaining, removed = filter_high_missingness_features(
        df, feature_cols, max_missing_fraction=max_missing_fraction,
    )
    df_imputed, impute_summary = impute_missing_median(df_filtered, remaining)

    summary = {
        "removed_features": removed,
        "n_features_removed": len(removed),
        "n_features_remaining": len(remaining),
        **impute_summary,
    }
    return df_imputed, remaining, summary


def drop_near_zero_mad_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    plate_col: str,
    control_col: str,
    negcon_values: Sequence[str],
    norm_control: str,
    mad_epsilon: float = 1e-3,
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """Drop features whose negative-control MAD is near-zero in any plate.

    ``mad_robustize`` divides each feature by its negative-control MAD. A
    feature that is (near-)constant within a given plate's negative controls
    has a MAD near zero there, and pycytominer's small fixed epsilon in that
    division is not enough to prevent the result from blowing up to an
    enormous, numerically meaningless magnitude (observed on a real dataset:
    a vesicle/nucleolus child-object count feature, near-constant per plate,
    reached ~1e19 after normalization). That corrupted feature then dominates
    any downstream correlation/cosine similarity computation.

    This must run *before* normalization, not after: once the blow-up has
    happened, the feature's variance looks artificially *high*, so a
    post-normalization ``variance_threshold`` feature-selection step cannot
    catch it -- it only removes low-variance features.

    Returns ``(df_without_dropped_columns, remaining_feature_cols, removed_report)``,
    where ``removed_report`` has one ``{"feature", "plate", "mad"}`` entry per
    (feature, plate) combination that triggered removal, for auditability.
    """
    norm_ctrl = resolve_normalization_control(df, control_col, negcon_values, norm_control)

    near_zero_mad_features: dict[str, list[dict]] = {}
    for plate, sub in df.groupby(plate_col, sort=False):
        controls = sub.loc[sub[control_col] == norm_ctrl, feature_cols]
        if controls.empty:
            continue
        mad = (controls - controls.median()).abs().median()
        for feature, value in mad.items():
            if value < mad_epsilon:
                near_zero_mad_features.setdefault(feature, []).append(
                    {"feature": feature, "plate": plate, "mad": float(value)}
                )

    removed = sorted(near_zero_mad_features)
    removed_report = [record for records in near_zero_mad_features.values() for record in records]
    remaining = [c for c in feature_cols if c not in near_zero_mad_features]
    df_out = df.drop(columns=removed) if removed else df.copy()
    return df_out, remaining, removed_report


def resolve_normalization_control(
    df: pd.DataFrame,
    control_col: str,
    negcon_values: Sequence[str],
    norm_control: str,
) -> str:
    """Resolve which control-type value to normalize each plate against.

    Fallback order (matching the source notebook exactly):
      1. ``norm_control`` if it is present among the dataset's control types.
      2. Otherwise the first of ``negcon_values`` that is present.
      3. Otherwise raise ``ValueError`` — there is no silent global fallback.
    """
    available_ctrls = df[control_col].dropna().unique() if control_col in df.columns else []
    available_candidates = [v for v in negcon_values if v in available_ctrls]

    if norm_control in available_ctrls:
        return norm_control
    if available_candidates:
        return available_candidates[0]
    raise ValueError(
        f"No valid negative-control wells found for normalisation. "
        f"norm_control='{norm_control}', negcon_values={list(negcon_values)}, "
        f"available control types={list(available_ctrls)}."
    )


def normalize_per_plate_mad_robustize(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | str,
    *,
    plate_col: str,
    control_col: str,
    negcon_values: Sequence[str],
    norm_control: str,
    method: str = "mad_robustize",
    min_control_wells: int = 5,
) -> pd.DataFrame:
    """Normalize each plate independently against its own negative-control wells.

    Ports the source notebook's per-plate ``pycytominer.normalize`` loop
    (cell ~18, "Fix 1.5"): each plate is checked to have at least
    ``min_control_wells`` wells of the resolved control type; if not, a
    ``ValueError`` is raised rather than silently falling back to a
    cross-plate/global control pool.

    ``feature_cols`` may be an explicit list of feature-column names or the
    pycytominer sentinel ``"infer"``.
    """
    from pycytominer import normalize as pc_normalize

    norm_ctrl = resolve_normalization_control(df, control_col, negcon_values, norm_control)

    norm_parts = []
    for plate, sub in df.groupby(plate_col, sort=False):
        n_ctrl = int((sub[control_col] == norm_ctrl).sum()) if control_col in sub.columns else 0
        if n_ctrl < min_control_wells:
            raise ValueError(
                f"Plate '{plate}' has only {n_ctrl} negative-control wells "
                f"(control_type='{norm_ctrl}'). At least {min_control_wells} are "
                f"required for robust per-plate normalisation. Consider merging "
                f"plates or adjusting the control configuration."
            )
        sub_norm = pc_normalize(
            profiles=sub,
            features=feature_cols,
            meta_features="infer",
            samples=f"{control_col} == '{norm_ctrl}'",
            method=method,
        )
        norm_parts.append(sub_norm)

    return pd.concat(norm_parts, axis=0, ignore_index=True)


def drop_extreme_magnitude_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    max_abs_value: float = 100.0,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Drop normalized features whose magnitude is implausibly large.

    A safety net independent of *why* a feature ended up this way --
    :func:`drop_near_zero_mad_features` prevents the specific near-zero-MAD
    cause, but this check protects against any other future cause of the
    same symptom. A genuinely MAD-robustized feature should be a small
    number even for a strong biological outlier; anything past
    ``max_abs_value`` is a numerical artifact, not biology, and would
    otherwise dominate any downstream correlation/distance computation.

    Returns ``(df_without_dropped_columns, remaining_feature_cols, removed_with_max_abs)``.
    """
    max_abs = df[feature_cols].abs().max()
    removed = max_abs[max_abs > max_abs_value].sort_values(ascending=False)
    remaining = [c for c in feature_cols if c not in removed.index]
    df_out = df.drop(columns=removed.index.tolist()) if not removed.empty else df.copy()
    return df_out, remaining, removed.to_dict()
