"""Auditable diagnostics and optional profile derivation for cell-count confounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class CellCountEvidence:
    classification: str
    max_abs_pc_correlation: float
    max_abs_control_pc_correlation: float | None
    low_count_fraction: float
    message: str


def pca_cell_count_associations(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    count_col: str,
    *,
    subset: str = "all_wells",
    n_components: int = 10,
) -> pd.DataFrame:
    """Return correlations of PCA coordinates with log10 cells per well."""
    if len(df) < 3:
        return pd.DataFrame(columns=["subset", "pc", "pearson_r", "variance_explained"])
    counts = pd.to_numeric(df[count_col], errors="coerce").to_numpy(float)
    matrix = df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    valid = np.isfinite(counts) & (counts > 0) & np.isfinite(matrix).all(axis=1)
    if valid.sum() < 3 or np.unique(counts[valid]).size < 2:
        return pd.DataFrame(columns=["subset", "pc", "pearson_r", "variance_explained"])
    matrix = matrix[valid]
    log_count = np.log10(counts[valid])
    n_components = min(int(n_components), matrix.shape[0], matrix.shape[1])
    coordinates = PCA(n_components=n_components, random_state=42).fit_transform(matrix)
    model = PCA(n_components=n_components, random_state=42).fit(matrix)
    return pd.DataFrame(
        {
            "subset": subset,
            "pc": [f"PC{i + 1}" for i in range(n_components)],
            "pearson_r": [float(np.corrcoef(coordinates[:, i], log_count)[0, 1]) for i in range(n_components)],
            "variance_explained": model.explained_variance_ratio_,
        }
    )


def classify_cell_count_evidence(
    all_wells: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    low_count_fraction: float,
    correlation_threshold: float = 0.40,
) -> CellCountEvidence:
    """Conservative gate: technical confounding requires evidence in controls."""
    all_max = float(all_wells["pearson_r"].abs().max()) if not all_wells.empty else 0.0
    control_max = float(controls["pearson_r"].abs().max()) if not controls.empty else None
    if control_max is not None and control_max >= correlation_threshold:
        classification = "technical_confounding_supported"
        message = "Cell count is associated with morphology inside negative controls; advanced correction may be justified."
    elif all_max >= correlation_threshold:
        classification = "biological_or_inconclusive"
        message = "The association is present globally but not established inside controls; do not regress it away automatically."
    else:
        classification = "no_strong_evidence"
        message = "No strong PCA association with cell count was detected; keep the canonical NB02 checkpoint."
    return CellCountEvidence(classification, all_max, control_max, float(low_count_fraction), message)


def control_based_residualize(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    count_col: str,
    control_mask: Sequence[bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residualize feature slopes fitted only in negative-control wells."""
    result = df.copy()
    counts = pd.to_numeric(df[count_col], errors="raise").to_numpy(float)
    controls = np.asarray(control_mask, dtype=bool)
    if controls.sum() < 3 or np.unique(counts[controls]).size < 2:
        raise ValueError("Control-based regression needs >=3 controls with varying positive cell counts.")
    log_count = np.log10(np.clip(counts, 1, None))
    reference = float(np.median(log_count[controls]))
    centered = log_count[controls] - log_count[controls].mean()
    denominator = float(centered @ centered)
    matrix = df[list(feature_cols)].to_numpy(float)
    slopes = (centered[:, None] * matrix[controls]).sum(axis=0) / denominator
    result.loc[:, list(feature_cols)] = matrix - (log_count - reference)[:, None] * slopes[None, :]
    audit = pd.DataFrame({"feature": list(feature_cols), "slope_per_log10_cell": slopes})
    return result, audit


def filter_wells(
    df: pd.DataFrame,
    count_col: str,
    minimum_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a table and return a complete kept/excluded audit table."""
    counts = pd.to_numeric(df[count_col], errors="coerce")
    keep = counts.ge(int(minimum_cells))
    audit_cols = [c for c in df.columns if c.startswith("Metadata_")]
    audit = df[audit_cols].copy()
    audit["cell_count"] = counts
    audit["minimum_cells"] = int(minimum_cells)
    audit["decision"] = np.where(keep, "kept", "excluded")
    audit["reason"] = np.where(keep, "meets_threshold", "below_manual_cell_count_threshold")
    return df.loc[keep].reset_index(drop=True), audit
