"""Single-cell curation, outlier QC, and clustering/classification helpers.

Ported from ``06_single_cell_analysis.ipynb``, which duplicated its own
feature-cleaning logic twice (cells ~14 and ~25 -- the latter more complete:
identifier-column removal, low-variance removal, an IQR-stability guard
before ``RobustScaler``) and its cell-subsampling logic four times (cells
~32/36/40, byte-identical each time). This module keeps exactly one version
of each, used consistently by the marimo conversion of that notebook.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_NON_PHENOTYPIC_PATTERNS = (
    "Number_Object_Number",
    "ObjectNumber",
    "ImageNumber",
    "GroupNumber",
    "GroupIndex",
    "Parent_",
)


def mahal_outliers_within_well(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    plate_col: str,
    well_col: str,
    percentile: float = 99,
    random_state: int = 42,
    min_cells_per_well: int = 10,
) -> pd.Series:
    """Flag outlier cells via within-well Mahalanobis distance (SC-16).

    Returns a boolean Series aligned to ``df.index`` (``True`` = outlier).
    Wells with fewer than ``min_cells_per_well`` cells, or fewer than 2
    features, are skipped (left as ``False``), matching the source
    notebook's guard against `MinCovDet` failing on too-small groups.
    """
    from sklearn.covariance import MinCovDet

    outlier_mask = pd.Series(False, index=df.index)
    for (_plate, _well), grp in df.groupby([plate_col, well_col]):
        X_grp = grp[feature_cols].fillna(0).to_numpy()
        if X_grp.shape[0] < min_cells_per_well or X_grp.shape[1] < 2:
            continue
        try:
            mcd = MinCovDet(random_state=random_state, support_fraction=0.75)
            mcd.fit(X_grp)
            dist = mcd.mahalanobis(X_grp)
            threshold = np.percentile(dist, percentile)
            outlier_mask.loc[grp.index] = dist > threshold
        except Exception:  # noqa: BLE001 - matches source: skip wells where MCD fails
            pass
    return outlier_mask


def curate_single_cell_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    non_phenotypic_patterns: Sequence[str] = DEFAULT_NON_PHENOTYPIC_PATTERNS,
    min_unique: int = 3,
    scaling_clip: float = 20.0,
) -> tuple[np.ndarray, list[str], dict]:
    """Canonical single-cell feature-matrix preparation (cells ~25-26).

    Pipeline: drop identifier-like columns -> coerce numeric, inf->NaN ->
    drop features with < ``min_unique`` distinct values or zero/non-finite
    variance -> median-impute -> drop features with a near-zero IQR (which
    would blow up under `RobustScaler`) -> `RobustScaler` -> clip to
    +/- ``scaling_clip``.

    Returns ``(X_scaled, feature_cols_model, summary)`` where ``summary``
    reports how many features were dropped at each step and the final clip
    count, for the notebook to print/log.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    identifier_features = [
        c for c in feature_cols if any(p in c for p in non_phenotypic_patterns)
    ]
    feature_cols_model = [c for c in feature_cols if c not in identifier_features]

    X_raw = df[feature_cols_model].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    n_unique = X_raw.nunique(dropna=True)
    variance = X_raw.var(skipna=True)
    finite_variance = np.isfinite(variance)
    variable_mask = finite_variance & (n_unique >= min_unique) & (variance > 0)

    removed_low_variance = X_raw.columns[~variable_mask].tolist()
    feature_cols_model = X_raw.columns[variable_mask].tolist()
    X_raw = X_raw[feature_cols_model]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    q25 = np.percentile(X_imputed, 25, axis=0)
    q75 = np.percentile(X_imputed, 75, axis=0)
    iqr = q75 - q25
    iqr_tolerance = 1e-12
    stable_iqr_mask = np.isfinite(iqr) & (iqr > iqr_tolerance)

    removed_unstable_iqr = [
        feature_cols_model[i] for i in np.where(~stable_iqr_mask)[0]
    ]
    feature_cols_model = [
        feature_cols_model[i] for i in np.where(stable_iqr_mask)[0]
    ]
    X_imputed = X_imputed[:, stable_iqr_mask]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    if not np.isfinite(X_scaled).all():
        raise ValueError("Non-finite values remain after single-cell preprocessing.")

    n_clipped = int(np.sum((X_scaled < -scaling_clip) | (X_scaled > scaling_clip)))
    X_scaled = np.clip(X_scaled, -scaling_clip, scaling_clip)

    summary = {
        "n_identifier_features_removed": len(identifier_features),
        "n_low_variance_removed": len(removed_low_variance),
        "n_unstable_iqr_removed": len(removed_unstable_iqr),
        "n_features_final": len(feature_cols_model),
        "n_values_clipped": n_clipped,
    }
    return X_scaled, feature_cols_model, summary


def subsample_for_embedding(
    X: np.ndarray,
    df: pd.DataFrame,
    n_cells: int,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Simple (non-stratified) random subsample for PCA/UMAP/clustering.

    Ports the source notebook's ``rng.choice(..., replace=False)`` sampling
    (cells ~32/36/40, identical each time), used when the full single-cell
    matrix is too large for tractable dimensionality reduction. For
    stratified-by-treatment sampling, use
    :func:`hca_pipeline.modelling.balanced_sample` instead -- this function
    is for the separate "just make PCA/UMAP tractable" use case.
    """
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(X.shape[0], size=min(n_cells, X.shape[0]), replace=False)
    df_sampled = df.iloc[sample_idx].copy().reset_index(drop=True)
    return X[sample_idx], df_sampled


def sweep_hdbscan_params(
    X: np.ndarray,
    candidate_settings: Sequence[dict],
    *,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Compare HDBSCAN parameter combinations (cells ~66-68).

    ``candidate_settings`` is a list of ``{"min_cluster_size": ..., "min_samples": ...}``
    dicts. Returns one row per setting: cluster count, noise fraction,
    smallest/largest cluster size, median membership probability, and
    relative validity (if available).
    """
    import hdbscan

    records = []
    for params in candidate_settings:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric=metric,
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(X)
        clustered_mask = labels != -1
        cluster_sizes = pd.Series(labels[clustered_mask]).value_counts()

        records.append(
            {
                **params,
                "n_clusters": cluster_sizes.size,
                "noise_fraction": float(np.mean(labels == -1)),
                "clustered_cells": int(clustered_mask.sum()),
                "smallest_cluster": int(cluster_sizes.min()) if not cluster_sizes.empty else 0,
                "largest_cluster": int(cluster_sizes.max()) if not cluster_sizes.empty else 0,
                "median_membership_probability": (
                    float(np.median(clusterer.probabilities_[clustered_mask]))
                    if clustered_mask.any()
                    else np.nan
                ),
                "relative_validity": getattr(clusterer, "relative_validity_", np.nan),
            }
        )
    return pd.DataFrame(records)


def fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 30,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> np.ndarray:
    """Fit HDBSCAN with one parameter setting and return cluster labels (-1 = noise)."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
        prediction_data=True,
        # Keep notebook execution portable to CI, containers, and other
        # environments where process semaphores or CPU discovery are limited.
        core_dist_n_jobs=1,
    )
    return clusterer.fit_predict(X)


def train_lgbm_classifier_with_shap(
    X: np.ndarray,
    labels: Sequence,
    groups: Sequence,
    *,
    feature_names: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    early_stopping_rounds: int = 30,
    n_shap_samples: int = 500,
) -> dict:
    """Train a grouped-CV LightGBM classifier and compute SHAP importances (cells ~107-108).

    ``groups`` (e.g. well IDs) is used with `GroupShuffleSplit` so no well
    appears in both train and test (avoids well-level leakage).

    Returns a dict with keys ``classification_report`` (DataFrame),
    ``mean_abs_shap`` (Series, sorted descending), ``classifier``,
    ``label_encoder``, ``train_idx``, ``test_idx`` -- or, if `lightgbm`
    and/or `shap` aren't installed, a ``{"message": ...}`` sentinel dict
    (the established optional-dependency convention used throughout this
    pipeline) instead of raising.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return {"message": "lightgbm not installed - classifier skipped."}

    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(np.asarray(labels))
    groups_arr = np.asarray(groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups_arr))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    classifier = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=-1)],
    )

    y_pred = classifier.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    result = {
        "classification_report": pd.DataFrame(report).T,
        "classifier": classifier,
        "label_encoder": label_encoder,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_wells": set(groups_arr[train_idx]),
        "test_wells": set(groups_arr[test_idx]),
    }

    try:
        import shap
    except ImportError:
        result["shap_message"] = "shap not installed - SHAP importance skipped."
        return result

    explainer = shap.TreeExplainer(classifier)
    n_shap = min(n_shap_samples, len(X_test))
    rng = np.random.default_rng(random_state)
    shap_idx = rng.choice(len(X_test), size=n_shap, replace=False)
    shap_values = explainer.shap_values(X_test[shap_idx])

    # Multiclass TreeExplainer output shape differs across shap versions:
    # older releases return a list of per-class (n_samples, n_features)
    # arrays; newer releases return one (n_samples, n_features, n_classes)
    # array. Both reduce to a per-feature importance by averaging |value|
    # across classes, then across samples.
    if isinstance(shap_values, list):
        shap_abs = np.abs(np.stack(shap_values, axis=0)).mean(axis=0)
    elif np.asarray(shap_values).ndim == 3:
        shap_abs = np.abs(shap_values).mean(axis=-1)
    else:
        shap_abs = np.abs(shap_values)
    result["mean_abs_shap"] = pd.Series(
        shap_abs.mean(axis=0), index=list(feature_names), name="mean_abs_shap"
    ).sort_values(ascending=False)
    result["shap_sample_idx"] = shap_idx
    return result
