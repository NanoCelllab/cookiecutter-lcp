import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 06 — Single-Cell Analysis

    **Pipeline step:** 6 of 6
    **Input:** `single_cell_ready.parquet` (from NB02 cache — annotated, QC-cleaned, pre-aggregation)
    **Output:** Single-cell PCA/UMAP/clustering figures, LightGBM classification report, SHAP CSVs

    > **Memory note:** this notebook can load hundreds of thousands of
    > single cells with hundreds of features. If you hit an out-of-memory
    > error, lower "Max cells per treatment" in the config below.

    ### What this notebook does

    1. Loads single-cell profiles and runs QC audits (SC-14, SC-15, SC-16).
    2. Curates features: drops identifiers/low-variance columns, imputes,
       scales with `RobustScaler`.
    3. Balanced sampling stratified by treatment.
    4. PCA and (if installed) UMAP on the sampled single-cell data.
    5. HDBSCAN and KMeans clustering on the top PCs.
    6. LightGBM classifier (treatment label) with `GroupShuffleSplit` by well.
    7. SHAP feature importance + cross-check with NB04 LDA loadings (SC-17).

    ### Fixes relative to the original notebook

    - **De-duplicated feature curation and sampling.** The source notebook
      computed its feature-cleaning logic twice (a simpler pass, then a
      more complete one a few cells later) and re-ran cell subsampling
      four times with identical logic — remnants of iterative
      development. This version calls each step exactly once, via
      `hca_pipeline.single_cell`.
    - **Fixed a real bug: balanced sampling was silently discarded.** The
      source notebook's own docs state step 3 is "balanced sampling
      stratified by treatment," and it does compute one — but the very
      same variable names (`df_sampled`/`X_sampled`) get reassigned by
      later cells to a *plain random* subsample instead, so the balanced
      sample was never actually used for PCA, clustering, or the LightGBM
      classifier. This version performs the balanced, per-treatment-capped
      sample exactly once and uses it consistently everywhere downstream.
    - **The "phenotypic recovery axis" section (a bespoke geometric
      analysis assuming two specific reference-state treatment labels)
      has been extracted into its own optional notebook** rather than
      hard-required here — see the notebook alongside this one.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 0 — Imports and shared utilities
    """)
    return


@app.cell
def _():
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    try:
        import umap.umap_ as umap

        UMAP_OK = True
    except ImportError:
        UMAP_OK = False

    try:
        import hdbscan

        HDBSCAN_OK = True
    except ImportError:
        HDBSCAN_OK = False

    sns.set_theme(context="talk", style="whitegrid", font_scale=1.0)
    plt.rcParams.update(
        {"figure.dpi": 120, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False}
    )
    return (
        HDBSCAN_OK,
        KMeans,
        PCA,
        UMAP_OK,
        np,
        pd,
        plt,
        silhouette_score,
        sns,
        umap,
    )


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell
def _(Path):
    import sys

    _notebook_path = Path(__file__).resolve()
    _repo_root_candidate = None
    for _p in (_notebook_path, *_notebook_path.parents):
        if (_p / "pixi.toml").exists() or (_p / ".git").exists() or (_p / "pixi.lock").exists():
            _repo_root_candidate = _p
            break
    if _repo_root_candidate is None:
        raise FileNotFoundError(
            "Could not find repo root (pixi.toml / .git / pixi.lock) starting "
            f"from {_notebook_path}. Ensure this notebook lives inside the project."
        )

    REPO_ROOT = _repo_root_candidate
    _pipelines_dir = REPO_ROOT / "workspace"
    if not (_pipelines_dir / "hca_pipeline").exists():
        raise FileNotFoundError(f"hca_pipeline package directory not found: {_pipelines_dir / 'hca_pipeline'}")
    sys.path.insert(0, str(_pipelines_dir))

    from hca_pipeline.config import ExperimentConfig
    from hca_pipeline.feature_select import infer_feature_cols
    from hca_pipeline.io import write_csv_protected, write_parquet_protected
    from hca_pipeline.modelling import balanced_sample
    from hca_pipeline.single_cell import (
        curate_single_cell_features,
        fit_hdbscan,
        mahal_outliers_within_well,
        subsample_for_embedding,
        sweep_hdbscan_params,
        train_lgbm_classifier_with_shap,
    )

    print(f"  ✓  Shared utilities loaded from hca_pipeline ({_pipelines_dir})")
    return (
        ExperimentConfig,
        REPO_ROOT,
        balanced_sample,
        curate_single_cell_features,
        fit_hdbscan,
        mahal_outliers_within_well,
        train_lgbm_classifier_with_shap,
        write_parquet_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration
    """)
    return


@app.cell
def _(REPO_ROOT, mo):
    _backend_dir = REPO_ROOT / "workspace" / "backend"
    available_experiment_ids = (
        sorted(p.name for p in _backend_dir.iterdir() if p.is_dir()) if _backend_dir.is_dir() else []
    )
    _default_experiment_id = (
        available_experiment_ids[0] if available_experiment_ids else "SET_EXPERIMENT_ID_HERE"
    )
    experiment_id_input = mo.ui.dropdown(
        options=available_experiment_ids or [_default_experiment_id],
        value=_default_experiment_id,
        label="Experiment ID",
    )
    experiment_id_input
    return (experiment_id_input,)


@app.cell
def _(ExperimentConfig, REPO_ROOT, experiment_id_input):
    loaded_config = ExperimentConfig.load(REPO_ROOT, experiment_id_input.value)
    return (loaded_config,)


@app.cell
def _(mo):
    n_cells_per_treatment_input = mo.ui.number(
        value=5000, start=100, stop=1_000_000,
        label="Max cells per treatment (balanced sample used for PCA/UMAP/clustering/classification)",
    )
    n_pca_components_input = mo.ui.number(value=50, start=2, stop=500, label="PCA components")
    nan_threshold_input = mo.ui.number(
        value=0.05, start=0.0, stop=1.0, step=0.01, label="SC-15: max feature NaN fraction"
    )
    mahal_percentile_input = mo.ui.number(
        value=99, start=50, stop=100, label="SC-16: Mahalanobis outlier percentile"
    )
    hdbscan_min_cluster_size_input = mo.ui.number(value=30, start=2, stop=1000, label="HDBSCAN min_cluster_size")
    hdbscan_min_samples_input = mo.ui.number(value=5, start=1, stop=200, label="HDBSCAN min_samples")
    kmeans_k_max_input = mo.ui.number(value=15, start=3, stop=50, label="KMeans: max k to try")
    overwrite_input = mo.ui.checkbox(value=False, label="Overwrite existing outputs")
    save_history_input = mo.ui.checkbox(value=False, label="Save timestamped provenance history")

    mo.vstack(
        [
            n_cells_per_treatment_input,
            n_pca_components_input,
            nan_threshold_input,
            mahal_percentile_input,
            hdbscan_min_cluster_size_input,
            hdbscan_min_samples_input,
            kmeans_k_max_input,
            overwrite_input,
            save_history_input,
        ]
    )
    return (
        hdbscan_min_cluster_size_input,
        hdbscan_min_samples_input,
        kmeans_k_max_input,
        mahal_percentile_input,
        n_cells_per_treatment_input,
        n_pca_components_input,
        nan_threshold_input,
        overwrite_input,
        save_history_input,
    )


@app.cell
def _(REPO_ROOT, experiment_id_input, loaded_config):
    from dataclasses import replace

    EXPERIMENT_ID = experiment_id_input.value
    CONFIG = replace(loaded_config, experiment_id=EXPERIMENT_ID)
    CONFIG.save(REPO_ROOT)
    print(f"  Experiment ID: {EXPERIMENT_ID}")
    return CONFIG, EXPERIMENT_ID


@app.cell
def _(EXPERIMENT_ID, REPO_ROOT):
    WORKSPACE_DIR = REPO_ROOT / "workspace"
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    PROFILES_DIR = WORKSPACE_DIR / "profiles" / EXPERIMENT_ID

    PROFILES_OUT_DIR = PROFILES_DIR / "outputs"
    CACHE_DIR = PROFILES_OUT_DIR / "cache"
    FIGS_DIR = ANALYSIS_DIR / "figures" / "single_cell"
    RESULTS_DIR = ANALYSIS_DIR / "results"

    for _d in (FIGS_DIR, RESULTS_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    INPUT_PARQUET = CACHE_DIR / "single_cell_ready.parquet"
    LDA_LOADINGS_CSV = RESULTS_DIR / "lda_loadings.csv"
    SHAP_SUMMARY_CSV = RESULTS_DIR / "shap_summary.csv"
    CLF_REPORT_CSV = RESULTS_DIR / "lgbm_classification_report.csv"
    GO_NOGO_REPORT = RESULTS_DIR / "quality_metrics" / "quality_metrics_report.md"

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PARQUET}\nRun NB02 first.")

    print(f"  Input:  {INPUT_PARQUET}")
    return (
        CLF_REPORT_CSV,
        FIGS_DIR,
        GO_NOGO_REPORT,
        INPUT_PARQUET,
        LDA_LOADINGS_CSV,
        RESULTS_DIR,
        SHAP_SUMMARY_CSV,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Load data
    """)
    return


@app.cell
def _(CONFIG, GO_NOGO_REPORT, INPUT_PARQUET, pd):
    df_sc = pd.read_parquet(INPUT_PARQUET)
    df_sc = df_sc.loc[:, ~df_sc.columns.str.contains("ImageNumber|ObjectNumber")]

    RESOLVED_CONFIG = CONFIG.resolve_columns(df_sc)

    if GO_NOGO_REPORT.exists():
        print("── NB03 Go/No-Go quality gate ──")
        report_text = GO_NOGO_REPORT.read_text()
        if "NO-GO" in report_text.upper():
            print("  ⚠️  NB03 returned NO-GO. Single-cell analysis may be unreliable.")
        else:
            print("  ✓  NB03 quality gate passed (or not NO-GO).")
    else:
        print("  ℹ️  NB03 Go/No-Go report not found yet.")

    feat_cols_raw = [c for c in df_sc.columns if c.startswith(("Cells_", "Cytoplasm_", "Nuclei_", "Vesicles_")) and not c.startswith("Metadata_")]
    print(f"  Shape: {df_sc.shape[0]:,} cells × {df_sc.shape[1]} cols")
    print(f"  Feature columns: {len(feat_cols_raw)}")
    print(f"  Treatments: {df_sc[RESOLVED_CONFIG.treatment_col].nunique()}")
    return RESOLVED_CONFIG, df_sc, feat_cols_raw


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Sanity checks (SC-14 → SC-16)

    - **SC-14:** cell-count-per-well distribution (violin, per treatment).
    - **SC-15:** feature completeness (% NaN per feature).
    - **SC-16:** outlier cells (Mahalanobis distance > configured
      percentile, computed within each well on a quick 20-component PCA).
    """)
    return


@app.cell
def _(FIGS_DIR, RESOLVED_CONFIG, df_sc, plt, sns):
    counts_per_well = (
        df_sc.groupby([RESOLVED_CONFIG.plate_col, RESOLVED_CONFIG.well_col, RESOLVED_CONFIG.treatment_col])
        .size()
        .reset_index(name="n_cells")
    )
    _order = counts_per_well.groupby(RESOLVED_CONFIG.treatment_col)["n_cells"].median().sort_values(ascending=False).index
    fig_sc14, ax_sc14 = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=counts_per_well, x=RESOLVED_CONFIG.treatment_col, y="n_cells",
        hue=RESOLVED_CONFIG.treatment_col, order=_order, hue_order=_order,
        palette="tab10", legend=False, ax=ax_sc14, inner="box", cut=0,
    )
    ax_sc14.set_xlabel("Treatment")
    ax_sc14.set_ylabel("Cells per well")
    ax_sc14.set_title("SC-14: Cell count distribution per well")
    plt.setp(ax_sc14.get_xticklabels(), rotation=30, ha="right")
    fig_sc14.tight_layout()
    fig_sc14.savefig(FIGS_DIR / "sc14_cell_count_violin.png", dpi=150, bbox_inches="tight")
    plt.close(fig_sc14)
    print(f"  ✓  Saved: sc14_cell_count_violin.png (median cells/well: {counts_per_well['n_cells'].median():.0f})")
    fig_sc14
    return


@app.cell
def _(df_sc, feat_cols_raw, nan_threshold_input):
    nan_frac = df_sc[feat_cols_raw].isna().mean()
    _nan_threshold = float(nan_threshold_input.value)
    n_high_nan = int((nan_frac > _nan_threshold).sum())
    print(f"── SC-15: Feature completeness ──")
    print(f"  Features with > {_nan_threshold:.0%} NaN: {n_high_nan} / {len(feat_cols_raw)}")
    print(f"  SC-15: {'WARN — dropping high-NaN features' if n_high_nan > 0 else 'PASS'}")
    return


@app.cell
def _(
    PCA,
    RESOLVED_CONFIG,
    df_sc,
    feat_cols_raw,
    mahal_outliers_within_well,
    mahal_percentile_input,
    np,
    pd,
):
    X_sc_raw_for_qc = df_sc[feat_cols_raw].fillna(0).replace([np.inf, -np.inf], 0).to_numpy()
    pca_qc = PCA(n_components=min(20, X_sc_raw_for_qc.shape[1]), random_state=42)
    X_sc_pca_qc = pca_qc.fit_transform(X_sc_raw_for_qc)

    df_pca_qc = pd.DataFrame(
        X_sc_pca_qc, index=df_sc.index, columns=[f"PC{i + 1}" for i in range(X_sc_pca_qc.shape[1])]
    )
    df_pca_qc[RESOLVED_CONFIG.plate_col] = df_sc[RESOLVED_CONFIG.plate_col].to_numpy()
    df_pca_qc[RESOLVED_CONFIG.well_col] = df_sc[RESOLVED_CONFIG.well_col].to_numpy()

    _mahal_percentile = float(mahal_percentile_input.value)
    outlier_mask = mahal_outliers_within_well(
        df_pca_qc,
        [f"PC{i + 1}" for i in range(X_sc_pca_qc.shape[1])],
        plate_col=RESOLVED_CONFIG.plate_col,
        well_col=RESOLVED_CONFIG.well_col,
        percentile=_mahal_percentile,
    )
    n_outliers = int(outlier_mask.sum())
    pct_outliers = n_outliers / len(df_sc) * 100
    print(f"── SC-16: Outlier cells (Mahalanobis, within-well) ──")
    print(f"  Outlier cells (>{_mahal_percentile:.0f}th pct): {n_outliers:,} ({pct_outliers:.1f}%)")
    print(f"  SC-16: {'WARN — check for imaging artefacts' if pct_outliers > 5 else 'PASS'}")

    df_sc_clean = df_sc.loc[~outlier_mask].copy()
    print(f"  After outlier removal: {df_sc_clean.shape[0]:,} cells")
    return (df_sc_clean,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Feature curation and balanced sampling

    One canonical curation pass (identifier removal → low-variance
    removal → median imputation → IQR-stability guard → `RobustScaler`
    → clip), followed by one balanced (per-treatment-capped) sample —
    replacing the original notebook's duplicated curation logic and its
    silently-discarded balanced-sampling step (see the fix noted above).
    """)
    return


@app.cell
def _(curate_single_cell_features, df_sc_clean, feat_cols_raw):
    X_sc_full, feature_cols_model, curation_summary = curate_single_cell_features(df_sc_clean, feat_cols_raw)
    print("── Feature curation ──")
    for _key, _value in curation_summary.items():
        print(f"  {_key}: {_value}")
    return X_sc_full, feature_cols_model


@app.cell
def _(
    RESOLVED_CONFIG,
    X_sc_full,
    balanced_sample,
    df_sc_clean,
    n_cells_per_treatment_input,
):
    df_sampled, X_sampled = balanced_sample(
        df_sc_clean, X_sc_full, RESOLVED_CONFIG.treatment_col,
        int(n_cells_per_treatment_input.value), random_state=42,
    )
    print(f"  Balanced sample: {X_sampled.shape[0]:,} cells × {X_sampled.shape[1]} features")
    print(df_sampled[RESOLVED_CONFIG.treatment_col].value_counts().to_string())
    return X_sampled, df_sampled


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — PCA and UMAP (single-cell)
    """)
    return


@app.cell
def _(FIGS_DIR, PCA, X_sampled, n_pca_components_input, np, plt):
    n_components_sc = min(int(n_pca_components_input.value), X_sampled.shape[0] - 1, X_sampled.shape[1])
    pca_sc = PCA(n_components=n_components_sc, random_state=42)
    X_sc_pca = pca_sc.fit_transform(X_sampled)

    _n_show = min(30, X_sc_pca.shape[1])
    _evr = pca_sc.explained_variance_ratio_[:_n_show]
    fig_scree, ax_scree = plt.subplots(figsize=(9, 4))
    ax_scree.bar(range(1, _n_show + 1), _evr * 100, color="#1976d2", alpha=0.8)
    ax_scree2 = ax_scree.twinx()
    ax_scree2.plot(range(1, _n_show + 1), np.cumsum(_evr) * 100, "o-", color="#d32f2f", markersize=4, linewidth=1.5)
    ax_scree2.axhline(80, color="grey", linestyle="--", linewidth=1)
    ax_scree2.set_ylabel("Cumulative variance (%)", color="#d32f2f")
    ax_scree.set_xlabel("Principal component")
    ax_scree.set_ylabel("Variance explained (%)")
    ax_scree.set_title("Single-cell PCA scree plot")
    fig_scree.tight_layout()
    fig_scree.savefig(FIGS_DIR / "sc_pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close(fig_scree)
    print(f"  PC1: {_evr[0]:.1%}  |  PC2: {_evr[1]:.1%}  |  Top-{_n_show}: {_evr.sum():.1%}")
    fig_scree
    return X_sc_pca, pca_sc


@app.cell
def _(FIGS_DIR, RESOLVED_CONFIG, X_sc_pca, df_sampled, np, pca_sc, plt, sns):
    fig_pca_scatter, axes_pca_scatter = plt.subplots(1, 2, figsize=(14, 5))
    _treatments = df_sampled[RESOLVED_CONFIG.treatment_col].astype(str).to_numpy()
    _plates = df_sampled[RESOLVED_CONFIG.plate_col].astype(str).to_numpy()

    for _ax, _labels, _title in [(axes_pca_scatter[0], _treatments, "Treatment"), (axes_pca_scatter[1], _plates, "Plate (batch check)")]:
        _unique_labels = sorted(np.unique(_labels))
        _palette = sns.color_palette("tab10", n_colors=len(_unique_labels))
        for _label, _color in zip(_unique_labels, _palette):
            _mask = _labels == _label
            _ax.scatter(X_sc_pca[_mask, 0], X_sc_pca[_mask, 1], label=_label, color=_color, s=10, alpha=0.45, linewidths=0, rasterized=True)
        _ax.set_xlabel(f"PC1 ({pca_sc.explained_variance_ratio_[0]:.1%})")
        _ax.set_ylabel(f"PC2 ({pca_sc.explained_variance_ratio_[1]:.1%})")
        _ax.set_title(_title)
        _ax.legend(fontsize=7, markerscale=2, frameon=True, loc="best")

    fig_pca_scatter.suptitle("Single-cell PCA", fontsize=13, fontweight="bold")
    fig_pca_scatter.tight_layout()
    fig_pca_scatter.savefig(FIGS_DIR / "sc_pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pca_scatter)
    print("✓ Saved: sc_pca_scatter.png")
    fig_pca_scatter
    return


@app.cell
def _(RESOLVED_CONFIG, X_sc_pca, df_sampled, silhouette_score):
    sil_plate = silhouette_score(X_sc_pca, df_sampled[RESOLVED_CONFIG.plate_col])
    sil_treatment = silhouette_score(X_sc_pca, df_sampled[RESOLVED_CONFIG.treatment_col])
    print(f"  Silhouette by plate (near 0 = good mixing): {sil_plate:.4f}")
    print(f"  Silhouette by treatment (further from 0 = more separation): {sil_treatment:.4f}")
    return


@app.cell
def _(RESOLVED_CONFIG, UMAP_OK, X_sc_pca, df_sampled, umap):
    if UMAP_OK:
        umap_model = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="euclidean", n_components=2, random_state=42, transform_seed=42, low_memory=True)
        X_sc_umap = umap_model.fit_transform(X_sc_pca)
        umap_df = df_sampled.reset_index(drop=True).copy()
        umap_df["UMAP1"] = X_sc_umap[:, 0]
        umap_df["UMAP2"] = X_sc_umap[:, 1]
        if RESOLVED_CONFIG.has_dose_axis:
            umap_df[RESOLVED_CONFIG.concentration_col] = __import__("pandas").to_numeric(
                umap_df[RESOLVED_CONFIG.concentration_col], errors="coerce"
            )
        print(f"  UMAP coordinates: {X_sc_umap.shape}")
    else:
        umap_df = df_sampled.reset_index(drop=True).copy()
        print("  ⚠️  UMAP skipped (umap-learn not installed).")
    return (umap_df,)


@app.cell
def _(FIGS_DIR, RESOLVED_CONFIG, UMAP_OK, plt, sns, umap_df):
    if UMAP_OK:
        fig_umap, axes_umap = plt.subplots(1, 2 + int(RESOLVED_CONFIG.has_dose_axis), figsize=(20 if RESOLVED_CONFIG.has_dose_axis else 14, 5.5))

        for _ax, _col, _title in [(axes_umap[0], RESOLVED_CONFIG.treatment_col, "Treatment"), (axes_umap[1], RESOLVED_CONFIG.plate_col, "Plate — batch check")]:
            _values = umap_df[_col].astype(str)
            _levels = sorted(_values.dropna().unique())
            _palette = sns.color_palette("tab10", len(_levels))
            for _level, _color in zip(_levels, _palette):
                _mask = _values.eq(_level).to_numpy()
                _ax.scatter(umap_df.loc[_mask, "UMAP1"], umap_df.loc[_mask, "UMAP2"], label=_level, color=_color, s=7, alpha=0.45, linewidths=0, rasterized=True)
            _ax.set_title(_title)
            _ax.legend(fontsize=7, markerscale=2.5, frameon=True)

        if RESOLVED_CONFIG.has_dose_axis:
            import pandas as _pd_umap

            _conc = _pd_umap.to_numeric(umap_df[RESOLVED_CONFIG.concentration_col], errors="coerce")
            _scatter = axes_umap[2].scatter(umap_df["UMAP1"], umap_df["UMAP2"], c=_conc, cmap="viridis", s=7, alpha=0.5, linewidths=0, rasterized=True)
            axes_umap[2].set_title("Concentration")
            fig_umap.colorbar(_scatter, ax=axes_umap[2], fraction=0.046, pad=0.04, label="Concentration")

        for _ax in axes_umap:
            _ax.set_xlabel("UMAP 1")
            _ax.set_ylabel("UMAP 2")
        fig_umap.suptitle("Single-cell UMAP — PCA input space", fontsize=15, fontweight="bold")
        fig_umap.tight_layout()
        fig_umap.savefig(FIGS_DIR / "sc_umap_overview.png", dpi=200, bbox_inches="tight")
        plt.close(fig_umap)
        print("✓ Saved: sc_umap_overview.png")
        _display = fig_umap
    else:
        _display = None
    _display
    return


@app.cell
def _(RESULTS_DIR, overwrite_input, umap_df, write_parquet_protected):
    umap_export_status = write_parquet_protected(
        umap_df, RESULTS_DIR / "single_cell_umap_coordinates.parquet", overwrite=bool(overwrite_input.value)
    )
    print(f"✓ UMAP coordinates {umap_export_status}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Clustering: HDBSCAN and KMeans
    """)
    return


@app.cell
def _(X_sc_pca):
    # Top-20 PCs used for clustering (both HDBSCAN and KMeans below), kept
    # independent of hdbscan's availability since KMeans doesn't need it.
    X_cluster = X_sc_pca[:, : min(20, X_sc_pca.shape[1])]
    return (X_cluster,)


@app.cell
def _(
    HDBSCAN_OK,
    RESOLVED_CONFIG,
    X_cluster,
    fit_hdbscan,
    hdbscan_min_cluster_size_input,
    hdbscan_min_samples_input,
    pd,
    umap_df,
):
    if HDBSCAN_OK:
        hdbscan_labels = fit_hdbscan(
            X_cluster,
            min_cluster_size=int(hdbscan_min_cluster_size_input.value),
            min_samples=int(hdbscan_min_samples_input.value),
        )
        umap_df_clustered = umap_df.copy()
        umap_df_clustered["Cluster"] = hdbscan_labels

        n_clusters_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        print("── HDBSCAN summary ──")
        print(f"  Cells: {len(hdbscan_labels):,}  Clusters: {n_clusters_hdbscan}  Noise: {(hdbscan_labels == -1).mean():.1%}")

        cluster_by_treatment = pd.crosstab(hdbscan_labels, umap_df_clustered[RESOLVED_CONFIG.treatment_col], normalize="columns")
        print(cluster_by_treatment.round(2).to_string())
    else:
        umap_df_clustered = umap_df.copy()
        print("  ⚠️  HDBSCAN skipped (hdbscan not installed).")
    return (umap_df_clustered,)


@app.cell
def _(FIGS_DIR, HDBSCAN_OK, plt, sns, umap_df_clustered):
    if HDBSCAN_OK:
        fig_hdb, ax_hdb = plt.subplots(figsize=(8, 7))
        sns.scatterplot(data=umap_df_clustered, x="UMAP1", y="UMAP2", hue="Cluster", palette="tab20", s=5, linewidth=0, ax=ax_hdb)
        ax_hdb.set_title("Single-cell states (HDBSCAN)")
        fig_hdb.tight_layout()
        fig_hdb.savefig(FIGS_DIR / "sc_hdbscan_umap.png", dpi=150, bbox_inches="tight")
        plt.close(fig_hdb)
        print("✓ Saved: sc_hdbscan_umap.png")
        _display = fig_hdb
    else:
        _display = None
    _display
    return


@app.cell
def _(
    FIGS_DIR,
    KMeans,
    X_cluster,
    kmeans_k_max_input,
    np,
    plt,
    silhouette_score,
):
    k_range = range(2, int(kmeans_k_max_input.value) + 1)
    inertias_sc, silhouettes_sc = [], []
    for _k in k_range:
        _km = KMeans(n_clusters=_k, random_state=42, n_init=10)
        _labels_k = _km.fit_predict(X_cluster)
        inertias_sc.append(_km.inertia_)
        silhouettes_sc.append(
            silhouette_score(X_cluster, _labels_k, sample_size=min(5000, len(_labels_k)), random_state=42)
        )

    fig_elbow, axes_elbow = plt.subplots(1, 2, figsize=(12, 4))
    axes_elbow[0].plot(list(k_range), inertias_sc, "o-", color="#1976d2")
    axes_elbow[0].set_xlabel("k")
    axes_elbow[0].set_ylabel("Inertia")
    axes_elbow[0].set_title("Elbow plot")
    axes_elbow[1].plot(list(k_range), silhouettes_sc, "o-", color="#d32f2f")
    axes_elbow[1].set_xlabel("k")
    axes_elbow[1].set_ylabel("Silhouette score")
    axes_elbow[1].set_title("Silhouette")
    fig_elbow.suptitle("Single-cell KMeans — choosing k", fontsize=12, fontweight="bold")
    fig_elbow.tight_layout()
    fig_elbow.savefig(FIGS_DIR / "sc_kmeans_elbow.png", dpi=150, bbox_inches="tight")
    plt.close(fig_elbow)
    best_k_sc = list(k_range)[int(np.argmax(silhouettes_sc))]
    print(f"  Best silhouette at k={best_k_sc}: {max(silhouettes_sc):.3f}")
    fig_elbow
    return (best_k_sc,)


@app.cell
def _(
    FIGS_DIR,
    KMeans,
    RESOLVED_CONFIG,
    UMAP_OK,
    X_cluster,
    best_k_sc,
    df_sampled,
    pd,
    plt,
    sns,
    umap,
):
    K_SC = best_k_sc
    kmeans_sc = KMeans(n_clusters=K_SC, random_state=42, n_init=10)
    sc_cluster_labels = kmeans_sc.fit_predict(X_cluster)

    df_clustered = df_sampled.copy()
    df_clustered["cluster"] = sc_cluster_labels
    composition = pd.crosstab(sc_cluster_labels, df_clustered[RESOLVED_CONFIG.treatment_col]).apply(lambda r: r / r.sum(), axis=1).round(2)
    print("Cluster composition (fraction per treatment):")
    print(composition.to_string())

    if UMAP_OK:
        reducer_sc = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
        X_sc_umap_clusters = reducer_sc.fit_transform(X_cluster)

        fig_kmeans_umap, axes_kmeans_umap = plt.subplots(1, 2, figsize=(14, 5))
        _palette = sns.color_palette("tab20", K_SC)
        for _k in range(K_SC):
            _mask = sc_cluster_labels == _k
            axes_kmeans_umap[0].scatter(X_sc_umap_clusters[_mask, 0], X_sc_umap_clusters[_mask, 1], label=f"Cluster {_k}", color=_palette[_k], s=8, alpha=0.5, linewidths=0)
        axes_kmeans_umap[0].set_title(f"KMeans clusters (k={K_SC})")
        axes_kmeans_umap[0].legend(fontsize=7, markerscale=2)

        _treatments_list = df_clustered[RESOLVED_CONFIG.treatment_col].tolist()
        _unique_t = sorted(set(_treatments_list))
        _palette_t = sns.color_palette("tab10", len(_unique_t))
        for _lbl, _col in zip(_unique_t, _palette_t):
            _mask = [t == _lbl for t in _treatments_list]
            axes_kmeans_umap[1].scatter(X_sc_umap_clusters[_mask, 0], X_sc_umap_clusters[_mask, 1], label=_lbl, color=_col, s=8, alpha=0.5, linewidths=0)
        axes_kmeans_umap[1].set_title("Treatment (true labels)")
        axes_kmeans_umap[1].legend(fontsize=7, markerscale=2)

        for _ax in axes_kmeans_umap:
            _ax.set_xlabel("UMAP 1")
            _ax.set_ylabel("UMAP 2")
        fig_kmeans_umap.suptitle("Single-cell UMAP", fontsize=12, fontweight="bold")
        fig_kmeans_umap.tight_layout()
        fig_kmeans_umap.savefig(FIGS_DIR / "sc_umap_clusters.png", dpi=150, bbox_inches="tight")
        plt.close(fig_kmeans_umap)
        print("  ✓  Saved: sc_umap_clusters.png")
        _display = fig_kmeans_umap
    else:
        _display = None
    _display
    return K_SC, df_clustered


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — LightGBM classifier + SHAP

    **Label:** treatment
    **CV strategy:** `GroupShuffleSplit` with well as group (no well-level leakage)
    **SHAP:** `TreeExplainer` on the trained model
    """)
    return


@app.cell
def _(
    CLF_REPORT_CSV,
    RESOLVED_CONFIG,
    SHAP_SUMMARY_CSV,
    X_sampled,
    df_clustered,
    feature_cols_model,
    train_lgbm_classifier_with_shap,
):
    lgbm_result = train_lgbm_classifier_with_shap(
        X_sampled,
        df_clustered[RESOLVED_CONFIG.treatment_col].to_numpy(),
        df_clustered[RESOLVED_CONFIG.well_col].to_numpy(),
        feature_names=feature_cols_model,
    )

    if "message" in lgbm_result:
        print(f"  ⚠️  {lgbm_result['message']}")
    else:
        lgbm_result["classification_report"].to_csv(CLF_REPORT_CSV)
        print(f"✓ Classification report saved: {CLF_REPORT_CSV.name}")
        print(f"  Train wells: {len(lgbm_result['train_wells'])}  Test wells: {len(lgbm_result['test_wells'])}")
        print(f"  Well overlap between train/test: {len(lgbm_result['train_wells'] & lgbm_result['test_wells'])}")

        if "mean_abs_shap" in lgbm_result:
            lgbm_result["mean_abs_shap"].to_csv(SHAP_SUMMARY_CSV)
            print(f"✓ SHAP summary saved: {SHAP_SUMMARY_CSV.name}")
        elif "shap_message" in lgbm_result:
            print(f"  ⚠️  {lgbm_result['shap_message']}")
    return (lgbm_result,)


@app.cell
def _(FIGS_DIR, lgbm_result, plt):
    if "mean_abs_shap" in lgbm_result:
        _top30 = lgbm_result["mean_abs_shap"].head(30)
        fig_shap, ax_shap = plt.subplots(figsize=(8, 10))
        ax_shap.barh(_top30.index[::-1], _top30.to_numpy()[::-1], color="#1976d2", alpha=0.85)
        ax_shap.set_xlabel("Mean |SHAP value|")
        ax_shap.set_title("SHAP feature importance (top 30, mean across classes)")
        ax_shap.tick_params(axis="y", labelsize=8)
        fig_shap.tight_layout()
        fig_shap.savefig(FIGS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig_shap)
        print("  ✓  Saved: shap_importance.png")
        _display = fig_shap
    else:
        _display = None
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — SC-17: SHAP vs. LDA loadings cross-check

    If the top SHAP features and top LDA loadings agree, this is
    convergent evidence that the morphological signal is real and not an
    artefact of the supervised method.
    """)
    return


@app.cell
def _(FIGS_DIR, LDA_LOADINGS_CSV, lgbm_result, pd, plt):
    print("── SC-17: SHAP vs. LDA loadings cross-check ──")
    if "mean_abs_shap" not in lgbm_result or not LDA_LOADINGS_CSV.exists():
        print("  ⚠️  SHAP summary or LDA loadings not available. Run NB04 and Section 7 above first.")
        _display = None
    else:
        lda_df_sc = pd.read_csv(LDA_LOADINGS_CSV, index_col=0)
        top_shap = set(lgbm_result["mean_abs_shap"].nlargest(20).index)
        lda_axes = [c for c in lda_df_sc.columns if c.startswith("LDA")]

        overlaps = []
        for axis in lda_axes:
            top_lda = set(lda_df_sc[axis].abs().nlargest(20).index)
            overlap = top_shap & top_lda
            overlaps.append(len(overlap))
            print(f"  {axis}: {len(overlap)}/20 features overlap with top-20 SHAP")

        fig_sc17, ax_sc17 = plt.subplots(figsize=(8, 4))
        ax_sc17.bar(lda_axes, overlaps, color="#1976d2", alpha=0.85)
        ax_sc17.axhline(10, color="green", linestyle="--", linewidth=1.2, label="Strong convergence (10)")
        ax_sc17.axhline(5, color="orange", linestyle="--", linewidth=1.2, label="Moderate convergence (5)")
        ax_sc17.set_xlabel("LDA axis")
        ax_sc17.set_ylabel("Overlap with top-20 SHAP features")
        ax_sc17.set_title("SC-17: SHAP vs. LDA loadings overlap (top 20 features)")
        ax_sc17.legend(fontsize=9)
        fig_sc17.tight_layout()
        fig_sc17.savefig(FIGS_DIR / "sc17_shap_lda_overlap.png", dpi=150, bbox_inches="tight")
        plt.close(fig_sc17)
        print("  ✓  Saved: sc17_shap_lda_overlap.png")
        _display = fig_sc17
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 — Final summary and provenance
    """)
    return


@app.cell
def _(
    CONFIG,
    EXPERIMENT_ID,
    K_SC,
    REPO_ROOT,
    RESULTS_DIR,
    df_clustered,
    df_sc,
    df_sc_clean,
    feature_cols_model,
    json,
    platform,
    save_history_input,
    subprocess,
):
    from datetime import datetime, timezone

    def _run_git(args, root):
        try:
            result = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None if result.returncode == 0 else None

    print("═" * 65)
    print("NB06 complete")
    print("═" * 65)
    print(f"  Cells loaded          : {df_sc.shape[0]:,}")
    print(f"  After outlier removal : {df_sc_clean.shape[0]:,}")
    print(f"  Sampled for analysis  : {df_clustered.shape[0]:,}")
    print(f"  Features used         : {len(feature_cols_model)}")
    print(f"  KMeans k              : {K_SC}")

    provenance_nb06 = {
        "schema_version": 1,
        "pipeline": {
            "notebook": "06_single_cell_analysis.py",
            "experiment_id": EXPERIMENT_ID,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "configuration": {"channels": CONFIG.channels},
        "dataset": {
            "n_cells_loaded": int(df_sc.shape[0]),
            "n_cells_after_outlier_removal": int(df_sc_clean.shape[0]),
            "n_cells_sampled": int(df_clustered.shape[0]),
            "n_features": int(len(feature_cols_model)),
            "kmeans_k": int(K_SC),
        },
        "version_control": {"git_commit": _run_git(["rev-parse", "HEAD"], REPO_ROOT) or "unknown"},
        "environment": {"python_version": platform.python_version()},
    }
    provenance_nb06_path = RESULTS_DIR / "provenance_nb06.json"
    with provenance_nb06_path.open("w", encoding="utf-8") as _f:
        json.dump(provenance_nb06, _f, indent=2, ensure_ascii=False)
    print(f"✓ Provenance saved: {provenance_nb06_path}")

    if save_history_input.value:
        _timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _history_path = RESULTS_DIR / f"provenance_nb06_{_timestamp}.json"
        if _history_path.exists():
            raise FileExistsError(f"Historical provenance file already exists: {_history_path}")
        with _history_path.open("w", encoding="utf-8") as _f:
            json.dump(provenance_nb06, _f, indent=2, ensure_ascii=False)
        print(f"✓ Historical record: {_history_path}")
    else:
        print("  Historical record: disabled")
    return


@app.cell
def _():
    import json
    import platform
    import subprocess

    return json, platform, subprocess


@app.cell
def _(mo):
    mo.md(r"""
    ## Export a PDF report

    Optional. Renders this notebook — markdown, code, and outputs — into a
    paginated PDF and saves it under `reports/` for this experiment,
    alongside `results/` and `figures/`. Rendering re-runs the notebook
    headlessly in a fresh process, so the report reflects whatever is
    currently saved in `experiment_config.json` (written by the
    configuration cell above each time this notebook runs), not any
    unsaved changes to the widgets above.
    """)
    return


@app.cell
def _(mo):
    export_report_button = mo.ui.run_button(
        label="Export this notebook as a PDF report", kind="success"
    )
    export_report_button
    return (export_report_button,)


@app.cell
def _(EXPERIMENT_ID, Path, REPO_ROOT, export_report_button, mo):
    mo.stop(not export_report_button.value)

    from hca_pipeline.report_export import export_notebook_pdf

    _notebook_file = Path(__file__).resolve()
    _reports_dir = REPO_ROOT / "workspace" / "analysis" / EXPERIMENT_ID / "reports"
    _reports_dir.mkdir(parents=True, exist_ok=True)
    _report_path = _reports_dir / f"{_notebook_file.stem}.pdf"

    with mo.status.spinner(title="Rendering PDF report (re-runs this notebook headlessly)"):
        export_notebook_pdf(
            _notebook_file,
            _report_path,
            title=f"{EXPERIMENT_ID} — {_notebook_file.stem}",
        )

    mo.md(f"✓ Report saved: `{_report_path}`")
    return


if __name__ == "__main__":
    app.run()
