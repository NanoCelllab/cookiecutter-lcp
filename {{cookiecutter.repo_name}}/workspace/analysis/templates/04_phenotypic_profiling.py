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
    # 04 — Phenotypic Profiling
    ## PCA · UMAP · Profile QC · Optional Batch Correction · LDA · Feature Importance · Clustering

    **Pipeline step:** 4 of 6
    **Position in pipeline:** NB03 (quality metrics, Go/No-Go) → **NB04 (phenotypic profiling)** → NB05 (phenotypic fingerprints)

    This notebook is meant to be run interactively (`marimo edit`), not
    headlessly: NB03's Go/No-Go gate has already checked replicate signal,
    but interpreting PCA/UMAP structure, choosing the final modelling space,
    and judging cluster quality below are human-in-the-loop decisions.

    **Input:** `per_well_features_selected.parquet` (from NB02), optional `cv_summary.csv` (from NB02)
    **Output:** profile-QC summaries, PCA/UMAP figures, optional Harmony-corrected coordinates,
    LDA models, loadings, CV results, confusion matrices, KMeans models, phenotypic fingerprints,
    and an evidence-based uncorrected-vs-Harmony comparison.

    ### Analysis sections
    | Section | Method | Purpose |
    |---------|--------|---------|
    | 1 | LDA bias check | Assess within-group variability, homoscedasticity, and sample size |
    | 2 | PCA | Unsupervised baseline and global variance structure |
    | 3 | UMAP | Non-linear visual exploration before correction |
    | 4 | Phenotypic profile QC | Visual and statistical assessment of plate effects |
    | 5 | Build latent spaces | Optional Harmony batch correction, in parallel with the uncorrected space |
    | 6 | Parallel modelling | UMAP + treatment/dose LDA + leave-one-plate-out CV + KMeans, run once per space |
    | 7 | Phenotypic fingerprints | Cohen's *d* by Treatment × Concentration, heatmaps and radar overview |
    | 8 | Comparison & recommendation | Evidence-based uncorrected-vs-Harmony assessment |
    | 9 | Provenance | Per-space and comparison provenance records |
    | 10 | Integrity checks | Confirm expected outputs exist and are internally consistent |

    ### Sanity checks
    | ID | Check |
    |----|-------|
    | SC-11 | Visual and statistical plate-effect assessment |
    | SC-12 | In-sample versus cross-validated LDA accuracy |
    | SC-13 | KMeans cluster purity |

    > Batch correction is not applied automatically. This notebook first measures technical
    > structure, reports the evidence, and then follows the explicit `FINAL_MODELLING_SPACE` choice.

    This is a **marimo** port of `03_phenotypic_profiling_parallel_spaces.ipynb`. The core
    modelling routine (`run_modelling_space`) now lives in `hca_pipeline.modelling`, already
    smoke-tested; this notebook is mostly orchestration — config, data loading, calling the
    shared routine once per latent space, plotting, and protected output writes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 0 — Imports and plot style
    """)
    return


@app.cell
def _():
    import datetime
    import json
    import subprocess
    import warnings
    from dataclasses import replace
    from pathlib import Path

    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    try:
        from umap import UMAP

        _UMAP_OK = True
    except ImportError:
        warnings.warn("umap-learn not installed — UMAP sections will be skipped.")
        UMAP = None
        _UMAP_OK = False

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )

    UMAP_OK = _UMAP_OK
    return (
        PCA,
        Path,
        UMAP,
        UMAP_OK,
        datetime,
        joblib,
        json,
        np,
        pd,
        plt,
        replace,
        subprocess,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0b — Shared pipeline utilities (bootstrap)
    """)
    return


@app.cell
def _(Path):
    # Locate the repo root before hca_pipeline can be imported (bootstrap:
    # can't import find_repo_root from the package until sys.path includes
    # workspace). __file__ is used instead of cwd so this notebook
    # behaves identically under `marimo edit`, a plain `python` run, or an
    # automated headless run from any working directory.
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
    from hca_pipeline.io import write_csv_protected
    from hca_pipeline.modelling import (
        _qc_record as qc_record,  # leading-underscore rename: see NB conversion notes
        absolute_change,
        get_qc_metric,
        run_modelling_space,
        safe_relative_change,
        validate_output_path,
    )
    from hca_pipeline.plotting import plot_fingerprint_heatmap, scatter_panel
    from hca_pipeline.stats import build_fingerprint_matrix, calculate_within_plate_effects, cohens_d

    print(f"  ✓  Shared utilities loaded from hca_pipeline ({_pipelines_dir})")
    return (
        ExperimentConfig,
        REPO_ROOT,
        absolute_change,
        build_fingerprint_matrix,
        calculate_within_plate_effects,
        get_qc_metric,
        infer_feature_cols,
        plot_fingerprint_heatmap,
        qc_record,
        run_modelling_space,
        safe_relative_change,
        scatter_panel,
        validate_output_path,
        write_csv_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Configuration

    Pick the experiment and the algorithmic choices for this run. Column names and control
    vocabulary come from the saved `ExperimentConfig` (shared with every other notebook in the
    pipeline); the choices below (which latent spaces to run, thresholds, random seed) are
    algorithmic decisions local to this notebook and are not persisted to `experiment_config.json`.
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
def _(loaded_config, mo):
    run_analysis_spaces_input = mo.ui.multiselect(
        options=["uncorrected", "harmony"],
        value=["uncorrected", "harmony"],
        label="Latent modelling spaces to run",
    )
    final_modelling_space_input = mo.ui.dropdown(
        options=["uncorrected", "harmony"],
        value="uncorrected",
        label="Final modelling space (explicit downstream choice)",
    )
    overwrite_input = mo.ui.checkbox(
        value=loaded_config.overwrite_existing_outputs,
        label="Overwrite existing outputs",
    )
    save_history_input = mo.ui.checkbox(
        value=loaded_config.save_provenance_history,
        label="Save timestamped provenance history",
    )
    mo.vstack([run_analysis_spaces_input, final_modelling_space_input, overwrite_input, save_history_input])
    return (
        final_modelling_space_input,
        overwrite_input,
        run_analysis_spaces_input,
        save_history_input,
    )


@app.cell
def _(mo):
    random_state_input = mo.ui.number(value=42, start=0, stop=10_000, label="Random seed")
    n_pca_components_input = mo.ui.number(value=50, start=2, stop=500, label="PCA components")
    n_qc_permutations_input = mo.ui.number(
        value=999, start=99, stop=9_999, label="PERMANOVA/PERMDISP permutations"
    )
    cv_warn_threshold_input = mo.ui.number(
        value=1.5, start=0.0, stop=10.0, step=0.1, label="Within-group median-CV warning threshold"
    )
    homoscedasticity_ratio_input = mo.ui.number(
        value=3.0, start=1.0, stop=20.0, step=0.5, label="Homoscedasticity SD-ratio threshold"
    )
    mo.vstack(
        [
            random_state_input,
            n_pca_components_input,
            n_qc_permutations_input,
            cv_warn_threshold_input,
            homoscedasticity_ratio_input,
        ]
    )
    return (
        cv_warn_threshold_input,
        homoscedasticity_ratio_input,
        n_pca_components_input,
        n_qc_permutations_input,
        random_state_input,
    )


@app.cell
def _(mo):
    fingerprint_effect_threshold_input = mo.ui.number(
        value=0.50, start=0.0, stop=5.0, step=0.05, label="Fingerprint 'altered feature' |Cohen's d| threshold"
    )
    fingerprint_within_plate_input = mo.ui.checkbox(
        value=True, label="Combine Cohen's d within-plate (vs. global fallback)"
    )
    mo.vstack([fingerprint_effect_threshold_input, fingerprint_within_plate_input])
    return fingerprint_effect_threshold_input, fingerprint_within_plate_input


@app.cell
def _(mo):
    batch_r2_warn_input = mo.ui.number(
        value=0.05, start=0.0, stop=1.0, step=0.01, label="Negcon plate PERMANOVA R² warning threshold"
    )
    min_plate_r2_reduction_input = mo.ui.number(
        value=0.30, start=0.0, stop=1.0, step=0.01, label="Minimum relative plate-R² reduction for 'harmony'"
    )
    max_treatment_r2_loss_input = mo.ui.number(
        value=0.10, start=0.0, stop=1.0, step=0.01, label="Max relative treatment-R² loss tolerated"
    )
    max_treatment_silhouette_loss_input = mo.ui.number(
        value=0.05, start=0.0, stop=1.0, step=0.01, label="Max absolute treatment-silhouette loss tolerated"
    )
    max_replicability_loss_input = mo.ui.number(
        value=0.05, start=0.0, stop=1.0, step=0.01, label="Max relative dose-CV balanced-accuracy loss tolerated"
    )
    mo.vstack(
        [
            batch_r2_warn_input,
            min_plate_r2_reduction_input,
            max_treatment_r2_loss_input,
            max_treatment_silhouette_loss_input,
            max_replicability_loss_input,
        ]
    )
    return (
        batch_r2_warn_input,
        max_replicability_loss_input,
        max_treatment_r2_loss_input,
        max_treatment_silhouette_loss_input,
        min_plate_r2_reduction_input,
    )


@app.cell
def _(
    batch_r2_warn_input,
    cv_warn_threshold_input,
    experiment_id_input,
    final_modelling_space_input,
    fingerprint_effect_threshold_input,
    fingerprint_within_plate_input,
    homoscedasticity_ratio_input,
    max_replicability_loss_input,
    max_treatment_r2_loss_input,
    max_treatment_silhouette_loss_input,
    min_plate_r2_reduction_input,
    n_pca_components_input,
    n_qc_permutations_input,
    overwrite_input,
    random_state_input,
    run_analysis_spaces_input,
    save_history_input,
):
    EXPERIMENT_ID = experiment_id_input.value
    RUN_ANALYSIS_SPACES = list(run_analysis_spaces_input.value)
    FINAL_MODELLING_SPACE = final_modelling_space_input.value
    OVERWRITE_EXISTING_OUTPUTS = bool(overwrite_input.value)
    SAVE_PROVENANCE_HISTORY = bool(save_history_input.value)

    RANDOM_STATE = int(random_state_input.value)
    N_PCA_COMPONENTS = int(n_pca_components_input.value)
    N_QC_PERMUTATIONS = int(n_qc_permutations_input.value)
    CV_WARN_THRESHOLD = float(cv_warn_threshold_input.value)
    HOMOSCEDASTICITY_RATIO = float(homoscedasticity_ratio_input.value)

    FINGERPRINT_EFFECT_THRESHOLD = float(fingerprint_effect_threshold_input.value)
    FINGERPRINT_USE_WITHIN_PLATE_EFFECTS = bool(fingerprint_within_plate_input.value)

    BATCH_R2_WARN_THRESHOLD = float(batch_r2_warn_input.value)
    MIN_PLATE_R2_RELATIVE_REDUCTION = float(min_plate_r2_reduction_input.value)
    MAX_TREATMENT_R2_RELATIVE_LOSS = float(max_treatment_r2_loss_input.value)
    MAX_TREATMENT_SILHOUETTE_ABSOLUTE_LOSS = float(max_treatment_silhouette_loss_input.value)
    MAX_REPLICABILITY_RELATIVE_LOSS = float(max_replicability_loss_input.value)

    if not RUN_ANALYSIS_SPACES:
        raise ValueError("Select at least one latent modelling space to run.")
    if FINAL_MODELLING_SPACE not in RUN_ANALYSIS_SPACES:
        raise ValueError(
            f"FINAL_MODELLING_SPACE={FINAL_MODELLING_SPACE!r} was not selected to run. "
            f"Selected spaces: {RUN_ANALYSIS_SPACES}"
        )

    print("═" * 72)
    print("NB04 CONFIGURATION")
    print("═" * 72)
    print(f"  Experiment ID          : {EXPERIMENT_ID}")
    print(f"  Modelling spaces       : {RUN_ANALYSIS_SPACES}")
    print(f"  Final modelling space  : {FINAL_MODELLING_SPACE}")
    print(f"  Overwrite outputs      : {OVERWRITE_EXISTING_OUTPUTS}")
    print(f"  Save provenance history: {SAVE_PROVENANCE_HISTORY}")
    print(f"  Random seed            : {RANDOM_STATE}")
    return (
        BATCH_R2_WARN_THRESHOLD,
        CV_WARN_THRESHOLD,
        EXPERIMENT_ID,
        FINAL_MODELLING_SPACE,
        FINGERPRINT_EFFECT_THRESHOLD,
        FINGERPRINT_USE_WITHIN_PLATE_EFFECTS,
        HOMOSCEDASTICITY_RATIO,
        MAX_REPLICABILITY_RELATIVE_LOSS,
        MAX_TREATMENT_R2_RELATIVE_LOSS,
        MAX_TREATMENT_SILHOUETTE_ABSOLUTE_LOSS,
        MIN_PLATE_R2_RELATIVE_REDUCTION,
        N_PCA_COMPONENTS,
        N_QC_PERMUTATIONS,
        OVERWRITE_EXISTING_OUTPUTS,
        RANDOM_STATE,
        RUN_ANALYSIS_SPACES,
        SAVE_PROVENANCE_HISTORY,
    )


@app.cell
def _(
    EXPERIMENT_ID,
    OVERWRITE_EXISTING_OUTPUTS,
    SAVE_PROVENANCE_HISTORY,
    loaded_config,
    replace,
):
    # Column names/vocabulary are resolved against the real data once it is
    # loaded (see "resolve CONFIG" cell below); this draft only carries the
    # experiment-identity and output-protection choices made above.
    WIDGET_CONFIG = replace(
        loaded_config,
        experiment_id=EXPERIMENT_ID,
        overwrite_existing_outputs=OVERWRITE_EXISTING_OUTPUTS,
        save_provenance_history=SAVE_PROVENANCE_HISTORY,
    )
    return (WIDGET_CONFIG,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Paths
    """)
    return


@app.cell
def _(EXPERIMENT_ID, REPO_ROOT):
    WORKSPACE_DIR = REPO_ROOT / "workspace"
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    PROFILES_DIR = WORKSPACE_DIR / "profiles" / EXPERIMENT_ID
    MODELS_DIR = WORKSPACE_DIR / "models" / EXPERIMENT_ID

    PROFILES_OUT_DIR = PROFILES_DIR / "outputs"
    FIGS_DIR = ANALYSIS_DIR / "figures" / "phenotypic_profiling"
    RESULTS_DIR = ANALYSIS_DIR / "results"
    MODELS_OUT_DIR = MODELS_DIR / "outputs"

    COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparisons"
    COMPARISON_FIGURES_DIR = FIGS_DIR / "comparisons"
    FINGERPRINT_RESULTS_DIR = RESULTS_DIR / "fingerprints"
    FINGERPRINT_FIGURES_DIR = FIGS_DIR / "fingerprints"

    for _directory in (
        FIGS_DIR,
        RESULTS_DIR,
        MODELS_OUT_DIR,
        COMPARISON_RESULTS_DIR,
        COMPARISON_FIGURES_DIR,
        FINGERPRINT_RESULTS_DIR,
        FINGERPRINT_FIGURES_DIR,
    ):
        _directory.mkdir(parents=True, exist_ok=True)

    INPUT_PARQUET = PROFILES_OUT_DIR / "per_well_features_selected.parquet"
    CV_SUMMARY_CSV = RESULTS_DIR / "cv_summary.csv"

    assert INPUT_PARQUET.exists(), f"Input not found: {INPUT_PARQUET}"
    print(f"EXPERIMENT_ID : {EXPERIMENT_ID}")
    print(f"INPUT         : {INPUT_PARQUET}")
    print(f"RESULTS_DIR   : {RESULTS_DIR}")
    print(f"FIGS_DIR      : {FIGS_DIR}")
    print(f"MODELS_OUT_DIR: {MODELS_OUT_DIR}")
    return (
        COMPARISON_FIGURES_DIR,
        COMPARISON_RESULTS_DIR,
        CV_SUMMARY_CSV,
        FIGS_DIR,
        FINGERPRINT_FIGURES_DIR,
        FINGERPRINT_RESULTS_DIR,
        INPUT_PARQUET,
        MODELS_OUT_DIR,
        RESULTS_DIR,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Load data
    """)
    return


@app.cell
def _(INPUT_PARQUET, pd):
    df_loaded = pd.read_parquet(INPUT_PARQUET)
    df_loaded = df_loaded.loc[:, ~df_loaded.columns.str.contains("ImageNumber")]
    return (df_loaded,)


@app.cell
def _(REPO_ROOT, WIDGET_CONFIG, df_loaded):
    # Auto-detect the actual metadata column names/vocabulary present in this
    # experiment's data (rather than the source notebook's hardcoded
    # NEGCON_LABEL="Non-treated" / TREATMENT_COL="Metadata_Treatment" style
    # literals), then persist the resolved config for downstream notebooks.
    CONFIG = WIDGET_CONFIG.resolve_columns(df_loaded)
    _config_path = CONFIG.save(REPO_ROOT)

    print(f"  Resolved columns: plate={CONFIG.plate_col}, treatment={CONFIG.treatment_col}, "
          f"concentration={CONFIG.concentration_col}, control_type={CONFIG.control_type_col}")
    print(f"  Has dose axis   : {CONFIG.has_dose_axis}")
    print(f"  Negcon values   : {CONFIG.negcon_values}")
    print(f"  Saved config    : {_config_path}")
    return (CONFIG,)


@app.cell
def _(CONFIG, df_loaded, infer_feature_cols):
    meta_cols = [c for c in df_loaded.columns if c.startswith("Metadata_")]
    feat_cols = infer_feature_cols(df_loaded)

    print(f"Shape          : {df_loaded.shape}")
    print(f"Meta columns   : {len(meta_cols)}")
    print(f"Feature columns: {len(feat_cols)}")
    print("\nTreatment distribution:")
    print(df_loaded[CONFIG.treatment_col].value_counts().to_string())
    print(f"\nPlates: {sorted(df_loaded[CONFIG.plate_col].unique())}")
    return feat_cols, meta_cols


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Prepare feature matrix
    """)
    return


@app.cell
def _(CONFIG, df_loaded, feat_cols, np):
    from sklearn.preprocessing import StandardScaler

    df = df_loaded.dropna(subset=feat_cols).reset_index(drop=True)

    # "_dose_label" is the Treatment x Concentration composite class used
    # throughout this notebook (sample-size checks, dose-level LDA, KMeans).
    # Kept as a private-looking column *name* inside the DataFrame — this is
    # just a string key, not a marimo cross-cell variable, so the leading
    # underscore is harmless here.
    if CONFIG.has_dose_axis and CONFIG.concentration_col:
        df["_dose_label"] = (
            df[CONFIG.treatment_col].astype(str)
            + " | "
            + df[CONFIG.concentration_col].map(lambda v: f"{float(v):g}")
        )
    else:
        df["_dose_label"] = df[CONFIG.treatment_col].astype(str)

    X_raw = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"Feature matrix: {X.shape[0]} wells × {X.shape[1]} features")
    print(f"NaN after cleaning: {np.isnan(X).sum()}")
    return X, df


@app.cell
def _(CONFIG, df):
    # Data-driven negative-control mask, replacing the source notebook's
    # hardcoded NEGCON_LABEL="Non-treated" literal with the experiment's own
    # control-type vocabulary (Section: unify control vocabulary onto
    # ExperimentConfig).
    if CONFIG.control_type_col and CONFIG.control_type_col in df.columns:
        _negcon_values_lower = {v.lower() for v in CONFIG.negcon_values}
        negcon_mask = df[CONFIG.control_type_col].astype(str).str.lower().isin(_negcon_values_lower)
    else:
        raise ValueError(
            "Cannot identify negative-control wells: no control-type column "
            f"({CONFIG.control_type_col!r}) found for experiment {CONFIG.experiment_id!r}. "
            "Configure control_type_col in ExperimentConfig."
        )

    n_negcon = int(negcon_mask.sum())
    if n_negcon < 2:
        raise ValueError("At least two negative-control wells are required for downstream QC.")
    print(f"Negative-control wells: {n_negcon} / {len(df)}")
    return (negcon_mask,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 1 — LDA Bias Check

    Before fitting LDA, two potential sources of bias are assessed:

    1. **Within-group CV** (optional, from NB02 `cv_summary.csv`): high CV in the negative-control
       group indicates residual batch effects.
    2. **Homoscedasticity**: LDA assumes equal within-class covariance matrices. Groups whose
       per-feature SD is far greater than the negative-control SD on a large fraction of features
       are flagged.
    """)
    return


@app.cell
def _(CONFIG, CV_SUMMARY_CSV, CV_WARN_THRESHOLD, df, negcon_mask, pd):
    print("═" * 72)
    print("BIAS AND MODEL-ASSUMPTION CHECK — SECTION 1")
    print("═" * 72)

    print("\n── 1a. Within-group CV (optional QC imported from NB02) ───────────")

    _negcon_treatments = set(df.loc[negcon_mask, CONFIG.treatment_col].astype(str))

    if CV_SUMMARY_CSV.exists():
        cv_summary = pd.read_csv(CV_SUMMARY_CSV)
        _required_cv_columns = {"Metadata_Treatment", "median_CV"}
        _missing_cv_columns = _required_cv_columns - set(cv_summary.columns)

        if _missing_cv_columns:
            print("  ⚠️ CV summary was found, but its schema is not compatible with this notebook.")
            print(f"  Missing columns: {sorted(_missing_cv_columns)}")
        else:
            cv_summary = cv_summary.sort_values("median_CV", ascending=False)
            print(cv_summary.to_string(index=False))

            _flagged_cv = cv_summary.loc[cv_summary["median_CV"] > CV_WARN_THRESHOLD]
            if not _flagged_cv.empty:
                print(f"\n  ⚠️ Groups with median CV > {CV_WARN_THRESHOLD}:")
                for _, _row in _flagged_cv.iterrows():
                    print(f"    • {_row['Metadata_Treatment']}: median CV = {_row['median_CV']:.2f}")
                if _negcon_treatments & set(_flagged_cv["Metadata_Treatment"]):
                    print("\n  ⚠️ At least one negative-control treatment has elevated variability.")
                    print("     This may indicate residual technical variation or inconsistent control wells.")
                else:
                    print("\n  Interpretation: elevated CV is restricted to specific biological groups.")
            else:
                print(f"\n  ✓ All groups have median CV ≤ {CV_WARN_THRESHOLD}.")
    else:
        print("  ℹ️ Optional CV summary not available at the expected path.")
        print(f"     Expected: {CV_SUMMARY_CSV}")
        print("     The remaining assumption checks will continue normally.")
    return


@app.cell
def _(CONFIG, HOMOSCEDASTICITY_RATIO, df, feat_cols, negcon_mask, np, pd):
    print("\n── 1b. Within-group dispersion relative to negative controls ────")

    _negcon_treatments = set(df.loc[negcon_mask, CONFIG.treatment_col].astype(str))
    negcon_sd = df.loc[negcon_mask, feat_cols].std(ddof=1)

    _hom_results = []
    for _group, _subset in df.groupby(CONFIG.treatment_col, sort=False):
        if str(_group) in _negcon_treatments:
            continue
        _group_sd = _subset[feat_cols].std(ddof=1)
        _sd_ratio = (_group_sd / negcon_sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        _valid_ratios = _sd_ratio.dropna()
        _pct_high = (
            _valid_ratios.gt(HOMOSCEDASTICITY_RATIO).mean() * 100 if not _valid_ratios.empty else np.nan
        )
        _hom_results.append(
            {
                "Treatment": _group,
                "n_wells": len(_subset),
                "pct_features_high_SD": _pct_high,
                "median_SD_ratio": _valid_ratios.median(),
            }
        )

    hom_df = pd.DataFrame(_hom_results).sort_values("pct_features_high_SD", ascending=False)
    print(
        hom_df.to_string(
            index=False,
            formatters={
                "pct_features_high_SD": lambda v: f"{v:.1f}" if pd.notna(v) else "NA",
                "median_SD_ratio": lambda v: f"{v:.2f}" if pd.notna(v) else "NA",
            },
        )
    )

    _flagged_hom = hom_df.loc[hom_df["pct_features_high_SD"] > 10]
    if _flagged_hom.empty:
        print("\n  ✓ No group shows a broad increase in dispersion relative to the negative controls.")
    else:
        print(
            "\n  Groups with >10% of features showing an SD greater than "
            f"{HOMOSCEDASTICITY_RATIO:g}× the negative-control SD:"
        )
        for _, _row in _flagged_hom.iterrows():
            print(
                f"    • {_row['Treatment']}: {_row['pct_features_high_SD']:.1f}% of features; "
                f"median SD ratio = {_row['median_SD_ratio']:.2f}"
            )
        print(
            "      Elevated dispersion is not, by itself, a reason to exclude a biological "
            "group — it may reflect real biological heterogeneity (e.g. an alternative "
            "reference state) rather than a technical batch effect."
        )
        print("\n  Modelling implication:")
        print(
            "    Prefer regularized LDA (solver='lsqr', shrinkage='auto') rather than an "
            "unregularized shared covariance matrix when dispersion is flagged."
        )
    return (hom_df,)


@app.cell
def _(CONFIG, df, pd):
    print("\n── 1c. Sample size per modelling class ──────────────────")

    treatment_counts = df[CONFIG.treatment_col].value_counts()
    dose_counts = df["_dose_label"].value_counts()

    print(f"  Treatment-level model ({treatment_counts.size} classes):")
    print(treatment_counts.to_string())
    print(f"\n  Treatment x concentration model ({dose_counts.size} classes):")
    print(dose_counts.to_string())

    def _sample_size_summary(counts):
        return {
            "n_classes": int(counts.size),
            "minimum": int(counts.min()),
            "median": float(counts.median()),
            "maximum": int(counts.max()),
        }

    _treatment_summary = _sample_size_summary(treatment_counts)
    _dose_summary = _sample_size_summary(dose_counts)

    sample_size_table = pd.DataFrame(
        [
            {"Model level": "Treatment", "Classes": _treatment_summary["n_classes"],
             "Minimum wells/class": _treatment_summary["minimum"],
             "Median wells/class": _treatment_summary["median"],
             "Maximum wells/class": _treatment_summary["maximum"]},
            {"Model level": "Treatment x concentration", "Classes": _dose_summary["n_classes"],
             "Minimum wells/class": _dose_summary["minimum"],
             "Median wells/class": _dose_summary["median"],
             "Maximum wells/class": _dose_summary["maximum"]},
        ]
    )
    print("\n  Sample-size summary:")
    print(sample_size_table.to_string(index=False))

    _small_dose_classes = dose_counts[dose_counts < 10]
    if not _small_dose_classes.empty:
        print(f"\n  ℹ️ {len(_small_dose_classes)} treatment x concentration classes have fewer than 10 wells.")
        print("     Dose-level classification can still be performed, but performance estimates")
        print("     will be less stable than treatment-level estimates.")
    else:
        print("\n  ✓ All treatment x concentration classes contain at least 10 wells.")
    return


@app.cell
def _(hom_df):
    print("\n── Section 1 interpretation ───────────────────────────")
    print("  • Treatment-level classes are adequately represented.")
    print("  • Dose-level classes reflect the planned experimental replication and should be")
    print("    interpreted with greater uncertainty.")
    if (hom_df["pct_features_high_SD"] > 10).any():
        print("  • At least one treatment group shows elevated dispersion; regularized LDA is recommended.")
    print("  • No group should be excluded solely because it violates the equal-dispersion heuristic.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 2 — PCA (unsupervised baseline)

    **SC-11:** PCA coloured by plate should show no strong plate clustering after NB02 normalization.
    """)
    return


@app.cell
def _(FIGS_DIR, N_PCA_COMPONENTS, PCA, RANDOM_STATE, X, np, plt):
    pca_full = PCA(n_components=min(N_PCA_COMPONENTS, X.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca_full.fit_transform(X)

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _n_show = min(30, X_pca.shape[1])
    _evr = pca_full.explained_variance_ratio_[:_n_show]
    _ax.bar(range(1, _n_show + 1), _evr * 100, color="#1976d2", alpha=0.8)
    _ax2 = _ax.twinx()
    _ax2.plot(range(1, _n_show + 1), np.cumsum(_evr) * 100, "o-", color="#d32f2f", markersize=4, linewidth=1.5)
    _ax2.axhline(80, color="grey", linestyle="--", linewidth=1)
    _ax2.set_ylabel("Cumulative variance (%)", color="#d32f2f")
    _ax.set_xlabel("Principal component")
    _ax.set_ylabel("Variance explained (%)")
    _ax.set_title("PCA scree plot")
    _fig.tight_layout()
    _fig.savefig(FIGS_DIR / "pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close(_fig)
    print(f"  PC1: {_evr[0]:.1%}  |  PC2: {_evr[1]:.1%}  |  Top-{_n_show}: {_evr.sum():.1%}")
    _fig
    return X_pca, pca_full


@app.cell
def _(CONFIG, FIGS_DIR, X_pca, df, pca_full, plt, scatter_panel):
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter_panel(_axes[0], X_pca[:, 0], X_pca[:, 1], df[CONFIG.treatment_col].tolist(), "Treatment")
    scatter_panel(
        _axes[1], X_pca[:, 0], X_pca[:, 1], df[CONFIG.plate_col].tolist(),
        "Plate (SC-11: should not cluster)",
    )
    if CONFIG.concentration_col:
        scatter_panel(
            _axes[2], X_pca[:, 0], X_pca[:, 1], df[CONFIG.concentration_col].tolist(),
            "Concentration", continuous=True,
        )
    for _ax in _axes:
        _ax.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]:.1%})")
        _ax.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]:.1%})")
    _fig.suptitle("PCA — per-well profiles", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(FIGS_DIR / "pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(_fig)
    print("  ✓  Saved: pca_scatter.png")
    print("  SC-11: Check that the middle panel shows no plate-specific clusters.")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 3 — UMAP (non-linear embedding)
    """)
    return


@app.cell
def _(
    CONFIG,
    FIGS_DIR,
    RANDOM_STATE,
    UMAP,
    UMAP_OK,
    X_pca,
    df,
    plt,
    scatter_panel,
):
    if UMAP_OK:
        _reducer = UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=RANDOM_STATE, n_jobs=1)
        X_umap = _reducer.fit_transform(X_pca)

        _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
        scatter_panel(_axes[0], X_umap[:, 0], X_umap[:, 1], df[CONFIG.treatment_col].tolist(), "Treatment")
        scatter_panel(_axes[1], X_umap[:, 0], X_umap[:, 1], df[CONFIG.plate_col].tolist(), "Plate (batch check)")
        if CONFIG.concentration_col:
            scatter_panel(
                _axes[2], X_umap[:, 0], X_umap[:, 1], df[CONFIG.concentration_col].tolist(),
                "Concentration", continuous=True,
            )
        for _ax in _axes:
            _ax.set_xlabel("UMAP 1")
            _ax.set_ylabel("UMAP 2")
        _fig.suptitle("UMAP — per-well profiles (input: top PCs)", fontsize=13, fontweight="bold")
        _fig.tight_layout()
        _fig.savefig(FIGS_DIR / "umap_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(_fig)
        print("  ✓  Saved: umap_scatter.png")
        _display = _fig
    else:
        X_umap = None
        print("  ⚠️  UMAP skipped. Install with: pip install umap-learn")
        _display = None
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 4 — Phenotypic Profile Quality Control

    This asks whether the extracted profiles still contain systematic plate-associated structure
    **before** supervised modelling, combining PCA of negative controls only, silhouette scores,
    PERMANOVA, and PERMDISP.
    """)
    return


@app.cell
def _(
    CONFIG,
    FIGS_DIR,
    PCA,
    RANDOM_STATE,
    X,
    df,
    negcon_mask,
    plt,
    scatter_panel,
):
    if int(negcon_mask.sum()) >= 6 and df.loc[negcon_mask, CONFIG.plate_col].nunique() >= 2:
        X_negcon = X[negcon_mask.to_numpy()]
        _pca_negcon = PCA(
            n_components=min(2, X_negcon.shape[0] - 1, X_negcon.shape[1]), random_state=RANDOM_STATE
        )
        X_negcon_pca = _pca_negcon.fit_transform(X_negcon)

        _fig, _ax = plt.subplots(figsize=(8, 6))
        scatter_panel(
            _ax, X_negcon_pca[:, 0], X_negcon_pca[:, 1],
            df.loc[negcon_mask, CONFIG.plate_col].tolist(), "Negative controls only — plate mixing",
        )
        _ax.set_xlabel(f"PC1 ({_pca_negcon.explained_variance_ratio_[0]:.1%})")
        _ax.set_ylabel(f"PC2 ({_pca_negcon.explained_variance_ratio_[1]:.1%})")
        _fig.tight_layout()
        _fig.savefig(FIGS_DIR / "profile_qc_negcon_pca.png", dpi=150, bbox_inches="tight")
        plt.close(_fig)
        _display = _fig
    else:
        X_negcon = None
        X_negcon_pca = None
        print("  ⚠️  Too few negative controls or plates for a controls-only PCA.")
        _display = None
    _display
    return X_negcon, X_negcon_pca


@app.cell
def _(
    COMPARISON_RESULTS_DIR,
    CONFIG,
    N_PCA_COMPONENTS,
    N_QC_PERMUTATIONS,
    OVERWRITE_EXISTING_OUTPUTS,
    PCA,
    RANDOM_STATE,
    X_negcon,
    X_negcon_pca,
    X_pca,
    df,
    negcon_mask,
    pd,
    qc_record,
    write_csv_protected,
):
    _qc_dimensions = min(N_PCA_COMPONENTS, X_pca.shape[1])
    _X_qc = X_pca[:, :_qc_dimensions]
    _qc_records = [
        qc_record("all_wells", "plate", _X_qc, df[CONFIG.plate_col].values,
                   permutations=N_QC_PERMUTATIONS, random_state=RANDOM_STATE),
        qc_record("all_wells", "treatment", _X_qc, df[CONFIG.treatment_col].values,
                   permutations=N_QC_PERMUTATIONS, random_state=RANDOM_STATE),
    ]
    if X_negcon_pca is not None:
        _negcon_dimensions = min(_qc_dimensions, X_negcon.shape[0] - 1, X_negcon.shape[1])
        _X_negcon_qc = PCA(n_components=_negcon_dimensions, random_state=RANDOM_STATE).fit_transform(X_negcon)
        _qc_records.append(
            qc_record(
                "negative_controls", "plate", _X_negcon_qc,
                df.loc[negcon_mask, CONFIG.plate_col].values,
                permutations=N_QC_PERMUTATIONS, random_state=RANDOM_STATE,
            )
        )

    batch_qc_df = pd.DataFrame(_qc_records)
    _status = write_csv_protected(
        batch_qc_df, COMPARISON_RESULTS_DIR / "batch_qc_summary.csv", overwrite=OVERWRITE_EXISTING_OUTPUTS
    )
    print(batch_qc_df.to_string(index=False))
    print(f"  ✓  Batch-QC summary {_status}: {COMPARISON_RESULTS_DIR / 'batch_qc_summary.csv'}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 5 — Build equivalent latent modelling spaces

    The comparison is performed between matrices with the same dimensionality: **uncorrected**
    (`X_pca`) and, optionally, **harmony** (Harmony-corrected `X_pca`).
    """)
    return


@app.cell
def _(FIGS_DIR, MODELS_OUT_DIR, RESULTS_DIR, RUN_ANALYSIS_SPACES):
    SPACE_DIRECTORIES = {}
    for _space_name in RUN_ANALYSIS_SPACES:
        _directories = {
            "figures": FIGS_DIR / _space_name,
            "results": RESULTS_DIR / _space_name,
            "models": MODELS_OUT_DIR / _space_name,
            "embeddings": FIGS_DIR / _space_name / "embeddings",
            "lda": FIGS_DIR / _space_name / "lda",
            "clustering": FIGS_DIR / _space_name / "clustering",
        }
        for _d in _directories.values():
            _d.mkdir(parents=True, exist_ok=True)
        SPACE_DIRECTORIES[_space_name] = _directories

    for _space_name, _directories in SPACE_DIRECTORIES.items():
        print(f"Analysis space: {_space_name}")
        print(f"  Figures: {_directories['figures']}")
        print(f"  Results: {_directories['results']}")
        print(f"  Models : {_directories['models']}")
    return (SPACE_DIRECTORIES,)


@app.cell
def _(
    CONFIG,
    OVERWRITE_EXISTING_OUTPUTS,
    RANDOM_STATE,
    RUN_ANALYSIS_SPACES,
    SPACE_DIRECTORIES,
    X_pca,
    df,
    meta_cols,
    np,
    pd,
    validate_output_path,
    write_csv_protected,
):
    # Harmony batch correction is not (yet) part of the shared hca_pipeline
    # package (confirmed absent from modelling.py/stats.py/etc.) — it is kept
    # local to this notebook since it is this section's specific concern.
    X_pca_harmony = None
    if "harmony" in RUN_ANALYSIS_SPACES:
        try:
            import harmonypy as hm
        except ImportError as _error:
            raise ImportError(
                "Harmony analysis was requested, but harmonypy is not installed. "
                "Add harmonypy to the Pixi environment and rerun."
            ) from _error

        _harmony_meta = df[[CONFIG.plate_col]].astype(str).reset_index(drop=True)
        _harmony_result = hm.run_harmony(
            X_pca, _harmony_meta, vars_use=[CONFIG.plate_col], random_state=RANDOM_STATE, verbose=False,
        )

        _harmony_raw = np.asarray(_harmony_result.Z_corr)
        if _harmony_raw.shape == X_pca.shape:
            X_pca_harmony = _harmony_raw
        elif _harmony_raw.T.shape == X_pca.shape:
            X_pca_harmony = _harmony_raw.T
        else:
            raise ValueError(
                f"Unexpected Harmony output shape. Input PCA shape: {X_pca.shape}, "
                f"Harmony raw shape: {_harmony_raw.shape}"
            )
        if X_pca_harmony.shape[0] != len(df):
            raise ValueError("Harmony output does not match the number of wells.")

        _harmony_coordinates_df = pd.DataFrame(
            X_pca_harmony,
            columns=[f"Harmony_PC{i + 1}" for i in range(X_pca_harmony.shape[1])],
            index=df.index,
        )
        _harmony_coordinates_df = pd.concat([df[meta_cols], _harmony_coordinates_df], axis=1)
        _harmony_path = validate_output_path(
            SPACE_DIRECTORIES["harmony"]["results"] / "harmony_coordinates.csv", OVERWRITE_EXISTING_OUTPUTS
        )
        _status = write_csv_protected(_harmony_coordinates_df, _harmony_path, overwrite=OVERWRITE_EXISTING_OUTPUTS)
        print(f"Harmony-corrected matrix: {X_pca_harmony.shape} ({_status})")

    MODELLING_SPACES = {
        "uncorrected": {
            "label": "Uncorrected PCA space", "X_latent": X_pca,
            "description": "PCA coordinates from normalized features",
        }
    }
    if X_pca_harmony is not None:
        MODELLING_SPACES["harmony"] = {
            "label": "Harmony-corrected PCA space", "X_latent": X_pca_harmony,
            "description": "Harmony-corrected PCA coordinates",
        }

    _invalid_spaces = [name for name in RUN_ANALYSIS_SPACES if name not in MODELLING_SPACES]
    if _invalid_spaces:
        raise ValueError(f"Unavailable analysis spaces: {_invalid_spaces}")

    for _name in RUN_ANALYSIS_SPACES:
        _matrix = MODELLING_SPACES[_name]["X_latent"]
        if _matrix.shape != X_pca.shape:
            raise ValueError(f"Space '{_name}' has shape {_matrix.shape}; expected {X_pca.shape}.")
        print(f"{_name:12s}: {_matrix.shape} — {MODELLING_SPACES[_name]['description']}")
    return (MODELLING_SPACES,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 6 — Parallel modelling analyses

    The same UMAP, LDA, leave-one-plate-out validation, clustering, and multivariate QC are run
    in both latent spaces via the shared `run_modelling_space` routine. Coordinates/models are
    kept in memory here (`directories=None`) so this notebook can re-export them with the full
    metadata column set (including well identifiers), matching the original notebook's output
    schema exactly rather than the shared routine's slimmer default metadata subset.
    """)
    return


@app.cell
def _(
    CONFIG,
    MODELLING_SPACES,
    N_QC_PERMUTATIONS,
    OVERWRITE_EXISTING_OUTPUTS,
    RANDOM_STATE,
    RUN_ANALYSIS_SPACES,
    SPACE_DIRECTORIES,
    UMAP_OK,
    df,
    joblib,
    meta_cols,
    mo,
    negcon_mask,
    pd,
    plt,
    run_modelling_space,
    scatter_panel,
    validate_output_path,
    write_csv_protected,
):
    def _export_with_full_metadata(values, columns, directories_key, filename, extra=None):
        """Re-export a coordinate/label array with the full Metadata_* column set
        (+ "_dose_label"), matching the source notebook's output schema."""
        out_df = df[meta_cols + ["_dose_label"]].copy()
        if extra is not None:
            for col, series in extra.items():
                out_df[col] = series
        if values is not None:
            coord_df = pd.DataFrame(values, columns=columns, index=df.index)
            out_df = pd.concat([out_df, coord_df], axis=1)
        path = validate_output_path(directories_key / filename, OVERWRITE_EXISTING_OUTPUTS)
        return write_csv_protected(out_df, path, overwrite=OVERWRITE_EXISTING_OUTPUTS)

    analysis_results = {}
    _figures_to_display = []
    for _space_name in RUN_ANALYSIS_SPACES:
        _space = MODELLING_SPACES[_space_name]
        _directories = SPACE_DIRECTORIES[_space_name]
        print("\n" + "═" * 72)
        print(f"RUNNING MODELLING SPACE: {_space['label']}")
        print("═" * 72)

        _result = run_modelling_space(
            space_name=_space_name,
            space_label=_space["label"],
            X_latent=_space["X_latent"],
            metadata_df=df,
            treatment_col=CONFIG.treatment_col,
            plate_col=CONFIG.plate_col,
            dose_col=CONFIG.concentration_col if CONFIG.has_dose_axis else None,
            negcon_mask=negcon_mask.to_numpy(),
            directories=None,
            random_state=RANDOM_STATE,
            n_qc_permutations=N_QC_PERMUTATIONS,
            overwrite_existing_outputs=OVERWRITE_EXISTING_OUTPUTS,
            run_umap=UMAP_OK,
        )
        analysis_results[_space_name] = _result

        # ── Re-export tables with the full metadata column set ─────────────
        _status = write_csv_protected(
            _result["qc"], validate_output_path(_directories["results"] / "multivariate_qc.csv", OVERWRITE_EXISTING_OUTPUTS),
            overwrite=OVERWRITE_EXISTING_OUTPUTS,
        )
        print(f"  ✓ multivariate_qc.csv {_status}")

        if _result.get("umap_coordinates") is not None:
            _status = _export_with_full_metadata(
                _result["umap_coordinates"], ["UMAP1", "UMAP2"], _directories["results"], "umap_coordinates.csv"
            )
            print(f"  ✓ umap_coordinates.csv {_status}")

            _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
            scatter_panel(_axes[0], _result["umap_coordinates"][:, 0], _result["umap_coordinates"][:, 1],
                          df[CONFIG.treatment_col].tolist(), "Treatment")
            scatter_panel(_axes[1], _result["umap_coordinates"][:, 0], _result["umap_coordinates"][:, 1],
                          df[CONFIG.plate_col].tolist(), "Plate")
            if CONFIG.concentration_col:
                scatter_panel(_axes[2], _result["umap_coordinates"][:, 0], _result["umap_coordinates"][:, 1],
                              df[CONFIG.concentration_col].tolist(), "Concentration", continuous=True)
            for _ax in _axes:
                _ax.set_xlabel("UMAP 1")
                _ax.set_ylabel("UMAP 2")
            _fig.suptitle(f"UMAP — {_space['label']}", fontweight="bold")
            _fig.tight_layout()
            _fig.savefig(
                validate_output_path(_directories["embeddings"] / "umap.png", OVERWRITE_EXISTING_OUTPUTS),
                dpi=150, bbox_inches="tight",
            )
            plt.close(_fig)
            _figures_to_display.append(_fig)

        _treatment_lda = _result.get("treatment_lda", {})
        if isinstance(_treatment_lda.get("cv"), pd.DataFrame):
            _status = write_csv_protected(
                _treatment_lda["cv"],
                validate_output_path(_directories["results"] / "treatment_lda_cv.csv", OVERWRITE_EXISTING_OUTPUTS),
                overwrite=OVERWRITE_EXISTING_OUTPUTS,
            )
            print(f"  ✓ treatment_lda_cv.csv {_status}")
        if _treatment_lda.get("model") is not None:
            joblib.dump(
                _treatment_lda["model"],
                validate_output_path(_directories["models"] / "lda_treatment.pkl", OVERWRITE_EXISTING_OUTPUTS),
            )

        _dose_lda = _result.get("dose_lda", {})
        if _dose_lda.get("X_lda") is not None:
            _lda_columns = [f"LDA{i + 1}" for i in range(_dose_lda["X_lda"].shape[1])]
            _status = _export_with_full_metadata(
                _dose_lda["X_lda"], _lda_columns, _directories["results"], "lda_coordinates.csv"
            )
            print(f"  ✓ lda_coordinates.csv {_status}")

            if isinstance(_dose_lda.get("cv"), pd.DataFrame):
                _status = write_csv_protected(
                    _dose_lda["cv"],
                    validate_output_path(_directories["results"] / "dose_lda_cv.csv", OVERWRITE_EXISTING_OUTPUTS),
                    overwrite=OVERWRITE_EXISTING_OUTPUTS,
                )
                print(f"  ✓ dose_lda_cv.csv {_status}")
            if _dose_lda.get("confusion_matrix") is not None:
                _cm_path = validate_output_path(_directories["results"] / "confusion_matrix.csv", OVERWRITE_EXISTING_OUTPUTS)
                _dose_lda["confusion_matrix"].to_csv(_cm_path)
                print("  ✓ confusion_matrix.csv written")

            _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
            scatter_panel(_axes[0], _dose_lda["X_lda"][:, 0], _dose_lda["X_lda"][:, 1],
                          df[CONFIG.treatment_col].tolist(), "Treatment")
            scatter_panel(_axes[1], _dose_lda["X_lda"][:, 0], _dose_lda["X_lda"][:, 1],
                          df[CONFIG.plate_col].tolist(), "Plate")
            for _ax in _axes:
                _ax.set_xlabel("LD1")
                _ax.set_ylabel("LD2")
            _fig.suptitle(f"Dose-level LDA — {_space['label']}", fontweight="bold")
            _fig.tight_layout()
            _fig.savefig(
                validate_output_path(_directories["lda"] / "lda_dose.png", OVERWRITE_EXISTING_OUTPUTS),
                dpi=150, bbox_inches="tight",
            )
            plt.close(_fig)
            _figures_to_display.append(_fig)
        if _dose_lda.get("model") is not None:
            joblib.dump(
                _dose_lda["model"],
                validate_output_path(_directories["models"] / "lda_dose.pkl", OVERWRITE_EXISTING_OUTPUTS),
            )

        _clustering = _result.get("clustering", {})
        if _clustering.get("labels") is not None:
            _status = _export_with_full_metadata(
                None, [], _directories["results"], "clustering_results.csv",
                extra={"cluster": _clustering["labels"]},
            )
            print(f"  ✓ clustering_results.csv {_status}")
        if _clustering.get("model") is not None:
            joblib.dump(
                _clustering["model"],
                validate_output_path(_directories["models"] / "kmeans.pkl", OVERWRITE_EXISTING_OUTPUTS),
            )

        _status = write_csv_protected(
            pd.DataFrame([_result["summary"]]),
            validate_output_path(_directories["results"] / "analysis_summary.csv", OVERWRITE_EXISTING_OUTPUTS),
            overwrite=OVERWRITE_EXISTING_OUTPUTS,
        )
        print(f"  ✓ analysis_summary.csv {_status}")
        print(f"  Summary: {_result['summary']}")
    mo.vstack(_figures_to_display) if _figures_to_display else None
    return (analysis_results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 7 — Interpretable phenotypic fingerprints

    Fingerprints are calculated **once**, from the normalized CellProfiler features. Harmony
    components are not mapped back to Intensity/Texture/AreaShape categories. Cohen's *d* is
    calculated by experimental condition (Treatment x Concentration); each condition is compared
    with negative controls, preferably combined within-plate.

    Detailed feature-level inspection and expanded radar analyses live in
    `05_phenotypic_fingerprints.py`; this notebook retains only the compact overview needed to
    contextualize the modelling results above (mirroring the source notebook's own scope note).
    """)
    return


@app.cell
def _(CONFIG, df, negcon_mask, pd):
    def format_concentration(value):
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)

    # Feature-category vocabulary matching the already-committed fingerprint_*.csv
    # files on disk exactly (AreaShape/Correlation/... row labels) — kept local
    # rather than hca_pipeline.taxonomy.classify_feature, whose family labels
    # ("Area & shape", ...) differ and would break parity with historical outputs.
    _category_patterns = {
        "Intensity": "_Intensity_", "Texture": "_Texture_", "AreaShape": "_AreaShape_",
        "Granularity": "_Granularity_", "Correlation": "_Correlation_",
        "RadialDistribution": "_RadialDistribution_", "Location": "_Location_",
    }

    def feature_category(feature_name):
        for category, pattern in _category_patterns.items():
            if pattern in feature_name:
                return category
        if "Count" in feature_name or "Children" in feature_name:
            return "ObjectCount"
        return "Other"

    if CONFIG.has_dose_axis and CONFIG.concentration_col:
        _condition = (
            df[CONFIG.treatment_col].astype(str) + "__" + df[CONFIG.concentration_col].map(format_concentration)
        )
        _concentration_label = df[CONFIG.concentration_col].map(format_concentration)
    else:
        _condition = df[CONFIG.treatment_col].astype(str)
        _concentration_label = pd.Series("", index=df.index)

    df_fp = df.assign(Metadata_Condition=_condition, Metadata_Concentration_Label=_concentration_label)

    condition_table = (
        df_fp.loc[~negcon_mask, [CONFIG.treatment_col, "Metadata_Concentration_Label", "Metadata_Condition"]]
        .drop_duplicates()
        .rename(columns={"Metadata_Concentration_Label": "_concentration_label_sort"})
    )
    if CONFIG.has_dose_axis and CONFIG.concentration_col:
        condition_table[CONFIG.concentration_col] = df_fp.loc[condition_table.index, CONFIG.concentration_col]
        condition_table = condition_table.sort_values([CONFIG.treatment_col, CONFIG.concentration_col])
    else:
        condition_table[CONFIG.concentration_col or "Metadata_Concentration"] = 0.0
        condition_table = condition_table.sort_values([CONFIG.treatment_col])
    condition_table = condition_table.rename(columns={"_concentration_label_sort": "Metadata_Concentration_Label"})

    condition_order = condition_table["Metadata_Condition"].tolist()
    print(f"Conditions: {len(condition_order)}")
    return condition_order, condition_table, df_fp, feature_category


@app.cell
def _(
    CONFIG,
    FINGERPRINT_RESULTS_DIR,
    FINGERPRINT_USE_WITHIN_PLATE_EFFECTS,
    OVERWRITE_EXISTING_OUTPUTS,
    calculate_within_plate_effects,
    condition_table,
    df_fp,
    feat_cols,
    feature_category,
    negcon_mask,
    validate_output_path,
    write_csv_protected,
):
    _conc_col = CONFIG.concentration_col or "Metadata_Concentration"
    effect_long_df = calculate_within_plate_effects(
        df_fp, feat_cols,
        condition_table=condition_table,
        negative_control_mask=negcon_mask,
        treatment_column=CONFIG.treatment_col,
        concentration_column=_conc_col,
        plate_column=CONFIG.plate_col,
    )
    if not FINGERPRINT_USE_WITHIN_PLATE_EFFECTS:
        from hca_pipeline.stats import calculate_global_effects

        effect_long_df = calculate_global_effects(
            df_fp, feat_cols,
            condition_table=condition_table,
            negative_control_mask=negcon_mask,
            treatment_column=CONFIG.treatment_col,
            concentration_column=_conc_col,
            plate_column=CONFIG.plate_col,
        )

    effect_long_df["FeatureCategory"] = effect_long_df["feature"].map(feature_category)
    effect_long_df = effect_long_df.rename(
        columns={
            "condition": "Condition", "treatment": "Treatment", "concentration": "Concentration",
            "feature": "Feature", "effect_size": "Cohens_d", "n_plates": "N_plates",
            "estimation_mode": "Method",
        }
    )

    _path = validate_output_path(
        FINGERPRINT_RESULTS_DIR / "cohens_d_by_condition_long.csv", OVERWRITE_EXISTING_OUTPUTS
    )
    _status = write_csv_protected(effect_long_df, _path, overwrite=OVERWRITE_EXISTING_OUTPUTS)
    print(f"Fingerprint effects saved for {effect_long_df['Condition'].nunique()} conditions ({_status}).")
    return (effect_long_df,)


@app.cell
def _(
    FINGERPRINT_EFFECT_THRESHOLD,
    FINGERPRINT_RESULTS_DIR,
    OVERWRITE_EXISTING_OUTPUTS,
    build_fingerprint_matrix,
    condition_order,
    effect_long_df,
    np,
    validate_output_path,
    write_csv_protected,
):
    fingerprint_tables = {}
    for _name, _agg in [
        ("mean_absolute", lambda v: np.nanmean(np.abs(v))),
        ("mean_signed", "mean"),
        ("max_absolute", lambda v: np.nanmax(np.abs(v))),
        ("fraction_changed", lambda v: np.nanmean(np.abs(v) >= FINGERPRINT_EFFECT_THRESHOLD)),
    ]:
        _table = build_fingerprint_matrix(
            effect_long_df, grouping_level="FeatureCategory", metric_column="Cohens_d",
            aggregation=_agg, condition_column="Condition", condition_order=condition_order,
            drop_other=False,
        )
        fingerprint_tables[_name] = _table
        _status = write_csv_protected(
            _table.reset_index(),
            validate_output_path(FINGERPRINT_RESULTS_DIR / f"fingerprint_{_name}.csv", OVERWRITE_EXISTING_OUTPUTS),
            overwrite=OVERWRITE_EXISTING_OUTPUTS,
        )
        print(f"  ✓ fingerprint_{_name}.csv {_status}")
    return (fingerprint_tables,)


@app.cell
def _(
    FINGERPRINT_EFFECT_THRESHOLD,
    FINGERPRINT_FIGURES_DIR,
    fingerprint_tables,
    mo,
    plot_fingerprint_heatmap,
):
    # The saved CSVs (previous cell) keep the "Other" feature-category row for
    # completeness; heatmaps drop it, matching the source notebook's plotting
    # helper (which visually excludes "Other" but never removed it from disk).
    _heatmap_figs = [
        plot_fingerprint_heatmap(
            fingerprint_tables["mean_absolute"].drop(index="Other", errors="ignore"),
            title="Phenotypic fingerprint by feature category",
            colorbar_label="Mean absolute effect vs negative control",
            output_path=FINGERPRINT_FIGURES_DIR / "mean_absolute_heatmap.png",
        ),
        plot_fingerprint_heatmap(
            fingerprint_tables["mean_signed"].drop(index="Other", errors="ignore"),
            title="Directional phenotypic effect by feature category",
            colorbar_label="Mean signed effect vs negative control",
            output_path=FINGERPRINT_FIGURES_DIR / "mean_signed_heatmap.png",
            diverging=True,
        ),
        plot_fingerprint_heatmap(
            fingerprint_tables["max_absolute"].drop(index="Other", errors="ignore"),
            title="Maximum absolute effect by feature category",
            colorbar_label="Maximum absolute effect vs negative control",
            output_path=FINGERPRINT_FIGURES_DIR / "max_absolute_heatmap.png",
        ),
        plot_fingerprint_heatmap(
            fingerprint_tables["fraction_changed"].drop(index="Other", errors="ignore"),
            title="Fraction of altered features by category",
            colorbar_label=f"Fraction with |effect| ≥ {FINGERPRINT_EFFECT_THRESHOLD:g}",
            output_path=FINGERPRINT_FIGURES_DIR / "fraction_changed_heatmap.png",
            fixed_range=(0, 1),
        ),
    ]
    print("  ✓  Fingerprint heatmaps saved.")
    mo.vstack([f for f in _heatmap_figs if f is not None])
    return


@app.cell
def _(FINGERPRINT_FIGURES_DIR, fingerprint_tables, mo):
    from hca_pipeline.plotting import plot_condition_radar_grid

    # Small-multiples radar overview (one panel per condition). The source
    # notebook additionally laid these out in a treatment x concentration grid
    # with per-treatment shared axes; that richer layout is deliberately left
    # to 05_phenotypic_fingerprints.py (as the source notebook's own docstring
    # says: "expanded LCP radar analyses are performed in
    # 04_phenotypic_fingerprints.ipynb"). This section keeps only a compact
    # overview using the shared plotting helper as-is.
    _radar_figs = [
        plot_condition_radar_grid(
            fingerprint_tables["mean_absolute"].drop(index="Other", errors="ignore"),
            title="Phenotypic fingerprint magnitude",
            output_path=FINGERPRINT_FIGURES_DIR / "mean_absolute_radar_small_multiples.png",
        ),
        plot_condition_radar_grid(
            fingerprint_tables["fraction_changed"].drop(index="Other", errors="ignore"),
            title="Breadth of phenotypic response",
            output_path=FINGERPRINT_FIGURES_DIR / "fraction_changed_radar_small_multiples.png",
        ),
    ]
    print("  ✓  Fingerprint radar overview saved.")
    mo.vstack([f for f in _radar_figs if f is not None])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 8 — Comparison and recommendation

    The recommendation is evidence-based but never silently selects the downstream space —
    `FINAL_MODELLING_SPACE` remains an explicit user choice.
    """)
    return


@app.cell
def _(
    COMPARISON_RESULTS_DIR,
    OVERWRITE_EXISTING_OUTPUTS,
    RUN_ANALYSIS_SPACES,
    analysis_results,
    get_qc_metric,
    pd,
    validate_output_path,
    write_csv_protected,
):
    _comparison_rows = []
    for _space_name in RUN_ANALYSIS_SPACES:
        _summary = analysis_results[_space_name]["summary"]
        _comparison_rows.append(
            {
                "space": _space_name,
                "plate_permanova_R2": get_qc_metric(analysis_results, _space_name, "all_wells", "plate", "permanova_R2"),
                "plate_permanova_p": get_qc_metric(analysis_results, _space_name, "all_wells", "plate", "permanova_p"),
                "plate_silhouette": get_qc_metric(analysis_results, _space_name, "all_wells", "plate", "silhouette"),
                "treatment_permanova_R2": get_qc_metric(analysis_results, _space_name, "all_wells", "treatment", "permanova_R2"),
                "treatment_permanova_p": get_qc_metric(analysis_results, _space_name, "all_wells", "treatment", "permanova_p"),
                "treatment_silhouette": get_qc_metric(analysis_results, _space_name, "all_wells", "treatment", "silhouette"),
                "negcon_plate_permanova_R2": get_qc_metric(analysis_results, _space_name, "negative_controls", "plate", "permanova_R2"),
                "negcon_plate_permanova_p": get_qc_metric(analysis_results, _space_name, "negative_controls", "plate", "permanova_p"),
                "negcon_plate_silhouette": get_qc_metric(analysis_results, _space_name, "negative_controls", "plate", "silhouette"),
                "dose_cv_balanced_accuracy": _summary["dose_cv_balanced_accuracy"],
                "dose_cv_macro_f1": _summary["dose_cv_macro_f1"],
                "cluster_silhouette": _summary["cluster_silhouette"],
                "n_dimensions": _summary["n_dimensions"],
            }
        )

    comparison_df = pd.DataFrame(_comparison_rows).set_index("space")
    _path = validate_output_path(
        COMPARISON_RESULTS_DIR / "uncorrected_vs_harmony_summary.csv", OVERWRITE_EXISTING_OUTPUTS
    )
    _status = write_csv_protected(comparison_df.reset_index(), _path, overwrite=OVERWRITE_EXISTING_OUTPUTS)
    print(comparison_df.to_string())
    print(f"  ✓ uncorrected_vs_harmony_summary.csv {_status}")
    return (comparison_df,)


@app.cell
def _(
    BATCH_R2_WARN_THRESHOLD,
    FINAL_MODELLING_SPACE,
    MAX_REPLICABILITY_RELATIVE_LOSS,
    MAX_TREATMENT_R2_RELATIVE_LOSS,
    MAX_TREATMENT_SILHOUETTE_ABSOLUTE_LOSS,
    MIN_PLATE_R2_RELATIVE_REDUCTION,
    RUN_ANALYSIS_SPACES,
    absolute_change,
    comparison_df,
    np,
    safe_relative_change,
):
    recommendation = "inconclusive"
    recommendation_confidence = "low"
    recommendation_reasons = []
    recommendation_criteria = {}
    recommended_primary_space = None

    if {"uncorrected", "harmony"}.issubset(comparison_df.index):
        _original = comparison_df.loc["uncorrected"]
        _corrected = comparison_df.loc["harmony"]

        _plate_r2_relative_change = safe_relative_change(_corrected["plate_permanova_R2"], _original["plate_permanova_R2"])
        _plate_r2_reduction = -_plate_r2_relative_change if np.isfinite(_plate_r2_relative_change) else np.nan
        _plate_silhouette_change = absolute_change(_corrected["plate_silhouette"], _original["plate_silhouette"])

        _treatment_r2_relative_change = safe_relative_change(_corrected["treatment_permanova_R2"], _original["treatment_permanova_R2"])
        _treatment_silhouette_change = absolute_change(_corrected["treatment_silhouette"], _original["treatment_silhouette"])

        _balanced_accuracy_relative_change = safe_relative_change(_corrected["dose_cv_balanced_accuracy"], _original["dose_cv_balanced_accuracy"])
        _macro_f1_relative_change = safe_relative_change(_corrected["dose_cv_macro_f1"], _original["dose_cv_macro_f1"])

        _negcon_plate_p_original = _original["negcon_plate_permanova_p"]
        _negcon_plate_r2_original = _original["negcon_plate_permanova_R2"]
        _negcon_batch_detected = bool(
            np.isfinite(_negcon_plate_p_original) and np.isfinite(_negcon_plate_r2_original)
            and _negcon_plate_p_original <= 0.05 and _negcon_plate_r2_original >= BATCH_R2_WARN_THRESHOLD
        )

        recommendation_criteria = {
            "plate_R2_reduced": bool(np.isfinite(_plate_r2_reduction) and _plate_r2_reduction >= MIN_PLATE_R2_RELATIVE_REDUCTION),
            "plate_silhouette_reduced": bool(np.isfinite(_plate_silhouette_change) and _plate_silhouette_change < 0),
            "treatment_R2_preserved": bool((not np.isfinite(_treatment_r2_relative_change)) or _treatment_r2_relative_change >= -MAX_TREATMENT_R2_RELATIVE_LOSS),
            "treatment_silhouette_preserved": bool((not np.isfinite(_treatment_silhouette_change)) or _treatment_silhouette_change >= -MAX_TREATMENT_SILHOUETTE_ABSOLUTE_LOSS),
            "balanced_accuracy_preserved": bool((not np.isfinite(_balanced_accuracy_relative_change)) or _balanced_accuracy_relative_change >= -MAX_REPLICABILITY_RELATIVE_LOSS),
        }
        _all_core_criteria_met = all(recommendation_criteria.values())

        if _all_core_criteria_met and _negcon_batch_detected:
            recommendation = "harmony"
            recommendation_confidence = "high"
            recommended_primary_space = "harmony"
        elif _all_core_criteria_met:
            recommendation = "harmony_for_sensitivity_analysis"
            recommendation_confidence = "moderate"
            recommended_primary_space = "uncorrected"
        else:
            recommendation = "uncorrected"
            recommendation_confidence = "moderate"
            recommended_primary_space = "uncorrected"

        recommendation_reasons = [
            f"Plate PERMANOVA R² reduction: {_plate_r2_reduction:.1%}" if np.isfinite(_plate_r2_reduction) else "Plate PERMANOVA R² reduction: unavailable",
            f"Treatment PERMANOVA R² change: {_treatment_r2_relative_change:+.1%}" if np.isfinite(_treatment_r2_relative_change) else "Treatment PERMANOVA R² change: unavailable",
            f"Plate silhouette absolute change: {_plate_silhouette_change:+.3f}" if np.isfinite(_plate_silhouette_change) else "Plate silhouette change: unavailable",
            f"Treatment silhouette absolute change: {_treatment_silhouette_change:+.3f}" if np.isfinite(_treatment_silhouette_change) else "Treatment silhouette change: unavailable",
            f"Dose-CV balanced accuracy change: {_balanced_accuracy_relative_change:+.1%}" if np.isfinite(_balanced_accuracy_relative_change) else "Dose-CV balanced accuracy change: unavailable",
            f"Dose-CV macro-F1 change: {_macro_f1_relative_change:+.1%}" if np.isfinite(_macro_f1_relative_change) else "Dose-CV macro-F1 change: unavailable",
            f"Significant plate-associated structure among negative controls: {_negcon_batch_detected}",
        ]

    print("═" * 76)
    print("COMPARISON OF EQUIVALENT LATENT MODELLING SPACES")
    print("═" * 76)
    print(f"\nRecommendation : {recommendation}")
    print(f"Confidence     : {recommendation_confidence}")
    if recommendation_reasons:
        for _reason in recommendation_reasons:
            print(f"  • {_reason}")
    else:
        print("  • The required pair of modelling spaces was not available; no comparison was performed.")

    print(f"\n── Explicit downstream selection ──")
    print(f"Selected space : {FINAL_MODELLING_SPACE}")
    if FINAL_MODELLING_SPACE not in RUN_ANALYSIS_SPACES:
        raise ValueError(f"FINAL_MODELLING_SPACE={FINAL_MODELLING_SPACE!r} was not run.")
    if recommended_primary_space is None:
        print("  ℹ️ No unique primary space was selected automatically.")
    elif FINAL_MODELLING_SPACE == recommended_primary_space:
        print("  ✓ The explicit selection is consistent with the recommended primary space.")
    else:
        print("  ℹ️ The explicit selection differs from the recommended primary space (allowed).")
    return (
        recommendation,
        recommendation_confidence,
        recommendation_criteria,
        recommendation_reasons,
        recommended_primary_space,
    )


@app.cell
def _(
    COMPARISON_FIGURES_DIR,
    CONFIG,
    MODELLING_SPACES,
    RUN_ANALYSIS_SPACES,
    UMAP_OK,
    analysis_results,
    df,
    plt,
    scatter_panel,
):
    if UMAP_OK and {"uncorrected", "harmony"}.issubset(analysis_results) and {"uncorrected", "harmony"}.issubset(RUN_ANALYSIS_SPACES):
        _fig, _axes = plt.subplots(2, 3, figsize=(18, 11))
        for _row_index, _space_name in enumerate(["uncorrected", "harmony"]):
            _coordinates = analysis_results[_space_name]["umap_coordinates"]
            if _coordinates is None:
                continue
            _label = MODELLING_SPACES[_space_name]["label"]
            scatter_panel(_axes[_row_index, 0], _coordinates[:, 0], _coordinates[:, 1],
                          df[CONFIG.treatment_col].astype(str).tolist(), f"{_label} — Treatment")
            scatter_panel(_axes[_row_index, 1], _coordinates[:, 0], _coordinates[:, 1],
                          df[CONFIG.plate_col].astype(str).tolist(), f"{_label} — Plate")
            if CONFIG.concentration_col:
                scatter_panel(_axes[_row_index, 2], _coordinates[:, 0], _coordinates[:, 1],
                              df[CONFIG.concentration_col].tolist(), f"{_label} — Concentration", continuous=True)
            for _ax in _axes[_row_index]:
                _ax.set_xlabel("UMAP 1")
                _ax.set_ylabel("UMAP 2")
        _fig.suptitle("Comparison of equivalent latent modelling spaces", fontweight="bold")
        _fig.tight_layout()
        _fig.savefig(COMPARISON_FIGURES_DIR / "umap_uncorrected_vs_harmony.png", dpi=180, bbox_inches="tight")
        plt.close(_fig)
        print("  ✓  Saved side-by-side UMAP comparison.")
        _display = _fig
    else:
        print("  ℹ️  Side-by-side UMAP comparison skipped (needs both spaces + UMAP installed).")
        _display = None
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 9 — Provenance and completion
    """)
    return


@app.cell
def _(
    COMPARISON_FIGURES_DIR,
    COMPARISON_RESULTS_DIR,
    CONFIG,
    FINAL_MODELLING_SPACE,
    FINGERPRINT_USE_WITHIN_PLATE_EFFECTS,
    MODELLING_SPACES,
    OVERWRITE_EXISTING_OUTPUTS,
    REPO_ROOT,
    RUN_ANALYSIS_SPACES,
    SPACE_DIRECTORIES,
    X_pca,
    condition_order,
    datetime,
    df,
    feat_cols,
    json,
    recommendation,
    recommendation_confidence,
    recommendation_criteria,
    recommendation_reasons,
    recommended_primary_space,
    subprocess,
    validate_output_path,
):
    try:
        _git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT),
        ).stdout.strip()[:8]
    except Exception:
        _git_hash = "unknown"

    _common_provenance = {
        "notebook": "04_phenotypic_profiling.py",
        "experiment_id": CONFIG.experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": _git_hash,
        "n_wells": int(df.shape[0]),
        "n_cellprofiler_features": int(len(feat_cols)),
        "latent_dimensions": int(X_pca.shape[1]),
        "run_analysis_spaces": RUN_ANALYSIS_SPACES,
        "final_modelling_space": FINAL_MODELLING_SPACE,
        "automated_recommendation": recommendation,
        "recommendation_confidence": recommendation_confidence,
        "recommended_primary_space": recommended_primary_space,
        "fingerprints_source": "normalized original CellProfiler features",
        "fingerprint_condition_definition": "Treatment x Concentration",
        "fingerprint_effect_method": (
            "within-plate weighted Cohen's d" if FINGERPRINT_USE_WITHIN_PLATE_EFFECTS else "global Cohen's d"
        ),
        "n_fingerprint_conditions": len(condition_order),
    }

    _timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    def _write_with_optional_history(directory, payload):
        _path = validate_output_path(directory / "provenance.json", OVERWRITE_EXISTING_OUTPUTS)
        _path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if CONFIG.save_provenance_history:
            _history_path = directory / f"provenance_{_timestamp}.json"
            if _history_path.exists():
                raise FileExistsError(f"Historical provenance file already exists: {_history_path}")
            _history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for _space_name in RUN_ANALYSIS_SPACES:
        _provenance = {
            **_common_provenance,
            "space_name": _space_name,
            "space_description": MODELLING_SPACES[_space_name]["description"],
            "output_directories": {k: str(v) for k, v in SPACE_DIRECTORIES[_space_name].items()},
        }
        _write_with_optional_history(SPACE_DIRECTORIES[_space_name]["results"], _provenance)

    _comparison_provenance = {
        **_common_provenance,
        "recommendation_reasons": recommendation_reasons,
        "recommendation_criteria": recommendation_criteria,
        "comparison_results_dir": str(COMPARISON_RESULTS_DIR),
        "comparison_figures_dir": str(COMPARISON_FIGURES_DIR),
    }
    _write_with_optional_history(COMPARISON_RESULTS_DIR, _comparison_provenance)
    print(f"Historical record           : {'enabled' if CONFIG.save_provenance_history else 'disabled'}")

    print("═" * 72)
    print("NB04 COMPLETE")
    print("═" * 72)
    print(f"Spaces run                 : {', '.join(RUN_ANALYSIS_SPACES)}")
    print(f"Automated recommendation    : {recommendation}")
    print(f"Explicit final space        : {FINAL_MODELLING_SPACE}")
    print(f"Fingerprint conditions      : {len(condition_order)}")
    print(f"Comparison results          : {COMPARISON_RESULTS_DIR}")
    print(f"Comparison figures          : {COMPARISON_FIGURES_DIR}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 10 — Final integrity checks
    """)
    return


@app.cell
def _(
    COMPARISON_RESULTS_DIR,
    FINGERPRINT_RESULTS_DIR,
    RUN_ANALYSIS_SPACES,
    SPACE_DIRECTORIES,
    analysis_results,
    comparison_df,
    df,
):
    _integrity_errors = []

    for _space_name in RUN_ANALYSIS_SPACES:
        _summary = analysis_results[_space_name]["summary"]
        if _summary["n_wells"] != len(df):
            _integrity_errors.append(f"Space '{_space_name}': summary n_wells does not match df row count.")
        _required_files = ["analysis_summary.csv", "multivariate_qc.csv", "provenance.json"]
        for _filename in _required_files:
            _path = SPACE_DIRECTORIES[_space_name]["results"] / _filename
            if not _path.exists():
                _integrity_errors.append(f"Missing expected output: {_path}")

    if not (COMPARISON_RESULTS_DIR / "uncorrected_vs_harmony_summary.csv").exists():
        _integrity_errors.append("Missing comparisons/uncorrected_vs_harmony_summary.csv")
    if not (COMPARISON_RESULTS_DIR / "provenance.json").exists():
        _integrity_errors.append("Missing comparisons/provenance.json")
    if not (FINGERPRINT_RESULTS_DIR / "cohens_d_by_condition_long.csv").exists():
        _integrity_errors.append("Missing fingerprints/cohens_d_by_condition_long.csv")

    if set(comparison_df.index) != set(RUN_ANALYSIS_SPACES):
        _integrity_errors.append("comparison_df space index does not match RUN_ANALYSIS_SPACES.")

    if _integrity_errors:
        raise RuntimeError(
            "\nNB04 integrity checks failed\n============================\n\n"
            + "\n".join(f"  - {e}" for e in _integrity_errors)
        )

    print("═" * 72)
    print("NB04 INTEGRITY CHECKS PASSED")
    print("═" * 72)
    print(f"  Wells profiled          : {len(df):,}")
    print(f"  Spaces run              : {', '.join(RUN_ANALYSIS_SPACES)}")
    print("✓ All final integrity checks passed")
    return


@app.cell
def _():
    return


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
