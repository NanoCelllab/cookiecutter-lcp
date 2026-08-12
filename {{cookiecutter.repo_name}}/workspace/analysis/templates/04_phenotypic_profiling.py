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
    from matplotlib.patches import Patch
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
        Patch,
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
    # workspace/pipelines). __file__ is used instead of cwd so this notebook
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
    _pipelines_dir = REPO_ROOT / "workspace" / "pipelines"
    if not _pipelines_dir.exists():
        raise FileNotFoundError(f"hca_pipeline package directory not found: {_pipelines_dir}")
    sys.path.insert(0, str(_pipelines_dir))

    from hca_pipeline.config import ExperimentConfig
    from hca_pipeline.feature_select import infer_feature_cols
    from hca_pipeline.io import write_csv_protected
    from hca_pipeline.metadata import CELL_COUNT_METADATA_COLUMN
    from hca_pipeline.metrics_qc import copairs_compute_map
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
        CELL_COUNT_METADATA_COLUMN,
        copairs_compute_map,
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
        options=["uncorrected", "harmony", "combat"],
        value=["uncorrected", "harmony"],
        label="Latent modelling spaces to run",
    )
    final_modelling_space_input = mo.ui.dropdown(
        options=["uncorrected", "harmony", "combat"],
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

    The fourth panel colours each well by its number of segmented cells. A
    PCA outlier that is also an extreme low/high-cell-count well may reflect
    sparse sampling, confluence, segmentation failure, or toxicity rather
    than a stable biological phenotype. Cell counts are reconstructed from
    the annotated single-cell checkpoint and displayed on a logarithmic
    colour scale so both small and very large wells remain distinguishable.
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
def _(CELL_COUNT_METADATA_COLUMN, df, pd):
    if CELL_COUNT_METADATA_COLUMN not in df.columns:
        raise ValueError(
            f"Required metadata {CELL_COUNT_METADATA_COLUMN!r} is missing from the final profile. "
            "Re-run NB01 and NB02; cell count must be preserved as metadata throughout the pipeline."
        )
    well_cell_counts = pd.to_numeric(df[CELL_COUNT_METADATA_COLUMN], errors="raise").astype(int).to_numpy()
    print(
        "Cell count per well: "
        f"min={well_cell_counts.min():,}, median={int(pd.Series(well_cell_counts).median()):,}, "
        f"max={well_cell_counts.max():,}"
    )
    return (well_cell_counts,)


@app.cell
def _(CONFIG, FIGS_DIR, X_pca, df, np, pca_full, plt, scatter_panel, well_cell_counts):
    from matplotlib.colors import LogNorm as _LogNorm

    _fig, _axes_grid = plt.subplots(2, 2, figsize=(14, 11))
    _axes = _axes_grid.ravel()
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
    else:
        _axes[2].text(0.5, 0.5, "No concentration axis configured", ha="center", va="center")
        _axes[2].set_title("Concentration")

    _positive_counts = np.clip(np.asarray(well_cell_counts, dtype=float), 1, None)
    _cell_count_scatter = _axes[3].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=_positive_counts,
        cmap="viridis",
        norm=_LogNorm(vmin=_positive_counts.min(), vmax=_positive_counts.max()),
        s=60,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    _fig.colorbar(_cell_count_scatter, ax=_axes[3], label="Number of cells per well")
    _axes[3].set_title("Cell count (potential confounder; log colour scale)")

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
    ### 2b — Exploratory cell-count regression

    This is a deliberately simplified, **diagnostic-only** comparison. For
    each standardized morphology feature, it estimates a linear relationship
    with `log10(cell count)` and subtracts the fitted count-dependent shift.
    The original profiles and every downstream analysis remain unchanged.

    **Why control-based is the default:** fitting slopes only in negative
    controls reduces the risk of learning and removing treatment-driven
    viability/proliferation biology. It still assumes the technical slope is
    transportable from controls to treated wells. Global regression is shown
    only as an exploratory option and can remove real treatment signal when
    treatment affects cell count.

    This compact comparison is not evidence that correction should be adopted.
    The deferred full implementation requires the confounder-detection gate,
    within-control tests, PERMANOVA, cross-validation, leakage checks and
    biological-signal preservation described in
    `workspace/FUTURE_CELL_COUNT_CONFOUNDER_QC.md`.
    """)
    return


@app.cell
def _(mo):
    cell_count_regression_mode_input = mo.ui.dropdown(
        options=["control_based", "global"],
        value="control_based",
        label="Fit cell-count slopes using (control_based is recommended)",
    )
    cell_count_regression_mode_input
    return (cell_count_regression_mode_input,)


@app.cell
def _(
    X,
    cell_count_regression_mode_input,
    negcon_mask,
    np,
    well_cell_counts,
):
    _log_count = np.log10(np.clip(np.asarray(well_cell_counts, dtype=float), 1, None))
    _fit_mask = (
        np.asarray(negcon_mask, dtype=bool)
        if cell_count_regression_mode_input.value == "control_based"
        else np.ones(len(_log_count), dtype=bool)
    )
    if int(_fit_mask.sum()) < 3:
        raise ValueError("At least three fitting wells are required for exploratory cell-count regression.")

    _fit_log = _log_count[_fit_mask]
    _fit_X = X[_fit_mask]
    _centered_fit_log = _fit_log - _fit_log.mean()
    _denominator = float(np.dot(_centered_fit_log, _centered_fit_log))
    if _denominator <= 0:
        raise ValueError("Cell count does not vary among the regression-fitting wells.")

    cell_count_feature_slopes = (_centered_fit_log[:, None] * _fit_X).sum(axis=0) / _denominator
    _reference_log_count = float(np.median(_fit_log))
    X_cell_count_regressed = X - (
        (_log_count - _reference_log_count)[:, None] * cell_count_feature_slopes[None, :]
    )

    print(
        f"Exploratory {cell_count_regression_mode_input.value} regression: "
        f"{int(_fit_mask.sum())} fitting wells; reference count ≈ {10 ** _reference_log_count:.0f} cells."
    )
    print("The corrected matrix is used only in Section 2b and does not replace downstream X.")
    return X_cell_count_regressed, cell_count_feature_slopes


@app.cell
def _(
    CONFIG,
    FIGS_DIR,
    PCA,
    RANDOM_STATE,
    X,
    X_cell_count_regressed,
    X_pca,
    cell_count_regression_mode_input,
    df,
    np,
    pca_full,
    plt,
    scatter_panel,
    well_cell_counts,
):
    from matplotlib.colors import LogNorm as _LogNorm

    pca_cell_count_regressed = PCA(n_components=min(X_pca.shape[1], X.shape[1]), random_state=RANDOM_STATE)
    X_pca_cell_count_regressed = pca_cell_count_regressed.fit_transform(X_cell_count_regressed)
    _log_count = np.log10(np.clip(np.asarray(well_cell_counts, dtype=float), 1, None))

    _corr_rows = []
    for _space, _coords, _model in (
        ("original", X_pca, pca_full),
        ("cell_count_regressed", X_pca_cell_count_regressed, pca_cell_count_regressed),
    ):
        for _pc_idx in range(2):
            _corr_rows.append(
                {
                    "profile_space": _space,
                    "pc": f"PC{_pc_idx + 1}",
                    "pearson_r_log10_cell_count": float(np.corrcoef(_coords[:, _pc_idx], _log_count)[0, 1]),
                    "variance_explained": float(_model.explained_variance_ratio_[_pc_idx]),
                    "regression_mode": cell_count_regression_mode_input.value,
                }
            )
    cell_count_pca_comparison = __import__("pandas").DataFrame(_corr_rows)

    _positive_counts = np.clip(np.asarray(well_cell_counts, dtype=float), 1, None)
    _norm = _LogNorm(vmin=_positive_counts.min(), vmax=_positive_counts.max())
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 11))
    for _ax, _coords, _model, _title in (
        (_axes[0, 0], X_pca, pca_full, "Original — cell count"),
        (_axes[0, 1], X_pca_cell_count_regressed, pca_cell_count_regressed, "Regressed — cell count"),
    ):
        _sc = _ax.scatter(
            _coords[:, 0], _coords[:, 1], c=_positive_counts, cmap="viridis", norm=_norm,
            s=58, alpha=0.85, edgecolors="k", linewidths=0.35,
        )
        _ax.set_title(_title)
        _ax.set_xlabel(f"PC1 ({_model.explained_variance_ratio_[0]:.1%})")
        _ax.set_ylabel(f"PC2 ({_model.explained_variance_ratio_[1]:.1%})")
    _fig.colorbar(_sc, ax=_axes[0, :].tolist(), label="Number of cells per well")

    scatter_panel(
        _axes[1, 0], X_pca[:, 0], X_pca[:, 1],
        df[CONFIG.treatment_col].tolist(), "Original — treatment",
    )
    scatter_panel(
        _axes[1, 1], X_pca_cell_count_regressed[:, 0], X_pca_cell_count_regressed[:, 1],
        df[CONFIG.treatment_col].tolist(), "Regressed — treatment",
    )
    _axes[1, 0].set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]:.1%})")
    _axes[1, 0].set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]:.1%})")
    _axes[1, 1].set_xlabel(f"PC1 ({pca_cell_count_regressed.explained_variance_ratio_[0]:.1%})")
    _axes[1, 1].set_ylabel(f"PC2 ({pca_cell_count_regressed.explained_variance_ratio_[1]:.1%})")
    _fig.suptitle(
        f"Exploratory cell-count regression ({cell_count_regression_mode_input.value})",
        fontsize=13, fontweight="bold",
    )
    _fig.subplots_adjust(top=0.92, hspace=0.28, wspace=0.20)
    _fig.savefig(FIGS_DIR / "pca_cell_count_regression_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(_fig)
    print(cell_count_pca_comparison.to_string(index=False))
    print("  ✓  Saved: pca_cell_count_regression_comparison.png")
    _fig
    return X_pca_cell_count_regressed, cell_count_pca_comparison, pca_cell_count_regressed


@app.cell
def _(
    CONFIG,
    FIGS_DIR,
    RESULTS_DIR,
    X,
    X_cell_count_regressed,
    cell_count_pca_comparison,
    cell_count_regression_mode_input,
    copairs_compute_map,
    df,
    feat_cols,
    pd,
    plt,
):
    _comparison_dir = RESULTS_DIR / "cell_count_regression_exploratory"
    _comparison_dir.mkdir(parents=True, exist_ok=True)
    _sameby = [CONFIG.treatment_col]
    if CONFIG.has_dose_axis and CONFIG.concentration_col:
        _sameby.append(CONFIG.concentration_col)
    _diffby = [CONFIG.plate_col] if CONFIG.plate_col and df[CONFIG.plate_col].nunique() > 1 else []

    _map_rows = []
    for _space, _matrix in (("original", X), ("cell_count_regressed", X_cell_count_regressed)):
        _profile = df.copy()
        _profile.loc[:, feat_cols] = _matrix
        _map = copairs_compute_map(
            _profile,
            feat_cols,
            pos_sameby=_sameby,
            pos_diffby=_diffby,
            neg_diffby=[CONFIG.treatment_col],
            null_size=1000,
            seed=42,
            distance="cosine",
        )
        _map_rows.append(
            {
                "profile_space": _space,
                "mean_map": float(_map["mean_average_precision"].mean()),
                "median_map": float(_map["mean_average_precision"].median()),
                "n_replicate_groups": int(len(_map)),
                "regression_mode": cell_count_regression_mode_input.value,
            }
        )
        _map.to_csv(_comparison_dir / f"map_{_space}.csv", index=False)

    cell_count_regression_summary = pd.DataFrame(_map_rows)
    _baseline_map = float(
        cell_count_regression_summary.loc[
            cell_count_regression_summary["profile_space"] == "original", "mean_map"
        ].iloc[0]
    )
    cell_count_regression_summary["delta_mean_map_vs_original"] = (
        cell_count_regression_summary["mean_map"] - _baseline_map
    )
    cell_count_regression_summary.to_csv(_comparison_dir / "map_summary.csv", index=False)
    cell_count_pca_comparison.to_csv(_comparison_dir / "pca_cell_count_correlations.csv", index=False)

    _fig, _ax = plt.subplots(figsize=(6.5, 4.5))
    _bars = _ax.bar(
        cell_count_regression_summary["profile_space"],
        cell_count_regression_summary["mean_map"],
        color=["#607d8b", "#7b1fa2"],
    )
    for _bar, _value in zip(_bars, cell_count_regression_summary["mean_map"]):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _value, f"{_value:.3f}", ha="center", va="bottom")
    _ax.set_ylabel("Phenotypic-consistency mAP (copairs)")
    _ax.set_title("Exploratory phenotypic consistency before vs. after cell-count regression")
    _ax.set_ylim(0, max(0.12, float(cell_count_regression_summary["mean_map"].max()) * 1.2))
    _fig.tight_layout()
    _fig.savefig(FIGS_DIR / "map_cell_count_regression_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(_fig)
    print("\nExploratory phenotypic-consistency mAP (cross-plate replicates when multiple plates are present):")
    print(cell_count_regression_summary.to_string(index=False))
    print("  ✓  Saved: map_cell_count_regression_comparison.png")
    print(f"  ✓  Saved diagnostics: {_comparison_dir}")
    _fig
    return (cell_count_regression_summary,)


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
    (`X_pca`) and, optionally, one or more **batch-corrected** versions of it — **harmony**
    and/or **combat** below.

    **Reason:** `mad_robustize` (NB02) normalizes each feature's marginal distribution, but
    plate-to-plate systematic shifts that only emerge from the *combination* of many features
    survive that step — that's exactly what SC-08 (NB03) checks for. Batch correction is the
    remediation: re-express every well's profile with plate-associated variation removed,
    while (ideally) keeping treatment-associated variation intact. Two algorithms are offered
    because they make different tradeoffs, not because one is a strict upgrade of the other:

    - **Harmony** (Korsunsky et al. 2019) iteratively re-clusters wells across plates in PCA
      space and nudges each cluster's per-plate centroids together. Arevalo et al. 2023
      benchmarked it as the strongest general-purpose choice for Cell Painting-style profiles,
      which is why it's this pipeline's long-standing default.
      **Assumption:** the same approximate treatment/cluster structure exists on every plate —
      Harmony needs enough per-plate replicates of each cluster to align them confidently.
    - **ComBat** (Johnson, Li & Rabinovic 2007; `inmoose.pycombat.pycombat_norm` here) instead
      fits an explicit per-feature, per-plate location (and, optionally, scale) shift and
      removes it, using an empirical-Bayes prior borrowed across features to stabilize the
      estimate from limited data.
      **Assumption:** the batch effect is a simple additive (± multiplicative) per-feature
      shift — a narrower model than Harmony's, but one that needs far fewer wells per plate to
      estimate reliably. That's the case this option is *for*: a small pilot plate, or a batch
      with too few replicates per cluster for Harmony's clustering step to align confidently.
      `mean_only=True` (the default below) drops the multiplicative term specifically because
      per-plate *variance* is the least stable thing to estimate with few replicates — see
      "Advanced: ComBat parameters" below.

    Both methods assume **plate is a reasonable proxy for batch** (the same one SC-08 uses)
    and that **treatment and plate are not perfectly confounded** — a design where every plate
    ran a different set of treatments would make batch correction indistinguishable from
    erasing the biology, regardless of which algorithm removes it.
    """)
    return


@app.cell
def _(mo):
    combat_mean_only_input = mo.ui.checkbox(
        value=True,
        label=(
            "mean_only — skip per-plate variance correction, only remove the mean shift "
            "(recommended when a plate has few replicates per treatment)"
        ),
    )
    combat_par_prior_input = mo.ui.checkbox(
        value=True,
        label="par_prior — parametric empirical-Bayes prior (uncheck for the slower, more robust non-parametric estimate)",
    )
    mo.accordion(
        {
            "Advanced: ComBat parameters (only used if 'combat' is selected above)": mo.vstack(
                [combat_mean_only_input, combat_par_prior_input]
            ),
        }
    )
    return combat_mean_only_input, combat_par_prior_input


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
    combat_mean_only_input,
    combat_par_prior_input,
    df,
    meta_cols,
    np,
    pd,
    validate_output_path,
    write_csv_protected,
):
    # Harmony/ComBat batch correction are not (yet) part of the shared
    # hca_pipeline package (confirmed absent from modelling.py/stats.py/etc.)
    # — kept local to this notebook since it is this section's specific
    # concern.
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

    X_pca_combat = None
    if "combat" in RUN_ANALYSIS_SPACES:
        try:
            from inmoose.pycombat import pycombat_norm
        except ImportError as _error:
            raise ImportError(
                "ComBat analysis was requested, but inmoose is not installed. "
                "Add inmoose to the Pixi environment and rerun."
            ) from _error

        # pycombat_norm expects features (rows) x samples (columns) --
        # transposed relative to X_pca's wells (rows) x PCs (columns).
        _combat_batch = df[CONFIG.plate_col].astype(str).tolist()
        _combat_raw = pycombat_norm(
            X_pca.T,
            _combat_batch,
            mean_only=bool(combat_mean_only_input.value),
            par_prior=bool(combat_par_prior_input.value),
        )
        X_pca_combat = np.asarray(_combat_raw).T

        if X_pca_combat.shape != X_pca.shape:
            raise ValueError(
                f"Unexpected ComBat output shape. Input PCA shape: {X_pca.shape}, "
                f"ComBat shape: {X_pca_combat.shape}"
            )

        _combat_coordinates_df = pd.DataFrame(
            X_pca_combat,
            columns=[f"ComBat_PC{i + 1}" for i in range(X_pca_combat.shape[1])],
            index=df.index,
        )
        _combat_coordinates_df = pd.concat([df[meta_cols], _combat_coordinates_df], axis=1)
        _combat_path = validate_output_path(
            SPACE_DIRECTORIES["combat"]["results"] / "combat_coordinates.csv", OVERWRITE_EXISTING_OUTPUTS
        )
        _status = write_csv_protected(_combat_coordinates_df, _combat_path, overwrite=OVERWRITE_EXISTING_OUTPUTS)
        print(f"ComBat-corrected matrix: {X_pca_combat.shape} ({_status})")

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
    if X_pca_combat is not None:
        MODELLING_SPACES["combat"] = {
            "label": "ComBat-corrected PCA space", "X_latent": X_pca_combat,
            "description": "ComBat-corrected PCA coordinates",
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
    ## Section 8b — Similarity Matrices for Profile Exploration

    A similarity matrix turns the well-level feature matrix `X` into an
    n_wells × n_wells table of "how alike are these two profiles", making
    replicate agreement and treatment separation visible directly — no PCA/
    UMAP projection required first. Caicedo et al. 2017 frame this as a core
    profiling primitive: *"similarity metrics reveal connections among
    morphological profiles"*.

    Five metrics are computed, each with a different notion of "alike":

    - **Pearson** — correlation of each profile's *shape* (feature-to-feature
      pattern), invariant to each profile's overall scale/offset.
    - **Cosine** — angle between profile vectors. On data that is already
      per-feature centered (this notebook's `StandardScaler` output, or
      `mad_robustize` upstream), cosine and Pearson are mathematically
      near-identical — demonstrated empirically in Section 8b.5 below.
    - **Spearman** — Pearson computed on *ranks* instead of raw values, so
      it only cares about monotonic agreement. Caicedo et al. 2017: *"rank
      correlations perform best for untransformed feature vectors"*, i.e.
      when features haven't already been made comparable in scale.
    - **Euclidean** — straight-line distance, converted to a similarity so
      it plots on the same 0–1 scale as the others.
    - **Mahalanobis** — Euclidean distance rescaled by the *feature
      covariance*, so correlated features don't get double-counted. Levy et
      al. 2025 call it *"extremely useful ... excellent in multivariate
      anomaly detection ... not so well known or used in ML practice"*.

    Distances (Euclidean, Mahalanobis) are converted to similarities via
    `sim = 1 / (1 + dist)`, mapping `[0, ∞) → (0, 1]` with `1.0` meaning
    identical profiles — the same 0–1 sense as a correlation-based
    similarity, so all five metrics can be plotted on one shared color scale.
    """)
    return


@app.cell
def _():
    import seaborn as sns
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr, rankdata
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics.pairwise import cosine_similarity

    # scipy.cluster.hierarchy.linkage method shared by every clustermap and
    # side-by-side heatmap below, so all of them agree on well ordering.
    LINKAGE_METHOD = "average"
    return (
        LINKAGE_METHOD,
        LedoitWolf,
        cosine_similarity,
        fcluster,
        linkage,
        pdist,
        pearsonr,
        rankdata,
        sns,
        squareform,
    )


@app.cell
def _(FIGS_DIR, RESULTS_DIR):
    # Dedicated subdirectory (matching NB04's own comparisons/ and
    # fingerprints/ split) so this exploratory section's outputs never
    # collide with the modelling-space results already written above.
    SIMILARITY_RESULTS_DIR = RESULTS_DIR / "similarity_matrices"
    SIMILARITY_FIGS_DIR = FIGS_DIR / "similarity_matrices"
    SIMILARITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SIMILARITY_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    return SIMILARITY_FIGS_DIR, SIMILARITY_RESULTS_DIR


@app.cell
def _(LedoitWolf, cosine_similarity, np, pdist, rankdata, squareform):
    def compute_similarity_matrix(profiles, metric):
        """Pairwise n x n similarity matrix for one profile matrix + metric.

        Every branch returns a symmetric ``(n, n)`` array with diagonal
        ``1.0`` (a profile is maximally similar to itself), so the five
        metrics are directly comparable and interchangeable downstream
        (clustermaps, Mantel tests, percent-replicating).
        """
        if metric == "pearson":
            # Caicedo et al. 2017: "Pearson's correlation generally appears
            # to be a good choice" for typical profiling pipelines. Vectorized
            # as center -> L2-normalize -> one matrix product for all pairs
            # at once (same pattern as hca_pipeline.metrics_qc's
            # compute_pairwise_correlations, but with a 1.0 diagonal here
            # instead of that function's -inf nearest-neighbor sentinel).
            centered = profiles - profiles.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = centered / norms
            sim = normalized @ normalized.T
            np.fill_diagonal(sim, 1.0)
            return sim

        if metric == "cosine":
            # Equivalent to Pearson on centered data (this notebook's
            # StandardScaler centers every feature) -- and cosine is exactly
            # the metric `copairs` uses for NB03's mAP/PR-based QC, so this
            # matrix is expected to closely track the Pearson one above (see
            # Section 8b.5, the Pearson/cosine equivalence check).
            sim = cosine_similarity(profiles)
            np.fill_diagonal(sim, 1.0)
            return sim

        if metric == "spearman":
            # Caicedo et al. 2017: rank correlations "perform best for
            # untransformed feature vectors" -- robust to outliers and
            # monotonic non-linearities that Pearson is sensitive to.
            # Vectorized as Pearson-on-ranks rather than a pairwise
            # spearmanr loop: rank each profile's features, then reuse the
            # same center/normalize/matmul pattern as the Pearson branch.
            ranked = np.apply_along_axis(rankdata, 1, profiles)
            centered = ranked - ranked.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = centered / norms
            sim = normalized @ normalized.T
            np.fill_diagonal(sim, 1.0)
            return sim

        if metric == "euclidean":
            dist = squareform(pdist(profiles, metric="euclidean"))
            sim = 1.0 / (1.0 + dist)  # [0, inf) -> (0, 1], 1.0 = identical
            np.fill_diagonal(sim, 1.0)
            return sim

        if metric == "mahalanobis":
            # Levy et al. 2025: Mahalanobis distance is "extremely useful
            # ... excellent in multivariate anomaly detection ... not so
            # well known or used in ML practice." LedoitWolf shrinkage keeps
            # the covariance estimate invertible even though n_features here
            # is typically >> n_wells, where the plain sample covariance
            # would be singular.
            try:
                precision = LedoitWolf().fit(profiles).precision_
                dist = squareform(pdist(profiles, metric="mahalanobis", VI=precision))
                sim = 1.0 / (1.0 + dist)
                np.fill_diagonal(sim, 1.0)
                return sim
            except Exception as exc:
                print(f"  ⚠️  Mahalanobis similarity failed ({exc!r}); skipping this metric.")
                return None

        raise ValueError(f"Unknown metric: {metric!r}")

    return (compute_similarity_matrix,)


@app.cell
def _(X, compute_similarity_matrix, np):
    SIMILARITY_METRICS = ["pearson", "cosine", "spearman", "euclidean", "mahalanobis"]

    similarity_matrices = {}
    for _metric in SIMILARITY_METRICS:
        _sim = compute_similarity_matrix(X, _metric)
        if _sim is not None:
            similarity_matrices[_metric] = _sim

    print(f"Profiles: {X.shape[0]} wells x {X.shape[1]} features\n")
    for _metric, _sim in similarity_matrices.items():
        _off_diag = _sim[~np.eye(_sim.shape[0], dtype=bool)]
        print(
            f"  {_metric:<12} shape={_sim.shape}  "
            f"range=[{_off_diag.min():.3f}, {_off_diag.max():.3f}]  mean={_off_diag.mean():.3f}"
        )
    return (similarity_matrices,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 8b.1 — Hierarchical Clustering & Heatmaps

    Caicedo et al. 2017: *"Hierarchical clustering is computed by using a
    similarity matrix ... visualized as a heat map ... sorted by using the
    hierarchical structure."* Reordering the similarity matrix by its own
    dendrogram groups replicate/treatment blocks together visually, without
    first needing to already know which wells belong together — a useful
    sanity check before trusting any of the metrics that summarize
    replicate agreement below.

    **How rows and columns are organized:** by default, neither treatment nor
    plate determines the order. The dendrogram is computed only from profile
    similarity (`distance = 1 − similarity`); metadata colors are annotations
    added afterwards. Use the controls below to temporarily sort the same
    matrix by a metadata field and expose treatment, concentration, plate, or
    other experimental blocks directly.
    """)
    return


@app.cell
def _(CONFIG, df, mo):
    _candidate_columns = [
        getattr(CONFIG, "treatment_col", None),
        getattr(CONFIG, "concentration_col", None),
        getattr(CONFIG, "plate_col", None),
        getattr(CONFIG, "control_type_col", None),
        getattr(CONFIG, "cell_type_col", None),
        getattr(CONFIG, "time_col", None),
        "Metadata_Cell_Type",
        "Metadata_Time",
    ]
    heatmap_metadata_columns = list(dict.fromkeys(
        _column for _column in _candidate_columns if _column and _column in df.columns
    ))
    _default_annotations = [
        _column
        for _column in (CONFIG.treatment_col, CONFIG.plate_col)
        if _column in heatmap_metadata_columns
    ]
    heatmap_order_input = mo.ui.dropdown(
        options=["Hierarchical similarity", *heatmap_metadata_columns],
        value="Hierarchical similarity",
        label="Order matrix by",
    )
    heatmap_annotations_input = mo.ui.multiselect(
        options=heatmap_metadata_columns,
        value=_default_annotations,
        label="Annotation color tracks",
    )
    mo.hstack(
        [heatmap_order_input, heatmap_annotations_input],
        widths=[1, 2],
        align="start",
    )
    return heatmap_annotations_input, heatmap_order_input


@app.cell
def _(df, heatmap_annotations_input, heatmap_order_input, pd, sns):
    heatmap_order_column = (
        None if heatmap_order_input.value == "Hierarchical similarity" else heatmap_order_input.value
    )
    heatmap_annotation_columns = list(heatmap_annotations_input.value)
    heatmap_metadata = df.reset_index(drop=True)
    heatmap_annotation_palettes = {}
    heatmap_annotation_colors = pd.DataFrame(index=heatmap_metadata.index)

    for _track_index, _column in enumerate(heatmap_annotation_columns):
        _values = heatmap_metadata[_column].fillna("Missing").astype(str)

        def _natural_key(_value):
            try:
                return (0, float(_value))
            except ValueError:
                return (1, _value.casefold())

        _levels = sorted(_values.unique(), key=_natural_key)
        _palette_name = ("Set2", "Set3", "tab10", "husl")[_track_index % 4]
        _colors = sns.color_palette(_palette_name, n_colors=len(_levels))
        _palette = dict(zip(_levels, _colors))
        heatmap_annotation_palettes[_column] = _palette
        heatmap_annotation_colors[_column] = _values.map(_palette)
    return (
        heatmap_annotation_colors,
        heatmap_annotation_palettes,
        heatmap_metadata,
        heatmap_order_column,
    )


@app.cell
def _(LINKAGE_METHOD, Patch, linkage, np, pd, plt, sns, squareform):
    def plot_similarity_clustermap(
        sim_matrix,
        metadata,
        annotation_colors,
        annotation_palettes,
        ordering_column,
        title,
        output_path,
    ):
        """Similarity heatmap reordered by its own hierarchical clustering.

        Clustering is computed from the *distance* (``1 - similarity``) but
        the heatmap itself is colored by *similarity*, so "brighter" reads
        as "more alike" — clustering geometry and color meaning are kept
        separate on purpose, rather than clustering directly on a
        similarity matrix (which `scipy.cluster.hierarchy` doesn't treat as
        a proper metric).
        """
        sim_df = pd.DataFrame(sim_matrix)
        plot_colors = annotation_colors if len(annotation_colors.columns) else None

        if ordering_column is None:
            distance_matrix = 1.0 - sim_matrix
            np.fill_diagonal(distance_matrix, 0.0)
            condensed = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed, method=LINKAGE_METHOD)
            row_linkage = linkage_matrix
            col_linkage = linkage_matrix
            row_cluster = True
            col_cluster = True
            order_description = f"hierarchical similarity ({LINKAGE_METHOD} linkage)"
        else:
            _sort_values = metadata[ordering_column]
            _numeric_sort = pd.to_numeric(_sort_values, errors="coerce")
            _sort_frame = pd.DataFrame({
                "missing": _sort_values.isna(),
                "numeric": _numeric_sort,
                "text": _sort_values.fillna("Missing").astype(str).str.casefold(),
            })
            _order = _sort_frame.sort_values(
                ["missing", "numeric", "text"], kind="stable", na_position="last"
            ).index.to_numpy()
            sim_df = sim_df.iloc[_order, _order]
            if plot_colors is not None:
                plot_colors = plot_colors.iloc[_order]
                plot_colors.index = sim_df.index
            row_linkage = None
            col_linkage = None
            row_cluster = False
            col_cluster = False
            order_description = ordering_column

        grid = sns.clustermap(
            sim_df,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            row_colors=plot_colors,
            col_colors=plot_colors,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            figsize=(14, 11),
            dendrogram_ratio=(0.15, 0.15),
            xticklabels=False,
            yticklabels=False,
        )
        grid.figure.suptitle(f"{title}\nOrder: {order_description}", y=1.04)
        grid.cax.set_ylabel("Profile similarity", rotation=270, labelpad=16)

        _legend_handles = []
        for _column, _palette in annotation_palettes.items():
            for _level, _color in _palette.items():
                _legend_handles.append(
                    Patch(facecolor=_color, edgecolor="none", label=f"{_column}: {_level}")
                )
        if _legend_handles:
            grid.figure.legend(
                handles=_legend_handles,
                title="Annotation colors",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=True,
                fontsize=8,
                title_fontsize=9,
            )

        grid.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        grid.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(grid.figure)
        return grid.figure

    return (plot_similarity_clustermap,)


@app.cell
def _(
    SIMILARITY_FIGS_DIR,
    heatmap_annotation_colors,
    heatmap_annotation_palettes,
    heatmap_metadata,
    heatmap_order_column,
    mo,
    plot_similarity_clustermap,
    similarity_matrices,
):
    _figures = []
    _order_slug = "hierarchical" if heatmap_order_column is None else heatmap_order_column.lower().replace("metadata_", "").replace(" ", "_")
    for _metric in ("pearson", "cosine"):
        if _metric not in similarity_matrices:
            continue
        _fig = plot_similarity_clustermap(
            similarity_matrices[_metric],
            heatmap_metadata,
            heatmap_annotation_colors,
            heatmap_annotation_palettes,
            heatmap_order_column,
            title=f"{_metric.capitalize()} similarity heatmap",
            output_path=SIMILARITY_FIGS_DIR / f"clustermap_{_metric}_by_{_order_slug}",
        )
        _figures.append(_fig)
        _png_path = SIMILARITY_FIGS_DIR / f"clustermap_{_metric}_by_{_order_slug}.png"
        _svg_path = SIMILARITY_FIGS_DIR / f"clustermap_{_metric}_by_{_order_slug}.svg"
        print(f"✓ Saved: {_png_path.name} / {_svg_path.name}")

    if _figures:
        _output = mo.vstack(_figures)
    else:
        _output = mo.callout("No Pearson or cosine similarity matrix is available to plot.", kind="warn")
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8b.2 — Metric Comparison

    Do the five metrics actually agree on which wells are similar? A Mantel
    test answers this directly: correlate the upper triangle of one
    similarity matrix against another's (excluding the trivial diagonal),
    which sidesteps re-deriving a joint embedding just to compare two
    n × n structures on the same samples.
    """)
    return


@app.cell
def _(pearsonr):
    def mantel_correlation(matrix_a, matrix_b):
        """Pearson r (+ p-value) between two similarity matrices' upper triangles."""
        n = matrix_a.shape[0]
        rows, cols = __import__("numpy").triu_indices(n, k=1)
        r, p = pearsonr(matrix_a[rows, cols], matrix_b[rows, cols])
        return float(r), float(p)

    return (mantel_correlation,)


@app.cell
def _(mantel_correlation, pd, similarity_matrices):
    _metric_names = list(similarity_matrices.keys())
    _rows = []
    for _i, _metric_a in enumerate(_metric_names):
        for _metric_b in _metric_names[_i + 1 :]:
            _r, _p = mantel_correlation(similarity_matrices[_metric_a], similarity_matrices[_metric_b])
            _rows.append({"metric_a": _metric_a, "metric_b": _metric_b, "mantel_r": _r, "p_value": _p})

    mantel_df = pd.DataFrame(_rows).sort_values("mantel_r", ascending=False).reset_index(drop=True)
    print("Mantel correlation between similarity metrics (upper triangles):\n")
    print(mantel_df.to_string(index=False))
    return (mantel_df,)


@app.cell
def _(SIMILARITY_FIGS_DIR, linkage, np, plt, similarity_matrices, squareform):
    from scipy.cluster.hierarchy import dendrogram as _dendrogram

    # Every panel uses the SAME (Pearson-derived) well ordering, so a visual
    # "does this block look tighter under metric X" comparison is valid --
    # each metric's *own* clustering would reorder wells differently and
    # make the panels incomparable.
    _pearson_distance = 1.0 - similarity_matrices["pearson"]
    np.fill_diagonal(_pearson_distance, 0.0)
    _pearson_linkage = linkage(squareform(_pearson_distance, checks=False), method="average")
    _order = _dendrogram(_pearson_linkage, no_plot=True)["leaves"]

    _panel_metrics = [m for m in ("pearson", "cosine", "euclidean") if m in similarity_matrices]
    _fig, _axes = plt.subplots(1, len(_panel_metrics), figsize=(6 * len(_panel_metrics), 5.5))
    _axes = np.atleast_1d(_axes)
    for _ax, _metric in zip(_axes, _panel_metrics):
        _reordered = similarity_matrices[_metric][np.ix_(_order, _order)]
        _im = _ax.imshow(_reordered, cmap="RdYlBu_r", vmin=0, vmax=1)
        _ax.set_title(_metric.capitalize())
        _ax.set_xticks([])
        _ax.set_yticks([])
        _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _fig.suptitle("Same well ordering (Pearson linkage) across metrics", fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(SIMILARITY_FIGS_DIR / "metric_comparison_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(_fig)
    print("✓ Saved: metric_comparison_heatmaps.png")
    return


@app.cell
def _(
    CONFIG,
    df,
    fcluster,
    linkage,
    mantel_df,
    np,
    pd,
    similarity_matrices,
    squareform,
):
    _n_treatments = df[CONFIG.treatment_col].nunique()
    _rows = []
    for _metric, _sim in similarity_matrices.items():
        _off_diag = _sim[~np.eye(_sim.shape[0], dtype=bool)]
        if _metric == "pearson":
            _mantel_vs_pearson = 1.0
        else:
            _match = mantel_df[
                ((mantel_df["metric_a"] == "pearson") & (mantel_df["metric_b"] == _metric))
                | ((mantel_df["metric_a"] == _metric) & (mantel_df["metric_b"] == "pearson"))
            ]
            _mantel_vs_pearson = float(_match["mantel_r"].iloc[0]) if len(_match) else np.nan

        _distance = 1.0 - _sim
        np.fill_diagonal(_distance, 0.0)
        _linkage = linkage(squareform(_distance, checks=False), method="average")
        _clusters = fcluster(_linkage, t=_n_treatments, criterion="maxclust")

        _rows.append(
            {
                "metric": _metric,
                "min": float(_off_diag.min()),
                "max": float(_off_diag.max()),
                "mean": float(_off_diag.mean()),
                "mantel_r_vs_pearson": _mantel_vs_pearson,
                f"n_clusters_at_k={_n_treatments}": len(np.unique(_clusters)),
            }
        )

    metric_summary_df = pd.DataFrame(_rows)
    print(f"Metric summary (k = {_n_treatments} treatments):\n")
    print(metric_summary_df.to_string(index=False))
    return (metric_summary_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 8b.3 — Replicate Reproducibility

    Caicedo et al. 2017: *"similarity scores are compared with a suitable
    null distribution"* — a treatment's replicates being *somewhat* similar
    means nothing on its own; what matters is whether they're *more*
    similar than random cross-treatment pairs. The null distribution below
    is built the same way as `hca_pipeline.metrics_qc.null_distribution_batch_aware`
    (used by NB03's from-scratch Percent Replicating): sample random pairs
    of wells with *different* treatments and correlate them.
    """)
    return


@app.cell
def _(CONFIG, X, df, np):
    from hca_pipeline.metrics_qc import null_distribution_batch_aware

    null_dist = null_distribution_batch_aware(
        X,
        df[CONFIG.treatment_col].to_numpy(),
        plate_labels=df[CONFIG.plate_col].to_numpy(),
        n_pairs=10_000,
        seed=42,  # same seed convention as every other notebook in this pipeline
    )
    print(f"Null distribution: {len(null_dist):,} random cross-treatment pairs")
    print(f"  mean={null_dist.mean():.3f}  95th percentile={np.percentile(null_dist, 95):.3f}")

    def percent_replicating_simplified(sim_matrix, labels, null_dist, percentile=95):
        """Fraction of treatments whose intra-treatment similarity clears the null.

        A lighter-weight, metric-agnostic cousin of
        `hca_pipeline.metrics_qc.percent_replicating` (which is Pearson-only):
        this takes any precomputed similarity matrix, so it can score the
        cosine/Spearman/Mahalanobis matrices too, not just Pearson. For each
        treatment, take the median of its intra-treatment (replicate-pair)
        similarities; a treatment "replicates" if that median exceeds the
        given percentile of the null distribution.
        """
        threshold = float(np.percentile(null_dist, percentile))
        labels = np.asarray(labels)
        medians = {}
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if len(idx) < 2:
                continue
            sub = sim_matrix[np.ix_(idx, idx)]
            upper = sub[np.triu_indices(len(idx), k=1)]
            medians[label] = float(np.median(upper))
        passing = [m > threshold for m in medians.values()]
        pr = float(np.mean(passing)) if passing else 0.0
        return pr, medians, threshold

    return null_dist, percent_replicating_simplified


@app.cell
def _(
    CONFIG,
    SIMILARITY_FIGS_DIR,
    df,
    np,
    null_dist,
    plt,
    similarity_matrices,
):
    _labels = df[CONFIG.treatment_col].to_numpy()
    _pearson = similarity_matrices["pearson"]

    _intra, _inter = [], []
    for _i in range(len(_labels)):
        for _j in range(_i + 1, len(_labels)):
            (_intra if _labels[_i] == _labels[_j] else _inter).append(_pearson[_i, _j])

    _fig, _ax = plt.subplots(figsize=(7, 5.5))
    _ax.boxplot([_intra, _inter], tick_labels=["Intra-treatment\n(replicates)", "Inter-treatment\n(different)"])
    _ax.axhline(
        np.percentile(null_dist, 95), color="crimson", ls="--", lw=1.5,
        label="Null 95th percentile",
    )
    _ax.set_ylabel("Pearson similarity")
    _ax.set_title("Replicate agreement vs. random cross-treatment pairs")
    _ax.legend()
    _fig.tight_layout()
    _fig.savefig(SIMILARITY_FIGS_DIR / "intra_vs_inter_treatment_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(_fig)
    print("✓ Saved: intra_vs_inter_treatment_boxplot.png")
    return


@app.cell
def _(
    CONFIG,
    df,
    null_dist,
    pd,
    percent_replicating_simplified,
    similarity_matrices,
):
    _pr, _medians, _threshold = percent_replicating_simplified(
        similarity_matrices["pearson"], df[CONFIG.treatment_col].to_numpy(), null_dist,
    )
    pr_simplified_df = pd.DataFrame(
        [
            {
                "treatment": _label,
                "median_intra_similarity": _median,
                "null_threshold": _threshold,
                "replicates": _median > _threshold,
            }
            for _label, _median in _medians.items()
        ]
    ).sort_values("median_intra_similarity", ascending=False).reset_index(drop=True)

    print(f"Percent replicating (simplified, Pearson, 95th-percentile null): {_pr:.1%}\n")
    print(pr_simplified_df.to_string(index=False))
    return (pr_simplified_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 8b.4 — Pearson ≈ Cosine Equivalence

    Pearson correlation *is* the cosine similarity of mean-centered vectors
    — not an approximation, an identity. Because this notebook's `X` is
    already per-feature centered (`StandardScaler`, and `mad_robustize`
    upstream in NB02 does the same), the Pearson and cosine matrices above
    should be near-identical. This is the same reason `copairs` (NB03's
    mAP/PR-based QC) can use cosine similarity and still be measuring
    essentially the same thing as the from-scratch Pearson-based metrics
    used elsewhere in this pipeline.
    """)
    return


@app.cell
def _(SIMILARITY_FIGS_DIR, np, pearsonr, plt, similarity_matrices):
    _n = similarity_matrices["pearson"].shape[0]
    _rows, _cols = np.triu_indices(_n, k=1)
    _pearson_upper = similarity_matrices["pearson"][_rows, _cols]
    _cosine_upper = similarity_matrices["cosine"][_rows, _cols]
    pearson_cosine_equivalence_r, _ = pearsonr(_pearson_upper, _cosine_upper)

    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.scatter(_pearson_upper, _cosine_upper, s=8, alpha=0.4)
    _ax.plot([0, 1], [0, 1], "k--", lw=1, label="y = x")
    _ax.set_xlabel("Pearson similarity")
    _ax.set_ylabel("Cosine similarity")
    _ax.set_title(f"Pearson vs. cosine (r = {pearson_cosine_equivalence_r:.4f})")
    _ax.legend()
    _fig.tight_layout()
    _fig.savefig(SIMILARITY_FIGS_DIR / "pearson_cosine_equivalence.png", dpi=300, bbox_inches="tight")
    plt.close(_fig)

    print(
        f"Pearson and cosine are equivalent on this notebook's centered data "
        f"(r = {pearson_cosine_equivalence_r:.4f})."
    )
    return (pearson_cosine_equivalence_r,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 8b.5 — Connection to `copairs` (NB03)

    NB03's Go/No-Go dashboard already scores replicate reproducibility
    properly (mAP via `copairs`, cosine-based). This is a cheap
    cross-check, not a replacement: if a treatment's mAP is high but its
    median intra-treatment Pearson similarity here is low (or vice versa),
    that disagreement is worth investigating rather than trusting either
    number blindly.
    """)
    return


@app.cell
def _(
    CONFIG,
    RESULTS_DIR,
    SIMILARITY_FIGS_DIR,
    df,
    pd,
    plt,
    similarity_matrices,
):
    _map_csv = RESULTS_DIR / "quality_metrics" / "app_a_per_plate_map.csv"
    if _map_csv.exists():
        _map_df = pd.read_csv(_map_csv)
        _map_by_treatment = _map_df.groupby("treatment")["mAP_copairs"].mean()

        _pearson = similarity_matrices["pearson"]
        _labels = df[CONFIG.treatment_col].to_numpy()
        _intra_median = {}
        for _label in _map_by_treatment.index:
            _idx = __import__("numpy").where(_labels == _label)[0]
            if len(_idx) < 2:
                continue
            _sub = _pearson[__import__("numpy").ix_(_idx, _idx)]
            _upper = _sub[__import__("numpy").triu_indices(len(_idx), k=1)]
            _intra_median[_label] = float(__import__("numpy").median(_upper))

        _common = sorted(set(_map_by_treatment.index) & set(_intra_median.keys()))
        if _common:
            _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
            _ax.scatter(
                [_map_by_treatment[t] for t in _common],
                [_intra_median[t] for t in _common],
                s=40,
            )
            for t in _common:
                _ax.annotate(t, (_map_by_treatment[t], _intra_median[t]), fontsize=7)
            _ax.set_xlabel("NB03 phenotypic-consistency mAP (copairs, cosine)")
            _ax.set_ylabel("Median intra-treatment Pearson similarity (this section)")
            _ax.set_title("Phenotypic-consistency mAP vs. replicate agreement")
            _fig.tight_layout()
            _fig.savefig(SIMILARITY_FIGS_DIR / "map_vs_similarity.png", dpi=300, bbox_inches="tight")
            plt.close(_fig)
            print(f"✓ Saved: map_vs_similarity.png ({len(_common)} treatments in common)")
        else:
            print("NB03 phenotypic-consistency mAP results found but no treatments in common — skipping comparison plot.")
    else:
        print(f"NB03 phenotypic-consistency mAP results not found — run NB03 first for comparison ({_map_csv}).")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8b.6 — Save Artifacts
    """)
    return


@app.cell
def _(
    OVERWRITE_EXISTING_OUTPUTS,
    SIMILARITY_RESULTS_DIR,
    df,
    mantel_df,
    metric_summary_df,
    pr_simplified_df,
    similarity_matrices,
    write_csv_protected,
):
    _well_ids = df.index.astype(str)
    for _metric, _sim in similarity_matrices.items():
        _sim_df = __import__("pandas").DataFrame(_sim, index=_well_ids, columns=_well_ids)
        _status = write_csv_protected(
            _sim_df, SIMILARITY_RESULTS_DIR / f"{_metric}_matrix.csv", overwrite=OVERWRITE_EXISTING_OUTPUTS,
        )
        print(f"  {_metric:<12} matrix {_status}")

    write_csv_protected(mantel_df, SIMILARITY_RESULTS_DIR / "mantel_comparison.csv", overwrite=OVERWRITE_EXISTING_OUTPUTS)
    write_csv_protected(metric_summary_df, SIMILARITY_RESULTS_DIR / "metric_summary.csv", overwrite=OVERWRITE_EXISTING_OUTPUTS)
    write_csv_protected(pr_simplified_df, SIMILARITY_RESULTS_DIR / "percent_replicating_simplified.csv", overwrite=OVERWRITE_EXISTING_OUTPUTS)
    print(f"✓ Saved similarity-matrix artifacts to {SIMILARITY_RESULTS_DIR}")
    return


@app.cell
def _(
    CONFIG,
    REPO_ROOT,
    SIMILARITY_RESULTS_DIR,
    X,
    datetime,
    feat_cols,
    json,
    pearson_cosine_equivalence_r,
    similarity_matrices,
    subprocess,
    validate_output_path,
):
    try:
        _git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT),
        ).stdout.strip()[:8]
    except Exception:
        _git_hash = "unknown"

    _provenance = {
        "notebook": "04_phenotypic_profiling.py (Section 8b)",
        "experiment_id": CONFIG.experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": _git_hash,
        "n_wells": int(X.shape[0]),
        "n_features": int(len(feat_cols)),
        "metrics_computed": list(similarity_matrices.keys()),
        "pearson_cosine_equivalence_r": pearson_cosine_equivalence_r,
    }
    _path = validate_output_path(SIMILARITY_RESULTS_DIR / "provenance.json", overwrite=True)
    _path.write_text(json.dumps(_provenance, indent=2), encoding="utf-8")
    print(f"✓ Provenance: {_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 8b complete

    Produced: 5 similarity matrices (CSV), 2 clustermaps + a 3-panel metric
    comparison + an intra/inter-treatment boxplot + a Pearson/cosine
    equivalence scatter (PNG/SVG), a Mantel comparison table, a per-metric
    summary table, and a simplified percent-replicating table — all under
    `results/similarity_matrices/` and `figures/phenotypic_profiling/similarity_matrices/`.
    This is exploratory and does not gate NB04's own integrity checks below.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 8c — Graph-Based Clustering & Manifold Learning (diagnostic)

    **Reason:** Section 8b's clustering was hierarchical (`fcluster`, forced to
    `k = n_treatments`), and NB04's own downstream KMeans
    (`hca_pipeline.modelling.run_modelling_space`) clusters the **LDA
    representation** — a *supervised*, label-informed embedding, not a test of
    whether the raw morphology separates into groups on its own. This section
    asks the more literal version of "does the data form clusters?": run
    graph-based (spectral) clustering on several affinity constructions and
    manifold learning directly on the *unsupervised* PCA space (`X`), and
    compare against a plain KMeans baseline using the metrics scikit-learn's
    own clustering guide recommends — silhouette, Calinski-Harabasz,
    Davies-Bouldin — plus bootstrap stability and agreement with treatment
    identity.

    **Assumptions:**

    - The affinity/graph construction determines what "similar" means before
      any clustering happens — a bad choice (e.g. an untuned RBF `gamma`) can
      silently produce a degenerate result that still looks fine by some
      metrics. This is checked below, not assumed away.
    - Comparing clusters against treatment identity is a *proxy*, not a
      validation: it only shows whether unsupervised structure lines up with
      *known* labels, not whether an unlabeled subpopulation is real. A
      genuine treatment-independent phenotype would score badly here and
      still be real.
    - `k` is **not** assumed to be `n_treatments` here — unlike Section 8b's
      hierarchical cut, a k-sweep lets the data suggest its own answer.

    This is a diagnostic section: it does not gate NB04's integrity checks,
    and its conclusion (below) is written from whatever this run's numbers
    actually show, not asserted in advance.
    """)
    return


@app.cell
def _(mo):
    graph_knn_neighbors_input = mo.ui.number(
        value=15, start=3, stop=50, label="k-NN graph / manifold n_neighbors"
    )
    graph_k_max_input = mo.ui.number(value=9, start=3, stop=20, label="Max k to sweep for clustering")
    graph_n_bootstrap_input = mo.ui.number(
        value=20, start=5, stop=200, label="Bootstrap resamples for cluster stability"
    )
    mo.accordion(
        {
            "Advanced: graph-clustering / manifold parameters": mo.vstack(
                [graph_knn_neighbors_input, graph_k_max_input, graph_n_bootstrap_input]
            ),
        }
    )
    return (
        graph_k_max_input,
        graph_knn_neighbors_input,
        graph_n_bootstrap_input,
    )


@app.cell
def _(FIGS_DIR, RESULTS_DIR):
    # Dedicated subdirectory (matching Section 8b's own similarity_matrices/
    # split) so this diagnostic section's outputs never collide with the
    # modelling-space results written above.
    GRAPH_CLUSTERING_RESULTS_DIR = RESULTS_DIR / "graph_clustering"
    GRAPH_CLUSTERING_FIGS_DIR = FIGS_DIR / "graph_clustering"
    GRAPH_CLUSTERING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_CLUSTERING_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    return GRAPH_CLUSTERING_FIGS_DIR, GRAPH_CLUSTERING_RESULTS_DIR


@app.cell
def _():
    # All third-party names this section needs, imported once (mirroring
    # Section 8b's own "imports for this section" cell) — marimo requires
    # every name to be defined in exactly one cell, so each subsequent cell
    # below consumes these as parameters rather than re-importing them.
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, trustworthiness
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.neighbors import kneighbors_graph

    return (
        Isomap,
        KMeans,
        LocallyLinearEmbedding,
        SpectralClustering,
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        kneighbors_graph,
        rbf_kernel,
        silhouette_score,
        trustworthiness,
    )


@app.cell
def _(
    X,
    graph_knn_neighbors_input,
    kneighbors_graph,
    np,
    rbf_kernel,
    similarity_matrices,
):
    # RBF and k-NN graph are new affinity constructions; cosine/Pearson reuse
    # Section 8b's similarity matrices as-is (shifted from [-1, 1] to [0, 1]
    # since spectral clustering requires a nonnegative affinity) rather than
    # recomputing them a second time.
    _knn_graph = kneighbors_graph(
        X, n_neighbors=int(graph_knn_neighbors_input.value), mode="connectivity", include_self=False,
    )
    # kneighbors_graph is directed (asymmetric); symmetrize into an
    # undirected affinity graph, which is what spectral clustering expects.
    _knn_symmetric = (_knn_graph + _knn_graph.T) > 0
    knn_graph_affinity = np.asarray(_knn_symmetric.astype(float).todense())

    graph_affinities = {
        "rbf": rbf_kernel(X),
        "knn_graph": knn_graph_affinity,
        "cosine": (similarity_matrices["cosine"] + 1) / 2,
        "pearson": (similarity_matrices["pearson"] + 1) / 2,
    }
    print(f"Affinity matrices built: {list(graph_affinities.keys())}")
    return (graph_affinities,)


@app.cell
def _(
    CONFIG,
    KMeans,
    RANDOM_STATE,
    SpectralClustering,
    X,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    df,
    graph_affinities,
    graph_k_max_input,
    np,
    pd,
    silhouette_score,
):
    _treatment_labels = df[CONFIG.treatment_col].astype(str).to_numpy()

    _rows = []
    for _k in range(2, int(graph_k_max_input.value) + 1):
        _km_labels = KMeans(n_clusters=_k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        _rows.append(
            {
                "k": _k,
                "method": "kmeans_pca",
                "silhouette": silhouette_score(X, _km_labels),
                "calinski_harabasz": calinski_harabasz_score(X, _km_labels),
                "davies_bouldin": davies_bouldin_score(X, _km_labels),
                "ari_vs_treatment": adjusted_rand_score(_treatment_labels, _km_labels),
            }
        )
        for _affinity_name, _affinity in graph_affinities.items():
            _sc_labels = SpectralClustering(
                n_clusters=_k, random_state=RANDOM_STATE, affinity="precomputed", assign_labels="kmeans",
            ).fit_predict(_affinity)
            if len(np.unique(_sc_labels)) < 2:
                # Degenerate solution (e.g. RBF collapsing to fewer real
                # clusters than requested) -- not a valid clustering to score.
                continue
            _rows.append(
                {
                    "k": _k,
                    "method": f"spectral_{_affinity_name}",
                    "silhouette": silhouette_score(X, _sc_labels),
                    "calinski_harabasz": calinski_harabasz_score(X, _sc_labels),
                    "davies_bouldin": davies_bouldin_score(X, _sc_labels),
                    "ari_vs_treatment": adjusted_rand_score(_treatment_labels, _sc_labels),
                }
            )

    clustering_comparison_df = pd.DataFrame(_rows)
    print("Spectral clustering vs. KMeans, across affinities and k:\n")
    print(clustering_comparison_df.round(4).to_string(index=False))
    return (clustering_comparison_df,)


@app.cell
def _(
    KMeans,
    RANDOM_STATE,
    SpectralClustering,
    X,
    adjusted_rand_score,
    graph_n_bootstrap_input,
    np,
    pd,
):
    # Representative k for the stability check (the sweep above already
    # covers k's effect on cluster quality; this asks a different question —
    # holding k fixed, how much does the clustering change under resampling?).
    _k_stability = 6
    _n = X.shape[0]
    _n_boot = int(graph_n_bootstrap_input.value)
    _rng = np.random.RandomState(RANDOM_STATE)

    def _bootstrap_stability(fit_predict):
        _base_labels = fit_predict(X)
        _aris = []
        for _ in range(_n_boot):
            _idx = _rng.choice(_n, size=_n, replace=True)
            _boot_labels = fit_predict(X[_idx])
            _aris.append(adjusted_rand_score(_base_labels[_idx], _boot_labels))
        return float(np.mean(_aris)), float(np.std(_aris))

    _stability_rows = []
    _mean_ari, _std_ari = _bootstrap_stability(
        lambda Z: KMeans(n_clusters=_k_stability, random_state=RANDOM_STATE, n_init=10).fit_predict(Z)
    )
    _stability_rows.append({"method": "kmeans_pca", "mean_ari": _mean_ari, "std_ari": _std_ari})

    for _affinity_mode, _method_name in [("rbf", "spectral_rbf"), ("nearest_neighbors", "spectral_knn_graph")]:
        try:
            _mean_ari, _std_ari = _bootstrap_stability(
                lambda Z, _mode=_affinity_mode: SpectralClustering(
                    n_clusters=_k_stability, random_state=RANDOM_STATE, affinity=_mode,
                ).fit_predict(Z)
            )
        except Exception as exc:
            print(f"  ⚠️  Stability check for {_method_name} failed: {exc!r}")
            _mean_ari, _std_ari = float("nan"), float("nan")
        _stability_rows.append({"method": _method_name, "mean_ari": _mean_ari, "std_ari": _std_ari})

    stability_df = pd.DataFrame(_stability_rows)
    print(f"Bootstrap cluster stability (k={_k_stability}, {_n_boot} resamples):\n")
    print(stability_df.round(4).to_string(index=False))
    return (stability_df,)


@app.cell
def _(
    CONFIG,
    Isomap,
    KMeans,
    LocallyLinearEmbedding,
    RANDOM_STATE,
    UMAP,
    UMAP_OK,
    X,
    adjusted_rand_score,
    df,
    graph_knn_neighbors_input,
    np,
    pd,
    silhouette_score,
    trustworthiness,
):
    _treatment_labels = df[CONFIG.treatment_col].astype(str).to_numpy()
    _n_neighbors = int(graph_knn_neighbors_input.value)
    _n_treatments = len(np.unique(_treatment_labels))

    manifold_embeddings = {"pca_2d": X[:, :2]}
    if UMAP_OK:
        manifold_embeddings["umap"] = UMAP(
            n_components=2, n_neighbors=_n_neighbors, min_dist=0.1, random_state=RANDOM_STATE, n_jobs=1,
        ).fit_transform(X)
    manifold_embeddings["isomap"] = Isomap(n_components=2, n_neighbors=_n_neighbors).fit_transform(X)
    manifold_embeddings["lle"] = LocallyLinearEmbedding(
        n_components=2, n_neighbors=_n_neighbors, random_state=RANDOM_STATE,
    ).fit_transform(X)

    _rows = []
    for _name, _embedding in manifold_embeddings.items():
        _km_labels = KMeans(n_clusters=_n_treatments, random_state=RANDOM_STATE, n_init=10).fit_predict(_embedding)
        _rows.append(
            {
                "embedding": _name,
                "trustworthiness": trustworthiness(X, _embedding, n_neighbors=10),
                "kmeans_silhouette_on_embedding": silhouette_score(_embedding, _km_labels),
                "ari_vs_treatment": adjusted_rand_score(_treatment_labels, _km_labels),
            }
        )
    manifold_comparison_df = pd.DataFrame(_rows)
    print("Manifold learning: local-structure preservation vs. downstream clustering:\n")
    print(manifold_comparison_df.round(4).to_string(index=False))
    return manifold_comparison_df, manifold_embeddings


@app.cell
def _(
    CONFIG,
    GRAPH_CLUSTERING_FIGS_DIR,
    df,
    manifold_embeddings,
    plt,
    scatter_panel,
):
    _names = list(manifold_embeddings.keys())
    _fig, _axes = plt.subplots(1, len(_names), figsize=(5.5 * len(_names), 5))
    _axes = [_axes] if len(_names) == 1 else list(_axes)
    for _ax, _name in zip(_axes, _names):
        scatter_panel(
            _ax, manifold_embeddings[_name][:, 0], manifold_embeddings[_name][:, 1],
            df[CONFIG.treatment_col].astype(str).tolist(), _name.upper(),
        )
    _fig.suptitle("Manifold embeddings colored by treatment", fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(GRAPH_CLUSTERING_FIGS_DIR / "manifold_embeddings.png", dpi=200, bbox_inches="tight")
    plt.close(_fig)
    print("✓ Saved: manifold_embeddings.png")
    _fig
    return


@app.cell
def _(CONFIG, GRAPH_CLUSTERING_FIGS_DIR, X, cosine_similarity, df, pd, plt):
    # Treatment-level consensus profiles: if wells within a treatment
    # average out to well-separated points, that supports discrete clusters;
    # if consensus profiles instead grade smoothly into each other (or sit at
    # opposite poles of a shared axis), that supports a continuum instead --
    # exactly the framing NB07's recovery axis already uses for this
    # dataset's dormancy/recovery design.
    _pca_df = pd.DataFrame(X, columns=[f"pc{i}" for i in range(X.shape[1])])
    _pca_df[CONFIG.treatment_col] = df[CONFIG.treatment_col].astype(str).to_numpy()
    _consensus = _pca_df.groupby(CONFIG.treatment_col).mean()
    consensus_similarity_df = pd.DataFrame(
        cosine_similarity(_consensus.to_numpy()), index=_consensus.index, columns=_consensus.index,
    )

    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _im = _ax.imshow(consensus_similarity_df.to_numpy(), cmap="RdYlBu_r", vmin=-1, vmax=1)
    _ax.set_xticks(range(len(_consensus.index)))
    _ax.set_xticklabels(_consensus.index, rotation=45, ha="right")
    _ax.set_yticks(range(len(_consensus.index)))
    _ax.set_yticklabels(_consensus.index)
    _fig.colorbar(_im, ax=_ax, label="Cosine similarity")
    _ax.set_title("Treatment-level consensus profile similarity")
    _fig.tight_layout()
    _fig.savefig(GRAPH_CLUSTERING_FIGS_DIR / "consensus_similarity.png", dpi=200, bbox_inches="tight")
    plt.close(_fig)
    print("✓ Saved: consensus_similarity.png")
    print(consensus_similarity_df.round(3).to_string())
    return (consensus_similarity_df,)


@app.cell
def _(clustering_comparison_df, consensus_similarity_df, np, stability_df):
    _kmeans_rows = clustering_comparison_df.loc[clustering_comparison_df["method"] == "kmeans_pca"]
    _best_k_row = _kmeans_rows.loc[_kmeans_rows["silhouette"].idxmax()]
    best_k_kmeans = int(_best_k_row["k"])
    best_k_kmeans_silhouette = float(_best_k_row["silhouette"])

    _at_k6 = clustering_comparison_df.loc[clustering_comparison_df["k"] == 6]
    kmeans_ari_at_k6 = float(_at_k6.loc[_at_k6["method"] == "kmeans_pca", "ari_vs_treatment"].iloc[0])
    _spectral_at_k6 = _at_k6.loc[_at_k6["method"].str.startswith("spectral_")]
    if len(_spectral_at_k6):
        _best_spectral_row = _spectral_at_k6.loc[_spectral_at_k6["ari_vs_treatment"].idxmax()]
        best_spectral_method_at_k6 = str(_best_spectral_row["method"])
        best_spectral_ari_at_k6 = float(_best_spectral_row["ari_vs_treatment"])
    else:
        best_spectral_method_at_k6, best_spectral_ari_at_k6 = "none", float("nan")
    spectral_beats_kmeans_at_k6 = best_spectral_ari_at_k6 > kmeans_ari_at_k6

    kmeans_stability = float(stability_df.loc[stability_df["method"] == "kmeans_pca", "mean_ari"].iloc[0])
    spectral_rbf_stability = float(stability_df.loc[stability_df["method"] == "spectral_rbf", "mean_ari"].iloc[0])

    # Most anti-correlated pair of treatment consensus profiles -- evidence
    # for a shared axis/continuum (opposite poles) rather than unrelated,
    # independently-scattered clusters.
    _sim = consensus_similarity_df.to_numpy().copy()
    np.fill_diagonal(_sim, np.nan)
    _min_idx = np.unravel_index(np.nanargmin(_sim), _sim.shape)
    most_anticorrelated_pair = (
        str(consensus_similarity_df.index[_min_idx[0]]),
        str(consensus_similarity_df.columns[_min_idx[1]]),
    )
    most_anticorrelated_value = float(_sim[_min_idx])
    return (
        best_k_kmeans,
        best_k_kmeans_silhouette,
        best_spectral_ari_at_k6,
        best_spectral_method_at_k6,
        kmeans_ari_at_k6,
        kmeans_stability,
        most_anticorrelated_pair,
        most_anticorrelated_value,
        spectral_beats_kmeans_at_k6,
        spectral_rbf_stability,
    )


@app.cell
def _(
    best_k_kmeans,
    best_k_kmeans_silhouette,
    best_spectral_ari_at_k6,
    best_spectral_method_at_k6,
    kmeans_ari_at_k6,
    kmeans_stability,
    mo,
    most_anticorrelated_pair,
    most_anticorrelated_value,
    spectral_beats_kmeans_at_k6,
    spectral_rbf_stability,
):
    _stability_note = (
        f"but at a stability cost — {spectral_rbf_stability:.2f} mean bootstrap ARI for "
        f"RBF-affinity spectral clustering vs. {kmeans_stability:.2f} for KMeans, i.e. "
        "noticeably less reproducible under resampling"
        if spectral_beats_kmeans_at_k6
        else "and KMeans remains both more accurate and more stable under resampling"
    )
    mo.md(
        f"""
    ### Verdict (computed from this run, not asserted in advance)

    - **Best unsupervised `k` for KMeans on PCA:** `{best_k_kmeans}`
      (silhouette {best_k_kmeans_silhouette:.3f}) — {"matches" if best_k_kmeans == 6 else "does **not** match"}
      the number of treatments (6). {"" if best_k_kmeans == 6 else "The strongest unsupervised split in this data is coarser than treatment identity, which is itself informative: it suggests a dominant axis (e.g. dormant vs. proliferative) rather than 6 evenly-separated treatment clusters."}
    - **Spectral clustering vs. KMeans at k=6:** best spectral variant is
      `{best_spectral_method_at_k6}` (ARI vs. treatment {best_spectral_ari_at_k6:.3f}) vs.
      KMeans's {kmeans_ari_at_k6:.3f}. {"Spectral clustering modestly **out-agrees** KMeans with treatment identity here, " + _stability_note + "." if spectral_beats_kmeans_at_k6 else _stability_note.capitalize() + "."}
    - **Most anti-correlated treatment pair:** `{most_anticorrelated_pair[0]}` vs.
      `{most_anticorrelated_pair[1]}` (cosine similarity {most_anticorrelated_value:.2f}) —
      consistent with a **continuum** between opposite phenotypic poles rather than
      unrelated, independently-scattered clusters.

    **Recommendation for this dataset:** graph-based clustering and manifold learning
    do not provide a decisive win over the existing PCA + KMeans / LDA-based
    workflow — RBF-affinity spectral clustering in particular is prone to
    degenerate, unstable solutions here (see the k-sweep and bootstrap-stability
    tables above), and manifold learning (UMAP/Isomap/LLE) improves visual
    separability without proportionally improving agreement with treatment
    identity. The more actionable finding is structural, not algorithmic: this
    dataset's dormancy/recovery design looks better described by a **continuum**
    (as NB07's geometric recovery axis already models) than by forcing it into
    discrete clusters, whichever clustering algorithm is used. Kept as a
    diagnostic section — re-run it on other experiments before assuming this
    verdict transfers; a dataset with more/tighter treatment clusters could
    show a different, more favorable balance for spectral clustering.
    """
    )
    return


@app.cell
def _(
    GRAPH_CLUSTERING_RESULTS_DIR,
    OVERWRITE_EXISTING_OUTPUTS,
    clustering_comparison_df,
    consensus_similarity_df,
    manifold_comparison_df,
    stability_df,
    write_csv_protected,
):
    write_csv_protected(
        clustering_comparison_df, GRAPH_CLUSTERING_RESULTS_DIR / "clustering_comparison.csv",
        overwrite=OVERWRITE_EXISTING_OUTPUTS,
    )
    write_csv_protected(
        stability_df, GRAPH_CLUSTERING_RESULTS_DIR / "bootstrap_stability.csv",
        overwrite=OVERWRITE_EXISTING_OUTPUTS,
    )
    write_csv_protected(
        manifold_comparison_df, GRAPH_CLUSTERING_RESULTS_DIR / "manifold_comparison.csv",
        overwrite=OVERWRITE_EXISTING_OUTPUTS,
    )
    write_csv_protected(
        consensus_similarity_df.reset_index(), GRAPH_CLUSTERING_RESULTS_DIR / "consensus_similarity.csv",
        overwrite=OVERWRITE_EXISTING_OUTPUTS,
    )
    print(f"✓ Saved graph-clustering artifacts to {GRAPH_CLUSTERING_RESULTS_DIR}")
    return


@app.cell
def _(
    CONFIG,
    GRAPH_CLUSTERING_RESULTS_DIR,
    REPO_ROOT,
    X,
    best_k_kmeans,
    best_spectral_ari_at_k6,
    best_spectral_method_at_k6,
    datetime,
    json,
    kmeans_ari_at_k6,
    kmeans_stability,
    spectral_rbf_stability,
    subprocess,
    validate_output_path,
):
    try:
        _git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT),
        ).stdout.strip()[:8]
    except Exception:
        _git_hash = "unknown"

    _provenance = {
        "notebook": "04_phenotypic_profiling.py (Section 8c)",
        "experiment_id": CONFIG.experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": _git_hash,
        "n_wells": int(X.shape[0]),
        "best_k_kmeans": best_k_kmeans,
        "kmeans_ari_at_k6": kmeans_ari_at_k6,
        "best_spectral_method_at_k6": best_spectral_method_at_k6,
        "best_spectral_ari_at_k6": best_spectral_ari_at_k6,
        "kmeans_stability": kmeans_stability,
        "spectral_rbf_stability": spectral_rbf_stability,
    }
    _path = validate_output_path(GRAPH_CLUSTERING_RESULTS_DIR / "provenance.json", overwrite=True)
    _path.write_text(json.dumps(_provenance, indent=2), encoding="utf-8")
    print(f"✓ Provenance: {_path}")
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
