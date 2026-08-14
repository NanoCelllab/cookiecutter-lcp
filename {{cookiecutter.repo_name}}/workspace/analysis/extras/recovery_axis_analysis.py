import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    def print(*values, sep=" ", end="\n"):
        """Display console-style progress in both notebook and App mode."""
        message = sep.join(str(value) for value in values) + end
        mo.output.append(mo.plain_text(message))

    return mo, print


@app.cell
def _(mo):
    mo.md(r"""
    # Extra — Single-Cell Phenotypic Recovery Axis

    > **This notebook is OPTIONAL and not part of the required 01→06 pipeline.**
    > It only applies to datasets with a **two-reference-state experimental
    > design** — a *baseline/dormant* state and a *proliferative/recovered*
    > state (e.g. a G0-arrest/dormancy-recovery study, or an analogous
    > "baseline vs. reference-recovered" biology). A dataset without that
    > design (a different perturbation type, no dormancy biology) simply has
    > nothing for this notebook to compute, and the notebook will say so and
    > stop gracefully rather than raising an error.

    ## Why this is an optional extra

    A recovery axis encodes a specific two-reference-state design rather than
    a generic profiling step. This notebook therefore loads its checkpoint
    directly and computes only the PCA and validation needed for that design;
    it does not depend on another notebook's in-memory state.

    ## What this notebook does

    1. Loads `single_cell_ready.parquet` and resolves metadata columns.
    2. Lets you pick the baseline and proliferative reference states from a
       dropdown populated with this dataset's actual treatment values. If
       either is left unset (or isn't present in the data), the analysis
       **stops gracefully** with a clear message — it does not raise.
    3. Curates and scales the single-cell feature matrix, balanced-samples
       cells per treatment (`hca_pipeline.modelling.balanced_sample`), and
       fits a PCA — a minimal stand-in for NB06's `X_sc_pca`.
    4. Builds a geometric recovery axis between the two reference centroids
       in PCA space, trains a grouped (by well) logistic classifier for
       proliferative-vs-baseline, aggregates recovery scores to well and
       condition level, plots dose-response (if a dose axis exists), and
       back-projects the axis to original features using the shared
       `hca_pipeline.taxonomy` classifier.

    ## Outputs

    Written under a **dedicated `recovery_axis/` subdirectory** (both
    `results/recovery_axis/` and `figures/recovery_axis/`) so nothing here
    collides with `06_single_cell_analysis.py`'s own outputs.

    > Cells are observations, but wells are the experimental units.
    > Inferential comparisons use well-level summaries.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What this notebook cannot tell you

    Recovery scores are conditional on the selected baseline and proliferative
    references being valid, replicated, and distinguishable. The axis measures
    similarity along that contrast; it does not prove restoration of mechanism,
    function, or cell fate, and it may miss orthogonal phenotypes.
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
    import json
    import platform
    import subprocess
    import warnings
    from dataclasses import replace
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Sequence

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler

    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )
    pd.set_option("display.max_columns", 80)
    return (
        StratifiedGroupKFold,
        LogisticRegression,
        PCA,
        Path,
        RobustScaler,
        Sequence,
        StandardScaler,
        average_precision_score,
        balanced_accuracy_score,
        datetime,
        json,
        make_pipeline,
        np,
        pd,
        platform,
        plt,
        replace,
        roc_auc_score,
        smf,
        subprocess,
        timezone,
    )


@app.cell
def _(Path, print):
    # Locate the repo root before hca_pipeline can be imported (bootstrap:
    # can't import find_repo_root from the package until sys.path includes
    # workspace). __file__ is used instead of cwd so this notebook
    # behaves identically regardless of the launch working directory.
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
    from hca_pipeline.normalize import clean_features_before_normalization
    from hca_pipeline.plotting import save_figure_protected
    from hca_pipeline.taxonomy import build_taxonomy_table

    print(f"  ✓  Shared utilities loaded from hca_pipeline ({_pipelines_dir})")
    return (
        ExperimentConfig,
        REPO_ROOT,
        balanced_sample,
        build_taxonomy_table,
        clean_features_before_normalization,
        infer_feature_cols,
        save_figure_protected,
        write_csv_protected,
        write_parquet_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Pick the experiment folder. Defaults are pre-filled from a previously
    saved `experiment_config.json` when one exists.
    """)
    return


@app.cell
def _(REPO_ROOT, mo):
    _backend_dir = REPO_ROOT / "workspace" / "backend"
    available_experiment_ids = (
        sorted(p.name for p in _backend_dir.iterdir() if p.is_dir()) if _backend_dir.is_dir() else []
    )
    _default_experiment_id = available_experiment_ids[0] if available_experiment_ids else "SET_EXPERIMENT_ID_HERE"

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
    random_state_input = mo.ui.number(value=42, start=0, stop=10_000, label="Random seed")
    n_per_class_input = mo.ui.number(
        value=5_000,
        start=100,
        stop=50_000,
        step=100,
        label="Max cells sampled per treatment (hca_pipeline.modelling.balanced_sample)",
    )
    n_pca_components_input = mo.ui.number(
        value=50, start=2, stop=200, label="PCA components fit on the sampled cells"
    )
    n_recovery_pcs_input = mo.ui.number(
        value=30, start=2, stop=200, label="PCA dimensions used for the recovery axis"
    )
    nan_threshold_input = mo.ui.number(
        value=0.05, start=0.0, stop=1.0, step=0.01, label="Max fraction missing before a feature is dropped"
    )
    proliferative_probability_threshold_input = mo.ui.number(
        value=0.50, start=0.0, stop=1.0, step=0.05, label="Proliferative-probability classification threshold"
    )
    recovery_score_threshold_input = mo.ui.number(
        value=0.50, start=-2.0, stop=2.0, step=0.05, label="Recovery-score 'recovery-like' threshold"
    )
    overwrite_input = mo.ui.checkbox(
        value=loaded_config.overwrite_existing_outputs, label="Overwrite existing outputs"
    )
    save_history_input = mo.ui.checkbox(
        value=loaded_config.save_provenance_history, label="Save timestamped provenance history"
    )
    mo.accordion({"Advanced recovery-axis and output settings": mo.vstack([
        random_state_input, n_per_class_input, n_pca_components_input,
        n_recovery_pcs_input, nan_threshold_input,
        proliferative_probability_threshold_input, recovery_score_threshold_input,
        overwrite_input, save_history_input,
    ])})
    return (
        n_pca_components_input,
        n_per_class_input,
        n_recovery_pcs_input,
        nan_threshold_input,
        overwrite_input,
        proliferative_probability_threshold_input,
        random_state_input,
        recovery_score_threshold_input,
        save_history_input,
    )


@app.cell
def _(
    REPO_ROOT,
    experiment_id_input,
    loaded_config,
    n_pca_components_input,
    n_per_class_input,
    n_recovery_pcs_input,
    nan_threshold_input,
    overwrite_input,
    proliferative_probability_threshold_input,
    random_state_input,
    recovery_score_threshold_input,
    replace,
    save_history_input,
    print,
):
    EXPERIMENT_ID = experiment_id_input.value
    RANDOM_STATE = int(random_state_input.value)
    N_PER_CLASS = int(n_per_class_input.value)
    N_PCA_COMPONENTS = int(n_pca_components_input.value)
    N_RECOVERY_PCS_TARGET = int(n_recovery_pcs_input.value)
    NAN_THRESHOLD = float(nan_threshold_input.value)
    PROLIFERATIVE_PROBABILITY_THRESHOLD = float(proliferative_probability_threshold_input.value)
    RECOVERY_SCORE_THRESHOLD = float(recovery_score_threshold_input.value)

    CONFIG = replace(
        loaded_config,
        experiment_id=EXPERIMENT_ID,
        overwrite_existing_outputs=bool(overwrite_input.value),
        save_provenance_history=bool(save_history_input.value),
    )
    _config_path = CONFIG.save(REPO_ROOT)

    print("═" * 72)
    print("RECOVERY-AXIS CONFIGURATION (optional extension)")
    print("═" * 72)
    print(f"  Experiment ID:                    {EXPERIMENT_ID}")
    print(f"  Random seed:                      {RANDOM_STATE}")
    print(f"  Cells sampled per treatment:       {N_PER_CLASS}")
    print(f"  PCA components fit:                {N_PCA_COMPONENTS}")
    print(f"  PCA dims used for recovery axis:   {N_RECOVERY_PCS_TARGET}")
    print(f"  Max missing fraction:              {NAN_THRESHOLD}")
    print(f"  Proliferative-probability thresh.: {PROLIFERATIVE_PROBABILITY_THRESHOLD}")
    print(f"  Recovery-score threshold:          {RECOVERY_SCORE_THRESHOLD}")
    print(f"  Saved config:                      {_config_path}")
    return (
        CONFIG,
        EXPERIMENT_ID,
        NAN_THRESHOLD,
        N_PCA_COMPONENTS,
        N_PER_CLASS,
        N_RECOVERY_PCS_TARGET,
        PROLIFERATIVE_PROBABILITY_THRESHOLD,
        RANDOM_STATE,
        RECOVERY_SCORE_THRESHOLD,
    )


@app.cell
def _(EXPERIMENT_ID, REPO_ROOT, print):
    WORKSPACE_DIR = REPO_ROOT / "workspace"
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    PROFILES_DIR = WORKSPACE_DIR / "profiles" / EXPERIMENT_ID

    PROFILES_OUT_DIR = PROFILES_DIR / "outputs"
    # Deliberately namespaced under `recovery_axis/` — NOT the bare `results/`
    # / `figures/single_cell/` directories NB06 uses — so this optional
    # notebook's outputs never collide with 06_single_cell_analysis.py's.
    RESULTS_DIR = ANALYSIS_DIR / "results" / "recovery_axis"
    FIGS_DIR = ANALYSIS_DIR / "figures" / "recovery_axis"

    INPUT_PARQUET = PROFILES_OUT_DIR / "single_cell_ready.parquet"
    _legacy_input = PROFILES_OUT_DIR / "cache" / "single_cell_ready.parquet"
    if not INPUT_PARQUET.exists() and _legacy_input.exists():
        INPUT_PARQUET = _legacy_input
        print(
            "  ℹ Using the legacy NB02 cache location for compatibility. "
            "Rerun NB02 once to promote this file to a declared output."
        )

    for _directory in (RESULTS_DIR, FIGS_DIR):
        _directory.mkdir(parents=True, exist_ok=True)

    print(f"  Analysis directory: {ANALYSIS_DIR}")
    print(f"  Input parquet:      {INPUT_PARQUET}")
    print(f"  Results directory:  {RESULTS_DIR}")
    print(f"  Figures directory:  {FIGS_DIR}")
    return FIGS_DIR, INPUT_PARQUET, RESULTS_DIR


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Load single-cell profiles

    Loaded from NB02's declared single-cell output — this notebook does not assume
    `06_single_cell_analysis.py` has been run first.
    """)
    return


@app.cell
def _(INPUT_PARQUET, pd, print):
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PARQUET}\n"
            "Run NB02 (aggregate / normalize / feature-select) first — this "
            "optional notebook re-derives its own single-cell PCA from that "
            "cache rather than reusing NB06's in-memory state."
        )
    df_sc = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded single-cell profiles: {df_sc.shape[0]:,} cells x {df_sc.shape[1]:,} columns")
    return (df_sc,)


@app.cell
def _(CONFIG, df_sc, print):
    _resolved = CONFIG.resolve_columns(df_sc)
    PLATE_COL = _resolved.plate_col
    WELL_COL = _resolved.well_col
    TREATMENT_COL = _resolved.treatment_col
    CONC_COL = _resolved.concentration_col

    print(f"  Detected: plate={PLATE_COL}, well={WELL_COL}, treatment={TREATMENT_COL}, concentration={CONC_COL}")
    print(df_sc[TREATMENT_COL].value_counts().to_string())
    return CONC_COL, PLATE_COL, TREATMENT_COL, WELL_COL


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Reference-state configuration (this is where the notebook becomes optional)

    Choose the baseline/dormant and proliferative/recovered reference
    states from this dataset's actual treatment values. Leaving either
    unset — or a dataset that doesn't have anything matching this
    two-reference-state design — is expected for most LCP experiments;
    the notebook stops gracefully in that case instead of raising an error.
    """)
    return


@app.cell
def _(TREATMENT_COL, df_sc, mo):
    _treatment_values = sorted(df_sc[TREATMENT_COL].dropna().astype(str).unique().tolist())
    _dropdown_options = [""] + _treatment_values
    _default_baseline = ""
    _default_proliferative = ""

    baseline_label_input = mo.ui.dropdown(
        options=_dropdown_options,
        value=_default_baseline,
        label="Baseline / dormant reference state",
    )
    proliferative_label_input = mo.ui.dropdown(
        options=_dropdown_options,
        value=_default_proliferative,
        label="Proliferative / recovered reference state",
    )
    mo.vstack([baseline_label_input, proliferative_label_input])
    return baseline_label_input, proliferative_label_input


@app.cell
def _(
    TREATMENT_COL,
    baseline_label_input,
    df_sc,
    mo,
    print,
    proliferative_label_input,
):
    _treatment_counts = df_sc[TREATMENT_COL].astype(str).value_counts()
    _baseline_value = baseline_label_input.value
    _proliferative_value = proliferative_label_input.value

    _reasons = []
    if not _baseline_value:
        _reasons.append("no baseline/dormant reference state selected")
    if not _proliferative_value:
        _reasons.append("no proliferative/recovered reference state selected")
    if _baseline_value and _proliferative_value and _baseline_value == _proliferative_value:
        _reasons.append("baseline and proliferative reference states must differ")
    if _baseline_value and _treatment_counts.get(_baseline_value, 0) == 0:
        _reasons.append(f"baseline reference state {_baseline_value!r} has no cells in this dataset")
    if _proliferative_value and _treatment_counts.get(_proliferative_value, 0) == 0:
        _reasons.append(f"proliferative reference state {_proliferative_value!r} has no cells in this dataset")

    _recovery_axis_configured = not _reasons

    mo.stop(
        not _recovery_axis_configured,
        mo.md(
            "### Recovery-axis analysis skipped\n\n"
            "This is an **optional, experiment-specific** extension that only "
            "applies to datasets with a two-reference-state design (e.g. a "
            "dormancy/recovery experiment with a baseline state and a "
            "proliferative-recovered state). It does not apply to every LCP "
            "dataset, and this dataset (as currently configured) does not "
            "qualify.\n\n"
            + ("**Reason(s):** " + "; ".join(_reasons) + ".\n\n" if _reasons else "")
            + "Select both reference states above (using the actual treatment "
            "labels present in this dataset) to run the full analysis."
        ),
    )

    BASELINE_LABEL = _baseline_value
    PROLIFERATIVE_LABEL = _proliferative_value
    print("Recovery-axis configuration")
    print(f"  Baseline reference       : {BASELINE_LABEL}")
    print(f"  Proliferative reference  : {PROLIFERATIVE_LABEL}")
    print(f"  Baseline cells available     : {_treatment_counts.get(BASELINE_LABEL, 0):,}")
    print(f"  Proliferative cells available: {_treatment_counts.get(PROLIFERATIVE_LABEL, 0):,}")
    return BASELINE_LABEL, PROLIFERATIVE_LABEL


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Lightweight single-cell PCA (stand-in for NB06's `X_sc_pca`)

    Curate and scale the single-cell feature matrix, balanced-sample cells
    per treatment, and fit a PCA. This is intentionally minimal: enough to
    get a PCA-reduced matrix, a fitted PCA object, and the feature-column
    list the recovery-axis math and back-projection need — not a full
    reproduction of NB06's outlier removal / clustering / SHAP machinery.
    """)
    return


@app.cell
def _(
    BASELINE_LABEL,
    CONC_COL,
    NAN_THRESHOLD,
    PROLIFERATIVE_LABEL,
    clean_features_before_normalization,
    df_sc,
    infer_feature_cols,
    print,
):
    print(f"Curating single-cell features for the recovery axis between {BASELINE_LABEL!r} and {PROLIFERATIVE_LABEL!r}")

    # Non-phenotypic identifier-like columns (object linkage IDs, not
    # measurements). `infer_feature_cols` already drops the *_ImageNumber /
    # *_ObjectNumber suffix forms; these two extra patterns catch the
    # remaining CellProfiler ID columns (e.g. "..._Number_Object_Number",
    # "Nuclei_Parent_NucleiCP") that would otherwise pollute the PCA.
    _NON_PHENOTYPIC_PATTERNS = ("Parent_", "Number_Object_Number")
    _feat_cols_raw = [
        c for c in infer_feature_cols(df_sc) if not any(pattern in c for pattern in _NON_PHENOTYPIC_PATTERNS)
    ]
    if not _feat_cols_raw:
        raise ValueError("No usable feature columns were detected in single_cell_ready.parquet.")

    df_sc_cleaned, feature_cols_after_missingness, _clean_summary = clean_features_before_normalization(
        df_sc, _feat_cols_raw, max_missing_fraction=NAN_THRESHOLD
    )

    print(f"  Feature columns detected      : {len(_feat_cols_raw):,}")
    print(f"  Removed (high missingness)    : {_clean_summary['n_features_removed']:,}")
    print(f"  Remaining after missingness   : {len(feature_cols_after_missingness):,}")
    print(f"  Missing values imputed        : {_clean_summary['n_missing_before']:,} -> {_clean_summary['n_missing_after']:,}")
    return df_sc_cleaned, feature_cols_after_missingness


@app.cell
def _(RobustScaler, df_sc_cleaned, feature_cols_after_missingness, np, pd, print):
    _X_raw = df_sc_cleaned[feature_cols_after_missingness].apply(pd.to_numeric, errors="coerce")
    _variance = _X_raw.var(skipna=True).to_numpy()
    _n_unique = _X_raw.nunique(dropna=True).to_numpy()
    _variable_mask = np.isfinite(_variance) & (_n_unique >= 3) & (_variance > 0)
    feature_cols_model = [c for c, keep in zip(feature_cols_after_missingness, _variable_mask) if keep]
    _n_removed_low_variance = len(feature_cols_after_missingness) - len(feature_cols_model)
    if not feature_cols_model:
        raise ValueError("Feature curation removed every feature as constant or non-finite.")

    _X_model = _X_raw[feature_cols_model].to_numpy(dtype=float)
    _scaler = RobustScaler()
    _X_scaled = _scaler.fit_transform(_X_model)

    _CLIP_VALUE = 20.0
    _n_clipped = int(np.sum((_X_scaled < -_CLIP_VALUE) | (_X_scaled > _CLIP_VALUE)))
    X_sc_scaled = np.clip(_X_scaled, -_CLIP_VALUE, _CLIP_VALUE)

    if not np.isfinite(X_sc_scaled).all():
        raise ValueError("Non-finite values remain in the scaled single-cell feature matrix.")

    print(f"  Removed (near-zero variance)  : {_n_removed_low_variance:,}")
    print(f"  Features entering PCA         : {len(feature_cols_model):,}")
    print(f"  Values clipped at ±{_CLIP_VALUE:g}          : {_n_clipped:,}")
    print(f"  Scaled matrix                 : {X_sc_scaled.shape[0]:,} cells × {X_sc_scaled.shape[1]:,} features")
    return X_sc_scaled, feature_cols_model


@app.cell
def _(
    BASELINE_LABEL,
    CONC_COL,
    N_PCA_COMPONENTS,
    N_PER_CLASS,
    PCA,
    PLATE_COL,
    PROLIFERATIVE_LABEL,
    RANDOM_STATE,
    TREATMENT_COL,
    WELL_COL,
    X_sc_scaled,
    balanced_sample,
    df_sc_cleaned,
    print,
):
    _df_sampled_full, X_sampled = balanced_sample(df_sc_cleaned, X_sc_scaled, TREATMENT_COL, N_PER_CLASS, RANDOM_STATE)

    # Keep only the metadata this analysis actually needs — the source
    # notebook's df_sc_sampled dragged along all ~1,800 raw feature columns,
    # which is why its single_cell_recovery_scores.parquet output ballooned
    # to hundreds of MB for no benefit once X_sc_pca has been computed.
    _meta_cols_needed = [c for c in [PLATE_COL, WELL_COL, TREATMENT_COL, CONC_COL] if c and c in _df_sampled_full.columns]
    df_sc_sampled = _df_sampled_full[_meta_cols_needed].reset_index(drop=True).copy()

    _n_components = min(N_PCA_COMPONENTS, X_sampled.shape[0] - 1, X_sampled.shape[1])
    if _n_components < 2:
        raise ValueError(
            f"Recovery-axis PCA requires at least two components; only {_n_components} is available."
        )
    if _n_components < N_PCA_COMPONENTS:
        print(
            f"ℹ PCA components reduced from {N_PCA_COMPONENTS} to {_n_components} "
            "to match sampled cells and curated features."
        )
    pca_sc = PCA(n_components=_n_components, random_state=RANDOM_STATE)
    X_sc_pca = pca_sc.fit_transform(X_sampled)

    print("Balanced sample + PCA (lightweight re-derivation of NB06's X_sc_pca)")
    print(f"  Cells sampled  : {X_sampled.shape[0]:,}")
    print(f"  PCA components : {X_sc_pca.shape[1]}")
    print(f"  PC1 variance   : {pca_sc.explained_variance_ratio_[0]:.1%}")
    print(f"  Baseline={BASELINE_LABEL!r} / Proliferative={PROLIFERATIVE_LABEL!r} cell counts:")
    print(df_sc_sampled[TREATMENT_COL].value_counts().to_string())
    return X_sc_pca, df_sc_sampled, pca_sc


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — Geometric recovery axis

    The baseline and proliferative controls define a directed axis in PCA
    space:

    \[
    \mathbf{v}_{recovery}=\boldsymbol{\mu}_{proliferative}-\boldsymbol{\mu}_{baseline}
    \]

    Each cell is decomposed into a component parallel to this transition and
    an orthogonal component representing a phenotype not explained by
    recovery.

    - `recovery_score = 0`: baseline centroid;
    - `recovery_score = 1`: proliferative centroid;
    - values between 0 and 1: partial recovery;
    - values above 1: displacement beyond the proliferative centroid;
    - values below 0: displacement opposite to recovery.
    """)
    return


@app.cell
def _(
    BASELINE_LABEL,
    N_RECOVERY_PCS_TARGET,
    PLATE_COL,
    PROLIFERATIVE_LABEL,
    TREATMENT_COL,
    WELL_COL,
    X_sc_pca,
    df_sc_sampled,
    np,
    print,
):
    N_RECOVERY_PCS = min(N_RECOVERY_PCS_TARGET, X_sc_pca.shape[1])

    _required_columns = [PLATE_COL, WELL_COL, TREATMENT_COL]
    _missing_columns = [c for c in _required_columns if c not in df_sc_sampled.columns]
    if _missing_columns:
        raise KeyError(f"Missing metadata columns: {_missing_columns}")
    if X_sc_pca.shape[0] != len(df_sc_sampled):
        raise ValueError(f"PCA rows ({X_sc_pca.shape[0]:,}) and metadata rows ({len(df_sc_sampled):,}) are not aligned.")
    if not np.isfinite(X_sc_pca).all():
        raise ValueError("X_sc_pca contains NaN or infinite values.")

    _counts = df_sc_sampled[TREATMENT_COL].value_counts()
    for _label in [BASELINE_LABEL, PROLIFERATIVE_LABEL]:
        if _counts.get(_label, 0) == 0:
            # Not the "no two-reference-state design" case (that's already
            # handled gracefully above) — this would mean sampling somehow
            # dropped a label that was present in df_sc, i.e. a real bug.
            raise ValueError(
                f"Reference label {_label!r} was present before sampling but has zero "
                "sampled cells (unexpected — indicates a sampling inconsistency)."
            )

    X_recovery = np.asarray(X_sc_pca[:, :N_RECOVERY_PCS], dtype=float)
    recovery_meta = df_sc_sampled.reset_index(drop=True).copy()
    print(
        f"Cells: {len(recovery_meta):,} | PCs: {X_recovery.shape[1]} | "
        f"Wells: {recovery_meta[[PLATE_COL, WELL_COL]].drop_duplicates().shape[0]}"
    )
    return N_RECOVERY_PCS, X_recovery, recovery_meta


@app.cell
def _(
    BASELINE_LABEL,
    PROLIFERATIVE_LABEL,
    TREATMENT_COL,
    X_recovery,
    np,
    print,
    recovery_meta,
):
    baseline_mask = recovery_meta[TREATMENT_COL].eq(BASELINE_LABEL).to_numpy()
    proliferative_mask = recovery_meta[TREATMENT_COL].eq(PROLIFERATIVE_LABEL).to_numpy()
    baseline_centroid = X_recovery[baseline_mask].mean(axis=0)
    proliferative_centroid = X_recovery[proliferative_mask].mean(axis=0)
    recovery_vector = proliferative_centroid - baseline_centroid
    _norm_sq = float(recovery_vector @ recovery_vector)
    if not np.isfinite(_norm_sq) or _norm_sq <= 0:
        raise ValueError("Reference centroids do not define a valid recovery axis.")
    unit_vector = recovery_vector / np.sqrt(_norm_sq)
    _centered = X_recovery - baseline_centroid
    recovery_score = (_centered @ recovery_vector) / _norm_sq
    _parallel = np.outer(_centered @ unit_vector, unit_vector)
    orthogonal_distance = np.linalg.norm(_centered - _parallel, axis=1)
    distance_to_baseline = np.linalg.norm(X_recovery - baseline_centroid, axis=1)
    distance_to_proliferative = np.linalg.norm(X_recovery - proliferative_centroid, axis=1)

    cell_recovery_df = recovery_meta.copy()
    cell_recovery_df["recovery_score"] = recovery_score
    cell_recovery_df["orthogonal_distance"] = orthogonal_distance
    cell_recovery_df["distance_to_baseline"] = distance_to_baseline
    cell_recovery_df["distance_to_proliferative"] = distance_to_proliferative
    cell_recovery_df["closer_to_proliferative"] = distance_to_proliferative < distance_to_baseline

    print(cell_recovery_df.loc[baseline_mask, ["recovery_score", "orthogonal_distance"]].median().rename(BASELINE_LABEL))
    print(
        cell_recovery_df.loc[proliferative_mask, ["recovery_score", "orthogonal_distance"]]
        .median()
        .rename(PROLIFERATIVE_LABEL)
    )
    return cell_recovery_df, unit_vector


@app.cell
def _(
    BASELINE_LABEL,
    CONFIG,
    FIGS_DIR,
    INPUT_PARQUET,
    PLATE_COL,
    PROLIFERATIVE_LABEL,
    TREATMENT_COL,
    WELL_COL,
    cell_recovery_df,
    np,
    plt,
    print,
    save_figure_protected,
):
    _reference_geometric_wells = (
        cell_recovery_df.loc[cell_recovery_df[TREATMENT_COL].isin([BASELINE_LABEL, PROLIFERATIVE_LABEL])]
        .groupby([PLATE_COL, WELL_COL, TREATMENT_COL], observed=True)
        .agg(
            n_cells=("recovery_score", "size"),
            recovery_median=("recovery_score", "median"),
            recovery_iqr=("recovery_score", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            orthogonal_median=("orthogonal_distance", "median"),
            closer_to_proliferative_fraction=("closer_to_proliferative", "mean"),
        )
        .reset_index()
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for _pos, _label in enumerate([BASELINE_LABEL, PROLIFERATIVE_LABEL]):
        _v = _reference_geometric_wells.loc[_reference_geometric_wells[TREATMENT_COL].eq(_label), "recovery_median"]
        _axes[0].scatter(np.full(len(_v), _pos), _v, s=55, alpha=0.8)
        _o = _reference_geometric_wells.loc[_reference_geometric_wells[TREATMENT_COL].eq(_label), "orthogonal_median"]
        _axes[1].scatter(np.full(len(_o), _pos), _o, s=55, alpha=0.8)
    _axes[0].axhline(0, ls="--", lw=1)
    _axes[0].axhline(1, ls="--", lw=1)
    for _ax in _axes:
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels([BASELINE_LABEL, PROLIFERATIVE_LABEL])
    _axes[0].set_ylabel("Median recovery score per well")
    _axes[1].set_ylabel("Median orthogonal distance per well")
    _axes[0].set_title("Reference separation")
    _axes[1].set_title("Deviation from recovery axis")
    _fig.tight_layout()
    _figure_path = FIGS_DIR / "reference_geometric_validation.png"
    _figure_status = save_figure_protected(
        _fig, _figure_path, overwrite=CONFIG.overwrite_existing_outputs,
        dpi=180, bbox_inches="tight",
    )
    plt.close(_fig)
    print(f"✓ Reference geometry figure {_figure_status}: {_figure_path} ({len(_reference_geometric_wells)} reference wells)")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Probabilistic proliferative classifier

    A regularized logistic-regression classifier is trained using only the
    two reference states. Validation is grouped by well to prevent
    cell-level information leakage. The output is interpreted as the
    probability that a cell resembles the proliferative reference
    phenotype.
    """)
    return


@app.cell
def _(
    BASELINE_LABEL,
    PLATE_COL,
    PROLIFERATIVE_LABEL,
    TREATMENT_COL,
    WELL_COL,
    X_recovery,
    pd,
    print,
    recovery_meta,
):
    reference_mask = recovery_meta[TREATMENT_COL].isin([BASELINE_LABEL, PROLIFERATIVE_LABEL]).to_numpy()
    X_reference = X_recovery[reference_mask]
    reference_meta = recovery_meta.loc[reference_mask].reset_index(drop=True)
    y_reference = reference_meta[TREATMENT_COL].eq(PROLIFERATIVE_LABEL).astype(int).to_numpy()
    reference_groups = (reference_meta[PLATE_COL].astype(str) + "::" + reference_meta[WELL_COL].astype(str)).to_numpy()
    print(
        f"Reference cells: {len(reference_meta):,} | "
        f"Wells: {pd.Series(reference_groups).nunique()} | "
        f"Positive class: {y_reference.mean():.1%}"
    )
    return X_reference, reference_groups, reference_meta, y_reference


@app.cell
def _(
    StratifiedGroupKFold,
    LogisticRegression,
    PROLIFERATIVE_PROBABILITY_THRESHOLD,
    RANDOM_STATE,
    StandardScaler,
    X_reference,
    average_precision_score,
    balanced_accuracy_score,
    make_pipeline,
    np,
    pd,
    reference_groups,
    roc_auc_score,
    y_reference,
    print,
):
    _reference_well_classes = pd.DataFrame(
        {"group": reference_groups, "label": y_reference}
    ).drop_duplicates()
    _wells_per_class = _reference_well_classes.groupby("label")["group"].nunique()
    if len(_wells_per_class) < 2 or int(_wells_per_class.min()) < 2:
        raise ValueError(
            "Grouped classifier validation requires at least two independent wells in each "
            f"reference state; observed wells per class: {_wells_per_class.to_dict()}."
        )
    n_splits = min(5, int(_wells_per_class.min()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_probability = np.full(len(y_reference), np.nan)
    _cv_prediction = np.full(len(y_reference), -1)
    _fold_records = []
    for _fold, (_tr, _te) in enumerate(cv.split(X_reference, y_reference, groups=reference_groups), start=1):
        if len(np.unique(y_reference[_tr])) < 2 or len(np.unique(y_reference[_te])) < 2:
            raise ValueError(
                f"Grouped validation fold {_fold} does not contain both reference states."
            )
        _model = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
        )
        _model.fit(X_reference[_tr], y_reference[_tr])
        _p = _model.predict_proba(X_reference[_te])[:, 1]
        _pred = (_p >= PROLIFERATIVE_PROBABILITY_THRESHOLD).astype(int)
        cv_probability[_te] = _p
        _cv_prediction[_te] = _pred
        _fold_records.append(
            {
                "fold": _fold,
                "n_train": len(_tr),
                "n_test": len(_te),
                "balanced_accuracy": balanced_accuracy_score(y_reference[_te], _pred),
                "roc_auc": roc_auc_score(y_reference[_te], _p),
                "pr_auc": average_precision_score(y_reference[_te], _p),
            }
        )
    cv_metrics_df = pd.DataFrame(_fold_records)
    print(cv_metrics_df.to_string(index=False))
    print(cv_metrics_df[["balanced_accuracy", "roc_auc", "pr_auc"]].agg(["mean", "std"]).to_string())
    return cv_metrics_df, cv_probability


@app.cell
def _(
    BASELINE_LABEL,
    CONFIG,
    FIGS_DIR,
    PLATE_COL,
    PROLIFERATIVE_LABEL,
    PROLIFERATIVE_PROBABILITY_THRESHOLD,
    TREATMENT_COL,
    WELL_COL,
    cv_probability,
    np,
    plt,
    print,
    reference_meta,
    save_figure_protected,
):
    reference_cv_df = reference_meta.copy()
    reference_cv_df["cv_proliferative_probability"] = cv_probability
    reference_cv_wells = (
        reference_cv_df.groupby([PLATE_COL, WELL_COL, TREATMENT_COL], observed=True)
        .agg(
            n_cells=("cv_proliferative_probability", "size"),
            probability_median=("cv_proliferative_probability", "median"),
            proliferative_like_fraction=(
                "cv_proliferative_probability",
                lambda x: (x >= PROLIFERATIVE_PROBABILITY_THRESHOLD).mean(),
            ),
        )
        .reset_index()
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for _pos, _label in enumerate([BASELINE_LABEL, PROLIFERATIVE_LABEL]):
        _v = reference_cv_wells.loc[reference_cv_wells[TREATMENT_COL].eq(_label), "probability_median"]
        _axes[0].scatter(np.full(len(_v), _pos), _v, s=55, alpha=0.8)
    _axes[0].axhline(PROLIFERATIVE_PROBABILITY_THRESHOLD, ls="--", lw=1)
    _axes[0].set_xticks([0, 1])
    _axes[0].set_xticklabels([BASELINE_LABEL, PROLIFERATIVE_LABEL])
    _axes[0].set_ylabel("Median out-of-fold probability")
    for _label in [BASELINE_LABEL, PROLIFERATIVE_LABEL]:
        _axes[1].hist(
            reference_cv_df.loc[reference_cv_df[TREATMENT_COL].eq(_label), "cv_proliferative_probability"],
            bins=40,
            alpha=0.6,
            density=True,
            label=_label,
        )
    _axes[1].axvline(PROLIFERATIVE_PROBABILITY_THRESHOLD, ls="--", lw=1)
    _axes[1].set_xlabel("Out-of-fold proliferative probability")
    _axes[1].set_ylabel("Density")
    _axes[1].legend()
    _fig.tight_layout()
    _figure_path = FIGS_DIR / "reference_classifier_validation.png"
    _figure_status = save_figure_protected(
        _fig, _figure_path, overwrite=CONFIG.overwrite_existing_outputs,
        dpi=180, bbox_inches="tight",
    )
    plt.close(_fig)
    print(f"✓ Classifier-validation figure {_figure_status}: {_figure_path} ({len(reference_cv_wells)} reference wells)")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — Fit the final classifier and project all cells

    After grouped validation, the final classifier is trained using all
    baseline and proliferative reference cells. Non-reference-treated cells
    are projected without influencing the fitted reference boundary.
    """)
    return


@app.cell
def _(
    CONFIG,
    LogisticRegression,
    PROLIFERATIVE_PROBABILITY_THRESHOLD,
    RANDOM_STATE,
    RECOVERY_SCORE_THRESHOLD,
    RESULTS_DIR,
    StandardScaler,
    X_recovery,
    X_reference,
    cell_recovery_df,
    make_pipeline,
    print,
    write_parquet_protected,
    y_reference,
):
    final_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
    )
    final_classifier.fit(X_reference, y_reference)
    cell_recovery_df["proliferative_probability"] = final_classifier.predict_proba(X_recovery)[:, 1]
    cell_recovery_df["proliferative_like"] = (
        cell_recovery_df["proliferative_probability"] >= PROLIFERATIVE_PROBABILITY_THRESHOLD
    )
    cell_recovery_df["geometric_recovery_like"] = cell_recovery_df["recovery_score"] >= RECOVERY_SCORE_THRESHOLD

    _output_path = RESULTS_DIR / "single_cell_recovery_scores.parquet"
    _status = write_parquet_protected(cell_recovery_df, _output_path, overwrite=CONFIG.overwrite_existing_outputs)
    print(f"✓ Final reference classifier fitted; single-cell scores {_status}.")
    print(f"  File: {_output_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — Aggregate by well

    Primary outcomes include median recovery score, proliferative-like
    fraction, and orthogonal distance. A strong phenotype with high
    orthogonal distance should not be called recovery-like.
    """)
    return


@app.cell
def _(
    CONC_COL,
    CONFIG,
    PLATE_COL,
    RESULTS_DIR,
    TREATMENT_COL,
    WELL_COL,
    cell_recovery_df,
    print,
    write_csv_protected,
):
    _group_columns = [PLATE_COL, WELL_COL, TREATMENT_COL]
    if CONC_COL and CONC_COL in cell_recovery_df.columns:
        _group_columns.append(CONC_COL)

    well_recovery_df = (
        cell_recovery_df.groupby(_group_columns, observed=True)
        .agg(
            n_cells=("recovery_score", "size"),
            recovery_median=("recovery_score", "median"),
            recovery_mean=("recovery_score", "mean"),
            recovery_q10=("recovery_score", lambda x: x.quantile(0.10)),
            recovery_q90=("recovery_score", lambda x: x.quantile(0.90)),
            recovery_iqr=("recovery_score", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            geometric_recovery_fraction=("geometric_recovery_like", "mean"),
            proliferative_probability_median=("proliferative_probability", "median"),
            proliferative_like_fraction=("proliferative_like", "mean"),
            closer_to_proliferative_fraction=("closer_to_proliferative", "mean"),
            orthogonal_distance_median=("orthogonal_distance", "median"),
            orthogonal_distance_q90=("orthogonal_distance", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )

    _output_path = RESULTS_DIR / "well_level_recovery_summary.csv"
    _status = write_csv_protected(well_recovery_df, _output_path, overwrite=CONFIG.overwrite_existing_outputs)
    print(f"✓ Well-level recovery summary {_status} ({len(well_recovery_df)} wells).")
    print(f"  File: {_output_path}")
    return (well_recovery_df,)


@app.cell
def _(
    CONC_COL,
    CONFIG,
    RESULTS_DIR,
    TREATMENT_COL,
    WELL_COL,
    well_recovery_df,
    print,
    write_csv_protected,
):
    _condition_columns = [TREATMENT_COL] + ([CONC_COL] if CONC_COL and CONC_COL in well_recovery_df.columns else [])
    condition_recovery_df = (
        well_recovery_df.groupby(_condition_columns, observed=True)
        .agg(
            n_wells=(WELL_COL, "size"),
            recovery_median_of_wells=("recovery_median", "median"),
            recovery_sd_across_wells=("recovery_median", "std"),
            proliferative_probability_median_of_wells=("proliferative_probability_median", "median"),
            proliferative_like_fraction_median=("proliferative_like_fraction", "median"),
            orthogonal_distance_median_of_wells=("orthogonal_distance_median", "median"),
        )
        .reset_index()
    )
    _output_path = RESULTS_DIR / "condition_level_recovery_summary.csv"
    _status = write_csv_protected(condition_recovery_df, _output_path, overwrite=CONFIG.overwrite_existing_outputs)
    print(f"✓ Condition-level recovery summary {_status} ({len(condition_recovery_df)} conditions).")
    print(f"  File: {_output_path}")
    print(condition_recovery_df.to_string(index=False))
    return (condition_recovery_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 — Recovery versus alternative phenotype

    Conditions near the proliferative end of the axis with low orthogonal
    distance are the strongest recovery-like candidates. Conditions with
    high orthogonal distance may be strongly perturbed but biologically
    distinct from proliferation.
    """)
    return


@app.cell
def _(CONFIG, FIGS_DIR, TREATMENT_COL, plt, print, save_figure_protected, well_recovery_df):
    _fig, _ax = plt.subplots(figsize=(9, 7))
    for _treatment, _sub in well_recovery_df.groupby(TREATMENT_COL, observed=True):
        _ax.scatter(_sub["recovery_median"], _sub["orthogonal_distance_median"], s=65, alpha=0.8, label=str(_treatment))
    _ax.axvline(0, ls="--", lw=1)
    _ax.axvline(1, ls="--", lw=1)
    _ax.set_xlabel("Median recovery score per well")
    _ax.set_ylabel("Median orthogonal distance per well")
    _ax.set_title("Recovery-like displacement versus alternative phenotype")
    _ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _fig.tight_layout()
    _figure_path = FIGS_DIR / "recovery_parallel_vs_orthogonal.png"
    _figure_status = save_figure_protected(
        _fig, _figure_path, overwrite=CONFIG.overwrite_existing_outputs,
        dpi=180, bbox_inches="tight",
    )
    plt.close(_fig)
    print(f"✓ Recovery/orthogonal figure {_figure_status}: {_figure_path}")
    _fig
    return


@app.cell
def _(
    CONFIG,
    FIGS_DIR,
    PROLIFERATIVE_PROBABILITY_THRESHOLD,
    TREATMENT_COL,
    np,
    plt,
    print,
    save_figure_protected,
    well_recovery_df,
):
    _order = well_recovery_df[TREATMENT_COL].astype(str).drop_duplicates().tolist()
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
    _metrics = [
        ("recovery_median", "Median recovery score"),
        ("proliferative_probability_median", "Median proliferative probability"),
        ("proliferative_like_fraction", "Fraction proliferative-like"),
    ]
    for _ax, (_metric, _ylabel) in zip(_axes, _metrics):
        for _pos, _t in enumerate(_order):
            _vals = well_recovery_df.loc[well_recovery_df[TREATMENT_COL].astype(str).eq(_t), _metric]
            _jitter = np.linspace(-0.08, 0.08, max(len(_vals), 1))
            _ax.scatter(_pos + _jitter[: len(_vals)], _vals, s=45, alpha=0.8)
        _ax.set_xticks(range(len(_order)))
        _ax.set_xticklabels(_order, rotation=45, ha="right")
        _ax.set_ylabel(_ylabel)
    _axes[0].axhline(0, ls="--", lw=1)
    _axes[0].axhline(1, ls="--", lw=1)
    _axes[1].axhline(PROLIFERATIVE_PROBABILITY_THRESHOLD, ls="--", lw=1)
    _axes[2].set_ylim(-0.02, 1.02)
    _fig.suptitle("Well-level phenotypic recovery metrics", fontweight="bold")
    _fig.tight_layout()
    _figure_path = FIGS_DIR / "well_level_recovery_overview.png"
    _figure_status = save_figure_protected(
        _fig, _figure_path, overwrite=CONFIG.overwrite_existing_outputs,
        dpi=180, bbox_inches="tight",
    )
    plt.close(_fig)
    print(f"✓ Well-level overview {_figure_status}: {_figure_path}")
    _fig
    return


@app.cell
def _(
    BASELINE_LABEL,
    CONC_COL,
    CONFIG,
    FIGS_DIR,
    PROLIFERATIVE_LABEL,
    TREATMENT_COL,
    pd,
    plt,
    print,
    save_figure_protected,
    well_recovery_df,
):
    if CONC_COL and CONC_COL in well_recovery_df.columns:
        _dose_df = well_recovery_df.loc[~well_recovery_df[TREATMENT_COL].isin([BASELINE_LABEL, PROLIFERATIVE_LABEL])].copy()
        _dose_df[CONC_COL] = pd.to_numeric(_dose_df[CONC_COL], errors="coerce")
        _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
        _metrics = [
            ("recovery_median", "Median recovery score"),
            ("proliferative_like_fraction", "Proliferative-like fraction"),
            ("orthogonal_distance_median", "Median orthogonal distance"),
        ]
        for _treatment, _sub in _dose_df.groupby(TREATMENT_COL, observed=True):
            _summary = (
                _sub.groupby(CONC_COL, observed=True)
                .agg(
                    recovery_median=("recovery_median", "median"),
                    proliferative_like_fraction=("proliferative_like_fraction", "median"),
                    orthogonal_distance_median=("orthogonal_distance_median", "median"),
                )
                .reset_index()
                .sort_values(CONC_COL)
            )
            for _ax, (_metric, _ylabel) in zip(_axes, _metrics):
                _ax.plot(_summary[CONC_COL], _summary[_metric], marker="o", lw=1.5, label=str(_treatment))
                _ax.set_xlabel("Concentration")
                _ax.set_ylabel(_ylabel)
        _axes[0].axhline(0, ls="--", lw=1)
        _axes[0].axhline(1, ls="--", lw=1)
        _axes[1].set_ylim(-0.02, 1.02)
        _axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        _fig.suptitle("Dose-dependent phenotypic recovery", fontweight="bold")
        _fig.tight_layout()
        _figure_path = FIGS_DIR / "dose_response_recovery.png"
        _figure_status = save_figure_protected(
            _fig, _figure_path, overwrite=CONFIG.overwrite_existing_outputs,
            dpi=180, bbox_inches="tight",
        )
        plt.close(_fig)
        print(f"✓ Dose-response figure {_figure_status}: {_figure_path}")
        _display = _fig
    else:
        print("Concentration metadata unavailable; dose-response plots skipped.")
        _display = None
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 — Homogeneous recovery or proliferative-like subpopulation?

    Condition means can conceal different single-cell response patterns.
    Compare density and ECDF plots to distinguish a homogeneous partial
    shift from enrichment of a proliferative-like minority.
    """)
    return


@app.cell
def _(CONFIG, FIGS_DIR, TREATMENT_COL, cell_recovery_df, np, plt, print, save_figure_protected):
    _conditions = cell_recovery_df[TREATMENT_COL].astype(str).drop_duplicates().tolist()
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for _condition in _conditions:
        _values = (
            cell_recovery_df.loc[cell_recovery_df[TREATMENT_COL].astype(str).eq(_condition), "recovery_score"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values()
            .to_numpy()
        )
        if len(_values) == 0:
            continue
        _axes[0].hist(np.clip(_values, -2, 3), bins=80, density=True, histtype="step", lw=1.3, label=_condition)
        _axes[1].plot(_values, np.arange(1, len(_values) + 1) / len(_values), lw=1.3, label=_condition)
    for _ax in _axes:
        _ax.axvline(0, ls="--", lw=1)
        _ax.axvline(1, ls="--", lw=1)
    _axes[0].set_xlabel("Recovery score")
    _axes[0].set_ylabel("Density")
    _axes[0].set_title("Single-cell recovery distributions")
    _axes[1].set_xlabel("Recovery score")
    _axes[1].set_ylabel("Empirical cumulative probability")
    _axes[1].set_title("Recovery-score ECDF")
    _axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _fig.tight_layout()
    _figure_path = FIGS_DIR / "single_cell_recovery_distributions.png"
    _figure_status = save_figure_protected(
        _fig, _figure_path, overwrite=CONFIG.overwrite_existing_outputs,
        dpi=180, bbox_inches="tight",
    )
    plt.close(_fig)
    print(f"✓ Single-cell distribution figure {_figure_status}: {_figure_path}")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11 — Well-level inference

    A plate-adjusted linear model on well-level median recovery score.
    Choose a model appropriate to the experimental design (mixed model,
    permutation test, hierarchical bootstrap, ordered dose-trend test are
    all reasonable alternatives) — the number of cells is never the
    inferential sample size, wells are.
    """)
    return


@app.cell
def _(CONC_COL, PLATE_COL, TREATMENT_COL, pd, print, smf, well_recovery_df):
    model_df = well_recovery_df.copy()
    if CONC_COL and CONC_COL in model_df.columns:
        model_df[CONC_COL] = pd.to_numeric(model_df[CONC_COL], errors="coerce")

    _plate_term = f" + C({PLATE_COL})" if model_df[PLATE_COL].nunique() > 1 else ""
    recovery_model_formula = f"recovery_median ~ C({TREATMENT_COL}){_plate_term}"
    recovery_model = smf.ols(formula=recovery_model_formula, data=model_df).fit(cov_type="HC3")
    print(recovery_model.summary())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12 — Feature-level interpretation

    The recovery direction is projected back to original features. These
    weights describe the baseline-to-proliferative reference transition and
    are summarized by feature family / compartment / channel using the
    shared `hca_pipeline.taxonomy` classifier (the same one
    `05_phenotypic_fingerprints.py` uses), rather than a notebook-local
    reimplementation.
    """)
    return


@app.cell
def _(
    CONFIG,
    N_RECOVERY_PCS,
    RESULTS_DIR,
    build_taxonomy_table,
    feature_cols_model,
    np,
    pca_sc,
    pd,
    unit_vector,
    print,
    write_csv_protected,
):
    _n_pcs_used = min(N_RECOVERY_PCS, pca_sc.components_.shape[0], unit_vector.size)
    _components = pca_sc.components_[:_n_pcs_used, :]
    _axis_vector = unit_vector[:_n_pcs_used]

    if _components.shape[1] != len(feature_cols_model):
        raise ValueError(
            "PCA component width does not match feature_cols_model.\n"
            f"PCA width: {_components.shape[1]:,}\n"
            f"Features : {len(feature_cols_model):,}"
        )

    _weights = _axis_vector @ _components
    recovery_feature_weights_df = (
        pd.DataFrame(
            {
                "feature": feature_cols_model,
                "recovery_axis_weight": _weights,
                "absolute_weight": np.abs(_weights),
            }
        )
        .sort_values("absolute_weight", ascending=False)
        .reset_index(drop=True)
    )

    _taxonomy_df = build_taxonomy_table(recovery_feature_weights_df["feature"].tolist(), channels=CONFIG.channels)
    recovery_feature_weights_df = recovery_feature_weights_df.merge(_taxonomy_df, on="feature", how="left")

    _output_path = RESULTS_DIR / "recovery_axis_feature_weights.csv"
    _status = write_csv_protected(recovery_feature_weights_df, _output_path, overwrite=CONFIG.overwrite_existing_outputs)
    print(f"✓ Recovery-axis feature weights {_status}")
    print(f"  File: {_output_path}")
    print(recovery_feature_weights_df.head(30).to_string(index=False))
    return (recovery_feature_weights_df,)


@app.cell
def _(CONFIG, RESULTS_DIR, print, recovery_feature_weights_df, write_csv_protected):
    family_weights_df = (
        recovery_feature_weights_df.groupby("feature_family", as_index=False)
        .agg(
            n_features=("feature", "size"),
            total_absolute_weight=("absolute_weight", "sum"),
            mean_absolute_weight=("absolute_weight", "mean"),
            maximum_absolute_weight=("absolute_weight", "max"),
            signed_weight=("recovery_axis_weight", "sum"),
        )
        .sort_values("total_absolute_weight", ascending=False)
        .reset_index(drop=True)
    )
    family_weights_df["weight_fraction"] = (
        family_weights_df["total_absolute_weight"] / family_weights_df["total_absolute_weight"].sum()
    )

    _output_path = RESULTS_DIR / "recovery_axis_feature_family_weights.csv"
    _status = write_csv_protected(family_weights_df, _output_path, overwrite=CONFIG.overwrite_existing_outputs)
    print(f"✓ Feature-family composition {_status}")
    print(f"  File: {_output_path}")
    print(family_weights_df.to_string(index=False))
    return (family_weights_df,)


@app.cell
def _(pd, print, recovery_feature_weights_df):
    _moment_mask = recovery_feature_weights_df["feature"].str.contains(
        r"HuMoment|CentralMoment|NormalizedMoment", regex=True, na=False
    )
    moment_summary = pd.Series(
        {
            "moment_features": int(_moment_mask.sum()),
            "total_features": len(recovery_feature_weights_df),
            "fraction_of_features": float(_moment_mask.mean()),
            "fraction_of_absolute_axis_weight": float(
                recovery_feature_weights_df.loc[_moment_mask, "absolute_weight"].sum()
                / recovery_feature_weights_df["absolute_weight"].sum()
            ),
            "fraction_of_top_30": float(
                recovery_feature_weights_df.head(30)["feature"]
                .str.contains(r"HuMoment|CentralMoment|NormalizedMoment", regex=True, na=False)
                .mean()
            ),
        },
        name="value",
    )
    print(moment_summary.to_frame().to_string())
    return (moment_summary,)


@app.cell
def _(family_weights_df, mo, moment_summary):
    _top_family = family_weights_df.iloc[0]
    mo.md(
        f"""
        ### Recovery-axis feature composition

        The phenotypic transition from the baseline reference to the
        proliferative reference is primarily supported by
        **{_top_family['feature_family']}** features, which account for
        approximately **{_top_family['weight_fraction']:.0%}** of the total
        absolute recovery-axis weight (n = {int(_top_family['n_features']):,}
        features).

        Moment-based shape descriptors make up
        **{moment_summary['fraction_of_features']:.0%}** of the available
        features but **{moment_summary['fraction_of_absolute_axis_weight']:.0%}**
        of the total absolute axis weight and
        **{moment_summary['fraction_of_top_30']:.0%}** of the top 30
        features by weight.

        Because texture and shape families often contain many correlated
        measurements calculated across multiple scales and orientations,
        summed weights should not be interpreted as independent biological
        contributions. This composition describes the geometry of the
        baseline-to-proliferative transition only; feature-level
        differences and grouped biological signatures (see the
        `biological_signature` / `channel` columns in
        `recovery_axis_feature_weights.csv`) are required to determine the
        direction and biological meaning of the underlying changes.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation guide

    | Recovery score | Orthogonal distance | Interpretation |
    |---|---:|---|
    | Near 0 | Low | Baseline-like |
    | Between 0 and 1 | Low | Partial recovery-like transition |
    | Near 1 | Low | Proliferative-like recovery |
    | Above 1 | Low | Transition beyond the proliferative centroid |
    | Any value | High | Alternative phenotype not captured by the reference transition |
    | Mixed single-cell distribution | Variable | Possible proliferative-like subpopulation rather than homogeneous recovery |

    Concordance between the geometric score and probabilistic classifier
    strengthens the interpretation. Discordance should be investigated
    rather than averaged away.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13 — Provenance
    """)
    return


@app.cell
def _(
    BASELINE_LABEL,
    CONFIG,
    EXPERIMENT_ID,
    FIGS_DIR,
    N_PCA_COMPONENTS,
    N_PER_CLASS,
    N_RECOVERY_PCS,
    PROLIFERATIVE_LABEL,
    PROLIFERATIVE_PROBABILITY_THRESHOLD,
    RANDOM_STATE,
    RECOVERY_SCORE_THRESHOLD,
    REPO_ROOT,
    RESULTS_DIR,
    Sequence,
    cell_recovery_df,
    cv_metrics_df,
    datetime,
    json,
    np,
    pd,
    platform,
    print,
    subprocess,
    timezone,
    well_recovery_df,
):
    NOTEBOOK_NAME = "extras/recovery_axis_analysis.py"
    from hca_pipeline.provenance import canonicalize_provenance, provenance_json

    def run_git_command(arguments: Sequence[str], repo_root) -> str | None:
        """Run a read-only Git command and return stripped stdout."""
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=str(repo_root), capture_output=True, text=True, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    git_commit_full = run_git_command(["rev-parse", "HEAD"], REPO_ROOT)
    git_branch = run_git_command(["branch", "--show-current"], REPO_ROOT) or "unknown"
    git_status = run_git_command(["status", "--porcelain"], REPO_ROOT)

    provenance = {
        "schema_version": 2,
        "pipeline": {
            "notebook": NOTEBOOK_NAME,
            "experiment_id": EXPERIMENT_ID,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "optional_extension": True,
            "requires_two_reference_states": True,
        },
        "configuration": {
            "baseline_label": BASELINE_LABEL,
            "proliferative_label": PROLIFERATIVE_LABEL,
            "random_state": int(RANDOM_STATE),
            "n_per_class": int(N_PER_CLASS),
            "n_pca_components": int(N_PCA_COMPONENTS),
            "n_recovery_pcs": int(N_RECOVERY_PCS),
            "proliferative_probability_threshold": float(PROLIFERATIVE_PROBABILITY_THRESHOLD),
            "recovery_score_threshold": float(RECOVERY_SCORE_THRESHOLD),
            "overwrite_existing_outputs": bool(CONFIG.overwrite_existing_outputs),
        },
        "dataset": {
            "n_cells_scored": int(len(cell_recovery_df)),
            "n_wells_summarized": int(len(well_recovery_df)),
            "cv_balanced_accuracy_mean": float(cv_metrics_df["balanced_accuracy"].mean()),
            "cv_roc_auc_mean": float(cv_metrics_df["roc_auc"].mean()),
        },
        "version_control": {
            "git_commit": git_commit_full or "unknown",
            "git_commit_short": git_commit_full[:8] if git_commit_full else "unknown",
            "git_branch": git_branch,
            "working_tree_dirty": bool(git_status) if git_status is not None else None,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
        "outputs": {
            "single_cell_recovery_scores_parquet": str(RESULTS_DIR / "single_cell_recovery_scores.parquet"),
            "well_level_recovery_summary_csv": str(RESULTS_DIR / "well_level_recovery_summary.csv"),
            "condition_level_recovery_summary_csv": str(RESULTS_DIR / "condition_level_recovery_summary.csv"),
            "recovery_axis_feature_weights_csv": str(RESULTS_DIR / "recovery_axis_feature_weights.csv"),
            "recovery_axis_feature_family_weights_csv": str(RESULTS_DIR / "recovery_axis_feature_family_weights.csv"),
            "figure_directory": str(FIGS_DIR),
        },
    }

    _declared_outputs = [
        RESULTS_DIR / "single_cell_recovery_scores.parquet",
        RESULTS_DIR / "well_level_recovery_summary.csv",
        RESULTS_DIR / "condition_level_recovery_summary.csv",
        RESULTS_DIR / "recovery_axis_feature_weights.csv",
        RESULTS_DIR / "recovery_axis_feature_family_weights.csv",
    ]
    provenance = canonicalize_provenance(
        provenance,
        notebook=NOTEBOOK_NAME,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
        dependencies=[INPUT_PARQUET],
        outputs=_declared_outputs,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    provenance_latest_path = RESULTS_DIR / "provenance_recovery_axis_latest.json"
    _provenance_payload = provenance_json(provenance)
    if provenance_latest_path.exists() and not CONFIG.overwrite_existing_outputs:
        _latest_status = "unchanged (existing provenance protected)"
    else:
        _latest_status = "replaced" if provenance_latest_path.exists() else "created"
        provenance_latest_path.write_text(_provenance_payload, encoding="utf-8")

    provenance_history_path = None
    if CONFIG.save_provenance_history:
        _timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        provenance_history_path = RESULTS_DIR / f"provenance_recovery_axis_{_timestamp}.json"
        provenance_history_path.write_text(_provenance_payload, encoding="utf-8")

    print("═" * 72)
    print("RECOVERY-AXIS EXTRA PROVENANCE")
    print("═" * 72)
    print(f"  Notebook:            {NOTEBOOK_NAME}")
    print(f"  Experiment:          {EXPERIMENT_ID}")
    print(f"  Baseline reference:  {BASELINE_LABEL}")
    print(f"  Proliferative ref.:  {PROLIFERATIVE_LABEL}")
    print(f"  Git commit:          {provenance['version_control']['git_commit_short']}")
    print(f"✓ Latest provenance:   {_latest_status} — {provenance_latest_path}")
    print(f"  Schema:              v{provenance['schema_version']}")
    if provenance_history_path is not None:
        print(f"✓ Historical record:  {provenance_history_path}")
    return (provenance_latest_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 14 — Final integrity checks and execution summary
    """)
    return


@app.cell
def _(
    BASELINE_LABEL,
    CONC_COL,
    EXPERIMENT_ID,
    FIGS_DIR,
    PROLIFERATIVE_LABEL,
    Path,
    RESULTS_DIR,
    cell_recovery_df,
    condition_recovery_df,
    cv_metrics_df,
    family_weights_df,
    print,
    provenance_latest_path,
    recovery_feature_weights_df,
    well_recovery_df,
):
    _required_outputs = [
        RESULTS_DIR / "single_cell_recovery_scores.parquet",
        RESULTS_DIR / "well_level_recovery_summary.csv",
        RESULTS_DIR / "condition_level_recovery_summary.csv",
        RESULTS_DIR / "recovery_axis_feature_weights.csv",
        RESULTS_DIR / "recovery_axis_feature_family_weights.csv",
        provenance_latest_path,
        FIGS_DIR / "reference_geometric_validation.png",
        FIGS_DIR / "reference_classifier_validation.png",
        FIGS_DIR / "recovery_parallel_vs_orthogonal.png",
        FIGS_DIR / "well_level_recovery_overview.png",
        FIGS_DIR / "single_cell_recovery_distributions.png",
    ]
    if CONC_COL and CONC_COL in cell_recovery_df.columns:
        _required_outputs.append(FIGS_DIR / "dose_response_recovery.png")
    _missing_outputs = [p for p in _required_outputs if not Path(p).exists()]
    if _missing_outputs:
        raise RuntimeError(
            "Recovery-axis extra integrity check failed — required output files are missing: "
            + ", ".join(str(p) for p in _missing_outputs)
        )

    print("═" * 72)
    print("RECOVERY-AXIS EXTRA COMPLETED")
    print("═" * 72)
    print(f"  Experiment:              {EXPERIMENT_ID}")
    print(f"  Baseline reference:      {BASELINE_LABEL}")
    print(f"  Proliferative reference: {PROLIFERATIVE_LABEL}")
    print(f"  Single cells scored:     {len(cell_recovery_df):,}")
    print(f"  Wells summarized:        {len(well_recovery_df):,}")
    print(f"  Conditions summarized:   {len(condition_recovery_df):,}")
    print(f"  Classifier CV balanced accuracy: {cv_metrics_df['balanced_accuracy'].mean():.3f}")
    print(f"  Classifier CV ROC AUC:           {cv_metrics_df['roc_auc'].mean():.3f}")
    print(f"  Recovery-axis features ranked:   {len(recovery_feature_weights_df):,}")
    print(f"  Feature families:                {len(family_weights_df):,}")
    print("\n  Results directory:")
    print(f"    {RESULTS_DIR}")
    print("  Figures directory:")
    print(f"    {FIGS_DIR}")
    print("\n  Provenance:")
    print(f"    {provenance_latest_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Save the analysis record

    Save the notebook's **current session** without rerunning cells. HTML
    preserves rich outputs and expanded details; PDF provides a clean reading
    copy without code. Chromium can save both directly to the experiment's
    `reports/` folder after authorization. Safari provides separate downloads.
    """)
    return


@app.cell
def _(EXPERIMENT_ID, mo):
    from hca_pipeline.report_export import SessionReportSaver

    _report_saver = SessionReportSaver(
        basename=f"{EXPERIMENT_ID}_extra_recovery_axis_analysis",
        suggested_directory=f"workspace/analysis/{EXPERIMENT_ID}/reports",
    )
    mo.ui.anywidget(_report_saver)
    return


if __name__ == "__main__":
    app.run()
