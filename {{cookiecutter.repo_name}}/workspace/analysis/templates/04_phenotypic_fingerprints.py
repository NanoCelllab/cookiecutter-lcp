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
    # 04 — Phenotypic Fingerprints for Live Cell Painting

    **Pipeline step:** 4 of 6
    **Purpose:** turn per-well normalized CellProfiler features into
    biologically interpretable phenotypic fingerprints — per-feature effect
    sizes, a feature taxonomy, and fingerprint matrices/plots at multiple
    grouping levels.

    Inspired by the grouped Cell Painting radar-plot workflow developed by
    Jonne, adapted here to the LCP (Acridine Orange GFP/PI ± Hoechst) assay.

    ## Questions addressed

    - Which feature families are most strongly altered?
    - Which cellular compartments carry the phenotype?
    - Which fluorescence channels contribute most strongly?
    - How do channel × feature-family and compartment × feature-family
      fingerprints change across doses?
    - Are the effects broad or driven by a small number of extreme features?

    ## Important conceptual distinction

    Fingerprints are calculated from the **normalized CellProfiler
    features**. Harmony-corrected PCA coordinates are not used here because
    Harmony components do not map directly to interpretable categories such
    as `Intensity`, `Texture`, `AreaShape`, `AOGFP`, or `Nuclei`.

    ## Changes relative to the legacy `.ipynb` version

    This marimo notebook fixes two issues in the original notebook rather
    than porting it verbatim:

    1. **Graceful no-dose-axis path.** The legacy notebook unconditionally
       assumed `Metadata_Concentration` exists (conditions = Treatment ×
       Concentration; dose-overlay radar plots). If that column is absent it
       crashed. This version checks `CONFIG.resolve_columns(df).has_dose_axis`
       (the same pattern already used by `hca_pipeline.metrics_qc.run_dose_response`
       for NB05) and, when there is no dose axis, builds conditions from
       treatment alone and skips only the dose-overlay radar section with a
       clear printed message — everything else still runs.
    2. **`SPECIAL_TREATMENTS` is a widget, not a hardcoded literal.** The
       legacy notebook hardcoded `{"Non-dormant"}` — a treatment name specific
       to this study — to route certain conditions into a separate panel
       instead of the dose grid. Here it is a `mo.ui.multiselect` populated
       from this experiment's actual treatment values, defaulting to *empty*,
       so a different dataset isn't stuck with a meaningless default.

    The legacy notebook's final "BONUS: Cell abundance and object counts"
    section (~14 cells) was **dropped**, not ported: it referenced a
    `feat_cols` name never defined in that notebook, used
    `phenotypic_score_df` before it was defined, and had a hand-edited
    "replace me" placeholder column name — it never actually ran end to end.

    ## Main outputs

    - per-feature effect-size table;
    - feature taxonomy and taxonomy-coverage diagnostics;
    - fingerprint matrices (CSV) for 6 grouping levels × 4 metrics;
    - heatmaps and small-multiple / dose-overlay radar plots;
    - a top-altered-features-per-condition table;
    - provenance JSON.
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
    from dataclasses import replace
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Sequence

    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_colwidth", 100)

    FIGURE_DPI = 180
    RADAR_MAX_COLUMNS = 4
    return (
        FIGURE_DPI,
        Path,
        RADAR_MAX_COLUMNS,
        Sequence,
        datetime,
        json,
        np,
        pd,
        platform,
        replace,
        subprocess,
        timezone,
    )


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
    from hca_pipeline.io import write_csv_protected, write_summary_table_protected
    from hca_pipeline.plotting import (
        plot_condition_radar_grid,
        plot_dose_overlay_radars,
        plot_fingerprint_heatmap,
    )
    from hca_pipeline.stats import (
        build_fingerprint_matrix,
        calculate_global_effects,
        calculate_within_plate_effects,
    )
    from hca_pipeline.taxonomy import build_taxonomy_table

    print(f"  ✓  Shared utilities loaded from hca_pipeline ({_pipelines_dir})")
    return (
        ExperimentConfig,
        REPO_ROOT,
        build_fingerprint_matrix,
        build_taxonomy_table,
        calculate_global_effects,
        calculate_within_plate_effects,
        infer_feature_cols,
        plot_condition_radar_grid,
        plot_dose_overlay_radars,
        plot_fingerprint_heatmap,
        write_csv_protected,
        write_summary_table_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Pick the experiment folder and confirm the analysis-choice parameters
    below. Defaults for the experiment-design fields (channels, overwrite
    policy) are pre-filled from a previously saved `experiment_config.json`
    when one exists. The effect-size and grouping parameters are algorithm
    choices for *this* notebook, not experiment-design facts, so they are
    plain widgets and are not written back to `experiment_config.json`.
    """)
    return


@app.cell
def _(REPO_ROOT, mo):
    _backend_dir = REPO_ROOT / "workspace" / "backend"
    available_experiment_ids = (
        sorted(p.name for p in _backend_dir.iterdir() if p.is_dir())
        if _backend_dir.is_dir()
        else []
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
    _all_grouping_levels = [
        "feature_family",
        "compartment",
        "channel",
        "compartment_x_family",
        "channel_x_family",
        "biological_signature",
    ]

    channels_input = mo.ui.multiselect(
        options=["GFP", "PI", "Hoechst", "DAPI", "Brightfield"],
        value=loaded_config.channels or ["GFP", "PI"],
        label="Imaging channels used (drives the taxonomy's channel axis)",
    )
    effect_mode_input = mo.ui.dropdown(
        options=["within_plate_weighted", "global"],
        value="within_plate_weighted",
        label="Effect-size estimation mode",
    )
    min_samples_input = mo.ui.number(
        value=2, start=2, stop=1000, step=1, label="Minimum samples per group (Cohen's d)"
    )
    effect_threshold_input = mo.ui.number(
        value=0.8,
        start=0.0,
        stop=10.0,
        step=0.05,
        label="Effect-size threshold (|d| considered 'altered')",
    )
    grouping_levels_input = mo.ui.multiselect(
        options=_all_grouping_levels,
        value=_all_grouping_levels,
        label="Fingerprint grouping levels to compute",
    )
    radar_groupings_input = mo.ui.multiselect(
        options=_all_grouping_levels,
        value=_all_grouping_levels,
        label="Grouping levels to render as radar plots",
    )
    drop_other_input = mo.ui.checkbox(
        value=True, label="Drop the 'Other' taxonomy category from figures"
    )
    top_features_input = mo.ui.number(
        value=30, start=1, stop=500, step=1, label="Top altered features to keep per condition"
    )
    overwrite_input = mo.ui.checkbox(
        value=loaded_config.overwrite_existing_outputs, label="Overwrite existing outputs"
    )
    save_history_input = mo.ui.checkbox(
        value=loaded_config.save_provenance_history, label="Save timestamped provenance history"
    )

    mo.vstack(
        [
            channels_input,
            effect_mode_input,
            min_samples_input,
            effect_threshold_input,
            grouping_levels_input,
            radar_groupings_input,
            drop_other_input,
            top_features_input,
            overwrite_input,
            save_history_input,
        ]
    )
    return (
        channels_input,
        drop_other_input,
        effect_mode_input,
        effect_threshold_input,
        grouping_levels_input,
        min_samples_input,
        overwrite_input,
        radar_groupings_input,
        save_history_input,
        top_features_input,
    )


@app.cell
def _(
    REPO_ROOT,
    channels_input,
    drop_other_input,
    effect_mode_input,
    effect_threshold_input,
    experiment_id_input,
    grouping_levels_input,
    loaded_config,
    min_samples_input,
    overwrite_input,
    radar_groupings_input,
    replace,
    save_history_input,
    top_features_input,
):
    EXPERIMENT_ID = experiment_id_input.value

    CONFIG = replace(
        loaded_config,
        experiment_id=EXPERIMENT_ID,
        channels=list(channels_input.value),
        overwrite_existing_outputs=bool(overwrite_input.value),
        save_provenance_history=bool(save_history_input.value),
    )
    _config_path = CONFIG.save(REPO_ROOT)

    EFFECT_ESTIMATION_MODE = effect_mode_input.value
    MIN_SAMPLES_PER_GROUP = int(min_samples_input.value)
    EFFECT_THRESHOLD = float(effect_threshold_input.value)
    GROUPING_LEVELS = list(grouping_levels_input.value)
    RADAR_GROUPINGS = list(radar_groupings_input.value)
    DROP_OTHER_FROM_FIGURES = bool(drop_other_input.value)
    TOP_FEATURES_PER_CONDITION = int(top_features_input.value)

    if not GROUPING_LEVELS:
        raise ValueError("Select at least one grouping level to analyze.")

    print("═" * 72)
    print("CONFIGURATION VALIDATED")
    print("═" * 72)
    print(f"  Experiment ID:          {EXPERIMENT_ID}")
    print(f"  Channels:               {', '.join(CONFIG.channels) or '(none selected)'}")
    print(f"  Effect estimation mode: {EFFECT_ESTIMATION_MODE}")
    print(f"  Effect threshold:       {EFFECT_THRESHOLD:g}")
    print(f"  Min samples per group:  {MIN_SAMPLES_PER_GROUP}")
    print(f"  Grouping levels:        {GROUPING_LEVELS}")
    print(f"  Radar groupings:        {RADAR_GROUPINGS}")
    print(f"  Drop 'Other' in figs:   {DROP_OTHER_FROM_FIGURES}")
    print(f"  Top features/condition: {TOP_FEATURES_PER_CONDITION}")
    print(f"  Repository root:        {REPO_ROOT}")
    print(f"  Saved config:           {_config_path}")
    return (
        CONFIG,
        DROP_OTHER_FROM_FIGURES,
        EFFECT_ESTIMATION_MODE,
        EFFECT_THRESHOLD,
        EXPERIMENT_ID,
        GROUPING_LEVELS,
        MIN_SAMPLES_PER_GROUP,
        RADAR_GROUPINGS,
        TOP_FEATURES_PER_CONDITION,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Input and output directories

    Missing directories are created automatically; existing directories are
    reused. Existing legacy outputs elsewhere in the project are not moved
    or deleted.
    """)
    return


@app.cell
def _(EXPERIMENT_ID, REPO_ROOT):
    WORKSPACE_DIR = REPO_ROOT / "workspace"
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    PROFILE_OUTPUT_DIR = WORKSPACE_DIR / "profiles" / EXPERIMENT_ID / "outputs"

    INPUT_PARQUET = PROFILE_OUTPUT_DIR / "per_well_features_selected.parquet"

    RESULTS_DIR = ANALYSIS_DIR / "results" / "phenotypic_fingerprints"
    FIGURES_DIR = ANALYSIS_DIR / "figures" / "phenotypic_fingerprints"
    TABLES_DIR = RESULTS_DIR / "tables"
    TAXONOMY_DIR = RESULTS_DIR / "taxonomy"
    PROVENANCE_DIR = RESULTS_DIR / "provenance"

    for _directory in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR, TAXONOMY_DIR, PROVENANCE_DIR):
        _directory.mkdir(parents=True, exist_ok=True)

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input profile not found.\nExpected path: {INPUT_PARQUET}")

    print(f"Input profile : {INPUT_PARQUET}")
    print(f"Results       : {RESULTS_DIR}")
    print(f"Figures       : {FIGURES_DIR}")
    return (
        FIGURES_DIR,
        INPUT_PARQUET,
        PROVENANCE_DIR,
        RESULTS_DIR,
        TABLES_DIR,
        TAXONOMY_DIR,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Load and validate the normalized profile
    """)
    return


@app.cell
def _(INPUT_PARQUET, pd):
    df_loaded = pd.read_parquet(INPUT_PARQUET)

    _unnamed_columns = [c for c in df_loaded.columns if str(c).startswith("Unnamed:")]
    if _unnamed_columns:
        df_loaded = df_loaded.drop(columns=_unnamed_columns, errors="ignore")

    if df_loaded.empty:
        raise ValueError("The input profile contains no wells.")

    print(f"Loaded profile: {df_loaded.shape[0]:,} wells × {df_loaded.shape[1]:,} columns")
    return (df_loaded,)


@app.cell
def _(CONFIG, df_loaded):
    # This is the fix for the legacy notebook's unconditional dose-axis
    # assumption: CONFIG.resolve_columns(df) is the same pattern already used
    # by hca_pipeline.metrics_qc.run_dose_response for NB05. has_dose_axis is
    # False (rather than raising) whenever no concentration/dose column can be
    # resolved against the actual data.
    CONFIG_RESOLVED = CONFIG.resolve_columns(df_loaded)

    PLATE_COL = CONFIG_RESOLVED.plate_col
    WELL_COL = CONFIG_RESOLVED.well_col
    TREATMENT_COL = CONFIG_RESOLVED.treatment_col
    CONTROL_COL = CONFIG_RESOLVED.control_type_col
    CONC_COL = CONFIG_RESOLVED.concentration_col
    HAS_DOSE_AXIS = CONFIG_RESOLVED.has_dose_axis

    required_columns = {
        "plate": PLATE_COL,
        "well": WELL_COL,
        "treatment": TREATMENT_COL,
        "control type": CONTROL_COL,
    }
    missing_resolved = [
        name
        for name, column in required_columns.items()
        if column is None or column not in df_loaded.columns
    ]
    if missing_resolved:
        raise ValueError(
            "Required metadata columns could not be resolved: "
            f"{missing_resolved}. Available Metadata_* columns: "
            f"{[c for c in df_loaded.columns if c.startswith('Metadata_')]}"
        )

    print("Resolved metadata columns:")
    for _name, _column in required_columns.items():
        print(f"  {_name:14s}: {_column}")

    if HAS_DOSE_AXIS:
        print(f"  {'concentration':14s}: {CONC_COL}")
        print("\n✓ Dose/concentration axis detected — conditions will be Treatment × Concentration.")
    else:
        print(
            "\n⚠ No concentration/dose column detected for this experiment "
            "(CONFIG.has_dose_axis is False). Conditions will be built from "
            "treatment alone, and the dose-overlay radar section will be "
            "skipped later — everything else still runs normally."
        )
    return (
        CONC_COL,
        CONFIG_RESOLVED,
        CONTROL_COL,
        HAS_DOSE_AXIS,
        PLATE_COL,
        TREATMENT_COL,
    )


@app.cell
def _(df_loaded, infer_feature_cols, np):
    feature_cols = infer_feature_cols(df_loaded)
    if not feature_cols:
        raise ValueError("No CellProfiler feature columns were inferred.")

    _feature_values = df_loaded[feature_cols].replace([np.inf, -np.inf], np.nan)
    _invalid_count = int(_feature_values.isna().sum().sum())
    if _invalid_count > 0:
        raise ValueError(
            "The input profile contains missing or infinite feature values. "
            f"Invalid values detected: {_invalid_count:,}"
        )

    print(f"Feature columns: {len(feature_cols):,}")
    return (feature_cols,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Build explicit experimental conditions

    The condition label is kept separate from the treatment and
    concentration columns to avoid losing dose-specific effects during
    Cohen's d calculation. When this experiment has no dose axis (see the
    warning above, if shown), conditions collapse to treatment alone instead
    of raising.
    """)
    return


@app.cell
def _(
    CONC_COL,
    CONFIG,
    CONTROL_COL,
    HAS_DOSE_AXIS,
    TREATMENT_COL,
    df_loaded,
    np,
):
    def format_concentration(value) -> str:
        """Format concentration values consistently for labels."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isnan(numeric):
            return "NA"
        return f"{numeric:g}"

    df = df_loaded.copy()

    if HAS_DOSE_AXIS:
        df["Metadata_Concentration_Sort"] = df[CONC_COL]
        df["Metadata_Concentration_Label"] = df[CONC_COL].map(format_concentration)
        df["Metadata_Condition"] = (
            df[TREATMENT_COL].astype(str) + " | " + df["Metadata_Concentration_Label"]
        )
    else:
        # Graceful no-dose-axis path (see the config-resolution cell above):
        # build conditions from treatment alone instead of crashing on a
        # missing concentration column.
        df["Metadata_Concentration_Sort"] = 0
        df["Metadata_Concentration_Label"] = "NA"
        df["Metadata_Condition"] = df[TREATMENT_COL].astype(str)

    CONDITION_CONC_COL = "Metadata_Concentration_Sort"

    _negcon_values_lower = {str(v).lower() for v in CONFIG.negcon_values}
    negative_control_mask = df[CONTROL_COL].astype(str).str.lower().isin(_negcon_values_lower)

    condition_table = (
        df.loc[
            ~negative_control_mask,
            [TREATMENT_COL, CONDITION_CONC_COL, "Metadata_Concentration_Label", "Metadata_Condition"],
        ]
        .drop_duplicates()
        .sort_values([TREATMENT_COL, CONDITION_CONC_COL], kind="stable")
        .reset_index(drop=True)
    )

    condition_order = condition_table["Metadata_Condition"].tolist()

    print(f"Negative-control wells: {int(negative_control_mask.sum()):,}")
    print(
        f"Conditions: {len(condition_order):,}"
        + (" (treatment × concentration)" if HAS_DOSE_AXIS else " (treatment only — no dose axis)")
    )

    condition_table
    return (
        CONDITION_CONC_COL,
        condition_order,
        condition_table,
        df,
        negative_control_mask,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — LCP feature taxonomy

    Each feature is parsed into several complementary labels using the
    shared `hca_pipeline.taxonomy` classifier: **feature family**
    (Intensity, Texture, AreaShape, Granularity, ...), **compartment**
    (Cells, Cytoplasm, Nuclei, Vesicles, ...), **channel** (driven by this
    experiment's configured channel list, e.g. `GFP`/`PI`), **compartment ×
    family**, **channel × family**, and a higher-level **biological
    signature**. The taxonomy is exported before fingerprints are calculated
    so that grouping decisions remain transparent and auditable.
    """)
    return


@app.cell
def _(
    CONFIG,
    GROUPING_LEVELS,
    TAXONOMY_DIR,
    build_taxonomy_table,
    feature_cols,
    write_csv_protected,
):
    # hca_pipeline.taxonomy.DEFAULT_COMPARTMENT_PREFIXES only distinguishes the
    # 4 top-level CellProfiler objects (Cells/Cytoplasm/Nuclei/Vesicles) — the
    # same coarse vocabulary infer_feature_cols uses to detect feature columns
    # at all. The legacy NB04 notebook additionally split out Nucleoli (a
    # child object of Nuclei) and disambiguated "Cells_Mean_Vesicles_" /
    # "Nuclei_Mean_Nucleoli_" child-object aggregate features from their
    # parent compartment. This is the standard LCP object hierarchy (not a
    # per-experiment design choice), so it is reconstructed here as an
    # explicit, more specific prefix list — order matters: the more specific
    # prefixes must come first so they win over the shorter, more general
    # ones.
    lcp_compartment_prefixes = (
        "Cells_Mean_Vesicles_",
        "Nuclei_Mean_Nucleoli_",
        "Vesicles_",
        "Nucleoli_",
        "Cytoplasm_",
        "Nuclei_",
        "Cells_",
    )

    feature_taxonomy = build_taxonomy_table(
        feature_cols,
        channels=CONFIG.channels,
        compartment_prefixes=lcp_compartment_prefixes,
    )

    _taxonomy_path = TAXONOMY_DIR / "lcp_feature_taxonomy.csv"
    _taxonomy_status = write_csv_protected(
        feature_taxonomy, _taxonomy_path, overwrite=CONFIG.overwrite_existing_outputs
    )
    print(f"✓ Feature taxonomy {_taxonomy_status}")
    print(f"  File: {_taxonomy_path}")

    for _grouping_level in GROUPING_LEVELS:
        print(f"\n{_grouping_level}:")
        print(
            feature_taxonomy[_grouping_level]
            .value_counts(dropna=False)
            .rename("n_features")
            .to_frame()
        )
    return (feature_taxonomy,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Taxonomy coverage overview

    This diagnostic shows how many features are represented in each family,
    compartment, and channel. Large `Other` or `Channel-independent` groups
    are not automatically errors: many shape and spatial measurements do not
    reference a fluorescence channel.
    """)
    return


@app.cell
def _(
    CONFIG,
    GROUPING_LEVELS,
    TAXONOMY_DIR,
    feature_taxonomy,
    pd,
    write_summary_table_protected,
):
    _taxonomy_summary = []
    for _grouping_level in GROUPING_LEVELS:
        _counts = feature_taxonomy[_grouping_level].value_counts(dropna=False)
        for _category, _count in _counts.items():
            _taxonomy_summary.append(
                {"grouping_level": _grouping_level, "category": _category, "n_features": int(_count)}
            )

    taxonomy_summary_df = pd.DataFrame(_taxonomy_summary)
    _taxonomy_summary_status = write_summary_table_protected(
        taxonomy_summary_df,
        TAXONOMY_DIR / "taxonomy_summary.csv",
        overwrite=CONFIG.overwrite_existing_outputs,
    )
    print(f"✓ Taxonomy summary {_taxonomy_summary_status}")

    taxonomy_summary_df.head(20)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — Calculate per-feature effect sizes

    Cohen's d is calculated feature by feature against the negative
    controls. For `within_plate_weighted` mode (the recommended default):
    the condition is compared with negative controls from the same plate,
    one Cohen's d is obtained per plate, and plate-specific effects are
    combined using the effective sample size as weight — this reduces the
    risk of mixing treatment effects with residual plate differences. For
    `global` mode, each condition is compared directly against all negative
    controls pooled across plates.
    """)
    return


@app.cell
def _(
    CONDITION_CONC_COL,
    CONFIG,
    EFFECT_ESTIMATION_MODE,
    EFFECT_THRESHOLD,
    MIN_SAMPLES_PER_GROUP,
    PLATE_COL,
    TABLES_DIR,
    TREATMENT_COL,
    calculate_global_effects,
    calculate_within_plate_effects,
    condition_table,
    df,
    feature_cols,
    feature_taxonomy,
    negative_control_mask,
    write_csv_protected,
):
    if EFFECT_ESTIMATION_MODE == "within_plate_weighted":
        feature_effects = calculate_within_plate_effects(
            df,
            feature_cols,
            condition_table=condition_table,
            negative_control_mask=negative_control_mask,
            treatment_column=TREATMENT_COL,
            concentration_column=CONDITION_CONC_COL,
            plate_column=PLATE_COL,
            min_samples=MIN_SAMPLES_PER_GROUP,
        )
    elif EFFECT_ESTIMATION_MODE == "global":
        feature_effects = calculate_global_effects(
            df,
            feature_cols,
            condition_table=condition_table,
            negative_control_mask=negative_control_mask,
            treatment_column=TREATMENT_COL,
            concentration_column=CONDITION_CONC_COL,
            plate_column=PLATE_COL,
            min_samples=MIN_SAMPLES_PER_GROUP,
        )
    else:
        raise ValueError("EFFECT_ESTIMATION_MODE must be 'within_plate_weighted' or 'global'.")

    feature_effects = feature_effects.merge(
        feature_taxonomy, on="feature", how="left", validate="many_to_one"
    )
    feature_effects["absolute_effect"] = feature_effects["effect_size"].abs()
    feature_effects["is_altered"] = feature_effects["absolute_effect"].ge(EFFECT_THRESHOLD)

    _effect_table_path = TABLES_DIR / "per_feature_effect_sizes.csv"
    _effect_table_status = write_csv_protected(
        feature_effects, _effect_table_path, overwrite=CONFIG.overwrite_existing_outputs
    )

    print(f"Effect-size rows:  {len(feature_effects):,}")
    print(f"Finite effects:    {feature_effects['effect_size'].notna().sum():,}")
    print(f"Effect table {_effect_table_status}")
    print(f"  File: {_effect_table_path}")

    feature_effects.head()
    return (feature_effects,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — Build fingerprint matrices

    For each grouping level, four complementary summaries are generated:

    - **mean absolute effect** — average response magnitude;
    - **mean signed effect** — average response direction;
    - **maximum absolute effect** — strongest feature response;
    - **fraction altered** — fraction of features with `|d| ≥ threshold`.
    """)
    return


@app.cell
def _(EFFECT_THRESHOLD):
    FINGERPRINT_METRICS = {
        "mean_absolute": {
            "column": "absolute_effect",
            "aggregation": "mean",
            "label": "Mean absolute Cohen's d",
        },
        "mean_signed": {
            "column": "effect_size",
            "aggregation": "mean",
            "label": "Mean signed Cohen's d",
        },
        "max_absolute": {
            "column": "absolute_effect",
            "aggregation": "max",
            "label": "Maximum absolute Cohen's d",
        },
        "fraction_altered": {
            "column": "is_altered",
            "aggregation": "mean",
            "label": f"Fraction of features |d| ≥ {EFFECT_THRESHOLD:g}",
        },
    }
    return (FINGERPRINT_METRICS,)


@app.cell
def _(
    CONFIG,
    DROP_OTHER_FROM_FIGURES,
    FIGURES_DIR,
    FINGERPRINT_METRICS,
    GROUPING_LEVELS,
    TABLES_DIR,
    build_fingerprint_matrix,
    condition_order,
    feature_effects,
    write_csv_protected,
):
    fingerprint_matrices = {}

    for _grouping_level in GROUPING_LEVELS:
        fingerprint_matrices[_grouping_level] = {}

        _grouping_results_dir = TABLES_DIR / _grouping_level
        _grouping_figures_dir = FIGURES_DIR / _grouping_level
        for _directory in (_grouping_results_dir, _grouping_figures_dir):
            _directory.mkdir(parents=True, exist_ok=True)

        for _metric_name, _metric_spec in FINGERPRINT_METRICS.items():
            _matrix = build_fingerprint_matrix(
                feature_effects,
                grouping_level=_grouping_level,
                metric_column=_metric_spec["column"],
                aggregation=_metric_spec["aggregation"],
                condition_order=condition_order,
                drop_other=DROP_OTHER_FROM_FIGURES,
            )
            fingerprint_matrices[_grouping_level][_metric_name] = _matrix

            write_csv_protected(
                _matrix.reset_index(),
                _grouping_results_dir / f"{_metric_name}.csv",
                overwrite=CONFIG.overwrite_existing_outputs,
            )

    print(f"✓ Fingerprint matrices generated for {len(GROUPING_LEVELS)} grouping level(s).")
    return (fingerprint_matrices,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 — Heatmaps
    """)
    return


@app.cell
def _(
    FIGURES_DIR,
    FIGURE_DPI,
    FINGERPRINT_METRICS,
    fingerprint_matrices,
    plot_fingerprint_heatmap,
):
    _n_heatmaps = 0
    for _grouping_level, _metric_matrices in fingerprint_matrices.items():
        _grouping_figures_dir = FIGURES_DIR / _grouping_level
        for _metric_name, _matrix in _metric_matrices.items():
            _metric_spec = FINGERPRINT_METRICS[_metric_name]
            plot_fingerprint_heatmap(
                _matrix,
                title=(
                    f"{_grouping_level.replace('_', ' ').title()} — "
                    f"{_metric_name.replace('_', ' ').title()}"
                ),
                colorbar_label=_metric_spec["label"],
                output_path=_grouping_figures_dir / f"{_metric_name}_heatmap.png",
                diverging=(_metric_name == "mean_signed"),
                fixed_range=(0, 1) if _metric_name == "fraction_altered" else None,
                dpi=FIGURE_DPI,
            )
            _n_heatmaps += 1

    print(f"✓ Saved {_n_heatmaps} fingerprint heatmap(s) to {FIGURES_DIR}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 — Radar plots

    The directional (mean-signed) fingerprint is deliberately excluded from
    radar plots because positive and negative axes are difficult to
    interpret in polar geometry. Radar plots are generated for the mean
    absolute effect, the relative fingerprint shape (each condition
    normalized to its own maximum), and the fraction of altered features.

    Dose-overlay radars (one panel per treatment, concentrations overlaid)
    are generated **only when this experiment has a dose/concentration
    axis** — see the graceful-skip note in section 3 above.
    """)
    return


@app.cell
def _(
    CONDITION_CONC_COL,
    FIGURES_DIR,
    FIGURE_DPI,
    HAS_DOSE_AXIS,
    RADAR_GROUPINGS,
    RADAR_MAX_COLUMNS,
    TREATMENT_COL,
    condition_table,
    fingerprint_matrices,
    np,
    plot_condition_radar_grid,
    plot_dose_overlay_radars,
):
    if not HAS_DOSE_AXIS:
        print(
            "⚠ No dose/concentration axis for this experiment — skipping "
            "dose-overlay radar plots. Condition-grid and relative-shape "
            "radar plots are still generated below."
        )

    for _grouping_level in RADAR_GROUPINGS:
        if _grouping_level not in fingerprint_matrices:
            continue

        _grouping_figures_dir = FIGURES_DIR / _grouping_level
        _mean_absolute = fingerprint_matrices[_grouping_level]["mean_absolute"]
        _fraction_altered = fingerprint_matrices[_grouping_level]["fraction_altered"]

        if not _mean_absolute.empty:
            _relative_shape = _mean_absolute.divide(
                _mean_absolute.max(axis=0).replace(0, np.nan), axis=1
            ).fillna(0)

            plot_condition_radar_grid(
                _mean_absolute,
                title=f"{_grouping_level.replace('_', ' ').title()} — Mean absolute effect",
                output_path=_grouping_figures_dir / "mean_absolute_radar_grid.png",
                max_columns=RADAR_MAX_COLUMNS,
                dpi=FIGURE_DPI,
            )

            if HAS_DOSE_AXIS:
                plot_dose_overlay_radars(
                    _mean_absolute,
                    condition_table=condition_table,
                    treatment_column=TREATMENT_COL,
                    concentration_column=CONDITION_CONC_COL,
                    title=f"{_grouping_level.replace('_', ' ').title()} — Dose overlays",
                    output_path=_grouping_figures_dir / "mean_absolute_dose_overlay.png",
                    max_columns=RADAR_MAX_COLUMNS,
                    dpi=FIGURE_DPI,
                )

            plot_condition_radar_grid(
                _relative_shape,
                title=f"{_grouping_level.replace('_', ' ').title()} — Relative fingerprint shape",
                output_path=_grouping_figures_dir / "relative_shape_radar_grid.png",
                max_columns=RADAR_MAX_COLUMNS,
                dpi=FIGURE_DPI,
            )

        if not _fraction_altered.empty:
            plot_condition_radar_grid(
                _fraction_altered,
                title=f"{_grouping_level.replace('_', ' ').title()} — Fraction of altered features",
                output_path=_grouping_figures_dir / "fraction_altered_radar_grid.png",
                max_columns=RADAR_MAX_COLUMNS,
                dpi=FIGURE_DPI,
            )

    print(f"✓ Radar plots generated for {len(RADAR_GROUPINGS)} grouping level(s).")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11 — Special-treatment panels

    Some conditions (for example, a reference biological state that is not
    part of a conventional dose series) are better viewed on their own
    rather than forced into the dose-grid interpretation. Select any such
    treatments below — the default is empty, so nothing is special-cased
    unless explicitly chosen.
    """)
    return


@app.cell
def _(TREATMENT_COL, condition_table, mo):
    _treatment_options = sorted(condition_table[TREATMENT_COL].astype(str).unique())
    special_treatments_input = mo.ui.multiselect(
        options=_treatment_options,
        value=[],
        label="Special treatments (plotted separately, outside the dose grid)",
    )
    special_treatments_input
    return (special_treatments_input,)


@app.cell
def _(
    FIGURES_DIR,
    FIGURE_DPI,
    RADAR_GROUPINGS,
    RADAR_MAX_COLUMNS,
    TREATMENT_COL,
    condition_table,
    fingerprint_matrices,
    plot_condition_radar_grid,
    special_treatments_input,
):
    _special_treatments = set(special_treatments_input.value)

    if _special_treatments:
        _special_condition_labels = condition_table.loc[
            condition_table[TREATMENT_COL].astype(str).isin(_special_treatments),
            "Metadata_Condition",
        ].tolist()

        _special_figures_dir = FIGURES_DIR / "special_conditions"
        _special_figures_dir.mkdir(parents=True, exist_ok=True)

        for _grouping_level in RADAR_GROUPINGS:
            _matrix = fingerprint_matrices.get(_grouping_level, {}).get("mean_absolute")
            if _matrix is None or _matrix.empty:
                continue

            _available = [c for c in _special_condition_labels if c in _matrix.columns]
            if not _available:
                continue

            plot_condition_radar_grid(
                _matrix[_available],
                title=f"Special conditions — {_grouping_level.replace('_', ' ').title()}",
                output_path=_special_figures_dir / f"{_grouping_level}_mean_absolute.png",
                max_columns=RADAR_MAX_COLUMNS,
                dpi=FIGURE_DPI,
            )

        print(f"✓ Special-condition radar panels generated for: {sorted(_special_treatments)}")
    else:
        print("No special treatments selected — skipping the special-condition panel section.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12 — Top altered features

    Radar plots summarize categories. This table preserves feature-level
    resolution and identifies the strongest contributors to each condition.
    """)
    return


@app.cell
def _(
    CONFIG,
    TABLES_DIR,
    TOP_FEATURES_PER_CONDITION,
    feature_effects,
    write_csv_protected,
):
    top_features = (
        feature_effects.dropna(subset=["effect_size"])
        .sort_values(["condition", "absolute_effect"], ascending=[True, False])
        .groupby("condition", group_keys=False)
        .head(TOP_FEATURES_PER_CONDITION)
    )

    _top_features_path = TABLES_DIR / "top_altered_features_by_condition.csv"
    _top_features_status = write_csv_protected(
        top_features, _top_features_path, overwrite=CONFIG.overwrite_existing_outputs
    )
    print(f"✓ Top altered features table {_top_features_status}")
    print(f"  File: {_top_features_path}")

    top_features[
        [
            "condition",
            "feature",
            "effect_size",
            "feature_family",
            "compartment",
            "channel",
            "biological_signature",
        ]
    ].head(30)
    return (top_features,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpreting the fingerprint metrics

    Each grouping level (feature family, compartment, channel, compartment ×
    family, channel × family, biological signature) is summarized with four
    complementary metrics:

    | Metric | Question answered |
    |---|---|
    | **Mean absolute effect** | Which categories are most strongly perturbed, regardless of direction? |
    | **Mean signed effect** | Do features in this category tend to increase or decrease? (Caution: opposing changes can cancel out and hide a real response — interpret together with the absolute-effect heatmap.) |
    | **Maximum absolute effect** | Does the category contain at least one exceptionally responsive feature? (Sensitive to single outliers — do not treat as representative of the whole category.) |
    | **Fraction altered** | Is the response broad (many features change) or narrow (driven by a few)? |

    Radar plots additionally show:

    - **Relative fingerprint shape** — each condition normalized to its own
      maximum, comparing fingerprint *composition* rather than overall
      magnitude.
    - **Dose-overlay radars** — one panel per treatment with all
      concentrations overlaid, showing whether magnitude and shape progress
      consistently with dose. *(Only produced when this experiment has a
      concentration/dose axis.)*

    These are descriptive, biologically-informed summaries of CellProfiler
    feature groups — not direct measurements of a specific molecular pathway
    or mechanism. Large values indicate that features from that group
    deviate more from the negative control; they do not by themselves
    establish causality. Fingerprints remain in the normalized CellProfiler
    feature space and should not be interpreted as Harmony-corrected
    features.
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
    CONFIG,
    CONFIG_RESOLVED,
    EFFECT_ESTIMATION_MODE,
    EFFECT_THRESHOLD,
    EXPERIMENT_ID,
    FIGURES_DIR,
    GROUPING_LEVELS,
    HAS_DOSE_AXIS,
    INPUT_PARQUET,
    MIN_SAMPLES_PER_GROUP,
    PROVENANCE_DIR,
    RADAR_GROUPINGS,
    REPO_ROOT,
    RESULTS_DIR,
    Sequence,
    TOP_FEATURES_PER_CONDITION,
    condition_order,
    datetime,
    df,
    feature_cols,
    json,
    np,
    pd,
    platform,
    special_treatments_input,
    subprocess,
    timezone,
):
    def run_git_command(arguments: Sequence[str], repo_root) -> str | None:
        """Run a read-only Git command and return stripped stdout."""
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
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
        "schema_version": 1,
        "pipeline": {
            "notebook": "04_phenotypic_fingerprints.py",
            "experiment_id": EXPERIMENT_ID,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "configuration": {
            "effect_estimation_mode": EFFECT_ESTIMATION_MODE,
            "effect_threshold": EFFECT_THRESHOLD,
            "minimum_samples_per_group": MIN_SAMPLES_PER_GROUP,
            "grouping_levels": GROUPING_LEVELS,
            "radar_groupings": RADAR_GROUPINGS,
            "top_features_per_condition": TOP_FEATURES_PER_CONDITION,
            "special_treatments": list(special_treatments_input.value),
            "has_dose_axis": bool(HAS_DOSE_AXIS),
            "plate_column": CONFIG_RESOLVED.plate_col,
            "well_column": CONFIG_RESOLVED.well_col,
            "treatment_column": CONFIG_RESOLVED.treatment_col,
            "concentration_column": CONFIG_RESOLVED.concentration_col,
            "control_type_column": CONFIG_RESOLVED.control_type_col,
            "negative_control_values": CONFIG.negcon_values,
            "overwrite_existing_outputs": bool(CONFIG.overwrite_existing_outputs),
        },
        "dataset": {
            "input_file": str(INPUT_PARQUET),
            "n_wells": int(len(df)),
            "n_features": int(len(feature_cols)),
            "n_conditions": int(len(condition_order)),
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
            "results_directory": str(RESULTS_DIR),
            "figures_directory": str(FIGURES_DIR),
        },
    }

    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    provenance_latest_path = PROVENANCE_DIR / "phenotypic_fingerprints_provenance.json"
    provenance_latest_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    provenance_history_path = None
    if CONFIG.save_provenance_history:
        _timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        provenance_history_path = (
            PROVENANCE_DIR / f"phenotypic_fingerprints_provenance_{_timestamp}.json"
        )
        if provenance_history_path.exists():
            raise FileExistsError(f"Historical provenance file already exists: {provenance_history_path}")
        provenance_history_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print("═" * 72)
    print("PHENOTYPIC FINGERPRINT ANALYSIS — PROVENANCE")
    print("═" * 72)
    print(f"Experiment        : {EXPERIMENT_ID}")
    print(f"Wells             : {len(df):,}")
    print(f"Features          : {len(feature_cols):,}")
    print(f"Conditions        : {len(condition_order):,}")
    print(f"Effect mode       : {EFFECT_ESTIMATION_MODE}")
    print(f"Effect threshold  : {EFFECT_THRESHOLD:g}")
    print(f"Dose axis         : {'yes' if HAS_DOSE_AXIS else 'no'}")
    print(f"Results           : {RESULTS_DIR}")
    print(f"Figures           : {FIGURES_DIR}")
    print(f"\n✓ Latest provenance: {provenance_latest_path}")
    if provenance_history_path is not None:
        print(f"✓ Historical record: {provenance_history_path}")
    else:
        print("  Historical record:  disabled")
    return (provenance_latest_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 14 — Final integrity checks and execution summary
    """)
    return


@app.cell
def _(
    EXPERIMENT_ID,
    FIGURES_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    TAXONOMY_DIR,
    condition_order,
    df,
    feature_cols,
    feature_effects,
    feature_taxonomy,
    fingerprint_matrices,
    provenance_latest_path,
    top_features,
):
    integrity_errors = []

    if feature_taxonomy["feature"].nunique() != len(feature_cols):
        integrity_errors.append("feature_taxonomy does not have exactly one row per feature column.")

    if not feature_effects["condition"].isin(condition_order).all():
        integrity_errors.append("feature_effects contains conditions absent from condition_order.")

    if len(fingerprint_matrices) == 0:
        integrity_errors.append("No fingerprint matrices were generated.")

    if top_features.empty:
        integrity_errors.append("top_altered_features table is empty.")

    _required_paths = [
        TAXONOMY_DIR / "lcp_feature_taxonomy.csv",
        TABLES_DIR / "per_feature_effect_sizes.csv",
        TABLES_DIR / "top_altered_features_by_condition.csv",
        provenance_latest_path,
    ]
    _missing_outputs = [p for p in _required_paths if not p.exists()]
    if _missing_outputs:
        integrity_errors.append(
            "Required output files are missing: " + ", ".join(str(p) for p in _missing_outputs)
        )

    if integrity_errors:
        raise RuntimeError(
            "\nNB04 integrity checks failed\n============================\n\n"
            + "\n".join(f"  - {e}" for e in integrity_errors)
        )

    print("═" * 72)
    print("NB04 COMPLETED")
    print("═" * 72)
    print("✓ All final integrity checks passed\n")
    print(f"  Experiment:          {EXPERIMENT_ID}")
    print(f"  Wells:               {len(df):,}")
    print(f"  Features:            {len(feature_cols):,}")
    print(f"  Conditions:          {len(condition_order):,}")
    print(f"  Grouping levels:     {len(fingerprint_matrices):,}")
    print(f"  Top-features rows:   {len(top_features):,}")
    print("\n  Results:")
    print(f"    {RESULTS_DIR}")
    print("\n  Figures:")
    print(f"    {FIGURES_DIR}")
    print("\nNext step: NB05 — Quality metrics")
    return


if __name__ == "__main__":
    app.run()
