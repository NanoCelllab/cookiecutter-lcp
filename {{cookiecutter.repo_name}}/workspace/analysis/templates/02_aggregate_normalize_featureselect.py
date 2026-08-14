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
    # 02 — Annotate · Clean · Aggregate · Normalize · FeatureSelect

    **Pipeline step:** 2 of 6
    **Input:** `single_cell_profiles.parquet` (from NB01; CSV supported as fallback)
    **Output:** `per_well_features_selected.parquet` — per-well median profiles,
    normalized and feature-selected

    ### What this notebook does

    1. Loads single-cell profiles and annotates with platemap metadata.
    2. Removes wells not in the platemap layout (QC).
    3. Aggregates single-cell data to per-well median profiles.
    4. Cleans (missingness filter + imputation) and normalizes plate-wise
       with pycytominer `mad_robustize`.
    5. Selects features with pycytominer `feature_select`.
    6. Runs sanity checks SC-06 → SC-10, including a batch-effect PCA
       and within-group CV/correlation diagnostic.
    7. Saves `cv_summary.csv` for NB04's optional LDA bias check.

    Valid checkpoints may be reused to avoid repeating expensive stages.
    Every reuse, recomputation, exclusion, warning, and written output is
    reported explicitly below.
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
    from dataclasses import replace
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    pd.set_option("display.max_columns", 200)
    return PCA, Path, StandardScaler, np, pd, plt, replace, silhouette_score


@app.cell
def _(Path, mo, print):
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

    from hca_pipeline.config import SUPPORTED_PLATE_FORMATS, ExperimentConfig, validate_configuration
    from hca_pipeline.feature_select import (
        infer_feature_cols,
        select_features,
        technical_identifier_columns,
    )
    from hca_pipeline.io import (
        checkpoint_matches_plate_scope,
        resolve_resume_stage,
        validate_checkpoint_df,
        write_csv_protected,
        write_parquet_protected,
        write_summary_table_protected,
    )
    from hca_pipeline.metadata import (
        CELL_COUNT_METADATA_COLUMN,
        add_cell_count_metadata,
        annotate_per_plate,
        dedupe_meta,
        enforce_pascalcase_metadata_columns,
        ensure_core_metadata,
        find_wells_missing_from_layout,
        norm_well,
        read_barcode_platemap,
        read_platemap_layout,
    )
    from hca_pipeline.normalize import (
        clean_features_before_normalization,
        drop_extreme_magnitude_features,
        drop_near_zero_mad_features,
        normalize_per_plate_mad_robustize,
    )

    mo.callout(
        mo.md(f"### Shared utilities ready\n\n✅ Loaded `hca_pipeline` from `{_pipelines_dir}`."),
        kind="success",
    )
    return (
        CELL_COUNT_METADATA_COLUMN,
        ExperimentConfig,
        REPO_ROOT,
        add_cell_count_metadata,
        annotate_per_plate,
        checkpoint_matches_plate_scope,
        clean_features_before_normalization,
        dedupe_meta,
        drop_extreme_magnitude_features,
        drop_near_zero_mad_features,
        enforce_pascalcase_metadata_columns,
        ensure_core_metadata,
        find_wells_missing_from_layout,
        infer_feature_cols,
        norm_well,
        normalize_per_plate_mad_robustize,
        read_barcode_platemap,
        select_features,
        technical_identifier_columns,
        validate_checkpoint_df,
        validate_configuration,
        write_csv_protected,
        write_parquet_protected,
        write_summary_table_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Select the experiment to process. Its saved configuration from NB01 is
    loaded automatically. Most NB02 parameters have safe defaults and are
    available under **Advanced settings** only when they need review.
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
    negcon_values_input = mo.ui.text(
        value=", ".join(loaded_config.negcon_values),
        label="Negative-control values (comma-separated; matched against the control-type column)",
    )
    norm_control_input = mo.ui.dropdown(
        options=loaded_config.negcon_values or ["negcon"],
        value=(loaded_config.negcon_values or ["negcon"])[0],
        label="Control value to normalize against (must be one of the negative-control values)",
    )
    use_checkpoints_input = mo.ui.checkbox(
        value=True,
        label="Reuse on-disk checkpoints when available (skip recomputation)",
    )
    max_missing_fraction_input = mo.ui.number(
        value=0.20, start=0.0, stop=1.0, step=0.01,
        label="Max fraction of missing values allowed per feature",
    )
    mad_epsilon_input = mo.ui.number(
        value=1e-3, start=0.0, stop=1.0, step=1e-4,
        label="SC-07b: min negative-control MAD per feature (below this, drop before normalizing)",
    )
    max_normalized_magnitude_input = mo.ui.number(
        value=100.0, start=1.0, stop=1_000_000.0, step=1.0,
        label="SC-09b: max plausible |normalized feature value| (above this, drop after normalizing)",
    )
    cv_warn_threshold_input = mo.ui.number(
        value=1.5, start=0.0, stop=100.0, label="Within-group CV warning threshold",
    )
    corr_warn_threshold_input = mo.ui.number(
        value=0.20, start=0.0, stop=1.0, step=0.01,
        label="Replicate-correlation warning threshold",
    )
    exclude_wells_input = mo.ui.text_area(
        value="",
        label='Wells to exclude, one "Plate:Well" per line (optional)',
    )
    overwrite_input = mo.ui.checkbox(
        value=loaded_config.overwrite_existing_outputs,
        label="Overwrite existing outputs",
    )
    save_history_input = mo.ui.checkbox(
        value=loaded_config.save_provenance_history,
        label="Save timestamped provenance history",
    )

    advanced_settings = mo.accordion(
        {
            "Advanced settings": mo.vstack(
                [
                    mo.md(
                        "Change these values only when the experimental design or QC review "
                        "requires it. The validated values are summarized below before processing."
                    ),
                    negcon_values_input,
                    norm_control_input,
                    use_checkpoints_input,
                    max_missing_fraction_input,
                    mad_epsilon_input,
                    max_normalized_magnitude_input,
                    cv_warn_threshold_input,
                    corr_warn_threshold_input,
                    exclude_wells_input,
                    overwrite_input,
                    save_history_input,
                ],
                gap=1,
            )
        }
    )
    advanced_settings
    return (
        corr_warn_threshold_input,
        cv_warn_threshold_input,
        exclude_wells_input,
        mad_epsilon_input,
        max_missing_fraction_input,
        max_normalized_magnitude_input,
        negcon_values_input,
        norm_control_input,
        overwrite_input,
        save_history_input,
        use_checkpoints_input,
    )


@app.cell
def _(
    REPO_ROOT,
    experiment_id_input,
    loaded_config,
    negcon_values_input,
    norm_control_input,
    overwrite_input,
    mo,
    print,
    replace,
    save_history_input,
    use_checkpoints_input,
    validate_configuration,
):
    EXPERIMENT_ID = experiment_id_input.value
    BACKEND_DIR = REPO_ROOT / "workspace" / "backend"
    validate_configuration(
        experiment_id=EXPERIMENT_ID,
        plate_format=loaded_config.plate_format,
        min_cells_per_well=loaded_config.min_cells_per_well,
        experiments_dir=BACKEND_DIR,
    )

    NEGCON_VALUES = [v.strip() for v in negcon_values_input.value.split(",") if v.strip()]
    NORM_CONTROL = norm_control_input.value
    if not NEGCON_VALUES:
        raise ValueError(
            "At least one negative-control value is required. Open Advanced settings "
            "and enter the control vocabulary used in the platemap."
        )
    if NORM_CONTROL not in NEGCON_VALUES:
        raise ValueError(
            f"Normalization reference {NORM_CONTROL!r} is not present in the configured "
            f"negative-control values: {NEGCON_VALUES}."
        )

    CONFIG = replace(
        loaded_config,
        experiment_id=EXPERIMENT_ID,
        negcon_values=NEGCON_VALUES,
        overwrite_existing_outputs=bool(overwrite_input.value),
        save_provenance_history=bool(save_history_input.value),
    )
    CONFIG.save(REPO_ROOT)

    OVERWRITE_EXISTING_OUTPUTS = bool(overwrite_input.value)
    mo.callout(
        mo.md(
            f"""
            ### Configuration validated

            | Setting | Effective value |
            |---|---|
            | Experiment | `{EXPERIMENT_ID}` |
            | Analysis mode | **{CONFIG.analysis_mode}** |
            | Negative-control values | {', '.join(f'`{v}`' for v in NEGCON_VALUES)} |
            | Normalization reference | `{NORM_CONTROL}` |
            | Reuse checkpoints | **{bool(use_checkpoints_input.value)}** |
            | Overwrite existing outputs | **{OVERWRITE_EXISTING_OUTPUTS}** |
            | Save timestamped provenance | **{CONFIG.save_provenance_history}** |
            """
        ),
        kind="success",
    )
    return (
        CONFIG,
        EXPERIMENT_ID,
        NEGCON_VALUES,
        NORM_CONTROL,
        OVERWRITE_EXISTING_OUTPUTS,
    )


@app.cell
def _(EXPERIMENT_ID, REPO_ROOT, mo, print):
    WORKSPACE_DIR = REPO_ROOT / "workspace"
    METADATA_DIR = WORKSPACE_DIR / "metadata" / EXPERIMENT_ID
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    PROFILES_DIR = WORKSPACE_DIR / "profiles" / EXPERIMENT_ID

    ANALYSIS_OUT_DIR = ANALYSIS_DIR / "outputs"
    PROFILES_OUT_DIR = PROFILES_DIR / "outputs"
    FIGS_DIR = ANALYSIS_DIR / "figures" / "aggregation"
    CACHE_DIR = PROFILES_OUT_DIR / "cache"
    RESULTS_DIR = ANALYSIS_DIR / "results"

    BARCODE_PLATEMAP_CSV = METADATA_DIR / "barcode_platemap.csv"
    PLATEMAP_DIR = METADATA_DIR / "platemap"
    INPUT_CSV = ANALYSIS_OUT_DIR / "single_cell_profiles.csv"
    INPUT_PARQUET = ANALYSIS_OUT_DIR / "single_cell_profiles.parquet"

    SC_ANNOTATED_PARQUET = CACHE_DIR / "single_cell_annotated.parquet"
    SC_READY_PARQUET = PROFILES_OUT_DIR / "single_cell_ready.parquet"
    LEGACY_SC_READY_PARQUET = CACHE_DIR / "single_cell_ready.parquet"
    PW_AGGREGATED_PARQUET = CACHE_DIR / "per_well_aggregated.parquet"
    PW_NORMALIZED_PARQUET = CACHE_DIR / "per_well_normalized.parquet"
    PW_FEATURES_SELECTED_PARQUET = PROFILES_OUT_DIR / "per_well_features_selected.parquet"
    PW_FEATURES_SELECTED_CSV = PROFILES_OUT_DIR / "per_well_features_selected.csv"

    CV_SUMMARY_CSV = RESULTS_DIR / "cv_summary.csv"
    WITHIN_GROUP_VARIABILITY_CSV = RESULTS_DIR / "within_group_variability.csv"

    for _d in (ANALYSIS_OUT_DIR, PROFILES_OUT_DIR, FIGS_DIR, CACHE_DIR, RESULTS_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    mo.callout(
        mo.md(
            f"""
            ### Paths ready

            - Preferred NB01 input: `{INPUT_PARQUET}`
            - CSV fallback: `{INPUT_CSV}`
            - Final Parquet: `{PW_FEATURES_SELECTED_PARQUET}`
            - Declared single-cell output: `{SC_READY_PARQUET}`
            - Interoperability CSV: `{PW_FEATURES_SELECTED_CSV}`
            - Figures: `{FIGS_DIR}`
            """
        ),
        kind="info",
    )
    return (
        BARCODE_PLATEMAP_CSV,
        CV_SUMMARY_CSV,
        FIGS_DIR,
        INPUT_CSV,
        INPUT_PARQUET,
        LEGACY_SC_READY_PARQUET,
        PLATEMAP_DIR,
        PW_AGGREGATED_PARQUET,
        PW_FEATURES_SELECTED_CSV,
        PW_FEATURES_SELECTED_PARQUET,
        PW_NORMALIZED_PARQUET,
        RESULTS_DIR,
        SC_ANNOTATED_PARQUET,
        SC_READY_PARQUET,
        WITHIN_GROUP_VARIABILITY_CSV,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Load single-cell profiles

    NB02 first looks for the Parquet produced by NB01, then uses the CSV copy
    only as a compatibility fallback. Not finding either file is an expected
    first-run condition: the notebook stops here with instructions instead of
    allowing later cells to fail.
    """)
    return


@app.cell
def _(INPUT_CSV, INPUT_PARQUET, add_cell_count_metadata, dedupe_meta, ensure_core_metadata, mo, pd, print):
    print("INPUT DISCOVERY")
    print(f"  Parquet: {'found' if INPUT_PARQUET.exists() else 'not found'} — {INPUT_PARQUET}")
    print(f"  CSV:     {'found' if INPUT_CSV.exists() else 'not found'} — {INPUT_CSV}")
    if INPUT_PARQUET.exists():
        df_loaded = pd.read_parquet(INPUT_PARQUET)
        _input_kind = "Parquet"
        _input_path = INPUT_PARQUET
    elif INPUT_CSV.exists():
        df_loaded = pd.read_csv(INPUT_CSV, low_memory=False)
        _input_kind = "CSV fallback"
        _input_path = INPUT_CSV
    else:
        mo.stop(True, mo.callout(mo.md("**NB02 stopped: NB01 output was not found.**\n\n" f"Checked:\n- `{INPUT_PARQUET}`\n- `{INPUT_CSV}`\n\nRun NB01 through `NB01 COMPLETED` and rerun this cell. Downstream cells were stopped to avoid cascading errors."), kind="warn"))

    df_loaded = add_cell_count_metadata(dedupe_meta(ensure_core_metadata(df_loaded)))
    print(f"✓ Loaded {_input_kind}: {_input_path}")
    print(f"  Cells: {len(df_loaded):,} · columns: {df_loaded.shape[1]:,}")
    return (df_loaded,)


@app.cell
def _(CONFIG, df_loaded, pd, print):
    available_plates = sorted(df_loaded[CONFIG.plate_col].dropna().astype(str).unique())
    analysis_plates = CONFIG.resolve_plate_scope(available_plates)
    df_analysis_scope = df_loaded.loc[
        df_loaded[CONFIG.plate_col].astype(str).isin(analysis_plates)
    ].copy()
    if df_analysis_scope.empty:
        raise ValueError("Plate selection produced an empty analysis dataset.")

    excluded_plates = [plate for plate in available_plates if plate not in analysis_plates]
    analysis_scope_summary = pd.DataFrame(
        [
            {
                "Plate": plate,
                "Included": plate in analysis_plates,
                "Cells": int(df_loaded[CONFIG.plate_col].astype(str).eq(plate).sum()),
                "Reason if excluded": CONFIG.excluded_plate_reasons.get(plate, ""),
            }
            for plate in available_plates
        ]
    )
    print("═" * 72)
    print(f"ANALYSIS SCOPE — {CONFIG.analysis_mode.upper()}")
    print("═" * 72)
    print(f"  Included plates: {len(analysis_plates)}/{len(available_plates)}")
    for _plate in analysis_plates:
        print(f"    ✓ {_plate}")
    for _plate in excluded_plates:
        print(f"    — {_plate}: {CONFIG.excluded_plate_reasons.get(_plate, 'no reason recorded')}")
    if CONFIG.analysis_mode == "preliminary":
        print("  ⚠️  Preliminary mode: downstream estimates are exploratory and may lack complete replication.")
    analysis_scope_summary
    return analysis_plates, analysis_scope_summary, df_analysis_scope, excluded_plates


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Annotate with platemap (SC-06)

    ### SC-06 — Treatment-annotation coverage

    **What it evaluates:** whether every loaded cell can be linked through its
    plate and well to the treatment metadata in the platemap.

    **How to read it:** PASS means every cell has a treatment label. WARN means
    some cells remain unannotated; the reported count and fraction show the
    extent of the problem. This is advisory here because the rows are preserved,
    but treatment-based analyses may omit or misgroup them. Review barcode and
    platemap mappings before interpreting treatment effects.

    This check confirms annotation coverage only; it does not confirm that the
    experimental labels themselves are biologically correct.
    """)
    return


@app.cell
def _(
    BARCODE_PLATEMAP_CSV,
    CELL_COUNT_METADATA_COLUMN,
    LEGACY_SC_READY_PARQUET,
    PLATEMAP_DIR,
    SC_ANNOTATED_PARQUET,
    add_cell_count_metadata,
    annotate_per_plate,
    analysis_plates,
    checkpoint_matches_plate_scope,
    dedupe_meta,
    df_analysis_scope,
    infer_feature_cols,
    read_barcode_platemap,
    print,
    use_checkpoints_input,
    validate_checkpoint_df,
):
    _reuse_annotated = use_checkpoints_input.value and checkpoint_matches_plate_scope(
        SC_ANNOTATED_PARQUET, analysis_plates
    )
    if _reuse_annotated:
        import pandas as _pd

        df_annotated = _pd.read_parquet(SC_ANNOTATED_PARQUET)
        validate_checkpoint_df(df_annotated, "annotated", infer_feature_cols(df_annotated))
        print(f"✓ Reused checkpoint: {SC_ANNOTATED_PARQUET}")
    else:
        if not BARCODE_PLATEMAP_CSV.exists():
            raise FileNotFoundError(f"barcode_platemap.csv not found: {BARCODE_PLATEMAP_CSV}")
        if not PLATEMAP_DIR.exists():
            raise FileNotFoundError(f"PLATEMAP_DIR not found: {PLATEMAP_DIR}")

        barcode_index = read_barcode_platemap(BARCODE_PLATEMAP_CSV)
        if use_checkpoints_input.value and SC_ANNOTATED_PARQUET.exists():
            print("  ℹ️  Invalidated annotated checkpoint because its plate scope changed")
        df_annotated = dedupe_meta(annotate_per_plate(df_analysis_scope, barcode_index, PLATEMAP_DIR))
        df_annotated.to_parquet(SC_ANNOTATED_PARQUET, index=False)
        print(f"✓ Annotated and cached: {SC_ANNOTATED_PARQUET}")

    _migrate_count_metadata = CELL_COUNT_METADATA_COLUMN not in df_annotated.columns
    df_annotated = add_cell_count_metadata(df_annotated)
    if _migrate_count_metadata:
        df_annotated.to_parquet(SC_ANNOTATED_PARQUET, index=False)
        print(f"  ✓ Added {CELL_COUNT_METADATA_COLUMN} to annotated checkpoint")

    print(f"  Shape: {df_annotated.shape}")
    return (df_annotated,)


@app.cell
def _(CONFIG, df_annotated, print):
    RESOLVED_CONFIG = CONFIG.resolve_columns(df_annotated)
    print("── SC-06: Annotation coverage ──")
    if RESOLVED_CONFIG.treatment_col not in df_annotated.columns:
        raise ValueError(f"{RESOLVED_CONFIG.treatment_col} is missing after annotation.")

    n_missing_treatment = int(df_annotated[RESOLVED_CONFIG.treatment_col].isna().sum())
    if n_missing_treatment > 0:
        print(
            f"  ⚠️  {n_missing_treatment} cells ({n_missing_treatment / len(df_annotated):.1%}) "
            "have no treatment annotation."
        )
        sc06_status = "WARN"
        print("  SC-06: WARN — review barcode/platemap mappings before treatment-level interpretation")
    else:
        sc06_status = "PASS"
        print("  ✓  All cells have treatment annotation")
        print("  SC-06: PASS")

    print(f"  Dose axis:  {'present (' + RESOLVED_CONFIG.concentration_col + ')' if RESOLVED_CONFIG.has_dose_axis else 'absent'}")
    print(f"  Time axis:  {'present (' + RESOLVED_CONFIG.time_col + ')' if RESOLVED_CONFIG.has_time_axis else 'absent'}")
    return RESOLVED_CONFIG, n_missing_treatment, sc06_status


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Remove wells not in the platemap layout

    Wells present in the data but missing from the platemap layout (or
    explicitly excluded) are dropped before aggregation. The notebook reports
    both affected wells and the exact number of single-cell rows removed.
    """)
    return


@app.cell
def _(
    BARCODE_PLATEMAP_CSV,
    CELL_COUNT_METADATA_COLUMN,
    PLATEMAP_DIR,
    SC_READY_PARQUET,
    add_cell_count_metadata,
    analysis_plates,
    checkpoint_matches_plate_scope,
    df_annotated,
    exclude_wells_input,
    find_wells_missing_from_layout,
    infer_feature_cols,
    norm_well,
    pd,
    print,
    read_barcode_platemap,
    use_checkpoints_input,
    validate_checkpoint_df,
):
    _ready_source = (
        SC_READY_PARQUET
        if SC_READY_PARQUET.exists()
        else LEGACY_SC_READY_PARQUET
    )
    _reuse_ready = use_checkpoints_input.value and checkpoint_matches_plate_scope(
        _ready_source, analysis_plates
    )
    if _reuse_ready:
        import pandas as _pd

        df_ready = _pd.read_parquet(_ready_source)
        validate_checkpoint_df(df_ready, "ready", infer_feature_cols(df_ready))
        print(f"✓ Reused single-cell output: {_ready_source}")
        if _ready_source == LEGACY_SC_READY_PARQUET:
            df_ready.to_parquet(SC_READY_PARQUET, index=False)
            print(f"  ✓ Promoted legacy cache file to declared NB02 output: {SC_READY_PARQUET}")
    else:
        if use_checkpoints_input.value and SC_READY_PARQUET.exists():
            print("  ℹ️  Invalidated ready checkpoint because its plate scope changed")
        barcode_index_for_exclusion = read_barcode_platemap(BARCODE_PLATEMAP_CSV)
        missing_pairs = find_wells_missing_from_layout(
            df_annotated, barcode_index_for_exclusion, PLATEMAP_DIR
        )
        if missing_pairs:
            print(f"⚠️  {len(missing_pairs)} well(s) not in platemap layout — will be removed:")
            for _plate, _well in missing_pairs[:20]:
                print(f"    {_plate}  {_well}")
        else:
            print("✓  All wells present in platemap layout")

        exclude_wells: dict[str, list[str]] = {}
        _invalid_exclusion_lines = []
        for _line in exclude_wells_input.value.splitlines():
            _line = _line.strip()
            if not _line:
                continue
            if ":" not in _line:
                _invalid_exclusion_lines.append(_line)
                continue
            _plate, _well = _line.split(":", 1)
            exclude_wells.setdefault(_plate.strip(), []).append(_well.strip())
        if _invalid_exclusion_lines:
            print("⚠ Ignored malformed manual exclusions; expected one Plate:Well pair per line:")
            for _line in _invalid_exclusion_lines:
                print(f"    - {_line}")

        df_ready = df_annotated.copy()
        df_ready["__well_norm__"] = df_ready["Metadata_Well"].map(norm_well)
        remove_mask = pd.Series(False, index=df_ready.index)
        for _plate, _well in missing_pairs:
            remove_mask |= (df_ready["Metadata_Plate"] == _plate) & (df_ready["__well_norm__"] == _well)
        for _plate, _wells in exclude_wells.items():
            _wells_norm = [norm_well(w) for w in _wells]
            remove_mask |= (df_ready["Metadata_Plate"] == _plate) & (df_ready["__well_norm__"].isin(_wells_norm))

        n_removed = int(remove_mask.sum())
        df_ready = df_ready.loc[~remove_mask].drop(columns="__well_norm__", errors="ignore")
        df_ready.to_parquet(SC_READY_PARQUET, index=False)
        print(f"  Removed {n_removed} rows → {df_ready.shape[0]:,} cells remaining")
        print(f"✓ Declared single-cell output written: {SC_READY_PARQUET}")

    _migrate_count_metadata = CELL_COUNT_METADATA_COLUMN not in df_ready.columns
    df_ready = add_cell_count_metadata(df_ready)
    if _migrate_count_metadata:
        df_ready.to_parquet(SC_READY_PARQUET, index=False)
        print(f"  ✓ Added {CELL_COUNT_METADATA_COLUMN} to ready checkpoint")
    _ready_wells = df_ready.groupby(["Metadata_Plate", "Metadata_Well"]).ngroups
    print(f"✓ Ready for aggregation: {len(df_ready):,} cells across {_ready_wells:,} wells")
    return (df_ready,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — Aggregate per well (median)

    ### SC-07 — One profile per observed well

    **What it evaluates:** whether median aggregation produces exactly one row
    for every plate/well pair retained after annotation and exclusions.

    **How to read it:** PASS means no wells were silently lost or duplicated.
    A mismatch is blocking because downstream normalization assumes one profile
    per well. If it fails, inspect missing or inconsistent grouping metadata;
    it does not by itself indicate a biological QC failure.
    """)
    return


@app.cell
def _(
    CELL_COUNT_METADATA_COLUMN,
    PW_AGGREGATED_PARQUET,
    RESOLVED_CONFIG,
    analysis_plates,
    checkpoint_matches_plate_scope,
    df_ready,
    infer_feature_cols,
    print,
    use_checkpoints_input,
    validate_checkpoint_df,
):
    _reuse_aggregated = use_checkpoints_input.value and checkpoint_matches_plate_scope(
        PW_AGGREGATED_PARQUET, analysis_plates, plate_col=RESOLVED_CONFIG.plate_col
    )
    if _reuse_aggregated:
        import pandas as _pd

        df_aggregated = _pd.read_parquet(PW_AGGREGATED_PARQUET)
        if CELL_COUNT_METADATA_COLUMN not in df_aggregated.columns:
            _well_keys = [RESOLVED_CONFIG.plate_col, RESOLVED_CONFIG.well_col]
            _count_lookup = df_ready[_well_keys + [CELL_COUNT_METADATA_COLUMN]].drop_duplicates(_well_keys)
            df_aggregated = df_aggregated.merge(
                _count_lookup, on=_well_keys, how="left", validate="one_to_one"
            )
            df_aggregated.to_parquet(PW_AGGREGATED_PARQUET, index=False)
            print(f"  ✓ Added {CELL_COUNT_METADATA_COLUMN} to aggregated checkpoint")
        validate_checkpoint_df(df_aggregated, "aggregated", infer_feature_cols(df_aggregated))
        print(f"✓ Reused checkpoint: {PW_AGGREGATED_PARQUET}")
    else:
        if use_checkpoints_input.value and PW_AGGREGATED_PARQUET.exists():
            print("  ℹ️  Invalidated aggregated checkpoint because its plate scope changed")
        _candidate_strata = [
            RESOLVED_CONFIG.plate_col,
            RESOLVED_CONFIG.well_col,
            RESOLVED_CONFIG.control_type_col,
            RESOLVED_CONFIG.treatment_col,
        ]
        if RESOLVED_CONFIG.has_dose_axis:
            _candidate_strata.append(RESOLVED_CONFIG.concentration_col)
        if RESOLVED_CONFIG.has_time_axis:
            _candidate_strata.append(RESOLVED_CONFIG.time_col)
        # Cell_Type isn't part of ExperimentConfig's core schema (not every
        # experiment has multiple cell lines), but when the platemap supplied
        # one it's real experiment metadata and must survive aggregation.
        _candidate_strata.append("Metadata_Cell_Type")
        _candidate_strata.append(CELL_COUNT_METADATA_COLUMN)
        strata_cols = [c for c in _candidate_strata if c and c in df_ready.columns]

        agg_feature_cols = infer_feature_cols(df_ready)
        print(f"Aggregating {len(agg_feature_cols)} features across {len(strata_cols)} strata columns...")

        df_aggregated = (
            df_ready[strata_cols + agg_feature_cols]
            .groupby(strata_cols, as_index=False)
            .median(numeric_only=True)
        )

        df_aggregated.to_parquet(PW_AGGREGATED_PARQUET, index=False)
        print(f"✓ Cached: {PW_AGGREGATED_PARQUET}")

    expected_wells = df_ready.groupby(
        [RESOLVED_CONFIG.plate_col, RESOLVED_CONFIG.well_col]
    ).ngroups
    print("── SC-07: One profile per observed well ──")
    print(f"  Aggregated rows: {df_aggregated.shape[0]:,}")
    print(f"  Expected wells: {expected_wells:,}")
    if df_aggregated.shape[0] != expected_wells:
        raise ValueError(
            f"SC-07 failed: aggregation produced {df_aggregated.shape[0]} rows for "
            f"{expected_wells} observed wells. Check missing/inconsistent grouping metadata."
        )
    sc07_status = "PASS"
    print("  ✓ SC-07 PASS — no wells were lost or duplicated during aggregation")
    return df_aggregated, expected_wells, sc07_status


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Clean and normalize (pycytominer `mad_robustize`)

    Two guards bracket the normalization step itself:

    - **SC-07b** (before): drops any feature whose negative-control MAD is
      near-zero in any plate — `mad_robustize` dividing by a near-zero MAD
      can blow that feature up to a numerically meaningless magnitude (seen
      on real data: ~1e19 for a feature that should be a small integer
      count), which then dominates any downstream correlation/distance
      computation. This must run *before* normalization: after the blow-up,
      the feature's variance looks artificially high, so SC-10's
      `variance_threshold` feature-selection step cannot catch it.
    - **SC-09b** (after): a safety net, independent of cause — drops any
      *normalized* feature whose magnitude is still implausibly large, in
      case something other than a near-zero MAD produces the same symptom.

    Both guards are **corrective warnings**: WARN means problematic features
    were found and removed so the notebook could continue safely. PASS means no
    feature crossed the configured threshold. When a validated checkpoint is
    reused, the summary reports REUSED because the earlier removal evidence is
    not recomputed. Neither check measures biological treatment activity.
    """)
    return


@app.cell
def _(
    CELL_COUNT_METADATA_COLUMN,
    NORM_CONTROL,
    PW_NORMALIZED_PARQUET,
    RESOLVED_CONFIG,
    analysis_plates,
    checkpoint_matches_plate_scope,
    clean_features_before_normalization,
    df_aggregated,
    drop_extreme_magnitude_features,
    drop_near_zero_mad_features,
    enforce_pascalcase_metadata_columns,
    infer_feature_cols,
    mad_epsilon_input,
    max_missing_fraction_input,
    max_normalized_magnitude_input,
    normalize_per_plate_mad_robustize,
    print,
    use_checkpoints_input,
    validate_checkpoint_df,
):
    _reuse_normalized = use_checkpoints_input.value and checkpoint_matches_plate_scope(
        PW_NORMALIZED_PARQUET, analysis_plates, plate_col=RESOLVED_CONFIG.plate_col
    )
    if _reuse_normalized:
        import pandas as _pd

        df_normalized = _pd.read_parquet(PW_NORMALIZED_PARQUET)
        if CELL_COUNT_METADATA_COLUMN not in df_normalized.columns:
            _well_keys = [RESOLVED_CONFIG.plate_col, RESOLVED_CONFIG.well_col]
            _count_lookup = df_aggregated[_well_keys + [CELL_COUNT_METADATA_COLUMN]].drop_duplicates(_well_keys)
            df_normalized = df_normalized.merge(
                _count_lookup, on=_well_keys, how="left", validate="one_to_one"
            )
            df_normalized.to_parquet(PW_NORMALIZED_PARQUET, index=False)
            print(f"  ✓ Added {CELL_COUNT_METADATA_COLUMN} to normalized checkpoint")
        validate_checkpoint_df(df_normalized, "normalized", infer_feature_cols(df_normalized), strict=True)
        print(f"✓ Reused checkpoint: {PW_NORMALIZED_PARQUET}")
        sc07b_status = "REUSED"
        sc09b_status = "REUSED"
        n_near_zero_mad_features = None
        n_extreme_magnitude_features = None
    else:
        if use_checkpoints_input.value and PW_NORMALIZED_PARQUET.exists():
            print("  ℹ️  Invalidated normalized checkpoint because its plate scope changed")
        pre_norm_feature_cols = infer_feature_cols(df_aggregated)
        df_cleaned, cleaned_feature_cols, clean_summary = clean_features_before_normalization(
            df_aggregated, pre_norm_feature_cols, max_missing_fraction=float(max_missing_fraction_input.value),
        )
        print(
            f"  Removed {clean_summary['n_features_removed']} high-missingness features; "
            f"imputed {clean_summary['n_missing_before']} remaining NaN(s)."
        )

        print("── SC-07b: near-zero-MAD feature guard (pre-normalization) ──")
        df_cleaned, cleaned_feature_cols, near_zero_mad_report = drop_near_zero_mad_features(
            df_cleaned,
            cleaned_feature_cols,
            plate_col=RESOLVED_CONFIG.plate_col,
            control_col=RESOLVED_CONFIG.control_type_col,
            negcon_values=RESOLVED_CONFIG.negcon_values,
            norm_control=NORM_CONTROL,
            mad_epsilon=float(mad_epsilon_input.value),
        )
        _near_zero_mad_features = sorted({r["feature"] for r in near_zero_mad_report})
        if _near_zero_mad_features:
            sc07b_status = "WARN"
            print(f"  ⚠️  Dropped {len(_near_zero_mad_features)} feature(s) with near-zero control MAD:")
            for _feature in _near_zero_mad_features:
                _plates = [r["plate"] for r in near_zero_mad_report if r["feature"] == _feature]
                print(f"      - {_feature} (MAD ≈ 0 on: {', '.join(str(p) for p in _plates)})")
        else:
            sc07b_status = "PASS"
            print("  ✓  No near-zero-MAD features detected.")
        n_near_zero_mad_features = len(_near_zero_mad_features)

        df_normalized = normalize_per_plate_mad_robustize(
            df_cleaned,
            "infer",
            plate_col=RESOLVED_CONFIG.plate_col,
            control_col=RESOLVED_CONFIG.control_type_col,
            negcon_values=RESOLVED_CONFIG.negcon_values,
            norm_control=NORM_CONTROL,
        )
        df_normalized = enforce_pascalcase_metadata_columns(df_normalized)

        print("── SC-09b: extreme-magnitude feature guard (post-normalization) ──")
        _post_norm_feature_cols = infer_feature_cols(df_normalized)
        df_normalized, _remaining_feature_cols, extreme_magnitude_report = drop_extreme_magnitude_features(
            df_normalized, _post_norm_feature_cols, max_abs_value=float(max_normalized_magnitude_input.value),
        )
        if extreme_magnitude_report:
            sc09b_status = "WARN"
            print(f"  ⚠️  Dropped {len(extreme_magnitude_report)} feature(s) with implausible normalized magnitude:")
            for _feature, _max_abs in extreme_magnitude_report.items():
                print(f"      - {_feature} (max|value| = {_max_abs:.3e})")
        else:
            sc09b_status = "PASS"
            print("  ✓  No implausible post-normalization magnitudes detected.")
        n_extreme_magnitude_features = len(extreme_magnitude_report)

        df_normalized.to_parquet(PW_NORMALIZED_PARQUET, index=False)
        print(f"✓ Normalized and cached: {PW_NORMALIZED_PARQUET}  →  {df_normalized.shape}")
    print(
        f"✓ Normalized profile ready: {len(df_normalized):,} wells · "
        f"{len(infer_feature_cols(df_normalized)):,} features"
    )
    return (
        df_normalized,
        n_extreme_magnitude_features,
        n_near_zero_mad_features,
        sc07b_status,
        sc09b_status,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — Technical structure and replicate consistency

    ### SC-08 — Residual plate structure after normalization

    The paired PCA plots compare profiles before and after normalization. Plate
    silhouette is reported only when at least two plates are available: values
    near zero indicate better mixing, while positive separation suggests
    residual plate-associated structure. This is an **advisory diagnostic**, not
    an automatic batch-correction decision and not evidence of treatment effect.

    ### SC-09 — Within-condition replicate consistency

    This check groups wells by treatment (and dose when available), then reports
    median pairwise profile correlation and a complementary coefficient of
    variation. PASS means every evaluable group meets the configured correlation
    threshold; WARN identifies groups to inspect. SKIP means no group has at
    least two replicates. Correlation measures consistency, not phenotypic
    activity or mechanism-of-action distinctiveness.
    """)
    return


@app.cell
def _(
    clean_features_before_normalization,
    df_aggregated,
    infer_feature_cols,
    max_missing_fraction_input,
):
    # Cleaned-but-not-yet-normalized view of the aggregated data, used only for
    # SC-08's "before normalization" panel below -- comparing PCA against the
    # raw aggregated data (which can still contain inf/NaN) would crash
    # StandardScaler, and wouldn't isolate the effect of normalization anyway.
    df_cleaned_for_diagnostic, _cleaned_feat_cols_diag, _clean_summary_diag = (
        clean_features_before_normalization(
            df_aggregated,
            infer_feature_cols(df_aggregated),
            max_missing_fraction=float(max_missing_fraction_input.value),
        )
    )
    return (df_cleaned_for_diagnostic,)


@app.cell
def _(
    FIGS_DIR,
    OVERWRITE_EXISTING_OUTPUTS,
    PCA,
    RESOLVED_CONFIG,
    StandardScaler,
    df_cleaned_for_diagnostic,
    df_normalized,
    infer_feature_cols,
    np,
    plt,
    print,
    silhouette_score,
):
    def _pca_coordinates(feature_matrix):
        if min(feature_matrix.shape) < 2:
            raise ValueError(
                "SC-08 requires at least two well profiles and two usable features for PCA; "
                f"received matrix shape {feature_matrix.shape}."
            )
        pca = PCA(n_components=2, random_state=42)
        scaled = StandardScaler().fit_transform(feature_matrix)
        return pca.fit_transform(scaled), pca.explained_variance_ratio_

    def _plot_pca_by_plate(feature_matrix, plate_labels, title, ax):
        coordinates, variance_ratio = _pca_coordinates(feature_matrix)
        for label in sorted(set(plate_labels)):
            mask = np.asarray(plate_labels) == label
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], label=label, s=60, alpha=0.8, edgecolors="k", linewidths=0.4)
        ax.set_xlabel(f"PC1 ({variance_ratio[0]:.1%})")
        ax.set_ylabel(f"PC2 ({variance_ratio[1]:.1%})")
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=0.8)

    feat_cols_norm = infer_feature_cols(df_normalized)
    normalized_values = df_normalized[feat_cols_norm].replace([np.inf, -np.inf], np.nan)
    if int(normalized_values.isna().sum().sum()) > 0:
        raise ValueError("Missing or infinite values remain in the normalized feature matrix.")

    plates_norm = df_normalized[RESOLVED_CONFIG.plate_col].astype(str).to_numpy()
    X_norm = normalized_values.to_numpy(dtype=float)

    feat_cols_raw = infer_feature_cols(df_cleaned_for_diagnostic)
    X_raw = df_cleaned_for_diagnostic[feat_cols_raw].to_numpy(dtype=float)
    plates_raw = df_cleaned_for_diagnostic[RESOLVED_CONFIG.plate_col].astype(str).to_numpy()

    fig_sc08, axes_sc08 = plt.subplots(1, 2, figsize=(13, 5))
    _plot_pca_by_plate(X_raw, plates_raw, "Before normalization", axes_sc08[0])
    _plot_pca_by_plate(X_norm, plates_norm, "After normalization", axes_sc08[1])
    fig_sc08.suptitle("SC-08: Plate-effect assessment", fontsize=13, fontweight="bold")
    fig_sc08.tight_layout()
    _sc08_path = FIGS_DIR / "sc08_plate_effect_pca.png"
    if _sc08_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        print(f"ℹ Existing SC-08 figure protected; current figure is displayed but not written: {_sc08_path}")
    else:
        _figure_action = "replaced" if _sc08_path.exists() else "created"
        fig_sc08.savefig(_sc08_path, dpi=150, bbox_inches="tight")
        print(f"✓ SC-08 figure {_figure_action} and displayed: {_sc08_path} ({_sc08_path.stat().st_size:,} bytes)")

    unique_plates = np.unique(plates_norm)
    if len(unique_plates) > 1 and X_norm.shape[0] > len(unique_plates):
        plate_silhouette = silhouette_score(X_norm, plates_norm)
        sc08_status = "INFO"
        print(f"  SC-08 INFO — silhouette by plate: {plate_silhouette:.3f} (near 0 = better mixing)")
    else:
        plate_silhouette = None
        sc08_status = "SKIP"
        print("  SC-08 SKIP — silhouette requires at least two plates and enough well profiles")
    fig_sc08
    return feat_cols_norm, plate_silhouette, sc08_status


@app.cell
def _(
    CV_SUMMARY_CSV,
    FIGS_DIR,
    OVERWRITE_EXISTING_OUTPUTS,
    RESOLVED_CONFIG,
    WITHIN_GROUP_VARIABILITY_CSV,
    analysis_plates,
    corr_warn_threshold_input,
    cv_warn_threshold_input,
    df_normalized,
    feat_cols_norm,
    mo,
    np,
    pd,
    plt,
    print,
    write_summary_table_protected,
):
    replicate_group_cols = [RESOLVED_CONFIG.treatment_col]
    if RESOLVED_CONFIG.has_dose_axis:
        replicate_group_cols.append(RESOLVED_CONFIG.concentration_col)
    missing_group_columns = [c for c in replicate_group_cols if c not in df_normalized.columns]
    if missing_group_columns:
        raise ValueError(f"Replicate grouping columns are missing: {missing_group_columns}")

    correlation_rows = []
    cv_rows = []
    for group_values, group_df in df_normalized.groupby(replicate_group_cols, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_metadata = dict(zip(replicate_group_cols, group_values))
        n_replicates = len(group_df)

        correlation_record = {
            **group_metadata, "n_replicates": n_replicates,
            "median_pairwise_corr": np.nan, "mean_pairwise_corr": np.nan,
        }
        if n_replicates >= 2:
            group_matrix = group_df[feat_cols_norm].to_numpy(dtype=float)
            correlation_matrix = np.corrcoef(group_matrix)
            upper = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            correlation_record.update(
                median_pairwise_corr=np.nanmedian(upper), mean_pairwise_corr=np.nanmean(upper)
            )
        correlation_rows.append(correlation_record)

        group_cv = (group_df[feat_cols_norm].std() / group_df[feat_cols_norm].mean().abs()).replace(
            [np.inf, -np.inf], np.nan
        )
        cv_rows.append({**group_metadata, "median_CV": group_cv.median()})

    corr_df = pd.DataFrame(correlation_rows).sort_values("median_pairwise_corr", ascending=False, na_position="last")
    cv_summary = pd.DataFrame(cv_rows).sort_values("median_CV", ascending=False, na_position="last")

    variability_combined = corr_df.merge(cv_summary, on=replicate_group_cols, how="outer")
    try:
        _cv_status = write_summary_table_protected(
            cv_summary, CV_SUMMARY_CSV, overwrite=OVERWRITE_EXISTING_OUTPUTS
        )
        _variability_status = write_summary_table_protected(
            variability_combined,
            WITHIN_GROUP_VARIABILITY_CSV,
            overwrite=OVERWRITE_EXISTING_OUTPUTS,
        )
    except FileExistsError as _error:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**A protected NB02 summary differs from the current result.**\n\n"
                    f"`{_error}`\n\nReview the changed configuration/data. If replacement is intended, "
                    "enable **Overwrite existing outputs** under Advanced settings and rerun."
                ),
                kind="danger",
            ),
        )
    print(f"✓ CV summary {_cv_status}: {CV_SUMMARY_CSV}")
    print(f"✓ Within-group variability table {_variability_status}: {WITHIN_GROUP_VARIABILITY_CSV}")

    corr_warn_threshold = float(corr_warn_threshold_input.value)
    cv_warn_threshold = float(cv_warn_threshold_input.value)
    evaluable_correlation = corr_df[corr_df["median_pairwise_corr"].notna()]
    low_correlation = evaluable_correlation[
        evaluable_correlation["median_pairwise_corr"] < corr_warn_threshold
    ]
    if evaluable_correlation.empty:
        sc09_status = "SKIP"
        print("SC-09 SKIP — no condition has at least two replicate wells")
    elif not low_correlation.empty:
        sc09_status = "WARN"
        print(f"⚠️  {len(low_correlation)} condition(s) below correlation threshold {corr_warn_threshold:.2f}")
    else:
        sc09_status = "PASS"
        print(f"✓  All evaluable conditions have median pairwise correlation ≥ {corr_warn_threshold:.2f}")

    def _condition_label(frame):
        return frame[replicate_group_cols].astype(str).agg(" | ".join, axis=1)

    corr_plot = corr_df.copy()
    cv_plot = cv_summary.copy()
    corr_plot["Condition"] = _condition_label(corr_plot)
    cv_plot["Condition"] = _condition_label(cv_plot)

    fig_sc09, axes_sc09 = plt.subplots(1, 2, figsize=(15, 5))
    axes_sc09[0].barh(corr_plot["Condition"], corr_plot["median_pairwise_corr"])
    axes_sc09[0].axvline(corr_warn_threshold, linestyle="--", linewidth=1.2, label=f"Alert ({corr_warn_threshold:.2f})")
    axes_sc09[0].set_xlabel("Median pairwise correlation")
    axes_sc09[0].set_title("Replicate correlation")
    axes_sc09[0].legend(fontsize=8)

    axes_sc09[1].barh(cv_plot["Condition"], cv_plot["median_CV"])
    axes_sc09[1].axvline(cv_warn_threshold, linestyle="--", linewidth=1.2, label=f"Reference ({cv_warn_threshold})")
    axes_sc09[1].set_xlabel("Median CV (interpret cautiously after RobustMAD)")
    axes_sc09[1].set_title("CV — complementary only")
    axes_sc09[1].legend(fontsize=8)

    fig_sc09.tight_layout()
    _sc09_path = FIGS_DIR / "sc09_within_group_variability.png"
    if _sc09_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        print(f"ℹ Existing SC-09 figure protected; current figure is displayed but not written: {_sc09_path}")
    else:
        _figure_action = "replaced" if _sc09_path.exists() else "created"
        fig_sc09.savefig(_sc09_path, dpi=150, bbox_inches="tight")
        print(f"✓ SC-09 figure {_figure_action} and displayed: {_sc09_path} ({_sc09_path.stat().st_size:,} bytes)")
    fig_sc09
    return corr_df, cv_summary, low_correlation, sc09_status, variability_combined


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — Feature selection (pycytominer `feature_select`)

    ### SC-10 — Feature-retention check

    **What it evaluates:** how many normalized morphology features remain after
    removing low-variance, highly correlated, or otherwise unsuitable features.

    **How to read it:** PASS means at least 50 usable features remain. WARN means
    fewer than 50 remain and downstream models may be unstable or biologically
    narrow; review the earlier missingness and normalization guards before
    continuing. The threshold is advisory and does not measure phenotype
    activity. An empty profile or a matrix containing missing/infinite values is
    a blocking failure.
    """)
    return


@app.cell
def _(
    CELL_COUNT_METADATA_COLUMN,
    OVERWRITE_EXISTING_OUTPUTS,
    PW_FEATURES_SELECTED_CSV,
    PW_FEATURES_SELECTED_PARQUET,
    analysis_plates,
    checkpoint_matches_plate_scope,
    df_normalized,
    enforce_pascalcase_metadata_columns,
    feat_cols_norm,
    infer_feature_cols,
    mo,
    print,
    select_features,
    technical_identifier_columns,
    use_checkpoints_input,
    validate_checkpoint_df,
    write_csv_protected,
    write_parquet_protected,
):
    _feature_scope_matches = checkpoint_matches_plate_scope(
        PW_FEATURES_SELECTED_PARQUET, analysis_plates
    )
    import pandas as _pd

    _checkpoint_identifiers = []
    if use_checkpoints_input.value and _feature_scope_matches and PW_FEATURES_SELECTED_PARQUET.exists():
        _checkpoint_schema = _pd.read_parquet(PW_FEATURES_SELECTED_PARQUET)
        _checkpoint_identifiers = technical_identifier_columns(_checkpoint_schema)
    _feature_checkpoint_current = _feature_scope_matches and not _checkpoint_identifiers

    if use_checkpoints_input.value and _feature_checkpoint_current and not OVERWRITE_EXISTING_OUTPUTS:

        df_feature_selected = _pd.read_parquet(PW_FEATURES_SELECTED_PARQUET)
        if CELL_COUNT_METADATA_COLUMN not in df_feature_selected.columns:
            _well_keys = ["Metadata_Plate", "Metadata_Well"]
            _count_lookup = df_normalized[_well_keys + [CELL_COUNT_METADATA_COLUMN]].drop_duplicates(_well_keys)
            df_feature_selected = df_feature_selected.merge(
                _count_lookup, on=_well_keys, how="left", validate="one_to_one"
            )
            df_feature_selected.to_parquet(PW_FEATURES_SELECTED_PARQUET, index=False)
            print(f"  ✓ Added {CELL_COUNT_METADATA_COLUMN} to feature-selected checkpoint")
        validate_checkpoint_df(
            df_feature_selected, "feature_selected", infer_feature_cols(df_feature_selected), strict=True,
        )
        print(f"✓ Reused checkpoint: {PW_FEATURES_SELECTED_PARQUET}")
    else:
        if use_checkpoints_input.value and PW_FEATURES_SELECTED_PARQUET.exists() and not _feature_scope_matches:
            print("  ℹ️  Invalidated feature-selected checkpoint because its plate scope changed")
        if _checkpoint_identifiers:
            print("  ℹ️  Invalidated legacy feature-selected checkpoint: technical identifier columns were detected")
            print(f"     Columns to remove: {len(_checkpoint_identifiers):,}")
        df_feature_selected = select_features(df_normalized, "infer")

        _selected_identifiers = technical_identifier_columns(df_feature_selected)
        if _selected_identifiers:
            df_feature_selected = df_feature_selected.drop(columns=_selected_identifiers)
            print(f"  ✓ Removed {len(_selected_identifiers):,} technical identifier column(s) from the final profile")

        df_feature_selected = enforce_pascalcase_metadata_columns(df_feature_selected)

        final_feature_cols = infer_feature_cols(df_feature_selected)
        if df_feature_selected.empty:
            raise ValueError("The final profile contains no wells.")
        if not final_feature_cols:
            raise ValueError("The final profile contains no features.")

        final_values = df_feature_selected[final_feature_cols].replace([float("inf"), float("-inf")], None)
        if int(final_values.isna().sum().sum()) > 0:
            raise ValueError("The final feature matrix still contains missing/inf values.")

        required_meta = [
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_Treatment",
            CELL_COUNT_METADATA_COLUMN,
        ]
        missing_meta = [c for c in required_meta if c not in df_feature_selected.columns]
        if missing_meta:
            raise ValueError(f"Required metadata columns missing: {missing_meta}")

        try:
            export_status = write_parquet_protected(
                df_feature_selected,
                PW_FEATURES_SELECTED_PARQUET,
                overwrite=(
                    OVERWRITE_EXISTING_OUTPUTS
                    or not _feature_scope_matches
                    or bool(_checkpoint_identifiers)
                ),
            )
        except FileExistsError as _error:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        "**The protected final Parquet differs from the current result.**\n\n"
                        f"`{_error}`\n\nReview the changed configuration/data. If replacement is intended, "
                        "enable **Overwrite existing outputs** under Advanced settings and rerun."
                    ),
                    kind="danger",
                ),
            )
        _status_label = {"created": "CREATED", "unchanged": "ALREADY CURRENT", "replaced": "REPLACED"}.get(export_status, export_status)
        print(f"✓ Feature-selected profile: {_status_label}")
        print(f"  File: {PW_FEATURES_SELECTED_PARQUET}")
        print(f"  Verified: {PW_FEATURES_SELECTED_PARQUET.stat().st_size:,} bytes")

    final_feature_cols = infer_feature_cols(df_feature_selected)
    n_before = len(feat_cols_norm)
    n_after = len(final_feature_cols)
    if df_feature_selected.empty:
        raise ValueError("The final profile contains no wells.")
    if not final_feature_cols:
        raise ValueError("The final profile contains no features.")
    final_values = df_feature_selected[final_feature_cols].replace(
        [float("inf"), float("-inf")], None
    )
    if int(final_values.isna().sum().sum()) > 0:
        raise ValueError("The final feature matrix still contains missing/inf values.")

    print("── SC-10: Feature-retention check ──")
    print(f"  Features before:  {n_before:,}")
    print(f"  Features after:   {n_after:,}")
    print(f"  Features removed: {n_before - n_after:,} ({(n_before - n_after) / n_before:.1%})")
    sc10_status = "WARN" if n_after < 50 else "PASS"
    if sc10_status == "WARN":
        print("  SC-10 WARN — fewer than 50 features remain; review earlier cleaning guards")
    else:
        print("  ✓ SC-10 PASS — sufficient usable features remain for downstream profiling")

    try:
        csv_export_status = write_csv_protected(
            df_feature_selected,
            PW_FEATURES_SELECTED_CSV,
            overwrite=OVERWRITE_EXISTING_OUTPUTS,
        )
    except FileExistsError as _error:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**The protected interoperability CSV differs from the current result.**\n\n"
                    f"`{_error}`\n\nEnable **Overwrite existing outputs** only after confirming that "
                    "the new schema is expected."
                ),
                kind="danger",
            ),
        )
    print(f"✓ Interoperability CSV {csv_export_status}: {PW_FEATURES_SELECTED_CSV}")
    return df_feature_selected, sc10_status


@app.cell
def _(
    expected_wells,
    low_correlation,
    n_extreme_magnitude_features,
    n_missing_treatment,
    n_near_zero_mad_features,
    plate_silhouette,
    sc06_status,
    sc07_status,
    sc07b_status,
    sc08_status,
    sc09_status,
    sc09b_status,
    sc10_status,
    mo,
    pd,
):
    _near_zero_evidence = (
        "Validated normalized checkpoint reused; original removal evidence was not recomputed"
        if n_near_zero_mad_features is None
        else f"{n_near_zero_mad_features:,} near-zero-control-MAD feature(s) detected and removed"
    )
    _extreme_evidence = (
        "Validated normalized checkpoint reused; original removal evidence was not recomputed"
        if n_extreme_magnitude_features is None
        else f"{n_extreme_magnitude_features:,} extreme-magnitude feature(s) detected and removed"
    )
    sanity_check_summary = pd.DataFrame(
        [
            {
                "Check": "SC-06 — Treatment annotation coverage",
                "Status": sc06_status,
                "Evidence": f"{n_missing_treatment:,} cell(s) without treatment annotation",
                "How to act": "Review barcode/platemap mappings if WARN",
            },
            {
                "Check": "SC-07 — One profile per observed well",
                "Status": sc07_status,
                "Evidence": f"Aggregation retained all {expected_wells:,} observed well(s)",
                "How to act": "Blocking if counts differ; inspect grouping metadata",
            },
            {
                "Check": "SC-07b — Near-zero control MAD guard",
                "Status": sc07b_status,
                "Evidence": _near_zero_evidence,
                "How to act": "Review removed features and negative-control coverage if WARN",
            },
            {
                "Check": "SC-08 — Residual plate structure",
                "Status": sc08_status,
                "Evidence": (
                    f"Post-normalization plate silhouette = {plate_silhouette:.3f}"
                    if plate_silhouette is not None
                    else "Plate silhouette unavailable for a single plate or insufficient samples"
                ),
                "How to act": "Interpret plate separation; do not automatically batch-correct",
            },
            {
                "Check": "SC-09 — Replicate consistency",
                "Status": sc09_status,
                "Evidence": f"{len(low_correlation):,} evaluable condition(s) below the correlation threshold",
                "How to act": "Inspect low-consistency conditions and replicate quality",
            },
            {
                "Check": "SC-09b — Extreme normalized magnitude guard",
                "Status": sc09b_status,
                "Evidence": _extreme_evidence,
                "How to act": "Review normalization and removed features if WARN",
            },
            {
                "Check": "SC-10 — Feature retention",
                "Status": sc10_status,
                "Evidence": "Final feature matrix is finite and non-empty",
                "How to act": "Review cleaning thresholds if fewer than 50 features remain",
            },
        ]
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Sanity-check interpretation summary

                PASS confirms the stated technical condition, WARN identifies a
                review item or a corrective removal, INFO is descriptive, SKIP
                means the check is not valid for this design, and REUSED means a
                validated checkpoint was loaded without recomputing its original
                removal evidence. These checks do not establish phenotypic activity.
                """
            ),
            sanity_check_summary,
        ]
    )
    return (sanity_check_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 — Final summary and provenance
    """)
    return


@app.cell
def _(df_feature_selected, infer_feature_cols, print):
    feat_cols_final = infer_feature_cols(df_feature_selected)
    print("═" * 60)
    print("NB02 complete")
    print("═" * 60)
    print(f"  Wells    : {df_feature_selected.shape[0]}")
    print(f"  Features : {len(feat_cols_final)}")
    print(f"  Treatments: {df_feature_selected['Metadata_Treatment'].nunique()}")
    print()
    print("Treatment distribution:")
    print(df_feature_selected["Metadata_Treatment"].value_counts().to_string())
    print()
    print("Next step: NB03 — Quality Metrics (Go/No-Go gate)")
    return (feat_cols_final,)


@app.cell
def _(
    CONFIG,
    CV_SUMMARY_CSV,
    EXPERIMENT_ID,
    INPUT_PARQUET,
    NEGCON_VALUES,
    NORM_CONTROL,
    PW_FEATURES_SELECTED_CSV,
    PW_FEATURES_SELECTED_PARQUET,
    REPO_ROOT,
    RESULTS_DIR,
    SC_READY_PARQUET,
    WITHIN_GROUP_VARIABILITY_CSV,
    analysis_plates,
    corr_warn_threshold_input,
    df_feature_selected,
    feat_cols_final,
    json,
    max_missing_fraction_input,
    platform,
    print,
    subprocess,
):
    from datetime import datetime, timezone
    from hca_pipeline.provenance import canonicalize_provenance, provenance_json

    def run_git_command(arguments, repo_root):
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=str(repo_root), capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None if result.returncode == 0 else None

    provenance_nb02 = {
        "schema_version": 2,
        "pipeline": {
            "notebook": "02_aggregate_normalize_featureselect.py",
            "experiment_id": EXPERIMENT_ID,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "configuration": {
            "negcon_values": NEGCON_VALUES,
            "norm_control": NORM_CONTROL,
            "max_missing_fraction": float(max_missing_fraction_input.value),
            "correlation_warn_threshold": float(corr_warn_threshold_input.value),
            "treatment_col": CONFIG.treatment_col,
            "concentration_col": CONFIG.concentration_col,
            "has_dose_axis": CONFIG.has_dose_axis,
            "analysis_mode": CONFIG.analysis_mode,
            "overwrite_existing_outputs": CONFIG.overwrite_existing_outputs,
            "included_plates": analysis_plates,
            "excluded_plate_reasons": CONFIG.excluded_plate_reasons,
            "required_reference_treatments": CONFIG.required_reference_treatments,
        },
        "dataset": {
            "n_wells": int(df_feature_selected.shape[0]),
            "n_features": int(len(feat_cols_final)),
            "n_plates": int(df_feature_selected[CONFIG.plate_col].nunique()),
        },
        "version_control": {
            "git_commit": (run_git_command(["rev-parse", "HEAD"], REPO_ROOT) or "unknown"),
        },
        "environment": {"python_version": platform.python_version()},
        "outputs": {
            "per_well_features_selected_parquet": str(PW_FEATURES_SELECTED_PARQUET),
            "per_well_features_selected_csv": str(PW_FEATURES_SELECTED_CSV),
            "cv_summary_csv": str(CV_SUMMARY_CSV),
            "within_group_variability_csv": str(WITHIN_GROUP_VARIABILITY_CSV),
        },
    }
    _declared_outputs = [
        SC_READY_PARQUET,
        PW_FEATURES_SELECTED_PARQUET,
        PW_FEATURES_SELECTED_CSV,
        CV_SUMMARY_CSV,
        WITHIN_GROUP_VARIABILITY_CSV,
    ]
    provenance_nb02 = canonicalize_provenance(
        provenance_nb02,
        notebook="02_aggregate_normalize_featureselect.py",
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
        dependencies=[INPUT_PARQUET],
        outputs=_declared_outputs,
    )
    _provenance_payload = provenance_json(provenance_nb02)
    provenance_nb02_path = RESULTS_DIR / "provenance_nb02.json"
    if provenance_nb02_path.exists() and not CONFIG.overwrite_existing_outputs:
        _provenance_status = "unchanged (existing provenance protected)"
    else:
        _provenance_status = "replaced" if provenance_nb02_path.exists() else "created"
        provenance_nb02_path.write_text(_provenance_payload, encoding="utf-8")
    print(f"✓ Provenance {_provenance_status}: {provenance_nb02_path}")
    print(f"  Schema v{provenance_nb02['schema_version']} · inputs hashed: {len(provenance_nb02['dependencies'])} · outputs recorded: {len(provenance_nb02['outputs'])}")

    if CONFIG.save_provenance_history:
        _timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _history_path = RESULTS_DIR / f"provenance_nb02_{_timestamp}.json"
        if _history_path.exists():
            raise FileExistsError(f"Historical provenance file already exists: {_history_path}")
        _history_path.write_text(_provenance_payload, encoding="utf-8")
        print(f"✓ Historical record: {_history_path}")
    else:
        print("  Historical record: disabled")
    return (provenance_nb02_path,)


@app.cell
def _(CV_SUMMARY_CSV, FIGS_DIR, PW_FEATURES_SELECTED_CSV, PW_FEATURES_SELECTED_PARQUET, WITHIN_GROUP_VARIABILITY_CSV, df_feature_selected, feat_cols_final, mo, print, provenance_nb02_path):
    _required = [PW_FEATURES_SELECTED_PARQUET, PW_FEATURES_SELECTED_CSV, CV_SUMMARY_CSV, WITHIN_GROUP_VARIABILITY_CSV, FIGS_DIR / "sc08_plate_effect_pca.png", FIGS_DIR / "sc09_within_group_variability.png", provenance_nb02_path]
    _missing = [path for path in _required if not path.is_file()]
    mo.stop(bool(_missing), mo.callout(mo.md("**NB02 did not complete successfully.**\n\nMissing required outputs:\n" + "\n".join(f"- `{path}`" for path in _missing) + "\n\nDo not continue to NB03."), kind="danger"))
    print("═" * 72)
    print("NB02 COMPLETED — ALL REQUIRED OUTPUTS VERIFIED")
    print("═" * 72)
    print(f"  Final wells: {len(df_feature_selected):,} · features: {len(feat_cols_final):,}")
    for _path in _required:
        print(f"  ✓ {_path} ({_path.stat().st_size:,} bytes)")
    print("\n✓ Safe to continue to NB03")
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
    ## Save the analysis record

    Save two complementary snapshots of the notebook's **current session**:

    - **HTML** preserves the complete record, including code and rich outputs;
    - **PDF** is a clean reading copy without code inputs.

    Saving does not rerun cells or regenerate stochastic results. On Chromium,
    select `workspace/analysis/<experiment>/reports` the first time and the
    browser will remember it. Safari presents separate HTML and PDF download
    buttons because it cannot write directly to a chosen folder.
    """)
    return


@app.cell
def _(EXPERIMENT_ID, mo):
    from hca_pipeline.report_export import SessionReportSaver

    _report_saver = SessionReportSaver(
        basename=f"{EXPERIMENT_ID}_02_aggregate_normalize_featureselect",
        suggested_directory=f"workspace/analysis/{EXPERIMENT_ID}/reports",
    )
    mo.ui.anywidget(_report_saver)
    return


if __name__ == "__main__":
    app.run()
