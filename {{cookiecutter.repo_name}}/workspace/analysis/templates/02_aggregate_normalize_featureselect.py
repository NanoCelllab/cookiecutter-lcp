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

    Each stage below checks for its own checkpoint parquet on disk and
    skips recomputation when one is already present — this replaces the
    original notebook's single "resume from the most advanced stage"
    ladder with N independent, always-correct read-cache-or-compute
    cells (a cleaner fit for marimo's reactive model, at the minor cost
    of re-validating — not recomputing — already-cached earlier stages
    on a full rerun).

    > The original notebook also had a ~48-cell block of one-off
    > scratch investigation and a self-contained image-QC mini-pipeline
    > entangled with this one. Neither is part of the documented 7-step
    > flow above; the image-QC toolkit now lives in
    > `hca_pipeline.image_qc` for use in a separate, optional notebook,
    > and the scratch cells were not ported (they were ad hoc debugging
    > of specific wells/plates, not reusable pipeline logic).
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

    from hca_pipeline.config import SUPPORTED_PLATE_FORMATS, ExperimentConfig, validate_configuration
    from hca_pipeline.feature_select import infer_feature_cols, select_features
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

    print(f"  ✓  Shared utilities loaded from hca_pipeline ({_pipelines_dir})")
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
        validate_checkpoint_df,
        validate_configuration,
        write_parquet_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Loads the experiment chosen (and configured) in NB01. Only
    NB02-specific parameters are asked here — plate format, channels,
    and control vocabulary already came from `experiment_config.json`.
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
    overwrite_input = mo.ui.checkbox(value=False, label="Overwrite existing outputs")
    save_history_input = mo.ui.checkbox(
        value=loaded_config.save_provenance_history,
        label="Save timestamped provenance history",
    )

    mo.vstack(
        [
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
        ]
    )
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
    replace,
    save_history_input,
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

    CONFIG = replace(
        loaded_config,
        experiment_id=EXPERIMENT_ID,
        negcon_values=NEGCON_VALUES,
        save_provenance_history=bool(save_history_input.value),
    )
    CONFIG.save(REPO_ROOT)

    OVERWRITE_EXISTING_OUTPUTS = bool(overwrite_input.value)
    print(f"  Experiment ID:      {EXPERIMENT_ID}")
    print(f"  Negcon values:      {NEGCON_VALUES}")
    print(f"  Normalize against:  {NORM_CONTROL}")
    return (
        CONFIG,
        EXPERIMENT_ID,
        NEGCON_VALUES,
        NORM_CONTROL,
        OVERWRITE_EXISTING_OUTPUTS,
    )


@app.cell
def _(EXPERIMENT_ID, REPO_ROOT):
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
    SC_READY_PARQUET = CACHE_DIR / "single_cell_ready.parquet"
    PW_AGGREGATED_PARQUET = CACHE_DIR / "per_well_aggregated.parquet"
    PW_NORMALIZED_PARQUET = CACHE_DIR / "per_well_normalized.parquet"
    PW_FEATURES_SELECTED_PARQUET = PROFILES_OUT_DIR / "per_well_features_selected.parquet"

    CV_SUMMARY_CSV = RESULTS_DIR / "cv_summary.csv"
    WITHIN_GROUP_VARIABILITY_CSV = RESULTS_DIR / "within_group_variability.csv"

    for _d in (ANALYSIS_OUT_DIR, PROFILES_OUT_DIR, FIGS_DIR, CACHE_DIR, RESULTS_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    print(f"  Input:  {INPUT_PARQUET}")
    print(f"  Output: {PW_FEATURES_SELECTED_PARQUET}")
    return (
        BARCODE_PLATEMAP_CSV,
        CV_SUMMARY_CSV,
        FIGS_DIR,
        INPUT_CSV,
        INPUT_PARQUET,
        PLATEMAP_DIR,
        PW_AGGREGATED_PARQUET,
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
    """)
    return


@app.cell
def _(INPUT_CSV, INPUT_PARQUET, add_cell_count_metadata, dedupe_meta, ensure_core_metadata, pd):
    if INPUT_PARQUET.exists():
        df_loaded = pd.read_parquet(INPUT_PARQUET)
        print(f"  Loaded Parquet input: {INPUT_PARQUET}")
    elif INPUT_CSV.exists():
        df_loaded = pd.read_csv(INPUT_CSV, low_memory=False)
        print(f"  Loaded CSV fallback: {INPUT_CSV}")
    else:
        raise FileNotFoundError(f"Input file not found. Expected one of:\n  {INPUT_PARQUET}\n  {INPUT_CSV}")

    df_loaded = add_cell_count_metadata(dedupe_meta(ensure_core_metadata(df_loaded)))
    print(f"  Shape: {df_loaded.shape}")
    return (df_loaded,)


@app.cell
def _(CONFIG, df_loaded, pd):
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

    Merges the single-cell/well profiles with each plate's platemap
    layout, then confirms every cell received a treatment annotation.
    """)
    return


@app.cell
def _(
    BARCODE_PLATEMAP_CSV,
    CELL_COUNT_METADATA_COLUMN,
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
def _(CONFIG, df_annotated):
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
        print("  SC-06: WARN")
    else:
        print("  ✓  All cells have treatment annotation")
        print("  SC-06: PASS")

    print(f"  Dose axis:  {'present (' + RESOLVED_CONFIG.concentration_col + ')' if RESOLVED_CONFIG.has_dose_axis else 'absent'}")
    print(f"  Time axis:  {'present (' + RESOLVED_CONFIG.time_col + ')' if RESOLVED_CONFIG.has_time_axis else 'absent'}")
    return (RESOLVED_CONFIG,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Remove wells not in the platemap layout

    Wells present in the data but missing from the platemap layout (or
    explicitly excluded) are dropped before aggregation.
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
    read_barcode_platemap,
    use_checkpoints_input,
    validate_checkpoint_df,
):
    _reuse_ready = use_checkpoints_input.value and checkpoint_matches_plate_scope(
        SC_READY_PARQUET, analysis_plates
    )
    if _reuse_ready:
        import pandas as _pd

        df_ready = _pd.read_parquet(SC_READY_PARQUET)
        validate_checkpoint_df(df_ready, "ready", infer_feature_cols(df_ready))
        print(f"✓ Reused checkpoint: {SC_READY_PARQUET}")
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
        for _line in exclude_wells_input.value.splitlines():
            _line = _line.strip()
            if not _line or ":" not in _line:
                continue
            _plate, _well = _line.split(":", 1)
            exclude_wells.setdefault(_plate.strip(), []).append(_well.strip())

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
        print(f"✓ Cached: {SC_READY_PARQUET}")

    _migrate_count_metadata = CELL_COUNT_METADATA_COLUMN not in df_ready.columns
    df_ready = add_cell_count_metadata(df_ready)
    if _migrate_count_metadata:
        df_ready.to_parquet(SC_READY_PARQUET, index=False)
        print(f"  ✓ Added {CELL_COUNT_METADATA_COLUMN} to ready checkpoint")
    return (df_ready,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — Aggregate per well (median) — SC-07
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

        expected_wells = df_ready.groupby([RESOLVED_CONFIG.plate_col, RESOLVED_CONFIG.well_col]).ngroups
        print("── SC-07: Aggregation shape ──")
        print(f"  Wells × features: {df_aggregated.shape[0]} × {len(agg_feature_cols)}")
        print(f"  Expected wells  : {expected_wells}")
        if df_aggregated.shape[0] != expected_wells:
            raise ValueError(
                f"Wells were lost during aggregation: got {df_aggregated.shape[0]}, expected "
                f"{expected_wells}. Check for missing/inconsistent metadata."
            )
        print("  ✓  Well count matches — SC-07: PASS")

        df_aggregated.to_parquet(PW_AGGREGATED_PARQUET, index=False)
        print(f"✓ Cached: {PW_AGGREGATED_PARQUET}")
    return (df_aggregated,)


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
            print(f"  ⚠️  Dropped {len(_near_zero_mad_features)} feature(s) with near-zero control MAD:")
            for _feature in _near_zero_mad_features:
                _plates = [r["plate"] for r in near_zero_mad_report if r["feature"] == _feature]
                print(f"      - {_feature} (MAD ≈ 0 on: {', '.join(str(p) for p in _plates)})")
        else:
            print("  ✓  No near-zero-MAD features detected.")

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
            print(f"  ⚠️  Dropped {len(extreme_magnitude_report)} feature(s) with implausible normalized magnitude:")
            for _feature, _max_abs in extreme_magnitude_report.items():
                print(f"      - {_feature} (max|value| = {_max_abs:.3e})")
        else:
            print("  ✓  No implausible post-normalization magnitudes detected.")

        df_normalized.to_parquet(PW_NORMALIZED_PARQUET, index=False)
        print(f"✓ Normalized and cached: {PW_NORMALIZED_PARQUET}  →  {df_normalized.shape}")
    return (df_normalized,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — SC-08: plate-effect PCA · SC-09: within-group variability
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
    PCA,
    RESOLVED_CONFIG,
    StandardScaler,
    df_cleaned_for_diagnostic,
    df_normalized,
    infer_feature_cols,
    np,
    plt,
    silhouette_score,
):
    def _pca_coordinates(feature_matrix):
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
    fig_sc08.savefig(FIGS_DIR / "sc08_plate_effect_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig_sc08)
    print(f"✓ Figure saved: {FIGS_DIR / 'sc08_plate_effect_pca.png'}")

    unique_plates = np.unique(plates_norm)
    if len(unique_plates) > 1 and X_norm.shape[0] > len(unique_plates):
        plate_silhouette = silhouette_score(X_norm, plates_norm)
        print(f"  Silhouette by plate after normalization: {plate_silhouette:.3f} (near 0 = better mixing)")
    else:
        print("  Silhouette skipped: single plate or too few samples.")
    fig_sc08
    return (feat_cols_norm,)


@app.cell
def _(
    CV_SUMMARY_CSV,
    FIGS_DIR,
    RESOLVED_CONFIG,
    WITHIN_GROUP_VARIABILITY_CSV,
    analysis_plates,
    corr_warn_threshold_input,
    cv_warn_threshold_input,
    df_normalized,
    feat_cols_norm,
    np,
    pd,
    plt,
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

    cv_summary.to_csv(CV_SUMMARY_CSV, index=False)
    print(f"✓ cv_summary.csv saved → {CV_SUMMARY_CSV}")

    variability_combined = corr_df.merge(cv_summary, on=replicate_group_cols, how="outer")
    variability_combined.to_csv(WITHIN_GROUP_VARIABILITY_CSV, index=False)
    print(f"✓ within_group_variability.csv saved → {WITHIN_GROUP_VARIABILITY_CSV}")

    corr_warn_threshold = float(corr_warn_threshold_input.value)
    cv_warn_threshold = float(cv_warn_threshold_input.value)
    low_correlation = corr_df[corr_df["median_pairwise_corr"] < corr_warn_threshold]
    if not low_correlation.empty:
        print(f"⚠️  {len(low_correlation)} condition(s) below correlation threshold {corr_warn_threshold:.2f}")
    else:
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
    fig_sc09.savefig(FIGS_DIR / "sc09_within_group_variability.png", dpi=150, bbox_inches="tight")
    plt.close(fig_sc09)
    print(f"✓ Figure saved: {FIGS_DIR / 'sc09_within_group_variability.png'}")
    fig_sc09
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — Feature selection (pycytominer `feature_select`) — SC-10
    """)
    return


@app.cell
def _(
    CELL_COUNT_METADATA_COLUMN,
    OVERWRITE_EXISTING_OUTPUTS,
    PW_FEATURES_SELECTED_PARQUET,
    analysis_plates,
    checkpoint_matches_plate_scope,
    df_normalized,
    enforce_pascalcase_metadata_columns,
    feat_cols_norm,
    infer_feature_cols,
    select_features,
    use_checkpoints_input,
    validate_checkpoint_df,
    write_parquet_protected,
):
    _feature_scope_matches = checkpoint_matches_plate_scope(
        PW_FEATURES_SELECTED_PARQUET, analysis_plates
    )
    if use_checkpoints_input.value and _feature_scope_matches and not OVERWRITE_EXISTING_OUTPUTS:
        import pandas as _pd

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
        n_before = len(feat_cols_norm)
        df_feature_selected = select_features(df_normalized, "infer")
        final_feature_cols = infer_feature_cols(df_feature_selected)
        n_after = len(final_feature_cols)

        print("── SC-10: Feature selection report ──")
        print(f"  Features before : {n_before}")
        print(f"  Features after  : {n_after}")
        print(f"  Features removed: {n_before - n_after} ({(n_before - n_after) / n_before:.1%})")
        print("  SC-10: WARN — very few features remaining" if n_after < 50 else "  SC-10: PASS")

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

        export_status = write_parquet_protected(
            df_feature_selected,
            PW_FEATURES_SELECTED_PARQUET,
            overwrite=OVERWRITE_EXISTING_OUTPUTS or not _feature_scope_matches,
        )
        print(f"✓ Feature-selected profile {export_status}: {PW_FEATURES_SELECTED_PARQUET}")
    return (df_feature_selected,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 — Final summary and provenance
    """)
    return


@app.cell
def _(df_feature_selected, infer_feature_cols):
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
    NEGCON_VALUES,
    NORM_CONTROL,
    PW_FEATURES_SELECTED_PARQUET,
    REPO_ROOT,
    RESULTS_DIR,
    WITHIN_GROUP_VARIABILITY_CSV,
    corr_warn_threshold_input,
    df_feature_selected,
    feat_cols_final,
    json,
    max_missing_fraction_input,
    platform,
    subprocess,
):
    from datetime import datetime, timezone

    def run_git_command(arguments, repo_root):
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=str(repo_root), capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None if result.returncode == 0 else None

    provenance_nb02 = {
        "schema_version": 1,
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
            "cv_summary_csv": str(CV_SUMMARY_CSV),
            "within_group_variability_csv": str(WITHIN_GROUP_VARIABILITY_CSV),
        },
    }
    provenance_nb02_path = RESULTS_DIR / "provenance_nb02.json"
    with provenance_nb02_path.open("w", encoding="utf-8") as _f:
        json.dump(provenance_nb02, _f, indent=2, ensure_ascii=False)
    print(f"✓ Provenance saved: {provenance_nb02_path}")

    if CONFIG.save_provenance_history:
        _timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _history_path = RESULTS_DIR / f"provenance_nb02_{_timestamp}.json"
        if _history_path.exists():
            raise FileExistsError(f"Historical provenance file already exists: {_history_path}")
        with _history_path.open("w", encoding="utf-8") as _f:
            json.dump(provenance_nb02, _f, indent=2, ensure_ascii=False)
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
