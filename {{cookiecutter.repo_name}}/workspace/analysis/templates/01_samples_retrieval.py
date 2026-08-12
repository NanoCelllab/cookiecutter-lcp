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
    # 01 — Samples Retrieval

    **Pipeline step:** 1 of 6
    **Purpose:** retrieve, validate, consolidate, and document the raw
    single-cell profiles used by the downstream analysis.

    > This notebook performs **structural validation and quality-control
    > reporting only**. It does not normalize, aggregate, impute, or
    > select features. Those transformations belong to NB02.

    ## Input and output

    **Accepted raw inputs:** CellProfiler SQLite database(s), or one legacy
    merged single-cell CSV
    **Primary output:** `single_cell_profiles.parquet`
    **Interoperability output:** `single_cell_profiles.csv`
    **QC outputs:** cell-count table, plate maps, feature-integrity report
    **Provenance output:** `provenance_nb01_latest.json`

    ## Execution logic

    1. Configure the experiment (widgets below) and validate the choice.
    2. Discover the available raw input and report what will be used.
    3. Reconstruct the plate-level datasets.
    4. Run blocking structural validations (SC-01 → SC-03).
    5. Report non-blocking cell-count QC findings (SC-04).
    6. Generate plate maps using a shared color scale (SC-05).
    7. Summarize cell counts and protect the export.
    8. Diagnose missing and infinite feature values.
    9. Safely export outputs without silent overwriting.
    10. Record provenance and run final integrity checks.

    ## Validation policy

    | Type | Examples | Behaviour |
    |---|---|---|
    | **Blocking validation** | missing plate/well metadata, incompatible plate schemas, invalid well identifiers | stop execution with an error |
    | **QC warning** | wells below the minimum cell count, partially missing features, infinite values | report and preserve data for NB02 |
    | **Output protection** | existing output differs from the current in-memory result | do not overwrite unless explicitly enabled |
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
    import re
    import sqlite3
    import subprocess
    from dataclasses import replace
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Sequence

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import Normalize

    pd.set_option("display.max_columns", 200)
    return (
        Normalize,
        Path,
        Sequence,
        datetime,
        json,
        np,
        patches,
        pd,
        platform,
        plt,
        re,
        replace,
        sqlite3,
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

    from hca_pipeline.config import (
        SUPPORTED_PLATE_FORMATS,
        COMPARTMENT_PREFIXES,
        ExperimentConfig,
        validate_configuration,
    )
    from hca_pipeline.io import (
        write_parquet_protected,
        write_csv_protected,
        write_summary_table_protected,
    )
    from hca_pipeline.metadata import (
        add_cell_count_metadata,
        norm_well,
        ensure_core_metadata,
        dedupe_meta,
        read_barcode_platemap,
        read_platemap_layout,
    )
    from hca_pipeline.feature_select import infer_feature_cols

    print(f"  ✓  Shared utilities loaded from hca_pipeline ({_pipelines_dir})")
    return (
        COMPARTMENT_PREFIXES,
        ExperimentConfig,
        REPO_ROOT,
        SUPPORTED_PLATE_FORMATS,
        add_cell_count_metadata,
        dedupe_meta,
        ensure_core_metadata,
        infer_feature_cols,
        norm_well,
        read_barcode_platemap,
        read_platemap_layout,
        validate_configuration,
        write_csv_protected,
        write_parquet_protected,
        write_summary_table_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Pick the experiment folder and confirm the plate geometry and QC
    threshold. Defaults are pre-filled from a previously saved
    `experiment_config.json` when one exists, so re-running this
    notebook (or moving on to later pipeline steps) doesn't require
    re-entering the same choices.
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
def _(SUPPORTED_PLATE_FORMATS, loaded_config, mo):
    plate_format_input = mo.ui.dropdown(
        options=[str(k) for k in sorted(SUPPORTED_PLATE_FORMATS)],
        value=str(loaded_config.plate_format),
        label="Plate format (wells)",
    )
    min_cells_input = mo.ui.number(
        value=loaded_config.min_cells_per_well,
        start=1,
        stop=100_000,
        label="Minimum cells per well",
    )
    channels_input = mo.ui.multiselect(
        options=["GFP", "PI", "Hoechst", "DAPI", "Brightfield"],
        value=loaded_config.channels or ["GFP", "PI"],
        label="Imaging channels used (saved for later pipeline steps)",
    )
    image_root_input = mo.ui.text(
        value=loaded_config.image_root or "",
        label="Raw image directory, relative to repo root (leave blank if none — "
        "00_image_quality.py self-skips without this)",
    )
    overwrite_input = mo.ui.checkbox(
        value=loaded_config.overwrite_existing_outputs,
        label="Overwrite existing outputs",
    )
    save_history_input = mo.ui.checkbox(
        value=loaded_config.save_provenance_history,
        label="Save timestamped provenance history",
    )
    mo.vstack(
        [
            plate_format_input,
            min_cells_input,
            channels_input,
            image_root_input,
            overwrite_input,
            save_history_input,
        ]
    )
    return (
        channels_input,
        image_root_input,
        min_cells_input,
        overwrite_input,
        plate_format_input,
        save_history_input,
    )


@app.cell
def _(
    REPO_ROOT,
    channels_input,
    experiment_id_input,
    image_root_input,
    loaded_config,
    min_cells_input,
    overwrite_input,
    plate_format_input,
    replace,
    save_history_input,
    validate_configuration,
):
    EXPERIMENT_ID = experiment_id_input.value
    PLATE_FORMAT = int(plate_format_input.value)
    MIN_CELLS_PER_WELL = int(min_cells_input.value)

    BACKEND_DIR = REPO_ROOT / "workspace" / "backend"
    validate_configuration(
        experiment_id=EXPERIMENT_ID,
        plate_format=PLATE_FORMAT,
        min_cells_per_well=MIN_CELLS_PER_WELL,
        experiments_dir=BACKEND_DIR,
    )

    CONFIG = replace(
        loaded_config,
        experiment_id=EXPERIMENT_ID,
        plate_format=PLATE_FORMAT,
        min_cells_per_well=MIN_CELLS_PER_WELL,
        channels=list(channels_input.value),
        image_root=image_root_input.value.strip() or None,
        overwrite_existing_outputs=bool(overwrite_input.value),
        save_provenance_history=bool(save_history_input.value),
    )
    _config_path = CONFIG.save(REPO_ROOT)

    print("═" * 72)
    print("CONFIGURATION VALIDATED")
    print("═" * 72)
    print(f"  Experiment ID:       {EXPERIMENT_ID}")
    print(f"  Plate format:        {PLATE_FORMAT}-well")
    print(f"  Minimum cells/well:  {MIN_CELLS_PER_WELL}")
    print(f"  Channels:            {', '.join(CONFIG.channels) or '(none selected)'}")
    print(f"  Image root:          {CONFIG.image_root or '(none — image QC disabled)'}")
    print(f"  Repository root:     {REPO_ROOT}")
    print(f"  Saved config:        {_config_path}")
    return CONFIG, EXPERIMENT_ID, MIN_CELLS_PER_WELL, PLATE_FORMAT


@app.cell
def _(EXPERIMENT_ID, REPO_ROOT):
    WORKSPACE_DIR = REPO_ROOT / "workspace"
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    PROFILES_DIR = WORKSPACE_DIR / "profiles" / EXPERIMENT_ID
    DB_ROOT = WORKSPACE_DIR / "backend" / EXPERIMENT_ID

    OUTPUT_DIR = ANALYSIS_DIR / "outputs"
    RESULTS_DIR = ANALYSIS_DIR / "results"
    FIGS_DIR = ANALYSIS_DIR / "figures" / "sample_retrieval"

    OUTPUT_CSV = OUTPUT_DIR / "single_cell_profiles.csv"
    OUTPUT_PARQUET = OUTPUT_DIR / "single_cell_profiles.parquet"
    COUNTS_CSV = OUTPUT_DIR / "object_count_per_well.csv"

    for _directory in (OUTPUT_DIR, RESULTS_DIR, FIGS_DIR):
        _directory.mkdir(parents=True, exist_ok=True)

    print(f"  Database directory:  {DB_ROOT}")
    print(f"  Analysis directory:  {ANALYSIS_DIR}")
    return (
        COUNTS_CSV,
        DB_ROOT,
        FIGS_DIR,
        OUTPUT_CSV,
        OUTPUT_PARQUET,
        RESULTS_DIR,
        WORKSPACE_DIR,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Locate and validate the experiment input

    The notebook first looks for CellProfiler SQLite databases (`.db`,
    `.sqlite`, or `.sqlite3`) and falls back to one legacy merged CSV.

    `single_cell_profiles.parquet` is the **first output of this notebook**.
    It is therefore completely normal for it not to exist on the first run;
    its absence is reported as information, never as an input error.
    """)
    return


@app.cell
def _(DB_ROOT, OUTPUT_PARQUET, mo):
    _candidate_databases = sorted(
        p
        for pattern in ("*.db", "*.sqlite", "*.sqlite3")
        for p in DB_ROOT.glob(pattern)
    ) if DB_ROOT.is_dir() else []
    _candidate_csvs = sorted(DB_ROOT.glob("*.csv")) if DB_ROOT.is_dir() else []

    if _candidate_databases:
        INPUT_KIND = "sqlite"
        INPUT_FILES = _candidate_databases
        _source_message = (
            f"Found **{len(INPUT_FILES)} CellProfiler database(s)**. "
            "They will be merged into one single-cell table."
        )
    elif len(_candidate_csvs) == 1:
        INPUT_KIND = "csv"
        INPUT_FILES = _candidate_csvs
        _source_message = "Found **one legacy merged single-cell CSV**."
    elif len(_candidate_csvs) > 1:
        INPUT_KIND = "ambiguous"
        INPUT_FILES = []
        _source_message = (
            "Found more than one CSV and cannot choose safely: "
            + ", ".join(p.name for p in _candidate_csvs)
        )
    else:
        INPUT_KIND = "missing"
        INPUT_FILES = []
        _source_message = (
            f"No CellProfiler database or legacy CSV was found in `{DB_ROOT}`."
        )

    _parquet_message = (
        f"An earlier output exists at `{OUTPUT_PARQUET}`; it will be protected "
        "from silent overwrite."
        if OUTPUT_PARQUET.exists()
        else "No previous parquet exists yet — this is the expected state on the first run."
    )
    _kind = "success" if INPUT_KIND in {"sqlite", "csv"} else "danger"
    _discovery_callout = mo.callout(
        mo.md(
            f"**Input discovery**\n\n{_source_message}\n\n"
            f"**Output checkpoint**\n\n{_parquet_message}"
        ),
        kind=_kind,
    )
    mo.stop(
        INPUT_KIND in {"missing", "ambiguous"},
        mo.vstack(
            [
                _discovery_callout,
                mo.callout(
                    mo.md(
                        "Add one or more CellProfiler `.db` files, or exactly one legacy "
                        f"CSV, to `{DB_ROOT}`, then rerun this cell."
                    ),
                    kind="danger",
                ),
            ]
        ),
    )
    _discovery_callout
    return INPUT_FILES, INPUT_KIND


@app.cell
def _(Path, dedupe_meta, ensure_core_metadata, norm_well, pd, sqlite3):
    def read_legacy_single_cell_csv(csv_path: Path) -> pd.DataFrame:
        """Read and validate a previously generated single-cell profile CSV.

        Intended for legacy experiments in which CellProfiler compartment
        tables were merged before the current SQLite-based pipeline was
        standardized.
        """
        csv_path = Path(csv_path)

        if not csv_path.is_file():
            raise FileNotFoundError(
                "\nLegacy single-cell CSV not found\n"
                "================================\n\n"
                f"Expected file:\n  {csv_path}\n\n"
                "Check that the experiment's database folder contains it."
            )

        file_size_mb = csv_path.stat().st_size / (1024**2)
        print("Reading legacy single-cell CSV")
        print("--------------------------------")
        print(f"  File:      {csv_path}")
        print(f"  File size: {file_size_mb:,.2f} MB")

        df = pd.read_csv(csv_path, low_memory=False)

        if df.empty:
            raise ValueError(f"{csv_path.name}: the CSV contains no rows.")

        original_rows, original_columns = df.shape
        print(f"  Raw data:  {original_rows:,} rows × {original_columns:,} columns")

        # Remove accidental DataFrame index columns created during CSV export.
        unnamed_columns = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed_columns:
            df = df.drop(columns=unnamed_columns, errors="ignore")
            print(f"  Cleanup:   removed {len(unnamed_columns)} exported index column(s)")
        else:
            print("  Cleanup:   no exported index columns detected")

        # Standardize core metadata column names when possible.
        original_column_names = set(df.columns)
        df = ensure_core_metadata(df)
        renamed_metadata = sorted(set(df.columns) - original_column_names)
        if renamed_metadata:
            print("  Metadata:  standardized core metadata column names")
        else:
            print("  Metadata:  core metadata columns already standardized")

        # Some CellProfiler exports merge per-compartment (Nuclei/Cytoplasm/
        # Cells) tables without dropping each one's own copy of
        # Plate/Well/Site/QCFlag -- pandas silently re-labels the repeats as
        # Metadata_Plate.1, Metadata_Plate.2, ... on read. Collapsing them
        # here, at the source, keeps every downstream parquet file free of
        # them rather than relying on each consumer to clean up after itself.
        _n_columns_before_dedupe = df.shape[1]
        df = dedupe_meta(df)
        _n_duplicates_removed = _n_columns_before_dedupe - df.shape[1]
        if _n_duplicates_removed:
            print(f"  Metadata:  collapsed {_n_duplicates_removed} duplicated metadata column(s)")
        else:
            print("  Metadata:  no duplicated metadata columns detected")

        required_metadata = ["Metadata_Plate", "Metadata_Well"]
        missing_metadata = [c for c in required_metadata if c not in df.columns]
        if missing_metadata:
            raise ValueError(
                "\nRequired metadata columns are missing\n"
                "=====================================\n\n"
                f"Missing columns:\n" + "\n".join(f"  - {c}" for c in missing_metadata)
                + "\n\nAvailable metadata-like columns:\n"
                + "\n".join(f"  - {c}" for c in df.columns if str(c).startswith("Metadata_"))
            )

        # Normalize plate and well identifiers.
        df["Metadata_Plate"] = df["Metadata_Plate"].astype("string").str.strip()
        df["Metadata_Well"] = df["Metadata_Well"].map(norm_well)

        # Remove rows without usable plate or well identifiers.
        invalid_metadata = (
            df["Metadata_Plate"].isna()
            | df["Metadata_Well"].isna()
            | df["Metadata_Plate"].eq("")
            | df["Metadata_Well"].eq("")
        )
        if invalid_metadata.any():
            n_invalid = int(invalid_metadata.sum())
            invalid_preview = (
                df.loc[invalid_metadata, ["Metadata_Plate", "Metadata_Well"]]
                .head(10)
                .to_string(index=False)
            )
            raise ValueError(
                "\nInvalid plate or well metadata detected\n"
                "=======================================\n\n"
                f"{n_invalid:,} row(s) have missing or invalid "
                "Metadata_Plate/Metadata_Well values.\n\n"
                f"First affected rows:\n{invalid_preview}"
            )

        return df

    def read_cellprofiler_sqlite(sqlite_path: Path) -> pd.DataFrame:
        """Merge CellProfiler image/cell/cytoplasm/nuclei tables."""
        sqlite_path = Path(sqlite_path)
        file_size_mb = sqlite_path.stat().st_size / (1024**2)
        print(f"  Opening {sqlite_path.name} ({file_size_mb:,.1f} MB)", flush=True)

        with sqlite3.connect(sqlite_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            required = {"Per_Image", "Per_Cells", "Per_Cytoplasm", "Per_Nuclei"}
            missing = sorted(required - tables)
            if missing:
                raise ValueError(
                    f"{sqlite_path.name} is missing required CellProfiler table(s): "
                    + ", ".join(missing)
                )

            print("    Reading image metadata", flush=True)
            image_columns = [
                row[1] for row in connection.execute("PRAGMA table_info(Per_Image)")
            ]
            selected_image_columns = [
                column
                for column in image_columns
                if column == "ImageNumber" or column.startswith("Image_Metadata_")
            ]
            image = pd.read_sql_query(
                "SELECT " + ", ".join(f'\"{c}\"' for c in selected_image_columns) + " FROM Per_Image",
                connection,
            ).rename(columns=lambda c: c.removeprefix("Image_") if c != "ImageNumber" else c)

            cytoplasm_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(Per_Cytoplasm)")
            }
            nucleus_parent_column = next(
                (
                    column
                    for column in (
                        "Cytoplasm_Parent_Nuclei",
                        "Cytoplasm_Parent_NucleiWithBorders",
                    )
                    if column in cytoplasm_columns
                ),
                None,
            )
            if nucleus_parent_column is None:
                raise ValueError(
                    f"{sqlite_path.name}: Per_Cytoplasm has no supported nucleus-parent column."
                )

            print("    Reading Cells measurements", flush=True)
            cells = pd.read_sql_query("SELECT * FROM Per_Cells", connection)
            print("    Reading Cytoplasm measurements", flush=True)
            cytoplasm = pd.read_sql_query("SELECT * FROM Per_Cytoplasm", connection)
            print("    Reading Nuclei measurements", flush=True)
            nuclei = pd.read_sql_query("SELECT * FROM Per_Nuclei", connection)

        merged = cells.merge(
            cytoplasm,
            left_on=["ImageNumber", "Cells_Number_Object_Number"],
            right_on=["ImageNumber", "Cytoplasm_Parent_Cells"],
            how="inner",
            validate="one_to_one",
        ).merge(
            nuclei,
            left_on=["ImageNumber", nucleus_parent_column],
            right_on=["ImageNumber", "Nuclei_Number_Object_Number"],
            how="inner",
            validate="one_to_one",
        ).merge(image, on="ImageNumber", how="left", validate="many_to_one")

        if merged.empty:
            raise ValueError(f"{sqlite_path.name}: compartment merge produced no cells.")
        merged = ensure_core_metadata(dedupe_meta(merged))
        merged["Metadata_Plate"] = merged["Metadata_Plate"].astype("string").str.strip()
        merged["Metadata_Well"] = merged["Metadata_Well"].map(norm_well)
        print(f"    ✓ {len(merged):,} cells × {merged.shape[1]:,} columns", flush=True)
        return merged

    return read_cellprofiler_sqlite, read_legacy_single_cell_csv


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Load the single-cell profiles and reconstruct plates

    The CSV is read exactly once. `df_all` preserves the complete
    single-cell table; `plate_dfs` provides one validated DataFrame per
    plate for plate-level QC.

    If `00_image_quality.py` has been run for this experiment, its
    `results/image_quality/excluded_sites.csv` (Plate/Well/Site combinations
    that failed a blur/saturation/SNR threshold) is applied here, before any
    structural validation below — so debris/out-of-focus/mis-exposed fields
    never reach NB02's aggregation. If that file doesn't exist (image QC
    hasn't been run, or `image_root` isn't configured for this experiment),
    this step is a no-op.
    """)
    return


@app.cell
def _(
    INPUT_FILES,
    INPUT_KIND,
    RESULTS_DIR,
    mo,
    pd,
    read_cellprofiler_sqlite,
    read_legacy_single_cell_csv,
):
    print("═" * 72, flush=True)
    print("LOADING SINGLE-CELL PROFILES", flush=True)
    print("═" * 72, flush=True)
    try:
        if INPUT_KIND == "sqlite":
            _loaded_parts = []
            for _index, _path in enumerate(INPUT_FILES, start=1):
                print(f"[{_index}/{len(INPUT_FILES)}] {_path.name}", flush=True)
                _loaded_parts.append(read_cellprofiler_sqlite(_path))
            df_all = pd.concat(_loaded_parts, ignore_index=True, sort=False)
        else:
            df_all = read_legacy_single_cell_csv(INPUT_FILES[0])
    except Exception as _error:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**Input loading failed**\n\n"
                    f"`{type(_error).__name__}: {_error}`\n\n"
                    "The pipeline stopped here, so downstream cells will not emit "
                    "secondary/cascading errors."
                ),
                kind="danger",
            ),
        )
    print(f"✓ Input loaded: {len(df_all):,} cells total", flush=True)

    _excluded_sites_csv = RESULTS_DIR / "image_quality" / "excluded_sites.csv"
    if not _excluded_sites_csv.exists():
        print("ℹ️  No image-quality exclusion list found — skipping (image QC not run for this experiment).")
    else:
        _excluded = pd.read_csv(_excluded_sites_csv)
        _excluded["Metadata_Plate"] = _excluded["Metadata_Plate"].astype("string").str.strip()
        _excluded["Metadata_Well"] = _excluded["Metadata_Well"].astype("string").str.strip()

        if "Metadata_Site" in df_all.columns and "Metadata_Site" in _excluded.columns:
            # Match on a trailing-underscore-insensitive plate key: raw image
            # folder names (what NB00's image_qc.infer_plate reads) and this
            # experiment's single-cell/platemap convention have been observed
            # to disagree on a trailing "_" (e.g. "..._Plate_1" vs.
            # "..._Plate_1_") even though they refer to the same plate. The
            # canonical Metadata_Plate values in df_all are left untouched --
            # this normalization is only used to build the join key.
            _join_cols = ["_join_plate", "Metadata_Well", "Metadata_Site"]
            _excluded["Metadata_Site"] = _excluded["Metadata_Site"].astype("string")
            _excluded["_join_plate"] = _excluded["Metadata_Plate"].str.rstrip("_")
            _excluded_keys = _excluded[_join_cols].drop_duplicates()

            _df_all_keys = df_all[["Metadata_Plate", "Metadata_Well", "Metadata_Site"]].copy()
            _df_all_keys["Metadata_Site"] = _df_all_keys["Metadata_Site"].astype("string")
            _df_all_keys["_join_plate"] = _df_all_keys["Metadata_Plate"].str.rstrip("_")

            _n_matched_keys = len(
                _excluded_keys.merge(_df_all_keys[_join_cols].drop_duplicates(), on=_join_cols, how="inner")
            )
            print(
                f"  Match check: {_n_matched_keys}/{len(_excluded_keys)} excluded plate/well/site "
                "combinations found a matching entry in df_all. A low match count can mean "
                "plate/well/site identifiers differ between the raw-image folder names and this "
                "experiment's single-cell metadata, OR that a flagged site already contributes zero "
                "cells to df_all (e.g. CellProfiler segmented nothing on a badly under/overexposed "
                "field) — inspect the unmatched rows in excluded_sites.csv if this matters for your "
                "dataset, don't assume it's a bug."
            )

            _n_before = len(df_all)
            _merged = _df_all_keys.merge(_excluded_keys, on=_join_cols, how="left", indicator=True)
            df_all = df_all.loc[_merged["_merge"].to_numpy() == "left_only"].reset_index(drop=True)
            _n_dropped = _n_before - len(df_all)
            print(
                f"✓ Applied image-QC exclusions (site-level): dropped {_n_dropped:,} of {_n_before:,} "
                f"cells ({_n_matched_keys:,} of {len(_excluded_keys):,} flagged plate/well/site "
                "combination(s) actually matched and were excluded)"
            )
        else:
            # No per-site granularity in this experiment's single-cell table.
            # Dropping a whole well because one of several imaged sites
            # failed QC could discard good cells too, so this case is
            # surfaced for manual review instead of auto-excluded.
            print(
                "⚠️  df_all has no Metadata_Site column, so the site-level image-QC exclusion "
                "list can't be safely applied automatically. Review the flagged plate/well "
                "combinations below and exclude them manually if warranted:"
            )
            _flagged_wells = _excluded[["Metadata_Plate", "Metadata_Well"]].drop_duplicates()
            for _row in _flagged_wells.itertuples(index=False):
                print(f"    - {_row.Metadata_Plate} / {_row.Metadata_Well}")
    return (df_all,)


@app.cell
def _(df_all):
    plate_dfs = {
        str(plate_id): df_plate.reset_index(drop=True)
        for plate_id, df_plate in df_all.groupby("Metadata_Plate", sort=True, observed=True)
    }
    if not plate_dfs:
        raise ValueError("No plates could be reconstructed from Metadata_Plate.")

    print("Loaded dataset")
    print("==============")
    print(f"  Cells:               {len(df_all):,}")
    print(f"  Columns:             {df_all.shape[1]:,}")
    print(f"  Plates:              {len(plate_dfs):,}")
    print(
        "  Unique wells:        "
        f"{df_all[['Metadata_Plate', 'Metadata_Well']].drop_duplicates().shape[0]:,}"
    )
    print("\n  Cells by plate:")
    for _plate_id, _df_plate in plate_dfs.items():
        print(
            f"    - {_plate_id}: {len(_df_plate):,} cells, "
            f"{_df_plate['Metadata_Well'].nunique():,} wells"
        )
    return (plate_dfs,)


@app.cell
def _(CONFIG, mo, plate_dfs):
    detected_plates = list(plate_dfs)
    _configured_plates = CONFIG.resolve_plate_scope(detected_plates)
    _reason_lines = "\n".join(
        f"{_plate}: {_reason}"
        for _plate, _reason in CONFIG.excluded_plate_reasons.items()
        if _plate in detected_plates
    )

    included_plates_input = mo.ui.multiselect(
        options=detected_plates,
        value=_configured_plates,
        label="Plates included in downstream analysis",
    )
    analysis_mode_input = mo.ui.dropdown(
        options=["preliminary", "final"],
        value=CONFIG.analysis_mode if CONFIG.analysis_mode in {"preliminary", "final"} else "final",
        label="Analysis mode",
    )
    required_references_input = mo.ui.text(
        value=", ".join(CONFIG.required_reference_treatments),
        label="Required reference treatments (comma-separated)",
        full_width=True,
    )
    exclusion_reasons_input = mo.ui.text_area(
        value=_reason_lines,
        label='Excluded plate reasons, one "Plate: reason" per line',
        full_width=True,
    )

    mo.vstack(
        [
            mo.md("## 3b — Analysis plate scope"),
            mo.md(
                "NB01 always exports every detected plate. This selection is persisted and applied "
                "only when NB02 builds downstream analysis checkpoints."
            ),
            included_plates_input,
            mo.hstack([analysis_mode_input, required_references_input], widths=[1, 2]),
            exclusion_reasons_input,
        ]
    )
    return (
        analysis_mode_input,
        detected_plates,
        exclusion_reasons_input,
        included_plates_input,
        required_references_input,
    )


@app.cell
def _(
    CONFIG,
    REPO_ROOT,
    analysis_mode_input,
    detected_plates,
    exclusion_reasons_input,
    included_plates_input,
    mo,
    pd,
    plate_dfs,
    replace,
    required_references_input,
):
    included_plates = list(included_plates_input.value)
    if not included_plates:
        raise ValueError("Select at least one plate for downstream analysis.")

    _excluded_plates = [plate for plate in detected_plates if plate not in included_plates]
    _parsed_reasons = {}
    for _line in exclusion_reasons_input.value.splitlines():
        _line = _line.strip()
        if not _line or ":" not in _line:
            continue
        _plate, _reason = _line.split(":", 1)
        _parsed_reasons[_plate.strip()] = _reason.strip()
    excluded_plate_reasons = {
        plate: _parsed_reasons.get(plate, "Excluded from downstream analysis")
        for plate in _excluded_plates
    }
    required_reference_treatments = [
        value.strip() for value in required_references_input.value.split(",") if value.strip()
    ]

    ANALYSIS_CONFIG = replace(
        CONFIG,
        included_plates=included_plates,
        excluded_plate_reasons=excluded_plate_reasons,
        required_reference_treatments=required_reference_treatments,
        analysis_mode=analysis_mode_input.value,
    )
    _scope_config_path = ANALYSIS_CONFIG.save(REPO_ROOT)

    plate_scope_df = pd.DataFrame(
        [
            {
                "Plate": plate,
                "Included": plate in included_plates,
                "Cells": len(plate_dfs[plate]),
                "Wells": plate_dfs[plate]["Metadata_Well"].nunique(),
                "Reason if excluded": excluded_plate_reasons.get(plate, ""),
            }
            for plate in detected_plates
        ]
    )
    mo.vstack(
        [
            mo.md(
                f"**Persisted analysis scope:** `{len(included_plates)}/{len(detected_plates)}` plates · "
                f"mode `{ANALYSIS_CONFIG.analysis_mode}` · `{_scope_config_path}`"
            ),
            plate_scope_df,
        ]
    )
    return ANALYSIS_CONFIG, included_plates, plate_scope_df


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Structural validation and cell-count QC

    ### Blocking checks

    - **SC-01:** each plate contains rows and columns.
    - **SC-02:** all plates contain the same feature columns.
    - **SC-03:** required plate and well metadata are present and complete.

    ### Non-blocking check

    - **SC-04:** wells below the configured minimum cell count are
      reported but not removed.

    A failure in SC-01–SC-03 stops execution. SC-04 produces a warning
    for later review.
    """)
    return


@app.cell
def _(COMPARTMENT_PREFIXES, MIN_CELLS_PER_WELL, plate_dfs):
    print("═" * 65)
    print("SANITY CHECKS — NB01")
    print("═" * 65)

    # ── SC-01: Per-plate shape report ──────────────────────────────────────
    print("\n── SC-01: Per-plate shape ──────────────────────────────────────")
    sc01_pass = True
    for _pid, _df_p in plate_dfs.items():
        _n_cells, _n_cols = _df_p.shape
        _status = "✓" if _n_cells > 0 and _n_cols > 0 else "✗"
        if _n_cells == 0 or _n_cols == 0:
            sc01_pass = False
        print(f"  {_status}  {_pid}: {_n_cells:,} cells × {_n_cols} cols")
    print(f"  SC-01: {'PASS' if sc01_pass else 'FAIL — check plates above'}")

    # ── SC-02: Column consistency ───────────────────────────────────────────
    # Uses the same COMPARTMENT_PREFIXES as infer_feature_cols, so this check
    # and the feature-integrity diagnostic in Section 8 can never disagree
    # about what counts as a feature column.
    print("\n── SC-02: Column consistency ───────────────────────────────────")
    feat_sets = {
        _pid: {
            c
            for c in _df_p.columns
            if c.startswith(COMPARTMENT_PREFIXES) and not c.startswith("Metadata_")
        }
        for _pid, _df_p in plate_dfs.items()
    }
    sc02_pass = len({frozenset(v) for v in feat_sets.values()}) == 1
    if sc02_pass:
        print(f"  ✓  All plates have identical feature columns ({len(next(iter(feat_sets.values())))} features)")
    else:
        print("  ❌  Column mismatch across plates:")
        _all_cols = set().union(*feat_sets.values())
        _common_cols = set.intersection(*feat_sets.values())
        for _pid, _cols in feat_sets.items():
            _missing = _all_cols - _cols
            _extra = _cols - _common_cols
            if _missing or _extra:
                print(f"    {_pid}: missing {len(_missing)}, extra {len(_extra)}")
        raise ValueError(
            "Feature columns differ across plates (SC-02 FAIL). "
            "Do not concatenate until the CellProfiler outputs are harmonized."
        )
    print("  SC-02: PASS")

    # ── SC-03: Missing metadata ─────────────────────────────────────────────
    print("\n── SC-03: Missing metadata ─────────────────────────────────────")
    _blocking_cols = ["Metadata_Plate", "Metadata_Well"]
    _nonblocking_cols = ["Metadata_Site"]
    sc03_blocking_pass = True
    sc03_nonblocking_pass = True
    for _pid, _df_p in plate_dfs.items():
        for _col in _blocking_cols:
            if _col not in _df_p.columns:
                print(f"  ✗  {_pid}: blocking column '{_col}' missing entirely")
                sc03_blocking_pass = False
            elif _df_p[_col].isna().any():
                print(f"  ✗  {_pid}: blocking column '{_col}' has {_df_p[_col].isna().sum()} NaN values")
                sc03_blocking_pass = False
        for _col in _nonblocking_cols:
            if _col not in _df_p.columns:
                print(f"  ⚠️  {_pid}: non-blocking column '{_col}' missing (OK if single-site)")
            elif _df_p[_col].isna().any():
                print(f"  ⚠️  {_pid}: '{_col}' has {_df_p[_col].isna().sum()} NaN values (non-blocking)")
                sc03_nonblocking_pass = False
    if sc03_blocking_pass:
        print("  ✓  No missing blocking metadata (Plate, Well) in any plate")
    if not sc03_nonblocking_pass:
        print("  ⚠️  Some non-blocking metadata (Site) has issues — OK for single-site pipelines")
    if not sc03_blocking_pass:
        raise ValueError(
            "SC-03 FAIL: Metadata_Plate or Metadata_Well is missing or has NaN values. "
            "These columns are required for downstream annotation and aggregation."
        )
    print(f"  SC-03: {'PASS' if sc03_blocking_pass else 'FAIL — blocking metadata missing'}")

    # ── SC-04: Cell count per well ──────────────────────────────────────────
    print("\n── SC-04: Cell count per well ───────────────────────────────────")
    sc04_pass = True
    low_wells = []
    for _pid, _df_p in plate_dfs.items():
        if "Metadata_Well" not in _df_p.columns:
            print(f"  ⚠️  {_pid}: Metadata_Well missing — skipping SC-04")
            continue
        _counts = _df_p.groupby("Metadata_Well").size()
        _low = _counts[_counts < MIN_CELLS_PER_WELL]
        if not _low.empty:
            sc04_pass = False
            for _well, _n in _low.items():
                low_wells.append((_pid, _well, _n))
                print(f"  ⚠️  {_pid}  {_well}: {_n} cells (< {MIN_CELLS_PER_WELL})")
    if sc04_pass:
        print(f"  ✓  All wells have ≥ {MIN_CELLS_PER_WELL} cells")
    print(f"  SC-04: {'PASS' if sc04_pass else f'WARN — {len(low_wells)} low-count well(s)'}")

    print("\n" + "═" * 65)
    print(
        f"SC-01: {'PASS' if sc01_pass else 'FAIL'}  |  "
        f"SC-02: {'PASS' if sc02_pass else 'FAIL'}  |  "
        f"SC-03: {'PASS' if sc03_blocking_pass else 'FAIL'}  |  "
        f"SC-04: {'PASS' if sc04_pass else 'WARN'}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — Plate-level visual QC

    SC-05 generates one cell-count plate map per plate. All maps use
    the same upper color limit so that color intensity remains
    directly comparable across plates.

    Treatment labels (if shown) come from a best-effort, read-only lookup
    of the platemap — NB01 itself never annotates `df_all`/`plate_dfs`
    with treatment metadata (that's NB02's job); this only enriches the
    QC image. If no platemap is found, plate maps fall back to cell
    counts only.
    """)
    return


@app.cell
def _(mo):
    blur_plate_maps_input = mo.ui.checkbox(
        value=False, label="Blur plate map image (smoothed heatmap instead of sharp per-well grid)"
    )
    show_treatment_labels_input = mo.ui.checkbox(
        value=True, label="Show treatment labels on plate maps (if a platemap is available)"
    )
    mo.vstack([blur_plate_maps_input, show_treatment_labels_input])
    return blur_plate_maps_input, show_treatment_labels_input


@app.cell
def _(
    ANALYSIS_CONFIG,
    EXPERIMENT_ID,
    WORKSPACE_DIR,
    mo,
    pd,
    plate_dfs,
    read_barcode_platemap,
    read_platemap_layout,
    show_treatment_labels_input,
):
    well_treatments_by_plate: dict[str, dict[str, str]] = {}
    _coverage_rows = []
    _metadata_dir = WORKSPACE_DIR / "metadata" / EXPERIMENT_ID
    _barcode_csv = _metadata_dir / "barcode_platemap.csv"
    _platemap_dir = _metadata_dir / "platemap"
    if _barcode_csv.exists() and _platemap_dir.is_dir():
        try:
            _barcode_index = read_barcode_platemap(_barcode_csv)
            for _plate_id in plate_dfs:
                _row = _barcode_index.loc[_barcode_index["Metadata_Plate"] == _plate_id]
                if _row.empty:
                    continue
                _pm_path = _platemap_dir / _row["filename"].iloc[0]
                if not _pm_path.exists():
                    continue
                _pm = read_platemap_layout(_pm_path)
                if show_treatment_labels_input.value:
                    well_treatments_by_plate[_plate_id] = dict(
                        zip(_pm["Metadata_Well"], _pm["Metadata_Treatment"].astype(str))
                    )
                _observed_wells = set(plate_dfs[_plate_id]["Metadata_Well"].astype(str))
                _pm_observed = _pm.loc[_pm["Metadata_Well"].astype(str).isin(_observed_wells)]
                for _treatment, _count in _pm_observed["Metadata_Treatment"].astype(str).value_counts().items():
                    _coverage_rows.append(
                        {"Plate": _plate_id, "Treatment": _treatment, "Observed wells": int(_count)}
                    )
            print(f"  ✓  Loaded platemap coverage for {len(set(r['Plate'] for r in _coverage_rows))}/{len(plate_dfs)} plate(s)")
        except Exception as error:
            print(f"  ⚠️  Could not load platemap coverage: {error}")
    else:
        print("  ℹ️  Platemap not found — plate coverage cannot be assessed")

    plate_treatment_coverage = pd.DataFrame(
        _coverage_rows, columns=["Plate", "Treatment", "Observed wells"]
    )
    _coverage_table = (
        plate_treatment_coverage.pivot_table(
            index="Plate", columns="Treatment", values="Observed wells", fill_value=0, aggfunc="sum"
        ).astype(int)
        if not plate_treatment_coverage.empty
        else pd.DataFrame()
    )
    _missing_rows = []
    for _plate in ANALYSIS_CONFIG.included_plates:
        _present = set(
            plate_treatment_coverage.loc[
                plate_treatment_coverage["Plate"].eq(_plate), "Treatment"
            ].astype(str)
        )
        _missing = [ref for ref in ANALYSIS_CONFIG.required_reference_treatments if ref not in _present]
        _missing_rows.append(
            {"Plate": _plate, "Missing required references": ", ".join(_missing) or "None"}
        )
    _reference_status = pd.DataFrame(_missing_rows)
    mo.vstack(
        [
            mo.md("### Plate × treatment coverage"),
            _coverage_table if not _coverage_table.empty else mo.md("No platemap coverage available."),
            mo.md("### Required-reference check for included plates"),
            _reference_status,
        ]
    )
    return plate_treatment_coverage, well_treatments_by_plate


@app.cell
def _(Normalize, Path, np, patches, pd, plt, re):
    def plot_cell_count_heatmap(
        df_plate: pd.DataFrame,
        plate_id: str,
        out_dir: Path,
        min_cells: int,
        plate_format: int,
        vmax: float | None = None,
        smooth_image: bool = False,
        well_treatments: dict[str, str] | None = None,
    ) -> plt.Figure | None:
        """Plot and return a publication-friendly plate map of cell counts per well.

        Wells below `min_cells` are outlined and marked with a warning symbol.
        `smooth_image=True` renders the heatmap with smoothed (blurred)
        interpolation between wells instead of the default sharp grid.
        `well_treatments` (well ID -> label) optionally adds a treatment
        label below the cell count for each well, when available.
        """
        if "Metadata_Well" not in df_plate.columns:
            print(f"  ⚠️  {plate_id}: Metadata_Well is missing — skipping cell-count heatmap.")
            return

        plate_dimensions = {
            6: ("AB", 3),
            12: ("ABC", 4),
            24: ("ABCD", 6),
            48: ("ABCDEF", 8),
            96: ("ABCDEFGH", 12),
            384: ("ABCDEFGHIJKLMNOP", 24),
        }
        if plate_format not in plate_dimensions:
            raise ValueError(
                f"Unsupported plate format: {plate_format}. "
                f"Supported formats: {sorted(plate_dimensions)}"
            )

        rows_string, n_columns = plate_dimensions[plate_format]
        rows = list(rows_string)
        columns = list(range(1, n_columns + 1))

        counts = (
            df_plate.groupby("Metadata_Well", observed=True)
            .size()
            .rename("Cell_Count")
            .reset_index()
        )
        counts["Row"] = counts["Metadata_Well"].str.extract(r"^([A-Z])", expand=False)
        counts["Column"] = pd.to_numeric(
            counts["Metadata_Well"].str.extract(r"(\d+)$", expand=False), errors="coerce"
        )

        invalid_wells = (
            counts["Row"].isna()
            | counts["Column"].isna()
            | ~counts["Row"].isin(rows)
            | ~counts["Column"].isin(columns)
        )
        if invalid_wells.any():
            raise ValueError(
                f"{plate_id}: invalid well identifiers detected: "
                f"{counts.loc[invalid_wells, 'Metadata_Well'].tolist()}"
            )

        grid = (
            counts.pivot(index="Row", columns="Column", values="Cell_Count")
            .reindex(index=rows, columns=columns)
        )

        values = grid.to_numpy(dtype=float)
        observed_values = values[np.isfinite(values)]
        if observed_values.size == 0:
            print(f"  ⚠️  {plate_id}: no valid cell counts found — skipping heatmap.")
            return

        observed_max = float(np.nanmax(observed_values))
        if vmax is None:
            robust_max = float(np.nanpercentile(observed_values, 98))
            plot_vmax = max(robust_max, float(min_cells), 1.0)
        else:
            plot_vmax = max(float(vmax), float(min_cells), 1.0)

        norm = Normalize(vmin=0, vmax=plot_vmax, clip=True)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#E8E8E8")

        figure_width = max(8.0, n_columns * 0.72)
        figure_height = max(4.8, len(rows) * 0.70)
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
        image = ax.imshow(
            values, cmap=cmap, norm=norm,
            interpolation="gaussian" if smooth_image else "none",
            aspect="equal",
        )

        ax.set_xticks(np.arange(n_columns))
        ax.set_xticklabels(columns)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Plate column", labelpad=10, fontsize=10)
        ax.set_ylabel("Plate row", labelpad=10, fontsize=10)
        ax.tick_params(axis="both", which="major", length=0, labelsize=9, pad=5)
        ax.set_xticks(np.arange(-0.5, n_columns, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", linewidth=1.2, color="white")
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Character budget for treatment labels, scaled to how wide each well
        # actually is on this plate format (a 384-well plate has much less
        # room per well than a 96-well plate). Deliberately conservative
        # (assumes ~1 fontsize-width per character, well above DejaVu Sans's
        # actual average) so labels fit without needing the clip-path safety
        # net below -- a first pass at 0.62x/char still let labels overflow
        # wide enough that clipping cut into both edges instead of one.
        treatment_fontsize = 5.5
        cell_width_points = (figure_width / n_columns) * 72
        max_treatment_chars = max(4, int(0.85 * cell_width_points / treatment_fontsize) - 1)

        low_count_wells = []
        for row_index, row_name in enumerate(rows):
            for column_index, column_number in enumerate(columns):
                value = grid.loc[row_name, column_number]
                if pd.isna(value):
                    ax.text(column_index, row_index, "—", ha="center", va="center", fontsize=9, color="#777777")
                    continue

                value = int(value)
                well_id = f"{row_name}{column_number:02d}"
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "black" if luminance > 0.58 else "white"
                is_low = value < min_cells
                count_text = f"{value:,}" + (" !" if is_low else "")

                treatment_label = None
                if well_treatments:
                    treatment_label = well_treatments.get(well_id)
                    if treatment_label and len(treatment_label) > max_treatment_chars:
                        treatment_label = treatment_label[: max_treatment_chars - 1] + "…"

                if is_low:
                    low_count_wells.append((well_id, value))
                    ax.add_patch(
                        patches.Rectangle(
                            (column_index - 0.47, row_index - 0.47),
                            0.94,
                            0.94,
                            fill=False,
                            edgecolor="#C62828",
                            linewidth=2.2,
                            joinstyle="round",
                        )
                    )

                # Count is always bold and a touch larger for readability;
                # the optional treatment label below it is smaller/lighter
                # so it doesn't compete visually with the primary number.
                ax.text(
                    column_index,
                    row_index - (0.16 if treatment_label else 0),
                    count_text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=text_color,
                )
                if treatment_label:
                    treatment_text = ax.text(
                        column_index,
                        row_index + 0.22,
                        treatment_label,
                        ha="center",
                        va="center",
                        fontsize=treatment_fontsize,
                        fontweight="normal",
                        color=text_color,
                        alpha=0.9,
                        clip_on=True,
                    )
                    # Belt-and-suspenders: even if the character budget above
                    # underestimates this font's actual width, hard-clip to
                    # this well's own cell so text can never bleed into a
                    # neighboring well.
                    treatment_text.set_clip_path(
                        patches.Rectangle(
                            (column_index - 0.49, row_index - 0.49),
                            0.98,
                            0.98,
                            transform=ax.transData,
                        )
                    )

        n_observed = int(np.isfinite(values).sum())
        n_low = len(low_count_wells)
        clean_plate_id = str(plate_id).strip()

        fig.suptitle(f"Cell count per well — {clean_plate_id}", fontsize=14, fontweight="semibold", y=1.06)
        ax.set_title(
            f"{n_observed} observed wells · {n_low} below QC threshold (< {min_cells:,} cells)",
            fontsize=9.5,
            pad=30,
        )

        colorbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.035, shrink=0.88)
        colorbar.set_label("Cells per well", fontsize=10, labelpad=10)
        colorbar.ax.tick_params(labelsize=8, length=3)
        colorbar.outline.set_visible(False)
        if observed_max > plot_vmax:
            colorbar.ax.set_title(f"≥ {plot_vmax:,.0f}", fontsize=8, pad=6)

        qc_patch = patches.Patch(
            facecolor="none", edgecolor="#C62828", linewidth=2, label=f"Below threshold (< {min_cells:,})"
        )
        missing_patch = patches.Patch(facecolor="#E8E8E8", edgecolor="none", label="No observations")
        ax.legend(
            handles=[qc_patch, missing_patch],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=2,
            frameon=False,
            fontsize=8.5,
            handlelength=1.5,
        )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_plate_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean_plate_id).strip("_")
        png_path = out_dir / f"sc05_cell_count_plate_map_{safe_plate_id}.png"
        pdf_path = out_dir / f"sc05_cell_count_plate_map_{safe_plate_id}.pdf"

        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"  ✓  {plate_id}: plate map saved ({n_observed} observed wells; {n_low} below threshold)")
        print(f"     PNG: {png_path.name}")
        print(f"     PDF: {pdf_path.name}")
        return fig

    return (plot_cell_count_heatmap,)


@app.cell
def _(
    FIGS_DIR,
    MIN_CELLS_PER_WELL,
    PLATE_FORMAT,
    blur_plate_maps_input,
    mo,
    plate_dfs,
    plot_cell_count_heatmap,
    well_treatments_by_plate: dict[str, dict[str, str]],
):
    global_max = max(
        df_plate.groupby("Metadata_Well", observed=True).size().max()
        for df_plate in plate_dfs.values()
    )

    print("═" * 72)
    print("SC-05 — CELL-COUNT PLATE MAPS")
    print("═" * 72)
    print(f"  Shared color-scale maximum: {global_max:,.0f} cells per well\n")

    plate_map_figures = []
    for _plate_id, _df_plate in plate_dfs.items():
        _figure = plot_cell_count_heatmap(
            df_plate=_df_plate,
            plate_id=_plate_id,
            out_dir=FIGS_DIR,
            min_cells=MIN_CELLS_PER_WELL,
            plate_format=PLATE_FORMAT,
            vmax=global_max,
            smooth_image=blur_plate_maps_input.value,
            well_treatments=well_treatments_by_plate.get(_plate_id),
        )
        if _figure is not None:
            plate_map_figures.append(_figure)

    mo.vstack(plate_map_figures)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Cell-count summary and protected export

    `counts_df` contains one row per plate/well combination.
    """)
    return


@app.cell
def _(MIN_CELLS_PER_WELL, add_cell_count_metadata, df_all, pd):
    def normalize_count_table(table: pd.DataFrame) -> pd.DataFrame:
        """Return a standardized cell-count table for safe comparison."""
        required_columns = ["Metadata_Plate", "Metadata_Well", "n_cells"]
        missing_columns = [c for c in required_columns if c not in table.columns]
        if missing_columns:
            raise ValueError(f"Cell-count table is missing required columns: {missing_columns}")

        normalized = table[required_columns].copy()
        normalized["Metadata_Plate"] = normalized["Metadata_Plate"].astype("string").str.strip()
        normalized["Metadata_Well"] = normalized["Metadata_Well"].astype("string").str.strip()
        normalized["n_cells"] = pd.to_numeric(normalized["n_cells"], errors="raise").astype("int64")
        return normalized.sort_values(["Metadata_Plate", "Metadata_Well"]).reset_index(drop=True)

    counts_df = normalize_count_table(
        df_all.groupby(["Metadata_Plate", "Metadata_Well"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    df_all_with_counts = add_cell_count_metadata(df_all)
    low_count_wells = counts_df.loc[counts_df["n_cells"] < MIN_CELLS_PER_WELL].copy()

    print("═" * 72)
    print("CELL COUNT PER WELL")
    print("═" * 72)
    print(f"  Plates:              {counts_df['Metadata_Plate'].nunique():,}")
    print(f"  Wells:               {len(counts_df):,}")
    print(f"  Total cells:         {counts_df['n_cells'].sum():,}")
    print("\n  Cells per well:")
    print(f"    Minimum:            {counts_df['n_cells'].min():,}")
    print(f"    Median:             {counts_df['n_cells'].median():,.0f}")
    print(f"    Mean:               {counts_df['n_cells'].mean():,.1f}")
    print(f"    Maximum:            {counts_df['n_cells'].max():,}")
    print(f"\n  QC threshold:        ≥ {MIN_CELLS_PER_WELL:,} cells")
    print(f"  Wells passing QC:    {len(counts_df) - len(low_count_wells):,}")
    print(f"  Wells below QC:      {len(low_count_wells):,}")
    if not low_count_wells.empty:
        print("\n  Wells below threshold:")
        for row in low_count_wells.itertuples(index=False):
            print(f"    - {row.Metadata_Plate} / {row.Metadata_Well}: {row.n_cells:,} cells")
    return counts_df, df_all_with_counts, low_count_wells


@app.cell
def _(CONFIG, COUNTS_CSV, counts_df, write_summary_table_protected):
    counts_export_status = write_summary_table_protected(
        counts_df, COUNTS_CSV, overwrite=CONFIG.overwrite_existing_outputs
    )
    print(f"✓ Count-summary file {counts_export_status}")
    print(f"  File: {COUNTS_CSV}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — Feature-integrity diagnostic

    This is a **report-only** step. It counts missing and infinite
    values but does not modify `df_all_with_counts`, and NB02 does not read this file
    back in — NB02 computes its own missingness/imputation decisions
    independently. The report saved here exists purely for human
    auditability: a per-feature breakdown of what's missing or infinite
    *before* any cleaning happens, in case that's relevant to interpreting
    NB02's later choices.
    """)
    return


@app.cell
def _(
    CONFIG,
    RESULTS_DIR,
    df_all_with_counts,
    infer_feature_cols,
    np,
    pd,
    write_summary_table_protected,
):
    feature_columns = infer_feature_cols(df_all_with_counts)
    if not feature_columns:
        raise ValueError("No feature columns were detected in df_all_with_counts.")

    raw_feature_values = df_all_with_counts[feature_columns]
    finite_view = raw_feature_values.replace([np.inf, -np.inf], np.nan)

    feature_integrity_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "n_missing_or_infinite": finite_view.isna().sum().to_numpy(dtype=int),
            "fraction_missing_or_infinite": finite_view.isna().mean().to_numpy(dtype=float),
            "n_positive_inf": np.isposinf(raw_feature_values.to_numpy()).sum(axis=0).astype(int),
            "n_negative_inf": np.isneginf(raw_feature_values.to_numpy()).sum(axis=0).astype(int),
        }
    )
    feature_integrity_df["status"] = np.select(
        [
            feature_integrity_df["fraction_missing_or_infinite"].eq(0),
            feature_integrity_df["fraction_missing_or_infinite"].eq(1),
        ],
        ["complete", "fully_missing"],
        default="partially_missing",
    )
    feature_integrity_df = feature_integrity_df.sort_values(
        ["fraction_missing_or_infinite", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    fully_missing_features = feature_integrity_df.loc[
        feature_integrity_df["status"].eq("fully_missing"), "feature"
    ].tolist()
    partially_missing_features = feature_integrity_df.loc[
        feature_integrity_df["status"].eq("partially_missing"), "feature"
    ].tolist()
    n_positive_inf = int(feature_integrity_df["n_positive_inf"].sum())
    n_negative_inf = int(feature_integrity_df["n_negative_inf"].sum())
    n_infinite = n_positive_inf + n_negative_inf
    n_complete_features = int(feature_integrity_df["status"].eq("complete").sum())

    print("═" * 72)
    print("FEATURE-INTEGRITY DIAGNOSTIC")
    print("═" * 72)
    print(f"  Cells evaluated:                  {len(df_all_with_counts):,}")
    print(f"  Feature columns:                  {len(feature_columns):,}")
    print("\n  Feature completeness:")
    print(f"    Complete features:              {n_complete_features:,}")
    print(f"    Partially missing features:     {len(partially_missing_features):,}")
    print(f"    Fully missing features:         {len(fully_missing_features):,}")
    print("\n  Infinite values:")
    print(f"    Positive infinity:              {n_positive_inf:,}")
    print(f"    Negative infinity:              {n_negative_inf:,}")

    if fully_missing_features or partially_missing_features or n_infinite:
        print("\n⚠ Missing or non-finite values were detected.")
        print("  No filtering or imputation was applied in NB01.")
    else:
        print("\n✓ No missing or infinite feature values were detected.")

    FEATURE_INTEGRITY_CSV = RESULTS_DIR / "feature_integrity_nb01.csv"
    feature_integrity_status = write_summary_table_protected(
        feature_integrity_df, FEATURE_INTEGRITY_CSV, overwrite=CONFIG.overwrite_existing_outputs
    )
    print(f"\n✓ Feature-integrity report {feature_integrity_status}")
    print(f"  File: {FEATURE_INTEGRITY_CSV}")
    return (
        FEATURE_INTEGRITY_CSV,
        feature_columns,
        fully_missing_features,
        n_infinite,
        partially_missing_features,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — Protected export of single-cell profiles

    Parquet is the primary pipeline format because it preserves data
    types and is efficient for wide tables. CSV is retained for
    interoperability.
    """)
    return


@app.cell
def _(
    CONFIG,
    OUTPUT_CSV,
    OUTPUT_PARQUET,
    df_all_with_counts,
    write_csv_protected,
    write_parquet_protected,
):
    print("═" * 72)
    print("SINGLE-CELL PROFILE EXPORT")
    print("═" * 72)
    print(
        f"  Current dataset:     {len(df_all_with_counts):,} cells × "
        f"{df_all_with_counts.shape[1]:,} columns\n"
    )

    parquet_export_status = write_parquet_protected(
        df_all_with_counts, OUTPUT_PARQUET, overwrite=CONFIG.overwrite_existing_outputs
    )
    print(f"✓ Parquet profile {parquet_export_status}")
    print(f"  File: {OUTPUT_PARQUET}")

    csv_export_status = write_csv_protected(
        df_all_with_counts, OUTPUT_CSV, overwrite=CONFIG.overwrite_existing_outputs
    )
    print(f"\n✓ CSV interoperability copy {csv_export_status}")
    print(f"  File: {OUTPUT_CSV}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 — Provenance

    The provenance record documents the local code state and execution
    context. It only runs read-only Git commands; it does **not**
    create commits or push anything to a remote repository.
    """)
    return


@app.cell
def _(
    ANALYSIS_CONFIG,
    CONFIG,
    COUNTS_CSV,
    EXPERIMENT_ID,
    FEATURE_INTEGRITY_CSV,
    FIGS_DIR,
    MIN_CELLS_PER_WELL,
    OUTPUT_CSV,
    OUTPUT_PARQUET,
    PLATE_FORMAT,
    REPO_ROOT,
    RESULTS_DIR,
    Sequence,
    datetime,
    df_all_with_counts,
    feature_columns,
    fully_missing_features,
    json,
    low_count_wells,
    n_infinite,
    np,
    partially_missing_features,
    pd,
    platform,
    subprocess,
    timezone,
):
    NOTEBOOK_NAME = "01_samples_retrieval.py"

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
            "notebook": NOTEBOOK_NAME,
            "experiment_id": EXPERIMENT_ID,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "configuration": {
            "plate_format": int(PLATE_FORMAT),
            "minimum_cells_per_well": int(MIN_CELLS_PER_WELL),
            "overwrite_existing_outputs": bool(CONFIG.overwrite_existing_outputs),
            "analysis_mode": ANALYSIS_CONFIG.analysis_mode,
            "included_plates": ANALYSIS_CONFIG.included_plates,
            "excluded_plate_reasons": ANALYSIS_CONFIG.excluded_plate_reasons,
            "required_reference_treatments": ANALYSIS_CONFIG.required_reference_treatments,
        },
        "dataset": {
            "n_cells": int(df_all_with_counts.shape[0]),
            "n_columns": int(df_all_with_counts.shape[1]),
            "n_features": int(len(feature_columns)),
            "n_plates": int(df_all_with_counts["Metadata_Plate"].nunique()),
            "n_wells": int(
                df_all_with_counts[["Metadata_Plate", "Metadata_Well"]].drop_duplicates().shape[0]
            ),
            "n_low_count_wells": int(len(low_count_wells)),
            "n_fully_missing_features": int(len(fully_missing_features)),
            "n_partially_missing_features": int(len(partially_missing_features)),
            "n_infinite_values": int(n_infinite),
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
            "single_cell_parquet": str(OUTPUT_PARQUET),
            "single_cell_csv": str(OUTPUT_CSV),
            "cell_count_csv": str(COUNTS_CSV),
            "feature_integrity_csv": str(FEATURE_INTEGRITY_CSV),
            "figure_directory": str(FIGS_DIR),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    provenance_latest_path = RESULTS_DIR / "provenance_nb01_latest.json"
    with provenance_latest_path.open("w", encoding="utf-8") as _f:
        json.dump(provenance, _f, indent=2, ensure_ascii=False)

    provenance_history_path = None
    if CONFIG.save_provenance_history:
        _timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        provenance_history_path = RESULTS_DIR / f"provenance_nb01_{_timestamp}.json"
        if provenance_history_path.exists():
            raise FileExistsError(f"Historical provenance file already exists: {provenance_history_path}")
        with provenance_history_path.open("w", encoding="utf-8") as _f:
            json.dump(provenance, _f, indent=2, ensure_ascii=False)

    print("═" * 72)
    print("NB01 PROVENANCE")
    print("═" * 72)
    print(f"  Notebook:            {provenance['pipeline']['notebook']}")
    print(f"  Experiment:          {provenance['pipeline']['experiment_id']}")
    print(f"  Executed at UTC:     {provenance['pipeline']['executed_at_utc']}")
    print(f"\n  Git commit:          {provenance['version_control']['git_commit_short']}")
    print(f"  Git branch:          {provenance['version_control']['git_branch']}")
    dirty_state = provenance["version_control"]["working_tree_dirty"]
    if dirty_state is True:
        print("  Working tree:        modified — uncommitted changes present")
    elif dirty_state is False:
        print("  Working tree:        clean")
    else:
        print("  Working tree:        unknown")
    print(f"\n✓ Latest provenance:   {provenance_latest_path}")
    if provenance_history_path is not None:
        print(f"✓ Historical record:  {provenance_history_path}")
    else:
        print("  Historical record:  disabled")
    return (provenance_latest_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 — Final integrity checks and execution summary

    These checks verify agreement among the principal in-memory
    objects and exported summaries. They do not transform the data.
    """)
    return


@app.cell
def _(
    COUNTS_CSV,
    EXPERIMENT_ID,
    FEATURE_INTEGRITY_CSV,
    FIGS_DIR,
    OUTPUT_CSV,
    OUTPUT_PARQUET,
    Path,
    counts_df,
    df_all_with_counts,
    feature_columns,
    fully_missing_features,
    n_infinite,
    partially_missing_features,
    plate_dfs,
    provenance_latest_path,
):
    integrity_errors = []

    if int(counts_df["n_cells"].sum()) != len(df_all_with_counts):
        integrity_errors.append("The sum of counts_df['n_cells'] does not equal len(df_all_with_counts).")

    df_plate_ids = set(df_all_with_counts["Metadata_Plate"].astype(str))
    count_plate_ids = set(counts_df["Metadata_Plate"].astype(str))
    if df_plate_ids != count_plate_ids:
        integrity_errors.append("Plate identifiers differ between df_all_with_counts and counts_df.")

    expected_wells = df_all_with_counts[["Metadata_Plate", "Metadata_Well"]].drop_duplicates().shape[0]
    if len(counts_df) != expected_wells:
        integrity_errors.append(
            "The number of rows in counts_df does not equal the number of "
            "unique plate/well combinations in df_all_with_counts."
        )

    required_outputs = [OUTPUT_PARQUET, OUTPUT_CSV, COUNTS_CSV, FEATURE_INTEGRITY_CSV, provenance_latest_path]
    missing_outputs = [p for p in required_outputs if not Path(p).exists()]
    if missing_outputs:
        integrity_errors.append(
            "Required output files are missing: " + ", ".join(str(p) for p in missing_outputs)
        )

    if integrity_errors:
        raise RuntimeError(
            "\nNB01 integrity checks failed\n"
            "============================\n\n"
            + "\n".join(f"  - {e}" for e in integrity_errors)
        )

    print("═" * 72)
    print("NB01 COMPLETED")
    print("═" * 72)
    print("✓ All final integrity checks passed\n")
    print(f"  Experiment:          {EXPERIMENT_ID}")
    print(f"  Plates loaded:       {len(plate_dfs):,}")
    print(f"  Observed wells:      {len(counts_df):,}")
    print(f"  Single cells:        {len(df_all_with_counts):,}")
    print(f"  Feature columns:     {len(feature_columns):,}")
    print(f"  Fully missing:       {len(fully_missing_features):,}")
    print(f"  Partially missing:   {len(partially_missing_features):,}")
    print(f"  Infinite values:     {n_infinite:,}")
    print("\n  Primary output:")
    print(f"    {OUTPUT_PARQUET}")
    print("\n  QC outputs:")
    print(f"    {COUNTS_CSV}")
    print(f"    {FEATURE_INTEGRITY_CSV}")
    print(f"    {FIGS_DIR}")
    print("\n  Provenance:")
    print(f"    {provenance_latest_path}")
    print("\nNext step: NB02 — Annotate · Clean · Aggregate · Normalize · Feature Select")
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
