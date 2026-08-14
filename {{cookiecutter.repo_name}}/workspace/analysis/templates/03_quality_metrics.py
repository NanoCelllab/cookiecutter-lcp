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
    # 03 — Quality Metrics & Go/No-Go Dashboard

    **Pipeline step:** 3 of 6
    **Position in pipeline:** NB02 (aggregate/normalize/feature-select) → **NB03 (quality metrics)** → NB04 (phenotypic profiling)

    This notebook is the pipeline's QC gate, deliberately placed *before*
    phenotypic profiling/fingerprints: it only needs NB02's output, and its
    pass/fail criteria (PR/mAP thresholds below) are protocol-defined and shown
    explicitly with the evidence supporting each decision.

    **Input:** `per_well_features_selected.parquet` (from NB02)
    **Outputs:** `results/quality_metrics/*.csv`, `figures/quality_metrics/*.png`,
    `quality_metrics_report.md`, `provenance.json`

    ---

    ## Overview

    This is the pipeline's **Go/No-Go quality gate**. It computes three
    quality metrics — **Percent Replicating (PR)**, **Percent Matching
    (PM)**, and **Mean Average Precision (mAP)** — to assess whether a
    Live Cell Painting experiment (Acridine Orange dye, GFP/PI channels)
    produces reproducible, distinguishable phenotypic profiles, then
    combines four applications and a final decision into checks **SC-19
    through SC-22**.

    | Metric | Question it answers | Range | Good signal |
    |--------|---------------------|-------|-------------|
    | **PR** (Percent Replicating) | What fraction of treatments show replicate consistency above null? | 0–1 | >0.25 (minimum), 0.84–0.94 (well-executed) |
    | **PM** (Percent Matching) | What fraction of profiles have their nearest neighbor be the same treatment? | 0–1 | >0.50 |
    | **mAP** (Mean Average Precision) | Retrieval score whose biological meaning depends on how positive and negative pairs are defined (see below). | 0–1 | >0.50, FDR-significant |

    ### Always qualify mAP by its profiling task

    **mAP is a retrieval statistic, not a biological endpoint by itself.** In
    every table and figure below it is therefore qualified as one of:

    - **Phenotypic consistency (PC mAP):** positives are replicate profiles of
      the same treatment (or the same treatment–dose); asks whether replicates
      retrieve one another. Applications A–C use this meaning. In Application C,
      PC mAP measures replicate consistency *within each dose*; dose ordering is
      assessed separately by SC-21.
    - **Phenotypic activity (activity mAP):** treatments are retrieved against
      negative-control references; asks whether a reproducible phenotype is
      detectable relative to control. Application D uses this meaning.
    - **Phenotypic distinctiveness (MoA mAP):** positives share a mechanism of
      action and negatives have a different MoA; asks whether MoA classes are
      separable. **NB03 does not compute MoA mAP**, because no MoA-based positive
      pair definition is used here.

    All of the metric logic (from-scratch Pearson implementations,
    `copairs`-based cosine implementations, and the four QC applications)
    already lives in `hca_pipeline.metrics_qc` — this notebook only wires
    the experiment configuration, the loaded data, and these functions
    together, then writes the reports.

    ### Four applications → four sanity checks

    | Application | Question | Check |
    |---|---|---|
    | A: Per-plate QC | Does each plate independently show replicate consistency? | SC-19 |
    | B: Cross-plate batch | Do replicates agree across plates? Is there a batch effect? | SC-20 |
    | C: Dose-response | Do increasing doses show consistent ordering? | SC-21 |
    | D: Treatment vs control | Can treatments be distinguished from negative controls? | SC-22 |

    Section 8 (time-course) and Application B (cross-plate) run
    conditionally — automatically skipped with an informative message when
    the experiment has only one time point or one plate.

    ### References

    - Caicedo et al. 2020 — PR/PM formalization [doi:10.1038/s41592-020-0851-3]
    - Kalinin et al. 2025 — mAP framework + `copairs` library [doi:10.1038/s41467-025-60306-2]
    - Arevalo et al. 2024 — Batch effects in profiling [doi:10.1038/s41467-024-50613-5]
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

    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 200)
    return Path, datetime, json, pd, platform, replace, subprocess, timezone


@app.cell
def _(Path, mo, print):
    # Locate the repo root before hca_pipeline can be imported (bootstrap:
    # can't import find_repo_root from the package until sys.path includes
    # workspace). __file__ is used instead of cwd so this notebook
    # behaves identically under `marimo edit`, a plain `python` run, or an
    # automated run from any working directory.
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
    from hca_pipeline.metrics_qc import (
        crossvalidate_scratch_vs_copairs_map,
        generate_go_nogo_dashboard,
        plot_cross_plate_batch,
        plot_dose_response,
        plot_go_nogo_dashboard,
        plot_per_plate_qc,
        plot_time_course,
        plot_treatment_vs_control,
        run_cross_plate_batch,
        run_dose_response,
        run_per_plate_qc,
        run_time_course,
        run_treatment_vs_control,
    )

    mo.callout(
        mo.md(f"### Shared utilities ready\n\n✅ Loaded `hca_pipeline` from `{_pipelines_dir}`."),
        kind="success",
    )
    return (
        ExperimentConfig,
        REPO_ROOT,
        crossvalidate_scratch_vs_copairs_map,
        generate_go_nogo_dashboard,
        infer_feature_cols,
        plot_cross_plate_batch,
        plot_dose_response,
        plot_go_nogo_dashboard,
        plot_per_plate_qc,
        plot_time_course,
        plot_treatment_vs_control,
        run_cross_plate_batch,
        run_dose_response,
        run_per_plate_qc,
        run_time_course,
        run_treatment_vs_control,
        write_summary_table_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Pick the experiment folder; plate geometry, column names, and control
    vocabulary are pre-filled from the `experiment_config.json` saved by
    earlier pipeline steps. Only the experiment selector is shown initially;
    output policy and QC-methodology thresholds are available under
    **Advanced settings**. Effective values are summarized before analysis.
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
    overwrite_input = mo.ui.checkbox(
        value=loaded_config.overwrite_existing_outputs,
        label="Overwrite existing outputs",
    )
    save_history_input = mo.ui.checkbox(
        value=loaded_config.save_provenance_history,
        label="Save timestamped provenance history",
    )
    output_settings = mo.accordion(
        {
            "Advanced settings · Output policy": mo.vstack(
                [
                    mo.md(
                        "Change these only when intentionally replacing prior QC results or "
                        "retaining timestamped provenance records."
                    ),
                    overwrite_input,
                    save_history_input,
                ]
            )
        }
    )
    output_settings
    return overwrite_input, save_history_input


@app.cell
def _(
    REPO_ROOT,
    experiment_id_input,
    loaded_config,
    overwrite_input,
    mo,
    print,
    replace,
    save_history_input,
):
    EXPERIMENT_ID = experiment_id_input.value
    if EXPERIMENT_ID == "SET_EXPERIMENT_ID_HERE":
        raise SystemExit("Please select a valid experiment ID from the dropdown above.")

    RAW_CONFIG = replace(
        loaded_config,
        experiment_id=EXPERIMENT_ID,
        overwrite_existing_outputs=bool(overwrite_input.value),
        save_provenance_history=bool(save_history_input.value),
    )
    _config_path = RAW_CONFIG.save(REPO_ROOT)

    OVERWRITE_EXISTING_OUTPUTS = RAW_CONFIG.overwrite_existing_outputs
    mo.callout(
        mo.md(
            f"""
            ### Experiment configuration loaded

            | Setting | Effective value |
            |---|---|
            | Experiment | `{EXPERIMENT_ID}` |
            | Analysis mode | **{RAW_CONFIG.analysis_mode}** |
            | Negative controls | {', '.join(f'`{v}`' for v in RAW_CONFIG.negcon_values) or 'not configured'} |
            | Positive controls | {', '.join(f'`{v}`' for v in RAW_CONFIG.poscon_values) or 'not configured'} |
            | Overwrite existing outputs | **{OVERWRITE_EXISTING_OUTPUTS}** |
            | Timestamped provenance | **{RAW_CONFIG.save_provenance_history}** |
            | Saved configuration | `{_config_path}` |
            """
        ),
        kind="success",
    )
    return EXPERIMENT_ID, OVERWRITE_EXISTING_OUTPUTS, RAW_CONFIG


@app.cell
def _(mo):
    mo.md(r"""
    ### QC-methodology parameters

    These govern *how* the metrics below are computed and thresholded —
    they are properties of this analysis, not of the experiment itself,
    so they live as notebook widgets rather than in `ExperimentConfig`.
    Defaults implement the current pipeline protocol. Change them only when a
    documented assay-specific rationale justifies a different decision rule.
    """)
    return


@app.cell
def _(mo):
    null_percentile_input = mo.ui.number(
        value=95, start=50, stop=99, step=1,
        label="Null-distribution percentile (PR threshold, Caicedo 2020 standard)",
    )
    n_null_pairs_input = mo.ui.number(
        value=10000, start=100, stop=100_000, step=100,
        label="From-scratch null pairs (random cross-treatment pairs)",
    )
    null_size_input = mo.ui.number(
        value=10000, start=100, stop=100_000, step=100,
        label="copairs null size — dose-response, treatment-vs-control, cross-validation",
    )
    null_size_copairs_local_input = mo.ui.number(
        value=5000, start=100, stop=100_000, step=100,
        label="copairs null size — per-plate QC, cross-plate batch, time-course",
    )
    random_state_input = mo.ui.number(
        value=42, start=0, stop=2_147_483_647, step=1, label="Random seed",
    )
    pr_fraction_threshold_input = mo.ui.number(
        value=0.25, start=0.0, stop=1.0, step=0.01,
        label="PR-fraction threshold — minimum fraction of treatments passing PR per plate",
    )
    negcon_pr_threshold_input = mo.ui.number(
        value=0.10, start=0.0, stop=1.0, step=0.01, label="Negcon PR should be ≤ this",
    )
    poscon_pr_threshold_input = mo.ui.number(
        value=0.50, start=0.0, stop=1.0, step=0.01, label="Poscon PR should be > this",
    )
    batch_drop_threshold_input = mo.ui.number(
        value=0.30, start=0.0, stop=1.0, step=0.01,
        label="Cross-plate PR drop > this ⇒ batch effect",
    )
    min_reps_for_pr_input = mo.ui.number(
        value=5, start=2, stop=1000, step=1,
        label="Minimum replicates per plate for a stable PR estimate",
    )
    min_within_pr_for_flag_input = mo.ui.number(
        value=0.30, start=0.0, stop=1.0, step=0.01,
        label="Only flag a batch effect if within-plate PR exceeds this",
    )
    methodology_settings = mo.accordion(
        {
            "Advanced settings · QC methodology": mo.vstack(
                [
                    mo.md(
                        "These thresholds change the QC decision itself. Record and justify "
                        "any deviation from the defaults in the analysis report."
                    ),
                    null_percentile_input,
                    n_null_pairs_input,
                    null_size_input,
                    null_size_copairs_local_input,
                    random_state_input,
                    pr_fraction_threshold_input,
                    negcon_pr_threshold_input,
                    poscon_pr_threshold_input,
                    batch_drop_threshold_input,
                    min_reps_for_pr_input,
                    min_within_pr_for_flag_input,
                ]
            )
        }
    )
    methodology_settings
    return (
        batch_drop_threshold_input,
        min_reps_for_pr_input,
        min_within_pr_for_flag_input,
        n_null_pairs_input,
        negcon_pr_threshold_input,
        null_percentile_input,
        null_size_copairs_local_input,
        null_size_input,
        poscon_pr_threshold_input,
        pr_fraction_threshold_input,
        random_state_input,
    )


@app.cell
def _(
    batch_drop_threshold_input,
    min_reps_for_pr_input,
    min_within_pr_for_flag_input,
    n_null_pairs_input,
    negcon_pr_threshold_input,
    null_percentile_input,
    null_size_copairs_local_input,
    null_size_input,
    poscon_pr_threshold_input,
    pr_fraction_threshold_input,
    random_state_input,
    mo,
    print,
):
    NULL_PERCENTILE = float(null_percentile_input.value)
    N_NULL_PAIRS = int(n_null_pairs_input.value)
    NULL_SIZE = int(null_size_input.value)
    NULL_SIZE_COPAIRS_LOCAL = int(null_size_copairs_local_input.value)
    RANDOM_STATE = int(random_state_input.value)
    PR_FRACTION_THRESHOLD = float(pr_fraction_threshold_input.value)
    NEGCON_PR_THRESHOLD = float(negcon_pr_threshold_input.value)
    POSCON_PR_THRESHOLD = float(poscon_pr_threshold_input.value)
    BATCH_DROP_THRESHOLD = float(batch_drop_threshold_input.value)
    MIN_REPS_FOR_PR = int(min_reps_for_pr_input.value)
    MIN_WITHIN_PR_FOR_FLAG = float(min_within_pr_for_flag_input.value)

    mo.callout(
        mo.md(
            f"""
            ### QC methodology finalized

            | Parameter | Value |
            |---|---:|
            | Null percentile | {NULL_PERCENTILE:g} |
            | From-scratch null pairs | {N_NULL_PAIRS:,} |
            | copairs null size (global / local) | {NULL_SIZE:,} / {NULL_SIZE_COPAIRS_LOCAL:,} |
            | Random seed | {RANDOM_STATE} |
            | Minimum plate PR fraction | {PR_FRACTION_THRESHOLD:.2f} |
            | Negative-control PR maximum | {NEGCON_PR_THRESHOLD:.2f} |
            | Positive-control PR minimum | {POSCON_PR_THRESHOLD:.2f} |
            | Cross-plate PR-drop warning | {BATCH_DROP_THRESHOLD:.2f} |
            | Minimum replicates for stable PR | {MIN_REPS_FOR_PR} |
            """
        ),
        kind="info",
    )
    return (
        BATCH_DROP_THRESHOLD,
        MIN_REPS_FOR_PR,
        MIN_WITHIN_PR_FOR_FLAG,
        NEGCON_PR_THRESHOLD,
        NULL_PERCENTILE,
        NULL_SIZE,
        NULL_SIZE_COPAIRS_LOCAL,
        N_NULL_PAIRS,
        POSCON_PR_THRESHOLD,
        PR_FRACTION_THRESHOLD,
        RANDOM_STATE,
    )


@app.cell
def _(EXPERIMENT_ID, REPO_ROOT, mo, print):
    WORKSPACE_DIR = REPO_ROOT / "workspace"
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    PROFILES_DIR = WORKSPACE_DIR / "profiles" / EXPERIMENT_ID

    FIGS_DIR = ANALYSIS_DIR / "figures" / "quality_metrics"
    RESULTS_DIR = ANALYSIS_DIR / "results" / "quality_metrics"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mo.callout(
        mo.md(
            f"""
            ### QC paths ready

            - NB02 profiles: `{PROFILES_DIR / 'outputs'}`
            - Results: `{RESULTS_DIR}`
            - Figures: `{FIGS_DIR}`
            """
        ),
        kind="info",
    )
    return FIGS_DIR, PROFILES_DIR, RESULTS_DIR


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Data loading & auto-detection

    Loads the feature-selected parquet from NB02 and resolves the actual
    metadata columns present in the data (handling casing variations)
    via `CONFIG.resolve_columns(df)`, which also sets
    `CONFIG.has_dose_axis` / `CONFIG.has_time_axis` based on what is
    actually found — no separate column-detection reimplementation needed
    here.
    """)
    return


@app.cell
def _(PROFILES_DIR, mo, pd, print):
    _candidate_paths = [
        PROFILES_DIR / "outputs" / "per_well_features_selected.parquet",
        PROFILES_DIR / "outputs" / "per_well_features_selected_merged.parquet",
    ]
    INPUT_PATH = next((p for p in _candidate_paths if p.exists()), None)
    _checked_paths = "\n".join(f"- `{path}`" for path in _candidate_paths)
    print("INPUT DISCOVERY")
    for _path in _candidate_paths:
        print(f"  {'✓ found' if _path.exists() else '— not found'}: {_path}")
    mo.stop(INPUT_PATH is None, mo.callout(mo.md("**NB03 stopped:** the NB02 feature-selected parquet was not found.\n\n" f"Paths checked:\n{_checked_paths}\n\nRun NB02 to completion and rerun this cell. Downstream cells were stopped to avoid cascading errors."), kind="warn"))

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Shape: {df.shape[0]} wells × {df.shape[1]} columns")
    return INPUT_PATH, df


@app.cell
def _(RAW_CONFIG, df, print):
    CONFIG = RAW_CONFIG.resolve_columns(df)

    print("=== Auto-Detection Results ===")
    print(f"  Plate column:      {CONFIG.plate_col}")
    print(f"  Well column:       {CONFIG.well_col}")
    print(f"  Treatment column:  {CONFIG.treatment_col}")
    print(f"  Concentration:     {CONFIG.concentration_col} (has_dose_axis={CONFIG.has_dose_axis})")
    print(f"  Time column:       {CONFIG.time_col} (has_time_axis={CONFIG.has_time_axis})")
    print(f"  Control type:      {CONFIG.control_type_col}")
    print(f"  Negcon values:     {CONFIG.negcon_values}")
    print(f"  Poscon values:     {CONFIG.poscon_values}")
    print(f"  Trt values:        {CONFIG.trt_values}")
    return (CONFIG,)


@app.cell
def _(CONFIG, df, infer_feature_cols, np, print):
    feat_cols = infer_feature_cols(df)
    if not feat_cols:
        raise ValueError("No feature columns were detected in df.")

    print(f"Feature columns:  {len(feat_cols)} "
          f"(prefixes: {sorted({c.split('_')[0] for c in feat_cols})})")

    _feature_values = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    _nan_count = int(_feature_values.isna().sum().sum())
    if _nan_count:
        raise ValueError(
            f"NB03 received {_nan_count:,} missing or infinite feature values. "
            "NB02 should export a finite feature matrix; rerun or review NB02 instead "
            "of silently replacing values with zero here."
        )

    print(f"\n=== Experiment Summary ===")
    print(f"  Plates: {df[CONFIG.plate_col].nunique()}")
    for _plate, _n in df.groupby(CONFIG.plate_col).size().items():
        print(f"    {_plate}: {_n} wells")
    print(f"  Total wells: {len(df)}")
    print(f"  Treatments: {df[CONFIG.treatment_col].nunique()}")
    if CONFIG.has_dose_axis:
        print(f"  Concentrations: {df[CONFIG.concentration_col].nunique()} unique values")
        print("\n  Treatment × concentration distribution:")
        print(df.groupby([CONFIG.treatment_col, CONFIG.concentration_col]).size()
              .reset_index(name="n_wells").to_string(index=False))
    if CONFIG.has_time_axis:
        print(f"  Time points: {sorted(df[CONFIG.time_col].unique())}")
    else:
        print("  Time points: 1 (single time point — time-course analysis will be skipped)")
    return (feat_cols,)


@app.cell
def _(CONFIG, df, mo, pd, print):
    control_inventory = (
        df.groupby([CONFIG.control_type_col, CONFIG.treatment_col], dropna=False)
        .size()
        .reset_index(name="n_wells")
        .sort_values([CONFIG.control_type_col, CONFIG.treatment_col])
    )
    _negcon_mask = df[CONFIG.control_type_col].isin(CONFIG.negcon_values)
    _poscon_mask = df[CONFIG.control_type_col].isin(CONFIG.poscon_values)
    n_negcon_wells = int(_negcon_mask.sum())
    n_poscon_wells = int(_poscon_mask.sum())
    print("CONTROL INVENTORY")
    print(f"  Negative-control wells: {n_negcon_wells:,}")
    print(f"  Positive-control wells: {n_poscon_wells:,}")
    mo.stop(
        n_negcon_wells == 0,
        mo.callout(
            mo.md(
                "**No configured negative-control wells were found.**\n\n"
                "Activity mAP and Cohen's d require a negative-control reference. "
                "Review the control vocabulary and platemap metadata before continuing."
            ),
            kind="danger",
        ),
    )
    if n_poscon_wells == 0:
        print(
            "  ⚠ No configured positive-control wells were found. QC will continue so the "
            "missing evidence is recorded as an explicit critical FAIL."
        )
    mo.vstack(
        [
            mo.md(
                """
                ### Control inventory

                Negative controls define the activity reference and are required
                to continue. Positive controls validate that the assay can produce
                a detectable phenotype; if absent, the notebook continues only to
                record that missing evidence as a critical QC failure.
                """
            ),
            control_inventory,
        ]
    )
    return control_inventory, n_negcon_wells, n_poscon_wells


@app.cell
def _(mo):
    mo.md(r"""
    ### SC-18 — Profile integrity and replication readiness

    **What it evaluates:** whether NB02 supplied one finite profile per well,
    complete grouping metadata, and enough replicates for stable PR estimates.

    Duplicate wells or missing treatment/control labels are **blocking** because
    they invalidate pair definitions. Low replicate counts are **advisory**:
    metrics can still be computed, but their uncertainty is higher. PASS here
    confirms structural readiness only; it does not mean the controls worked or
    that treatments produced a phenotype.
    """)
    return


@app.cell
def _(CONFIG, MIN_REPS_FOR_PR, df, mo, print):
    reps_per_plate = df.groupby([CONFIG.plate_col, CONFIG.treatment_col]).size()
    reps_total = df.groupby(CONFIG.treatment_col).size()

    low_rep_plates = reps_per_plate[reps_per_plate < MIN_REPS_FOR_PR]
    low_rep_total = reps_total[reps_total < 10]

    print("=== Replicate Count Assessment ===")
    if len(low_rep_plates) > 0:
        print(f"  WARNING: {len(low_rep_plates)} treatment×plate combinations have "
              f"< {MIN_REPS_FOR_PR} replicates:")
        for (_plate, _trt), _n in low_rep_plates.items():
            print(f"    {_trt} on {_plate}: {_n} replicates (PR may be unstable)")
    else:
        print(f"  All treatments have ≥ {MIN_REPS_FOR_PR} replicates per plate ✓")

    if len(low_rep_total) > 0:
        print(f"  WARNING: {len(low_rep_total)} treatments have < 10 total replicates:")
        for _trt, _n in low_rep_total.items():
            print(f"    {_trt}: {_n} total (cross-plate PR recommended over within-plate)")
    else:
        print("  All treatments have ≥ 10 total replicates ✓")

    sc18_checks = [
        ("Dataset contains at least one well profile", not df.empty),
        (
            "Exactly one profile per plate/well",
            not df.duplicated([CONFIG.plate_col, CONFIG.well_col]).any(),
        ),
        ("No empty treatment labels", not df[CONFIG.treatment_col].isna().any()),
        ("No empty control-type labels", not df[CONFIG.control_type_col].isna().any()),
    ]
    print("\n=== SC-18: Profile integrity and replication readiness ===")
    for _name, _passed in sc18_checks:
        print(f"  [{'PASS' if _passed else 'FAIL'}] {_name}")
    sc18_pass = all(_passed for _, _passed in sc18_checks)
    sc18_status = "PASS" if sc18_pass else "FAIL"
    print(f"  SC-18 overall: {sc18_status}")
    mo.stop(
        not sc18_pass,
        mo.callout(
            mo.md(
                "**SC-18 failed: the profile table is not structurally valid for QC.**\n\n"
                "Review duplicate plate/well rows and missing treatment/control metadata in NB02."
            ),
            kind="danger",
        ),
    )
    return low_rep_plates, low_rep_total, sc18_checks, sc18_status


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — From-scratch vs. copairs: implementation check + sensitivity analysis

    `hca_pipeline.metrics_qc` provides two independent implementations of
    replicate-consistency / retrieval metrics:

    1. **From-scratch** (Pearson correlation, transparent) —
       `compute_pairwise_correlations`, `null_distribution_batch_aware`,
       `percent_replicating`, `percent_matching`, `mean_average_precision`.
    2. **`copairs`** (FDR-corrected p-values, configurable pair definitions,
       any `scipy`-supported distance metric) — `copairs_compute_map` and
       friends.

    Comparing them is **two different questions**, not one:

    - **Implementation check** — run copairs with `distance="correlation"`
      (Pearson, matching the from-scratch metric exactly). Both sides then
      rank neighbors identically, so a disagreement beyond the threshold here
      is a real signal that one implementation has a bug. This is the only
      one of the two comparisons with a PASS/INVESTIGATE threshold.
    - **Sensitivity analysis** — run copairs with its default
      `distance="cosine"`. Pearson and cosine similarity generally rank
      neighbors differently, so disagreement here reflects that deliberate
      methodological choice, not a bug — there is **no pass/fail threshold**
      for this comparison; a fixed threshold would conflate "different
      similarity metric, as designed" with "implementation error" (the
      earlier version of this notebook did exactly that).

    Both copairs runs also differ from the from-scratch result in
    aggregation (copairs reports a **macro-average**; from-scratch reports a
    **micro-average**) — a small, expected difference in the summary
    statistic itself, not in the per-treatment AP values.
    """)
    return


@app.cell
def _(
    CONFIG,
    NULL_SIZE,
    RANDOM_STATE,
    crossvalidate_scratch_vs_copairs_map,
    df,
    feat_cols,
    print,
):
    crossval_result = crossvalidate_scratch_vs_copairs_map(
        df, feat_cols, CONFIG.treatment_col, null_size=NULL_SIZE, seed=RANDOM_STATE,
    )
    _impl = crossval_result["implementation_check"]
    _sens = crossval_result["sensitivity_analysis"]

    print("=== Implementation check: From-scratch (Pearson) vs copairs (Pearson) ===\n")
    print(f"{'Treatment':<20} {'From-scratch':>12} {'copairs':>10} {'Diff':>8}")
    print("-" * 52)
    for _trt in sorted(_impl["per_treatment"].keys()):
        _row = _impl["per_treatment"][_trt]
        _flag = " ✓" if abs(_row["diff"]) <= _impl["agreement_threshold"] else " **"
        print(f"{_trt:<20} {_row['from_scratch']:>12.3f} {_row['copairs']:>10.3f} {_row['diff']:>8.3f}{_flag}")
    print(f"\nMax per-treatment difference: {_impl['max_abs_diff']:.4f}")
    print(f"Implementation check: {'PASS' if _impl['agrees'] else 'INVESTIGATE'} "
          f"(threshold: ±{_impl['agreement_threshold']})")
    print(f"copairs (Pearson) macro-average PC mAP: {_impl['map_copairs_macro']:.3f}")
    print(f"From-scratch micro-average PC mAP:      {crossval_result['map_from_scratch_micro']:.3f}")

    print("\n=== Sensitivity analysis: Pearson (from-scratch) vs cosine (copairs) ===\n")
    print("No pass/fail threshold — disagreement here reflects the similarity-metric")
    print("choice, not an implementation error.\n")
    print(f"{'Treatment':<20} {'From-scratch':>12} {'copairs':>10} {'Diff':>8}")
    print("-" * 52)
    for _trt in sorted(_sens["per_treatment"].keys()):
        _row = _sens["per_treatment"][_trt]
        print(f"{_trt:<20} {_row['from_scratch']:>12.3f} {_row['copairs']:>10.3f} {_row['diff']:>8.3f}")
    print(f"\nMax per-treatment difference: {_sens['max_abs_diff']:.4f}")
    print(f"copairs (cosine) macro-average PC mAP: {_sens['map_copairs_macro']:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Application A: Per-plate QC

    **Question:** Does each plate independently show replicate consistency?

    For each plate: within-plate PR, within-plate PM, and within-plate
    **phenotypic-consistency mAP (PC mAP)**
    (from-scratch + copairs, cross-validated). **SC-19**: negcon PR ≤
    threshold, poscon PR > threshold.

    **How to read SC-19:** PASS requires both control classes to be present on
    every plate and to satisfy their configured consistency rules. Missing
    negative or positive controls produce FAIL rather than an automatic pass.
    A failure is critical: inspect the platemap/control vocabulary and control
    wells before interpreting treatment phenotypes. SC-19 evaluates replicate
    consistency, not activity relative to control; positive-control activity is
    evaluated separately in SC-22.
    """)
    return


@app.cell
def _(
    CONFIG,
    NEGCON_PR_THRESHOLD,
    NULL_PERCENTILE,
    NULL_SIZE_COPAIRS_LOCAL,
    N_NULL_PAIRS,
    POSCON_PR_THRESHOLD,
    PR_FRACTION_THRESHOLD,
    RANDOM_STATE,
    df,
    feat_cols,
    print,
    run_per_plate_qc,
):
    qc_result = run_per_plate_qc(
        df, feat_cols, CONFIG.treatment_col, CONFIG.plate_col, CONFIG.control_type_col,
        negcon_values=CONFIG.negcon_values, poscon_values=CONFIG.poscon_values,
        null_percentile=NULL_PERCENTILE, n_null=N_NULL_PAIRS, seed=RANDOM_STATE,
        null_size_copairs=NULL_SIZE_COPAIRS_LOCAL,
        negcon_pr_threshold=NEGCON_PR_THRESHOLD, poscon_pr_threshold=POSCON_PR_THRESHOLD,
        pr_fraction_threshold=PR_FRACTION_THRESHOLD,
    )

    print("=== Application A: Per-Plate QC ===\n")
    print("--- QC Flags ---")
    print(qc_result["qc_flags"].to_string(index=False))
    print("\n--- SC-19 Status ---")
    print(qc_result["sc19_status"].to_string(index=False))
    _missing_control_evidence = qc_result["sc19_status"][
        ~qc_result["sc19_status"]["negcon_available"]
        | ~qc_result["sc19_status"]["poscon_available"]
    ]
    if not _missing_control_evidence.empty:
        print("\nSC-19 FAIL — one or more plates lack evaluable negative- or positive-control profiles")
    print("\n--- Percent Matching (PM), per plate ---")
    print(qc_result["pm_results"].to_string(index=False))
    print("\n--- Per-plate PR ---")
    print(qc_result["pr_results"][[CONFIG.plate_col, "treatment", "control_type",
        "median_corr", "null_threshold", "passes_pr"]].to_string(index=False))
    print("\n--- Phenotypic-consistency mAP (PC mAP) Cross-Validation ---")
    print(
        qc_result["map_results"][
            [CONFIG.plate_col, "treatment", "mAP_from_scratch", "mAP_copairs", "mAP_diff"]
        ]
        .rename(
            columns={
                "mAP_from_scratch": "PC_mAP_from_scratch",
                "mAP_copairs": "PC_mAP_copairs",
                "mAP_diff": "PC_mAP_difference",
            }
        )
        .to_string(index=False)
    )
    return (qc_result,)


@app.cell
def _(CONFIG, FIGS_DIR, OVERWRITE_EXISTING_OUTPUTS, plot_per_plate_qc, print, qc_result):
    _output_path = FIGS_DIR / "app_a_per_plate_qc.png"
    _existed = _output_path.exists()
    _write_path = None if _existed and not OVERWRITE_EXISTING_OUTPUTS else str(_output_path)
    _figure = plot_per_plate_qc(qc_result, CONFIG.plate_col, output_path=_write_path)
    print(
        f"ℹ Existing figure protected; current figure displayed but not written: {_output_path}"
        if _write_path is None
        else f"✓ Figure {'replaced' if _existed else 'created'} and displayed: {_output_path}"
    )
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — Application B: Cross-plate batch assessment

    **Question:** Do replicates agree across plates? Is there a batch
    effect? Only runs when more than one plate is present — otherwise
    `run_cross_plate_batch` returns a `{"message": ...}` sentinel and this
    section reports that it was skipped. **SC-20**: cross-plate PR drop
    below the configured threshold (non-critical / warning-level check).

    Here, both within- and cross-plate mAP are **PC mAP**: positives are the
    same treatment, with the cross-plate calculation requiring positives to
    come from different plates.

    **How to read SC-20:** SKIP is the correct result for a single plate, not a
    failure. With multiple plates, WARN means cross-plate consistency drops
    beyond the configured limit for otherwise reproducible treatments. This is
    evidence of technical plate structure, not proof that biological effects
    are absent.
    """)
    return


@app.cell
def _(
    BATCH_DROP_THRESHOLD,
    CONFIG,
    MIN_WITHIN_PR_FOR_FLAG,
    NULL_PERCENTILE,
    NULL_SIZE_COPAIRS_LOCAL,
    N_NULL_PAIRS,
    RANDOM_STATE,
    df,
    feat_cols,
    print,
    run_cross_plate_batch,
):
    batch_result = run_cross_plate_batch(
        df, feat_cols, CONFIG.treatment_col, CONFIG.plate_col, CONFIG.control_type_col,
        negcon_values=CONFIG.negcon_values, poscon_values=CONFIG.poscon_values,
        null_percentile=NULL_PERCENTILE, n_null=N_NULL_PAIRS, seed=RANDOM_STATE,
        null_size_copairs=NULL_SIZE_COPAIRS_LOCAL,
        batch_drop_threshold=BATCH_DROP_THRESHOLD, min_within_pr_for_flag=MIN_WITHIN_PR_FOR_FLAG,
    )

    if "message" in batch_result:
        print(f"SC-20 SKIP — {batch_result['message']}")
    else:
        print("=== Application B: Cross-Plate Batch Assessment ===\n")
        print(f"Plates: {batch_result['n_plates']}")
        print(f"Cross-plate PR: {batch_result['cross_plate_pr_value']:.3f}")
        print(f"Cross-plate null threshold: {batch_result['cross_plate_threshold']:.3f}")
        print("\n--- Batch Assessment ---")
        print(batch_result["batch_assessment"][["treatment", "within_plate_pr", "cross_plate_pr",
            "pr_drop_pct", "cross_plate_map", "batch_effect_meaningful"]].to_string(index=False))
        _sc20 = batch_result["sc20_status"]
        print("\n--- SC-20 ---")
        print(f"  Meaningful batch-affected: {_sc20['n_batch_affected']}")
        print(f"  Affected: {_sc20['batch_affected_treatments']}")
        print(f"  SC-20: {'PASS' if _sc20['sc20_pass'] else 'WARN'}")
    return (batch_result,)


@app.cell
def _(FIGS_DIR, OVERWRITE_EXISTING_OUTPUTS, batch_result, plot_cross_plate_batch, print):
    _output_path = FIGS_DIR / "app_b_cross_plate_batch.png"
    _write_path = None if _output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS else str(_output_path)
    _figure = plot_cross_plate_batch(batch_result, output_path=_write_path)
    print(
        "ℹ Figure skipped: cross-plate analysis is not applicable."
        if _figure is None
        else f"ℹ Existing figure protected; current figure displayed but not written: {_output_path}"
        if _write_path is None
        else f"✓ Figure written and displayed: {_output_path}"
    )
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Application C: Dose-response consistency

    **Question:** Do increasing doses show consistent ordering in profile
    space? Auto-detects treatments with multiple concentrations. Only runs
    when a concentration column is present and multi-dose treatments are
    detected — otherwise skipped with a message. **SC-21**: adjacent doses
    should correlate higher than non-adjacent doses (monotonicity,
    non-critical / informational check).

    The displayed dose-retrieval score is **PC mAP**: positives are replicates
    of the same treatment–dose and negatives are other doses of that treatment.
    It measures within-dose consistency, not dose monotonicity or MoA
    distinctiveness.

    **How to read SC-21:** PASS supports orderly dose-related change; WARN
    identifies a non-monotonic series to inspect; SKIP means the design has no
    evaluable multi-dose treatment. It is informational and does not determine
    whether a phenotype is active.
    """)
    return


@app.cell
def _(CONFIG, NULL_SIZE, RANDOM_STATE, df, feat_cols, print, run_dose_response):
    dose_result = run_dose_response(
        df, feat_cols, CONFIG.treatment_col, CONFIG.concentration_col,
        null_size_copairs=NULL_SIZE, seed=RANDOM_STATE,
    )

    if "message" in dose_result:
        print(f"SC-21 SKIP — {dose_result['message']}")
    else:
        print("=== Application C: Dose-Response ===\n")
        print(f"Multi-dose treatments: {dose_result['multi_dose_treatments']}")
        for _r in dose_result["dose_correlations"]:
            print(f"\n{_r['treatment']} ({_r['n_doses']} doses: {_r['doses']})")
            print(f"  Mean pairwise corr: {_r['mean_pairwise_corr']:.3f}")
            print(f"  Adjacent: {_r['adjacent_corr_mean']:.3f}" if _r["adjacent_corr_mean"] is not None else "  Adjacent: N/A")
        print("\n--- SC-21 ---")
        print(dose_result["sc21_status"].to_string(index=False))
        if dose_result["dose_map"] is not None:
            print("\n--- Dose-level phenotypic-consistency mAP (PC mAP) ---")
            print(
                dose_result["dose_map"][["Metadata_Treatment_Dose", "mean_average_precision",
                    "corrected_p_value"]]
                .rename(columns={"mean_average_precision": "PC_mean_average_precision"})
                .to_string(index=False)
            )
    return (dose_result,)


@app.cell
def _(FIGS_DIR, OVERWRITE_EXISTING_OUTPUTS, dose_result, plot_dose_response, print):
    _output_path = FIGS_DIR / "app_c_dose_response.png"
    _write_path = None if _output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS else str(_output_path)
    _figure = plot_dose_response(dose_result, output_path=_write_path)
    print(
        "ℹ Figure skipped: dose-response analysis is not applicable."
        if _figure is None
        else f"ℹ Existing figure protected; current figure displayed but not written: {_output_path}"
        if _write_path is None
        else f"✓ Figure written and displayed: {_output_path}"
    )
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — Application D: Treatment vs control separation

    **Question:** Can treatment profiles be distinguished from negative
    controls? For each treatment: **phenotypic-activity mAP** vs negcon
    (copairs, reference
    indexing), FDR-corrected p-value, and Cohen's d effect size. **SC-22**:
    positive-control activity mAP should be high and configured negative-control
    treatments should remain near the pooled negative-control baseline.
    Experimental treatments with activity mAP ≈ null are reported as an
    informative biological result, not a QC failure.

    **Cohen's d and activity mAP answer different questions.** Cohen's d measures
    *effect magnitude* (how big is the change?); activity mAP measures
    *detectability* (how consistently is that change recovered across
    replicates?). A small but highly reproducible phenotype can score
    higher on activity mAP than a large but heterogeneous one — looking at either
    metric alone can't distinguish these cases:

    | | High activity mAP | Low activity mAP |
    |---|---|---|
    | **High Cohen's d** | Strong, consistent phenotype | Strong but heterogeneous phenotype |
    | **Low Cohen's d** | Subtle but reproducible phenotype | No detectable (or weak) phenotype |

    - **High d + High activity mAP** — robust phenotypic signature: large and reproducible changes.
    - **High d + Low activity mAP** — large effect size but heterogeneous response; the phenotype exists but isn't consistently recovered.
    - **Low d + High activity mAP** — small but highly reproducible phenotype: subtle changes, consistently detected.
    - **Low d + Low activity mAP** — no detectable phenotypic signature, or changes indistinguishable from the negative control.

    **How to read SC-22:** this critical control/phenotype check fails when a
    positive control lacks activity, when a configured negative-control
    treatment departs from the pooled negative-control baseline. Inactive tested
    treatments are listed as INFO because an inactive compound does not mean the
    experiment failed. Missing positive-control evidence is reported as FAIL,
    never PASS. A failure calls for control-well/platemap
    review and may justify rerunning with a scientifically supported control
    definition; it does not justify silently relabeling controls.
    """)
    return


@app.cell
def _(
    CONFIG,
    NULL_SIZE,
    RANDOM_STATE,
    df,
    feat_cols,
    print,
    run_treatment_vs_control,
):
    tvc_result = run_treatment_vs_control(
        df, feat_cols, CONFIG.treatment_col, CONFIG.control_type_col,
        negcon_values=CONFIG.negcon_values, poscon_values=CONFIG.poscon_values,
        plate_col=CONFIG.plate_col, null_size_copairs=NULL_SIZE, seed=RANDOM_STATE,
    )

    print("=== Application D: Treatment vs Control ===\n")
    print(
        tvc_result["results"]
        .rename(
            columns={
                "mAP_vs_negcon": "activity_mAP_vs_negcon",
                "normalized_mAP": "normalized_activity_mAP",
            }
        )
        .to_string(index=False)
    )
    print("\n--- SC-22 ---")
    for _check in tvc_result["sc22_status"]["checks"]:
        _status = (
            "INFO"
            if not _check.get("critical", True)
            else "PASS"
            if _check["pass"]
            else "FAIL"
        )
        print(f"  [{_status}] {_check['description']}")
    print(f"  Overall: {'PASS' if tvc_result['sc22_status']['sc22_pass'] else 'FAIL'}")

    _non_negcon = tvc_result["results"][~tvc_result["results"]["is_negcon"]]
    if len(_non_negcon) > 1:
        print(f"\nCohen's d vs phenotypic-activity mAP correlation: r = {_non_negcon['cohens_d'].corr(_non_negcon['mAP_vs_negcon']):.3f}")

    print("\n--- Effect magnitude x detectability quadrant (per treatment) ---")
    for _row in _non_negcon.itertuples(index=False):
        print(
            f"  {_row.treatment:<20} d={_row.cohens_d:.3f}  activity mAP={_row.mAP_vs_negcon:.3f}  "
            f"→ {_row.effect_detectability_quadrant}"
        )
    return (tvc_result,)


@app.cell
def _(CONFIG, mo, qc_result, tvc_result):
    _control_profiles = tvc_result["results"][
        tvc_result["results"]["is_negcon"] | tvc_result["results"]["is_poscon"]
    ][
        [
            "treatment",
            "control_type",
            "is_negcon",
            "is_poscon",
            "mAP_vs_negcon",
            "normalized_mAP",
            "fdr_significant",
            "cohens_d",
            "effect_detectability_quadrant",
        ]
    ].rename(
        columns={
            "mAP_vs_negcon": "activity_mAP_vs_negcon",
            "normalized_mAP": "normalized_activity_mAP",
        }
    )
    _sc19_failures = int((~qc_result["sc19_status"]["sc19_pass"]).sum())
    _sc22_failures = sum(
        not _check["pass"] for _check in tvc_result["sc22_status"]["checks"]
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ### Integrated control-performance review

                This table brings the control evidence together. SC-19 checks
                **within-plate consistency**; SC-22 checks **activity relative to
                the negative-control reference**. A valid positive control should
                be reproducible and active. A negative-control treatment should
                remain near the pooled negative-control baseline.

                - Configured negative controls: {', '.join(f'`{v}`' for v in CONFIG.negcon_values) or 'none'}
                - Configured positive controls: {', '.join(f'`{v}`' for v in CONFIG.poscon_values) or 'none'}
                - Plates failing SC-19: **{_sc19_failures}**
                - Failed SC-22 control/activity checks: **{_sc22_failures}**

                If control evidence fails, treat the Go/No-Go result as an
                experimental/QC problem to investigate before downstream
                interpretation. Do not silently remove or relabel a control.
                """
            ),
            _control_profiles,
        ]
    )
    return (_control_profiles,)


@app.cell
def _(FIGS_DIR, OVERWRITE_EXISTING_OUTPUTS, plot_treatment_vs_control, print, tvc_result):
    _output_path = FIGS_DIR / "app_d_treatment_vs_control.png"
    _write_path = None if _output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS else str(_output_path)
    _figure = plot_treatment_vs_control(tvc_result, output_path=_write_path)
    print(
        f"ℹ Existing figure protected; current figure displayed but not written: {_output_path}"
        if _write_path is None
        else f"✓ Figure written and displayed: {_output_path}"
    )
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — Time-course analysis (conditional)

    Only runs if `CONFIG.has_time_axis` is `True` (a time/exposure column
    with more than one unique value was detected during column
    resolution). With a single time point, SKIP is the correct result rather
    than an error. The framework activates automatically for time-course
    experiments with no notebook changes.
    """)
    return


@app.cell
def _(
    CONFIG,
    NULL_PERCENTILE,
    NULL_SIZE_COPAIRS_LOCAL,
    N_NULL_PAIRS,
    RANDOM_STATE,
    df,
    feat_cols,
    print,
    run_time_course,
):
    time_result = run_time_course(
        df, feat_cols, CONFIG.treatment_col, CONFIG.time_col, CONFIG.control_type_col,
        CONFIG.negcon_values, CONFIG.poscon_values, plate_col=CONFIG.plate_col,
        null_percentile=NULL_PERCENTILE, n_null=N_NULL_PAIRS, seed=RANDOM_STATE,
        null_size_copairs=NULL_SIZE_COPAIRS_LOCAL,
    )

    if "message" in time_result:
        print(time_result["message"])
        print("The framework is ready for time-course experiments — this section "
              "will activate automatically when time metadata is present.")
    else:
        print("=== Section 8: Time-Course Analysis ===\n")
        print(f"Time points: {time_result['time_values']}")
        print(f"Optimal time: {time_result['optimal_time']}")
        print("\n--- Summary ---")
        print(time_result["time_course_summary"][[time_result["time_col"], "pr_fraction",
            "n_passing", "n_treatments", "mAP_copairs"]].to_string(index=False))
    return (time_result,)


@app.cell
def _(FIGS_DIR, OVERWRITE_EXISTING_OUTPUTS, plot_time_course, print, time_result):
    _output_path = FIGS_DIR / "section8_time_course.png"
    _write_path = None if _output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS else str(_output_path)
    _figure = plot_time_course(time_result, output_path=_write_path)
    print(
        "ℹ Figure skipped: time-course analysis is not applicable."
        if _figure is None
        else f"ℹ Existing figure protected; current figure displayed but not written: {_output_path}"
        if _write_path is None
        else f"✓ Figure written and displayed: {_output_path}"
    )
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 — Summary & Go/No-Go decision

    Aggregates all four applications into a single dashboard and a
    **GO / GO WITH CAUTION / NO-GO** verdict.

    | Decision | Criteria |
    |----------|----------|
    | **GO** | All critical and non-critical checks pass |
    | **GO WITH CAUTION** | All critical checks pass; some non-critical warnings |
    | **NO-GO** | One or more critical checks fail |

    **Critical:** SC-19 (control validation), PR-25% (PR fraction ≥
    threshold per plate), SC-22 (treatment phenotype detectability).
    **Non-critical:** SC-20 (batch effect), SC-21 (dose monotonicity),
    TIME (informational).
    """)
    return


@app.cell
def _(
    CONFIG,
    batch_result,
    control_inventory,
    dose_result,
    generate_go_nogo_dashboard,
    qc_result,
    sc18_status,
    time_result,
    tvc_result,
    pd,
    print,
):
    dashboard = generate_go_nogo_dashboard(qc_result, batch_result, dose_result, tvc_result, time_result)
    _sc18_row = pd.DataFrame(
        [
            {
                "check_id": "SC-18",
                "name": "Profile integrity and replication readiness",
                "n_pass": int(sc18_status == "PASS"),
                "n_total": 1,
                "pass": sc18_status == "PASS",
                "critical": True,
                "details": "Profile table is structurally valid for QC",
            }
        ]
    )
    dashboard["checks"] = pd.concat([_sc18_row, dashboard["checks"]], ignore_index=True)
    dashboard["analysis_mode"] = CONFIG.analysis_mode
    dashboard["reported_decision"] = (
        f"PRELIMINARY — {dashboard['decision']}"
        if CONFIG.analysis_mode == "preliminary"
        else dashboard["decision"]
    )

    print("=== Section 9: Go/No-Go Dashboard ===\n")
    print(f"DECISION: {dashboard['reported_decision']}")
    if CONFIG.analysis_mode == "preliminary":
        print("This is an early plate-scope assessment; rerun in final mode after all planned plates are included.")
    print(f"Rationale: {dashboard['rationale']}\n")
    print("--- Checks ---")
    print(dashboard["checks"][["check_id", "name", "n_pass", "n_total", "pass", "critical",
        "details"]].to_string(index=False))
    print("\n--- Metric Summary ---")
    for _k, _v in dashboard["metric_summary"].items():
        print(f"  {_k}: {_v:.3f}" if isinstance(_v, float) else f"  {_k}: {_v}")
    return (dashboard,)


@app.cell
def _(batch_result, dose_result, mo, pd, qc_result, sc18_status, tvc_result):
    _sc19_pass = bool(qc_result["sc19_status"]["sc19_pass"].all())
    if "message" in batch_result:
        _sc20_status = "SKIP"
        _sc20_evidence = batch_result["message"]
    else:
        _sc20_status = "PASS" if batch_result["sc20_status"]["sc20_pass"] else "WARN"
        _sc20_evidence = (
            f"{batch_result['sc20_status']['n_batch_affected']} treatment(s) with meaningful cross-plate PR loss"
        )
    if "message" in dose_result:
        _sc21_status = "SKIP"
        _sc21_evidence = dose_result["message"]
    else:
        _sc21_table = dose_result["sc21_status"]
        _sc21_pass = bool(_sc21_table["sc21_pass"].all()) if "sc21_pass" in _sc21_table else True
        _sc21_status = "PASS" if _sc21_pass else "WARN"
        _sc21_evidence = f"{int(_sc21_table['sc21_pass'].sum())}/{len(_sc21_table)} multi-dose treatment(s) pass" if "sc21_pass" in _sc21_table else "No evaluable monotonicity result"

    sanity_check_summary = pd.DataFrame(
        [
            {
                "Check": "SC-18 — Profile integrity and replication readiness",
                "Status": sc18_status,
                "Evidence": "One finite profile per well with complete grouping metadata",
                "How to act": "Blocking if FAIL; return to NB02",
            },
            {
                "Check": "SC-19 — Per-plate control consistency",
                "Status": "PASS" if _sc19_pass else "FAIL",
                "Evidence": f"{int(qc_result['sc19_status']['sc19_pass'].sum())}/{len(qc_result['sc19_status'])} plate(s) pass",
                "How to act": "Review missing or inconsistent control wells if FAIL",
            },
            {
                "Check": "SC-20 — Cross-plate consistency",
                "Status": _sc20_status,
                "Evidence": _sc20_evidence,
                "How to act": "SKIP is expected for one plate; inspect technical structure if WARN",
            },
            {
                "Check": "SC-21 — Dose-response ordering",
                "Status": _sc21_status,
                "Evidence": _sc21_evidence,
                "How to act": "Inspect non-monotonic dose series if WARN",
            },
            {
                "Check": "SC-22 — Control activity and phenotype detectability",
                "Status": "PASS" if tvc_result["sc22_status"]["sc22_pass"] else "FAIL",
                "Evidence": "; ".join(
                    _check["description"]
                    for _check in tvc_result["sc22_status"]["checks"]
                    if _check.get("critical", True)
                ),
                "How to act": "Review control identity/performance before downstream interpretation if FAIL",
            },
        ]
    )
    mo.vstack(
        [
            mo.md(
                """
                ### Sanity-check interpretation summary

                PASS supports the stated technical criterion; WARN requires
                review but is not automatically blocking; FAIL invalidates a
                critical QC criterion; SKIP means the analysis is not defined
                for this design. None of these labels alone establishes a
                mechanism of action.
                """
            ),
            sanity_check_summary,
        ]
    )
    return (sanity_check_summary,)


@app.cell
def _(FIGS_DIR, OVERWRITE_EXISTING_OUTPUTS, dashboard, plot_go_nogo_dashboard, print):
    _output_path = FIGS_DIR / "section9_go_nogo_dashboard.png"
    _write_path = None if _output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS else str(_output_path)
    _figure = plot_go_nogo_dashboard(dashboard, output_path=_write_path)
    print(
        f"ℹ Existing figure protected; current figure displayed but not written: {_output_path}"
        if _write_path is None
        else f"✓ Figure written and displayed: {_output_path}"
    )
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 — Protected export of results

    All tables are small per-plate/per-treatment summaries, so they are
    written with `write_summary_table_protected` (full-content compare)
    rather than the header-only `write_csv_protected` used for large
    per-well/per-cell tables elsewhere in the pipeline.
    """)
    return


@app.cell
def _(
    CONFIG,
    RESULTS_DIR,
    batch_result,
    dashboard,
    dose_result,
    mo,
    pd,
    print,
    qc_result,
    sc18_checks,
    sanity_check_summary,
    tvc_result,
    write_summary_table_protected,
):
    _overwrite = CONFIG.overwrite_existing_outputs

    def _export(table, filename):
        try:
            status = write_summary_table_protected(
                table, RESULTS_DIR / filename, overwrite=_overwrite
            )
        except FileExistsError as _error:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        "**A protected NB03 table differs from the current result.**\n\n"
                        f"`{_error}`\n\nReview the changed inputs or QC settings. If replacement is "
                        "intentional, enable **Overwrite existing outputs** under Advanced settings."
                    ),
                    kind="danger",
                ),
            )
        print(f"  {filename}: {status}")

    print("=== Application A outputs ===")
    _export(
        pd.DataFrame(
            [
                {"check": _name, "pass": _passed}
                for _name, _passed in sc18_checks
            ]
        ),
        "sc18_profile_integrity.csv",
    )
    _export(control_inventory, "control_inventory.csv")
    _export(sanity_check_summary, "sanity_check_summary.csv")
    _export(qc_result["pr_results"], "app_a_per_plate_pr.csv")
    _export(qc_result["pm_results"], "app_a_per_plate_pm.csv")
    _export(
        qc_result["map_results"].rename(
            columns={
                "mAP_from_scratch": "PC_mAP_from_scratch",
                "mAP_copairs": "PC_mAP_copairs",
                "mAP_diff": "PC_mAP_difference",
            }
        ),
        "app_a_per_plate_map.csv",
    )
    _export(qc_result["qc_flags"], "app_a_qc_flags.csv")
    _export(qc_result["sc19_status"], "app_a_sc19_status.csv")

    print("=== Application B outputs ===")
    if "message" not in batch_result:
        _export(batch_result["batch_assessment"], "app_b_batch_assessment.csv")
        _export(
            batch_result["cross_plate_map"][[CONFIG.treatment_col, "mean_average_precision",
                "p_value", "corrected_p_value"]].rename(
                    columns={"mean_average_precision": "cross_plate_PC_mean_average_precision"}
                ),
            "app_b_cross_plate_map.csv",
        )
    else:
        print(f"  skipped: {batch_result['message']}")

    print("=== Application C outputs ===")
    if "message" not in dose_result:
        if dose_result.get("dose_correlations"):
            _dose_corr_rows = []
            for _r in dose_result["dose_correlations"]:
                _n = _r["n_doses"]
                for _i in range(_n):
                    for _j in range(_i + 1, _n):
                        _dose_corr_rows.append({
                            "treatment": _r["treatment"], "dose_1": _r["doses"][_i],
                            "dose_2": _r["doses"][_j], "correlation": _r["corr_matrix"][_i, _j],
                            "is_adjacent": _j == _i + 1,
                        })
            import pandas as _pd
            _export(_pd.DataFrame(_dose_corr_rows), "app_c_dose_correlations.csv")
        if dose_result.get("dose_map") is not None:
            _export(
                dose_result["dose_map"].rename(
                    columns={
                        "mean_average_precision": "PC_mean_average_precision",
                        "mean_normalized_average_precision": "normalized_PC_mean_average_precision",
                    }
                ),
                "app_c_dose_map.csv",
            )
        _export(dose_result["sc21_status"], "app_c_sc21_status.csv")
    else:
        print(f"  skipped: {dose_result['message']}")

    print("=== Application D outputs ===")
    _export(
        tvc_result["results"].rename(
            columns={
                "mAP_vs_negcon": "activity_mAP_vs_negcon",
                "normalized_mAP": "normalized_activity_mAP",
            }
        ),
        "app_d_treatment_vs_control.csv",
    )

    print("=== Section 9 outputs ===")
    _export(dashboard["checks"], "section9_dashboard_checks.csv")
    return


@app.cell
def _(CONFIG, EXPERIMENT_ID, RESULTS_DIR, dashboard, mo, print):
    _report_lines = [
        f"# Quality Metrics Go/No-Go Report — {EXPERIMENT_ID}",
        f"## Decision: {dashboard['reported_decision']}",
        f"\n**Analysis mode:** {dashboard['analysis_mode']}",
        f"\n**Rationale:** {dashboard['rationale']}",
        "\n## Metric Summary",
        f"- Plates: {dashboard['metric_summary']['n_plates']}",
        f"- Treatments: {dashboard['metric_summary']['n_treatments']}",
        f"- Mean PR fraction: {dashboard['metric_summary']['mean_pr_fraction']:.1%}",
        f"- Cross-plate PR: {dashboard['metric_summary']['cross_plate_pr']}",
        f"- Mean treatment activity mAP: {dashboard['metric_summary']['mean_treatment_map']:.3f}",
        f"- FDR significant treatments: {dashboard['metric_summary']['n_fdr_significant']}",
        f"- Positive-control activity mAP: {dashboard['metric_summary']['poscon_map']}",
        "\n## Check Details",
    ]
    for _, _row in dashboard["checks"].iterrows():
        _status = "PASS" if _row["pass"] else "FAIL" if _row["critical"] else "WARN"
        _crit = "CRITICAL" if _row["critical"] else "non-critical"
        _report_lines.append(f"- [{_status}] {_row['check_id']} ({_crit}): {_row['details']}")

    _report_path = RESULTS_DIR / "quality_metrics_report.md"
    _report_text = "\n".join(_report_lines)
    if _report_path.exists() and _report_path.read_text(encoding="utf-8") == _report_text:
        _report_status = "unchanged"
    elif _report_path.exists() and not CONFIG.overwrite_existing_outputs:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**The protected Markdown QC report differs from the current result.**\n\n"
                    f"Existing file: `{_report_path}`\n\nReview the new QC decision. Enable "
                    "**Overwrite existing outputs** only if replacement is intended."
                ),
                kind="danger",
            ),
        )
    else:
        _report_status = "replaced" if _report_path.exists() else "created"
        _report_path.write_text(_report_text, encoding="utf-8")
    print(f"✓ Go/No-Go report {_report_status}: {_report_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11 — Provenance
    """)
    return


@app.cell
def _(
    CONFIG,
    EXPERIMENT_ID,
    INPUT_PATH,
    REPO_ROOT,
    RESULTS_DIR,
    dashboard,
    datetime,
    df,
    feat_cols,
    json,
    n_negcon_wells,
    n_poscon_wells,
    platform,
    print,
    subprocess,
    timezone,
):
    from hca_pipeline.provenance import canonicalize_provenance, provenance_json

    _legacy_provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "analysis_mode": CONFIG.analysis_mode,
            "included_plates": CONFIG.included_plates,
            "excluded_plate_reasons": CONFIG.excluded_plate_reasons,
            "required_reference_treatments": CONFIG.required_reference_treatments,
            "overwrite_existing_outputs": CONFIG.overwrite_existing_outputs,
        },
        "dataset": {
            "n_wells": int(df.shape[0]),
            "n_features": int(len(feat_cols)),
            "n_negcon_wells": n_negcon_wells,
            "n_poscon_wells": n_poscon_wells,
        },
        "analysis": {
            "decision": dashboard["decision"],
            "reported_decision": dashboard["reported_decision"],
            "rationale": dashboard["rationale"],
            "metric_summary": dashboard["metric_summary"],
        },
    }

    _declared_outputs = [
        RESULTS_DIR / "quality_metrics_report.md",
        RESULTS_DIR / "section9_dashboard_checks.csv",
        RESULTS_DIR / "app_a_per_plate_map.csv",
        RESULTS_DIR / "app_d_treatment_vs_control.csv",
    ]
    provenance = canonicalize_provenance(
        _legacy_provenance,
        notebook="03_quality_metrics.py",
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
        dependencies=[INPUT_PATH],
        outputs=_declared_outputs,
    )

    _prov_path = RESULTS_DIR / "provenance.json"
    _prov_existed = _prov_path.exists()
    _provenance_payload = provenance_json(provenance)
    if _prov_existed and not CONFIG.overwrite_existing_outputs:
        _provenance_status = "unchanged (existing provenance protected)"
    else:
        _prov_path.write_text(_provenance_payload, encoding="utf-8")
        _provenance_status = "replaced" if _prov_existed else "created"

    print(f"✓ Latest provenance {_provenance_status}: {_prov_path}")
    print(f"  Schema:     v{provenance['schema_version']}")
    print(f"  Notebook:   {provenance['pipeline']['notebook']}")
    print(f"  Experiment: {provenance['pipeline']['experiment_id']}")
    print(f"  Timestamp:  {provenance['pipeline']['executed_at_utc']}")
    print(f"  Git hash:   {provenance['version_control']['git_commit_short']}")
    print(f"  Inputs:     {len(provenance['dependencies'])} hashed dependency record(s)")
    print(f"  Decision:   {provenance['analysis']['reported_decision']}")
    if CONFIG.save_provenance_history:
        _history_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _history_path = RESULTS_DIR / f"provenance_{_history_timestamp}.json"
        if _history_path.exists():
            raise FileExistsError(f"Historical provenance file already exists: {_history_path}")
        _history_path.write_text(
            _provenance_payload, encoding="utf-8"
        )
        print(f"✓ Timestamped provenance created: {_history_path}")
    else:
        print("  Timestamped provenance: disabled")
    return (provenance,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 12 — Final integrity checks
    """)
    return


@app.cell
def _(EXPERIMENT_ID, FIGS_DIR, Path, RESULTS_DIR, dashboard, print, provenance):
    _required_outputs = [
        RESULTS_DIR / "app_a_qc_flags.csv",
        RESULTS_DIR / "sc18_profile_integrity.csv",
        RESULTS_DIR / "control_inventory.csv",
        RESULTS_DIR / "sanity_check_summary.csv",
        RESULTS_DIR / "app_a_sc19_status.csv",
        RESULTS_DIR / "app_d_treatment_vs_control.csv",
        RESULTS_DIR / "section9_dashboard_checks.csv",
        RESULTS_DIR / "quality_metrics_report.md",
        RESULTS_DIR / "provenance.json",
        FIGS_DIR / "app_a_per_plate_qc.png",
        FIGS_DIR / "app_d_treatment_vs_control.png",
        FIGS_DIR / "section9_go_nogo_dashboard.png",
    ]
    _missing = [p for p in _required_outputs if not Path(p).exists()]
    if _missing:
        raise RuntimeError(
            "NB03 integrity check failed — required output files are missing: "
            + ", ".join(str(p) for p in _missing)
        )

    print("═" * 72)
    print("NB03 COMPLETED")
    print("═" * 72)
    print("✓ All final integrity checks passed\n")
    print(f"  Experiment:  {EXPERIMENT_ID}")
    print(f"  Decision:    {dashboard['reported_decision']}")
    print(f"  Rationale:   {dashboard['rationale']}")
    print("\n  Results directory:")
    print(f"    {RESULTS_DIR}")
    print("  Figures directory:")
    print(f"    {FIGS_DIR}")
    print(f"  Provenance: {provenance['pipeline']['executed_at_utc']} (schema v{provenance['schema_version']})")
    print("\nNext step: NB04 — Phenotypic profiling")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notebook complete

    This notebook computed quality metrics (PR, PM, and task-qualified mAP) for the Live
    Cell Painting experiment, from-scratch and via `copairs`, across four
    applications:

    1. **Per-plate QC** — PC mAP: replicate consistency within each plate
    2. **Cross-plate batch** — PC mAP: replicate consistency across plates
    3. **Dose-response** — PC mAP within each treatment–dose; SC-21 for ordering
    4. **Treatment vs control** — activity mAP: phenotype detection vs negcon

    MoA-based phenotypic-distinctiveness mAP is not computed in NB03. These
    applications are combined into the SC-18 → SC-22 **Go/No-Go decision**.

    ### Key outputs
    - `results/quality_metrics/*.csv` — all metric tables
    - `figures/quality_metrics/*.png` — all visualizations
    - `results/quality_metrics/quality_metrics_report.md` — human-readable Go/No-Go report
    - `results/quality_metrics/provenance.json` — run provenance
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Save the analysis record

    Save the notebook's **current session** without rerunning cells:

    - **HTML** preserves code and the complete rich-output record;
    - **PDF** provides a clean reading copy without code inputs.

    Chromium can write both files directly to the experiment's `reports/`
    folder after authorization. Safari shows separate HTML and PDF download
    buttons because it cannot write directly to a selected folder.
    """)
    return


@app.cell
def _(EXPERIMENT_ID, mo):
    from hca_pipeline.report_export import SessionReportSaver

    _report_saver = SessionReportSaver(
        basename=f"{EXPERIMENT_ID}_03_quality_metrics",
        suggested_directory=f"workspace/analysis/{EXPERIMENT_ID}/reports",
    )
    mo.ui.anywidget(_report_saver)
    return


if __name__ == "__main__":
    app.run()
