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
    # 00 — Image Quality Gate

    **Pipeline step:** 0 of 6 (optional)
    **Position in pipeline:** raw images → **NB00 (image quality)** → NB01 (samples retrieval)

    **Input:** raw microscopy images under `CONFIG.image_root`
    **Output:** `results/image_quality/excluded_sites.csv` (consumed by NB01),
    per-image/per-well QC tables, threshold-diagnostic and flagged-vs-passed
    thumbnail figures, `provenance.json`

    ---

    ## Overview

    This notebook scans raw images for blur (`focus_score`, a Laplacian-variance
    proxy for `PowerLogLogSlope`), over/under-exposure (`saturated_fraction`,
    `zero_fraction`), and signal quality (`snr`), using
    `hca_pipeline.image_qc` (pixel-based metrics — no CellProfiler
    `MeasureImageQuality` output exists for this dataset to reuse instead).
    A site with any channel image failing a threshold is flagged for
    exclusion **before** it reaches NB02's well-level aggregation, so a
    handful of debris/out-of-focus/mis-exposed fields can't skew a
    MAD-robustized well profile.

    **This is a "human calibrates once, then autonomous" gate, not a fully
    autonomous one.** The thresholds below default to the 1st/99th
    percentile of *this run's own* metric distributions — a reasonable
    starting point, not a validated cutoff. Run this notebook interactively
    (`marimo edit`) at least once per experiment, look at the histograms and
    the flagged/passed thumbnail grid, and adjust the sliders before trusting
    the exclusion list. Once calibrated, re-running headlessly
    (`pixi run python3`) reuses whatever threshold values were last saved
    into this cell's widget defaults.

    **Self-skips** if `CONFIG.image_root` isn't set (see NB01's config
    wizard) — datasets with no raw images available proceed straight to
    NB01 unaffected.
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
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Sequence

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 80)
    return (
        Path,
        Sequence,
        datetime,
        json,
        np,
        pd,
        platform,
        plt,
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
    from hca_pipeline.io import write_csv_protected, write_summary_table_protected
    from hca_pipeline.image_qc import (
        DEFAULT_CHANNEL_KEYWORDS,
        aggregate_per_well_image_qc,
        read_image,
        robust_rescale_image,
        scan_images,
    )

    print(f"  ✓  Shared utilities loaded from hca_pipeline ({_pipelines_dir})")
    return (
        DEFAULT_CHANNEL_KEYWORDS,
        ExperimentConfig,
        REPO_ROOT,
        aggregate_per_well_image_qc,
        read_image,
        robust_rescale_image,
        scan_images,
        write_csv_protected,
        write_summary_table_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Reads the `image_root` set via NB01's config wizard. This notebook
    doesn't have its own experiment-design widgets — it only consumes
    `ExperimentConfig`, it doesn't write it.
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
    EXPERIMENT_ID = experiment_id_input.value
    CONFIG = ExperimentConfig.load(REPO_ROOT, EXPERIMENT_ID)
    print(f"  Experiment ID:  {EXPERIMENT_ID}")
    print(f"  Image root:     {CONFIG.image_root or '(not configured — this notebook will self-skip)'}")
    return CONFIG, EXPERIMENT_ID


@app.cell
def _(CONFIG, EXPERIMENT_ID, REPO_ROOT):
    ANALYSIS_DIR = REPO_ROOT / "workspace" / "analysis" / EXPERIMENT_ID
    RESULTS_DIR = ANALYSIS_DIR / "results" / "image_quality"
    FIGS_DIR = ANALYSIS_DIR / "figures" / "image_quality"

    IMAGE_QC_ENABLED = bool(CONFIG.image_root)
    IMAGE_ROOT_PATH = (REPO_ROOT / CONFIG.image_root) if IMAGE_QC_ENABLED else None

    if IMAGE_QC_ENABLED:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Image QC enabled — scanning: {IMAGE_ROOT_PATH}")
    else:
        print(
            "ℹ️  NB00 SKIPPED — CONFIG.image_root is not set for this experiment.\n"
            "   Set it via NB01's config wizard if raw images are available."
        )
    return ANALYSIS_DIR, FIGS_DIR, IMAGE_QC_ENABLED, IMAGE_ROOT_PATH, RESULTS_DIR


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Scan raw images

    Every supported image under `image_root` is read once and reduced to
    per-image intensity/background/noise/focus/saturation metrics
    (`hca_pipeline.image_qc.calculate_image_metrics`). Plate/Well/Site/Channel
    identifiers are inferred from the folder and file names.
    """)
    return


@app.cell
def _(
    DEFAULT_CHANNEL_KEYWORDS,
    IMAGE_QC_ENABLED,
    IMAGE_ROOT_PATH,
    pd,
    scan_images,
):
    if not IMAGE_QC_ENABLED:
        image_qc_df = pd.DataFrame()
        failed_images = []
    else:
        image_qc_df, failed_images = scan_images(
            IMAGE_ROOT_PATH, channel_keywords=DEFAULT_CHANNEL_KEYWORDS
        )
        print(f"✓ Scanned {len(image_qc_df):,} image(s)")
        if failed_images:
            print(f"⚠️  {len(failed_images):,} image(s) failed to read/measure:")
            for _failure in failed_images[:10]:
                print(f"    - {_failure['filepath']}: {_failure['error']}")
            if len(failed_images) > 10:
                print(f"    ... and {len(failed_images) - 10} more")
    return failed_images, image_qc_df


@app.cell
def _(IMAGE_QC_ENABLED, aggregate_per_well_image_qc, image_qc_df, pd):
    if not IMAGE_QC_ENABLED or image_qc_df.empty:
        image_qc_per_well = pd.DataFrame()
    else:
        image_qc_per_well = aggregate_per_well_image_qc(image_qc_df)
        print(
            f"✓ Aggregated to {len(image_qc_per_well):,} plate/well/channel combination(s)"
        )
    return (image_qc_per_well,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Calibrate exclusion thresholds

    Defaults are the 1st/99th percentile of *this run's* own metric
    distributions — look at the histograms and the thumbnail grid in the
    sections below before trusting them, and adjust here if the default
    cutoff is clearly catching (or missing) the wrong images.
    """)
    return


@app.cell
def _(IMAGE_QC_ENABLED, image_qc_df, mo, pd):
    def _percentile_default(column, q, fallback):
        if not IMAGE_QC_ENABLED or image_qc_df.empty or column not in image_qc_df:
            return fallback
        value = image_qc_df[column].quantile(q)
        return float(value) if pd.notna(value) else fallback

    min_focus_score_input = mo.ui.number(
        value=round(_percentile_default("focus_score", 0.01, 0.0), 4),
        label="Minimum focus score (Laplacian variance) — below this is flagged out-of-focus",
    )
    max_saturated_fraction_input = mo.ui.number(
        value=round(_percentile_default("saturated_fraction", 0.99, 1.0), 4),
        start=0.0,
        stop=1.0,
        step=0.001,
        label="Maximum saturated-pixel fraction — above this is flagged overexposed",
    )
    max_zero_fraction_input = mo.ui.number(
        value=round(_percentile_default("zero_fraction", 0.99, 1.0), 4),
        start=0.0,
        stop=1.0,
        step=0.001,
        label="Maximum zero-pixel fraction — above this is flagged underexposed/black frame",
    )
    min_snr_input = mo.ui.number(
        value=round(_percentile_default("snr", 0.01, -1e9), 4),
        label="Minimum SNR — below this is flagged low-signal/debris",
    )
    mo.vstack(
        [
            min_focus_score_input,
            max_saturated_fraction_input,
            max_zero_fraction_input,
            min_snr_input,
        ]
    )
    return (
        max_saturated_fraction_input,
        max_zero_fraction_input,
        min_focus_score_input,
        min_snr_input,
    )


@app.cell
def _(
    IMAGE_QC_ENABLED,
    image_qc_df,
    max_saturated_fraction_input,
    max_zero_fraction_input,
    min_focus_score_input,
    min_snr_input,
    pd,
):
    if not IMAGE_QC_ENABLED or image_qc_df.empty:
        image_qc_df_flagged = pd.DataFrame()
        excluded_sites_df = pd.DataFrame(
            columns=["Metadata_Plate", "Metadata_Well", "Metadata_Site", "reason"]
        )
    else:
        _min_focus = float(min_focus_score_input.value)
        _max_saturated = float(max_saturated_fraction_input.value)
        _max_zero = float(max_zero_fraction_input.value)
        _min_snr = float(min_snr_input.value)

        image_qc_df_flagged = image_qc_df.copy()
        _fails_focus = image_qc_df_flagged["focus_score"] < _min_focus
        _fails_saturation = image_qc_df_flagged["saturated_fraction"] > _max_saturated
        _fails_zero = image_qc_df_flagged["zero_fraction"] > _max_zero
        _fails_snr = image_qc_df_flagged["snr"] < _min_snr
        image_qc_df_flagged["qc_pass"] = ~(
            _fails_focus | _fails_saturation | _fails_zero | _fails_snr
        )

        def _reason(row_fails_focus, row_fails_sat, row_fails_zero, row_fails_snr):
            reasons = []
            if row_fails_focus:
                reasons.append("out_of_focus")
            if row_fails_sat:
                reasons.append("overexposed")
            if row_fails_zero:
                reasons.append("underexposed")
            if row_fails_snr:
                reasons.append("low_snr")
            return ",".join(reasons)

        image_qc_df_flagged["fail_reason"] = [
            _reason(a, b, c, d)
            for a, b, c, d in zip(_fails_focus, _fails_saturation, _fails_zero, _fails_snr)
        ]

        _failed_images_df = image_qc_df_flagged.loc[~image_qc_df_flagged["qc_pass"]]
        # A site is excluded if ANY channel image captured for it fails --
        # single-cell tables don't carry a Channel column, so exclusion can
        # only be applied at Plate/Well/Site granularity, not per-channel.
        excluded_sites_df = (
            _failed_images_df.groupby(
                ["Metadata_Plate", "Metadata_Well", "Metadata_Site"], dropna=False
            )["fail_reason"]
            .apply(lambda reasons: ",".join(sorted(set(",".join(reasons).split(",")))))
            .reset_index()
            .rename(columns={"fail_reason": "reason"})
        )

        print(
            f"✓ {len(_failed_images_df):,} of {len(image_qc_df_flagged):,} image(s) failed a threshold "
            f"→ {len(excluded_sites_df):,} plate/well/site combination(s) flagged for exclusion"
        )
    return excluded_sites_df, image_qc_df_flagged


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Threshold diagnostics
    """)
    return


@app.cell
def _(FIGS_DIR, IMAGE_QC_ENABLED, image_qc_df_flagged, plt):
    if not IMAGE_QC_ENABLED or image_qc_df_flagged.empty:
        print("ℹ️  Skipping diagnostic histograms (image QC disabled or no images scanned).")
        _display = None
    else:
        _metrics_and_thresholds = [
            ("focus_score", "min_focus_score_input"),
            ("saturated_fraction", "max_saturated_fraction_input"),
            ("zero_fraction", "max_zero_fraction_input"),
            ("snr", "min_snr_input"),
        ]
        _fig, _axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for _ax, (_metric, _) in zip(_axes.ravel(), _metrics_and_thresholds):
            _values = image_qc_df_flagged[_metric].replace([float("inf"), float("-inf")], float("nan")).dropna()
            _ax.hist(_values, bins=60, color="#4C72B0", alpha=0.85)
            _n_flagged = int((~image_qc_df_flagged["qc_pass"]).sum())
            _ax.set_title(f"{_metric}  ({_n_flagged:,} image(s) flagged overall)")
            _ax.set_xlabel(_metric)
            _ax.set_ylabel("Image count")
        _fig.suptitle("Image-QC metric distributions", fontsize=13, fontweight="bold")
        _png_path = FIGS_DIR / "metric_distributions.png"
        _fig.savefig(_png_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(_fig)
        print(f"✓ Saved: {_png_path}")
        _display = _fig
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — Flagged vs. passed thumbnails (human sanity check)

    A visual spot check matters more than the metric thresholds themselves —
    a blur/brightness number chosen without looking at actual images is a
    guess. This grid samples a few flagged and a few passed images so you can
    confirm the thresholds are catching what they're supposed to.
    """)
    return


@app.cell
def _(
    FIGS_DIR,
    IMAGE_QC_ENABLED,
    Path,
    image_qc_df_flagged,
    plt,
    read_image,
    robust_rescale_image,
):
    if not IMAGE_QC_ENABLED or image_qc_df_flagged.empty:
        print("ℹ️  Skipping thumbnail grid (image QC disabled or no images scanned).")
        _display = None
    else:
        _n_sample = 6
        _flagged_sample = image_qc_df_flagged.loc[~image_qc_df_flagged["qc_pass"]].sample(
            n=min(_n_sample, int((~image_qc_df_flagged["qc_pass"]).sum())), random_state=0
        ) if (~image_qc_df_flagged["qc_pass"]).any() else image_qc_df_flagged.iloc[0:0]
        _passed_sample = image_qc_df_flagged.loc[image_qc_df_flagged["qc_pass"]].sample(
            n=min(_n_sample, int(image_qc_df_flagged["qc_pass"].sum())), random_state=0
        ) if image_qc_df_flagged["qc_pass"].any() else image_qc_df_flagged.iloc[0:0]

        _n_cols = max(len(_flagged_sample), len(_passed_sample), 1)
        _fig, _axes = plt.subplots(2, _n_cols, figsize=(2.4 * _n_cols, 5.2), squeeze=False)
        for _row_idx, (_label, _sample) in enumerate([("Flagged", _flagged_sample), ("Passed", _passed_sample)]):
            for _col_idx in range(_n_cols):
                _ax = _axes[_row_idx][_col_idx]
                _ax.axis("off")
                if _col_idx >= len(_sample):
                    continue
                _row = _sample.iloc[_col_idx]
                _image = read_image(Path(_row["filepath"]))
                _ax.imshow(robust_rescale_image(_image), cmap="gray")
                _ax.set_title(
                    f"{_label}\n{_row['Metadata_Plate']}/{_row['Metadata_Well']} site {_row['Metadata_Site']}\n"
                    f"{_row['Metadata_Channel']}",
                    fontsize=7,
                )
        _fig.suptitle("Sample of flagged vs. passed images", fontsize=12, fontweight="bold")
        _png_path = FIGS_DIR / "flagged_vs_passed_thumbnails.png"
        _fig.savefig(_png_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(_fig)
        print(f"✓ Saved: {_png_path}")
        _display = _fig
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5b — Browse flagged images one at a time

    Pick any single flagged image below to inspect it full-size alongside
    its metrics and the reason it failed — useful for spot-checking a
    specific image rather than only the small random sample above.
    """)
    return


@app.cell
def _(IMAGE_QC_ENABLED, image_qc_df_flagged, mo):
    if not IMAGE_QC_ENABLED or image_qc_df_flagged.empty or not (~image_qc_df_flagged["qc_pass"]).any():
        flagged_images_indexed = image_qc_df_flagged.iloc[0:0]
        flagged_image_picker = mo.ui.dropdown(
            options=["(no flagged images)"], value="(no flagged images)", label="Flagged image"
        )
    else:
        flagged_images_indexed = image_qc_df_flagged.loc[~image_qc_df_flagged["qc_pass"]].reset_index(drop=True)
        _options = {
            f"{row.Metadata_Plate}/{row.Metadata_Well} site {row.Metadata_Site} "
            f"({row.Metadata_Channel}) — {row.fail_reason}": i
            for i, row in flagged_images_indexed.iterrows()
        }
        flagged_image_picker = mo.ui.dropdown(
            options=_options,
            value=next(iter(_options)),
            label=f"Flagged image ({len(flagged_images_indexed):,} total)",
        )
    flagged_image_picker
    return flagged_image_picker, flagged_images_indexed


@app.cell
def _(
    Path,
    flagged_image_picker,
    flagged_images_indexed,
    mo,
    plt,
    read_image,
    robust_rescale_image,
):
    if flagged_images_indexed.empty or not isinstance(flagged_image_picker.value, int):
        _display = mo.md("_No flagged images to display._")
    else:
        _row = flagged_images_indexed.iloc[flagged_image_picker.value]
        _image = read_image(Path(_row["filepath"]))
        _fig, _ax = plt.subplots(figsize=(6, 6))
        _ax.imshow(robust_rescale_image(_image), cmap="gray")
        _ax.axis("off")
        _ax.set_title(
            f"{_row['Metadata_Plate']}/{_row['Metadata_Well']} site {_row['Metadata_Site']} "
            f"({_row['Metadata_Channel']})",
            fontsize=10,
        )
        _metrics_md = mo.md(
            f"""
**Fail reason:** {_row['fail_reason']}

**Focus score:** {_row['focus_score']:.4g}&nbsp;&nbsp;&nbsp;
**Saturated fraction:** {_row['saturated_fraction']:.4g}&nbsp;&nbsp;&nbsp;
**Zero fraction:** {_row['zero_fraction']:.4g}&nbsp;&nbsp;&nbsp;
**SNR:** {_row['snr']:.4g}

**File:** `{_row['filepath']}`
"""
        )
        plt.close(_fig)
        _display = mo.vstack([_fig, _metrics_md])
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Protected export
    """)
    return


@app.cell
def _(
    CONFIG,
    IMAGE_QC_ENABLED,
    RESULTS_DIR,
    excluded_sites_df,
    image_qc_df_flagged,
    image_qc_per_well,
    write_csv_protected,
    write_summary_table_protected,
):
    if not IMAGE_QC_ENABLED:
        print("ℹ️  Skipping export (image QC disabled for this experiment).")
        image_qc_metrics_csv = None
        image_qc_per_well_csv = None
        excluded_sites_csv = None
    else:
        image_qc_metrics_csv = RESULTS_DIR / "image_qc_metrics.csv"
        image_qc_per_well_csv = RESULTS_DIR / "image_qc_per_well.csv"
        excluded_sites_csv = RESULTS_DIR / "excluded_sites.csv"

        _status_metrics = write_csv_protected(
            image_qc_df_flagged, image_qc_metrics_csv, overwrite=CONFIG.overwrite_existing_outputs
        )
        print(f"✓ Per-image QC metrics {_status_metrics}: {image_qc_metrics_csv}")

        _status_per_well = write_summary_table_protected(
            image_qc_per_well, image_qc_per_well_csv, overwrite=CONFIG.overwrite_existing_outputs
        )
        print(f"✓ Per-well QC summary {_status_per_well}: {image_qc_per_well_csv}")

        _status_excluded = write_summary_table_protected(
            excluded_sites_df, excluded_sites_csv, overwrite=CONFIG.overwrite_existing_outputs
        )
        print(f"✓ Excluded-sites list {_status_excluded}: {excluded_sites_csv}")
    return excluded_sites_csv, image_qc_metrics_csv, image_qc_per_well_csv


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — Provenance
    """)
    return


@app.cell
def _(
    EXPERIMENT_ID,
    IMAGE_QC_ENABLED,
    RESULTS_DIR,
    REPO_ROOT,
    datetime,
    excluded_sites_csv,
    excluded_sites_df,
    image_qc_df_flagged,
    json,
    max_saturated_fraction_input,
    max_zero_fraction_input,
    min_focus_score_input,
    min_snr_input,
    platform,
    subprocess,
    timezone,
):
    def _run_git_command(args):
        try:
            result = subprocess.run(
                ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None if result.returncode == 0 else None

    _git_commit = _run_git_command(["rev-parse", "HEAD"])

    provenance = {
        "notebook": "00_image_quality.py",
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": (_git_commit or "unknown")[:8],
        "image_qc_enabled": IMAGE_QC_ENABLED,
        "n_images_scanned": int(len(image_qc_df_flagged)),
        "n_sites_excluded": int(len(excluded_sites_df)),
        "thresholds": {
            "min_focus_score": float(min_focus_score_input.value),
            "max_saturated_fraction": float(max_saturated_fraction_input.value),
            "max_zero_fraction": float(max_zero_fraction_input.value),
            "min_snr": float(min_snr_input.value),
        },
        "python_version": platform.python_version(),
    }

    if IMAGE_QC_ENABLED:
        _prov_path = RESULTS_DIR / "provenance.json"
        _prov_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
        print(f"✓ Provenance saved: {_prov_path}")
    else:
        print("ℹ️  Skipping provenance write (image QC disabled for this experiment).")

    print(f"  Notebook:    {provenance['notebook']}")
    print(f"  Experiment:  {provenance['experiment_id']}")
    print(f"  Git hash:    {provenance['git_hash']}")
    print(f"  Enabled:     {provenance['image_qc_enabled']}")
    print(f"  Excluded:    {provenance['n_sites_excluded']} site(s)")
    return (provenance,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — Final summary
    """)
    return


@app.cell
def _(EXPERIMENT_ID, IMAGE_QC_ENABLED, excluded_sites_csv, provenance):
    print("═" * 72)
    print("NB00 COMPLETED")
    print("═" * 72)
    print(f"  Experiment:  {EXPERIMENT_ID}")
    if IMAGE_QC_ENABLED:
        print(f"  Excluded sites list: {excluded_sites_csv}")
    else:
        print("  Image QC was disabled for this experiment — nothing to hand off to NB01.")
    print(f"  Provenance:  {provenance['timestamp']}")
    print("\nNext step: NB01 — Samples Retrieval")
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
