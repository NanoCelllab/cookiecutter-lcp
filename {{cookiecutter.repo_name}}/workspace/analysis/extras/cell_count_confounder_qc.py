import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Extra — Advanced cell-count confounder QC

    Use this notebook only when NB04 flags a plausible cell-count problem.
    Low cell count can be toxicity or proliferation biology, so **no well is
    removed and no feature is corrected automatically**.

    This notebook reads NB02's immutable checkpoints, records the evidence,
    and can create an explicitly activated derived checkpoint. After applying
    a decision, return to **NB03** (not directly to NB04) so every downstream
    metric is recomputed from the same data.
    """)
    return


@app.cell
def _():
    import json
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    return PCA, Path, datetime, json, np, pd, plt, sys, timezone


@app.cell
def _(Path, sys):
    _path = Path(__file__).resolve()
    REPO_ROOT = next(
        (p for p in (_path, *_path.parents) if (p / "pixi.toml").exists() or (p / ".git").exists()),
        None,
    )
    if REPO_ROOT is None:
        raise FileNotFoundError("Could not locate the project root from this notebook.")
    sys.path.insert(0, str(REPO_ROOT / "workspace"))

    from hca_pipeline.cell_count_qc import (
        classify_cell_count_evidence,
        control_based_residualize,
        filter_wells,
        pca_cell_count_associations,
    )
    from hca_pipeline.config import ExperimentConfig
    from hca_pipeline.feature_select import infer_feature_cols, select_features
    from hca_pipeline.io import file_sha256, write_active_profile_manifest, write_parquet_protected
    from hca_pipeline.metadata import CELL_COUNT_METADATA_COLUMN
    from hca_pipeline.normalize import (
        clean_features_before_normalization,
        drop_extreme_magnitude_features,
        drop_near_zero_mad_features,
        enforce_pascalcase_metadata_columns,
        normalize_per_plate_mad_robustize,
    )
    return (
        CELL_COUNT_METADATA_COLUMN,
        ExperimentConfig,
        REPO_ROOT,
        classify_cell_count_evidence,
        clean_features_before_normalization,
        control_based_residualize,
        drop_extreme_magnitude_features,
        drop_near_zero_mad_features,
        enforce_pascalcase_metadata_columns,
        filter_wells,
        file_sha256,
        infer_feature_cols,
        normalize_per_plate_mad_robustize,
        pca_cell_count_associations,
        select_features,
        write_active_profile_manifest,
        write_parquet_protected,
    )


@app.cell
def _(Path, mo):
    _inferred = Path(__file__).resolve().parents[1].name
    experiment_id = mo.ui.text(value=_inferred, label="Experiment ID")
    experiment_id
    return (experiment_id,)


@app.cell
def _(ExperimentConfig, REPO_ROOT, experiment_id):
    EXPERIMENT_ID = experiment_id.value.strip()
    CONFIG = ExperimentConfig.load(REPO_ROOT, EXPERIMENT_ID)
    PROFILE_DIR = REPO_ROOT / "workspace" / "profiles" / EXPERIMENT_ID
    ANALYSIS_DIR = REPO_ROOT / "workspace" / "analysis" / EXPERIMENT_ID
    CACHE_DIR = PROFILE_DIR / "cache"
    OUTPUT_DIR = PROFILE_DIR / "outputs"
    QC_RESULTS_DIR = ANALYSIS_DIR / "results" / "cell_count_qc"
    QC_FIGURES_DIR = ANALYSIS_DIR / "figures" / "cell_count_qc"
    for _directory in (QC_RESULTS_DIR, QC_FIGURES_DIR):
        _directory.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR, CACHE_DIR, CONFIG, EXPERIMENT_ID, OUTPUT_DIR, PROFILE_DIR, QC_FIGURES_DIR, QC_RESULTS_DIR


@app.cell
def _(CACHE_DIR, CELL_COUNT_METADATA_COLUMN, CONFIG, OUTPUT_DIR, infer_feature_cols, pd):
    CANONICAL_PER_WELL = OUTPUT_DIR / "per_well_features_selected.parquet"
    AGGREGATED_PER_WELL = CACHE_DIR / "per_well_aggregated.parquet"
    CANONICAL_SINGLE_CELL = CACHE_DIR / "single_cell_ready.parquet"
    _missing = [p for p in (CANONICAL_PER_WELL, AGGREGATED_PER_WELL, CANONICAL_SINGLE_CELL) if not p.exists()]
    if _missing:
        raise FileNotFoundError("Run NB02 first. Missing checkpoint(s):\n" + "\n".join(map(str, _missing)))

    df_profile = pd.read_parquet(CANONICAL_PER_WELL)
    CONFIG_RESOLVED = CONFIG.resolve_columns(df_profile)
    feature_cols = infer_feature_cols(df_profile)
    if CELL_COUNT_METADATA_COLUMN not in df_profile:
        raise ValueError(f"NB02 output lacks {CELL_COUNT_METADATA_COLUMN}; rerun NB01 and NB02.")
    print(f"✓ Loaded canonical NB02 profile: {len(df_profile):,} wells × {len(feature_cols):,} features")
    print(f"  This notebook has not changed the active pipeline checkpoint.")
    return AGGREGATED_PER_WELL, CANONICAL_PER_WELL, CANONICAL_SINGLE_CELL, CONFIG_RESOLVED, df_profile, feature_cols


@app.cell
def _(
    CELL_COUNT_METADATA_COLUMN,
    CONFIG_RESOLVED,
    classify_cell_count_evidence,
    df_profile,
    feature_cols,
    pca_cell_count_associations,
    pd,
):
    _control_mask = df_profile[CONFIG_RESOLVED.control_type_col].astype(str).isin(
        [str(v) for v in CONFIG_RESOLVED.negcon_values]
    )
    associations_all = pca_cell_count_associations(
        df_profile, feature_cols, CELL_COUNT_METADATA_COLUMN, subset="all_wells"
    )
    associations_controls = pca_cell_count_associations(
        df_profile.loc[_control_mask], feature_cols, CELL_COUNT_METADATA_COLUMN, subset="negative_controls"
    )
    associations = pd.concat([associations_all, associations_controls], ignore_index=True)
    _counts = pd.to_numeric(df_profile[CELL_COUNT_METADATA_COLUMN], errors="coerce")
    _exploratory_floor = max(1, int(_counts.quantile(0.10)))
    evidence = classify_cell_count_evidence(
        associations_all,
        associations_controls,
        low_count_fraction=float((_counts < _exploratory_floor).mean()),
    )
    print("── Evidence gate ──")
    print(f"  Classification: {evidence.classification}")
    print(f"  Max |r|, all wells: {evidence.max_abs_pc_correlation:.3f}")
    print(f"  Max |r|, controls : {evidence.max_abs_control_pc_correlation}")
    print(f"  Interpretation    : {evidence.message}")
    return associations, evidence


@app.cell
def _(CELL_COUNT_METADATA_COLUMN, CONFIG_RESOLVED, QC_FIGURES_DIR, associations, df_profile, np, pd, plt):
    _counts = pd.to_numeric(df_profile[CELL_COUNT_METADATA_COLUMN], errors="coerce")
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _axes[0].hist(_counts.dropna(), bins=min(30, max(5, int(np.sqrt(len(_counts))))), color="#1976d2")
    _axes[0].set(xlabel="Cells per well", ylabel="Wells", title="Cell-count distribution")
    for _subset, _part in associations.groupby("subset"):
        _axes[1].plot(_part["pc"], _part["pearson_r"].abs(), "o-", label=_subset)
    _axes[1].axhline(0.40, color="#d32f2f", linestyle="--", label="exploratory |r| gate")
    _axes[1].set(xlabel="Principal component", ylabel="|r| with log10(cell count)", title="PCA association")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _path = QC_FIGURES_DIR / "cell_count_evidence.png"
    _fig.savefig(_path, dpi=180, bbox_inches="tight")
    print(f"✓ Figure saved and displayed: {_path}")
    _fig
    return


@app.cell
def _(CELL_COUNT_METADATA_COLUMN, df_profile, mo, pd):
    _counts = pd.to_numeric(df_profile[CELL_COUNT_METADATA_COLUMN], errors="coerce")
    strategy = mo.ui.dropdown(
        options={
            "Keep canonical profile (recommended unless evidence is convincing)": "keep",
            "Exclude low-count wells, then re-normalize and re-select features": "exclude",
            "Control-based regression": "regress",
            "Exclude + re-normalize, then control-based regression": "hybrid",
        },
        value="keep",
        label="Reviewed decision",
    )
    minimum_cells = mo.ui.number(
        start=1, stop=max(2, int(_counts.max())), step=10,
        value=max(1, int(_counts.quantile(0.10))), label="Manual minimum cells per well",
    )
    confirm = mo.ui.checkbox(label="I reviewed the plots and understand that cell count can be biological")
    apply_decision = mo.ui.run_button(label="Create and activate reviewed checkpoint")
    mo.vstack([strategy, minimum_cells, confirm, apply_decision])
    return apply_decision, confirm, minimum_cells, strategy


@app.cell
def _(
    AGGREGATED_PER_WELL,
    CANONICAL_SINGLE_CELL,
    CELL_COUNT_METADATA_COLUMN,
    CONFIG_RESOLVED,
    PROFILE_DIR,
    QC_RESULTS_DIR,
    apply_decision,
    clean_features_before_normalization,
    confirm,
    control_based_residualize,
    datetime,
    drop_extreme_magnitude_features,
    drop_near_zero_mad_features,
    enforce_pascalcase_metadata_columns,
    evidence,
    file_sha256,
    feature_cols,
    filter_wells,
    infer_feature_cols,
    json,
    minimum_cells,
    normalize_per_plate_mad_robustize,
    pd,
    select_features,
    strategy,
    timezone,
    write_active_profile_manifest,
    write_parquet_protected,
    df_profile,
):
    if apply_decision.value:
        if not confirm.value:
            raise ValueError("Review the evidence and tick the confirmation before applying a decision.")
        _strategy = strategy.value
        _run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _derived_dir = PROFILE_DIR / "cell_count_qc" / _run_id
        _per_well_path = _derived_dir / "per_well_features_selected.parquet"
        _single_cell_path = _derived_dir / "single_cell_ready.parquet"
        _audit_path = QC_RESULTS_DIR / "well_decisions.csv"
        _slopes_path = QC_RESULTS_DIR / "regression_slopes.csv"

        if _strategy == "keep":
            _manifest = {
                "active": False,
                "source": "canonical_nb02",
                "decision": "keep",
                "decided_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _manifest_path = write_active_profile_manifest(PROFILE_DIR, _manifest)
            print(f"✓ Canonical NB02 checkpoint remains active: {_manifest_path}")
        else:
            if _strategy in {"regress", "hybrid"} and evidence.classification != "technical_confounding_supported":
                raise ValueError(
                    "Regression blocked: technical confounding was not supported inside negative controls. "
                    "Do not remove a potentially biological treatment effect."
                )

            _aggregated = pd.read_parquet(AGGREGATED_PER_WELL)
            if _strategy in {"exclude", "hybrid"}:
                _working, _audit = filter_wells(_aggregated, CELL_COUNT_METADATA_COLUMN, int(minimum_cells.value))
                if _working.empty:
                    raise ValueError("The chosen threshold excludes every well.")
                _audit.to_csv(_audit_path, index=False)
                _features = infer_feature_cols(_working)
                _working, _features, _ = clean_features_before_normalization(
                    _working, _features, max_missing_fraction=0.20
                )
                _working, _features, _ = drop_near_zero_mad_features(
                    _working, _features,
                    plate_col=CONFIG_RESOLVED.plate_col,
                    control_col=CONFIG_RESOLVED.control_type_col,
                    negcon_values=CONFIG_RESOLVED.negcon_values,
                    norm_control=CONFIG_RESOLVED.negcon_values[0], mad_epsilon=1e-8,
                )
                _working = normalize_per_plate_mad_robustize(
                    _working, "infer", plate_col=CONFIG_RESOLVED.plate_col,
                    control_col=CONFIG_RESOLVED.control_type_col,
                    negcon_values=CONFIG_RESOLVED.negcon_values,
                    norm_control=CONFIG_RESOLVED.negcon_values[0],
                )
                _working = enforce_pascalcase_metadata_columns(_working)
                _working, _, _ = drop_extreme_magnitude_features(
                    _working, infer_feature_cols(_working), max_abs_value=1e6
                )
                _derived = select_features(_working, "infer")
            else:
                _derived = df_profile.copy()
                _audit = pd.DataFrame()

            if _strategy in {"regress", "hybrid"}:
                _resolved = CONFIG_RESOLVED.resolve_columns(_derived)
                _control_mask = _derived[_resolved.control_type_col].astype(str).isin(
                    [str(v) for v in _resolved.negcon_values]
                )
                _derived, _slopes = control_based_residualize(
                    _derived, infer_feature_cols(_derived), CELL_COUNT_METADATA_COLUMN, _control_mask
                )
                _slopes.to_csv(_slopes_path, index=False)

            _single_cell = pd.read_parquet(CANONICAL_SINGLE_CELL)
            _keys = [CONFIG_RESOLVED.plate_col, CONFIG_RESOLVED.well_col]
            _kept = _derived[_keys].drop_duplicates()
            _single_cell = _single_cell.merge(_kept.assign(_keep=True), on=_keys, how="inner").drop(columns="_keep")
            _pw_status = write_parquet_protected(_derived, _per_well_path, overwrite=False)
            _sc_status = write_parquet_protected(_single_cell, _single_cell_path, overwrite=False)
            _decision = {
                "schema_version": 1,
                "strategy": _strategy,
                "minimum_cells": int(minimum_cells.value) if _strategy in {"exclude", "hybrid"} else None,
                "evidence_classification": evidence.classification,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_wells": int(len(_derived)),
                "n_single_cells": int(len(_single_cell)),
            }
            _decision_path = QC_RESULTS_DIR / "decision.json"
            _decision_path.write_text(json.dumps(_decision, indent=2) + "\n", encoding="utf-8")
            _manifest_path = write_active_profile_manifest(
                PROFILE_DIR,
                {
                    "active": True,
                    "source": "cell_count_qc",
                    "strategy": _strategy,
                    "per_well_path": str(_per_well_path.relative_to(PROFILE_DIR)),
                    "single_cell_path": str(_single_cell_path.relative_to(PROFILE_DIR)),
                    "per_well_sha256": file_sha256(_per_well_path),
                    "single_cell_sha256": file_sha256(_single_cell_path),
                    "decision_path": str(_decision_path),
                    "activated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"✓ Derived per-well checkpoint: {_pw_status} — {_per_well_path}")
            print(f"✓ Derived single-cell checkpoint: {_sc_status} — {_single_cell_path}")
            print(f"✓ Active checkpoint manifest: {_manifest_path}")
            print("NEXT: restart/rerun NB03, then continue with NB04 → NB05 → NB06.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Save analysis record

    Click once to save the currently displayed state as matching HTML and PDF
    files under `reports/`, without re-running any analysis cells.
    """)
    return


@app.cell
def _(mo):
    from hca_pipeline.report_export import make_live_report_capture

    report_capture = make_live_report_capture(mo)
    report_capture
    return (report_capture,)


@app.cell
def _(EXPERIMENT_ID, Path, REPO_ROOT, mo, report_capture):
    _capture = report_capture.value
    mo.stop(not _capture.get("capture_id") or not _capture.get("captured_html"))

    from hca_pipeline.report_export import save_live_report_snapshot

    _notebook_file = Path(__file__).resolve()
    _reports_dir = REPO_ROOT / "workspace" / "analysis" / EXPERIMENT_ID / "reports"
    with mo.status.spinner(title="Saving current session as HTML and PDF (no cell re-execution)"):
        _html_path, _pdf_path = save_live_report_snapshot(
            _capture["captured_html"],
            _reports_dir,
            _notebook_file.stem,
            title=f"{EXPERIMENT_ID} — {_notebook_file.stem}",
            captured_at=_capture["capture_id"],
        )
    mo.callout(
        mo.md(f"✓ Analysis record saved without re-running cells.\n\n- HTML: `{_html_path}`\n- PDF: `{_pdf_path}`"),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
