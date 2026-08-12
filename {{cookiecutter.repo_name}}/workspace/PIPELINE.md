# Live Cell Painting Analysis Pipeline

## Overview

This pipeline processes Live Cell Painting (Acridine Orange, GFP/PI ± Hoechst)
data from raw CellProfiler outputs through quality-controlled phenotypic
profiles and single-cell analysis. It follows the `cookiecutter-lcp`
directory structure for reproducibility.

The pipeline notebooks are **marimo notebooks** (plain `.py` files, not
`.ipynb`) — see [Marimo Notebooks](#marimo-notebooks) below for how that
changes the way you run and edit them. Superseded Jupyter versions are kept
under each experiment's `analysis/legacy_ipynb/` for reference, not deleted.

## Pipeline Flow

> **Deferred design:** a future dedicated cell-count confounder QC module,
> positioned between NB02 and the quality-metrics gate, is specified in
> [FUTURE_CELL_COUNT_CONFOUNDER_QC.md](FUTURE_CELL_COUNT_CONFOUNDER_QC.md).
> It is not implemented and the current pipeline does not exclude wells by
> cell count.

```
(NB00) → NB01 → NB02 → NB03 → NB04 → NB05 → NB06
          image     ↑
          QC gate    Go/No-Go gate
```

QC gates (NB00, NB03) have protocol-defined, objective pass/fail criteria and
are safe to run headlessly (`pixi run python3 <path>` — see
[Marimo Notebooks](#marimo-notebooks)). NB04–NB06 involve judgment calls
(interpreting a UMAP, choosing a modelling space, evaluating a cluster) and
are meant to be run interactively (`pixi run marimo edit <path>`), with a
human looking at every figure before moving on — don't treat a headless run
of those notebooks as a substitute for actually reviewing them.

| Step | Notebook | Input | Output | What it does |
|------|----------|-------|--------|-------------|
| 0 of 6 (optional) | `00_image_quality.py` | Raw images under `images/{EXPERIMENT_ID}/` | `results/image_quality/excluded_sites.csv`, per-image/per-well QC CSVs, figures | Per-image blur/saturation/SNR/focus metrics, human-tunable thresholds, flags debris/out-of-focus/over- or under-exposed sites for exclusion before aggregation. Self-skips if `image_root` isn't configured for this experiment. |
| 1 of 6 | `01_samples_retrieval.py` | Legacy single-cell CSV (or SQLite, per experiment); optional `excluded_sites.csv` (from NB00) | `single_cell_profiles.parquet` | Read/consolidate all discovered plates, apply NB00's image-QC exclusions if present, select the analysis plate scope, check required-reference coverage, structural sanity checks (SC-01→SC-05) |
| 2 of 6 | `02_aggregate_normalize_featureselect.py` | `single_cell_profiles.parquet` | `per_well_features_selected.parquet` | Filter to NB01's selected plate scope, annotate with platemap, aggregate to well-level, mad_robustize normalization, feature selection (SC-06→SC-10) |
| 3 of 6 | `03_quality_metrics.py` | `per_well_features_selected.parquet` | `results/quality_metrics/*.csv`, `figures/quality_metrics/*.png` | Percent Replicating, mAP, batch assessment, dose-response, Go/No-Go decision (SC-18→SC-22) — runs **before** biological interpretation, so profiling/fingerprints/single-cell analysis only happen on data that already cleared the replicate-signal bar |
| 4 of 6 | `04_phenotypic_profiling.py` | `per_well_features_selected.parquet` | LDA/PCA/UMAP figures, models, CSVs | PCA, UMAP, LDA with leave-one-plate-out CV, Cohen's d, KMeans clustering, uncorrected-vs-Harmony comparison (SC-11→SC-13) |
| 5 of 6 | `05_phenotypic_fingerprints.py` | `per_well_features_selected.parquet` | `results/phenotypic_fingerprints/*`, `figures/phenotypic_fingerprints/*` | Feature taxonomy, per-condition Cohen's d fingerprints, heatmaps/radar plots — dose-optional |
| 6 of 6 | `06_single_cell_analysis.py` | `single_cell_ready.parquet` (NB02 cache) | SHAP CSVs, classification report, single-cell figures | Single-cell PCA/UMAP, HDBSCAN + KMeans clustering, LightGBM + SHAP, Mahalanobis distance (SC-14→SC-17) |

**Optional, experiment-specific extension:** `07_recovery_axis_analysis.py`
computes a geometric "recovery axis" between two named reference states
(e.g. a dormancy baseline and a proliferative/recovered state). It is **not**
part of the required 01→06 sequence — it only applies to datasets built
around that specific two-reference-state design, and gracefully skips itself
if the two reference labels aren't configured. Do not treat it as a required
step for datasets without that biology.

## Go/No-Go Decision Point

**NB03 is the quality gate**, and it now runs right after aggregation —
*before* any biological interpretation — rather than after it. It produces a
Go/No-Go decision based on:

| Check | ID | Critical? | Pass condition |
|-------|----|-----------|----------------|
| Control validation | SC-19 | Yes | Negcon PR ≤ 0.10 AND poscon PR > 0.50 |
| PR fraction | PR-25% | Yes | ≥ 25% of treatments pass PR per plate |
| Batch effect | SC-20 | No | < 30% PR drop cross-plate (meaningful only) |
| Dose-response | SC-21 | No | Monotonic dose-response for multi-dose treatments (skipped if no dose axis) |
| Phenotype detectability | SC-22 | Yes | Poscon mAP > 0.5, all treatments show phenotype |

**NB06 reads NB03's Go/No-Go report** and warns if the experiment received NO-GO.

## Quality Metrics Reference

| Metric | Question | Range | Good signal | Reference |
|--------|----------|-------|-------------|-----------|
| PR (Percent Replicating) | What fraction of treatments replicate above null? | 0–1 | > 0.25 | Caicedo et al. 2020 [doi:10.1038/s41592-020-0851-3] |
| PM (Percent Matching) | What fraction of profiles match same treatment? | 0–1 | > 0.50 | Caicedo et al. 2020 |
| mAP (Mean Average Precision) | How well can we retrieve replicates from ranking? | 0–1 | > 0.50, FDR-significant | Kalinin et al. 2025 [doi:10.1038/s41467-025-60306-2] |

## Marimo Notebooks

Create the per-experiment workspace before opening a notebook:

```bash
pixi run new-experiment <EXPERIMENT_ID>
```

This safely copies every `.py` template into
`workspace/analysis/<EXPERIMENT_ID>/analysis/`, creates the associated working
directories, and refuses to overwrite an existing analysis. Place CellProfiler
outputs under `workspace/backend/<EXPERIMENT_ID>/` afterward.

Every notebook is a plain Python file (`app = marimo.App()`, one function per
cell), not a `.ipynb`. This matters for how you work with them:

- **Edit interactively:** `pixi run marimo edit workspace/analysis/<EXPERIMENT_ID>/analysis/01_samples_retrieval.py`
  opens the reactive editor in your browser. Widgets you see at the top of
  each notebook (experiment picker, thresholds, checkboxes) are `mo.ui`
  elements — change one and every cell that depends on it re-runs
  automatically.
- **Run headlessly / as a script:** `pixi run python3 workspace/analysis/<EXPERIMENT_ID>/analysis/01_samples_retrieval.py`
  executes every cell in dependency order with no browser attached. Every
  `mo.ui` widget falls back to its default `value`, so a plain script run is
  fully deterministic — this is what CI or a "just run the whole pipeline"
  workflow should use.
- **Validate a notebook's structure** (catches unresolved names, duplicate
  variable definitions, etc. — cheaper than a full run): `pixi run marimo check --strict <path>`.
- **No `globals()`-based "resume from wherever the kernel happens to be"
  tricks.** Marimo forbids redefining the same variable across cells, and a
  full script run always executes the whole graph — so notebooks use plain,
  explicit read-cache-or-compute cells for expensive steps (e.g. NB02's
  per-stage checkpoint parquets) instead.
- **One gotcha worth knowing if you edit these notebooks:** a variable name
  starting with `_` is treated by marimo as private to its own cell (the
  convention loop/scratch variables should use). If a name needs to be read
  by a *different* cell, it must not start with `_`, or `marimo check --fix`
  will silently drop it from that cell's exports.

### Exporting a PDF report

Use `pixi run export-report <path-to-notebook.py> -o <report.pdf>` — **not**
`marimo export pdf`. Both go through the same nbconvert-generated HTML, but
`marimo export pdf` hands it to Playwright with no page numbers and no fix
for nbconvert's `lab` template clipping (not wrapping) any source line wider
than the code box, which loses the tail end of longer lines silently.
`export-report` (`workspace/hca_pipeline/report_export.py`) drives
Playwright itself to add both: a page-number footer and a CSS override that
wraps long lines instead of clipping them. Pass `--no-code` to omit source
and keep only markdown/outputs, and `--title` to change the header text
(defaults to the notebook's filename).

Every pipeline notebook (NB00–NB07) also has an **"Export this notebook as a
PDF report" button** as its last cell, so you don't need the CLI at all
during a normal `marimo edit` session — click it and the PDF is written to
`analysis/{EXPERIMENT_ID}/reports/`. Clicking it re-runs the whole notebook
headlessly in a separate process to render it, so the report reflects
whatever is currently saved in `experiment_config.json`, not unsaved widget
changes you haven't let the notebook re-run with yet. (`demo_nanoparticle_showcase.py`
is the one exception — it's explicitly stateless and writes nothing to disk,
so it doesn't get this button.)

### New content style: explain-the-why + basic/advanced options

Starting with NB04's batch-correction section (Harmony vs. ComBat), **new**
algorithmic sections should follow the shape below. This is *not* a
retroactive rewrite — existing sections keep their current (leaner) style
until a deliberate later pass updates the whole pipeline to match; only new
work should use it from here on.

**1. Explain more, and say why — including the assumptions.**
Every markdown cell introducing a new algorithmic choice should cover:

- **What** it does, in one or two plain-language sentences.
- **Reason:** why this approach (or these alternatives) exist here
  specifically — the problem being solved, and the tradeoff each
  alternative makes. Cite the paper/benchmark behind a claim rather than
  just asserting it.
- **Assumptions:** state them explicitly (as bullets, if there's more than
  one) so a future reader can tell *when the assumption breaks* and the
  method stops being appropriate — not just that the method exists.

See `04_phenotypic_profiling.py`'s "Section 5" markdown cell for the
worked example: it explains Harmony vs. ComBat, each one's
**Assumption:**, and the assumption both share (plate is a valid batch
proxy; treatment and plate aren't confounded with each other).

**2. Split options into basic (visible) and advanced (accordion).**
Not every knob deserves equal visibility. A choice made *every run* (which
method, which spaces to compare) stays a plain, always-visible `mo.ui`
widget. A tuning knob that only matters in an edge case — one most runs
should leave at its default — goes inside `mo.accordion({...})`, collapsed
by default:

```python
@app.cell
def _(mo):
    advanced_a_input = mo.ui.checkbox(value=True, label="...")
    advanced_b_input = mo.ui.checkbox(value=True, label="...")
    mo.accordion(
        {
            "Advanced: <method> parameters (only used if '<method>' is selected above)": mo.vstack(
                [advanced_a_input, advanced_b_input]
            ),
        }
    )
    return advanced_a_input, advanced_b_input
```

Widgets inside the accordion work exactly as normal — `mo.accordion` only
changes how they're *displayed* (collapsed by default), not how marimo
wires their `.value` to other cells. Give the accordion label enough
context to say what it's for and when it matters (the "only used if X is
selected above" phrasing above is deliberate).

**Applied so far:** NB04 Section 5 (Harmony vs. ComBat batch correction;
ComBat's `mean_only`/`par_prior` are the accordion's advanced options).
Apply this to new sections/notebooks as they're written — retrofitting
existing sections is a separate, deliberate pass, not something to do
opportunistically while passing through for an unrelated change.

## Experiment Configuration

Dataset-agnostic settings (control-type vocabulary, resolved column names,
plate format, imaging channels, whether a dose or time axis is present) live
in one file per experiment:

```
workspace/metadata/{EXPERIMENT_ID}/experiment_config.json
```

NB01's config wizard creates/updates this file; every later notebook loads
it via `ExperimentConfig.load(repo_root, experiment_id)` and calls
`CONFIG.resolve_columns(df)` after loading its input data, so column names
are auto-detected against what's actually present rather than hardcoded.
Control vocabulary (`negcon_values`/`poscon_values`/`trt_values`) is a
**list** per role, not a single literal — different platemap conventions use
different labels for the same role.

NB01 also owns the **analysis plate scope**. Its plate selector writes
`included_plates`, `excluded_plate_reasons`, `required_reference_treatments`,
and `analysis_mode` to the experiment config. An empty `included_plates` list
remains backward-compatible and means “use every discovered plate.” NB01's
single-cell parquet deliberately retains every plate, preserving the raw
analytical asset; NB02 applies the scope before annotation and aggregation,
and invalidates an existing checkpoint whenever its plate set differs from
the selection. This makes it possible to inspect one or two early plates—or
temporarily exclude a plate with a missing control—without deleting data.

Use `analysis_mode: "preliminary"` for incomplete acquisitions. NB03 still
runs the objective QC checks, but labels its reported decision as preliminary
so an early GO/NO-GO result cannot be mistaken for the final experiment-level
decision. Switch to `"final"` and include all intended plates for the final
run.

`has_dose_axis`/`has_time_axis` are derived automatically: a notebook that
depends on a dose or time axis checks these flags and skips (with a clear
message) rather than assuming every dataset has that dimension.

`image_root` follows the same optional-field pattern: it's `None` unless set
via NB01's config wizard, in which case it should point at the experiment's
raw-image directory relative to the repo root (e.g.
`images/{EXPERIMENT_ID}/images`). `00_image_quality.py` self-skips with a
clear message when it's unset, and NB01 only applies an image-QC exclusion
list when `00_image_quality.py` has actually produced one.

## Shared Pipeline Library

`workspace/hca_pipeline/` is a Python package (not a single flat
file) imported by every notebook:

| Module | What it holds |
|--------|---------------|
| `config.py` | `ExperimentConfig`, plate-format table, `validate_configuration`, `find_column` |
| `io.py` | Repo-root discovery, protected (never-silently-overwrite) parquet/CSV/table writers, checkpoint resume helpers |
| `metadata.py` | Core metadata normalization, platemap reading/annotation |
| `feature_select.py` | Feature-column detection, pycytominer `feature_select` wrapper |
| `normalize.py` | Missingness filtering/imputation, per-plate `mad_robustize` normalization |
| `taxonomy.py` | Canonical CellProfiler feature-family/compartment/channel/biological-signature classifier (channels/prefixes are parameters, not hardcoded) |
| `stats.py` | Cohen's d, PERMANOVA/PERMDISP, fingerprint-matrix aggregation |
| `metrics_qc.py` | From-scratch and `copairs`-based PR/PM/mAP, the Go/No-Go dashboard |
| `modelling.py` | Reusable PCA/UMAP/LDA/KMeans "run one modelling space" runner, balanced sampling |
| `single_cell.py` | Mahalanobis within-well outlier QC, single-cell feature curation, HDBSCAN helpers, LightGBM+SHAP classifier |
| `plotting.py` | Generic heatmap/radar/scatter plotting helpers |
| `image_qc.py` | Per-image quality-control toolkit — scan/aggregate/threshold raw microscopy images, used by `00_image_quality.py` |
| `schema.py` | Parquet schema consistency checks (duplicate-column detection, cross-file comparison) — see "Cross-experiment schema consistency" below |
| `report_export.py` | Notebook → paginated PDF report rendering — see "Exporting a PDF report" above |

Import from the specific submodule you need
(`from hca_pipeline.config import ExperimentConfig`), or from the top-level
package for the most commonly used names.

## Cross-experiment schema consistency (DuckDB-readiness)

Every notebook writes Parquet checkpoints, and the eventual goal is to
compare/query many students' experiments together (e.g. with DuckDB's
`read_parquet('workspace/analysis/*/outputs/*.parquet')` over many
experiment folders at once). That only works if the files actually share a
schema — or at least don't contain columns that look like the *same* thing
under different names.

**Which checkpoint to compare on:** use `per_well_aggregated.parquet`
(`workspace/profiles/{EXPERIMENT_ID}/outputs/cache/`) for cross-student
comparison, not `per_well_features_selected.parquet`. Feature *selection*
(NB02) legitimately keeps a different subset of columns per experiment —
different channels, different CellProfiler modules run, different
variance/quality-filtering outcomes — so the selected-features file is
*expected* to have a different schema per student. The pre-selection
aggregated profile is the least likely to have already diverged.

**Two different checks, on purpose** (`hca_pipeline.schema`):

- `find_duplicate_like_columns` / `validate_checkpoint` — structural
  hygiene that should hold for *every* file, unconditionally: no column
  should look like a merge/duplicate artifact (`Metadata_Plate.1`,
  `Foo_x`/`Foo_y`, ...). This is exactly the bug `dedupe_meta` fixes (see
  below) — `validate_checkpoint` is the test that would have caught it
  before it reached a downstream notebook.
- `compare_schemas` — a wide column × file dtype comparison across several
  files at once, for judging whether *feature* differences between
  students are expected drift or an actual problem.

Run it from the CLI: `pixi run schema-check <file1.parquet> <file2.parquet> ...`
(add `--compare` for the wide comparison instead of per-file validation).

**Where the duplicate-column bug actually came from:** some CellProfiler
exports horizontally merge per-compartment (Nuclei/Cytoplasm/Cells) tables
without dropping each one's own copy of Plate/Well/Site/QCFlag — pandas
silently re-labels the repeats as `Metadata_Plate.1`, `Metadata_Plate.2`,
... on read to keep column names unique, and without cleanup they ride
along into every downstream file. `hca_pipeline.metadata.dedupe_meta` is
called right after NB01 reads a raw single-cell CSV, before anything is
saved, so this never reaches `single_cell_profiles.parquet` in the first
place — the fix belongs at the point of ingestion, not scattered across
every consumer. Since this is a property of the *raw data a student
provides* (not something this pipeline's own code introduces), expect it to
recur across different students' submissions in different amounts; run
`validate_checkpoint` on a new student's checkpoint before trusting it.

## Directory Structure

```
<repo_root>/
├── pixi.toml                          # environment definition
├── workspace/
│   ├── pipelines/
│   │   └── hca_pipeline/              # shared package (imported by all notebooks)
│   ├── metadata/{EXPERIMENT_ID}/
│   │   ├── barcode_platemap.csv
│   │   ├── platemap/
│   │   └── experiment_config.json     # dataset-agnostic config, created by NB01
│   ├── backend/{EXPERIMENT_ID}/       # legacy single-cell CSV / SQLite databases (NB01 input)
│   ├── images/{EXPERIMENT_ID}/        # raw microscopy images (NB00 input, if configured)
│   ├── analysis/{EXPERIMENT_ID}/
│   │   ├── analysis/                  # the marimo notebooks themselves
│   │   │   └── legacy_ipynb/          # superseded Jupyter versions, kept for reference
│   │   ├── outputs/                   # NB01 output (single_cell_profiles.parquet)
│   │   ├── results/
│   │   │   ├── image_quality/         # optional NB00
│   │   │   ├── cv_summary.csv         # NB02
│   │   │   ├── quality_metrics/       # NB03
│   │   │   ├── lda_*.csv              # NB04
│   │   │   ├── phenotypic_fingerprints/ # NB05
│   │   │   └── recovery_axis/         # optional NB07
│   │   ├── figures/
│   │   │   ├── image_quality/         # optional NB00
│   │   │   ├── sample_retrieval/      # NB01
│   │   │   ├── aggregation/           # NB02
│   │   │   ├── quality_metrics/       # NB03
│   │   │   ├── phenotypic_profiling/  # NB04
│   │   │   ├── phenotypic_fingerprints/ # NB05
│   │   │   ├── single_cell/           # NB06
│   │   │   └── recovery_axis/         # optional NB07
│   │   └── reports/                   # `pixi run export-report` PDFs, one per notebook
│   ├── profiles/{EXPERIMENT_ID}/
│   │   ├── outputs/
│   │   │   ├── per_well_features_selected.parquet  # NB02 → NB03/NB04/NB05
│   │   │   └── cache/                 # NB02's per-stage checkpoints
│   │   │       └── single_cell_ready.parquet       # NB02 → NB06/NB07
│   └── models/{EXPERIMENT_ID}/outputs/ # NB04 models (.pkl)
```

## Control Type Vocabulary

| Label | Meaning | Example |
|-------|---------|---------|
| `con` | Untreated negative control | Non-treated cells |
| `negcon` | Vehicle/matched negative control | Lactose (sugar vehicle) |
| `poscon` | Positive control (strong phenotype) | Amiodarone |
| `trt` | Treatment | NPPS, Bafilomicin, etc. |

Both `con` and `negcon` are treated as negative controls for quality metrics.
These are **defaults**, not fixed literals — each experiment's actual
vocabulary (which labels mean negcon/poscon/trt for *this* platemap) lives in
`experiment_config.json` and is editable via NB01's config wizard.

## Training Curriculum

Before running the pipeline, complete the training sessions:

| Session | Topic | Tier |
|---------|-------|------|
| 01 | Python environment setup | Tier 0 |
| 02 | Pandas basics | Tier 1a |
| 03 | Metadata and platemap handling | Tier 1b |
| 04 | Feature inspection | Tier 1c |
| 05 | Normalization | Tier 2a |
| 06 | Aggregation | Tier 2b |
| 07 | PCA and visualization | Tier 3a |
| 08 | LDA and clustering | Tier 3b/4a |
| 09 | SHAP and Mahalanobis distance | Tier 4b/5 |
| 10 | Quality metrics concepts (PR, mAP, copairs) | Tier 5 |

Sessions 01–09 use relative paths (`Path("data")`). The pipeline uses
`find_repo_root()` for absolute paths. Session 09 includes a bridge note
explaining the transition.

## Environment

All dependencies are pinned in `pixi.toml`:

```toml
[dependencies]
python = ">=3.10,<3.13"
ipykernel = ">=6.0"
pandas = ">=2.0"
numpy = ">=1.24"
matplotlib = ">=3.7"
seaborn = ">=0.12"
scikit-learn = ">=1.3"
pycytominer = ">=0.3"
umap-learn = ">=0.5"
joblib = ">=1.3"
lightgbm = ">=4.0"
shap = ">=0.44"
scipy = ">=1.11"
ipywidgets = ">=8.0"
jupyterlab = ">=4.0"
nbformat = ">=5.9"
pillow = ">=10.0"
tifffile = ">=2026.7.14,<2027"
harmonypy = ">=2.0.0,<3"
hdbscan = ">=0.8.44,<0.9"
marimo = ">=0.23"

[pypi-dependencies]
copairs = ">=0.5"
```

`jupyterlab`/`ipykernel`/`nbformat` remain because the `legacy_ipynb/`
reference notebooks are still openable if you want to compare against the
pre-marimo version of a given step; they aren't required to run the current
pipeline.

Install with: `pixi install`, then install the browser runtime used for PDF
reports with `pixi run install-browser`, and verify imports with
`pixi run check`. Edit or run a notebook with
`pixi run marimo edit <path>` / `pixi run python3 <path>` (see
[Marimo Notebooks](#marimo-notebooks) above).
