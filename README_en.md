# 🔬 Scientific Project Scaffold — Reproducible Research Template (NanoCell)

This repository provides a **structured and reproducible foundation** for scientific projects (MSc, PhD, or independent research).
It is designed as an **umbrella repository**, grouping experiments, metadata, and analyses in a consistent layout.

> 🔬 This template reflects the organizational, analytical, and reproducibility practices of the NanoCell Interactions Lab (UNICAMP).

---

## 💡 Why a single umbrella repository?

- Keeps your **entire research history** (metadata, pipelines, notebooks, analyses) in one place.
- Enables **comparison between experiments** and consistent reuse of tools.
- Supports **FAIR and reproducible research** — lightweight data and code are versioned; heavy data are stored externally.

> 📘 The repository name should be **general**, not experiment-specific.

Examples:
- `hca-nanotoxicology`
- `bioinfo-transcriptomics`
- `cp-senescence-multi-cell`

If using multiple cell lines, use underscores (e.g., `Huh7_Caco2`). Avoid spaces and special characters.

---

## 📂 Repository layout

Each experiment is stored in a folder named:

`YYYY_MM_DD_CellLine_Perturbation_Time`

Example:
`2025_05_09_Huh7_NPPS_24h`

```
<repo_root>/
├─ images/
│  └─ <experiment>/
│     ├─ images/         # raw microscopy images
│     └─ illum/          # illumination correction (optional)
├─ workspace/
│  ├─ metadata/
│  │  ├─ templates/               # generic barcode_platemap/platemap starting point
│  │  └─ <experiment>/            # copied from templates/, filled in
│  ├─ load_data_csv/<experiment>/
│  ├─ pipelines/
│  │  ├─ templates/               # generic assaydev/analysis .cppipe starting point
│  │  └─ <experiment>/            # this experiment's adapted .cppipe files
│  ├─ hca_pipeline/                # shared, assay-agnostic Python package (vendored, not per-experiment)
│  ├─ assaydev/<experiment>/outlines_qc/
│  ├─ segmentation/
│  │  └─ cellpose/
│  │     ├─ model/                # pretrained, consumable-as-is segmentation models
│  │     ├─ training/             # training data / reference
│  │     └─ objects/<experiment>/ # per-experiment segmentation outputs
│  ├─ analysis/
│  │  ├─ templates/               # generic notebook starting point
│  │  └─ <experiment>/analysis/   # this experiment's notebook copies + results/figures
│  ├─ profiles/<experiment>/
│  ├─ backend/<experiment>/
│  └─ models/<experiment>/        # trained ML analysis models (PCA/UMAP/LDA/etc.)
└─ workspace_dl/
   └─ <experiment>/notebooks/
```

> The `templates/` subfolders are permanent library assets, not one-off
> examples — copy from them into a new `<experiment>/` folder every time you
> start a new experiment, then adapt as needed. `hca_pipeline/` and
> `segmentation/cellpose/model|training/` hold things that are used as-is
> (shared code, pretrained models) and are never per-experiment.

---

## 🧪 Computational environment (NanoCell standard)

This template uses **Pixi** to manage a reproducible computational environment for HCI/HCA data analysis.

The standard NanoCell environment, `hca-analysis`, is designed for:

- notebooks (JupyterLab)
- tabular profile analysis
- quality control (QC)
- pycytominer workflows
- copairs / mAP analysis
- classical machine learning
- visualization and statistics

Core files:

```
pixi.toml   # environment recipe
pixi.lock   # exact resolved versions
.pixi/      # local environment (not versioned)
```

### Usage

Install environment:

```bash
pixi install
```

Start JupyterLab:

```bash
pixi run lab
```

Check environment:

```bash
pixi run check
```

### Best practices

- **Never commit `.pixi/`**
- **Always version `pixi.toml` and `pixi.lock`**
- **Do not update the main environment casually**

> ⚠️ Always test updates in a separate branch before modifying the main environment.

---

## ⚙️ Cookiecutter prompts

When creating the project:

- **Required fields:** used for naming
- **Optional fields:** help describe the project
- **Defaults:** used if you press Enter
- **Multiple cell lines:** use underscores (`Huh7_Caco2`)

---

## 🚀 Step 1 — Define experiment

Each experiment must have a unique identifier:

```
YYYY_MM_DD_CellLine_Perturbation_Time
```

Example:

```bash
EXP=2025_06_28_Huh7_NPPS_24h
```

---

## 🚀 Step 2 — Create structure

```bash
mkdir -p images/$EXP/images
mkdir -p images/$EXP/illum

mkdir -p workspace/metadata/$EXP/platemap
cp workspace/metadata/templates/* workspace/metadata/$EXP/

mkdir -p workspace/load_data_csv/$EXP

mkdir -p workspace/pipelines/$EXP
cp workspace/pipelines/templates/* workspace/pipelines/$EXP/

mkdir -p workspace/assaydev/$EXP/outlines_qc
mkdir -p workspace/segmentation/cellpose/objects/$EXP

mkdir -p workspace/analysis/$EXP/analysis
cp workspace/analysis/templates/* workspace/analysis/$EXP/analysis/

mkdir -p workspace/backend/$EXP
mkdir -p workspace/profiles/$EXP
mkdir -p workspace/models/$EXP

mkdir -p workspace_dl/$EXP/notebooks
```

`workspace/hca_pipeline/` and `workspace/segmentation/cellpose/{model,training}/`
are not created per-experiment — they already exist once at the repo root and
are shared/imported, not copied.

---

## ✅ Best practices

- **Naming:** `YYYY_MM_DD_CellLine_Treatment_Condition`
- **Version control:** track scripts, metadata, and environment
- **Heavy data:** store externally (e.g., REDU)
- **Reproducibility:** version pipelines, notebooks, and environment

---

## 🔗 Useful links

- CellProfiler: https://cellprofiler.org
- pycytominer: https://github.com/cytomining/pycytominer
- Cell Painting Gallery: https://broadinstitute.github.io/cellpainting-gallery/data_structure.html
- REDU (Unicamp): https://redu.unicamp.br

