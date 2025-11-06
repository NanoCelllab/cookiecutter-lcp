# 🔬 Scientific Project Scaffold — Reproducible Research Template

This repository provides a **structured and reproducible foundation** for your scientific project (e.g., MSc, PhD, or independent research).  
It serves as an **umbrella project**, grouping all related experiments, metadata, and analyses in a consistent layout.

---

## 💡 Why a single umbrella repository?

- Keeps your **entire research history** (metadata, pipelines, notebooks, analysis outputs) in one place.  
- Enables **comparison between experiments** and consistent reuse of analysis tools.  
- Supports **FAIR and reproducible research** practices — lightweight data and scripts are versioned; heavy data are archived elsewhere.  

> 📘 The repository name should be **general**, not experiment-specific.  
> Examples:  
> - `hca-nanotoxicology`  
> - `bioinfo-transcriptomics`  
> - `cp-senescence-multi-cell`  

If using multiple cell lines, join them with an underscore (e.g., `Huh7_Caco2`). Avoid spaces and special characters.

---

## 📂 Repository layout

Each experiment is stored as a folder named:  
`YYYY_MM_DD_CellLine_Perturbation_Time`  
(e.g., `2025_05_09_Huh7_NPPS_24h`).

```
<repo_root>/
├─ images/
│  └─ <experiment>/
│     ├─ images/         # raw microscopy images
│     └─ illum/          # optional illumination correction
├─ workspace/
│  ├─ metadata/<experiment>/     # barcodes, platemaps
│  ├─ load_data_csv/<experiment>/ # LoadData CSVs
│  ├─ pipelines/<experiment>/     # CellProfiler pipelines (.cppipe)
│  ├─ assaydev/<experiment>/outlines_qc/
│  ├─ analysis/<experiment>/analysis/
│  ├─ profiles/<experiment>/
│  ├─ backend/<experiment>/
│  ├─ models/
│  └─ cellpose/
└─ workspace_dl/
   └─ <experiment>/notebooks/
```

---

## ⚙️ About the Cookiecutter prompts

When you create the repository with `cookiecutter`, you will be asked for several fields.

### Guidelines
- **Required**: used for folder or repository naming (e.g., `modality`, `project_tag`).  
- **Optional**: domain-specific details that help describe your project (e.g., `tissue`, `drug`, `nanoparticle`).  
- **Defaults**: the value in parentheses will be used if you press **Enter**.  
- **Multiple cell lines**: use underscores (e.g., `Huh7_Caco2`).

---

## 🚀 Step 1 — Define your experiment name

Each experiment must have a **unique identifier**, stored in a variable called `EXP`.

```
YYYY_MM_DD_CellLine_Perturbation_Time
```

**Examples:**
- 2025_06_28_Huh7_NPPS_24h  
- 2025_06_28_HepG2_Doxo_48h  
- 2025_06_28_Huh7_Caco2_AgNP_72h

> 💡 Start with the **year (YYYY)** to keep folders sorted chronologically.  
> Include cell line, perturbation (NP, drug, or condition), and time point.

Define it in your terminal (edit the name only):

```bash
# Define your experiment
EXP=2025_06_28_Huh7_NPPS_24h
```

---

## 🚀 Step 2 — Create all folders for your experiment

Once `$EXP` is set, create the full structure automatically:

```bash
# Raw images and illumination correction
mkdir -p images/$EXP/images
mkdir -p images/$EXP/illum

# Metadata and LoadData
mkdir -p workspace/metadata/$EXP/platemap
mkdir -p workspace/load_data_csv/$EXP

# Pipelines (.cppipe)
mkdir -p workspace/pipelines/$EXP

# QC and analysis
mkdir -p workspace/assaydev/$EXP/outlines_qc
mkdir -p workspace/analysis/$EXP/analysis

# Backend and profiles
mkdir -p workspace/backend/$EXP
mkdir -p workspace/profiles/$EXP

# Deep learning (optional)
mkdir -p workspace_dl/$EXP/notebooks
```

Check the result with:
```bash
tree -L 3
```

---

## ✅ Best practices

- **Naming:** `YYYY_MM_DD_CellLine_Treatment_Condition`  
- **Version control:** commit only scripts, metadata, and light files.  
- **Heavy data:** store microscopy images and outputs in institutional storage (e.g., REDU/Unicamp).  
- **Reproducibility:** keep pipelines and metadata under version control.

---

## 🔗 Useful links

- [CellProfiler](https://cellprofiler.org)  
- [pycytominer](https://github.com/cytomining/pycytominer)  
- [Cell Painting Gallery](https://broadinstitute.github.io/cellpainting-gallery/data_structure.html)  
- [REDU (Unicamp)](https://redu.unicamp.br)
