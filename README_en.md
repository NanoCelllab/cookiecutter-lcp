# 🍪🔬 Live Cell Painting / HCA — Project Scaffold

Welcome! This repository is the **umbrella project** for your graduate research (MSc/PhD).  
You will keep **all experiments of your project here**, and each experiment lives in its own dated folder (e.g., `2025_05_09_Huh7_NPPS_24h`) mirrored across `images/`, `workspace/`, and `workspace_dl/`.

> 💡 **GitHub is for versioning code, metadata and light artifacts.**  
> ⚠️ **Do not push raw microscopy images or large binaries.** Store them in institutional storage (e.g., REDU/Unicamp) and keep only paths/metadata here.

---

## 🚀 Step 1 — Define your experiment name

Each experiment must have a **unique identifier** stored in a variable called `EXP`.  
It defines the folder names under `images/`, `workspace/`, and `workspace_dl/`.

Use the convention:
```
YYYY_MM_DD_CellLine_Perturbation_Time
```

Examples:
- `2025_06_28_Huh7_NPPS_24h`
- `2025_06_28_HepG2_Doxo_48h`
- `2025_06_28_Huh7_Caco2_AgNP_72h`

> 📘 Start with the **year** to keep folders sorted chronologically.  
> 🧠 Be descriptive but concise: include the **cell line**, **perturbation**, and **exposure time**.

Copy this snippet to define your experiment (edit the name only):

```bash
# Define your experiment
EXP=2025_06_28_Huh7_NPPS_24h
```

---

## 🚀 Step 2 — Create all folders for your experiment

After defining `$EXP`, copy and paste this block to create the full structure:

```bash
# Raw images and (optional) illumination correction
mkdir -p images/$EXP/images
mkdir -p images/$EXP/illum

# Metadata and LoadData
mkdir -p workspace/metadata/$EXP/platemap
mkdir -p workspace/load_data_csv/$EXP

# Pipelines (.cppipe)
mkdir -p workspace/pipelines/$EXP

# Assay development (QC) and analysis
mkdir -p workspace/assaydev/$EXP/outlines_qc
mkdir -p workspace/analysis/$EXP/analysis

# Backend outputs and profiles
mkdir -p workspace/backend/$EXP
mkdir -p workspace/profiles/$EXP

# Deep learning notebooks (optional)
mkdir -p workspace_dl/$EXP/notebooks
```

Check the result with:
```bash
tree -L 3
```

---

## ✅ Best practices

- **Naming**: `YYYY_MM_DD_CellLine_Treatment_Condition`  
- **Version control**: commit only scripts, metadata, and light files.  
- **Heavy data**: archive images in REDU.  
- **Reproducibility**: keep pipeline versions and metadata together.
