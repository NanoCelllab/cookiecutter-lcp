# 🍪🔪 Live Cell Painting Project

Welcome to your **Live Cell Painting (LCP)** project!  
This structure was automatically generated from [cookiecutter-lcp](https://github.com/NanoCelllab/cookiecutter-lcp) to ensure **organization, reproducibility, and best practices** across experiments.

---

## 📂 Folder Structure

```bash
<repo_name>/
├── .gitignore                     # ignores large files (images, outputs)
├── <cell_line>/                   # cell line used (e.g., Huh7)
│   └── <assay_slug>/              # assay (e.g., npps)
│       ├── <batch_tag>/           # data batch from Cytation
│       │   ├── images/            # raw image data organized by plate
│       │   │   └── <plate_id>/    # e.g., SQ00015167, PlateX...
│       │   └── illum/             # illumination correction files
│       │       └── <plate_id>/    
│       ├── workspace/             # analysis and processed outputs
│       │   ├── analysis/          # full feature extraction
│       │   │   └── <batch>/<plate>/
│       │   ├── assaydev/          # QC and segmentation testing
│       │   ├── backend/           # databases (.csv, .sqlite)
│       │   │   └── <batch>/<plate>/
│       │   ├── cellpose/          # Cellpose model outputs and logs
│       │   ├── load_data_csv/     # LoadData.csv files per batch/plate
│       │   │   └── <batch>/<plate>/
│       │   ├── metadata/          # platemaps, layouts, barcodes
│       │   │   └── <batch>/<plate>/
│       │   ├── models/            # trained models (Cellpose, etc.)
│       │   ├── pipelines/         # CellProfiler pipelines (.cppipe)
│       │   └── profiles/          # phenotypic profiles (pycytominer)
│       └── workspace_dl/          # deep learning analyses
└── README.md                      # this file
```

---

## 🧩 Batch and Plate Organization

Each **batch** represents a set of images exported from the Cytation microscope (or another imaging device).  
Within each batch, there may be one or multiple **plates**.

### Example:
```bash
Huh7/npps/
├── 20251109_batch1/
│   ├── images/PlateX/
│   └── illum/PlateX/
└── workspace/
    ├── analysis/20251109_batch1/PlateX/
    ├── backend/20251109_batch1/PlateX/
    ├── load_data_csv/20251109_batch1/PlateX/
    └── metadata/20251109_batch1/PlateX/
```

> 💡 **Tip:** The template can automatically create *demo batches*:
> `{{ cookiecutter.batch_tags | default(cookiecutter.batch_tag, true) }}`
> and *demo plates*:
> `{{ cookiecutter.plate_ids | default(cookiecutter.plate_id_example, true) }}`.
> You can rename or remove them once you have your real data.

---

## 🚀 How to Use

### 1️⃣ Initialize your repository

```bash
git init
git add .
git commit -m "Initial commit from LCP template"
git remote add origin <your-repo-URL>
git branch -M main
git push -u origin main
```

---

### 2️⃣ Add your raw images

Place your Cytation exports in:

```
<cell_line>/<assay_slug>/<batch_tag>/images/<plate_id>/
```

If illumination correction is used:

```
<cell_line>/<assay_slug>/<batch_tag>/illum/<plate_id>/
```

---

### 3️⃣ Set up metadata

Use **Load Data Generator** and **Layout Generator** to create `.csv` metadata files.  
Save them in:

```
<cell_line>/<assay_slug>/workspace/load_data_csv/<batch_tag>/<plate_id>/
<cell_line>/<assay_slug>/workspace/metadata/<batch_tag>/<plate_id>/
```

---

### 4️⃣ Run your analysis

Import or edit `.cppipe` pipelines in:

```
<cell_line>/<assay_slug>/workspace/pipelines/
```

- Adjust `assaydev.cppipe` parameters for quality control.  
- Run `analysis.cppipe` for full feature extraction.  
- Results will appear under `workspace/analysis/`.

---

### 5️⃣ Outputs

| Output Type | Location |
|--------------|-----------|
| **Single-cell features** | `workspace/analysis/<batch>/<plate>/` |
| **Aggregated profiles** | `workspace/profiles/` |
| **Databases (.sqlite/.csv)** | `workspace/backend/<batch>/<plate>/` |

---

## ⚠️ About `.gitkeep` Files

These files exist only to ensure **empty directories** are tracked by Git.  
- If the folder is empty → **keep** `.gitkeep`.  
- If it contains real files → you may **delete** it (optional).

---

## 📌 Best Practices

✅ **Naming conventions**  
Use `CellLine_Assay_Date` (e.g., `Huh7_NPPS_20250925`).  
Avoid spaces, accents, and capital letters.

✅ **Version control**  
Never upload raw images (`images/`, `illum/`) to GitHub.  
Only analysis files, metadata, and pipelines should be versioned.

✅ **Data storage**  
Use [REDU Unicamp](https://redu.unicamp.br/) or institutional repositories for heavy data.

✅ **Reproducibility**  
Keep pipelines, metadata, and notebooks versioned and updated.  
Document relevant changes via Git commits.

---

## 📚 Useful Resources

- [📘 CellProfiler Documentation](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.1/index.html)  
- [📊 pycytominer Documentation](https://github.com/cytomining/pycytominer)  
- [🍪 Cookiecutter LCP Template](https://github.com/NanoCelllab/cookiecutter-lcp)
