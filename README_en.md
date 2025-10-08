# 🍪🔬 Live Cell Painting — HCA Project

Welcome! This repository is your **main project** for *High Content Analysis (HCA)* using **Live Cell Painting (LCP)**.  
The idea is simple: you’ll have **one repository for your entire research project**, and **each experiment** becomes a separate folder named with the date, cell line, and assay (e.g., `2025_05_09_Huh7_NPPS_24h`), appearing **in parallel** under `images/`, `workspace/`, and `workspace_dl/`.

> 💡 **GitHub is for versioning code and text files** (tracking script and metadata changes).  
> ⚠️ **Do not upload raw microscopy images or large files!** Store them in [REDU (Unicamp)](https://redu.unicamp.br) or your institutional repository.

---

## 📂 Repository structure

```
<repo_root>/
├─ images/
│  └─ <experiment>/images/, illum/
├─ workspace/
│  ├─ metadata/<experiment>/barcode_platemap.csv, platemap/
│  ├─ pipelines/<experiment>/{assaydev.cppipe, analysis.cppipe, illum.cppipe}
│  ├─ assaydev/<experiment>/outlines_qc/
│  ├─ load_data_csv/<experiment>/
│  ├─ analysis/<experiment>/analysis/
│  ├─ profiles/<experiment>/
│  ├─ backend/<experiment>/
│  ├─ models/
│  └─ cellpose/
└─ workspace_dl/
   └─ <experiment>/notebooks/
```

---

## 🚀 Steps

1️⃣ **Create an experiment**
```bash
mkdir -p images/2025_05_09_Huh7_NPPS_24h/{images,illum}
mkdir -p workspace/{metadata,pipelines,assaydev,load_data_csv,analysis,profiles,backend}/2025_05_09_Huh7_NPPS_24h
mkdir -p workspace_dl/2025_05_09_Huh7_NPPS_24h/notebooks
```

2️⃣ **Add raw microscopy images**
- `images/<experiment>/images/`
- `images/<experiment>/illum/`

3️⃣ **Generate metadata**
- Use Load Data Generator & Layout Generator to create CSVs  
- Save in:
```
workspace/load_data_csv/<experiment>/
workspace/metadata/<experiment>/
```

4️⃣ **Prepare analysis**
- `assaydev.cppipe` → test & QC  
- `analysis.cppipe` → full feature extraction  
- `illum.cppipe` → optional illumination correction

5️⃣ **Results**
| Data type | Path |
|------------|------|
| Single-cell CSVs | workspace/analysis/<experiment>/analysis/ |
| Aggregated profiles | workspace/profiles/<experiment>/ |
| SQLite databases | workspace/backend/<experiment>/ |

---

## ✅ Best practices
- **Naming:** `YYYY_MM_DD_CellLine_Assay` (no spaces or accents).  
- **Version control:** commit only scripts, metadata, and light files.  
- **Data storage:** keep raw data in REDU.  
- **Reproducibility:** update metadata, notebooks, and pipelines.

---

## 🔗 Useful links
- [CellProfiler docs](https://cellprofiler.org/)
- [pycytominer](https://github.com/cytomining/pycytominer)
- [Cell Painting Gallery structure](https://broadinstitute.github.io/cellpainting-gallery/data_structure.html)
- [REDU (Unicamp)](https://redu.unicamp.br)

---

## 🧩 Quick checklist
- [ ] Created `<experiment>` folders in all roots
- [ ] Added raw images
- [ ] Generated `load_data.csv` and platemap
- [ ] Ran QC + analysis pipelines
- [ ] Pushed only lightweight files
- [ ] Archived data in REDU
