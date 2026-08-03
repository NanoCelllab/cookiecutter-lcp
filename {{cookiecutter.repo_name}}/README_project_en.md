# 🧬 [Project Title]

**Author:** [Full Name]  
**Advisor:** [Advisor Name]  
**Institution:** [Lab / Graduate Program]  
**Project start:** [YYYY-MM]  
**Last update:** [YYYY-MM-DD]

---

## 🧠 About this repository

This repository serves as the **main umbrella** for your MSc/PhD project.  
It centralizes all **experiments, metadata, pipelines, and analyses** performed throughout your research.

Each experiment has its own folder (e.g., `2025_06_28_Huh7_NPPS_24h`), created following the [cookiecutter-lcp](https://github.com/NanoCellLab/cookiecutter-lcp) structure.

> 💡 **Note:** only **light files** (scripts, metadata, notebooks) should be versioned here.  
> **Raw microscopy images and heavy files** must be stored in [REDU (Unicamp)](https://redu.unicamp.br) or another institutional repository.

---

## 🎯 Project objective

> ✏️ **Briefly describe your project's goal in 3–5 lines.**  
> What are you investigating? What questions are you trying to answer?

*Example:*  
This project investigates the effects of polystyrene nanoparticles (NPPS) on human liver cells (Huh-7) using image-based phenotypic profiling (Live Cell Painting). The aim is to identify morphological signatures of toxicity and relate them to molecular mechanisms.

---

## 🧪 Methodology overview

> ✏️ **List your main methods, assays, and tools.**

*Example:*
- Cell models: Huh-7, HepG2  
- Approach: Live Cell Painting (HCA)  
- Segmentation: CellProfiler + CellPose  
- Feature extraction: pycytominer  
- Analysis: PCA, UMAP, LDA, SHAP (Explainable AI)

---

## 📂 Repository structure

```
<repo_root>/
├── images/                    # raw and illumination-corrected images
├── workspace/
│   ├── metadata/              # templates/ + one folder per experiment
│   ├── load_data_csv/         # one folder per experiment
│   ├── pipelines/             # templates/ (.cppipe) + one folder per experiment
│   ├── hca_pipeline/          # shared Python package (not per-experiment)
│   ├── assaydev/              # one folder per experiment
│   ├── segmentation/cellpose/ # model/, training/ (shared) + objects/ per experiment
│   ├── analysis/              # templates/ (notebooks) + one folder per experiment
│   ├── profiles/              # one folder per experiment
│   ├── backend/               # one folder per experiment
│   └── models/                # trained ML models, one folder per experiment
└── workspace_dl/              # deep learning notebooks (optional)
```

Every `workspace/<subfolder>/` (except the shared `hca_pipeline/` and
`segmentation/cellpose/{model,training}/`) has one sub-folder per experiment,
named `YYYY_MM_DD_Cell_Treatment_Time`. Where a `templates/` sibling exists,
copy from it into your new experiment folder and adapt.

---

## 🧩 Main tools and libraries

| Category | Tools / Libraries |
|-----------|------------------|
| Image analysis | CellProfiler, CellPose |
| Feature extraction | pycytominer |
| Machine learning | scikit-learn, LightGBM |
| Visualization | matplotlib, seaborn, umap-learn |
| Version control | git + GitHub |
| Reproducibility | cookiecutter-lcp, conda/uv |

---

## 📈 Current project status

> ✏️ **Briefly describe what stage the project is in.**  
> Example: model standardization, data acquisition, model training, analysis, manuscript preparation, etc.

---

## 📅 Timeline and next steps

- [ ] Standardize image analysis pipeline  
- [ ] Extract phenotypic profiles  
- [ ] Perform dose–response analysis  
- [ ] Correlate with cell viability data  
- [ ] Prepare manuscript

---

## 🔗 Useful links

- [CellProfiler](https://cellprofiler.org)  
- [pycytominer](https://github.com/cytomining/pycytominer)  
- [Cell Painting Gallery](https://broadinstitute.github.io/cellpainting-gallery/data_structure.html)  
- [REDU (Unicamp)](https://redu.unicamp.br)

---

## 🧠 Final note

This repository was created using the **[cookiecutter-lcp](https://github.com/NanoCellLab/cookiecutter-lcp)** template, developed to ensure consistent and reproducible organization of image-based phenotypic profiling projects.  
Keep your repository updated and document every experiment clearly.
