# 🧬 Cookiecutter Live Cell Painting (LCP)

A **reproducible project generator** for Live Cell Painting and phenotypic profiling experiments.  
Developed by the **NanoCell Interactions Lab (Unicamp)** to standardize folder structures, streamline data organization, and promote reproducibility in image-based research.

---

## 📦 What this template does

This [Cookiecutter](https://cookiecutter.readthedocs.io/en/stable/) automatically creates a complete project scaffold, including:

- Organized folders for **batches** and **plates**
- Workspace directories for **analysis, metadata, pipelines, models, and profiles**
- Example files for **pipelines, notebooks, and models**
- Optional integration with **Git and Git-LFS** for reproducible version control

Ideal for:
- **Live Cell Painting** and **Cell Painting** assays  
- **High-content microscopy** and image-based screening  
- Projects with **multiple plates and batches**

---

## 🚀 How to use

### 1️⃣ Install Cookiecutter
```bash
pip install cookiecutter
```

### 2️⃣ Generate a new project
Run the template directly from GitHub:

```bash
cookiecutter gh:NanoCelllab/cookiecutter-lcp
```

During setup, you’ll see prompts like:
```
[1/21] Select ui_language (1=pt, 2=en)
...
[11/21] batch_tags (comma-separated list)
[12/21] plate_ids (comma-separated list)
...
```

Example input:
```
batch_tags: 250808_102354_Plate_1, 250813_094546_Plate_1, 250815_094252_Plate_1
plate_ids: 250808_101736_AONPPS, 250813_094546_AONPPS
```

---

## 📁 Example output structure

```
lcp-huh7-npps-20250509/
├── env/
│   └── environment.yml
├── Huh7/
│   └── npps/
│       ├── 250808_102354_Plate_1/
│       │   ├── images/
│       │   └── illum/
│       ├── workspace/
│       │   ├── metadata/
│       │   │   ├── barcode_platemap.csv
│       │   │   └── <batch>/platemap/
│       │   ├── analysis/
│       │   ├── pipelines/
│       │   ├── models/
│       │   └── profiles/
│       └── workspace_dl/
│           └── notebooks/
│               └── 00_setup.ipynb
└── README.md
```

---

## 🧠 Why use this template?

- **Standardizes** folder organization across assays and users  
- **Simplifies** downstream automation (CellProfiler, pycytominer, Zenodo, etc.)  
- **Ensures** reproducibility and data traceability  
- **Supports** multiple batches and plates automatically  

Used by the **NanoCell Interactions Lab (Unicamp)** and collaborators at the **Broad Institute**.

---

## ⚙️ Included hooks

- **pre_gen_project.py** → validates user inputs before generation  
- **post_gen_project.py** → dynamically creates all folders for each batch × plate combination and activates example files  

---

## 🔁 Updating your local copy

To refresh your local template:

```bash
rm -rf ~/.cookiecutters/cookiecutter-lcp
cookiecutter gh:NanoCelllab/cookiecutter-lcp
```

---

## 👩‍🔬 Authors

**Marcelo Bispo de Jesus** – NanoCell Interactions Lab, Unicamp  
Collaborators: Lucas.

---

## 📄 License

Distributed under the **MIT License**.  
See the [`LICENSE`](./LICENSE) file for details.

---

## 🔗 Related resources

- [Cell Painting – Broad Institute](https://www.broadinstitute.org/cell-painting)  
- [pycytominer – cytomining toolkit](https://github.com/cytomining/pycytominer)  
- [CellProfiler pipelines](https://cellprofiler.org/)
