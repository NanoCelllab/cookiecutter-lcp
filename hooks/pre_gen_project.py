# hooks/pre_gen_project.py
import re
import sys

def fail(msg: str):
    print(f"ERROR: {msg}")
    sys.exit(1)

# --- Pull context values ---
modality      = "{{ cookiecutter.modality }}".strip()
project_tag   = "{{ cookiecutter.project_tag }}".strip()
cell_models   = "{{ cookiecutter.cell_models }}".strip()
tissue        = "{{ cookiecutter.tissue }}".strip()
category      = "{{ cookiecutter.perturbation_category }}".strip().lower()
nanoparticle  = "{{ cookiecutter.nanoparticle }}".strip()
drug          = "{{ cookiecutter.drug }}".strip()
genetic       = "{{ cookiecutter.genetic }}".strip()
repo_name     = "{{ cookiecutter.repo_name }}".strip()

# --- REQUIRED fields: modality, project_tag must be non-empty ---
if not modality:
    fail("`modality` is required (e.g., lcp, hca, bioinfo).")
if not project_tag:
    fail("`project_tag` is required (e.g., nanotoxicology, hepatotoxicity).")

# --- Allowed characters: letters, numbers, dash, underscore, dot ---
safe_pat = re.compile(r"^[A-Za-z0-9._\-]+$")

def check_slug(label: str, value: str, allow_empty: bool = True):
    if not value and allow_empty:
        return
    if " " in value:
        fail(f"`{label}` contains spaces. Use '-' or '_' instead. Got: '{value}'")
    if not safe_pat.fullmatch(value):
        fail(f"`{label}` has invalid characters. Allowed: letters, numbers, '-', '_', '.'. Got: '{value}'")

# Validate required slugs
check_slug("modality", modality, allow_empty=False)
check_slug("project_tag", project_tag, allow_empty=False)

# Validate optional slugs
check_slug("tissue", tissue)
check_slug("cell_models", cell_models)

# If user provided multiple cell lines (comma-separated), validate each token loosely
if cell_models:
    cells = [c.strip() for c in cell_models.split(",") if c.strip()]
    for c in cells:
        check_slug("cell_models item", c)

# Category-specific optional fields
if category == "nanoparticle":
    if not nanoparticle:
        print("⚠️  Note: `perturbation_category` is 'nanoparticle' but `nanoparticle` is empty. Proceeding anyway.")
    else:
        check_slug("nanoparticle", nanoparticle)
elif category == "drug":
    if not drug:
        print("⚠️  Note: `perturbation_category` is 'drug' but `drug` is empty. Proceeding anyway.")
    else:
        check_slug("drug", drug)
elif category == "genetic":
    if not genetic:
        print("⚠️  Note: `perturbation_category` is 'genetic' but `genetic` is empty. Proceeding anyway.")
    else:
        check_slug("genetic", genetic)
elif category in ("none", "other"):
    pass
else:
    print(f"⚠️  Note: unknown `perturbation_category`='{category}'. Proceeding.")

# Final sanity on repo_name
if not repo_name:
    fail("Internal error: `repo_name` resolved to empty.")
check_slug("repo_name", repo_name, allow_empty=False)

print("✔ Pre-generation checks passed.")
