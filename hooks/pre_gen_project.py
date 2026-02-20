# hooks/pre_gen_project.py
import re, sys

def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

safe_pat = re.compile(r"^[A-Za-z0-9._\-]+$")  # letras, números, ., _, -

fields = {
    "modality": "{{ cookiecutter.modality }}".strip(),
    "project_tag": "{{ cookiecutter.project_tag }}".strip(),
    "cell_models": "{{ cookiecutter.cell_models }}".strip(),
    "tissue": "{{ cookiecutter.tissue }}".strip(),
    "perturbation_category": "{{ cookiecutter.perturbation_category }}".strip(),
    "nanoparticle": "{{ cookiecutter.nanoparticle }}".strip(),
    "drug": "{{ cookiecutter.drug }}".strip(),
    "genetic": "{{ cookiecutter.genetic }}".strip(),
    "repo_name": "{{ cookiecutter.repo_name }}".strip()
}

# 🚨 obrigatórios mínimos
if not fields["modality"]:
    fail("modality cannot be empty (e.g., lcp, hca, bioinfo).")
if not fields["project_tag"]:
    fail("project_tag cannot be empty (e.g., nanotoxicology, hepatotoxicity).")

# ✅ valida campos preenchidos (sem espaços ou caracteres ilegais)
for name, val in fields.items():
    if not val:
        continue
    if " " in val:
        fail(f"{name} contains spaces. Use '-' or '_' instead of spaces. Got: '{val}'")
    if not safe_pat.fullmatch(val):
        fail(f"{name} has invalid characters. Allowed: letters, numbers, '-', '_', '.'. Got: '{val}'")

def safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))

safe_print("✔ Pre-generation checks passed.")
