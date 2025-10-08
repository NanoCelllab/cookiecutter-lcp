# hooks/pre_gen_project.py
import re
import sys

def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

date = "{{ cookiecutter.experiment_date }}"
repo = "{{ cookiecutter.repo_name }}"
exp  = "{{ cookiecutter.experiment_name }}"
focus = "{{ cookiecutter.project_focus }}"
cell  = "{{ cookiecutter.cell_model }}"

# 1) Validate date as YYYY_MM_DD
if not re.fullmatch(r"\d{4}_\d{2}_\d{2}", date):
    fail("experiment_date must be in the format YYYY_MM_DD (e.g., 2025_06_28).")

# 2) Forbid spaces and weird chars in key fields
safe_pat = re.compile(r"^[A-Za-z0-9._\-]+$")

for name, val in [
    ("repo_name", repo),
    ("experiment_name", exp),
    ("project_focus", focus),
    ("cell_model", cell),
]:
    if " " in val:
        fail(f"{name} contains spaces. Use '-' or '_' instead of spaces. Got: '{val}'")
    if not safe_pat.fullmatch(val):
        fail(f"{name} has invalid characters. Allowed: letters, numbers, '-', '_', '.'. Got: '{val}'")

print("✔ Pre-generation checks passed.")
