# hooks/pre_gen_project.py
import re
import textwrap
from cookiecutter.exceptions import FailedHookException

def fail(msg: str):
    raise FailedHookException(msg)

# 🔹 Validações mínimas
assay_slug = "{{ cookiecutter.assay_slug }}"
if not re.fullmatch(r"[a-z0-9_-]+", assay_slug):
    fail("❌ 'assay_slug' deve conter apenas [a-z0-9_-], sem espaços e sem acentos.")

plate_format = "{{ cookiecutter.plate_format }}"
if plate_format not in ("6", "12", "24", "48", "96", "384"):
    fail("❌ 'plate_format' deve ser um dos valores válidos: 6, 12, 24, 48, 96 ou 384.")

# 🔹 Resumo amigável das escolhas
ctx = {
    "author_name": "{{ cookiecutter.author_name }}",
    "author_email": "{{ cookiecutter.author_email }}",
    "org_name": "{{ cookiecutter.org_name }}",
    "cell_line": "{{ cookiecutter.cell_line }}",
    "assay_name": "{{ cookiecutter.assay_name }}",
    "assay_slug": "{{ cookiecutter.assay_slug }}",
    "plate_format": "{{ cookiecutter.plate_format }}",
    "date_tag": "{{ cookiecutter.date_tag }}",
    "use_git_lfs": "{{ cookiecutter.use_git_lfs }}",
    "include_example_pipelines": "{{ cookiecutter.include_example_pipelines }}",
    "include_example_notebooks": "{{ cookiecutter.include_example_notebooks }}",
    "include_redu_packager": "{{ cookiecutter.include_redu_packager }}",
}

print(
    textwrap.dedent(f"""
    ===========================================================
    ✅ Resumo da configuração escolhida
    ===========================================================

    • Autor ............ {ctx['author_name']} ({ctx['author_email']})
    • Organização ...... {ctx['org_name']}
    • Linha celular .... {ctx['cell_line']}
    • Ensaio ........... {ctx['assay_name']} (slug: {ctx['assay_slug']})
    • Placa ............ {ctx['plate_format']}-well
    • Data (tag) ....... {ctx['date_tag']}

    • Git LFS .......... {ctx['use_git_lfs']}
    • Pipelines exemplo  {ctx['include_example_pipelines']}
    • Notebooks exemplo  {ctx['include_example_notebooks']}
    • REDU packager .... {ctx['include_redu_packager']}

    🚀 Agora o projeto será criado com essa configuração.
    Se algo estiver errado, interrompa (Ctrl+C) e rode de novo.
    ===========================================================
    """)
)
