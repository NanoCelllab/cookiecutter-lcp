# hooks/pre_gen_project.py
import re
import textwrap
from cookiecutter.exceptions import FailedHookException

LANG = "{{ cookiecutter.ui_language }}".lower()

def fail(msg: str):
    raise FailedHookException(msg)

# validações mínimas (independe de idioma)
assay_slug = "{{ cookiecutter.assay_slug }}"
if not re.fullmatch(r"[a-z0-9_-]+", assay_slug):
    if LANG == "pt":
        fail("❌ 'assay_slug' deve conter apenas [a-z0-9_-], sem espaços e sem acentos.")
    else:
        fail("❌ 'assay_slug' must contain only [a-z0-9_-], no spaces or accents.")

plate_format = "{{ cookiecutter.plate_format }}"
if plate_format not in ("6", "12", "24", "48", "96", "384"):
    if LANG == "pt":
        fail("❌ 'plate_format' deve ser um dos valores: 6, 12, 24, 48, 96 ou 384.")
    else:
        fail("❌ 'plate_format' must be one of: 6, 12, 24, 48, 96 or 384.")

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
    "include_example_models": "{{ cookiecutter.include_example_models }}",
    "include_redu_packager": "{{ cookiecutter.include_redu_packager }}",
}

if LANG == "pt":
    msg = f"""
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
    • Modelos exemplo .. {ctx['include_example_models']}
    • REDU packager .... {ctx['include_redu_packager']}

    🚀 O projeto será criado com essa configuração.
    Se algo estiver errado, interrompa (Ctrl+C) e rode novamente.
    ===========================================================
    """
else:
    msg = f"""
    ===========================================================
    ✅ Summary of your configuration
    ===========================================================
    • Author ........... {ctx['author_name']} ({ctx['author_email']})
    • Organization ..... {ctx['org_name']}
    • Cell line ........ {ctx['cell_line']}
    • Assay ............ {ctx['assay_name']} (slug: {ctx['assay_slug']})
    • Plate format ..... {ctx['plate_format']}-well
    • Date tag ......... {ctx['date_tag']}

    • Git LFS .......... {ctx['use_git_lfs']}
    • Example pipelines  {ctx['include_example_pipelines']}
    • Example notebooks  {ctx['include_example_notebooks']}
    • Example models ... {ctx['include_example_models']}
    • REDU packager .... {ctx['include_redu_packager']}

    🚀 The project will be created with this configuration.
    If anything looks wrong, abort (Ctrl+C) and run again.
    ===========================================================
    """
print(textwrap.dedent(msg))
