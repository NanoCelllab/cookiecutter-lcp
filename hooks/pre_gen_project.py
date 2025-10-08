# hooks/pre_gen_project.py
import textwrap

LANG = "{{ cookiecutter.ui_language }}".lower()

ctx = {
    "author_name": "{{ cookiecutter.author_name }}",
    "author_email": "{{ cookiecutter.author_email }}",
    "org_name": "{{ cookiecutter.org_name }}",
    "project_title": "{{ cookiecutter.project_title }}",
    "repo_name": "{{ cookiecutter.repo_name }}",
    "include_example_pipelines": "{{ cookiecutter.include_example_pipelines }}",
    "include_example_notebooks": "{{ cookiecutter.include_example_notebooks }}",
    "include_example_models": "{{ cookiecutter.include_example_models }}",
}

if LANG == "pt":
    msg = f"""
    ===========================================================
    ✅ Resumo da configuração escolhida
    ===========================================================
    • Autor ............ {ctx['author_name']} ({ctx['author_email']})
    • Organização ...... {ctx['org_name']}
    • Título do projeto  {ctx['project_title']}
    • Nome do repositório {ctx['repo_name']}

    • Pipelines exemplo  {ctx['include_example_pipelines']}
    • Notebooks exemplo  {ctx['include_example_notebooks']}
    • Modelos exemplo .. {ctx['include_example_models']}

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
    • Project title .... {ctx['project_title']}
    • Repo name ........ {ctx['repo_name']}

    • Example pipelines  {ctx['include_example_pipelines']}
    • Example notebooks  {ctx['include_example_notebooks']}
    • Example models ... {ctx['include_example_models']}

    🚀 The project will be created with this configuration.
    If anything looks wrong, abort (Ctrl+C) and run again.
    ===========================================================
    """
print(textwrap.dedent(msg))
