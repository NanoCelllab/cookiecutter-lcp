# hooks/post_gen_project.py
import pathlib
import subprocess
import shutil  # <- novo

def run(cmd):
    subprocess.run(cmd, check=True)

# Contexto direto via Jinja
ctx = {
    "use_git_lfs": "{{ cookiecutter.use_git_lfs }}",
    "include_example_pipelines": "{{ cookiecutter.include_example_pipelines }}",
    "include_example_notebooks": "{{ cookiecutter.include_example_notebooks }}",
    "include_example_models": "{{ cookiecutter.include_example_models }}",
    "include_redu_packager": "{{ cookiecutter.include_redu_packager }}",
    "cell_line": "{{ cookiecutter.cell_line }}",
    "assay_slug": "{{ cookiecutter.assay_slug }}"
}

use_lfs   = (ctx["use_git_lfs"].lower() == "yes")
inc_pipe  = (ctx["include_example_pipelines"].lower() == "yes")
inc_nb    = (ctx["include_example_notebooks"].lower() == "yes")
inc_models= (ctx["include_example_models"].lower() == "yes")
inc_redu  = (ctx["include_redu_packager"].lower() == "yes")

base = pathlib.Path(".")

# 1) git init
try:
    run(["git", "--version"])
    run(["git", "init"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", "Initial scaffold via Cookiecutter"])
except Exception as e:
    print(f"⚠️  Aviso: não foi possível rodar git automaticamente: {e}")

# 2) Git LFS (opcional)
if use_lfs:
    try:
        run(["git", "lfs", "install"])
        gitattributes = base / ".gitattributes"
        if gitattributes.exists():
            run(["git", "add", ".gitattributes"])
            run(["git", "commit", "-m", "Configure Git LFS"])
        else:
            print("ℹ️  .gitattributes não encontrado; crie um se quiser rastrear arquivos grandes via LFS.")
    except Exception as e:
        print(f"⚠️  Aviso: não foi possível configurar Git LFS: {e}")

# 3) Pipelines de exemplo (*.cppipe.example)
for p in base.rglob("*.cppipe.example"):
    try:
        if inc_pipe:
            p.rename(p.with_suffix(""))  # remove ".example"
        else:
            p.unlink()
    except Exception as e:
        print(f"⚠️  Aviso ao processar {p}: {e}")

# 4) REDU packager
for p in base.rglob("make_redu_package.py.example"):
    try:
        if inc_redu:
            p.rename(p.with_suffix(""))
        else:
            p.unlink()
    except Exception as e:
        print(f"⚠️  Aviso ao processar {p}: {e}")

# 5) Notebook starter (opcional)
if inc_nb:
    nbdir = base / ctx["cell_line"] / ctx["assay_slug"] / "workspace_dl" / "notebooks"
    try:
        nbdir.mkdir(parents=True, exist_ok=True)
        (nbdir / "00_setup.ipynb").write_text("{}")
    except Exception as e:
        print(f"⚠️  Aviso: não foi possível criar notebook starter: {e}")

# 6) Modelos Cellpose de exemplo (arquivos OU diretórios com sufixo .example)
models_dir = base / ctx["cell_line"] / ctx["assay_slug"] / "workspace" / "models"
if models_dir.exists():
    for p in models_dir.iterdir():
        try:
            if p.name.endswith(".example"):
                if inc_models:
                    target = models_dir / p.name[:-8]  # remove ".example"
                    p.rename(target)
                else:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
        except Exception as e:
            print(f"⚠️  Aviso ao processar modelo {p}: {e}")

print("\n✅ Projeto criado com sucesso!")
print("ℹ️  Dicas:")
print("   - Conecte ao GitHub:")
print("       git remote add origin <URL>")
print("       git branch -M main")
print("       git push -u origin main")
print("   - Ambiente conda:")
print("       conda env create -f env/environment.yml")
print("       conda activate lcp")
