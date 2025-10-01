# hooks/post_gen_project.py
import pathlib
import subprocess
import shutil

def run(cmd):
    subprocess.run(cmd, check=True)

# Contexto via Jinja
LANG = "{{ cookiecutter.ui_language }}".lower()
use_lfs   = "{{ cookiecutter.use_git_lfs }}".lower() == "yes"
inc_pipe  = "{{ cookiecutter.include_example_pipelines }}".lower() == "yes"
inc_nb    = "{{ cookiecutter.include_example_notebooks }}".lower() == "yes"
inc_models= "{{ cookiecutter.include_example_models }}".lower() == "yes"
inc_redu  = "{{ cookiecutter.include_redu_packager }}".lower() == "yes"
cell_line = "{{ cookiecutter.cell_line }}"
assay_slug= "{{ cookiecutter.assay_slug }}"

def log(pt, en):
    print(pt if LANG == "pt" else en)

base = pathlib.Path(".")  # <-- Faltava isso!

# 0) git init (não falha se git não estiver disponível)
try:
    run(["git", "--version"])
    run(["git", "init"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", "Initial scaffold via Cookiecutter"])
except Exception as e:
    log(f"⚠️  Aviso: não foi possível rodar git automaticamente: {e}",
        f"⚠️  Warning: could not run git automatically: {e}")

# 1) Git LFS (opcional)
if use_lfs:
    try:
        run(["git", "lfs", "install"])
        gitattributes = base / ".gitattributes"
        if gitattributes.exists():
            run(["git", "add", ".gitattributes"])
            run(["git", "commit", "-m", "Configure Git LFS"])
        else:
            log("ℹ️  .gitattributes não encontrado; crie um se quiser rastrear arquivos grandes via LFS.",
                "ℹ️  .gitattributes not found; create one to track large files via LFS.")
    except Exception as e:
        log(f"⚠️  Aviso: não foi possível configurar Git LFS: {e}",
            f"⚠️  Warning: could not configure Git LFS: {e}")

# 2) Normalização de pipelines .cppipe.example
for p in base.rglob("*.cppipe.example"):
    try:
        if inc_pipe:
            target = p.with_suffix("")  # remove apenas ".example" -> vira .cppipe
            p.rename(target)
            log(f"✓ Pipeline ativado: {target}",
                f"✓ Pipeline enabled: {target}")
        else:
            p.unlink()
            log(f"✗ Pipeline removido: {p}",
                f"✗ Pipeline removed: {p}")
    except Exception as e:
        log(f"⚠️ Erro ao processar pipeline {p}: {e}",
            f"⚠️ Error processing pipeline {p}: {e}")

# 3) Ativar/Remover REDU packager (ex.: make_redu_package.py.example)
for p in base.rglob("make_redu_package.py.example"):
    try:
        if inc_redu:
            target = p.with_suffix("")  # remove ".example"
            p.rename(target)
            log(f"✓ REDU packager ativado: {target}",
                f"✓ REDU packager enabled: {target}")
        else:
            p.unlink()
            log(f"✗ REDU packager removido: {p}",
                f"✗ REDU packager removed: {p}")
    except Exception as e:
        log(f"⚠️ Erro ao processar REDU packager {p}: {e}",
            f"⚠️ Error processing REDU packager {p}: {e}")

# 4) Notebook starter (opcional)
if inc_nb:
    nbdir = base / cell_line / assay_slug / "workspace_dl" / "notebooks"
    try:
        nbdir.mkdir(parents=True, exist_ok=True)
        (nbdir / "00_setup.ipynb").write_text("{}")
        log("✓ Notebook inicial criado em workspace_dl/notebooks/00_setup.ipynb",
            "✓ Starter notebook created at workspace_dl/notebooks/00_setup.ipynb")
    except Exception as e:
        log(f"⚠️  Aviso: não foi possível criar notebook starter: {e}",
            f"⚠️  Warning: could not create starter notebook: {e}")

# 5) Modelos em workspace/models (*.example como arquivo ou pasta)
models_dir = base / cell_line / assay_slug / "workspace" / "models"
if models_dir.exists():
    for p in models_dir.iterdir():
        try:
            if p.name.endswith(".example"):
                if inc_models:
                    target = p.with_name(p.name[:-8])  # remove ".example"
                    p.rename(target)
                    log(f"✓ Modelo ativado: {target}",
                        f"✓ Model enabled: {target}")
                else:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    log(f"✗ Modelo removido: {p}",
                        f"✗ Model removed: {p}")
        except Exception as e:
            log(f"⚠️ Erro ao processar modelo {p}: {e}",
                f"⚠️ Error processing model {p}: {e}")
else:
    log("ℹ️ Pasta de modelos não encontrada (pule se não usar).",
        "ℹ️ Models folder not found (skip if not using).")

# 6) Mensagem final
if LANG == "pt":
    print("\n✅ Projeto criado com sucesso!")
    print("ℹ️  Próximos passos:")
    print("   - Conecte ao GitHub:")
    print("       git remote add origin <URL>")
    print("       git branch -M main")
    print("       git push -u origin main")
    print("   - Ambiente conda:")
    print("       conda env create -f env/environment.yml")
    print("       conda activate lcp")
else:
    print("\n✅ Project created successfully!")
    print("ℹ️  Next steps:")
    print("   - Connect to GitHub:")
    print("       git remote add origin <URL>")
    print("       git branch -M main")
    print("       git push -u origin main")
    print("   - Conda environment:")
    print("       conda env create -f env/environment.yml")
    print("       conda activate lcp")
