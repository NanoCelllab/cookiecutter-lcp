# hooks/post_gen_project.py
import pathlib, subprocess, shutil

def run(cmd): subprocess.run(cmd, check=True)

# --- Contexto do Cookiecutter (Jinja) ---
LANG       = "{{ cookiecutter.ui_language }}".lower()
use_lfs    = "{{ cookiecutter.use_git_lfs }}".lower() == "yes"
inc_pipe   = "{{ cookiecutter.include_example_pipelines }}".lower() == "yes"
inc_nb     = "{{ cookiecutter.include_example_notebooks }}".lower() == "yes"
inc_models = "{{ cookiecutter.include_example_models }}".lower() == "yes"
inc_redu   = "{{ cookiecutter.include_redu_packager }}".lower() == "yes"
cell_line  = "{{ cookiecutter.cell_line }}"
assay_slug = "{{ cookiecutter.assay_slug }}"
batch_tag  = "{{ cookiecutter.batch_tag }}"
plate_id   = "{{ cookiecutter.plate_id_example }}"

base = pathlib.Path(".")

def log(pt, en): print(pt if LANG == "pt" else en)
def rel(p: pathlib.Path) -> str: 
    try: return str(p.resolve().relative_to(base.resolve()))
    except Exception: return str(p)

def safe_rename(src: pathlib.Path, dst: pathlib.Path):
    if dst.exists():
        if dst.is_dir(): shutil.rmtree(dst)
        else: dst.unlink()
    src.rename(dst)

# 0) git init (best effort)
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
        if not gitattributes.exists():
            gitattributes.write_text(
                "\n".join([
                    "*.tif filter=lfs diff=lfs merge=lfs -text",
                    "*.tiff filter=lfs diff=lfs merge=lfs -text",
                    "*.nd2 filter=lfs diff=lfs merge=lfs -text",
                    "*.czi filter=lfs diff=lfs merge=lfs -text",
                    "*.lif filter=lfs diff=lfs merge=lfs -text",
                    "*.npz filter=lfs diff=lfs merge=lfs -text",
                    "*.npy filter=lfs diff=lfs merge=lfs -text",
                ]) + "\n", encoding="utf-8"
            )
        run(["git", "add", ".gitattributes"])
        run(["git", "commit", "-m", "Configure Git LFS"])
    except Exception as e:
        log(f"⚠️  Aviso: não foi possível configurar Git LFS: {e}",
            f"⚠️  Warning: could not configure Git LFS: {e}")

# 2) Ativar pipelines .cppipe.example
for p in base.rglob("*.cppipe.example"):
    try:
        if inc_pipe:
            target = p.with_suffix("")  # remove ".example"
            safe_rename(p, target)
            log(f"✓ Pipeline ativado: {rel(target)}",
                f"✓ Pipeline enabled: {rel(target)}")
        else:
            p.unlink()
            log(f"✗ Pipeline removido: {rel(p)}",
                f"✗ Pipeline removed: {rel(p)}")
    except Exception as e:
        log(f"⚠️ Erro ao processar pipeline {rel(p)}: {e}",
            f"⚠️ Error processing pipeline {rel(p)}: {e}")

# 2.1) Ativar barcode_platemap.csv
for p in base.rglob("workspace/metadata/*/barcode_platemap.csv.example"):
    try:
        safe_rename(p, p.with_name("barcode_platemap.csv"))
        log("✓ barcode_platemap.csv ativado",
            "✓ barcode_platemap.csv enabled")
    except Exception as e:
        log(f"⚠️ Erro ao ativar barcode_platemap: {e}",
            f"⚠️ Error enabling barcode_platemap: {e}")

# 3) Ativar/Remover REDU packager
for p in base.rglob("make_redu_package.py.example"):
    try:
        if inc_redu:
            target = p.with_suffix("")
            safe_rename(p, target)
            log(f"✓ REDU packager ativado: {rel(target)}",
                f"✓ REDU packager enabled: {rel(target)}")
        else:
            p.unlink()
            log(f"✗ REDU packager removido: {rel(p)}",
                f"✗ REDU packager removed: {rel(p)}")
    except Exception as e:
        log(f"⚠️ Erro ao processar REDU packager {rel(p)}: {e}",
            f"⚠️ Error processing REDU packager {rel(p)}: {e}")

# 4) Notebook starter
if inc_nb:
    nbdir = base / cell_line / assay_slug / "workspace_dl" / "notebooks"
    try:
        nbdir.mkdir(parents=True, exist_ok=True)
        nbdir.joinpath("00_setup.ipynb").write_text(
            '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n',
            encoding="utf-8"
        )
        log(f"✓ Notebook inicial criado em {rel(nbdir / '00_setup.ipynb')}",
            f"✓ Starter notebook created at {rel(nbdir / '00_setup.ipynb')}")
    except Exception as e:
        log(f"⚠️  Aviso: não foi possível criar notebook starter: {e}",
            f"⚠️  Warning: could not create starter notebook: {e}")

# 5) Modelos em workspace/models (*.example)
models_dir = base / cell_line / assay_slug / "workspace" / "models"
if models_dir.exists():
    for p in models_dir.iterdir():
        try:
            if p.name.endswith(".example"):
                if inc_models:
                    target = p.with_name(p.name[:-8])  # remove ".example"
                    safe_rename(p, target)
                    log(f"✓ Modelo ativado: {rel(target)}",
                        f"✓ Model enabled: {rel(target)}")
                else:
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
                    log(f"✗ Modelo removido: {rel(p)}",
                        f"✗ Model removed: {rel(p)}")
        except Exception as e:
            log(f"⚠️ Erro ao processar modelo {rel(p)}: {e}",
                f"⚠️ Error processing model {rel(p)}: {e}")
else:
    log("ℹ️ Pasta de modelos não encontrada (pule se não usar).",
        "ℹ️ Models folder not found (skip if not using).")

# 5') Garantir <cell>/<assay>/<batch>/(images|illum)/<plate>/.gitkeep
raw_imgs  = base / cell_line / assay_slug / batch_tag / "images" / plate_id
raw_illum = base / cell_line / assay_slug / batch_tag / "illum"  / plate_id
for d in (raw_imgs, raw_illum):
    d.mkdir(parents=True, exist_ok=True)
    keep = d / ".gitkeep"
    if not keep.exists(): keep.write_text("")
log(f"✓ Pastas de batch criadas/garantidas: {rel(raw_imgs)} e {rel(raw_illum)}",
    f"✓ Ensured raw batch folders: {rel(raw_imgs)} and {rel(raw_illum)}")

# 6) Commit final das mudanças pós-ativação
try:
    run(["git", "add", "-A"])
    run(["git", "commit", "-m", "Activate examples and scaffold batch folders"])
except Exception as e:
    log(f"⚠️  Aviso: commit final não foi possível: {e}",
        f"⚠️  Warning: final commit not possible: {e}")

# 7) Mensagem final
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

