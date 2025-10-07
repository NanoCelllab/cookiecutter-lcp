# hooks/post_gen_project.py
import pathlib
import subprocess
import shutil

def run(cmd):
    subprocess.run(cmd, check=True)

# =========================
# Contexto via Jinja
# =========================
LANG       = "{{ cookiecutter.ui_language }}".lower()
use_lfs    = "{{ cookiecutter.use_git_lfs }}".lower() == "yes"
inc_pipe   = "{{ cookiecutter.include_example_pipelines }}".lower() == "yes"
inc_nb     = "{{ cookiecutter.include_example_notebooks }}".lower() == "yes"
inc_models = "{{ cookiecutter.include_example_models }}".lower() == "yes"
inc_redu   = "{{ cookiecutter.include_redu_packager }}".lower() == "yes"

cell_line  = "{{ cookiecutter.cell_line }}"
assay_slug = "{{ cookiecutter.assay_slug }}"

# Agora como STR possivelmente com vírgulas:
raw_batches = "{{ cookiecutter.batch_tags }}"
raw_plates  = "{{ cookiecutter.plate_ids }}"

def log(pt, en):
    print(pt if LANG == "pt" else en)

def _split_csv(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]

# Listas finais (aceita 1 ou vários valores)
def _split_csv(s: str):
    return [x.strip() for x in (s or "").split(",") if x.strip()]

batches = _split_csv(raw_batches)
plates  = _split_csv(raw_plates)

if not batches:
    log("⚠️  Nenhum 'batch_tags' informado; pulando criação de pastas de batch.",
        "⚠️  No 'batch_tags' provided; skipping batch folder creation.")
if not plates:
    log("⚠️  Nenhum 'plate_ids' informado; pulando criação de pastas de plate.",
        "⚠️  No 'plate_ids' provided; skipping plate folder creation.")


base = pathlib.Path(".")

# =========================
# 0) git init (melhor esforço)
# =========================
try:
    run(["git", "--version"])
    run(["git", "init"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", "Initial scaffold via Cookiecutter"])
except Exception as e:
    log(f"⚠️  Aviso: não foi possível rodar git automaticamente: {e}",
        f"⚠️  Warning: could not run git automatically: {e}")

# =========================
# 1) Git LFS (opcional)
# =========================
if use_lfs:
    try:
        run(["git", "lfs", "install"])
        gitattributes = base / ".gitattributes"
        if gitattributes.exists():
            run(["git", "add", ".gitattributes"])
            run(["git", "commit", "-m", "Configure Git LFS"])
        else:
            log("ℹ️  .gitattributes não encontrado; crie um para rastrear binários grandes com LFS.",
                "ℹ️  .gitattributes not found; create one to track large binaries with LFS.")
    except Exception as e:
        log(f"⚠️  Aviso: não foi possível configurar Git LFS: {e}",
            f"⚠️  Warning: could not configure Git LFS: {e}")

# =========================
# 2) Normalizar pipelines .cppipe.example
# =========================
for p in base.rglob("*.cppipe.example"):
    try:
        if inc_pipe:
            target = p.with_suffix("")  # remove ".example" -> vira .cppipe
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

# =========================
# 2.1) Ativar barcode_platemap.csv a partir de .example (qualquer batch)
# =========================
for p in base.rglob("workspace/metadata/*/barcode_platemap.csv.example"):
    try:
        p.rename(p.with_name("barcode_platemap.csv"))
        log("✓ barcode_platemap.csv ativado",
            "✓ barcode_platemap.csv enabled")
    except Exception as e:
        log(f"⚠️ Erro ao ativar barcode_platemap: {e}",
            f"⚠️ Error enabling barcode_platemap: {e}")

# =========================
# 3) Ativar/Remover REDU packager (ex.: make_redu_package.py.example)
# =========================
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

# =========================
# 4) Notebook starter (opcional)
# =========================
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

# =========================
# 5) Modelos em workspace/models (*.example como arquivo ou pasta)
# =========================
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

# =========================
# 6) Estruturas para múltiplos batches × plates
# =========================
created = []

# 6.1) Pastas de dados brutos (images/ e illum/)
for b in batches:
    for pid in plates:
        for sub in ("images", "illum"):
            d = base / cell_line / assay_slug / b / sub / pid
            d.mkdir(parents=True, exist_ok=True)
            k = d / ".gitkeep"
            if not k.exists():
                k.write_text("")
            created.append(d)

# 6.2) workspace: load_data_csv e backend por batch/plate
for b in batches:
    for pid in plates:
        for sub in ("load_data_csv", "backend"):
            d = base / cell_line / assay_slug / "workspace" / sub / b / pid
            d.mkdir(parents=True, exist_ok=True)
            k = d / ".gitkeep"
            if not k.exists():
                k.write_text("")

# 6.3) workspace: analysis por batch/plate
for b in batches:
    for pid in plates:
        d = base / cell_line / assay_slug / "workspace" / "analysis" / b / pid / "analysis"
        d.mkdir(parents=True, exist_ok=True)
        k = d / ".gitkeep"
        if not k.exists():
            k.write_text("")

# 6.4) workspace: metadata por batch (platemap e barcode)
for b in batches:
    d_meta = base / cell_line / assay_slug / "workspace" / "metadata" / b / "platemap"
    d_meta.mkdir(parents=True, exist_ok=True)
    k = d_meta / ".gitkeep"
    if not k.exists():
        k.write_text("")

# Log do que foi criado (raw)
for d in created:
    log(f"✓ Criado: {d}", f"✓ Created: {d}")

# 6.5) Se existir um barcode_platemap.csv.example na raiz (workspace/metadata),
#      copiar para cada batch e ativar como barcode_platemap.csv
root_example = base / cell_line / assay_slug / "workspace" / "metadata" / "barcode_platemap.csv.example"
if root_example.exists():
    for b in batches:
        dst = base / cell_line / assay_slug / "workspace" / "metadata" / b / "barcode_platemap.csv"
        if not dst.exists():
            dst.write_bytes(root_example.read_bytes())
            log(f"✓ barcode_platemap.csv copiado para {dst}",
                f"✓ barcode_platemap.csv copied to {dst}")
else:
    log("ℹ️ Nenhum barcode_platemap.csv.example na raiz de metadata; pulei cópia.",
        "ℹ️ No root metadata/barcode_platemap.csv.example; skipped copy.")


# =========================
# 7) Mensagem final
# =========================
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

