# --- Normalização de arquivos ".example" com logs verbosos ---

import shutil

LANG = "{{ cookiecutter.ui_language }}".lower()
inc_pipe   = "{{ cookiecutter.include_example_pipelines }}".lower() == "yes"
inc_models = "{{ cookiecutter.include_example_models }}".lower() == "yes"

def log(msg_pt, msg_en):
    print(msg_pt if LANG == "pt" else msg_en)

# 1) Pipelines (.cppipe.example)
for p in base.glob("**/*.cppipe.example"):
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

# 2) Modelos (qualquer arquivo/pasta *.example dentro de workspace/models)
models_dir = base / "{{ cookiecutter.cell_line }}" / "{{ cookiecutter.assay_slug }}" / "workspace" / "models"
if models_dir.exists():
    for p in models_dir.rglob("*.example"):
        try:
            if inc_models:
                # remove o sufixo ".example" de arquivo OU pasta
                target = p.with_name(p.name[:-8])  # corta ".example"
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
