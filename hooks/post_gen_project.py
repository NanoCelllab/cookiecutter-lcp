# hooks/post_gen_project.py (mínimo)
import pathlib, subprocess

def run(cmd): subprocess.run(cmd, check=True)

LANG = "{{ cookiecutter.ui_language }}".lower()
EXPERIMENT = "{{ cookiecutter.experiment_name }}"
inc_pipe   = "{{ cookiecutter.include_example_pipelines }}".lower() == "yes"
inc_nb     = "{{ cookiecutter.include_example_notebooks }}".lower() == "yes"
inc_models = "{{ cookiecutter.include_example_models }}".lower() == "yes"

def log(pt, en): print(pt if LANG == "pt" else en)

base = pathlib.Path(".")

# 0) git init (best-effort)
try:
    run(["git","--version"])
    run(["git","init"])
    run(["git","add","."])
    run(["git","commit","-m","Initial scaffold (LCP, per-experiment)"])
except Exception as e:
    log(f"⚠️  Aviso: não foi possível rodar git: {e}",
        f"⚠️  Warning: could not run git: {e}")

# 1) Estrutura por experimento (sem plates)
for p in [
    base/"images"/EXPERIMENT/"images",
    base/"images"/EXPERIMENT/"illum",
    base/"workspace"/"pipelines"/EXPERIMENT,
    base/"workspace"/"metadata"/EXPERIMENT/"platemap",
    base/"workspace"/"assaydev"/EXPERIMENT/"outlines_qc",
    base/"workspace"/"analysis"/EXPERIMENT/"analysis",   # CP vai popular aqui
    base/"workspace"/"load_data_csv"/EXPERIMENT,
    base/"workspace"/"backend"/EXPERIMENT,
    base/"workspace"/"profiles"/EXPERIMENT,
    base/"workspace"/"models",
    base/"workspace_dl"/EXPERIMENT/"notebooks",
]:
    p.mkdir(parents=True, exist_ok=True)
    (p/".gitkeep").touch(exist_ok=True)

# 2) Exemplos opcionais
if inc_pipe:
    for name in ["assaydev.cppipe","analysis.cppipe","illum.cppipe"]:
        (base/"workspace"/"pipelines"/EXPERIMENT/name).write_text("# CellProfiler pipeline\n")
if inc_nb:
    (base/"workspace_dl"/EXPERIMENT/"notebooks"/"00_setup.ipynb").write_text("{}")
if inc_models:
    (base/"workspace"/"models"/".gitkeep").touch(exist_ok=True)

# 3) Mensagem final
if LANG=="pt":
    print("\n✅ Estrutura criada (LCP por experimento).")
    print("Próximos passos:")
    print(" - Coloque dados brutos em images/<exp>/images")
    print(" - Adicione platemap/metadata em workspace/metadata/<exp>/")
    print(" - Ajuste pipelines em workspace/pipelines/<exp>/")
else:
    print("\n✅ Structure created (per-experiment LCP).")
    print("Next steps:")
    print(" - Put raw data in images/<exp>/images")
    print(" - Add platemap/metadata in workspace/metadata/<exp>/")
    print(" - Edit pipelines under workspace/pipelines/<exp>/")
