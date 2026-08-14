"""Create a safe per-experiment workspace from the pipeline templates."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import uuid
from pathlib import Path


VALID_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def create_experiment(repo_root: Path, experiment_id: str) -> Path:
    if not VALID_EXPERIMENT_ID.fullmatch(experiment_id) or experiment_id in {".", ".."}:
        raise ValueError(
            "Experiment ID must contain only letters, numbers, '.', '_' or '-', "
            "must start with a letter/number, and cannot contain a path separator."
        )

    workspace = repo_root / "workspace"
    notebook_templates = workspace / "analysis" / "templates"
    extra_templates = workspace / "analysis" / "extras"
    metadata_templates = workspace / "metadata" / "templates"
    experiment_root = workspace / "analysis" / experiment_id
    notebook_dir = experiment_root / "analysis"
    if experiment_root.exists():
        raise FileExistsError(
            f"Experiment analysis already exists: {experiment_root}\n"
            "Nothing was changed. Choose another ID or continue using the existing folder."
        )
    if not notebook_templates.is_dir():
        raise FileNotFoundError(f"Notebook template directory not found: {notebook_templates}")

    temporary_root = experiment_root.with_name(f".{experiment_id}.creating-{uuid.uuid4().hex}")
    try:
        (temporary_root / "analysis").mkdir(parents=True)
        for template in sorted(notebook_templates.glob("*.py")):
            shutil.copy2(template, temporary_root / "analysis" / template.name)
        if extra_templates.is_dir():
            (temporary_root / "analysis" / "extras").mkdir()
            for template in sorted(extra_templates.glob("*.py")):
                shutil.copy2(template, temporary_root / "analysis" / "extras" / template.name)
        for name in ("outputs", "results", "figures", "reports"):
            (temporary_root / name).mkdir()
        temporary_root.rename(experiment_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    # Raw inputs or metadata may have been placed before the notebooks were
    # initialized. Preserve them and only create/copy paths that are absent.
    (workspace / "backend" / experiment_id).mkdir(parents=True, exist_ok=True)
    (workspace / "profiles" / experiment_id).mkdir(parents=True, exist_ok=True)
    metadata_dir = workspace / "metadata" / experiment_id
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if metadata_templates.is_dir():
        for template in sorted(metadata_templates.iterdir()):
            destination = metadata_dir / template.name
            if template.is_file() and not destination.exists():
                shutil.copy2(template, destination)

    return notebook_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an experiment folder and copy the pipeline and optional extra notebooks safely."
    )
    parser.add_argument("experiment_id", help="For example: 2026_08_Huh7_NPPS_5_days")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        notebook_dir = create_experiment(repo_root, args.experiment_id)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    notebooks = sorted(notebook_dir.glob("*.py"))
    extras = sorted((notebook_dir / "extras").glob("*.py"))
    print(f"✓ Experiment created: {args.experiment_id}")
    print(f"  Notebooks: {notebook_dir} ({len(notebooks)} files)")
    print(f"  Optional project-specific extras: {notebook_dir / 'extras'} ({len(extras)} files)")
    print(f"  Raw CellProfiler inputs: workspace/backend/{args.experiment_id}/")
    print(f"  Metadata: workspace/metadata/{args.experiment_id}/")
    print("\nOpen the first notebook with:")
    print(f"  pixi run marimo edit {notebook_dir / '01_samples_retrieval.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
