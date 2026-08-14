"""Canonical provenance records for the Live Cell Painting pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PROVENANCE_SCHEMA_VERSION = 2
REQUIRED_PROVENANCE_BLOCKS = {
    "schema_version", "pipeline", "configuration", "dataset", "dependencies",
    "version_control", "environment", "outputs", "analysis",
}


def _git_value(repo_root: Path, arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=str(repo_root), capture_output=True,
            text=True, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def file_record(path: str | Path, *, role: str) -> dict[str, Any]:
    """Describe one dependency/output, hashing files without loading them whole."""
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "role": role,
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        return record
    if not resolved.is_file():
        record["kind"] = "directory"
        return record

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    record.update(
        {
            "kind": "file",
            "size_bytes": stat.st_size,
            "sha256": digest.hexdigest(),
            "modified_at_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    )
    return record


def _dependency_source(path: str | Path) -> str:
    """Infer the pipeline producer for a declared dependency path."""
    name = Path(path).name
    if name.startswith("single_cell_profiles"):
        return "01_samples_retrieval.py"
    if name in {"per_well_features_selected.parquet", "per_well_features_selected_merged.parquet", "single_cell_ready.parquet", "cv_summary.csv"}:
        return "02_aggregate_normalize_featureselect.py"
    if Path(path).suffix.lower() in {".sqlite", ".sqlite3", ".db", ".csv"}:
        return "external_or_cellprofiler_input"
    return "unknown"


def _records(paths: Iterable[str | Path], role: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records = []
    for path in paths:
        key = str(Path(path).expanduser().resolve())
        if key not in seen:
            record = file_record(path, role=role)
            if role == "input":
                record["source_notebook"] = _dependency_source(path)
            records.append(record)
            seen.add(key)
    return records


def canonicalize_provenance(
    legacy: Mapping[str, Any],
    *,
    notebook: str,
    experiment_id: str,
    repo_root: str | Path,
    dependencies: Iterable[str | Path] = (),
    outputs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Wrap an existing notebook record in the common schema-v2 envelope.

    Existing canonical blocks are retained. Notebook-specific or legacy flat
    fields move to ``analysis`` so no evidence is discarded during migration.
    """
    source = dict(legacy)
    repo = Path(repo_root).resolve()
    pipeline_existing = dict(source.get("pipeline", {}))
    executed_at = (
        pipeline_existing.get("executed_at_utc")
        or source.get("timestamp")
        or datetime.now(timezone.utc).isoformat()
    )
    commit = _git_value(repo, ["rev-parse", "HEAD"])
    branch = _git_value(repo, ["branch", "--show-current"])
    dirty_raw = _git_value(repo, ["status", "--porcelain"])

    canonical_keys = {
        "schema_version", "pipeline", "configuration", "dataset",
        "dependencies", "version_control", "environment", "outputs", "analysis",
    }
    analysis = dict(source.get("analysis", {}))
    analysis.update({key: value for key, value in source.items() if key not in canonical_keys})

    version_control = dict(source.get("version_control", {}))
    version_control.update(
        {
            "git_commit": commit or version_control.get("git_commit", "unknown"),
            "git_commit_short": (commit[:8] if commit else version_control.get("git_commit_short", "unknown")),
            "git_branch": branch or version_control.get("git_branch", "unknown"),
            "working_tree_dirty": bool(dirty_raw) if dirty_raw is not None else version_control.get("working_tree_dirty"),
        }
    )

    environment = dict(source.get("environment", {}))
    environment.update(
        {
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
    )

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "pipeline": {
            **pipeline_existing,
            "notebook": notebook,
            "experiment_id": experiment_id,
            "executed_at_utc": executed_at,
        },
        "configuration": dict(source.get("configuration", {})),
        "dataset": dict(source.get("dataset", {})),
        "dependencies": _records(dependencies, "input"),
        "version_control": version_control,
        "environment": environment,
        "outputs": _records(outputs, "output"),
        "analysis": analysis,
    }


def provenance_json(record: Mapping[str, Any]) -> str:
    """Serialize a provenance record consistently."""
    validate_provenance_record(record)
    return json.dumps(record, indent=2, ensure_ascii=False, default=str)


def validate_provenance_record(record: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a record does not satisfy schema v2."""
    missing = REQUIRED_PROVENANCE_BLOCKS - set(record)
    if missing:
        raise ValueError(f"Provenance record is missing required blocks: {sorted(missing)}")
    if record.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected provenance schema v{PROVENANCE_SCHEMA_VERSION}, "
            f"found {record.get('schema_version')!r}."
        )
    pipeline = record.get("pipeline", {})
    required_pipeline = {"notebook", "experiment_id", "executed_at_utc"}
    missing_pipeline = required_pipeline - set(pipeline)
    if missing_pipeline:
        raise ValueError(f"Provenance pipeline block is incomplete: {sorted(missing_pipeline)}")
    for block in ("dependencies", "outputs"):
        if not isinstance(record.get(block), list):
            raise ValueError(f"Provenance {block!r} block must be a list of file records.")
