"""Path resolution and protected file I/O for the LCP pipeline.

``find_repo_root`` lets every notebook work regardless of the current working
directory (VS Code / Jupyter / marimo all differ on this). The protected
write helpers formalize a pattern used repeatedly across the original
notebooks: never silently replace an existing output that differs from the
freshly computed result -- compare first, and only overwrite when the caller
opts in via ``overwrite=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def find_repo_root(
    start: Path,
    markers: Sequence[str] = (".git", "pyproject.toml", "pixi.toml"),
) -> Path:
    """Walk up from *start* to find the repo root.

    Returns the first parent directory containing any of the marker files.
    Falls back to a best-guess ancestor if no marker is found.
    """
    start = start.resolve()
    for p in (start, *start.parents):
        if any((p / m).exists() for m in markers):
            return p
    return start.parents[5] if len(start.parents) > 5 else start


def parquet_structure(path: Path) -> tuple[int, list[str]]:
    """Return Parquet row count and column names without loading the table."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    return (
        int(parquet_file.metadata.num_rows),
        list(parquet_file.schema_arrow.names),
    )


def checkpoint_matches_plate_scope(
    path: Path,
    included_plates,
    *,
    plate_col: str = "Metadata_Plate",
) -> bool:
    """Return whether a Parquet checkpoint contains exactly the requested plates."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        observed = set(pd.read_parquet(path, columns=[plate_col])[plate_col].dropna().astype(str))
    except (KeyError, ValueError):
        return False
    return observed == {str(plate) for plate in included_plates}


def write_parquet_protected(df: pd.DataFrame, path: Path, *, overwrite: bool) -> str:
    """Write *df* to *path* as Parquet unless an incompatible file already exists.

    The existence check compares row count and column names only (not full
    content) -- deliberately cheap, since these tables can be large single-cell
    profiles. Returns one of ``"created"``, ``"unchanged"``, ``"replaced"``.
    Raises ``FileExistsError`` if the existing file differs and ``overwrite=False``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing_rows, existing_columns = parquet_structure(path)
        if existing_rows == len(df) and existing_columns == list(df.columns):
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                "Existing Parquet file differs from the current dataset and "
                f"was not overwritten: {path}"
            )
        df.to_parquet(path, index=False)
        return "replaced"

    df.to_parquet(path, index=False)
    return "created"


def write_csv_protected(df: pd.DataFrame, path: Path, *, overwrite: bool) -> str:
    """Write *df* to *path* as CSV, comparing column headers only.

    This mirrors the parquet check's "cheap by design" intent for large
    interoperability copies. Returns one of ``"created"``, ``"unchanged"``,
    ``"replaced"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing_header = pd.read_csv(path, nrows=0).columns.tolist()
        if existing_header == list(df.columns):
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                "Existing CSV columns differ from the current dataset and "
                f"the file was not overwritten: {path}"
            )
        df.to_csv(path, index=False)
        return "replaced"

    df.to_csv(path, index=False)
    return "created"


def _tables_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Full-content comparison for small summary tables.

    Float columns are compared with a numeric tolerance (``np.allclose``,
    NaN-aware); everything else is compared as exact strings. This is what
    the original notebooks did ad hoc for cell-count and feature-integrity
    tables -- generalized here into one reusable check.
    """
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    for col in a.columns:
        if pd.api.types.is_float_dtype(a[col]) or pd.api.types.is_float_dtype(b[col]):
            if not np.allclose(
                a[col].to_numpy(dtype=float),
                b[col].to_numpy(dtype=float),
                equal_nan=True,
            ):
                return False
        elif not a[col].astype(str).equals(b[col].astype(str)):
            return False
    return True


def resolve_resume_stage(
    stage_paths: Sequence[tuple[str, Path]],
) -> tuple[str | None, Path | None]:
    """Given ``(stage_name, path)`` pairs ordered from most- to least-advanced,
    return the first stage whose checkpoint file exists on disk, or
    ``(None, None)`` if none do.

    Generalizes the resume ladder used by NB02 (six hardcoded ``elif``
    branches checking each checkpoint parquet in turn) into a single
    reusable, order-driven lookup.
    """
    for stage_name, path in stage_paths:
        if Path(path).exists():
            return stage_name, Path(path)
    return None, None


def validate_checkpoint_df(
    df: pd.DataFrame,
    label: str,
    feature_cols: Sequence[str],
    *,
    strict: bool = False,
) -> None:
    """Validate a resumed checkpoint DataFrame before reuse.

    ``strict=True`` raises if any feature value is missing/infinite (for
    stages -- e.g. "normalized", "feature_selected" -- where that should
    already have been resolved); ``strict=False`` allows it silently, since
    downstream cleaning steps haven't run yet.
    """
    if df.empty:
        raise ValueError(f"Checkpoint '{label}' loaded an empty DataFrame.")
    if not feature_cols:
        raise ValueError(f"Checkpoint '{label}' has no feature columns.")

    feature_values = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_invalid = int(feature_values.isna().sum().sum())
    if n_invalid > 0 and strict:
        raise ValueError(
            f"Checkpoint '{label}' contains {n_invalid} missing or infinite "
            "feature values. Delete or regenerate this checkpoint."
        )

    unnamed_columns = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_columns:
        raise ValueError(f"Checkpoint '{label}' contains exported index columns: {unnamed_columns}")


def write_summary_table_protected(df: pd.DataFrame, path: Path, *, overwrite: bool) -> str:
    """Write a small summary table (cell counts, feature-integrity report, ...)
    with a full-content protected-overwrite check.

    Unlike the parquet/CSV helpers above, this compares every value (not just
    shape/headers), since these tables are small enough that a full
    comparison is cheap and a content-level guarantee is worth having.
    Returns one of ``"created"``, ``"unchanged"``, ``"replaced"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path)
        if _tables_equal(existing, df):
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                "Existing table differs from the freshly computed result and "
                f"was not overwritten: {path}"
            )
        df.to_csv(path, index=False)
        return "replaced"

    df.to_csv(path, index=False)
    return "created"
