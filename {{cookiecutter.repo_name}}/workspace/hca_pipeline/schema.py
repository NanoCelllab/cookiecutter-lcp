"""Parquet schema consistency checks, for comparing many students' data in DuckDB.

DuckDB's ``read_parquet('.../*.parquet')`` over many experiment folders
only works cleanly if every file actually shares a schema -- or at least
doesn't contain columns that look like the *same* thing under different
names. Two different things need checking here, and this module keeps
them separate:

- **Structural hygiene** (:func:`find_duplicate_like_columns`): does this
  file have a column that looks like a merge/duplicate-column artifact
  (``Metadata_Plate.1``, ``Foo_x``/``Foo_y``, ...)? This should never
  happen, in anyone's data, and is checked unconditionally. It's exactly
  what caused NB01's raw single-cell exports to carry duplicated
  ``Metadata_Plate``/``Well``/``Site``/``QCFlag`` columns (one set per
  merged CellProfiler compartment table) -- :func:`~hca_pipeline.metadata.dedupe_meta`
  is the fix; this module is the test that would have caught it.
- **Schema comparison** (:func:`describe_schema`, :func:`compare_schemas`):
  do several files (e.g. many students' ``per_well_aggregated.parquet``)
  actually share the same columns and dtypes? Feature columns are
  *expected* to differ somewhat between experiments (different channels,
  different CellProfiler modules run, different feature-selection
  outcomes) -- this reports the difference rather than treating it as a
  failure, so a human can judge whether it's expected. ``per_well_aggregated.parquet``
  (pre-feature-selection) is the recommended checkpoint for cross-student
  comparison precisely because it's the least likely to have diverged yet;
  ``per_well_features_selected.parquet`` is expected to differ per
  experiment by design and isn't a good target for this comparison.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Matches a duplicated column's suffix, from either of two sources: an
# explicit pandas ``merge`` of two frames that both carry the same column
# (``Foo_x``/``Foo_y``), or a source file whose header already repeated a
# column name outright, which pandas/pyarrow re-label as ``Foo.1``, ``Foo.2``,
# ... on read to keep names unique. See hca_pipeline.metadata.dedupe_meta for
# the underlying story (CellProfiler per-compartment metadata duplication).
_DUPLICATE_SUFFIX_RE = re.compile(r"^(.+?)(?:_[xy]\d*|(?:\.\d+)+)$")

# Every checkpoint should have these, with a dtype pyarrow reports as one of
# these substrings (case-insensitive) -- Concentration/Cell_Type are left out
# since ExperimentConfig already treats those as optional, per-experiment axes.
DEFAULT_REQUIRED_METADATA_DTYPES: dict[str, str] = {
    "Metadata_Plate": "string",
    "Metadata_Well": "string",
    "Metadata_Treatment": "string",
}


def find_duplicate_like_columns(columns) -> dict[str, list[str]]:
    """Group columns whose name looks like a duplicate/merge artifact.

    Returns ``{base_name: [suffixed_variant, ...]}`` for every base name
    that has more than one matching column -- e.g.
    ``{"Metadata_Plate": ["Metadata_Plate.1", "Metadata_Plate.2"]}``. An
    empty dict means no suspicious duplicates were found.
    """
    columns = list(columns)
    column_set = set(columns)
    groups: dict[str, list[str]] = {}
    for col in columns:
        match = _DUPLICATE_SUFFIX_RE.match(col)
        if not match:
            continue
        base = match.group(1)
        if base in column_set:
            groups.setdefault(base, []).append(col)
    return groups


def describe_schema(path: Path) -> pd.DataFrame:
    """Column name + dtype for a Parquet file, without loading the data."""
    schema = pq.ParquetFile(path).schema_arrow
    return pd.DataFrame({"column": schema.names, "dtype": [str(t) for t in schema.types]})


def validate_checkpoint(
    path: Path,
    *,
    required_metadata_dtypes: dict[str, str] | None = DEFAULT_REQUIRED_METADATA_DTYPES,
) -> list[str]:
    """Structural checks that should hold for every experiment's checkpoint file.

    Returns a list of human-readable issues; an empty list means the file
    passed. Deliberately does not check *which* feature columns are present
    -- that's expected to vary between experiments (see the module
    docstring) and isn't a structural problem.
    """
    schema_df = describe_schema(path)
    issues: list[str] = []

    duplicates = find_duplicate_like_columns(schema_df["column"])
    for base, variants in duplicates.items():
        issues.append(f"Duplicate-looking columns for {base!r}: {variants}")

    is_metadata = schema_df["column"].str.startswith("Metadata_")
    feature_dtypes = schema_df.loc[~is_metadata, "dtype"]
    # bool is a legitimate, DuckDB-friendly non-metadata dtype (e.g. NB07's
    # recovery-axis flags) -- only string/object columns are a real problem.
    non_numeric = feature_dtypes[~feature_dtypes.str.contains(r"int|float|double|bool", case=False, regex=True)]
    if len(non_numeric):
        issues.append(f"{len(non_numeric)} non-metadata column(s) are not numeric (e.g. {non_numeric.iloc[0]!r})")

    if required_metadata_dtypes:
        dtype_by_column = dict(zip(schema_df["column"], schema_df["dtype"]))
        for column, expected_dtype in required_metadata_dtypes.items():
            if column not in dtype_by_column:
                issues.append(f"Missing required metadata column: {column!r}")
            elif expected_dtype.lower() not in dtype_by_column[column].lower():
                issues.append(
                    f"{column!r} has dtype {dtype_by_column[column]!r}, expected something matching {expected_dtype!r}"
                )

    return issues


def compare_schemas(paths: dict[str, Path]) -> pd.DataFrame:
    """Wide column x file dtype comparison across several files.

    One row per column (union across every file), one column per label in
    *paths* (e.g. one per student/experiment) -- makes it visually obvious
    which files are structurally consistent and which aren't. A cell reads
    ``"—"`` when that file doesn't have that column at all.
    """
    dtype_by_column_per_file = {label: dict(zip((d := describe_schema(path))["column"], d["dtype"])) for label, path in paths.items()}
    all_columns = sorted(set().union(*(d.keys() for d in dtype_by_column_per_file.values())))
    comparison = pd.DataFrame(
        {label: [dtype_by_column_per_file[label].get(col, "—") for col in all_columns] for label in paths},
        index=pd.Index(all_columns, name="column"),
    )
    return comparison


def schema_fingerprint(path: Path) -> str:
    """Short, stable hash of a file's (column, dtype) pairs.

    Cheap to stash in a notebook's ``provenance.json`` so a later run (or a
    different student's run of the same notebook) can tell at a glance
    whether the schema actually changed, without re-describing the file.
    """
    schema_df = describe_schema(path).sort_values("column")
    payload = "|".join(f"{c}:{t}" for c, t in zip(schema_df["column"], schema_df["dtype"]))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Parquet files to check")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show a wide column x file dtype comparison instead of per-file validation",
    )
    args = parser.parse_args()

    if args.compare:
        table = compare_schemas({str(p): p for p in args.paths})
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(table)
        return

    exit_code = 0
    for path in args.paths:
        issues = validate_checkpoint(path)
        if issues:
            exit_code = 1
            print(f"✗ {path}")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"✓ {path}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
