"""Feature-column detection and selection shared by every pipeline notebook."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .config import COMPARTMENT_PREFIXES

TECHNICAL_IDENTIFIER_PATTERNS = (
    "ImageNumber",
    "ObjectNumber",
    "Number_Object_Number",
    "Parent_",
)

DEFAULT_FEATURE_SELECT_OPERATIONS = (
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
)


def is_technical_identifier_column(column: str) -> bool:
    """Return whether *column* stores object identity/linkage rather than phenotype.

    The match is intentionally not suffix-only: aggregation can produce names
    such as ``Cells_Mean_Vesicles_Number_Object_Number`` while preserving the
    technical identifier token in the middle of the resulting column name.
    """
    column = str(column)
    return any(pattern in column for pattern in TECHNICAL_IDENTIFIER_PATTERNS)


def technical_identifier_columns(df: pd.DataFrame) -> list[str]:
    """List CellProfiler identity/linkage columns present in *df*."""
    return [c for c in df.columns if is_technical_identifier_column(c)]


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns: those starting with a compartment prefix
    (:data:`hca_pipeline.config.COMPARTMENT_PREFIXES`) and not starting with
    ``Metadata_``.

    Also excludes CellProfiler object-identity and parent-linkage columns,
    including identifier tokens preserved inside aggregated column names.

    Falls back to all non-``Metadata_`` numeric columns if no
    compartment-prefixed columns are found.
    """
    feat_cols = [
        c
        for c in df.columns
        if c.startswith(COMPARTMENT_PREFIXES)
        and not c.startswith("Metadata_")
        and not is_technical_identifier_column(c)
    ]
    if not feat_cols:
        feat_cols = [
            c
            for c in df.columns
            if not c.startswith("Metadata_")
            and not is_technical_identifier_column(c)
            and df[c].dtype in ("float64", "float32", "int64", "int32")
        ]
    return feat_cols


def select_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | str = "infer",
    operations: Sequence[str] = DEFAULT_FEATURE_SELECT_OPERATIONS,
) -> pd.DataFrame:
    """Run pycytominer ``feature_select`` (variance/correlation/NA/blocklist).

    ``feature_cols`` may be an explicit list or the pycytominer sentinel
    ``"infer"``. Thin wrapper kept here (rather than called inline from a
    notebook) so the exact operation list is a single source of truth.
    """
    from pycytominer import feature_select as pc_feature_select

    return pc_feature_select(
        profiles=df,
        features=feature_cols,
        operation=list(operations),
    )
