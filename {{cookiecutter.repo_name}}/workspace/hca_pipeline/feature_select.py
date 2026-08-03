"""Feature-column detection and selection shared by every pipeline notebook."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .config import COMPARTMENT_PREFIXES

TECHNICAL_SUFFIXES = ("_ImageNumber", "_ObjectNumber")

DEFAULT_FEATURE_SELECT_OPERATIONS = (
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
)


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns: those starting with a compartment prefix
    (:data:`hca_pipeline.config.COMPARTMENT_PREFIXES`) and not starting with
    ``Metadata_``.

    Also excludes columns ending with technical identifiers (``_ImageNumber``,
    ``_ObjectNumber``) that should never be treated as features.

    Falls back to all non-``Metadata_`` numeric columns if no
    compartment-prefixed columns are found.
    """
    feat_cols = [
        c
        for c in df.columns
        if c.startswith(COMPARTMENT_PREFIXES)
        and not c.startswith("Metadata_")
        and not c.endswith(TECHNICAL_SUFFIXES)
    ]
    if not feat_cols:
        feat_cols = [
            c
            for c in df.columns
            if not c.startswith("Metadata_")
            and not c.endswith(TECHNICAL_SUFFIXES)
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
