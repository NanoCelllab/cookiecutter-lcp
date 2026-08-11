"""Core metadata normalization shared by every pipeline notebook.

Includes platemap reading/annotation, ported from Sections 4-4b of
``02_aggregate_normalize_featureselect_cv.ipynb``. The source notebook
required every platemap CSV to have a ``Concentration`` column -- a hard
dose-axis assumption that breaks non-dose experiments (genetic
perturbations, single-concentration screens). Here ``Concentration`` and
``Cell_Type`` are optional, propagated only when present, the same way the
source notebook already treated ``Time_Point`` as optional.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def norm_well(x: str) -> str:
    """Normalise a well ID to zero-padded format: A1 -> A01, B12 -> B12."""
    s = str(x).strip().upper()
    m = re.match(r"^([A-H])0?([1-9]|1[0-2])$", s)
    return f"{m.group(1)}{int(m.group(2)):02d}" if m else s


def ensure_core_metadata(
    df: pd.DataFrame,
    plate_value: Optional[str] = None,
) -> pd.DataFrame:
    """Rename common CellProfiler metadata columns to the standard
    ``Metadata_Plate`` / ``Metadata_Well`` / ``Metadata_Site`` convention and
    normalise well IDs.

    Parameters
    ----------
    df : DataFrame with raw CellProfiler column names.
    plate_value : If ``Metadata_Plate`` is not present, fill it with this value.
    """
    rename_map = {
        "Image_Metadata_Plate": "Metadata_Plate",
        "Image_Metadata_Well": "Metadata_Well",
        "Image_Metadata_Site": "Metadata_Site",
        "Site": "Metadata_Site",
        "Well": "Metadata_Well",
        "Plate": "Metadata_Plate",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "Metadata_Plate" not in df.columns and plate_value is not None:
        df["Metadata_Plate"] = plate_value
    if "Metadata_Well" in df.columns:
        df["Metadata_Well"] = df["Metadata_Well"].map(norm_well)
    return df


# Matches a duplicated Metadata_* column's suffix, from either of two
# sources: an explicit pandas ``merge`` of two frames that both carry the
# same metadata column (``Metadata_Plate_x``/``Metadata_Plate_y``), or a
# source CSV whose header already repeated an entire metadata block --
# typical of CellProfiler exports that horizontally merge per-compartment
# (Nuclei/Cytoplasm/Cells) tables without dropping each one's own copy of
# Plate/Well/Site/QCFlag. pandas silently re-labels the latter as
# ``Metadata_Plate.1``, ``Metadata_Plate.2``, ... on read to keep column
# names unique; left alone, those ride along into every downstream file as
# distinct-looking but fully redundant columns.
_META_DUPLICATE_SUFFIX_RE = re.compile(r"^(Metadata_.+?)(?:_[xy]\d*|(?:\.\d+)+)$")
CELL_COUNT_METADATA_COLUMN = "Metadata_Cell_Count"


def add_cell_count_metadata(
    df: pd.DataFrame,
    *,
    plate_col: str = "Metadata_Plate",
    well_col: str = "Metadata_Well",
    output_col: str = CELL_COUNT_METADATA_COLUMN,
) -> pd.DataFrame:
    """Attach the number of single-cell rows in each plate/well as metadata.

    The returned column deliberately starts with ``Metadata_`` so feature
    inference, normalization, and feature selection never treat cell count as
    a morphological measurement. The count is repeated on every single-cell
    row and therefore survives a later per-well groupby as an invariant
    stratum value.
    """
    missing = [column for column in (plate_col, well_col) if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot calculate cell count metadata; missing columns: {missing}")

    result = df.copy()
    computed = result.groupby([plate_col, well_col], observed=True)[well_col].transform("size").astype("int64")
    if output_col in result.columns:
        existing = pd.to_numeric(result[output_col], errors="coerce")
        mismatched = existing.isna() | existing.ne(computed)
        if mismatched.any():
            raise ValueError(
                f"Existing {output_col!r} disagrees with the observed single-cell row count "
                f"for {int(mismatched.sum())} row(s)."
            )
    result[output_col] = computed
    return result


def dedupe_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicated Metadata_* columns into one per logical name.

    Groups every ``Metadata_*`` column by its base name (suffix stripped),
    coalesces each group by "first non-null wins" (preferring the
    unsuffixed name as the surviving column when present), and moves
    Plate/Well/Site to the front.
    """
    df = df.loc[:, ~df.columns.duplicated()].copy()

    groups: dict[str, list[str]] = {}
    for col in df.columns:
        if not col.startswith("Metadata_"):
            continue
        match = _META_DUPLICATE_SUFFIX_RE.match(col)
        base = match.group(1) if match else col
        groups.setdefault(base, []).append(col)

    for base, cols in groups.items():
        if len(cols) <= 1:
            continue
        target = base if base in cols else cols[0]
        for extra in cols:
            if extra == target:
                continue
            df[target] = df[target].where(df[target].notna(), df[extra])
            df.drop(columns=extra, inplace=True)
        if target != base:
            df = df.rename(columns={target: base})

    meta_keys = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
    meta_first = [c for c in meta_keys if c in df.columns]
    return df[meta_first + [c for c in df.columns if c not in meta_first]]


def read_barcode_platemap(index_csv: Path) -> pd.DataFrame:
    """Read ``barcode_platemap.csv``: maps each plate barcode to its platemap file."""
    idx = pd.read_csv(index_csv)
    rename_map = {"Assay_Plate_Barcode": "Metadata_Plate", "Plate_Map_Name": "filename"}
    for k, v in rename_map.items():
        if k in idx.columns and v not in idx.columns:
            idx = idx.rename(columns={k: v})
    if "plate_map_name" not in idx.columns and "filename" in idx.columns:
        idx["plate_map_name"] = idx["filename"].str.replace(".csv", "", regex=False)
    idx["filename"] = idx["filename"].astype(str).str.strip()
    idx["Metadata_Plate"] = idx["Metadata_Plate"].astype(str).str.strip()

    duplicates = idx["Metadata_Plate"].duplicated(keep=False)
    if duplicates.any():
        dup_plates = idx.loc[duplicates, "Metadata_Plate"].unique().tolist()
        raise ValueError(f"Duplicate plate barcodes found in barcode_platemap.csv: {dup_plates}")
    return idx


def read_platemap_layout(csv_path: Path) -> pd.DataFrame:
    """Read one platemap layout CSV and build its ``Metadata_*`` columns.

    Only ``well_position``, ``Treatment``, and ``Control_Type`` are strictly
    required. ``Concentration`` and ``Cell_Type`` are propagated to
    ``Metadata_Concentration``/``Metadata_Cell_Type`` only when present, and
    ``Time_Point`` (if present) is propagated to ``Metadata_Time`` -- the
    same optional-axis convention, applied consistently to every axis that
    isn't universal across experiment designs.
    """
    pm = pd.read_csv(csv_path)
    required = ["plate_map_name", "well_position", "Treatment", "Control_Type"]
    missing = [c for c in required if c not in pm.columns]
    if missing:
        raise KeyError(f"{csv_path.name}: missing columns {missing}")

    pm = pm.rename(columns={"well_position": "Metadata_Well"})
    pm["Metadata_Well"] = pm["Metadata_Well"].astype(str).map(norm_well)

    if pm["Metadata_Well"].duplicated().any():
        dup_wells = pm.loc[pm["Metadata_Well"].duplicated(), "Metadata_Well"].unique().tolist()
        raise ValueError(f"{csv_path.name} contains duplicated wells: {dup_wells}")

    pm["Metadata_Control_Type"] = pm["Control_Type"].astype(str).str.strip().str.lower()
    pm["Metadata_Treatment"] = pm["Treatment"]
    if "Concentration" in pm.columns:
        pm["Metadata_Concentration"] = pm["Concentration"]
    if "Cell_Type" in pm.columns:
        pm["Metadata_Cell_Type"] = pm["Cell_Type"]
    if "Time_Point" in pm.columns:
        pm["Metadata_Time"] = pm["Time_Point"]

    # Drop the raw, un-prefixed source columns now that each has a
    # Metadata_-prefixed counterpart (`annotate_per_plate` merges this
    # frame's columns wholesale onto the profile data -- leaving these in
    # would carry a second, un-prefixed copy of Treatment/Control_Type/
    # Cell_Type/Concentration into every downstream file). `plate_map_name`
    # has no Metadata_ counterpart and isn't used past this function; it's
    # dropped rather than promoted since nothing downstream needs it.
    return pm.drop(
        columns=["Treatment", "Control_Type", "Cell_Type", "Time_Point", "Concentration", "plate_map_name"],
        errors="ignore",
    )


def annotate_per_plate(
    df_profiles: pd.DataFrame,
    barcode_index: pd.DataFrame,
    platemap_dir: Path,
) -> pd.DataFrame:
    """Merge single-cell/well profiles with each plate's platemap layout."""
    outs = []
    for plate, sub in df_profiles.groupby("Metadata_Plate", sort=False):
        row = barcode_index.loc[barcode_index["Metadata_Plate"] == plate]
        if row.empty:
            raise KeyError(
                f"Plate '{plate}' was not found in barcode_platemap.csv. "
                "Every plate must have a matching entry."
            )
        pm_path = platemap_dir / row["filename"].iloc[0]
        if not pm_path.exists():
            raise FileNotFoundError(f"Plate '{plate}': platemap layout not found: {pm_path.name}")
        pm = read_platemap_layout(pm_path)
        merged = sub.merge(pm, on="Metadata_Well", how="left")
        outs.append(merged)
    return pd.concat(outs, axis=0, ignore_index=True)


def find_wells_missing_from_layout(
    df_annotated: pd.DataFrame,
    barcode_index: pd.DataFrame,
    platemap_dir: Path,
) -> list[tuple[str, str]]:
    """Return ``(plate, well)`` pairs present in the data but absent from the platemap layout."""
    missing_pairs = []
    for plate in df_annotated["Metadata_Plate"].dropna().unique():
        row = barcode_index.loc[barcode_index["Metadata_Plate"] == plate]
        if row.empty:
            continue
        pm_path = platemap_dir / row["filename"].iloc[0]
        if not pm_path.exists():
            continue
        pm = read_platemap_layout(pm_path)
        layout_wells = set(pm["Metadata_Well"])
        data_wells = set(df_annotated.loc[df_annotated["Metadata_Plate"] == plate, "Metadata_Well"])
        for w in sorted(data_wells - layout_wells):
            missing_pairs.append((plate, w))
    return missing_pairs


def enforce_pascalcase_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any ``Metadata_camelCase`` column to ``Metadata_PascalCase``.

    Deduplicates a rename step ("C4") that appeared twice, identically, in
    the source notebook (once after normalization, once after feature
    selection).
    """
    rename_map = {}
    for col in df.columns:
        if col.startswith("Metadata_"):
            rest = col[len("Metadata_"):]
            if rest and rest[0].islower():
                pascal = "Metadata_" + rest[0].upper() + rest[1:]
                if pascal not in df.columns:
                    rename_map[col] = pascal
    return df.rename(columns=rename_map) if rename_map else df
