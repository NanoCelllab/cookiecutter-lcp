"""Canonical feature-taxonomy classifier for CellProfiler-style LCP features.

Two notebooks in this repository independently grew their own version of this
classifier:

- ``04_phenotypic_fingerprints.ipynb`` (``classify_feature`` /
  ``infer_feature_family`` / ``infer_compartment`` / ``infer_channels``): the
  richer of the two, with compartment, channel, and biological-signature
  buckets in addition to measurement family.
- ``06_single_cell_analysis.ipynb`` (``classify_recovery_feature``): a
  stripped-down, family-only classifier with a slightly different bucket
  vocabulary and matching rule (see ``infer_feature_family`` below).

This module merges both into one classifier. Channel names and compartment
prefixes are function parameters rather than hardcoded literals, so a
dataset that only stains Hoechst (or that names compartments differently)
does not silently collapse every feature into "Other".
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Measurement family vocabulary
# ---------------------------------------------------------------------------
# Superset of the two notebooks' family lists. NB06 never introduced a family
# absent from NB04 (it simply omitted "Neighbors" and "Number"), so NB04's
# list is used as-is.
KNOWN_FAMILIES: tuple[str, ...] = (
    "RadialDistribution",
    "Granularity",
    "Correlation",
    "AreaShape",
    "Intensity",
    "Texture",
    "Location",
    "Neighbors",
    "Children",
    "Number",
)

DEFAULT_FAMILY_LABELS: Mapping[str, str] = {
    "RadialDistribution": "Radial distribution",
    "Granularity": "Granularity",
    "Correlation": "Correlation",
    "AreaShape": "Area & shape",
    "Intensity": "Intensity",
    "Texture": "Texture",
    "Location": "Location",
    "Neighbors": "Neighbors",
    "Children": "Object count",
    "Number": "Object count",
}

DEFAULT_COMPARTMENT_PREFIXES: tuple[str, ...] = (
    "Cells_",
    "Cytoplasm_",
    "Nuclei_",
    "Vesicles_",
)


def infer_feature_family(
    feature_name: str,
    *,
    families: Sequence[str] = KNOWN_FAMILIES,
    family_labels: Mapping[str, str] = DEFAULT_FAMILY_LABELS,
) -> str:
    """Infer the CellProfiler measurement family from a feature name.

    Matching rule: NB04 required the family token to be flanked by
    underscores (``_Texture_``) or to open the name (``Texture_...``). NB06
    only checked the flanked form. NB04's rule is a strict superset, so it is
    kept as the primary rule.

    Fallback: NB04 additionally special-cased the literal substring
    ``"cell_count"``; NB06 instead matched any ``"_Count"`` or ``"Children_"``
    substring (e.g. ``Cells_Children_Vesicles_Count``), which is broader.
    Both are reconciled here as a single case-insensitive ``"count"``
    substring check, which is a superset of both notebooks' fallbacks.
    """
    for family in families:
        if f"_{family}_" in feature_name or feature_name.startswith(f"{family}_"):
            return family_labels.get(family, family)

    if "count" in feature_name.lower():
        return family_labels.get("Children", "Object count")

    return "Other"


def infer_compartment(
    feature_name: str,
    *,
    compartment_prefixes: tuple[str, ...] = DEFAULT_COMPARTMENT_PREFIXES,
) -> str:
    """Infer the biological compartment from a configurable prefix list.

    ``compartment_prefixes`` is checked in order and the first match wins, so
    callers that need to disambiguate nested/child objects (e.g. NB04's
    ``"Cells_Mean_Vesicles_"`` -> ``"Vesicles"``) should list the more
    specific prefix before the more general one. The label for a matched
    prefix is derived from its final underscore-delimited token, so
    ``"Cells_Mean_Vesicles_"`` yields ``"Vesicles"`` and ``"Cells_"`` yields
    ``"Cells"`` without needing a separate label mapping.
    """
    for prefix in compartment_prefixes:
        if feature_name.startswith(prefix):
            stripped = prefix.rstrip("_")
            return stripped.rsplit("_", 1)[-1] if stripped else "Other"

    return "Other"


def infer_channels(
    feature_name: str,
    *,
    channels: Sequence[str],
) -> tuple[str, ...]:
    """Infer all configured image channels referenced by a feature name.

    NB04 hardcoded a pairwise "prefer the more specific label" rule (drop
    generic ``"GFP"`` when the more specific ``"AOGFP"`` is also present, same
    for ``"PI"``/``"AOPI"``). That is generalized here: for any two matched
    channels where one is a suffix of the other (e.g. ``"AOGFP"`` ends with
    ``"GFP"``), the shorter/generic one is dropped in favor of the longer,
    more specific one. This works for any channel vocabulary, not just the
    AOGFP/AOPI pair NB04 special-cased.
    """
    tokens = re.split(r"[_\-]", feature_name)
    found = [
        channel
        for channel in channels
        if channel in tokens or channel in feature_name
    ]
    found = list(dict.fromkeys(found))

    redundant: set[str] = set()
    for longer in found:
        for shorter in found:
            if longer != shorter and shorter not in redundant and longer.endswith(shorter):
                redundant.add(shorter)

    return tuple(channel for channel in found if channel not in redundant)


def format_channel_group(channels: Sequence[str]) -> str:
    """Create a concise label for one or more channels."""
    if not channels:
        return "Channel-independent"

    if len(channels) == 1:
        return channels[0]

    return " × ".join(sorted(channels))


def infer_biological_signature(
    *,
    family: str,
    compartment: str,
    channels: Sequence[str],
    ordered_channels: Sequence[str],
) -> str:
    """Map a classified feature to a broad biological interpretation bucket.

    NB04 hardcoded two channel-specific branches ("AOGFP signal
    organization", "AOPI signal organization"). Rather than hardcoding those
    channel names, this walks ``ordered_channels`` (the same configuration
    list passed to ``infer_channels``) in the caller-supplied priority order
    and returns "<channel> signal organization" for the first channel that is
    both present on this feature and paired with a signal-bearing family.
    This reproduces NB04's AOGFP-then-AOPI behavior when
    ``ordered_channels == ["AOGFP", "AOPI", ...]`` while also giving sensible,
    non-"Other" output for single-channel (e.g. Hoechst-only) datasets.
    """
    channel_set = set(channels)
    signal_families = {"Texture", "Granularity", "Intensity"}

    if compartment == "Vesicles":
        return "Vesicular / lysosomal organization"

    if compartment == "Nucleoli":
        return "Nucleolar organization"

    if compartment == "Nuclei":
        if family in signal_families:
            return "Nuclear organization"
        if family == "Area & shape":
            return "Nuclear morphology"

    for channel in ordered_channels:
        if channel in channel_set and family in signal_families:
            return f"{channel} signal organization"

    if family == "Area & shape":
        return "Cell morphology"

    if family == "Object count":
        return "Object abundance"

    if family in {"Correlation", "Radial distribution", "Location", "Neighbors"}:
        return "Spatial organization"

    if family == "Texture":
        return "Texture complexity"

    if family == "Intensity":
        return "Signal intensity"

    return "Other"


def classify_feature(
    feature_name: str,
    *,
    channels: Sequence[str],
    compartment_prefixes: tuple[str, ...] = DEFAULT_COMPARTMENT_PREFIXES,
    families: Sequence[str] = KNOWN_FAMILIES,
    family_labels: Mapping[str, str] = DEFAULT_FAMILY_LABELS,
) -> dict[str, str]:
    """Classify one CellProfiler feature name across every taxonomy axis.

    ``channels`` and ``compartment_prefixes`` are the dataset-specific
    configuration inputs: they are never hardcoded inside this function, so
    a dataset with a different channel or compartment vocabulary is
    classified correctly instead of falling back to "Other".
    """
    family = infer_feature_family(
        feature_name, families=families, family_labels=family_labels
    )
    compartment = infer_compartment(
        feature_name, compartment_prefixes=compartment_prefixes
    )
    feature_channels = infer_channels(feature_name, channels=channels)
    channel_group = format_channel_group(feature_channels)

    return {
        "feature": feature_name,
        "feature_family": family,
        "compartment": compartment,
        "channel": channel_group,
        "compartment_x_family": f"{compartment} | {family}",
        "channel_x_family": f"{channel_group} | {family}",
        "biological_signature": infer_biological_signature(
            family=family,
            compartment=compartment,
            channels=feature_channels,
            ordered_channels=channels,
        ),
    }


def build_taxonomy_table(
    feature_names: Sequence[str],
    *,
    channels: Sequence[str],
    compartment_prefixes: tuple[str, ...] = DEFAULT_COMPARTMENT_PREFIXES,
    families: Sequence[str] = KNOWN_FAMILIES,
    family_labels: Mapping[str, str] = DEFAULT_FAMILY_LABELS,
) -> pd.DataFrame:
    """Classify a full feature-column list into a taxonomy table.

    Returns a ``pandas.DataFrame`` with one row per feature and the columns
    produced by :func:`classify_feature`. This is what NB04's "taxonomy
    coverage overview" section consumed (built there as
    ``pd.DataFrame([classify_feature(f) for f in feature_cols])``).
    """
    records = [
        classify_feature(
            feature_name,
            channels=channels,
            compartment_prefixes=compartment_prefixes,
            families=families,
            family_labels=family_labels,
        )
        for feature_name in feature_names
    ]
    return pd.DataFrame(records)
