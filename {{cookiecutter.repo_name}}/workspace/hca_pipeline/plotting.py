"""Generic plotting helpers shared across LCP analysis notebooks.

Ports the scatter-panel helper from
``03_phenotypic_profiling_parallel_spaces.ipynb`` and the fingerprint-heatmap
/ radar-plot toolkit from ``04_phenotypic_fingerprints.ipynb``. All functions
are parameterized by the data, labels, and paths passed in rather than by any
one experiment's channel or treatment names.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure_protected(
    figure: plt.Figure,
    path: Path,
    *,
    overwrite: bool,
    **savefig_kwargs,
) -> str:
    """Save a figure while honoring the pipeline overwrite policy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return "preserved"
    status = "replaced" if path.exists() else "created"
    figure.savefig(path, **savefig_kwargs)
    return status

# ---------------------------------------------------------------------------
# Generic scatter panel (e.g. PCA/UMAP embeddings colored by metadata)
# ---------------------------------------------------------------------------


def categorical_palette(n_categories: int) -> list:
    """Return distinct categorical colors without repeating at 10 groups."""
    if n_categories <= 0:
        return []
    if n_categories <= 10:
        return list(plt.get_cmap("tab10").colors[:n_categories])
    if n_categories <= 20:
        return list(plt.get_cmap("tab20").colors[:n_categories])
    cmap = plt.get_cmap("turbo").resampled(n_categories)
    return [cmap(index) for index in range(n_categories)]


def add_categorical_legend(ax: plt.Axes, n_categories: int, **kwargs) -> None:
    """Place small legends inside and large legends outside the plotting area."""
    defaults = {"fontsize": 7, "markerscale": 0.8, "frameon": True}
    defaults.update(kwargs)
    if n_categories <= 10:
        ax.legend(loc="best", **defaults)
    else:
        columns = max(1, math.ceil(n_categories / 20))
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
            ncol=columns, **defaults,
        )


def scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: Sequence,
    title: str,
    *,
    continuous: bool = False,
    cmap: str = "viridis",
) -> None:
    """Scatter ``x``/``y`` on ``ax``, colored by categorical or continuous labels."""
    if continuous:
        sc = ax.scatter(
            x, y, c=labels, cmap=cmap, s=60, alpha=0.85, edgecolors="k", linewidths=0.4
        )
        plt.colorbar(sc, ax=ax, label="Concentration")
    else:
        unique = sorted(set(labels))
        palette = categorical_palette(len(unique))
        for lbl, color in zip(unique, palette):
            mask = [item == lbl for item in labels]
            ax.scatter(
                np.asarray(x)[mask],
                np.asarray(y)[mask],
                label=lbl,
                color=color,
                s=60,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.4,
            )
        add_categorical_legend(ax, len(unique))
    ax.set_title(title)


# ---------------------------------------------------------------------------
# Fingerprint heatmap
# ---------------------------------------------------------------------------


def plot_fingerprint_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    output_path: Path,
    diverging: bool = False,
    fixed_range: tuple[float, float] | None = None,
    dpi: int = 180,
    overwrite: bool = True,
) -> plt.Figure | None:
    """Render and save a category x condition fingerprint heatmap.

    Returns the saved (closed) ``Figure`` so callers can display it inline
    in a notebook, or ``None`` if ``matrix`` is empty and nothing was drawn.
    """
    if matrix.empty:
        return None
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        image = plt.imread(output_path)
        fig, ax = plt.subplots(figsize=(max(9, image.shape[1] / 180), max(4.5, image.shape[0] / 180)))
        ax.imshow(image)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.close(fig)
        return fig

    n_rows, n_columns = matrix.shape

    figure_width = max(9, 0.65 * n_columns + 4)
    figure_height = max(4.5, 0.42 * n_rows + 2.5)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    values = matrix.to_numpy(dtype=float)

    image_kwargs: dict[str, object] = {"aspect": "auto", "interpolation": "nearest"}

    if diverging:
        maximum = np.nanmax(np.abs(values))
        if not np.isfinite(maximum):
            maximum = 1.0
        image_kwargs.update({"cmap": "coolwarm", "vmin": -maximum, "vmax": maximum})
    elif fixed_range is not None:
        image_kwargs.update({"vmin": fixed_range[0], "vmax": fixed_range[1]})

    image = ax.imshow(values, **image_kwargs)

    ax.grid(False)
    ax.set_xticks(np.arange(n_columns))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Treatment × concentration")
    ax.set_ylabel("Feature group")

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label(colorbar_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Radar-plot toolkit
# ---------------------------------------------------------------------------

DEFAULT_RADAR_ABBREVIATIONS: Mapping[str, str] = {
    "Intensity": "Int.",
    "Texture": "Tex.",
    "Area & shape": "Shape",
    "Granularity": "Gran.",
    "Correlation": "Corr.",
    "Radial distribution": "Radial",
    "Location": "Loc.",
    "Object count": "Count",
    "Neighbors": "Neigh.",
    "Cells": "Cells",
    "Cytoplasm": "Cyto.",
    "Nuclei": "Nuclei",
    "Nucleoli": "Nucl.",
    "Vesicles": "Ves.",
    "Channel-independent": "No channel",
}


def abbreviate_radar_label(
    label: str,
    *,
    abbreviations: Mapping[str, str] = DEFAULT_RADAR_ABBREVIATIONS,
    max_length: int = 22,
) -> str:
    """Shorten a long radar-axis label without losing its meaning."""
    if label in abbreviations:
        return abbreviations[label]

    parts = [part.strip() for part in label.split("|")]
    abbreviated_parts = [abbreviations.get(part, part) for part in parts]
    compact = " · ".join(abbreviated_parts)

    if len(compact) > max_length:
        compact = compact[: max_length - 2] + "…"

    return compact


def radar_angles(n_categories: int) -> tuple[np.ndarray, np.ndarray]:
    """Return open and closed angular coordinates for a radar plot."""
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    return angles, closed_angles


def style_radar_axis(
    ax: plt.Axes,
    categories: Sequence[str],
    *,
    radial_max: float,
    title: str,
    abbreviations: Mapping[str, str] = DEFAULT_RADAR_ABBREVIATIONS,
) -> None:
    """Apply consistent styling to a polar radar axis."""
    angles, _ = radar_angles(len(categories))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(
        [abbreviate_radar_label(category, abbreviations=abbreviations) for category in categories],
        fontsize=8,
    )

    ax.set_ylim(0, radial_max)

    radial_ticks = np.linspace(0, radial_max, 4)[1:]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{value:.2g}" for value in radial_ticks], fontsize=7)

    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.spines["polar"].set_visible(False)
    ax.set_title(title, fontsize=10, pad=20)


def draw_radar_trace(
    ax: plt.Axes,
    values: Sequence[float],
    *,
    label: str | None = None,
    linewidth: float = 1.5,
    fill_alpha: float = 0.18,
) -> None:
    """Draw one radar trace using Matplotlib's default color cycle."""
    values = np.asarray(values, dtype=float)
    _, closed_angles = radar_angles(len(values))
    closed_values = np.concatenate([values, values[:1]])

    line = ax.plot(closed_angles, closed_values, linewidth=linewidth, label=label)[0]
    ax.fill(closed_angles, closed_values, alpha=fill_alpha, color=line.get_color())


def plot_condition_radar_grid(
    matrix: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    max_columns: int = 4,
    dpi: int = 180,
    overwrite: bool = True,
) -> plt.Figure | None:
    """Create one small radar per experimental condition (matrix column).

    Returns the saved (closed) ``Figure`` so callers can display it inline
    in a notebook, or ``None`` if ``matrix`` is empty and nothing was drawn.
    """
    if matrix.empty:
        return None
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        image = plt.imread(output_path)
        fig, ax = plt.subplots(figsize=(max(8, image.shape[1] / 180), max(5, image.shape[0] / 180)))
        ax.imshow(image)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.close(fig)
        return fig

    categories = matrix.index.tolist()
    conditions = matrix.columns.tolist()

    n_panels = len(conditions)
    n_columns = min(max_columns, max(1, n_panels))
    n_rows = math.ceil(n_panels / n_columns)

    radial_max = float(np.nanmax(matrix.to_numpy(dtype=float)))
    if not np.isfinite(radial_max) or radial_max <= 0:
        radial_max = 1.0

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 4.2 * n_rows),
        subplot_kw={"polar": True},
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, condition in zip(axes_flat, conditions):
        values = matrix[condition].fillna(0).to_numpy(dtype=float)
        style_radar_axis(ax, categories, radial_max=radial_max, title=condition)
        draw_radar_trace(ax, values)

    for ax in axes_flat[len(conditions) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=15, y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_dose_overlay_radars(
    matrix: pd.DataFrame,
    *,
    condition_table: pd.DataFrame,
    treatment_column: str,
    concentration_column: str,
    title: str,
    output_path: Path,
    condition_column: str = "Metadata_Condition",
    concentration_label_column: str = "Metadata_Concentration_Label",
    max_columns: int = 4,
    dpi: int = 180,
    overwrite: bool = True,
) -> plt.Figure | None:
    """Create one radar per treatment with all its concentrations overlaid.

    ``condition_table`` must have one row per condition with, at minimum,
    ``condition_column``, ``treatment_column``, ``concentration_column``, and
    ``concentration_label_column`` (matching ``stats.calculate_*_effects``).

    Returns the saved (closed) ``Figure`` so callers can display it inline
    in a notebook, or ``None`` if ``matrix`` is empty and nothing was drawn.
    """
    if matrix.empty:
        return None
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        image = plt.imread(output_path)
        fig, ax = plt.subplots(figsize=(max(8, image.shape[1] / 180), max(5, image.shape[0] / 180)))
        ax.imshow(image)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.close(fig)
        return fig

    categories = matrix.index.tolist()

    treatment_conditions = condition_table.loc[
        condition_table[condition_column].isin(matrix.columns)
    ].copy()

    treatments = treatment_conditions[treatment_column].drop_duplicates().tolist()

    n_panels = len(treatments)
    n_columns = min(max_columns, max(1, n_panels))
    n_rows = math.ceil(n_panels / n_columns)

    radial_max = float(np.nanmax(matrix.to_numpy(dtype=float)))
    if not np.isfinite(radial_max) or radial_max <= 0:
        radial_max = 1.0

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.5 * n_columns, 4.5 * n_rows),
        subplot_kw={"polar": True},
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, treatment in zip(axes_flat, treatments):
        treatment_rows = treatment_conditions.loc[
            treatment_conditions[treatment_column].eq(treatment)
        ].sort_values(concentration_column, kind="stable")

        style_radar_axis(ax, categories, radial_max=radial_max, title=str(treatment))

        for _, row in treatment_rows.iterrows():
            condition = row[condition_column]
            if condition not in matrix.columns:
                continue

            values = matrix[condition].fillna(0).to_numpy(dtype=float)
            draw_radar_trace(ax, values, label=row[concentration_label_column])

        ax.legend(
            title="Concentration",
            loc="upper right",
            bbox_to_anchor=(1.30, 1.15),
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )

    for ax in axes_flat[len(treatments) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=15, y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig
