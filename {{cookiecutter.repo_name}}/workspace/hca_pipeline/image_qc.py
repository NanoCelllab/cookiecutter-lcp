"""Per-image quality-control toolkit for raw microscopy images.

Ported out of ``02_aggregate_normalize_featureselect_cv.ipynb`` (cells
~55-70), where it was entangled with the profile-aggregation notebook and
used a hardcoded, machine-specific absolute path for the image root. Here
the image root is an explicit parameter (callers resolve it the same
repo-root-relative way as every other path in the pipeline), and the
channel-keyword vocabulary is a parameter rather than a module-level
constant -- the source notebook's default omitted Hoechst entirely despite
it being one of this assay's three possible channels; the default here
includes it.

Deliberately NOT ported: the source notebook's cells ~61-72, which were
one-off investigative comparisons hardcoded to specific wells (``B02``,
``B03``, ``G06``) and a single channel (``AOGFP``) from a specific
troubleshooting session -- scratch analysis, not a reusable pipeline step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_SUPPORTED_EXTENSIONS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})

DEFAULT_CHANNEL_KEYWORDS: dict[str, list[str]] = {
    "AOGFP": ["gfp", "aogfp"],
    "AOPI": ["propidium", "pi", "aopi"],
    "Hoechst": ["hoechst", "dapi"],
}


def read_image(image_path: Path) -> np.ndarray:
    """Read an image and return a two-dimensional float array."""
    import tifffile
    from PIL import Image

    suffix = image_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image = tifffile.imread(image_path)
    else:
        image = np.asarray(Image.open(image_path))

    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[-1] in {3, 4}:
            image = image[..., :3].mean(axis=-1)
        else:
            image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, but found shape {image.shape}")

    return image.astype(np.float64)


def infer_channel(filename: str, channel_keywords: Mapping[str, Sequence[str]]) -> str:
    """Infer the fluorescence channel from the filename via keyword matching."""
    filename_lower = filename.lower()
    for channel, keywords in channel_keywords.items():
        if any(keyword in filename_lower for keyword in keywords):
            return channel
    return "Unknown"


def infer_well(filename: str) -> str | None:
    """Infer a well identifier (A1 -> A01 style) from the filename."""
    import re

    match = re.search(r"(?<![A-Za-z0-9])([A-Ha-h])0?([1-9]|1[0-2])(?!\d)", filename)
    if match is None:
        match = re.match(r"([A-Ha-h])0?([1-9]|1[0-2])", filename)
    if match is None:
        return None

    row = match.group(1).upper()
    column = int(match.group(2))
    return f"{row}{column:02d}"


def infer_site(filename: str, channel: str, channel_keywords: Mapping[str, Sequence[str]]) -> int | None:
    """Infer the site number from numeric tokens immediately before the channel keyword."""
    import re

    stem = Path(filename).stem
    channel_positions = []
    for keyword in channel_keywords.get(channel, []):
        position = stem.lower().find(keyword.lower())
        if position >= 0:
            channel_positions.append(position)

    prefix = stem[: min(channel_positions)] if channel_positions else stem
    numbers = re.findall(r"\d+", prefix)
    return int(numbers[-1]) if numbers else None


def infer_plate(image_path: Path) -> str:
    """Infer the plate identifier from the parent folder name.

    Adapt this if the plate barcode appears elsewhere in your folder
    structure or filename convention.
    """
    return image_path.parent.name


def calculate_image_metrics(image: np.ndarray, background_percentile: float = 20) -> dict:
    """Calculate intensity, background, noise, SNR and focus metrics for one image.

    Background is estimated from pixels below ``background_percentile``.
    """
    from scipy.ndimage import laplace
    from scipy.stats import skew

    pixels = image.ravel()
    finite_pixels = pixels[np.isfinite(pixels)]
    if finite_pixels.size == 0:
        raise ValueError("Image contains no finite pixel values.")

    background_threshold = np.percentile(finite_pixels, background_percentile)
    background_pixels = finite_pixels[finite_pixels <= background_threshold]
    signal_pixels = finite_pixels[finite_pixels > background_threshold]
    if background_pixels.size == 0:
        background_pixels = finite_pixels
    if signal_pixels.size == 0:
        signal_pixels = finite_pixels

    background_mean = float(np.mean(background_pixels))
    background_median = float(np.median(background_pixels))
    background_sd = float(np.std(background_pixels, ddof=1) if background_pixels.size > 1 else 0)
    signal_mean = float(np.mean(signal_pixels))
    signal_median = float(np.median(signal_pixels))
    snr = (signal_mean - background_mean) / background_sd if background_sd > 0 else np.nan
    focus_score = float(np.var(laplace(image)))

    # Saturation relative to the 16-bit range -- matches the source notebook's
    # assumption. If a dataset uses a different bit depth, pass a different
    # dtype-aware threshold upstream; this function's default stays as-is to
    # match the assay this pipeline was built for (16-bit CellProfiler input).
    saturation_threshold = np.iinfo(np.uint16).max
    saturated_fraction = float(np.mean(finite_pixels >= saturation_threshold))
    zero_fraction = float(np.mean(finite_pixels == 0))

    return {
        "image_mean": float(np.mean(finite_pixels)),
        "image_median": float(np.median(finite_pixels)),
        "image_sd": float(np.std(finite_pixels, ddof=1)),
        "image_min": float(np.min(finite_pixels)),
        "image_max": float(np.max(finite_pixels)),
        "p01": float(np.percentile(finite_pixels, 1)),
        "p05": float(np.percentile(finite_pixels, 5)),
        "p50": float(np.percentile(finite_pixels, 50)),
        "p95": float(np.percentile(finite_pixels, 95)),
        "p99": float(np.percentile(finite_pixels, 99)),
        "histogram_skewness": float(skew(finite_pixels)),
        "background_threshold": float(background_threshold),
        "background_mean": background_mean,
        "background_median": background_median,
        "background_sd": background_sd,
        "signal_mean": signal_mean,
        "signal_median": signal_median,
        "snr": float(snr),
        "focus_score": focus_score,
        "zero_fraction": zero_fraction,
        "saturated_fraction": saturated_fraction,
        "n_pixels": int(finite_pixels.size),
    }


def scan_images(
    image_root: Path,
    *,
    supported_extensions: frozenset = DEFAULT_SUPPORTED_EXTENSIONS,
    channel_keywords: Mapping[str, Sequence[str]] = DEFAULT_CHANNEL_KEYWORDS,
    background_percentile: float = 20,
) -> tuple[pd.DataFrame, list[dict]]:
    """Scan every image under ``image_root``, computing per-image QC metrics.

    Returns ``(image_qc_df, failed_images)`` -- ``failed_images`` is a list
    of ``{"filepath": ..., "error": ...}`` records for any image that could
    not be read or measured (the scan continues past individual failures).
    """
    import tifffile

    image_paths = sorted(
        p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images were found in: {image_root}")

    records = []
    failed_images = []
    for image_path in image_paths:
        try:
            channel = infer_channel(image_path.name, channel_keywords)
            well = infer_well(image_path.name)
            site = infer_site(image_path.name, channel, channel_keywords)
            plate = infer_plate(image_path)
            image = read_image(image_path)
            metrics = calculate_image_metrics(image, background_percentile=background_percentile)
            records.append(
                {
                    "Metadata_Plate": plate,
                    "Metadata_Well": well,
                    "Metadata_Site": site,
                    "Metadata_Channel": channel,
                    "filename": image_path.name,
                    "filepath": str(image_path),
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "dtype_original": str(
                        tifffile.imread(image_path).dtype
                        if image_path.suffix.lower() in {".tif", ".tiff"}
                        else "unknown"
                    ),
                    **metrics,
                }
            )
        except Exception as error:  # noqa: BLE001 - continue scanning past bad images
            failed_images.append({"filepath": str(image_path), "error": str(error)})

    image_qc_df = pd.DataFrame(records)
    if image_qc_df.empty:
        raise ValueError("No images were processed successfully.")
    return image_qc_df, failed_images


DEFAULT_SUMMARY_METRICS = (
    "image_mean",
    "image_median",
    "image_sd",
    "background_mean",
    "background_sd",
    "signal_mean",
    "snr",
    "focus_score",
    "p01",
    "p99",
    "zero_fraction",
    "saturated_fraction",
)


def aggregate_per_well_image_qc(
    image_qc_df: pd.DataFrame,
    summary_metrics: Sequence[str] = DEFAULT_SUMMARY_METRICS,
) -> pd.DataFrame:
    """Aggregate per-image QC metrics to one row per plate/well/channel."""
    per_well_qc = image_qc_df.groupby(
        ["Metadata_Plate", "Metadata_Well", "Metadata_Channel"], dropna=False
    )[list(summary_metrics)].agg(["median", "mean", "std", "min", "max", "count"])
    per_well_qc.columns = [f"{metric}_{statistic}" for metric, statistic in per_well_qc.columns]
    return per_well_qc.reset_index()


def calculate_normalized_histogram(
    image: np.ndarray,
    bins: int = 256,
    intensity_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bin_centers, density)`` for a normalized pixel-intensity histogram."""
    pixels = image.ravel()
    pixels = pixels[np.isfinite(pixels)]
    if intensity_range is None:
        intensity_range = (float(np.min(pixels)), float(np.max(pixels)))

    histogram, bin_edges = np.histogram(pixels, bins=bins, range=intensity_range, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, histogram


def robust_rescale_image(image: np.ndarray) -> np.ndarray:
    """Percentile-based (5th-99th) rescale of an image to the [0, 1] range."""
    image = image.astype(float)
    low = np.percentile(image, 5)
    high = np.percentile(image, 99)
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def calculate_robust_contrast(image: np.ndarray) -> dict:
    """Percentile-based (20th-90th) contrast and contrast-to-noise ratio for an image.

    Uses the more complete of the source notebook's two near-identical
    copies of this function (cell ~73, which additionally reports
    ``low_pixel_sd``).
    """
    pixels = image[np.isfinite(image)]
    p20 = np.percentile(pixels, 20)
    p90 = np.percentile(pixels, 90)
    low_pixels = pixels[pixels <= p20]
    low_sd = np.std(low_pixels, ddof=1) if low_pixels.size > 1 else 0.0
    cnr = (p90 - p20) / low_sd if low_sd > 0 else np.nan

    return {
        "p20": p20,
        "p90": p90,
        "dynamic_range_20_90": p90 - p20,
        "low_pixel_sd": low_sd,
        "robust_cnr": cnr,
    }
