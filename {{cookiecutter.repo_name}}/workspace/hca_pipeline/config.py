"""Experiment configuration for the HCA analysis pipeline.

Centralizes the knobs every pipeline notebook needs to be dataset-agnostic:
which metadata columns exist, what control-type vocabulary this experiment
uses, and whether optional axes (dose, time) are present. An ExperimentConfig
is loaded from and saved to
``workspace/metadata/{EXPERIMENT_ID}/experiment_config.json`` so every
notebook in the pipeline reads the same resolved configuration instead of
re-deriving (and potentially disagreeing on) it independently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from numbers import Integral
from pathlib import Path
from typing import Optional, Sequence

SUPPORTED_PLATE_FORMATS = {
    6: {"rows": "AB", "columns": 3},
    12: {"rows": "ABC", "columns": 4},
    24: {"rows": "ABCD", "columns": 6},
    48: {"rows": "ABCDEF", "columns": 8},
    96: {"rows": "ABCDEFGH", "columns": 12},
    384: {"rows": "ABCDEFGHIJKLMNOP", "columns": 24},
}

# Single source of truth for CellProfiler compartment prefixes, shared by
# feature_select.infer_feature_cols and by any notebook cell that needs to
# check per-plate column consistency against the same vocabulary.
COMPARTMENT_PREFIXES = ("Cells_", "Cytoplasm_", "Nuclei_", "Vesicles_")


def validate_configuration(
    experiment_id: str,
    plate_format: int,
    min_cells_per_well: int,
    experiments_dir: str | Path,
) -> None:
    """Validate the user-defined pipeline configuration.

    Parameters
    ----------
    experiment_id
        Experiment identifier. It must exactly match an experiment folder name.
    plate_format
        Number of wells in the plate.
    min_cells_per_well
        Minimum number of cells required for a well to pass QC.
    experiments_dir
        Directory containing the experiment folders, normally
        ``workspace/backend``.

    Raises
    ------
    ValueError
        If one or more configuration values are missing or invalid.
    """

    errors: list[str] = []

    # ------------------------------------------------------------------
    # Experiment ID
    # ------------------------------------------------------------------
    experiment_id_is_valid = True

    if not isinstance(experiment_id, str) or not experiment_id.strip():
        experiment_id_is_valid = False
        errors.append(
            "EXPERIMENT_ID is missing.\n"
            "  Enter the experiment identifier exactly as it appears in "
            "the project folder name.\n"
            '  Example: EXPERIMENT_ID = "2025_01_MCF7_NPPS_24h"'
        )

    elif experiment_id == "SET_EXPERIMENT_ID_HERE":
        experiment_id_is_valid = False
        errors.append(
            "EXPERIMENT_ID has not been configured.\n"
            "  Replace SET_EXPERIMENT_ID_HERE with the exact name of the "
            "experiment folder."
        )

    elif experiment_id != experiment_id.strip():
        experiment_id_is_valid = False
        errors.append(
            "EXPERIMENT_ID contains leading or trailing spaces.\n"
            f"  Entered: {experiment_id!r}\n"
            f"  Use: {experiment_id.strip()!r}"
        )

    elif Path(experiment_id).name != experiment_id:
        experiment_id_is_valid = False
        errors.append(
            "EXPERIMENT_ID must contain only the folder name, but received:\n"
            f"  {experiment_id!r}\n"
            "  Do not enter a complete path."
        )

    # ------------------------------------------------------------------
    # Experiment folder
    # ------------------------------------------------------------------
    experiments_dir = Path(experiments_dir)

    if not experiments_dir.is_dir():
        errors.append(
            "The experiment directory was not found.\n"
            f"  Expected directory: {experiments_dir}\n"
            "  Confirm that the notebook is being run from inside the "
            "correct project repository."
        )

    elif experiment_id_is_valid:
        available_experiments = sorted(
            path.name for path in experiments_dir.iterdir() if path.is_dir()
        )

        # Compare strings directly so the check remains case-sensitive even on
        # case-insensitive file systems such as the default macOS filesystem.
        if experiment_id not in available_experiments:
            case_matches = [
                name
                for name in available_experiments
                if name.casefold() == experiment_id.casefold()
            ]

            if case_matches:
                errors.append(
                    "EXPERIMENT_ID does not exactly match the experiment "
                    "folder name.\n"
                    f"  Entered: {experiment_id!r}\n"
                    f"  Existing folder: {case_matches[0]!r}\n"
                    "  Copy the folder name exactly, including capitalization."
                )
            else:
                available_text = (
                    "\n    - " + "\n    - ".join(available_experiments)
                    if available_experiments
                    else "\n    No experiment folders were found."
                )

                errors.append(
                    "No experiment folder matches EXPERIMENT_ID.\n"
                    f"  Entered: {experiment_id!r}\n"
                    f"  Searched in: {experiments_dir}\n"
                    "  Available experiment folders:"
                    f"{available_text}"
                )

    # ------------------------------------------------------------------
    # Plate format
    # ------------------------------------------------------------------
    if plate_format is None:
        errors.append(
            "PLATE_FORMAT is missing.\n"
            "  Enter the number of wells in the plate used in this experiment.\n"
            "  Supported values: " + ", ".join(map(str, SUPPORTED_PLATE_FORMATS))
        )

    elif isinstance(plate_format, bool) or not isinstance(plate_format, Integral):
        errors.append(f"PLATE_FORMAT must be an integer, but received {plate_format!r}.")

    elif plate_format not in SUPPORTED_PLATE_FORMATS:
        errors.append(
            f"Unsupported PLATE_FORMAT: {plate_format!r}.\n"
            "  Supported values: " + ", ".join(map(str, SUPPORTED_PLATE_FORMATS))
        )

    # ------------------------------------------------------------------
    # Minimum cell-count threshold
    # ------------------------------------------------------------------
    if min_cells_per_well is None:
        errors.append(
            "MIN_CELLS_PER_WELL is missing.\n"
            "  Enter the minimum number of cells required for a well to "
            "pass QC.\n"
            "  Example: MIN_CELLS_PER_WELL = 100"
        )

    elif isinstance(min_cells_per_well, bool) or not isinstance(min_cells_per_well, Integral):
        errors.append(
            "MIN_CELLS_PER_WELL must be a whole number, but received "
            f"{min_cells_per_well!r}."
        )

    elif min_cells_per_well <= 0:
        errors.append(
            "MIN_CELLS_PER_WELL must be greater than zero, but received "
            f"{min_cells_per_well}."
        )

    # ------------------------------------------------------------------
    # Stop the notebook if any validation failed
    # ------------------------------------------------------------------
    if errors:
        message = (
            "\nConfiguration error\n"
            "===================\n\n"
            + "\n\n".join(f"{index}. {error}" for index, error in enumerate(errors, start=1))
            + "\n\nCorrect the values in the 'User configuration' section "
            "and run this cell again."
        )

        raise ValueError(message)


def find_column(
    df,
    target_names: Sequence[str],
    prefix: str = "Metadata_",
) -> Optional[str]:
    """Find a column matching any of *target_names*, case-insensitive.

    Tries exact match first, then case-insensitive match. Returns the actual
    column name from ``df.columns``, or ``None`` if not found.
    """
    for target in target_names:
        if target in df.columns:
            return target
        for col in df.columns:
            if col.lower() == target.lower():
                return col
    return None


@dataclass
class ExperimentConfig:
    """Resolved, dataset-agnostic configuration for one experiment.

    ``concentration_col``/``time_col`` are ``None`` when the corresponding
    axis doesn't apply to this experiment -- notebooks branch on that instead
    of assuming every dataset has a dose or time dimension. ``image_root`` is
    ``None`` unless raw microscopy images are available for this experiment,
    following the same pattern -- ``00_image_quality.py`` self-skips when it
    is unset. Control-type vocabulary is a *list* per role (not a single
    literal) since different platemap conventions use different labels for
    the same role.
    """

    experiment_id: str
    plate_format: int = 96
    min_cells_per_well: int = 100
    channels: list[str] = field(default_factory=list)
    image_root: Optional[str] = None

    plate_col: str = "Metadata_Plate"
    well_col: str = "Metadata_Well"
    site_col: str = "Metadata_Site"
    treatment_col: str = "Metadata_Treatment"
    concentration_col: Optional[str] = "Metadata_Concentration"
    time_col: Optional[str] = None
    control_type_col: str = "Metadata_Control_Type"

    negcon_values: list[str] = field(default_factory=lambda: ["negcon", "con"])
    poscon_values: list[str] = field(default_factory=lambda: ["poscon"])
    trt_values: list[str] = field(default_factory=lambda: ["trt"])

    has_dose_axis: bool = False
    has_time_axis: bool = False

    overwrite_existing_outputs: bool = False
    save_provenance_history: bool = False

    @staticmethod
    def config_path(repo_root: Path, experiment_id: str) -> Path:
        return repo_root / "workspace" / "metadata" / experiment_id / "experiment_config.json"

    @classmethod
    def load(cls, repo_root: Path, experiment_id: str) -> "ExperimentConfig":
        """Load a saved config, or return defaults for *experiment_id* if none exists yet."""
        path = cls.config_path(repo_root, experiment_id)
        if not path.exists():
            return cls(experiment_id=experiment_id)

        data = json.loads(path.read_text(encoding="utf-8"))
        known_fields = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known_fields}
        data["experiment_id"] = experiment_id
        return cls(**data)

    def save(self, repo_root: Path) -> Path:
        """Persist this config to disk, creating the metadata folder if needed."""
        path = self.config_path(repo_root, self.experiment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def resolve_columns(self, df) -> "ExperimentConfig":
        """Return a copy with column-name fields reconciled against *df* (case-tolerant).

        A configured column name that doesn't exist verbatim in ``df`` is
        replaced by whatever actually-present column matches case-insensitively;
        an optional column (``concentration_col``/``time_col``) that isn't
        found at all is set to ``None`` rather than left dangling.
        """
        plate_col = find_column(df, [self.plate_col, "Metadata_Plate"]) or self.plate_col
        well_col = find_column(df, [self.well_col, "Metadata_Well"]) or self.well_col
        treatment_col = (
            find_column(df, [self.treatment_col, "Metadata_Treatment"]) or self.treatment_col
        )
        control_type_col = find_column(df, [self.control_type_col, "Metadata_Control_Type"])
        concentration_col = None
        if self.concentration_col:
            concentration_col = find_column(
                df, [self.concentration_col, "Metadata_Concentration", "Metadata_Dose"]
            )
        time_col = None
        if self.time_col:
            time_col = find_column(df, [self.time_col, "Metadata_Time"])

        return replace(
            self,
            plate_col=plate_col,
            well_col=well_col,
            treatment_col=treatment_col,
            control_type_col=control_type_col or self.control_type_col,
            concentration_col=concentration_col,
            time_col=time_col,
            has_dose_axis=concentration_col is not None,
            has_time_axis=time_col is not None,
        )
