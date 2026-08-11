"""Shared, dataset-agnostic library for the Live Cell Painting analysis pipeline.

Replaces the old flat ``lcp_utils.py``. Notebooks generally import from the
specific submodule they need (``hca_pipeline.config``, ``hca_pipeline.io``,
...); the most commonly used names are also re-exported here for convenience.
"""

from .config import (
    COMPARTMENT_PREFIXES,
    SUPPORTED_PLATE_FORMATS,
    ExperimentConfig,
    find_column,
    validate_configuration,
)
from .feature_select import infer_feature_cols, select_features
from .io import (
    find_repo_root,
    parquet_structure,
    resolve_resume_stage,
    validate_checkpoint_df,
    write_csv_protected,
    write_parquet_protected,
    write_summary_table_protected,
)
from .metadata import (
    annotate_per_plate,
    dedupe_meta,
    ensure_core_metadata,
    enforce_pascalcase_metadata_columns,
    find_wells_missing_from_layout,
    norm_well,
    read_barcode_platemap,
    read_platemap_layout,
)
from .metrics_qc import (
    generate_go_nogo_dashboard,
    mean_average_precision,
    percent_replicating,
    run_dose_response,
    run_per_plate_qc,
)
from .modelling import balanced_sample, run_modelling_space
from .normalize import normalize_per_plate_mad_robustize
from .schema import (
    compare_schemas,
    describe_schema,
    find_duplicate_like_columns,
    schema_fingerprint,
    validate_checkpoint,
)
from .single_cell import (
    curate_single_cell_features,
    fit_hdbscan,
    mahal_outliers_within_well,
    subsample_for_embedding,
    sweep_hdbscan_params,
    train_lgbm_classifier_with_shap,
)
from .stats import cohens_d
from .taxonomy import build_taxonomy_table, classify_feature

__all__ = [
    "COMPARTMENT_PREFIXES",
    "SUPPORTED_PLATE_FORMATS",
    "ExperimentConfig",
    "find_column",
    "validate_configuration",
    "infer_feature_cols",
    "select_features",
    "find_repo_root",
    "parquet_structure",
    "resolve_resume_stage",
    "validate_checkpoint_df",
    "write_csv_protected",
    "write_parquet_protected",
    "write_summary_table_protected",
    "annotate_per_plate",
    "dedupe_meta",
    "ensure_core_metadata",
    "enforce_pascalcase_metadata_columns",
    "find_wells_missing_from_layout",
    "norm_well",
    "read_barcode_platemap",
    "read_platemap_layout",
    "generate_go_nogo_dashboard",
    "mean_average_precision",
    "percent_replicating",
    "run_dose_response",
    "run_per_plate_qc",
    "balanced_sample",
    "run_modelling_space",
    "normalize_per_plate_mad_robustize",
    "compare_schemas",
    "describe_schema",
    "find_duplicate_like_columns",
    "schema_fingerprint",
    "validate_checkpoint",
    "curate_single_cell_features",
    "fit_hdbscan",
    "mahal_outliers_within_well",
    "subsample_for_embedding",
    "sweep_hdbscan_params",
    "train_lgbm_classifier_with_shap",
    "cohens_d",
    "build_taxonomy_table",
    "classify_feature",
]
