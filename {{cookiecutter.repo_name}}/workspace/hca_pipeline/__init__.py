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
from .feature_select import (
    infer_feature_cols,
    is_technical_identifier_column,
    select_features,
    technical_identifier_columns,
)
from .io import (
    checkpoint_matches_plate_scope,
    find_repo_root,
    parquet_structure,
    resolve_resume_stage,
    validate_checkpoint_df,
    write_csv_protected,
    write_parquet_protected,
    write_summary_table_protected,
)
from .metadata import (
    CELL_COUNT_METADATA_COLUMN,
    add_cell_count_metadata,
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
from .normalize import normalize_per_plate_mad_robustize
from .provenance import (
    PROVENANCE_SCHEMA_VERSION,
    canonicalize_provenance,
    file_record,
    provenance_json,
    validate_provenance_record,
)
from .schema import (
    compare_schemas,
    describe_schema,
    find_duplicate_like_columns,
    schema_fingerprint,
    validate_checkpoint,
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
    "is_technical_identifier_column",
    "select_features",
    "technical_identifier_columns",
    "checkpoint_matches_plate_scope",
    "find_repo_root",
    "parquet_structure",
    "resolve_resume_stage",
    "validate_checkpoint_df",
    "write_csv_protected",
    "write_parquet_protected",
    "write_summary_table_protected",
    "CELL_COUNT_METADATA_COLUMN",
    "add_cell_count_metadata",
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
    "normalize_per_plate_mad_robustize",
    "PROVENANCE_SCHEMA_VERSION",
    "canonicalize_provenance",
    "file_record",
    "provenance_json",
    "validate_provenance_record",
    "compare_schemas",
    "describe_schema",
    "find_duplicate_like_columns",
    "schema_fingerprint",
    "validate_checkpoint",
    "cohens_d",
    "build_taxonomy_table",
    "classify_feature",
]
