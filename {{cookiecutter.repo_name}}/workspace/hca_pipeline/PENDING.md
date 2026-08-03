# hca_pipeline — pending

This is the vendored home for the shared, assay-agnostic Python package
imported by every analysis notebook (config/experiment resolution, metadata
annotation, normalization, feature selection, taxonomy, quality metrics,
modelling, single-cell tooling, plotting). It replaces per-notebook
duplication of this logic.

The actual package content is being ported from the reference implementation
in a follow-up pass, together with the marimo notebook templates under
`workspace/analysis/templates/` that depend on it. Until then this folder is
a placeholder.
