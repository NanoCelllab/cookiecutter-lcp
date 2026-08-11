# Future module: cell-count confounder QC and auditable well exclusion

## Status

**Deferred proposal — not implemented.** This document records the design and
acceptance criteria for a future, deliberately scoped implementation. The
current pipeline preserves `Metadata_Cell_Count` as metadata and visualizes it,
but does not use it to exclude wells.

The motivation is that low-cell-count wells may produce unstable aggregate
morphological profiles. In the current experiment, the strongest low-count
pattern appears within a control/reference population (`Non-dormant`), so it
cannot safely be interpreted as treatment biology or technical failure from a
plot alone.

## Intended position in the pipeline

Implement this as a dedicated notebook between aggregation/preliminary
profiling and the quality-metrics gate. Do not hide it inside NB02.

```text
NB01  Sample retrieval and plate selection
NB02  Aggregation, preliminary normalization and feature selection
NB03  Cell-count confounder QC (new)
NB04  Quality metrics / preliminary or final Go-No-Go (current NB03)
NB05  Phenotypic profiling (current NB04)
NB06  Phenotypic fingerprints (current NB05)
NB07  Single-cell analysis (current NB06)
```

The proposed shared implementation should live in:

```text
workspace/hca_pipeline/cell_count_qc.py
```

The notebook should contain the UI, explanations, plots and decision summary;
statistical and resampling routines should live in the shared module and have
unit tests.

## Non-negotiable design rules

1. Never remove wells merely because their cell count is low.
2. Preserve the complete NB02 inputs and never overwrite them silently.
3. Separate diagnosis, recommendation and application into explicit states.
4. Test technical-versus-biological explanations quantitatively.
5. Select a threshold against a proper null and with a retention penalty; do
   not select it from a raw-metric plateau.
6. Repeat the diagnosis before and after feature selection.
7. The within-control analysis is mandatory.
8. If exclusion is applied, rerun normalization and feature selection on the
   retained wells; do not only subset an already normalized final parquet.
9. Every exclusion must appear in an auditable manifest.
10. Downstream notebooks must consume a QC-approved checkpoint rather than
    independently reconstructing the exclusion.

## Inputs and immutable checkpoints

The module should consume these existing NB02 outputs:

- `single_cell_ready.parquet` for within-well subsampling;
- `per_well_aggregated.parquet` for reprocessing candidate thresholds;
- `per_well_normalized.parquet` for pre-feature-selection diagnosis;
- `per_well_features_selected.parquet` for post-feature-selection diagnosis.

`Metadata_Cell_Count` must remain metadata and must never enter the morphology
feature matrix returned by `infer_feature_cols()`.

## Proposed configuration

Add a nested configuration or equivalent validated fields to
`ExperimentConfig`:

```json
{
  "cell_count_qc": {
    "mode": "diagnose",
    "apply_exclusion": false,
    "threshold": null,
    "decision_source": null,
    "reference_treatments": ["Non-treated", "Non-dormant"],
    "candidate_thresholds": [0, 10, 20, 30, 40, 50, 75, 100, 150, 200],
    "permutations": 999,
    "bootstrap_iterations": 200,
    "random_seed": 42
  }
}
```

Supported modes:

- `diagnose`: calculate and recommend, but exclude nothing;
- `apply_recommended`: apply the recorded recommendation;
- `manual`: apply an explicit threshold and label it as a manual decision;
- `disabled`: retain everything and record that the QC was not run.

The default must be `diagnose` with `apply_exclusion: false`.

## Required analyses

### 1. PCA association before and after feature selection

Run PCA on standardized per-well morphology for both the normalized,
pre-feature-selection matrix and the feature-selected matrix. For at least
PC1–PC10, export:

- explained-variance fraction;
- Pearson coefficient and p-value versus raw cell count and `log10(count)`;
- Spearman coefficient and p-value versus both count representations;
- partial association after residualizing PC score and log-count against
  treatment, concentration, plate and imaging batch.

Categorical covariates should use an explicit design matrix. Missing or
aliased terms must be reported, not silently substituted. If imaging batch is
absent, report it as unavailable; do not use plate twice.

Generate coefficient-by-PC plots and scatter plots for PC1, PC2 and any PC
whose absolute partial association crosses a documented effect-size threshold.

### 2. Formal multivariate tests

Fit a distance-based multivariate model with terms for treatment,
concentration, log-count, plate/batch and treatment × log-count. Report
pseudo-F, partial R² and permutation p-value for each estimable term using at
least 999 reproducible permutations.

Do not rely on the single-factor convenience PERMANOVA API. Implement or use a
tested multi-term distance-based model with Freedman–Lane-style permutation of
residuals. Restrict permutations within the appropriate exchangeability block
(normally plate or imaging batch) where required by the design.

Also run Kruskal–Wallis tests of cell count across treatments and plates and
export median, IQR, range and group size. Treat p-values as evidence alongside
effect sizes, not as the decision rule by themselves.

### 3. Mandatory within-control analysis

Repeat PCA/count association and the estimable multivariate count test within:

1. strict negative controls defined by `Metadata_Control_Type`; and
2. each configured biological reference such as `Non-dormant`.

Do not automatically combine `Non-treated` and `Non-dormant`. In the current
configuration `Non-dormant` is a treatment/reference, not a negative-control
type, and the two populations must first be evaluated separately.

Persistent within-control association is strong evidence for a technical
effect. A treatment interaction without an independent count main effect is
more compatible with treatment-related viability or proliferation biology.

### 4. Profile reliability versus cell count

For every well, calculate distance to its replicate centroid using an explicit
replicate definition based on treatment, concentration and the appropriate
context. Avoid making plate part of the replicate definition when the purpose
is to measure cross-plate reproducibility; report within-plate and cross-plate
versions separately where possible.

Calculate pairwise cosine similarity and Pearson correlation between
replicates, along with pair minimum, arithmetic mean and geometric mean cell
counts. Export a per-well agreement score and plot smoothed relationships and
cell-count-bin summaries with sample sizes.

### 5. Single-cell bootstrap stability

For sufficiently populated wells, repeatedly subsample cells at feasible
levels selected from `10, 20, 30, 50, 75, 100, 150, 200`, recompute median
profiles and compare them with the full-well aggregate using cosine similarity.
Use fixed seeds, record the eligible well set at every level, and show
confidence intervals. If compute is constrained, use a preregistered reduced
set that covers the observed plateau; do not choose levels after viewing the
result.

### 6. Candidate-threshold sensitivity

Adapt the configured thresholds to the observed distribution while retaining
the unfiltered baseline. At every threshold:

1. subset `per_well_aggregated`;
2. rerun the same cleaning, per-plate normalization and feature selection;
3. record retained/removed wells overall, by treatment, condition and plate;
4. record lost or under-replicated conditions;
5. compute explained variance and raw/partial PC1/PC2 count associations;
6. compute centroid dispersion and replicate similarity;
7. compute copairs mAP with the pipeline's valid positive/negative pairing;
8. compute a treatment-label-permuted null mAP using valid exchangeability
   blocks and multiple permutations, not a single lucky shuffle.

Do not compare thresholds with different feature preprocessing or a changing
definition of replicates without reporting that change.

### 7. PCA stability with a sampling null

Compare PCA solutions on the intersection of wells/features that can validly
be aligned. Align signs/rotations before measuring change.

- Estimate within-threshold sampling variability at the unfiltered baseline
  and recommended threshold using bootstrap resampling and Procrustes distance.
- Compute between-threshold Procrustes distances.
- Compare between-threshold distance with the within-threshold bootstrap
  distribution/reference band.
- Also correlate pairwise sample distances across thresholds.

Document how duplicate wells produced by bootstrap sampling are handled.

### 8. Non-circular recommendation

Implement both rules if feasible:

1. **Permutation-null gap:** true mAP minus permuted-label mAP at each threshold;
   identify diminishing returns with a predefined tolerance.
2. **Retention-aware marginal gain:** change in mAP or null gap divided by the
   number of newly removed wells; define the minimum useful gain before the
   run.

The final recommendation must reconcile these with bootstrap stability,
partial correlations, loss of experimental coverage and the technical versus
biological classification. A stricter threshold must not win solely because it
removes more noisy observations.

It is valid and expected for the recommendation to be **no exclusion**.

## Decision synthesis

Classify the evidence as one of:

- `technical_confound_supported`;
- `biology_or_interaction_supported`;
- `batch_effect_more_likely`;
- `mixed_or_inconclusive`;
- `no_material_cell_count_effect`.

Technical confounding requires convergent evidence such as an independent
count term in the multivariate model, persistent partial PC association,
within-control structure and degraded low-count profile stability. A
significant interaction confined to biologically plausible cytotoxic or
anti-proliferative treatments should not trigger automatic exclusion.

## Outputs

Write all artifacts under dedicated directories:

```text
analysis/{EXPERIMENT_ID}/results/cell_count_qc/
analysis/{EXPERIMENT_ID}/figures/cell_count_qc/
```

Required outputs include:

- `cell_count_qc_report.md`;
- `cell_count_qc_decision.json`;
- `pc_cell_count_associations_pre_fs.csv`;
- `pc_cell_count_associations_post_fs.csv`;
- `permanova_terms.csv`;
- `kruskal_wallis_cell_count.csv`;
- `within_control_results.csv`;
- `well_profile_stability.csv`;
- `replicate_pair_similarity.csv`;
- `bootstrap_profile_stability.csv`;
- `threshold_sensitivity.csv`;
- `threshold_removals_by_group.csv`;
- `pca_stability.csv`;
- `excluded_wells_cell_count.csv`.

The exclusion manifest must contain at least:

- experiment ID;
- plate;
- well;
- treatment;
- concentration;
- imaging batch when available;
- cell count;
- applied threshold;
- exclusion reason;
- decision mode/source;
- configuration hash and timestamp.

Proposed derived checkpoints:

```text
profiles/{EXPERIMENT_ID}/outputs/per_well_features_selected_qc.parquet
profiles/{EXPERIMENT_ID}/outputs/cache/single_cell_ready_qc.parquet
```

If no exclusion is recommended, these may contain all rows, but provenance
must explicitly record the no-exclusion decision.

## Downstream and single-cell integration

The quality gate and all biological profiling notebooks should consume the
QC-approved per-well checkpoint. The single-cell notebook should consume the
QC-approved single-cell checkpoint or join the exclusion manifest by plate and
well.

The current single-cell workflow uses PCA, UMAP, HDBSCAN, KMeans and LightGBM
with SHAP—not single-cell LDA/GMM as assumed in the original proposal. Adapt
the sensitivity analysis accordingly:

- color PCA/UMAP by source-well cell count and exclusion status;
- test enrichment of below-threshold wells in HDBSCAN/KMeans regions;
- compare embeddings/neighborhoods with and without flagged cells;
- compare clusters using ARI/NMI on common cells;
- compare classifier performance and feature importance;
- report whether the well-level exclusion should propagate to single-cell
  interpretation.

## Future companion module: cell-count regression correction

### Status and dependency

**Deferred full implementation.** NB04 may contain a small, diagnostic-only
PCA/mAP comparison, but it must not be treated as an approved correction or as
a substitute for this specification.

Regression correction is downstream of the detection module in this document.
It may be enabled only when `cell_count_qc_decision.json` classifies the effect
as technical confounding. A manual toggle alone must not override a biological,
batch-dominated or inconclusive decision. Default: disabled.

Regression and exclusion solve different problems:

- exclusion addresses unstable variance from extremely sparse wells;
- regression addresses a systematic count-dependent mean shift among retained
  wells;
- the hybrid strategy first applies an independently justified stability
  threshold and then residualizes the remaining count association.

### Placement and immutable outputs

Apply correction after initial filtering/normalization and any justified
low-count exclusion, but before PCA, clustering, LDA, manifolds, copairs/mAP
and treatment-similarity analysis. Preserve `Metadata_Cell_Count` and the
uncorrected profiles. Write a separate corrected parquet; never silently
replace a canonical input.

If exclusion is part of the selected strategy, reconstruct the retained
profiles from the aggregated checkpoint and rerun normalization/feature
selection before fitting residual correction.

### Regression models

Default predictor: `log10(Metadata_Cell_Count)`.

For feature `j`, estimate a count-dependent shift and subtract that shift while
anchoring values at a documented reference count (normally the median fitting
count). Retain metadata and well identity unchanged.

Wells below configurable `min_cell_count_for_regression` do not estimate the
slope because of leverage/instability. They may receive the fitted correction
only if they were not excluded by the detection module. Record both thresholds
and their distinct meanings.

Required modes:

1. **Global:** fit `feature ~ log_count` on all eligible wells. Permit only when
   detection/Kruskal–Wallis evidence supports independence of treatment and
   count. Otherwise refuse because global slopes can absorb treatment biology.
2. **Control-based (preferred):** fit in validated negative controls and apply
   to all retained wells. Test transportability with per-treatment
   `treatment × log_count` interactions and FDR correction.
3. **Plate-aware additive:** default plate adjustment is
   `feature ~ log_count + plate`, allowing intercept shifts.
4. **Plate interaction:** consider `feature ~ log_count * plate` only when it
   improves AIC by more than 10 and does not degrade cross-validated mAP.

Diagnose nonlinearity by comparing linear and 3-df spline fits per feature.
Adopt nonlinear correction globally only when more than the configured
fraction (proposed default 20%) of practically count-associated features
(`R² > 0.10`) favor the spline by ΔAIC > 10. Otherwise retain the interpretable
linear model for every feature.

### Leakage, transportability and overfitting safeguards

- Refuse global regression when treatment materially determines cell count.
- For control-based fits, export treatment-versus-control slope differences
  and FDR-adjusted interaction p-values. Flag divergent slopes, especially in
  non-cytotoxic treatments.
- Report feature-level in-sample and 5-fold cross-validated R² for global
  models.
- For control-based models, report control-fit R² and predictive/transport R²
  in each treated group.
- Compare treatment PERMANOVA R² before and after correction. A relative loss
  above the configured tolerance (proposed 10%) makes regression harmful.
- A significant result alone is insufficient: practical feature-level signal
  requires FDR significance and configurable R² (default > 0.10).

### Before/after formal diagnostics

For uncorrected and corrected profiles, repeat PC1–PC10 Pearson, Spearman and
partial count associations and the full multi-term distance model. Correction
should reduce count and interaction R² while preserving treatment R².

Compare reproducibility using:

- replicate Pearson and cosine similarity;
- non-replicate similarity reported separately;
- distance to replicate centroid;
- copairs mAP plus permuted-label null mAP.

Always decompose an mAP change. An apparent improvement accompanied by a large
loss of replicate similarity may indicate removed biological structure rather
than useful denoising.

### Biological-signal preservation

Supplement PERMANOVA with:

- correlation/Mantel test of treatment-level distance matrices;
- nearest-neighbor preservation (proposed minimum 80%);
- treatment rank stability;
- dose-response monotonicity/profile correlation;
- positive-control and control-versus-treatment separation.

Flag regression when the treatment-distance correlation falls below 0.9,
nearest-neighbor preservation falls below 80%, dose response deteriorates or
positive controls lose separation.

### Four-strategy benchmark

Compare under identical preprocessing and pairing definitions:

1. no correction;
2. stability-derived well exclusion only;
3. regression only;
4. hybrid exclusion followed by regression.

For each strategy export retained wells/features, replicate and non-replicate
similarity, true/null mAP, raw/partial PC1/PC2 count association, treatment,
count and interaction PERMANOVA R², treatment separation, Procrustes stability
against a within-strategy bootstrap reference, treatment-distance correlation
and nearest-neighbor preservation.

Decision order:

1. prefer regression/hybrid only when mAP improves and count association falls
   without >10% relative loss of treatment R²;
2. prefer exclusion when it provides comparable benefit because it has fewer
   model/transport assumptions;
3. choose hybrid when exclusion and residualization add independent benefit;
4. choose no correction when neither provides material improvement;
5. reject regression when biological-signal safeguards fail, even if mAP rises.

Record the triggered rule and its metrics. “No correction” is a valid outcome.

### Covariate adjustment alternative

Support `log_count` as a nuisance covariate in models that accept covariates
(PERMANOVA, treatment-effect and dose-response models). Prefer covariate
adjustment for supervised inference because it does not alter profiles. Keep
residualized profiles as the option for unsupervised PCA/clustering/manifold
and retrieval analyses that cannot accept covariates. Document that these
approaches answer related but non-identical questions.

### Proposed configuration

```yaml
cell_count_correction:
  enabled: false
  predictor: log10
  mode: control_based
  plate_adjustment: additive
  nonlinear_diagnostic: true
  nonlinear_threshold_aic: 10
  nonlinear_fraction_threshold: 0.20
  min_cell_count_for_regression: 20
  r2_practical_threshold: 0.10
  fdr_alpha: 0.05
  cv_folds: 5
  compare_uncorrected: true
  covariate_adjustment_option: true
  hybrid_exclusion_threshold: null
```

When `enabled: true`, validate the detection decision and its input/checkpoint
hash before fitting. Refuse activation if the detection result is absent,
stale, biological, batch-dominated or inconclusive.

### Regression outputs

Required artifacts under `results/cell_count_qc/` and the corresponding figure
directory:

- `cell_count_regression_feature_diagnostics.csv`: coefficient, in-sample and
  CV/predictive R², raw/FDR p-values, direction, mode, plate adjustment,
  nonlinear flag and fitting minimum;
- `cell_count_slope_divergence_test.csv`: treatment/control slopes,
  difference, FDR interaction p-value and flag;
- `cell_count_correction_comparison.csv`: the complete four-strategy metric
  table and triggered decision rule;
- `cell_count_corrected_profiles.parquet`: recommended strategy only, never
  overwriting the original;
- `cell_count_correction_report.md`: activation evidence, mode choice,
  affected features/R² distribution, PCA/PERMANOVA changes, mAP decomposition,
  preservation tests, CV results, recommendation and hybrid exclusions.

The feature report must summarize the fraction FDR-significant, the fraction
both significant and `R² > 0.10`, and R² median/P75/P90/P95/maximum.

### Regression acceptance criteria

The full correction is acceptable only when:

1. activation is cryptographically/provenance-linked to a technical-confound
   detection decision;
2. raw and corrected profiles coexist;
3. control-based transportability and global leakage rules are enforced;
4. cross-validated diagnostics accompany every fitted mode;
5. mAP is decomposed and compared with a permutation null;
6. treatment PERMANOVA R² and treatment-distance topology are preserved;
7. the four strategies use identical preprocessing/evaluation definitions;
8. the chosen rule and every exclusion/correction parameter are auditable;
9. synthetic tests cover technical effects, treatment-mediated count effects,
   plate-specific slopes, nonlinear relationships and no-effect data;
10. downstream notebooks record the selected strategy and corrected checkpoint
    hash.

## Implementation phases

### Phase 1 — high evidentiary value

- shared module scaffolding and tests;
- pre/post-feature-selection PCA associations;
- partial correlations;
- multi-term PERMANOVA and Kruskal–Wallis summaries;
- mandatory within-control analyses;
- centroid/replicate reliability;
- threshold mAP and permuted-label null;
- decision report and dry-run exclusion manifest.

### Phase 2 — computational validation and propagation

- single-cell subsampling bootstrap;
- PCA bootstrap null and Procrustes comparisons;
- automatic but explicit application mode;
- final renormalization/feature selection checkpoints;
- downstream input migration;
- single-cell sensitivity analysis.

### Phase 3 — conditional regression correction

- implement the regression modes and nonlinear/plate diagnostics;
- enforce detection-gate activation and stale-decision protection;
- add leakage, slope-transportability and cross-validation tests;
- benchmark uncorrected, excluded, regressed and hybrid strategies;
- add PERMANOVA/mAP decomposition and biological-preservation safeguards;
- add covariate-adjusted alternatives and corrected-profile provenance;
- migrate downstream consumers only after all acceptance criteria pass.

Do not enable `apply_recommended` by default until both phases and their tests
are complete.

## Testing and acceptance criteria

Add unit tests with synthetic data where:

- count is a pure technical main effect;
- count is associated only through treatment interaction;
- count is a plate proxy;
- no count effect exists;
- a low-count control subset is unstable;
- the correct decision is no exclusion.

Test deterministic permutations/bootstrap under a fixed seed, correct
metadata/feature separation, checkpoint non-overwrite behavior, manifest joins
by plate+well, and backward compatibility when the QC configuration is absent.

The implementation is acceptable only when:

1. NB01/NB02 immutable outputs remain available;
2. pre/post-feature-selection conclusions are reported side by side;
3. within-control results are always present or explicitly non-estimable;
4. every term has effect size and permutation p-value;
5. mAP is compared with a permutation null at every valid threshold;
6. PCA change is interpreted against within-threshold variability;
7. lost treatments/replicates block unsafe recommendations;
8. exclusion requires explicit application mode;
9. all excluded wells are reproducible from the manifest;
10. downstream provenance records the QC decision and checkpoint hash.

## Compute and reproducibility notes

This module will be substantially more expensive than the present NB02/NB03.
Cache results by input hash, plate scope, feature set, thresholds, seed and
method parameters. Store the copairs cache in a configured writable location
(`COPAIRS_CACHE_DIR` is already supported by the pipeline). Provide a fast
development mode, but never allow reduced permutations/bootstrap iterations to
be labeled as a final analysis.
