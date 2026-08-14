"""Small reference tests for shared LCP statistical and audit utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hca_pipeline.metrics_qc import mean_average_precision
from hca_pipeline.plotting import categorical_palette
from hca_pipeline.provenance import canonicalize_provenance, validate_provenance_record
from hca_pipeline.stats import _one_factor_permanova, _one_factor_permdisp, cohens_d


class ReferenceStatisticsTests(unittest.TestCase):
    def test_cohens_d_uses_pooled_sample_variance(self):
        observed = cohens_d([2, 4, 6], [1, 2, 3])
        self.assertAlmostEqual(observed, 2 / np.sqrt(2.5), places=12)

    def test_permanova_detects_well_separated_groups_deterministically(self):
        matrix = np.array([[0, 0], [0.1, 0], [0, 0.1], [10, 10], [10.1, 10], [10, 10.1]])
        labels = np.array(["a", "a", "a", "b", "b", "b"])
        first = _one_factor_permanova(matrix, labels, permutations=99, random_state=7)
        second = _one_factor_permanova(matrix, labels, permutations=99, random_state=7)
        self.assertEqual(first, second)
        self.assertGreater(first["R2"], 0.99)
        self.assertLessEqual(first["p_value"], 0.05)

    def test_permdisp_handles_unequal_group_sizes(self):
        matrix = np.array([[0, 0], [0.1, 0], [0, 0.1], [1, 1], [1.1, 1], [1, 1.1], [0.9, 1]])
        labels = np.array(["a", "a", "a", "b", "b", "b", "b"])
        result = _one_factor_permdisp(matrix, labels, permutations=49, random_state=3)
        self.assertTrue(np.isfinite(result["pseudo_F"]))
        self.assertTrue(0 <= result["p_value"] <= 1)

    def test_pc_map_is_perfect_for_exact_replicate_profiles(self):
        profiles = np.array([[1, 0, -1], [1, 0, -1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
        labels = np.array(["a", "a", "b", "b"])
        overall, per_group, per_profile = mean_average_precision(profiles, labels)
        self.assertAlmostEqual(overall, 1.0)
        self.assertTrue(all(value == 1.0 for value in per_group.values()))
        self.assertTrue(all(value == 1.0 for value in per_profile))


class AuditUtilityTests(unittest.TestCase):
    def test_provenance_hash_and_source_notebook(self):
        with tempfile.TemporaryDirectory() as directory:
            dependency = Path(directory) / "per_well_features_selected.parquet"
            dependency.write_bytes(b"reference")
            record = canonicalize_provenance(
                {}, notebook="03_quality_metrics.py", experiment_id="example",
                repo_root=Path.cwd(), dependencies=[dependency], outputs=[],
            )
            validate_provenance_record(record)
            self.assertEqual(
                record["dependencies"][0]["source_notebook"],
                "02_aggregate_normalize_featureselect.py",
            )
            self.assertEqual(len(record["dependencies"][0]["sha256"]), 64)

    def test_categorical_palettes_do_not_repeat(self):
        for count in (10, 20, 25):
            palette = categorical_palette(count)
            self.assertEqual(len(palette), count)
            self.assertEqual(len({tuple(color) for color in palette}), count)


if __name__ == "__main__":
    unittest.main()
