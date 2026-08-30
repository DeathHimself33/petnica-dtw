from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_interpretable_dtw import (  # noqa: E402
    interpretable_dtw,
    summarize_component_errors,
)
from kimore_yu_xiong_dtw import FEATURE_NAMES, yu_xiong_dtw  # noqa: E402


class InterpretableDtwTests(unittest.TestCase):
    def test_identical_motion_has_zero_component_errors(self) -> None:
        vectors = np.zeros((3, 9, 3), dtype=float)
        vectors[:, :, 0] = 1.0

        alignment = interpretable_dtw(vectors, vectors)

        self.assertEqual(alignment.component_errors_degrees.shape, (3, 9))
        self.assertTrue(np.allclose(alignment.component_errors_degrees, 0.0))

    def test_component_errors_identify_the_changed_body_vector(self) -> None:
        reference = np.zeros((2, 9, 3), dtype=float)
        reference[:, :, 0] = 1.0
        sample = reference.copy()
        sample[:, 3] = (0.0, 1.0, 0.0)

        alignment = interpretable_dtw(sample, reference)

        expected = np.zeros((2, 9), dtype=float)
        expected[:, 3] = 90.0
        self.assertTrue(np.allclose(alignment.component_errors_degrees, expected))
        self.assertAlmostEqual(
            float(np.sum(alignment.component_errors_degrees)),
            alignment.total_angular_cost_degrees,
        )

        summaries = summarize_component_errors(alignment)
        self.assertEqual(tuple(item.name for item in summaries), FEATURE_NAMES)
        self.assertAlmostEqual(summaries[3].total_error_degrees, 180.0)
        self.assertAlmostEqual(summaries[3].mean_error_degrees, 90.0)
        self.assertAlmostEqual(summaries[3].maximum_error_degrees, 90.0)
        self.assertAlmostEqual(summaries[3].contribution_percent, 100.0)
        self.assertAlmostEqual(
            sum(item.contribution_percent for item in summaries),
            100.0,
        )
        self.assertTrue(
            all(
                item.contribution_percent == 0.0
                for index, item in enumerate(summaries)
                if index != 3
            )
        )

    def test_zero_error_summary_has_zero_contributions(self) -> None:
        vectors = np.zeros((3, 9, 3), dtype=float)
        vectors[:, :, 0] = 1.0

        summaries = summarize_component_errors(
            interpretable_dtw(vectors, vectors)
        )

        self.assertEqual(tuple(item.name for item in summaries), FEATURE_NAMES)
        for item in summaries:
            self.assertEqual(item.total_error_degrees, 0.0)
            self.assertEqual(item.mean_error_degrees, 0.0)
            self.assertEqual(item.maximum_error_degrees, 0.0)
            self.assertEqual(item.contribution_percent, 0.0)

    def test_path_and_score_match_the_unchanged_yu_xiong_baseline(self) -> None:
        reference = np.zeros((2, 9, 3), dtype=float)
        reference[:, :, 0] = 1.0
        sample = reference.copy()
        sample[1, 8] = (0.0, 1.0, 0.0)

        baseline = yu_xiong_dtw(sample, reference)
        interpretable = interpretable_dtw(sample, reference)

        self.assertTrue(np.array_equal(interpretable.path, baseline.path))
        self.assertAlmostEqual(
            interpretable.total_angular_cost_degrees,
            baseline.total_angular_cost_degrees,
        )
        self.assertAlmostEqual(interpretable.paper_score, baseline.paper_score)


if __name__ == "__main__":
    unittest.main()
