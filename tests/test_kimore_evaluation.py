from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_evaluation import (  # noqa: E402
    bootstrap_improvement_intervals,
    bootstrap_metric_intervals,
    regression_metrics,
    training_constant_values,
    validate_oof_indices,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_regression_metrics_have_expected_values(self) -> None:
        metrics = regression_metrics(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 3.0]),
        )

        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.0))
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertAlmostEqual(metrics["pearson"], 1.0)

    def test_constant_predictions_have_undefined_correlations(self) -> None:
        metrics = regression_metrics(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([2.0, 2.0, 2.0]),
        )

        self.assertTrue(math.isnan(metrics["spearman"]))
        self.assertTrue(math.isnan(metrics["pearson"]))

    def test_training_constants_do_not_use_test_scores(self) -> None:
        scores = np.asarray([10.0, 20.0, 1000.0])

        median, mean = training_constant_values(scores, train_indices=[0, 1])

        self.assertEqual(median, 15.0)
        self.assertEqual(mean, 15.0)


class EvaluationSafetyTests(unittest.TestCase):
    def test_oof_coverage_accepts_each_index_once(self) -> None:
        validate_oof_indices([[0, 2], [1], [3, 4]], sample_count=5)

    def test_oof_coverage_rejects_repeated_and_missing_indices(self) -> None:
        with self.assertRaisesRegex(AssertionError, "missing=.*3.*repeated=.*1"):
            validate_oof_indices([[0, 1], [1, 2]], sample_count=4)

    def test_subject_bootstrap_is_deterministic(self) -> None:
        actual = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
        predicted = np.asarray([11.0, 18.0, 33.0, 39.0, 48.0])
        groups = ["A", "B", "C", "D", "E"]

        first = bootstrap_metric_intervals(
            actual,
            predicted,
            groups,
            resamples=100,
            seed=123,
        )
        second = bootstrap_metric_intervals(
            actual,
            predicted,
            groups,
            resamples=100,
            seed=123,
        )

        self.assertEqual(first, second)
        self.assertLessEqual(first["mae"]["low"], first["mae"]["high"])

    def test_improvement_is_positive_for_a_perfect_model(self) -> None:
        actual = np.asarray([10.0, 20.0, 30.0, 40.0])
        constant = np.full(4, 25.0)
        improvements = bootstrap_improvement_intervals(
            actual,
            model_predictions=actual.copy(),
            median_predictions=constant,
            mean_predictions=constant,
            groups=["A", "B", "C", "D"],
            resamples=50,
            seed=456,
        )

        self.assertGreater(
            improvements["mae_reduction_vs_training_median"]["estimate"],
            0.0,
        )
        self.assertGreater(
            improvements["rmse_reduction_vs_training_mean"]["estimate"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
