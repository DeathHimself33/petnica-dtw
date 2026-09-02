from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_evaluation import (  # noqa: E402
    alignment_warp_diagnostics,
    bootstrap_improvement_intervals,
    bootstrap_metric_intervals,
    bootstrap_paired_metric_improvements,
    regression_metrics,
    run_cross_validated_evaluation,
    training_cohort_constant_values,
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

    def test_spearman_uses_average_ranks_for_ties(self) -> None:
        metrics = regression_metrics(
            np.asarray([1.0, 2.0, 2.0, 3.0]),
            np.asarray([1.0, 2.0, 3.0, 4.0]),
        )

        self.assertAlmostEqual(metrics["spearman"], 0.9486832980505138)

    def test_training_constants_do_not_use_test_scores(self) -> None:
        scores = np.asarray([10.0, 20.0, 1000.0])

        median, mean = training_constant_values(scores, train_indices=[0, 1])

        self.assertEqual(median, 15.0)
        self.assertEqual(mean, 15.0)

    def test_cohort_constants_use_matching_training_rows_only(self) -> None:
        scores = np.asarray([10.0, 20.0, 30.0, 1000.0])
        medians, means = training_cohort_constant_values(
            scores,
            cohorts=["A", "A", "B", "A"],
            train_indices=[0, 1, 2],
            test_indices=[3, 2],
        )

        self.assertEqual(medians.tolist(), [15.0, 30.0])
        self.assertEqual(means.tolist(), [15.0, 30.0])

    def test_warp_diagnostics_count_repeated_alignment_step(self) -> None:
        diagnostic = alignment_warp_diagnostics(
            sample_frames=4,
            reference_frames=3,
            path_length=4,
        )

        self.assertAlmostEqual(diagnostic["non_diagonal_step_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(
            diagnostic["minimum_required_non_diagonal_step_fraction"],
            1.0 / 3.0,
        )

    def test_evaluation_rejects_exercises_other_than_es3_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "defined only for Es3"):
            run_cross_validated_evaluation(
                manifest_path=Path("missing.csv"),
                exercise="Es2",
                output_dir=Path("unused-results"),
                figure_dir=Path("unused-figures"),
                bootstrap_resamples=1,
            )


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

    def test_paired_bootstrap_uses_positive_values_for_candidate_gains(self) -> None:
        actual = np.asarray([0.0, 1.0, 2.0, 3.0])
        candidate = actual.copy()
        comparator = actual + 1.0

        result = bootstrap_paired_metric_improvements(
            actual,
            candidate,
            comparator,
            groups=["A", "B", "C", "D"],
            resamples=50,
            seed=123,
        )

        self.assertAlmostEqual(result["mae_reduction"]["estimate"], 1.0)
        self.assertAlmostEqual(result["rmse_reduction"]["estimate"], 1.0)
        self.assertGreater(result["mae_reduction"]["valid_resamples"], 0)


if __name__ == "__main__":
    unittest.main()
