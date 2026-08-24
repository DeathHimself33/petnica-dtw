from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset import KimoreSample  # noqa: E402
from kimore_dtw import exact_dtw  # noqa: E402
from kimore_plain_dtw import (  # noqa: E402
    PreparedSample,
    fit_linear_calibration,
    select_training_reference,
)


def prepared_sample(
    sample_id: str,
    score: float,
    tracked_fraction: float,
) -> PreparedSample:
    sample = KimoreSample(
        sample_id=sample_id,
        subject_id=sample_id,
        cohort="synthetic",
        exercise="Es3",
        score=score,
        position_path=Path(f"{sample_id}.csv"),
        frames=3,
        audit_issues="",
    )
    return PreparedSample(
        sample=sample,
        feature=np.asarray([[0.0], [1.0], [0.0]]),
        shoulder_tracked_fraction=tracked_fraction,
    )


class ExactDtwTests(unittest.TestCase):
    def test_identical_sequences_have_zero_distance_and_diagonal_path(self) -> None:
        values = np.asarray([0.0, 1.0, 2.0])
        alignment = exact_dtw(values, values)

        self.assertEqual(alignment.aligned_rmse, 0.0)
        self.assertEqual(alignment.path.tolist(), [[0, 0], [1, 1], [2, 2]])

    def test_dtw_aligns_a_repeated_value_without_cost(self) -> None:
        sample = np.asarray([0.0, 0.0, 1.0, 2.0])
        reference = np.asarray([0.0, 1.0, 2.0])
        alignment = exact_dtw(sample, reference)

        self.assertEqual(alignment.aligned_rmse, 0.0)
        self.assertEqual(alignment.path[0].tolist(), [0, 0])
        self.assertEqual(alignment.path[-1].tolist(), [3, 2])
        self.assertEqual(len(alignment.path), 4)

    def test_nonfinite_features_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            exact_dtw(np.asarray([0.0, np.nan]), np.asarray([0.0, 1.0]))


class PlainBaselineTests(unittest.TestCase):
    def test_reference_is_selected_from_training_indices_only(self) -> None:
        samples = [
            prepared_sample("train_reliable", 45.0, 1.0),
            prepared_sample("train_best_tracking", 50.0, 0.9),
            prepared_sample("train_worse_tracking", 50.0, 0.7),
            prepared_sample("test_higher", 60.0, 1.0),
        ]

        selected = select_training_reference(samples, train_indices=[0, 1, 2])

        self.assertEqual(selected, 0)

    def test_reference_falls_back_to_best_tracking_fraction(self) -> None:
        samples = [
            prepared_sample("higher_score", 50.0, 0.80),
            prepared_sample("best_tracking", 45.0, 0.95),
            prepared_sample("also_best_tracking", 40.0, 0.95),
        ]

        selected = select_training_reference(samples, train_indices=[0, 1, 2])

        self.assertEqual(selected, 1)

    def test_linear_calibration_recovers_known_mapping(self) -> None:
        distances = np.asarray([0.0, 1.0, 2.0, 3.0])
        scores = 50.0 - 4.0 * distances
        calibration = fit_linear_calibration(distances, scores)

        self.assertAlmostEqual(calibration.intercept, 50.0)
        self.assertAlmostEqual(calibration.slope, -4.0)
        self.assertTrue(
            np.allclose(calibration.predict(np.asarray([1.5])), np.asarray([44.0]))
        )


if __name__ == "__main__":
    unittest.main()
