from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_ml_data import (  # noqa: E402
    EXERCISE_TO_INDEX,
    FeatureStandardizer,
    apply_feature_standardizer,
    fit_feature_standardizer,
    fold_train_test_indices,
    uniform_resample,
)
from kimore_yu_xiong_dtw import FEATURE_DIMENSIONS  # noqa: E402


class MLDataTests(unittest.TestCase):
    def test_exercise_indices_are_stable(self) -> None:
        self.assertEqual(
            EXERCISE_TO_INDEX,
            {"Es1": 0, "Es2": 1, "Es3": 2, "Es4": 3, "Es5": 4},
        )

    def test_uniform_resample_preserves_masks_and_zeros_unusable_frames(self) -> None:
        vectors = np.ones((5, FEATURE_DIMENSIONS, 3), dtype=np.float64)
        vectors *= np.arange(1, 6)[:, np.newaxis, np.newaxis]
        usable = np.asarray([True, True, False, True, True])
        observed = np.ones((5, FEATURE_DIMENSIONS), dtype=bool)
        observed[4, 2] = False

        result = uniform_resample(vectors, usable, observed, target_frames=3)

        self.assertEqual(result.source_frame_indices.tolist(), [0, 2, 4])
        self.assertEqual(result.frame_mask.tolist(), [True, False, True])
        np.testing.assert_array_equal(result.vectors[1], 0.0)
        self.assertFalse(result.component_observed_mask[1].any())
        self.assertFalse(result.component_observed_mask[2, 2])

    def test_split_rejects_subject_leakage_in_fold_metadata(self) -> None:
        subjects = ("A", "A", "B", "C")
        invalid_folds = np.asarray([1, 2, 1, 2])

        with self.assertRaisesRegex(AssertionError, "Subject leakage"):
            fold_train_test_indices(subjects, invalid_folds, test_fold=1)

    def test_split_returns_complete_subject_disjoint_indices(self) -> None:
        subjects = ("A", "A", "B", "C")
        folds = np.asarray([1, 1, 2, 2])

        train, test = fold_train_test_indices(subjects, folds, test_fold=1)

        self.assertEqual(train.tolist(), [2, 3])
        self.assertEqual(test.tolist(), [0, 1])

    def test_standardizer_uses_training_rows_and_valid_frames_only(self) -> None:
        features = np.zeros((3, 2, FEATURE_DIMENSIONS, 3), dtype=np.float32)
        features[0] = 1.0
        features[1] = 3.0
        features[2] = 1000.0
        frame_mask = np.ones((3, 2), dtype=bool)
        frame_mask[1, 1] = False

        standardizer = fit_feature_standardizer(
            features, frame_mask, train_indices=[0, 1]
        )

        np.testing.assert_allclose(standardizer.mean, 5.0 / 3.0)
        self.assertTrue(np.all(standardizer.mean < 2.0))
        transformed = apply_feature_standardizer(
            features, frame_mask, standardizer
        )
        np.testing.assert_array_equal(transformed[1, 1], 0.0)
        self.assertTrue(np.isfinite(transformed).all())

    def test_invalid_standardizer_scale_is_rejected(self) -> None:
        features = np.ones((1, 2, FEATURE_DIMENSIONS, 3), dtype=np.float32)
        frame_mask = np.ones((1, 2), dtype=bool)
        invalid = FeatureStandardizer(
            mean=np.zeros((FEATURE_DIMENSIONS, 3)),
            scale=np.zeros((FEATURE_DIMENSIONS, 3)),
        )

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            apply_feature_standardizer(features, frame_mask, invalid)


if __name__ == "__main__":
    unittest.main()
