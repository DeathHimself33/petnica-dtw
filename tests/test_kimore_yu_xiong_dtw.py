from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset import JOINT_INDEX, JOINT_NAMES, JointSequence, KimoreSample  # noqa: E402
from kimore_yu_xiong_dtw import (  # noqa: E402
    YuXiongPreparedSample,
    select_yu_xiong_reference,
    yu_xiong_dtw,
    yu_xiong_vectors,
)


def synthetic_sequence(yaw_degrees: float = 0.0, frames: int = 3) -> JointSequence:
    pose = np.zeros((len(JOINT_NAMES), 3), dtype=np.float64)
    pose[JOINT_INDEX["SpineBase"]] = [0.0, 0.75, 0.0]
    pose[JOINT_INDEX["SpineShoulder"]] = [0.0, 1.35, 0.0]
    pose[JOINT_INDEX["ShoulderLeft"]] = [-0.4, 1.35, 0.0]
    pose[JOINT_INDEX["ElbowLeft"]] = [-0.72, 1.02, 0.08]
    pose[JOINT_INDEX["WristLeft"]] = [-0.86, 0.68, 0.18]
    pose[JOINT_INDEX["ShoulderRight"]] = [0.4, 1.35, 0.0]
    pose[JOINT_INDEX["ElbowRight"]] = [0.72, 1.02, -0.08]
    pose[JOINT_INDEX["WristRight"]] = [0.86, 0.68, -0.18]
    pose[JOINT_INDEX["HipLeft"]] = [-0.24, 0.75, 0.0]
    pose[JOINT_INDEX["KneeLeft"]] = [-0.25, 0.38, 0.06]
    pose[JOINT_INDEX["AnkleLeft"]] = [-0.25, 0.04, 0.11]
    pose[JOINT_INDEX["HipRight"]] = [0.24, 0.75, 0.0]
    pose[JOINT_INDEX["KneeRight"]] = [0.25, 0.38, -0.06]
    pose[JOINT_INDEX["AnkleRight"]] = [0.25, 0.04, -0.11]

    radians = math.radians(yaw_degrees)
    rotation = np.asarray(
        [
            [math.cos(radians), 0.0, math.sin(radians)],
            [0.0, 1.0, 0.0],
            [-math.sin(radians), 0.0, math.cos(radians)],
        ]
    )
    rotated = pose @ rotation.T
    positions = np.repeat(rotated[np.newaxis, :, :], frames, axis=0)
    tracking = np.full((frames, len(JOINT_NAMES)), 2, dtype=np.int8)
    return JointSequence(positions=positions, tracking_states=tracking)


def prepared(sample_id: str, score: float, tracked: float) -> YuXiongPreparedSample:
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
    return YuXiongPreparedSample(
        sample=sample,
        vectors=yu_xiong_vectors(synthetic_sequence()),
        required_joints_tracked_fraction=tracked,
    )


class YuXiongFeatureTests(unittest.TestCase):
    def test_feature_shape_and_unit_lengths(self) -> None:
        vectors = yu_xiong_vectors(synthetic_sequence(frames=4))

        self.assertEqual(vectors.shape, (4, 9, 3))
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=2), 1.0))

    def test_global_yaw_changes_only_body_forward_feature(self) -> None:
        original = yu_xiong_vectors(synthetic_sequence(yaw_degrees=0.0))
        rotated = yu_xiong_vectors(synthetic_sequence(yaw_degrees=30.0))

        self.assertTrue(np.allclose(original[:, :8], rotated[:, :8], atol=1e-12))
        forward_dots = np.sum(original[:, 8] * rotated[:, 8], axis=1)
        self.assertTrue(np.allclose(forward_dots, math.cos(math.radians(30.0))))

    def test_zero_length_limb_is_rejected(self) -> None:
        sequence = synthetic_sequence()
        sequence.positions[:, JOINT_INDEX["WristLeft"]] = sequence.positions[
            :, JOINT_INDEX["ElbowLeft"]
        ]

        with self.assertRaisesRegex(ValueError, "zero-length"):
            yu_xiong_vectors(sequence)


class YuXiongDtwTests(unittest.TestCase):
    def test_identical_motion_has_diagonal_path_and_perfect_score(self) -> None:
        vectors = yu_xiong_vectors(synthetic_sequence(frames=3))
        alignment = yu_xiong_dtw(vectors, vectors)

        self.assertAlmostEqual(alignment.total_angular_cost_degrees, 0.0)
        self.assertAlmostEqual(alignment.paper_score, 100.0)
        self.assertEqual(alignment.path.tolist(), [[0, 0], [1, 1], [2, 2]])

    def test_body_yaw_difference_uses_ninth_dimension_in_score(self) -> None:
        sample = yu_xiong_vectors(synthetic_sequence(yaw_degrees=30.0))
        reference = yu_xiong_vectors(synthetic_sequence(yaw_degrees=0.0))
        alignment = yu_xiong_dtw(sample, reference)

        expected = 100.0 * (1.0 - 30.0 / (90.0 * 9.0))
        self.assertAlmostEqual(alignment.mean_angle_degrees, 30.0 / 9.0)
        self.assertAlmostEqual(alignment.paper_score, expected)

    def test_score_retains_raw_value_and_bounds_public_percentage(self) -> None:
        sample = np.zeros((1, 9, 3), dtype=float)
        sample[:, :, 0] = 1.0
        reference = -sample
        alignment = yu_xiong_dtw(sample, reference)

        self.assertAlmostEqual(alignment.paper_score_unclipped, -100.0)
        self.assertEqual(alignment.paper_score, 0.0)


class YuXiongReferenceTests(unittest.TestCase):
    def test_reference_uses_highest_training_score(self) -> None:
        samples = [
            prepared("train_reliable", 45.0, 1.0),
            prepared("train_higher_but_unreliable", 50.0, 0.9),
            prepared("test_highest", 50.0, 1.0),
        ]

        selected = select_yu_xiong_reference(samples, train_indices=[0, 1])

        self.assertEqual(selected, 1)

    def test_tracking_fraction_resolves_equal_score(self) -> None:
        samples = [
            prepared("less_tracked", 50.0, 0.8),
            prepared("better_tracked", 50.0, 0.9),
        ]

        selected = select_yu_xiong_reference(samples, train_indices=[0, 1])

        self.assertEqual(selected, 1)


if __name__ == "__main__":
    unittest.main()
