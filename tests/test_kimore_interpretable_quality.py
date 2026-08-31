from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset import (  # noqa: E402
    JOINT_INDEX,
    JOINT_NAMES,
    JointSequence,
    KimoreSample,
)
from kimore_interpretable_evaluation import _quality_aware_reference  # noqa: E402
from kimore_interpretable_quality import (  # noqa: E402
    apply_frame_quality_control,
    assess_component_quality,
    overall_quality_status,
)
from kimore_yu_xiong_dtw import (  # noqa: E402
    FEATURE_NAMES,
    YuXiongPreparedSample,
    yu_xiong_vectors,
)


def standing_sequence(frames: int = 5) -> JointSequence:
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
    positions = np.repeat(pose[np.newaxis, :, :], frames, axis=0)
    tracking = np.full((frames, len(JOINT_NAMES)), 2, dtype=np.int8)
    return JointSequence(positions=positions, tracking_states=tracking)


def reference_candidate(sample_id: str, score: float) -> YuXiongPreparedSample:
    sample = KimoreSample(
        sample_id=sample_id,
        subject_id=sample_id,
        cohort="synthetic",
        exercise="Es3",
        score=score,
        position_path=Path(f"{sample_id}.csv"),
        frames=5,
        audit_issues="",
    )
    return YuXiongPreparedSample(
        sample=sample,
        vectors=yu_xiong_vectors(standing_sequence()),
        required_joints_tracked_fraction=1.0,
    )


class InterpretableQualityTests(unittest.TestCase):
    def test_failed_sample_cannot_be_selected_as_reference(self) -> None:
        candidates = [
            reference_candidate("failed_high_score", 50.0),
            reference_candidate("usable_lower_score", 45.0),
        ]

        selected = _quality_aware_reference(
            candidates,
            np.asarray([0, 1]),
            ["fail", "warning"],
        )

        self.assertEqual(selected, 1)

    def test_valid_standing_motion_passes_all_nine_components(self) -> None:
        sequence = standing_sequence()

        quality = assess_component_quality(
            sequence,
            yu_xiong_vectors(sequence),
            exercise="Es3",
        )

        self.assertEqual(tuple(item.component_name for item in quality), FEATURE_NAMES)
        self.assertTrue(all(item.quality_status == "pass" for item in quality))
        self.assertEqual(overall_quality_status(quality), "pass")

    def test_inverted_upper_leg_fails_geometry_even_when_tracked(self) -> None:
        sequence = standing_sequence()
        hip = sequence.positions[:, JOINT_INDEX["HipLeft"]]
        sequence.positions[:, JOINT_INDEX["KneeLeft"]] = hip + (0.0, 0.3, 0.0)

        quality = assess_component_quality(
            sequence,
            yu_xiong_vectors(sequence),
            exercise="Es3",
        )
        left_upper_leg = quality[4]

        self.assertEqual(left_upper_leg.tracked_fraction, 1.0)
        self.assertEqual(left_upper_leg.anatomical_plausibility_fraction, 0.0)
        self.assertEqual(left_upper_leg.quality_status, "fail")
        self.assertIn(
            "anatomically_implausible_direction",
            left_upper_leg.quality_reasons,
        )
        self.assertEqual(quality[0].quality_status, "pass")
        self.assertEqual(overall_quality_status(quality), "fail")

    def test_tracking_check_applies_to_arm_components(self) -> None:
        sequence = standing_sequence()
        sequence.tracking_states[:, JOINT_INDEX["ElbowLeft"]] = 1

        quality = assess_component_quality(
            sequence,
            yu_xiong_vectors(sequence),
            exercise="Es3",
        )

        for component_index in (0, 1):
            self.assertEqual(quality[component_index].tracked_fraction, 0.0)
            self.assertEqual(quality[component_index].quality_status, "fail")
            self.assertIn(
                "tracking_below_10_percent",
                quality[component_index].quality_reasons,
            )
        self.assertEqual(quality[2].quality_status, "pass")

    def test_body_forward_detects_large_frame_to_frame_flip(self) -> None:
        sequence = standing_sequence(frames=7)
        rotation = np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        sequence.positions[3] = sequence.positions[3] @ rotation.T

        vectors = yu_xiong_vectors(sequence)
        original_vectors = vectors.copy()
        result = apply_frame_quality_control(
            sequence,
            vectors,
            exercise="Es3",
        )
        body_forward = result.component_summaries[8]

        self.assertAlmostEqual(body_forward.temporal_continuity_fraction, 2 / 3)
        self.assertEqual(body_forward.temporal_outlier_frames, 1)
        self.assertEqual(body_forward.interpolated_frames, 1)
        self.assertEqual(body_forward.unrepaired_invalid_frames, 0)
        self.assertEqual(body_forward.quality_status, "warning")
        self.assertIn(
            "short_frame_gaps_interpolated",
            body_forward.quality_reasons,
        )
        self.assertEqual(result.dropped_frames, 0)
        self.assertEqual(result.interpolated_frames, 1)
        self.assertEqual(result.retained_frames, 7)
        self.assertTrue(np.array_equal(vectors, original_vectors))
        self.assertTrue(
            np.allclose(np.linalg.norm(result.cleaned_vectors, axis=2), 1.0)
        )
        self.assertTrue(
            all(
                item.body_frame_invalid_frames == 1
                for item in result.component_summaries
            )
        )

    def test_short_internal_tracking_gap_is_interpolated(self) -> None:
        sequence = standing_sequence(frames=9)
        knee = JOINT_INDEX["KneeLeft"]
        sequence.tracking_states[4, knee] = 0
        sequence.positions[4, knee] = sequence.positions[4, JOINT_INDEX["HipLeft"]] + (
            0.0,
            0.3,
            0.0,
        )

        vectors = yu_xiong_vectors(sequence)
        original_vectors = vectors.copy()
        result = apply_frame_quality_control(
            sequence,
            vectors,
            exercise="Es3",
        )

        for component_index in (4, 5):
            component = result.component_summaries[component_index]
            self.assertEqual(component.untracked_frames, 1)
            self.assertEqual(component.interpolated_frames, 1)
            self.assertEqual(component.unrepaired_invalid_frames, 0)
            self.assertTrue(
                np.allclose(
                    result.repaired_vectors[4, component_index],
                    result.repaired_vectors[3, component_index],
                )
            )
        self.assertEqual(result.quality_status, "warning")
        self.assertEqual(result.dropped_frames, 0)
        self.assertEqual(result.cleaned_vectors.shape, (9, 9, 3))
        self.assertTrue(np.array_equal(vectors, original_vectors))
        self.assertTrue(
            np.allclose(np.linalg.norm(result.cleaned_vectors, axis=2), 1.0)
        )

    def test_unrepairable_edge_frame_is_removed_without_rejecting_recording(self) -> None:
        sequence = standing_sequence(frames=20)
        sequence.tracking_states[0, JOINT_INDEX["KneeLeft"]] = 0

        result = apply_frame_quality_control(
            sequence,
            yu_xiong_vectors(sequence),
            exercise="Es3",
        )

        self.assertEqual(result.quality_status, "warning")
        self.assertEqual(result.interpolated_frames, 0)
        self.assertEqual(result.dropped_frames, 1)
        self.assertEqual(result.retained_frames, 19)
        self.assertEqual(result.longest_dropped_run, 1)
        self.assertEqual(result.cleaned_vectors.shape, (19, 9, 3))

    def test_long_invalid_run_still_rejects_recording(self) -> None:
        sequence = standing_sequence(frames=12)
        sequence.tracking_states[2:10, JOINT_INDEX["KneeLeft"]] = 0

        result = apply_frame_quality_control(
            sequence,
            yu_xiong_vectors(sequence),
            exercise="Es3",
        )

        self.assertEqual(result.quality_status, "fail")
        self.assertEqual(result.interpolated_frames, 0)
        self.assertEqual(result.dropped_frames, 8)
        self.assertEqual(result.retained_frames, 4)
        self.assertEqual(result.longest_dropped_run, 8)
        self.assertIn(
            "retained_frame_fraction_below_80_percent",
            result.quality_reasons,
        )


if __name__ == "__main__":
    unittest.main()
