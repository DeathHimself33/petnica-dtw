from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset import (  # noqa: E402
    JOINT_INDEX,
    JOINT_NAMES,
    JointSequence,
    explain_position_exclusion,
    load_joint_positions,
)
from kimore_preprocessing import preprocess_sequence  # noqa: E402
from kimore_tracking_diagnostic import (  # noqa: E402
    centered_moving_median,
    count_tracking_states,
    longest_true_run,
    shoulder_yaw_degrees,
)


def synthetic_sequence() -> JointSequence:
    frames = 7
    positions = np.zeros((frames, len(JOINT_NAMES), 3), dtype=float)
    positions[:, :, 0] = 10.0
    positions[:, :, 1] = 20.0
    positions[:, :, 2] = 30.0
    positions[:, JOINT_INDEX["SpineShoulder"], 1] = 22.0
    positions[:, JOINT_INDEX["ShoulderLeft"], :] = [9.0, 22.0, 30.0]
    positions[:, JOINT_INDEX["ShoulderRight"], :] = [11.0, 22.0, 30.0]
    states = np.full((frames, len(JOINT_NAMES)), 2, dtype=np.int8)
    return JointSequence(positions=positions, tracking_states=states)


class LoaderValidationTests(unittest.TestCase):
    def write_recording(self, tracking_value: float, coordinate: float = 1.0) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            delete=False,
        )
        path = Path(temporary.name)
        with temporary:
            row = []
            for _ in JOINT_NAMES:
                row.extend((coordinate, 2.0, 3.0, tracking_value))
            csv.writer(temporary).writerow(row)
        self.addCleanup(path.unlink)
        return path

    def test_loader_rejects_fractional_tracking_state(self) -> None:
        path = self.write_recording(1.5)
        with self.assertRaisesRegex(ValueError, r"expected only 0, 1 or 2"):
            load_joint_positions(path)

    def test_loader_rejects_nonfinite_coordinate(self) -> None:
        path = self.write_recording(2.0, coordinate=float("nan"))
        with self.assertRaisesRegex(ValueError, r"non-finite coordinates"):
            load_joint_positions(path)

    def test_manifest_rejects_audited_nonnumeric_position_rows(self) -> None:
        reason = explain_position_exclusion(
            {
                "clinical_ts": "40",
                "position_path": "JointPosition.csv",
                "position_frames": "10",
                "position_columns": "100",
                "issues": "position has 1 nonnumeric rows",
            }
        )
        self.assertEqual(reason, "JointPosition contains nonnumeric rows")


class PreprocessingTests(unittest.TestCase):
    def test_preprocessing_centres_scales_and_preserves_input(self) -> None:
        sequence = synthetic_sequence()
        original_positions = sequence.positions.copy()
        processed = preprocess_sequence(sequence)

        self.assertAlmostEqual(processed.body_scale, 2.0)
        self.assertTrue(
            np.allclose(processed.positions[:, JOINT_INDEX["SpineBase"], :], 0.0)
        )
        self.assertTrue(np.array_equal(sequence.positions, original_positions))
        self.assertTrue(
            np.array_equal(processed.tracking_states, sequence.tracking_states)
        )

    def test_tracking_counts_use_kinect_state_meanings(self) -> None:
        counts = count_tracking_states(np.asarray([[2, 2, 1], [0, 2, 1]]))
        self.assertEqual((counts.tracked, counts.inferred, counts.untracked), (3, 2, 1))

    def test_moving_median_removes_an_isolated_spike(self) -> None:
        signal = np.asarray([0.0, 0.0, 20.0, 0.0, 0.0])
        self.assertEqual(centered_moving_median(signal, window=5).tolist(), [0.0] * 5)

    def test_longest_true_run_handles_internal_and_empty_runs(self) -> None:
        self.assertEqual(
            longest_true_run(np.asarray([False, True, True, False, True])),
            2,
        )
        self.assertEqual(longest_true_run(np.zeros(4, dtype=bool)), 0)

    def test_shoulder_yaw_is_translation_and_scale_invariant(self) -> None:
        sequence = synthetic_sequence()
        transformed = JointSequence(
            positions=sequence.positions * 3.5 + np.asarray([4.0, -2.0, 9.0]),
            tracking_states=sequence.tracking_states.copy(),
        )
        self.assertTrue(
            np.allclose(
                shoulder_yaw_degrees(sequence),
                shoulder_yaw_degrees(transformed),
            )
        )


if __name__ == "__main__":
    unittest.main()
