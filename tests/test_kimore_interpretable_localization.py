from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset import KimoreSample  # noqa: E402
from kimore_interpretable_dtw import interpretable_dtw  # noqa: E402
from kimore_interpretable_localization import (  # noqa: E402
    CANDIDATE_INTERPRETATION,
    annotation_queue_rows,
    iter_frame_timeline_rows,
    top_deviation_interval_rows,
)
from kimore_interpretable_quality import FrameQualityResult  # noqa: E402
from kimore_yu_xiong_dtw import YuXiongPreparedSample  # noqa: E402


def prepared_sample(sample_id: str, vectors: np.ndarray) -> YuXiongPreparedSample:
    return YuXiongPreparedSample(
        sample=KimoreSample(
            sample_id=sample_id,
            subject_id=sample_id,
            cohort="synthetic",
            exercise="Es3",
            score=40.0,
            position_path=Path(f"{sample_id}.csv"),
            frames=len(vectors),
            audit_issues="",
        ),
        vectors=vectors,
        required_joints_tracked_fraction=1.0,
    )


def passing_quality(vectors: np.ndarray) -> FrameQualityResult:
    frames = len(vectors)
    return FrameQualityResult(
        repaired_vectors=vectors.copy(),
        cleaned_vectors=vectors.copy(),
        component_summaries=(),
        retained_frame_indices=np.arange(frames),
        interpolated_component_mask=np.zeros((frames, 9), dtype=bool),
        dropped_frame_mask=np.zeros(frames, dtype=bool),
        total_frames=frames,
        interpolated_frames=0,
        interpolated_component_frames=0,
        dropped_frames=0,
        retained_frames=frames,
        retained_fraction=1.0,
        longest_dropped_run=0,
        quality_status="pass",
        quality_reasons=(),
    )


class InterpretableLocalizationTests(unittest.TestCase):
    def test_timeline_and_candidates_preserve_original_frame_location(self) -> None:
        reference_vectors = np.zeros((8, 9, 3), dtype=float)
        reference_vectors[:, :, 0] = 1.0
        sample_vectors = reference_vectors.copy()
        sample_vectors[6:, 3] = (0.0, 1.0, 0.0)
        sample = prepared_sample("sample", sample_vectors)
        reference = prepared_sample("reference", reference_vectors)
        sample_quality = passing_quality(sample_vectors)
        reference_quality = passing_quality(reference_vectors)
        alignment = interpretable_dtw(sample_vectors, reference_vectors)

        timeline = list(
            iter_frame_timeline_rows(
                1,
                sample,
                reference,
                alignment,
                sample_quality,
                reference_quality,
            )
        )

        self.assertEqual(len(timeline), 8 * 9)
        changed = [
            row
            for row in timeline
            if row["original_frame_index"] == 6
            and row["component_index"] == 3
        ]
        self.assertEqual(len(changed), 1)
        self.assertAlmostEqual(changed[0]["mean_angular_deviation_degrees"], 90.0)
        self.assertEqual(changed[0]["interpretation"], CANDIDATE_INTERPRETATION)

        intervals = top_deviation_interval_rows(
            1,
            sample,
            reference,
            alignment,
            sample_quality,
            reference_quality,
            window_count=4,
            limit=3,
        )
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[0]["candidate_rank"], 1)
        self.assertEqual(intervals[0]["component_name"], "right_lower_arm")
        self.assertGreaterEqual(intervals[0]["original_frame_start"], 6)

        queue = annotation_queue_rows(intervals)
        self.assertEqual(
            [row["candidate_id"] for row in queue],
            ["Es3:sample:1", "Es3:sample:2", "Es3:sample:3"],
        )
        self.assertEqual({row["exercise"] for row in queue}, {"Es3"})
        self.assertEqual(queue[0]["review_status"], "unreviewed")
        self.assertEqual(queue[0]["execution_label"], "")


if __name__ == "__main__":
    unittest.main()
