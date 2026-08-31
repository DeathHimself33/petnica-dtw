from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset import KimoreSample  # noqa: E402
from kimore_grouping import (  # noqa: E402
    assert_no_subject_leakage,
    make_subject_folds,
    subject_groups,
)


def sample(sample_id: str, subject_id: str) -> KimoreSample:
    return KimoreSample(
        sample_id=sample_id,
        subject_id=subject_id,
        cohort="synthetic",
        exercise="Es3",
        score=30.0,
        position_path=Path(f"{sample_id}.csv"),
        frames=100,
        audit_issues="",
    )


class SubjectGroupingTests(unittest.TestCase):
    def test_subject_groups_follow_sample_order(self) -> None:
        samples = [
            sample("A_Es3", "A"),
            sample("B_Es3", "B"),
            sample("A_Es4", "A"),
        ]

        self.assertEqual(subject_groups(samples).tolist(), ["A", "B", "A"])

    def test_group_folds_keep_each_subject_together(self) -> None:
        samples = [
            sample(f"{subject}_{exercise}", subject)
            for subject in ("A", "B", "C", "D")
            for exercise in ("Es3", "Es4")
        ]
        groups = subject_groups(samples)
        folds = make_subject_folds(samples, n_splits=4)
        tested_indices: set[int] = set()

        for fold in folds:
            assert_no_subject_leakage(
                groups,
                fold.train_indices,
                fold.test_indices,
            )
            tested_indices.update(int(index) for index in fold.test_indices)

        self.assertEqual(tested_indices, set(range(len(samples))))

    def test_overlap_is_reported(self) -> None:
        groups = ["A", "A", "B", "C"]

        with self.assertRaisesRegex(AssertionError, r"training and testing: A"):
            assert_no_subject_leakage(
                groups,
                train_indices=[0, 2],
                test_indices=[1, 3],
            )

    def test_too_many_splits_are_rejected(self) -> None:
        samples = [
            sample("A_Es3", "A"),
            sample("B_Es3", "B"),
        ]

        with self.assertRaisesRegex(ValueError, r"exceeds the 2 unique subjects"):
            make_subject_folds(samples, n_splits=3)

    def test_equal_groups_match_deterministic_group_kfold_allocation(self) -> None:
        samples = [sample(f"{subject}_Es3", subject) for subject in "ABCDE"]

        folds = make_subject_folds(samples, n_splits=2)

        self.assertEqual(folds[0].test_indices.tolist(), [0, 2, 4])
        self.assertEqual(folds[1].test_indices.tolist(), [1, 3])


if __name__ == "__main__":
    unittest.main()
