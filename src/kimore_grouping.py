"""Create subject-wise evaluation folds and detect subject leakage.

KIMORE recordings from the same person must stay on the same side of an
evaluation split.  Otherwise a model could be trained on one recording from a
person and tested on another recording from that same person.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from kimore_dataset import KimoreSample, read_manifest


@dataclass(frozen=True)
class SubjectFold:
    number: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def subject_groups(samples: Sequence[KimoreSample]) -> np.ndarray:
    """Return one subject ID for every sample, in the same order."""
    return np.asarray(
        [sample.subject_id for sample in samples],
        dtype=object,
    )


def assert_no_subject_leakage(
    groups: Sequence[str],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> None:
    """Raise an AssertionError if a subject occurs on both sides of a split."""
    group_array = np.asarray(groups, dtype=object)
    train_subjects = set(group_array[np.asarray(train_indices, dtype=int)])
    test_subjects = set(group_array[np.asarray(test_indices, dtype=int)])
    overlap = train_subjects.intersection(test_subjects)

    if overlap:
        overlapping_ids = ", ".join(sorted(overlap))
        raise AssertionError(
            "Subject leakage detected; present in training and testing: "
            f"{overlapping_ids}"
        )


def make_subject_folds(
    samples: Sequence[KimoreSample],
    n_splits: int = 5,
) -> list[SubjectFold]:
    """Split samples while keeping every subject entirely within one group."""
    if not samples:
        raise ValueError("Cannot create folds from an empty sample list")

    groups = subject_groups(samples)
    subject_count = len(set(groups))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > subject_count:
        raise ValueError(
            f"n_splits={n_splits} exceeds the {subject_count} unique subjects"
        )

    sample_indices = np.arange(len(samples))
    splitter = GroupKFold(n_splits=n_splits)
    folds: list[SubjectFold] = []

    for number, (train_indices, test_indices) in enumerate(
        splitter.split(sample_indices, groups=groups),
        start=1,
    ):
        assert_no_subject_leakage(
            groups,
            train_indices,
            test_indices,
        )
        folds.append(
            SubjectFold(
                number=number,
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )

    return folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to kimore_manifest.csv",
    )
    parser.add_argument(
        "--exercise",
        default="Es3",
        choices=[f"Es{i}" for i in range(1, 6)],
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of subject-wise folds (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, excluded = read_manifest(
        args.manifest.expanduser().resolve(),
        args.exercise,
    )
    groups = subject_groups(samples)
    folds = make_subject_folds(samples, args.splits)

    print(f"Exercise: {args.exercise}")
    print(f"Usable samples: {len(samples)}")
    print(f"Unique subjects: {len(set(groups))}")
    print(f"Manifest exclusions: {len(excluded)}")

    for fold in folds:
        train_subjects = set(groups[fold.train_indices])
        test_subjects = set(groups[fold.test_indices])
        print(
            f"Fold {fold.number}: "
            f"train={len(fold.train_indices)} samples/"
            f"{len(train_subjects)} subjects, "
            f"test={len(fold.test_indices)} samples/"
            f"{len(test_subjects)} subjects, "
            "overlap=0"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
