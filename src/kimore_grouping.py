"""Create subject-wise evaluation folds and detect subject leakage.

KIMORE recordings from the same person must stay on the same side of an
evaluation split.  Otherwise a model could be trained on one recording from a
person and tested on another recording from that same person.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

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


def make_subject_fold_assignments(
    samples: Sequence[KimoreSample],
    n_splits: int = 5,
) -> dict[str, int]:
    """Assign every subject to one zero-based fold.

    Build this mapping from the union of all exercises when several exercises
    will be evaluated or used by one model.  Reusing the returned mapping then
    guarantees that a subject cannot be in a training fold for one exercise
    and the corresponding test fold for another exercise.
    """
    if not samples:
        raise ValueError("Cannot create folds from an empty sample list")

    groups = subject_groups(samples)
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    subject_count = len(unique_groups)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > subject_count:
        raise ValueError(
            f"n_splits={n_splits} exceeds the {subject_count} unique subjects"
        )

    samples_per_group = np.bincount(group_indices)
    # Match deterministic ``GroupKFold(shuffle=False)`` allocation without
    # importing scikit-learn.  Heavier subjects are placed first in the
    # currently lightest fold; ties are deterministic.
    largest_first = np.argsort(samples_per_group, kind="stable")[::-1]
    samples_per_fold = np.zeros(n_splits, dtype=np.int64)
    group_to_fold = np.zeros(len(unique_groups), dtype=np.int64)
    for group_index in largest_first:
        lightest_fold = int(np.argmin(samples_per_fold))
        samples_per_fold[lightest_fold] += samples_per_group[group_index]
        group_to_fold[group_index] = lightest_fold

    return {
        str(group): int(group_to_fold[index])
        for index, group in enumerate(unique_groups)
    }


def make_subject_folds(
    samples: Sequence[KimoreSample],
    n_splits: int = 5,
    subject_fold_assignments: Mapping[str, int] | None = None,
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
    assignments = (
        make_subject_fold_assignments(samples, n_splits)
        if subject_fold_assignments is None
        else dict(subject_fold_assignments)
    )
    missing_subjects = sorted(set(groups).difference(assignments))
    if missing_subjects:
        raise ValueError(
            "Subject fold assignments are missing: "
            + ", ".join(str(subject) for subject in missing_subjects)
        )
    invalid_assignments = {
        str(subject): fold
        for subject, fold in assignments.items()
        if isinstance(fold, bool)
        or not isinstance(fold, (int, np.integer))
        or int(fold) < 0
        or int(fold) >= n_splits
    }
    if invalid_assignments:
        raise ValueError(
            f"Subject fold assignments must be integers in 0--{n_splits - 1}: "
            f"{invalid_assignments}"
        )
    assigned_folds = np.asarray(
        [int(assignments[str(subject)]) for subject in groups],
        dtype=np.int64,
    )
    folds: list[SubjectFold] = []

    for fold_index in range(n_splits):
        test_indices = sample_indices[assigned_folds == fold_index]
        train_indices = sample_indices[assigned_folds != fold_index]
        if len(test_indices) == 0:
            raise ValueError(
                f"Fold {fold_index + 1} has no samples for this exercise"
            )
        number = fold_index + 1
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
