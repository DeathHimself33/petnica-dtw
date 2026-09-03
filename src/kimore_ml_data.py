"""Build leakage-safe, fixed-length KIMORE inputs for temporal ML models.

The exported tensors deliberately remain unstandardized.  A model evaluation
must fit :class:`FeatureStandardizer` on the outer-training rows of each fold
and then apply it to that fold's training and test rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from kimore_dataset import KimoreSample, load_joint_positions, read_manifest
from kimore_grouping import assert_no_subject_leakage, make_subject_fold_assignments
from kimore_interpretable_quality import apply_frame_quality_control
from kimore_yu_xiong_dtw import FEATURE_DIMENSIONS, yu_xiong_vectors


EXERCISES = tuple(f"Es{number}" for number in range(1, 6))
EXERCISE_TO_INDEX = {exercise: index for index, exercise in enumerate(EXERCISES)}
DEFAULT_SEQUENCE_LENGTH = 128


@dataclass(frozen=True)
class ResampledSequence:
    vectors: np.ndarray  # (time, 9, 3), float32
    frame_mask: np.ndarray  # (time,), bool; false positions are zeroed
    component_observed_mask: np.ndarray  # (time, 9), bool
    source_frame_indices: np.ndarray  # (time,), int64


@dataclass(frozen=True)
class MLDataset:
    features: np.ndarray  # (samples, time, 9, 3), float32
    frame_mask: np.ndarray  # (samples, time), bool
    component_observed_mask: np.ndarray  # (samples, time, 9), bool
    targets: np.ndarray  # (samples,), float32
    exercise_indices: np.ndarray  # (samples,), int64
    fold_numbers: np.ndarray  # (samples,), int64; one-based
    sample_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    cohorts: tuple[str, ...]
    quality_statuses: tuple[str, ...]
    retained_fractions: np.ndarray  # (samples,), float32

    @property
    def sequence_length(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray  # (9, 3), float64
    scale: np.ndarray  # (9, 3), float64


def uniform_resample(
    vectors: np.ndarray,
    usable_frame_mask: np.ndarray,
    component_observed_mask: np.ndarray,
    target_frames: int = DEFAULT_SEQUENCE_LENGTH,
) -> ResampledSequence:
    """Sample a recording on a fixed progress grid while preserving QC masks."""
    values = np.asarray(vectors, dtype=np.float64)
    usable = np.asarray(usable_frame_mask, dtype=bool)
    observed = np.asarray(component_observed_mask, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != (FEATURE_DIMENSIONS, 3):
        raise ValueError(
            "Vectors must have shape (frames, "
            f"{FEATURE_DIMENSIONS}, 3); got {values.shape}"
        )
    if len(values) == 0:
        raise ValueError("Cannot resample an empty recording")
    if usable.shape != (len(values),):
        raise ValueError("Usable-frame mask does not match the recording")
    if observed.shape != (len(values), FEATURE_DIMENSIONS):
        raise ValueError("Component-observed mask does not match the recording")
    if target_frames < 2:
        raise ValueError("Target sequence length must be at least two frames")
    if not np.isfinite(values).all():
        raise ValueError("ML feature vectors must be finite")

    source_indices = np.rint(
        np.linspace(0, len(values) - 1, target_frames)
    ).astype(np.int64)
    sampled_vectors = values[source_indices].astype(np.float32)
    sampled_usable = usable[source_indices].copy()
    sampled_observed = observed[source_indices].copy()
    sampled_observed &= sampled_usable[:, np.newaxis]
    sampled_vectors[~sampled_usable] = 0.0
    return ResampledSequence(
        vectors=sampled_vectors,
        frame_mask=sampled_usable,
        component_observed_mask=sampled_observed,
        source_frame_indices=source_indices,
    )


def fold_train_test_indices(
    subject_ids: Sequence[str],
    fold_numbers: np.ndarray,
    test_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one subject-disjoint outer split from one-based fold numbers."""
    subjects = np.asarray(subject_ids, dtype=object)
    folds = np.asarray(fold_numbers, dtype=np.int64)
    if folds.shape != (len(subjects),):
        raise ValueError("Fold numbers must contain one value per sample")
    available = sorted(int(number) for number in np.unique(folds))
    if test_fold not in available:
        raise ValueError(
            f"Test fold {test_fold} is unavailable; choose one of {available}"
        )
    test_indices = np.flatnonzero(folds == test_fold)
    train_indices = np.flatnonzero(folds != test_fold)
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Both sides of an ML split must be non-empty")
    assert_no_subject_leakage(subjects, train_indices, test_indices)
    return train_indices, test_indices


def fit_feature_standardizer(
    features: np.ndarray,
    frame_mask: np.ndarray,
    train_indices: Sequence[int],
) -> FeatureStandardizer:
    """Fit channel statistics using valid training frames only."""
    values = np.asarray(features, dtype=np.float64)
    masks = np.asarray(frame_mask, dtype=bool)
    if values.ndim != 4 or values.shape[2:] != (FEATURE_DIMENSIONS, 3):
        raise ValueError(
            "ML features must have shape (samples, time, "
            f"{FEATURE_DIMENSIONS}, 3); got {values.shape}"
        )
    if masks.shape != values.shape[:2]:
        raise ValueError("Frame mask must match the sample and time dimensions")
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("Training indices must be a non-empty one-dimensional list")
    if int(indices.min()) < 0 or int(indices.max()) >= len(values):
        raise IndexError("Training index is outside the ML dataset")
    selected_values = values[indices]
    selected_masks = masks[indices]
    valid_values = selected_values[selected_masks]
    if len(valid_values) == 0:
        raise ValueError("Training split contains no QC-usable feature frames")
    mean = np.mean(valid_values, axis=0, dtype=np.float64)
    scale = np.std(valid_values, axis=0, dtype=np.float64)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return FeatureStandardizer(mean=mean, scale=scale)


def apply_feature_standardizer(
    features: np.ndarray,
    frame_mask: np.ndarray,
    standardizer: FeatureStandardizer,
) -> np.ndarray:
    """Apply fitted channel statistics and keep masked frames exactly zero."""
    values = np.asarray(features, dtype=np.float64)
    masks = np.asarray(frame_mask, dtype=bool)
    if values.ndim != 4 or values.shape[2:] != (FEATURE_DIMENSIONS, 3):
        raise ValueError("ML features have an invalid shape")
    if masks.shape != values.shape[:2]:
        raise ValueError("Frame mask must match the sample and time dimensions")
    if standardizer.mean.shape != (FEATURE_DIMENSIONS, 3):
        raise ValueError("Standardizer mean has an invalid shape")
    if standardizer.scale.shape != (FEATURE_DIMENSIONS, 3):
        raise ValueError("Standardizer scale has an invalid shape")
    if not np.isfinite(standardizer.mean).all():
        raise ValueError("Standardizer mean must be finite")
    if not np.isfinite(standardizer.scale).all() or np.any(
        standardizer.scale <= 0
    ):
        raise ValueError("Standardizer scales must be finite and positive")
    transformed = (values - standardizer.mean) / standardizer.scale
    transformed[~masks] = 0.0
    return transformed.astype(np.float32)


def _all_manifest_samples(
    manifest_path: Path,
) -> tuple[list[KimoreSample], dict[str, str]]:
    samples: list[KimoreSample] = []
    excluded: dict[str, str] = {}
    for exercise in EXERCISES:
        exercise_samples, exercise_excluded = read_manifest(manifest_path, exercise)
        samples.extend(exercise_samples)
        excluded.update(exercise_excluded)
    return samples, excluded


def build_ml_dataset(
    manifest_path: Path,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    n_splits: int = 5,
    include_qc_failed: bool = False,
    progress: Callable[[str], None] = print,
) -> tuple[MLDataset, dict[str, str]]:
    """Load all exercises and return model-ready tensors plus exclusions."""
    manifest_path = manifest_path.expanduser().resolve()
    samples, exclusions = _all_manifest_samples(manifest_path)
    assignments = make_subject_fold_assignments(samples, n_splits=n_splits)

    features: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    observed_masks: list[np.ndarray] = []
    targets: list[float] = []
    exercise_indices: list[int] = []
    fold_numbers: list[int] = []
    sample_ids: list[str] = []
    subject_ids: list[str] = []
    cohorts: list[str] = []
    quality_statuses: list[str] = []
    retained_fractions: list[float] = []

    progress(f"Preparing fixed-length ML inputs for {len(samples)} recordings...")
    for number, sample in enumerate(samples, start=1):
        sequence = load_joint_positions(sample.position_path)
        if len(sequence.positions) != sample.frames:
            raise ValueError(
                f"{sample.sample_id}: manifest says {sample.frames} frames but "
                f"the loader read {len(sequence.positions)}"
            )
        vectors = yu_xiong_vectors(sequence, allow_degenerate_frames=True)
        quality = apply_frame_quality_control(sequence, vectors, sample.exercise)
        if quality.quality_status == "fail" and not include_qc_failed:
            exclusions[sample.sample_id] = "full-body frame QC failure"
            continue

        usable = ~quality.dropped_frame_mask
        observed = ~quality.interpolated_component_mask
        observed &= usable[:, np.newaxis]
        fixed = uniform_resample(
            quality.repaired_vectors,
            usable,
            observed,
            target_frames=sequence_length,
        )
        if not fixed.frame_mask.any():
            exclusions[sample.sample_id] = "no usable frames on fixed ML grid"
            continue

        features.append(fixed.vectors)
        frame_masks.append(fixed.frame_mask)
        observed_masks.append(fixed.component_observed_mask)
        targets.append(sample.score)
        exercise_indices.append(EXERCISE_TO_INDEX[sample.exercise])
        fold_numbers.append(assignments[sample.subject_id] + 1)
        sample_ids.append(sample.sample_id)
        subject_ids.append(sample.subject_id)
        cohorts.append(sample.cohort)
        quality_statuses.append(quality.quality_status)
        retained_fractions.append(quality.retained_fraction)
        if number % 50 == 0 or number == len(samples):
            progress(f"Prepared {number}/{len(samples)} recordings")

    if not features:
        raise ValueError("No recordings remained for the ML dataset")
    dataset = MLDataset(
        features=np.stack(features).astype(np.float32),
        frame_mask=np.stack(frame_masks),
        component_observed_mask=np.stack(observed_masks),
        targets=np.asarray(targets, dtype=np.float32),
        exercise_indices=np.asarray(exercise_indices, dtype=np.int64),
        fold_numbers=np.asarray(fold_numbers, dtype=np.int64),
        sample_ids=tuple(sample_ids),
        subject_ids=tuple(subject_ids),
        cohorts=tuple(cohorts),
        quality_statuses=tuple(quality_statuses),
        retained_fractions=np.asarray(retained_fractions, dtype=np.float32),
    )
    for fold in range(1, n_splits + 1):
        fold_train_test_indices(dataset.subject_ids, dataset.fold_numbers, fold)
    return dataset, exclusions


def export_ml_dataset(
    dataset: MLDataset,
    exclusions: dict[str, str],
    output_path: Path,
) -> tuple[Path, Path]:
    """Write numeric tensors to NPZ and transparent schema details to JSON."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=dataset.features,
        frame_mask=dataset.frame_mask,
        component_observed_mask=dataset.component_observed_mask,
        targets=dataset.targets,
        exercise_indices=dataset.exercise_indices,
        fold_numbers=dataset.fold_numbers,
        sample_ids=np.asarray(dataset.sample_ids, dtype=np.str_),
        subject_ids=np.asarray(dataset.subject_ids, dtype=np.str_),
        cohorts=np.asarray(dataset.cohorts, dtype=np.str_),
        quality_statuses=np.asarray(dataset.quality_statuses, dtype=np.str_),
        retained_fractions=dataset.retained_fractions,
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "samples": len(dataset.targets),
                "sequence_length": dataset.sequence_length,
                "feature_shape": list(dataset.features.shape),
                "feature": "nine Yu-Xiong unit vectors in body-local coordinates",
                "exercises": EXERCISE_TO_INDEX,
                "fold_numbering": "one_based",
                "fold_policy": "shared subject assignment across all exercises",
                "normalization": (
                    "unstandardized; fit FeatureStandardizer on each outer-training "
                    "split only"
                ),
                "qc_failed_included": "fail" in dataset.quality_statuses,
                "excluded_samples": exclusions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path, metadata_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("kimore_audit_output/kimore_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ml_data/kimore_all_exercises_128.npz"),
    )
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--include-qc-failed", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset, exclusions = build_ml_dataset(
        args.manifest,
        sequence_length=args.sequence_length,
        n_splits=args.splits,
        include_qc_failed=args.include_qc_failed,
    )
    output_path, metadata_path = export_ml_dataset(dataset, exclusions, args.output)
    print(f"ML samples: {len(dataset.targets)}")
    print(f"Tensor shape: {dataset.features.shape}")
    print(f"NPZ: {output_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
