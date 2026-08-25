"""Run the first subject-disjoint, plain-DTW KIMORE baseline split.

The signal is shoulder-axis yaw: a direct and interpretable measurement of the
trunk rotation exercised in Es3.  One clinically high-quality execution is
selected from the training subjects only.  Exact DTW aligns every execution to
that reference, and a linear calibration fitted only on training distances
maps aligned RMSE to clinical Total Score.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from kimore_dataset import (
    JOINT_INDEX,
    JointSequence,
    KimoreSample,
    load_joint_positions,
    read_manifest,
)
from kimore_dtw import DtwAlignment, exact_dtw
from kimore_grouping import assert_no_subject_leakage, make_subject_folds, subject_groups
from kimore_preprocessing import preprocess_sequence
from kimore_tracking_diagnostic import shoulder_yaw_degrees


FEATURE_NAME = "shoulder_axis_yaw_degrees"
MIN_REFERENCE_SHOULDER_TRACKED_FRACTION = 0.99
SUPPORTED_EXERCISE = "Es3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class PreparedSample:
    sample: KimoreSample
    feature: np.ndarray  # shape: (frames, 1)
    shoulder_tracked_fraction: float


@dataclass(frozen=True)
class LinearCalibration:
    intercept: float
    slope: float

    def predict(self, distances: np.ndarray) -> np.ndarray:
        values = np.asarray(distances, dtype=np.float64)
        return self.intercept + self.slope * values


def shoulder_yaw_feature(sequence: JointSequence) -> np.ndarray:
    """Return the unwrapped shoulder-axis yaw as one feature per frame."""
    yaw = shoulder_yaw_degrees(sequence)
    return yaw[:, np.newaxis]


def prepare_sample(sample: KimoreSample) -> PreparedSample:
    sequence = load_joint_positions(sample.position_path)
    if len(sequence.positions) != sample.frames:
        raise ValueError(
            f"{sample.sample_id}: manifest says {sample.frames} frames but "
            f"the loader read {len(sequence.positions)}"
        )

    processed = preprocess_sequence(sequence)
    processed_sequence = JointSequence(
        positions=processed.positions,
        tracking_states=processed.tracking_states,
    )
    left = JOINT_INDEX["ShoulderLeft"]
    right = JOINT_INDEX["ShoulderRight"]
    both_shoulders_tracked = (
        (sequence.tracking_states[:, left] == 2)
        & (sequence.tracking_states[:, right] == 2)
    )
    return PreparedSample(
        sample=sample,
        feature=shoulder_yaw_feature(processed_sequence),
        shoulder_tracked_fraction=float(both_shoulders_tracked.mean()),
    )


def select_training_reference(
    prepared_samples: Sequence[PreparedSample],
    train_indices: Sequence[int],
) -> int:
    """Choose a well-tracked, high-score reference from training only."""
    candidate_indices = [int(index) for index in train_indices]
    if not candidate_indices:
        raise ValueError("Cannot select a reference from an empty training split")
    if min(candidate_indices) < 0 or max(candidate_indices) >= len(prepared_samples):
        raise IndexError("Training index is outside the prepared sample list")

    reliable_indices = [
        index
        for index in candidate_indices
        if prepared_samples[index].shoulder_tracked_fraction
        >= MIN_REFERENCE_SHOULDER_TRACKED_FRACTION
    ]
    if not reliable_indices:
        best_tracking_fraction = max(
            prepared_samples[index].shoulder_tracked_fraction
            for index in candidate_indices
        )
        reliable_indices = [
            index
            for index in candidate_indices
            if np.isclose(
                prepared_samples[index].shoulder_tracked_fraction,
                best_tracking_fraction,
            )
        ]

    return min(
        reliable_indices,
        key=lambda index: (
            -prepared_samples[index].sample.score,
            -prepared_samples[index].shoulder_tracked_fraction,
            prepared_samples[index].sample.sample_id,
        ),
    )


def fit_linear_calibration(
    distances: np.ndarray,
    scores: np.ndarray,
) -> LinearCalibration:
    """Fit score = intercept + slope * DTW distance by least squares."""
    x = np.asarray(distances, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Calibration needs equally sized 1D arrays with at least 2 rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Calibration inputs must be finite")
    if np.allclose(x, x[0]):
        return LinearCalibration(intercept=float(np.mean(y)), slope=0.0)

    design = np.column_stack((np.ones(len(x)), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return LinearCalibration(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
    )


def save_alignment_plot(
    reference: PreparedSample,
    comparison: PreparedSample,
    alignment: DtwAlignment,
    quality_label: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    reference_yaw = reference.feature[:, 0]
    comparison_yaw = comparison.feature[:, 0]
    path = alignment.path

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2))
    axes[0].plot(
        np.linspace(0.0, 100.0, len(reference_yaw)),
        reference_yaw,
        label=f"Reference: {reference.sample.sample_id}",
        linewidth=1.2,
    )
    axes[0].plot(
        np.linspace(0.0, 100.0, len(comparison_yaw)),
        comparison_yaw,
        label=f"Comparison: {comparison.sample.sample_id}",
        linewidth=1.0,
        alpha=0.85,
    )
    axes[0].set_title(
        f"{quality_label.capitalize()} test alignment before time warping"
    )
    axes[0].set_xlabel("Recording progress (%)")
    axes[0].set_ylabel("Shoulder-axis yaw (degrees)")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    path_steps = np.arange(len(path))
    axes[1].plot(
        path_steps,
        reference_yaw[path[:, 1]],
        label="Reference along DTW path",
        linewidth=1.2,
    )
    axes[1].plot(
        path_steps,
        comparison_yaw[path[:, 0]],
        label="Comparison along DTW path",
        linewidth=1.0,
        alpha=0.85,
    )
    axes[1].set_title(
        f"After alignment: aligned RMSE = {alignment.aligned_rmse:.3f} degrees, "
        f"path length = {len(path)}"
    )
    axes[1].set_xlabel("DTW path step")
    axes[1].set_ylabel("Shoulder-axis yaw (degrees)")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_prediction_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_first_fold(
    manifest_path: Path,
    exercise: str,
    output_dir: Path,
    figure_dir: Path,
) -> dict[str, object]:
    if exercise.casefold() != SUPPORTED_EXERCISE.casefold():
        raise ValueError(
            f"The shoulder-yaw plain-DTW method is defined only for "
            f"{SUPPORTED_EXERCISE}, not {exercise}"
        )
    samples, excluded = read_manifest(manifest_path, exercise)
    if len(samples) < 2:
        raise ValueError("The plain-DTW baseline needs at least two usable samples")

    folds = make_subject_folds(samples, n_splits=5)
    fold = folds[0]
    groups = subject_groups(samples)
    assert_no_subject_leakage(groups, fold.train_indices, fold.test_indices)

    print(f"Loading and preprocessing {len(samples)} usable {exercise} recordings...")
    prepared_samples = []
    for number, sample in enumerate(samples, start=1):
        prepared_samples.append(prepare_sample(sample))
        if number % 10 == 0 or number == len(samples):
            print(f"  prepared {number}/{len(samples)}")

    reference_index = select_training_reference(prepared_samples, fold.train_indices)
    reference = prepared_samples[reference_index]
    test_index_set = {int(index) for index in fold.test_indices}
    train_index_set = {int(index) for index in fold.train_indices}

    print(
        f"Training-only reference: {reference.sample.sample_id} "
        f"(TS={reference.sample.score:.1f}, shoulders tracked="
        f"{reference.shoulder_tracked_fraction:.1%})"
    )
    print("Calculating exact DTW alignments...")
    alignments: list[DtwAlignment] = []
    for number, prepared in enumerate(prepared_samples, start=1):
        alignment = exact_dtw(prepared.feature, reference.feature)
        alignments.append(alignment)
        if number % 10 == 0 or number == len(prepared_samples):
            print(f"  aligned {number}/{len(prepared_samples)}")

    distances = np.asarray(
        [alignment.aligned_rmse for alignment in alignments],
        dtype=np.float64,
    )
    scores = np.asarray([sample.score for sample in samples], dtype=np.float64)
    calibration = fit_linear_calibration(
        distances[fold.train_indices],
        scores[fold.train_indices],
    )
    predictions = calibration.predict(distances)

    rows: list[dict[str, object]] = []
    for index, (prepared, alignment, predicted_score) in enumerate(
        zip(prepared_samples, alignments, predictions)
    ):
        rows.append(
            {
                "fold": fold.number,
                "split": "train" if index in train_index_set else "test",
                "sample_id": prepared.sample.sample_id,
                "subject_id": prepared.sample.subject_id,
                "cohort": prepared.sample.cohort,
                "actual_ts": prepared.sample.score,
                "dtw_aligned_rmse_degrees": alignment.aligned_rmse,
                "predicted_ts": float(predicted_score),
                "frames": len(prepared.feature),
                "alignment_path_length": len(alignment.path),
                "feature": FEATURE_NAME,
                "reference_sample_id": reference.sample.sample_id,
                "reference_subject_id": reference.sample.subject_id,
                "reference_actual_ts": reference.sample.score,
                "calibration_intercept": calibration.intercept,
                "calibration_slope": calibration.slope,
            }
        )

    prediction_path = output_dir / "fold_1_predictions.csv"
    save_prediction_rows(rows, prediction_path)

    good_index = min(test_index_set, key=lambda index: distances[index])
    poor_index = max(test_index_set, key=lambda index: distances[index])
    good_plot_path = figure_dir / "fold_1_good_alignment.png"
    poor_plot_path = figure_dir / "fold_1_poor_alignment.png"
    save_alignment_plot(
        reference,
        prepared_samples[good_index],
        alignments[good_index],
        "closest",
        good_plot_path,
    )
    save_alignment_plot(
        reference,
        prepared_samples[poor_index],
        alignments[poor_index],
        "most distant",
        poor_plot_path,
    )

    test_actual = scores[fold.test_indices]
    test_predicted = predictions[fold.test_indices]
    test_mae = float(np.mean(np.abs(test_actual - test_predicted)))
    metadata: dict[str, object] = {
        "exercise": exercise,
        "feature": FEATURE_NAME,
        "usable_samples": len(samples),
        "excluded_samples": excluded,
        "fold": fold.number,
        "training_samples": len(fold.train_indices),
        "test_samples": len(fold.test_indices),
        "subject_overlap": 0,
        "reference_sample_id": reference.sample.sample_id,
        "reference_subject_id": reference.sample.subject_id,
        "reference_actual_ts": reference.sample.score,
        "reference_shoulder_tracked_fraction": reference.shoulder_tracked_fraction,
        "reference_selection": (
            "among fold-1 training recordings with at least 99% fully tracked "
            "shoulder frames, highest clinical TS; if none meet that tracking "
            "threshold, use the best-tracked training recording(s); remaining "
            "ties resolved by tracking fraction and sample ID"
        ),
        "distance": "sqrt(sum of squared feature differences along DTW path / path length)",
        "calibration": "ordinary least-squares linear regression fitted on fold-1 training rows only",
        "calibration_intercept": calibration.intercept,
        "calibration_slope": calibration.slope,
        "test_mae_preview": test_mae,
        "closest_test_sample_id": prepared_samples[good_index].sample.sample_id,
        "closest_test_distance": float(distances[good_index]),
        "most_distant_test_sample_id": prepared_samples[poor_index].sample.sample_id,
        "most_distant_test_distance": float(distances[poor_index]),
        "prediction_csv": portable_project_path(prediction_path),
        "closest_alignment_plot": portable_project_path(good_plot_path),
        "most_distant_alignment_plot": portable_project_path(poor_plot_path),
        "evaluation_note": (
            "Historical one-fold development preview, inspected while the reference "
            "quality rule was selected; it is not an untouched performance estimate. "
            "Use the five-fold outputs for the internal development result."
        ),
    }
    metadata_path = output_dir / "fold_1_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Fold-1 preview MAE: {test_mae:.3f} TS points")
    print(f"Predictions: {prediction_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"Closest alignment: {good_plot_path.resolve()}")
    print(f"Most distant alignment: {poor_plot_path.resolve()}")
    return metadata


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
        choices=["Es3"],
        help="Exercise supported by the frozen shoulder-yaw method (Es3 only)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plain_dtw"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/plain_dtw"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_first_fold(
        manifest_path=args.manifest.expanduser().resolve(),
        exercise=args.exercise,
        output_dir=args.output_dir.expanduser().resolve(),
        figure_dir=args.figure_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
