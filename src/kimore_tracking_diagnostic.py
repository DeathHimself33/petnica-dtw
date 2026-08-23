"""Audit KIMORE tracking quality and short-lived noise for one exercise.

The diagnostic does not alter recordings.  It quantifies tracked, inferred and
untracked joint observations, checks for missing coordinate values, and uses a
five-frame moving median only as a measuring tool for short-lived deviations in
the shoulder-yaw signal.  It writes per-sample and per-joint CSV files so that
preprocessing decisions are based on inspectable evidence.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kimore_dataset import (
    JOINT_INDEX,
    JOINT_NAMES,
    JointSequence,
    KimoreSample,
    load_joint_positions,
    read_manifest,
)


CORE_TRUNK_JOINTS = (
    "SpineBase",
    "SpineMid",
    "SpineShoulder",
    "Neck",
)

SAMPLE_FIELDS = (
    "sample_id",
    "subject_id",
    "cohort",
    "clinical_ts",
    "frames",
    "joint_observations",
    "tracked_observations",
    "inferred_observations",
    "untracked_observations",
    "tracked_fraction",
    "inferred_fraction",
    "untracked_fraction",
    "nonfinite_coordinates",
    "zero_position_observations",
    "core_trunk_fully_tracked_fraction",
    "shoulders_fully_tracked_fraction",
    "pelvis_yaw_median_degrees",
    "pelvis_yaw_mad_degrees",
    "shoulder_yaw_range_5_95_degrees",
    "shoulder_yaw_median5_residual_p95_degrees",
    "shoulder_yaw_median5_residual_max_degrees",
)

JOINT_FIELDS = (
    "joint_name",
    "observations",
    "tracked_fraction",
    "inferred_fraction",
    "untracked_fraction",
    "samples_with_inferred_frames",
    "samples_with_untracked_frames",
    "median_longest_inferred_or_untracked_run_frames",
    "maximum_longest_inferred_or_untracked_run_frames",
)


@dataclass(frozen=True)
class TrackingCounts:
    tracked: int
    inferred: int
    untracked: int

    @property
    def total(self) -> int:
        return self.tracked + self.inferred + self.untracked


def count_tracking_states(tracking_states: np.ndarray) -> TrackingCounts:
    """Count Kinect state 2 (tracked), 1 (inferred) and 0 (untracked)."""
    states = np.asarray(tracking_states)
    invalid = ~np.isin(states, (0, 1, 2))
    if invalid.any():
        invalid_values = ", ".join(
            str(value) for value in np.unique(states[invalid])[:8]
        )
        raise ValueError(
            f"Invalid tracking-state values: {invalid_values}; expected 0, 1 or 2"
        )

    counts = np.bincount(states.astype(int).ravel(), minlength=3)
    return TrackingCounts(
        tracked=int(counts[2]),
        inferred=int(counts[1]),
        untracked=int(counts[0]),
    )


def centered_moving_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Return an edge-padded moving median for an odd-sized window."""
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("Moving-median input must be a non-empty 1D array")
    if window < 1 or window % 2 == 0:
        raise ValueError("Moving-median window must be a positive odd integer")

    radius = window // 2
    padded = np.pad(signal, (radius, radius), mode="edge")
    windows = np.stack(
        [padded[offset : offset + len(signal)] for offset in range(window)],
        axis=1,
    )
    return np.median(windows, axis=1)


def longest_true_run(mask: np.ndarray) -> int:
    """Return the length of the longest consecutive True run."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Run-length input must be a 1D array")
    if not values.any():
        return 0

    padded = np.concatenate(([False], values, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return int(np.max(stops - starts))


def horizontal_axis_yaw_degrees(
    sequence: JointSequence,
    left_joint: str,
    right_joint: str,
) -> np.ndarray:
    """Return a left-to-right body-axis yaw around the Kinect Y axis."""
    left = JOINT_INDEX[left_joint]
    right = JOINT_INDEX[right_joint]
    body_axis = sequence.positions[:, right, :] - sequence.positions[:, left, :]
    horizontal_length = np.linalg.norm(body_axis[:, (0, 2)], axis=1)

    if not np.isfinite(body_axis).all():
        raise ValueError("Cannot calculate body-axis yaw from non-finite coordinates")
    if np.any(horizontal_length == 0):
        raise ValueError("Cannot calculate body-axis yaw from a zero-length horizontal axis")

    yaw_radians = np.arctan2(body_axis[:, 2], body_axis[:, 0])
    return np.degrees(np.unwrap(yaw_radians))


def shoulder_yaw_degrees(sequence: JointSequence) -> np.ndarray:
    """Return left-to-right shoulder-axis yaw around the Kinect Y axis."""
    return horizontal_axis_yaw_degrees(
        sequence,
        "ShoulderLeft",
        "ShoulderRight",
    )


def sample_diagnostic_row(
    sample: KimoreSample,
    sequence: JointSequence,
) -> dict[str, object]:
    """Create one inspectable tracking/noise summary for a recording."""
    counts = count_tracking_states(sequence.tracking_states)
    if counts.total == 0:
        raise ValueError("Cannot diagnose an empty tracking-state array")

    core_indices = [JOINT_INDEX[name] for name in CORE_TRUNK_JOINTS]
    core_fully_tracked = np.all(
        sequence.tracking_states[:, core_indices] == 2,
        axis=1,
    )
    left = JOINT_INDEX["ShoulderLeft"]
    right = JOINT_INDEX["ShoulderRight"]
    shoulders_fully_tracked = (
        (sequence.tracking_states[:, left] == 2)
        & (sequence.tracking_states[:, right] == 2)
    )

    yaw = shoulder_yaw_degrees(sequence)
    median_yaw = centered_moving_median(yaw, window=5)
    residual = np.abs(yaw - median_yaw)
    yaw_range = float(np.percentile(yaw, 95) - np.percentile(yaw, 5))
    pelvis_yaw = horizontal_axis_yaw_degrees(
        sequence,
        "HipLeft",
        "HipRight",
    )
    pelvis_yaw_median = float(np.median(pelvis_yaw))
    pelvis_yaw_mad = float(
        np.median(np.abs(pelvis_yaw - pelvis_yaw_median))
    )

    return {
        "sample_id": sample.sample_id,
        "subject_id": sample.subject_id,
        "cohort": sample.cohort,
        "clinical_ts": sample.score,
        "frames": len(sequence.positions),
        "joint_observations": counts.total,
        "tracked_observations": counts.tracked,
        "inferred_observations": counts.inferred,
        "untracked_observations": counts.untracked,
        "tracked_fraction": counts.tracked / counts.total,
        "inferred_fraction": counts.inferred / counts.total,
        "untracked_fraction": counts.untracked / counts.total,
        "nonfinite_coordinates": int((~np.isfinite(sequence.positions)).sum()),
        "zero_position_observations": int(
            np.all(sequence.positions == 0, axis=2).sum()
        ),
        "core_trunk_fully_tracked_fraction": float(core_fully_tracked.mean()),
        "shoulders_fully_tracked_fraction": float(shoulders_fully_tracked.mean()),
        "pelvis_yaw_median_degrees": pelvis_yaw_median,
        "pelvis_yaw_mad_degrees": pelvis_yaw_mad,
        "shoulder_yaw_range_5_95_degrees": yaw_range,
        "shoulder_yaw_median5_residual_p95_degrees": float(
            np.percentile(residual, 95)
        ),
        "shoulder_yaw_median5_residual_max_degrees": float(np.max(residual)),
    }


def joint_diagnostic_rows(
    sequences: list[JointSequence],
) -> list[dict[str, object]]:
    """Aggregate tracking-state quality by joint across recordings."""
    rows: list[dict[str, object]] = []
    for joint_name in JOINT_NAMES:
        joint_index = JOINT_INDEX[joint_name]
        per_sample_states = [
            sequence.tracking_states[:, joint_index] for sequence in sequences
        ]
        longest_nontracked_runs = [
            longest_true_run(states != 2) for states in per_sample_states
        ]
        counts = count_tracking_states(np.concatenate(per_sample_states))
        rows.append(
            {
                "joint_name": joint_name,
                "observations": counts.total,
                "tracked_fraction": counts.tracked / counts.total,
                "inferred_fraction": counts.inferred / counts.total,
                "untracked_fraction": counts.untracked / counts.total,
                "samples_with_inferred_frames": sum(
                    int(np.any(states == 1)) for states in per_sample_states
                ),
                "samples_with_untracked_frames": sum(
                    int(np.any(states == 0)) for states in per_sample_states
                ),
                "median_longest_inferred_or_untracked_run_frames": float(
                    np.median(longest_nontracked_runs)
                ),
                "maximum_longest_inferred_or_untracked_run_frames": max(
                    longest_nontracked_runs
                ),
            }
        )
    return rows


def collect_diagnostics(
    manifest_path: Path,
    exercise: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, str],
    dict[str, str],
]:
    samples, excluded = read_manifest(manifest_path, exercise)
    sample_rows: list[dict[str, object]] = []
    successful_sequences: list[JointSequence] = []
    failures: dict[str, str] = {}

    for sample in samples:
        try:
            sequence = load_joint_positions(sample.position_path)
            if len(sequence.positions) != sample.frames:
                raise ValueError(
                    f"Manifest says {sample.frames} frames, but the loader read "
                    f"{len(sequence.positions)}"
                )
            row = sample_diagnostic_row(sample, sequence)
        except (OSError, ValueError) as error:
            failures[sample.sample_id] = str(error)
            continue

        successful_sequences.append(sequence)
        sample_rows.append(row)

    joint_rows = (
        joint_diagnostic_rows(successful_sequences) if successful_sequences else []
    )
    return sample_rows, joint_rows, excluded, failures


def write_rows(
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fraction_array(
    rows: list[dict[str, object]],
    field: str,
) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def print_summary(
    sample_rows: list[dict[str, object]],
    joint_rows: list[dict[str, object]],
) -> None:
    total_observations = sum(int(row["joint_observations"]) for row in sample_rows)
    tracked = sum(int(row["tracked_observations"]) for row in sample_rows)
    inferred = sum(int(row["inferred_observations"]) for row in sample_rows)
    untracked = sum(int(row["untracked_observations"]) for row in sample_rows)

    print(f"Successful sample diagnostics: {len(sample_rows)}")
    print(f"Total frames: {sum(int(row['frames']) for row in sample_rows)}")
    print(f"Joint observations: {total_observations}")
    print(f"  Fully tracked: {tracked / total_observations:.2%}")
    print(f"  Inferred: {inferred / total_observations:.2%}")
    print(f"  Untracked: {untracked / total_observations:.2%}")
    print(
        "Non-finite coordinates: "
        f"{sum(int(row['nonfinite_coordinates']) for row in sample_rows)}"
    )
    print(
        "Zero XYZ observations: "
        f"{sum(int(row['zero_position_observations']) for row in sample_rows)}"
    )
    print(
        "Lowest core-trunk fully tracked fraction: "
        f"{np.min(fraction_array(sample_rows, 'core_trunk_fully_tracked_fraction')):.2%}"
    )

    worst_joint = min(joint_rows, key=lambda row: float(row["tracked_fraction"]))
    print(
        "Lowest joint fully tracked fraction: "
        f"{worst_joint['joint_name']}={float(worst_joint['tracked_fraction']):.2%}"
    )
    longest_run_joint = max(
        joint_rows,
        key=lambda row: int(row["maximum_longest_inferred_or_untracked_run_frames"]),
    )
    print(
        "Longest inferred/untracked run: "
        f"{longest_run_joint['joint_name']}="
        f"{int(longest_run_joint['maximum_longest_inferred_or_untracked_run_frames'])} "
        "frames"
    )

    pelvis_medians = fraction_array(sample_rows, "pelvis_yaw_median_degrees")
    print(
        "Per-recording median pelvis yaw (5th to 95th percentile): "
        f"{np.percentile(pelvis_medians, 5):.2f} to "
        f"{np.percentile(pelvis_medians, 95):.2f} degrees"
    )

    p95_residuals = fraction_array(
        sample_rows,
        "shoulder_yaw_median5_residual_p95_degrees",
    )
    maximum_residuals = fraction_array(
        sample_rows,
        "shoulder_yaw_median5_residual_max_degrees",
    )
    worst_index = int(np.argmax(maximum_residuals))
    print("Shoulder-yaw deviation from five-frame moving median:")
    print(f"  Median per-sample 95th percentile: {np.median(p95_residuals):.3f} degrees")
    print(f"  Worst per-sample 95th percentile: {np.max(p95_residuals):.3f} degrees")
    print(
        "  Largest isolated deviation: "
        f"{maximum_residuals[worst_index]:.3f} degrees "
        f"({sample_rows[worst_index]['sample_id']})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to kimore_manifest.csv")
    parser.add_argument(
        "--exercise",
        default="Es3",
        choices=[f"Es{i}" for i in range(1, 6)],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/preprocessing"),
        help="Directory for the two diagnostic CSV files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not manifest_path.is_file():
        print(f"Manifest does not exist: {manifest_path}", file=sys.stderr)
        return 2

    sample_rows, joint_rows, excluded, failures = collect_diagnostics(
        manifest_path,
        args.exercise,
    )
    print(f"Exercise: {args.exercise}")
    print(f"Manifest exclusions: {len(excluded)}")
    print(f"Diagnostic failures: {len(failures)}")
    for sample_id, reason in excluded.items():
        print(f"  Excluded {sample_id}: {reason}")
    for sample_id, reason in failures.items():
        print(f"  Failed {sample_id}: {reason}")

    if not sample_rows:
        print("No successful diagnostics were produced.", file=sys.stderr)
        return 1

    sample_output = output_dir / "sample_tracking_quality.csv"
    joint_output = output_dir / "joint_tracking_quality.csv"
    write_rows(sample_rows, SAMPLE_FIELDS, sample_output)
    write_rows(joint_rows, JOINT_FIELDS, joint_output)

    print()
    print_summary(sample_rows, joint_rows)
    print(f"Sample output: {sample_output}")
    print(f"Joint output: {joint_output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
