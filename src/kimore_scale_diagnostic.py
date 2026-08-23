from __future__ import annotations

import sys
import argparse
import csv
from pathlib import Path

import numpy as np

from kimore_dataset import JOINT_INDEX, load_joint_positions, read_manifest
from kimore_preprocessing import (
    diagnose_scale_candidates,
    preprocess_sequence,
)

OUTPUT_FIELDS = (
    "sample_id",
    "subject_id",
    "cohort",
    "clinical_ts",
    "frames",
    "torso_median_length",
    "torso_relative_mad",
    "torso_usable_fraction",
    "shoulders_median_length",
    "shoulders_relative_mad",
    "shoulders_usable_fraction",
)

def collect_diagnostics(
    manifest_path: Path,
    exercise: str,
) -> tuple[
    list[dict[str, object]],
    dict[str, str],
    dict[str, str],
]:
    samples, excluded = read_manifest(
        manifest_path,
        exercise,
    )

    rows: list[dict[str, object]] = []
    failures: dict[str, str] = {}

    for sample in samples:
        try:
            sequence = load_joint_positions(
                sample.position_path
            )
            original_positions = sequence.positions.copy()
            original_tracking_states = sequence.tracking_states.copy()

            if len(sequence.positions) != sample.frames:
                raise ValueError(
                    f"Manifest says {sample.frames} frames, "
                    f"but the loader read "
                    f"{len(sequence.positions)}"
                )

            diagnostics = diagnose_scale_candidates(
                sequence
            )
            processed = preprocess_sequence(sequence)

            if processed.positions.shape != sequence.positions.shape:
                raise ValueError(
                    "Preprocessing changed the position-array shape"
                )
            if not np.array_equal(sequence.positions, original_positions):
                raise ValueError("Preprocessing modified the input positions")
            if not np.array_equal(
                sequence.tracking_states,
                original_tracking_states,
            ):
                raise ValueError("Preprocessing modified the input tracking states")
            if not np.array_equal(
                processed.tracking_states,
                original_tracking_states,
            ):
                raise ValueError("Preprocessing did not preserve tracking states")

            spine_base = JOINT_INDEX["SpineBase"]
            spine_shoulder = JOINT_INDEX["SpineShoulder"]
            normalized_torso_lengths = np.linalg.norm(
                processed.positions[:, spine_shoulder, :]
                - processed.positions[:, spine_base, :],
                axis=1,
            )
            if not np.isclose(np.median(normalized_torso_lengths), 1.0):
                raise ValueError("Median normalized torso length is not one")
            torso = diagnostics["torso"]
            shoulders = diagnostics["shoulders"]
        except (OSError, ValueError) as error:
            failures[sample.sample_id] = str(error)
            continue

        rows.append(
            {
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "cohort": sample.cohort,
                "clinical_ts": sample.score,
                "frames": sample.frames,
                "torso_median_length": (
                    torso.median_length
                ),
                "torso_relative_mad": (
                    torso.relative_mad
                ),
                "torso_usable_fraction": (
                    torso.usable_fraction
                ),
                "shoulders_median_length": (
                    shoulders.median_length
                ),
                "shoulders_relative_mad": (
                    shoulders.relative_mad
                ),
                "shoulders_usable_fraction": (
                    shoulders.usable_fraction
                ),
            }
        )

    return rows, excluded, failures

def write_diagnostics(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def column_array(
    rows: list[dict[str, object]],
    field: str,
) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in rows],
        dtype=float,
    )


def print_candidate_summary(
    rows: list[dict[str, object]],
    name: str,
    label: str,
) -> None:
    lengths = column_array(
        rows,
        f"{name}_median_length",
    )
    relative_mads = column_array(
        rows,
        f"{name}_relative_mad",
    )
    usable_fractions = column_array(
        rows,
        f"{name}_usable_fraction",
    )

    print(f"{label}:")
    print(
        f"  Median scale: "
        f"{np.median(lengths):.4f}"
    )
    print(
        f"  Median relative MAD: "
        f"{np.median(relative_mads):.2%}"
    )
    print(
        f"  Worst relative MAD: "
        f"{np.max(relative_mads):.2%}"
    )
    print(
        f"  Median usable fraction: "
        f"{np.median(usable_fractions):.2%}"
    )
    print(
        f"  Lowest usable fraction: "
        f"{np.min(usable_fractions):.2%}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare torso-length and shoulder-width "
            "body-scale candidates for KIMORE."
        )
    )
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
        "--output",
        type=Path,
        default=Path("kimore_scale_diagnostics.csv"),
        help=(
            "Output CSV path "
            "(default: ./kimore_scale_diagnostics.csv)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not manifest_path.is_file():
        print(
            f"Manifest does not exist: {manifest_path}",
            file=sys.stderr,
        )
        return 2

    rows, excluded, failures = collect_diagnostics(
        manifest_path,
        args.exercise,
    )

    print(f"Exercise: {args.exercise}")
    print(f"Successful diagnostics: {len(rows)}")
    print(f"Manifest exclusions: {len(excluded)}")
    print(f"Diagnostic failures: {len(failures)}")

    for sample_id, reason in excluded.items():
        print(f"  Excluded {sample_id}: {reason}")

    for sample_id, reason in failures.items():
        print(f"  Failed {sample_id}: {reason}")

    if not rows:
        print(
            "No successful diagnostics were produced.",
            file=sys.stderr,
        )
        return 1

    write_diagnostics(rows, output_path)

    print()
    print_candidate_summary(
        rows,
        "torso",
        "Torso: SpineBase–SpineShoulder",
    )
    print()
    print_candidate_summary(
        rows,
        "shoulders",
        "Shoulders: ShoulderLeft–ShoulderRight",
    )

    torso_mads = column_array(
        rows,
        "torso_relative_mad",
    )
    shoulder_mads = column_array(
        rows,
        "shoulders_relative_mad",
    )

    torso_wins = int(
        np.sum(torso_mads < shoulder_mads)
    )
    shoulder_wins = int(
        np.sum(shoulder_mads < torso_mads)
    )
    ties = len(rows) - torso_wins - shoulder_wins

    print()
    print("Per-recording stability:")
    print(f"  Torso more stable: {torso_wins}")
    print(f"  Shoulders more stable: {shoulder_wins}")
    print(f"  Ties: {ties}")
    print(f"Output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
