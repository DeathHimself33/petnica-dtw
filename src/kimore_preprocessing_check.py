from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kimore_dataset import (
    JOINT_INDEX,
    find_sample,
    load_joint_positions,
    read_manifest,
)
from kimore_preprocessing import preprocess_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify KIMORE centring and "
            "body-size normalization."
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
        "--subject",
        default="B_ID1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()

    samples, _ = read_manifest(
        manifest_path,
        args.exercise,
    )
    sample = find_sample(
        samples,
        args.subject,
    )

    sequence = load_joint_positions(
        sample.position_path
    )
    original_positions = sequence.positions.copy()

    processed = preprocess_sequence(sequence)

    spine_base = JOINT_INDEX["SpineBase"]
    spine_shoulder = JOINT_INDEX["SpineShoulder"]

    normalized_torso_lengths = np.linalg.norm(
        processed.positions[:, spine_shoulder, :]
        - processed.positions[:, spine_base, :],
        axis=1,
    )

    checks = {
        "input was not modified": np.array_equal(
            sequence.positions,
            original_positions,
        ),
        "position shape preserved": (
            processed.positions.shape
            == sequence.positions.shape
        ),
        "tracking states preserved": np.array_equal(
            processed.tracking_states,
            sequence.tracking_states,
        ),
        "SpineBase is zero": np.allclose(
            processed.positions[:, spine_base, :],
            0.0,
        ),
        "coordinates are finite": np.isfinite(
            processed.positions
        ).all(),
        "median torso length is one": np.isclose(
            np.median(normalized_torso_lengths),
            1.0,
        ),
    }

    print(f"Sample: {sample.sample_id}")
    print(f"Raw shape: {sequence.positions.shape}")
    print(
        f"Processed shape: "
        f"{processed.positions.shape}"
    )
    print(f"Body scale: {processed.body_scale:.6f}")
    print(
        "Maximum absolute SpineBase coordinate: "
        f"{np.max(np.abs(processed.positions[:, spine_base, :])):.12f}"
    )
    print(
        "Median normalized torso length: "
        f"{np.median(normalized_torso_lengths):.6f}"
    )
    print()

    for name, passed in checks.items():
        result = "PASS" if passed else "FAIL"
        print(f"{result}: {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())