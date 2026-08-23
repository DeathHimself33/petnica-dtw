"""Create before/after plots for the KIMORE Es3 preprocessing decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kimore_dataset import (
    JOINT_INDEX,
    JointSequence,
    find_sample,
    load_joint_positions,
    read_manifest,
)
from kimore_preprocessing import preprocess_sequence
from kimore_tracking_diagnostic import shoulder_yaw_degrees


def save_joint_signal_plot(
    sample_id: str,
    score: float,
    raw: JointSequence,
    processed: JointSequence,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    joint_index = JOINT_INDEX["SpineShoulder"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)

    axes[0].plot(raw.positions[:, joint_index, 0], linewidth=1.1, color="#16697a")
    axes[0].set_title("Before: camera-relative SpineShoulder X")
    axes[0].set_ylabel("Kinect X coordinate")
    axes[0].grid(alpha=0.22)

    axes[1].plot(
        processed.positions[:, joint_index, 0],
        linewidth=1.1,
        color="#d97706",
    )
    axes[1].set_title("After: SpineBase-relative and torso-normalized X")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Torso lengths")
    axes[1].grid(alpha=0.22)

    figure.suptitle(
        f"{sample_id}: representative signal before/after preprocessing "
        f"(clinical TS {score:.1f})"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_shoulder_yaw_plot(
    sample_id: str,
    raw: JointSequence,
    processed: JointSequence,
    output_path: Path,
) -> float:
    import matplotlib.pyplot as plt

    raw_yaw = shoulder_yaw_degrees(raw)
    processed_yaw = shoulder_yaw_degrees(processed)
    maximum_difference = float(np.max(np.abs(raw_yaw - processed_yaw)))

    figure, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, sharey=True)
    axes[0].plot(raw_yaw, linewidth=1.1, color="#16697a")
    axes[0].set_title("Before: shoulder-axis yaw from raw coordinates")
    axes[0].set_ylabel("Yaw (degrees)")
    axes[0].grid(alpha=0.22)

    axes[1].plot(processed_yaw, linewidth=1.1, color="#d97706")
    axes[1].set_title("After: the same yaw is preserved")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Yaw (degrees)")
    axes[1].grid(alpha=0.22)

    figure.suptitle(
        f"{sample_id}: trunk-rotation measurement before/after preprocessing\n"
        f"maximum absolute difference = {maximum_difference:.3e} degrees"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return maximum_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to kimore_manifest.csv")
    parser.add_argument(
        "--exercise",
        default="Es3",
        choices=[f"Es{i}" for i in range(1, 6)],
    )
    parser.add_argument("--subject", default="B_ID1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/preprocessing"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, _ = read_manifest(args.manifest.expanduser().resolve(), args.exercise)
    sample = find_sample(samples, args.subject)
    raw = load_joint_positions(sample.position_path)
    preprocessed = preprocess_sequence(raw)
    processed = JointSequence(
        positions=preprocessed.positions,
        tracking_states=preprocessed.tracking_states,
    )

    output_dir = args.output_dir.expanduser().resolve()
    signal_path = output_dir / f"{sample.sample_id}_spine_shoulder_x_before_after.png"
    yaw_path = output_dir / f"{sample.sample_id}_shoulder_yaw_before_after.png"
    save_joint_signal_plot(
        sample.sample_id,
        sample.score,
        raw,
        processed,
        signal_path,
    )
    maximum_difference = save_shoulder_yaw_plot(
        sample.sample_id,
        raw,
        processed,
        yaw_path,
    )

    print(f"Sample: {sample.sample_id}")
    print(f"Signal plot: {signal_path}")
    print(f"Shoulder-yaw plot: {yaw_path}")
    print(
        "Maximum raw/processed shoulder-yaw difference: "
        f"{maximum_difference:.12g} degrees"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
