r"""Load KIMORE joint-position sequences selected from the audit manifest.

This is the first reusable data module for the interpretable-DTW project. It
does not clean or overwrite KIMORE. It selects samples with a clinical Total
Score and a structurally valid JointPosition CSV, then loads each recording as
``(frames, 25, 3)`` coordinates plus ``(frames, 25)`` tracking states.

Example (PowerShell):

    python kimore_dataset.py .\kimore_audit_output\kimore_manifest.csv

Choose a subject and save a diagnostic joint plot:

    python kimore_dataset.py .\kimore_audit_output\kimore_manifest.csv `
        --exercise Es3 --subject B_ID1 --joint SpineShoulder --plot
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "SpineBase",
    "SpineMid",
    "Neck",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
    "SpineShoulder",
    "HandTipLeft",
    "ThumbLeft",
    "HandTipRight",
    "ThumbRight",
)
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}


@dataclass(frozen=True)
class KimoreSample:
    sample_id: str
    subject_id: str
    cohort: str
    exercise: str
    score: float
    position_path: Path
    frames: int
    audit_issues: str


@dataclass(frozen=True)
class JointSequence:
    positions: np.ndarray  # shape: (frames, 25, 3)
    tracking_states: np.ndarray  # shape: (frames, 25)


def optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def optional_int(value: str | None) -> int | None:
    parsed = optional_float(value)
    return int(parsed) if parsed is not None else None


def explain_position_exclusion(row: dict[str, str]) -> str | None:
    """Return why a row cannot be used by a position-based model."""
    if optional_float(row.get("clinical_ts")) is None:
        return "missing clinical TS target"
    if not (row.get("position_path") or "").strip():
        issues = row.get("issues") or ""
        if "multiple JointPosition" in issues:
            return "multiple position recordings require manual resolution"
        return "missing JointPosition CSV"
    if optional_int(row.get("position_frames")) in (None, 0):
        return "empty JointPosition CSV"
    if optional_int(row.get("position_columns")) != 100:
        return "JointPosition rows do not have 100 values"
    issues = row.get("issues") or ""
    if "position has" in issues and "inconsistent-width rows" in issues:
        return "JointPosition contains inconsistent-width rows"
    return None


def read_manifest(
    manifest_path: Path, exercise: str = "Es3"
) -> tuple[list[KimoreSample], dict[str, str]]:
    selected: list[KimoreSample] = []
    excluded: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "subject_id",
            "cohort",
            "exercise",
            "clinical_ts",
            "position_frames",
            "position_columns",
            "position_path",
            "issues",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        for row in reader:
            if row["exercise"].casefold() != exercise.casefold():
                continue
            reason = explain_position_exclusion(row)
            if reason:
                excluded[row["sample_id"]] = reason
                continue
            score = optional_float(row["clinical_ts"])
            frames = optional_int(row["position_frames"])
            assert score is not None and frames is not None
            selected.append(
                KimoreSample(
                    sample_id=row["sample_id"],
                    subject_id=row["subject_id"],
                    cohort=row["cohort"],
                    exercise=row["exercise"],
                    score=score,
                    position_path=Path(row["position_path"]),
                    frames=frames,
                    audit_issues=row.get("issues") or "",
                )
            )
    return selected, excluded


def trimmed_row(row: list[str]) -> list[str]:
    while row and not row[-1].strip():
        row.pop()
    return row


def load_joint_positions(path: Path) -> JointSequence:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_row in enumerate(csv.reader(handle), start=1):
            row = trimmed_row(raw_row)
            if not row:
                continue
            if len(row) != 100:
                raise ValueError(
                    f"{path}: nonblank line {line_number} has {len(row)} values, expected 100"
                )
            try:
                rows.append([float(value) for value in row])
            except ValueError as error:
                raise ValueError(
                    f"{path}: nonnumeric value on line {line_number}"
                ) from error

    if not rows:
        raise ValueError(f"No joint-position frames found in {path}")

    raw = np.asarray(rows, dtype=np.float64).reshape(-1, len(JOINT_NAMES), 4)
    positions = raw[:, :, :3]
    tracking_states = raw[:, :, 3].astype(np.int8)
    return JointSequence(positions=positions, tracking_states=tracking_states)


def find_sample(samples: list[KimoreSample], subject_id: str | None) -> KimoreSample:
    if not samples:
        raise ValueError("No usable samples were selected")
    if subject_id is None:
        return samples[0]
    wanted = subject_id.casefold()
    for sample in samples:
        if sample.subject_id.casefold() == wanted:
            return sample
    available = ", ".join(sample.subject_id for sample in samples[:12])
    raise ValueError(
        f"Subject {subject_id!r} is not usable for this exercise. "
        f"First available subjects: {available}"
    )


def save_joint_plot(
    sample: KimoreSample,
    sequence: JointSequence,
    joint_name: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    try:
        joint_index = JOINT_INDEX[joint_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown joint {joint_name!r}. Choose one of: {', '.join(JOINT_NAMES)}"
        ) from error

    trajectory = sequence.positions[:, joint_index, :]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    for coordinate, label in enumerate(("X", "Y", "Z")):
        axis.plot(trajectory[:, coordinate], label=label, linewidth=1)
    axis.set_title(
        f"{sample.sample_id} – {joint_name} position – clinical TS {sample.score:.3f}"
    )
    axis.set_xlabel("Frame")
    axis.set_ylabel("Kinect coordinate")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to kimore_manifest.csv")
    parser.add_argument("--exercise", default="Es3", choices=[f"Es{i}" for i in range(1, 6)])
    parser.add_argument("--subject", help="Optional subject ID, for example B_ID1")
    parser.add_argument("--joint", default="SpineShoulder", choices=JOINT_NAMES)
    parser.add_argument("--plot", action="store_true", help="Save a diagnostic PNG")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("kimore_sample_signal.png"),
        help="Diagnostic plot path (default: ./kimore_sample_signal.png)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    samples, excluded = read_manifest(manifest_path, args.exercise)
    print(f"Exercise: {args.exercise}")
    print(f"Usable position-and-score samples: {len(samples)}")
    print(f"Excluded samples: {len(excluded)}")
    for sample_id, reason in excluded.items():
        print(f"  {sample_id}: {reason}")

    sample = find_sample(samples, args.subject)
    if not sample.position_path.is_file():
        raise FileNotFoundError(
            f"The manifest points to a file that is not available: {sample.position_path}. "
            "Rerun the audit if KIMORE was moved."
        )
    sequence = load_joint_positions(sample.position_path)
    if len(sequence.positions) != sample.frames:
        raise ValueError(
            f"Manifest says {sample.frames} frames but loader read {len(sequence.positions)}"
        )

    unique_states, state_counts = np.unique(
        sequence.tracking_states, return_counts=True
    )
    tracking_summary = ", ".join(
        f"{int(state)}={int(count)}" for state, count in zip(unique_states, state_counts)
    )
    print(f"Loaded sample: {sample.sample_id}")
    print(f"Cohort: {sample.cohort}")
    print(f"Clinical TS: {sample.score:.3f}")
    print(f"Position shape: {sequence.positions.shape}")
    print(f"Tracking-state counts: {tracking_summary}")
    print(f"All coordinates finite: {bool(np.isfinite(sequence.positions).all())}")

    if args.plot:
        output_path = args.plot_path.expanduser().resolve()
        save_joint_plot(sample, sequence, args.joint, output_path)
        print(f"Plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
