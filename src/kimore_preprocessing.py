from __future__ import annotations

import numpy as np

from kimore_dataset import JOINT_INDEX, JointSequence

from dataclasses import dataclass

SCALE_CANDIDATES = {
    "torso": ("SpineBase", "SpineShoulder"),
    "shoulders": ("ShoulderLeft", "ShoulderRight"),
}

@dataclass(frozen=True)
class ScaleDiagnostic:
    median_length: float
    relative_mad: float
    usable_fraction: float
    usable_frames: int
    total_frames: int

@dataclass(frozen=True)
class PreprocessedSequence:
    positions: np.ndarray
    tracking_states: np.ndarray
    body_scale: float

def tracked_bone_lengths(
    sequence: JointSequence,
    joint_a: str,
    joint_b: str,
) -> np.ndarray:
    index_a = JOINT_INDEX[joint_a]
    index_b = JOINT_INDEX[joint_b]

    both_tracked = (
        (sequence.tracking_states[:, index_a] == 2)
        & (sequence.tracking_states[:, index_b] == 2)
    )

    vectors = (
        sequence.positions[:, index_b, :]
        - sequence.positions[:, index_a, :]
    )
    lengths = np.linalg.norm(vectors, axis=1)

    valid = (
        both_tracked
        & np.isfinite(lengths)
        & (lengths > 0)
    )
    return lengths[valid]


def diagnose_bone_scale(
    sequence: JointSequence,
    joint_a: str,
    joint_b: str,
) -> ScaleDiagnostic:
    lengths = tracked_bone_lengths(
        sequence,
        joint_a,
        joint_b,
    )

    total_frames = len(sequence.positions)
    usable_frames = len(lengths)

    if usable_frames == 0:
        raise ValueError(
            f"No fully tracked frames for "
            f"{joint_a}–{joint_b}"
        )

    median_length = float(np.median(lengths))
    mad = float(
        np.median(np.abs(lengths - median_length))
    )
    relative_mad = mad / median_length

    return ScaleDiagnostic(
        median_length=median_length,
        relative_mad=relative_mad,
        usable_fraction=usable_frames / total_frames,
        usable_frames=usable_frames,
        total_frames=total_frames,
    )

def diagnose_scale_candidates(
    sequence: JointSequence,
) -> dict[str, ScaleDiagnostic]:
    diagnostics: dict[str, ScaleDiagnostic] = {}

    for name, joints in SCALE_CANDIDATES.items():
        joint_a, joint_b = joints
        diagnostics[name] = diagnose_bone_scale(
            sequence,
            joint_a,
            joint_b,
        )

    return diagnostics

def center_on_joint(
    sequence: JointSequence,
    joint_name: str = "SpineBase",
) -> np.ndarray:
    joint_index = JOINT_INDEX[joint_name]

    origin = sequence.positions[
        :,
        joint_index : joint_index + 1,
        :,
    ]

    return sequence.positions - origin

def preprocess_sequence(
    sequence: JointSequence,
) -> PreprocessedSequence:
    scale_diagnostic = diagnose_bone_scale(
        sequence,
        "SpineBase",
        "SpineShoulder",
    )
    body_scale = scale_diagnostic.median_length

    centered = center_on_joint(
        sequence,
        "SpineBase",
    )
    normalized = centered / body_scale

    if not np.isfinite(normalized).all():
        raise ValueError(
            "Preprocessing produced non-finite coordinates"
        )

    spine_base_index = JOINT_INDEX["SpineBase"]
    if not np.allclose(
        normalized[:, spine_base_index, :],
        0.0,
    ):
        raise ValueError(
            "SpineBase is not zero after centring"
        )

    return PreprocessedSequence(
        positions=normalized,
        tracking_states=(
            sequence.tracking_states.copy()
        ),
        body_scale=body_scale,
    )