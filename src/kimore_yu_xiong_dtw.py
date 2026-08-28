"""Yu--Xiong angular-DTW features and scoring for KIMORE skeletons.

Yu and Xiong (Sensors 2019, 19, 2882) compare eight normalized limb-bone
vectors in a body-local coordinate system and add the body's forward vector as
a ninth dimension.  A multidimensional DTW local cost is the sum of the nine
corresponding vector-angle differences.  Their Equation (5) converts the
optimal-path cost to a percentage score.

KIMORE has no separately recorded virtual-coach execution.  Reference
selection is therefore kept outside the paper algorithm and uses only the
current outer training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from kimore_dataset import (
    JOINT_INDEX,
    JointSequence,
    KimoreSample,
    load_joint_positions,
)


BONE_JOINTS = (
    ("ShoulderLeft", "ElbowLeft"),
    ("ElbowLeft", "WristLeft"),
    ("ShoulderRight", "ElbowRight"),
    ("ElbowRight", "WristRight"),
    ("HipLeft", "KneeLeft"),
    ("KneeLeft", "AnkleLeft"),
    ("HipRight", "KneeRight"),
    ("KneeRight", "AnkleRight"),
)
FEATURE_NAMES = (
    "left_upper_arm",
    "left_lower_arm",
    "right_upper_arm",
    "right_lower_arm",
    "left_upper_leg",
    "left_lower_leg",
    "right_upper_leg",
    "right_lower_leg",
    "body_forward",
)
FEATURE_NAME = "yu_xiong_8_local_bones_plus_body_forward"
FEATURE_DIMENSIONS = len(FEATURE_NAMES)
PAPER_MAX_ANGLE_DEGREES = 90.0

REQUIRED_JOINTS = tuple(
    sorted({joint_name for bone in BONE_JOINTS for joint_name in bone})
)


@dataclass(frozen=True)
class YuXiongPreparedSample:
    sample: KimoreSample
    vectors: np.ndarray  # shape: (frames, 9, 3), unit vectors
    required_joints_tracked_fraction: float


@dataclass(frozen=True)
class YuXiongAlignment:
    path: np.ndarray  # shape: (path steps, 2), [sample index, reference index]
    total_angular_cost_degrees: float
    mean_angle_degrees: float
    paper_score_unclipped: float
    paper_score: float


def _unit_vectors(vectors: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError(f"{name} must have XYZ as its final dimension")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")

    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    invalid = lengths[..., 0] <= np.finfo(np.float64).eps
    if invalid.any():
        first = tuple(int(index) for index in np.argwhere(invalid)[0])
        raise ValueError(f"{name} contains a zero-length vector at index {first}")
    return values / lengths


def yu_xiong_vectors(sequence: JointSequence) -> np.ndarray:
    """Return the paper's eight body-local bones and world forward vector.

    The frame axes follow the construction described in Section 2.1.3: torso
    up is the vector between hip and shoulder midpoints, left is the average of
    shoulder and hip left directions made orthogonal to up, and forward is
    ``up x left``.  The eight limb bones are expressed in that orthonormal
    frame; forward remains in Kinect/world coordinates so orientation is still
    represented as the ninth comparison dimension.
    """
    positions = np.asarray(sequence.positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (len(JOINT_INDEX), 3):
        raise ValueError(
            "Joint positions must have shape "
            f"(frames, {len(JOINT_INDEX)}, 3); got {positions.shape}"
        )
    if len(positions) == 0:
        raise ValueError("Cannot extract Yu--Xiong vectors from an empty sequence")
    if not np.isfinite(positions).all():
        raise ValueError("Joint positions contain non-finite values")

    shoulder_left = positions[:, JOINT_INDEX["ShoulderLeft"]]
    shoulder_right = positions[:, JOINT_INDEX["ShoulderRight"]]
    hip_left = positions[:, JOINT_INDEX["HipLeft"]]
    hip_right = positions[:, JOINT_INDEX["HipRight"]]

    shoulder_midpoint = (shoulder_left + shoulder_right) / 2.0
    hip_midpoint = (hip_left + hip_right) / 2.0
    up = _unit_vectors(shoulder_midpoint - hip_midpoint, "body up")

    left_raw = (
        (shoulder_left - shoulder_right) + (hip_left - hip_right)
    ) / 2.0
    left_orthogonal = left_raw - np.sum(left_raw * up, axis=1, keepdims=True) * up
    left = _unit_vectors(left_orthogonal, "body left")
    forward = _unit_vectors(np.cross(up, left), "body forward")
    # Recompute left to eliminate accumulated floating-point non-orthogonality.
    left = _unit_vectors(np.cross(forward, up), "orthogonal body left")

    world_bones = np.stack(
        [
            positions[:, JOINT_INDEX[end]] - positions[:, JOINT_INDEX[start]]
            for start, end in BONE_JOINTS
        ],
        axis=1,
    )
    world_bones = _unit_vectors(world_bones, "limb bones")
    local_bones = np.stack(
        (
            np.einsum("fbd,fd->fb", world_bones, left),
            np.einsum("fbd,fd->fb", world_bones, up),
            np.einsum("fbd,fd->fb", world_bones, forward),
        ),
        axis=2,
    )
    vectors = np.concatenate((local_bones, forward[:, np.newaxis, :]), axis=1)
    return _unit_vectors(vectors, "Yu--Xiong features")


def prepare_yu_xiong_sample(sample: KimoreSample) -> YuXiongPreparedSample:
    sequence = load_joint_positions(sample.position_path)
    if len(sequence.positions) != sample.frames:
        raise ValueError(
            f"{sample.sample_id}: manifest says {sample.frames} frames but "
            f"the loader read {len(sequence.positions)}"
        )

    required_indices = [JOINT_INDEX[name] for name in REQUIRED_JOINTS]
    fully_tracked = np.all(sequence.tracking_states[:, required_indices] == 2, axis=1)
    return YuXiongPreparedSample(
        sample=sample,
        vectors=yu_xiong_vectors(sequence),
        required_joints_tracked_fraction=float(np.mean(fully_tracked)),
    )


def select_yu_xiong_reference(
    prepared_samples: Sequence[YuXiongPreparedSample],
    train_indices: Sequence[int],
) -> int:
    """Choose the highest-TS KIMORE coach from training only.

    The original method assumes a separately recorded virtual coach. Clinical
    TS is therefore the primary proxy for coach quality; required-joint
    tracking and sample ID resolve ties deterministically.
    """
    candidates = [int(index) for index in train_indices]
    if not candidates:
        raise ValueError("Cannot select a reference from an empty training split")
    if min(candidates) < 0 or max(candidates) >= len(prepared_samples):
        raise IndexError("Training index is outside the prepared sample list")

    return min(
        candidates,
        key=lambda index: (
            -prepared_samples[index].sample.score,
            -prepared_samples[index].required_joints_tracked_fraction,
            prepared_samples[index].sample.sample_id,
        ),
    )


def _vector_sequence(values: np.ndarray, name: str) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    expected_suffix = (FEATURE_DIMENSIONS, 3)
    if vectors.ndim != 3 or vectors.shape[1:] != expected_suffix or len(vectors) == 0:
        raise ValueError(
            f"{name} must have shape (frames, {FEATURE_DIMENSIONS}, 3)"
        )
    return _unit_vectors(vectors, name)


def _angular_row_costs(sample_frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    dots = np.einsum("kd,mkd->mk", sample_frame, reference)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return np.sum(angles, axis=1)


def yu_xiong_dtw(sample: np.ndarray, reference: np.ndarray) -> YuXiongAlignment:
    """Run exact multidimensional angular DTW and apply paper Equation (5)."""
    sample_vectors = _vector_sequence(sample, "sample")
    reference_vectors = _vector_sequence(reference, "reference")
    sample_length = len(sample_vectors)
    reference_length = len(reference_vectors)
    accumulated = np.empty((sample_length, reference_length), dtype=np.float64)

    first_row = _angular_row_costs(sample_vectors[0], reference_vectors)
    accumulated[0] = np.cumsum(first_row)
    for sample_index in range(1, sample_length):
        row_costs = _angular_row_costs(
            sample_vectors[sample_index], reference_vectors
        )
        accumulated[sample_index, 0] = (
            accumulated[sample_index - 1, 0] + row_costs[0]
        )
        for reference_index in range(1, reference_length):
            accumulated[sample_index, reference_index] = row_costs[
                reference_index
            ] + min(
                accumulated[sample_index - 1, reference_index - 1],
                accumulated[sample_index - 1, reference_index],
                accumulated[sample_index, reference_index - 1],
            )

    sample_index = sample_length - 1
    reference_index = reference_length - 1
    reverse_path = [(sample_index, reference_index)]
    while sample_index > 0 or reference_index > 0:
        if sample_index == 0:
            reference_index -= 1
        elif reference_index == 0:
            sample_index -= 1
        else:
            candidates = (
                accumulated[sample_index - 1, reference_index - 1],
                accumulated[sample_index - 1, reference_index],
                accumulated[sample_index, reference_index - 1],
            )
            direction = int(np.argmin(candidates))
            if direction == 0:
                sample_index -= 1
                reference_index -= 1
            elif direction == 1:
                sample_index -= 1
            else:
                reference_index -= 1
        reverse_path.append((sample_index, reference_index))

    path = np.asarray(reverse_path[::-1], dtype=np.int64)
    total_cost = float(accumulated[-1, -1])
    mean_angle = total_cost / (len(path) * FEATURE_DIMENSIONS)
    paper_score_unclipped = 100.0 * (
        1.0 - total_cost / (
            PAPER_MAX_ANGLE_DEGREES * FEATURE_DIMENSIONS * len(path)
        )
    )
    # The paper assumes each angle is within 90 degrees and calls Eq. (5) a
    # 0--100 score. Real Kinect inputs can violate that assumption, so retain
    # the raw value for auditability and bound the public percentage score.
    paper_score = float(np.clip(paper_score_unclipped, 0.0, 100.0))
    return YuXiongAlignment(
        path=path,
        total_angular_cost_degrees=total_cost,
        mean_angle_degrees=float(mean_angle),
        paper_score_unclipped=float(paper_score_unclipped),
        paper_score=paper_score,
    )
