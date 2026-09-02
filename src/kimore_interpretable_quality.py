"""Frame-level input quality control for interpretable full-body DTW.

The raw KIMORE files are never modified. QC is applied to the nine unit-vector
features consumed by the Yu--Xiong DTW: short, bounded invalid runs are
interpolated on the unit sphere and remaining invalid frames are removed from
the QC sequence. The untouched vectors remain available for the all-sample
diagnostic output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kimore_dataset import JOINT_INDEX, JointSequence
from kimore_yu_xiong_dtw import BONE_JOINTS, FEATURE_DIMENSIONS, FEATURE_NAMES


BODY_FRAME_JOINTS = (
    "ShoulderLeft",
    "ShoulderRight",
    "HipLeft",
    "HipRight",
)
LEG_COMPONENT_INDICES = frozenset((4, 5, 6, 7))

# Five frames is a deliberately short hole (roughly 0.17 s at 30 FPS). A run
# is interpolated only when it has valid frames on both sides and the endpoint
# directions are no more than 90 degrees apart.
MAX_INTERPOLATED_GAP_FRAMES = 5
MAX_INTERPOLATION_ENDPOINT_ANGLE_DEGREES = 90.0

# At least 80% of frames must remain, and a single unresolved run may not
# occupy more than 10% of the recording.
MIN_RETAINED_FRAME_FRACTION = 0.80
MAX_UNREPAIRED_RUN_FRACTION = 0.10

# Fully tracked means Kinect state 2. State 1 (inferred) is retained when its
# geometry is plausible, but systematic inference remains a warning/failure.
# State 0 is invalid at frame level.
MIN_USABLE_TRACKED_FRACTION = 0.10
MIN_PASS_TRACKED_FRACTION = 0.50
MIN_PASS_STABLE_LENGTH_FRACTION = 0.90
MIN_PASS_TEMPORAL_CONTINUITY_FRACTION = 0.95
MIN_PASS_ANATOMICAL_PLAUSIBILITY_FRACTION = 0.80
BONE_LENGTH_RELATIVE_TOLERANCE = 0.25
MAX_FRAME_SOURCE_LENGTH_RELATIVE_ERROR = 0.50
MAX_FRAME_ANGLE_CHANGE_DEGREES = 45.0


@dataclass(frozen=True)
class ComponentQualitySummary:
    component_index: int
    component_name: str
    source_joints: tuple[str, ...]
    tracked_fraction: float
    stable_length_fraction: float
    temporal_continuity_fraction: float
    anatomical_plausibility_fraction: float | None
    not_fully_tracked_frames: int
    untracked_frames: int
    source_length_invalid_frames: int
    temporal_outlier_frames: int
    anatomical_invalid_frames: int
    body_frame_invalid_frames: int
    invalid_frames: int
    invalid_fraction: float
    invalid_run_count: int
    longest_invalid_run: int
    interpolated_frames: int
    unrepaired_invalid_frames: int
    unrepaired_invalid_fraction: float
    longest_unrepaired_run: int
    quality_status: str
    quality_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FrameQualityResult:
    """Repaired vectors, the retained QC sequence, and transparent diagnostics."""

    repaired_vectors: np.ndarray
    cleaned_vectors: np.ndarray
    component_summaries: tuple[ComponentQualitySummary, ...]
    retained_frame_indices: np.ndarray
    interpolated_component_mask: np.ndarray
    dropped_frame_mask: np.ndarray
    total_frames: int
    interpolated_frames: int
    interpolated_component_frames: int
    dropped_frames: int
    retained_frames: int
    retained_fraction: float
    longest_dropped_run: int
    quality_status: str
    quality_reasons: tuple[str, ...]


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Frame mask must be one-dimensional")
    padded = np.concatenate(([False], values, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _longest_true_run(mask: np.ndarray) -> int:
    runs = _true_runs(mask)
    return max((end - start for start, end in runs), default=0)


def _stable_mask(
    lengths: np.ndarray,
    baseline_frames: np.ndarray,
    relative_tolerance: float = BONE_LENGTH_RELATIVE_TOLERANCE,
) -> np.ndarray:
    values = np.asarray(lengths, dtype=np.float64)
    candidates = np.asarray(baseline_frames, dtype=bool)
    if values.ndim != 1 or len(values) == 0 or candidates.shape != values.shape:
        raise ValueError("Source-vector lengths and baseline mask must align")
    finite_positive = np.isfinite(values) & (
        values > np.finfo(np.float64).eps
    )
    baseline = values[candidates & finite_positive]
    if len(baseline) == 0:
        baseline = values[finite_positive]
    if len(baseline) == 0:
        return np.zeros(len(values), dtype=bool)
    median = float(np.median(baseline))
    relative_error = np.abs(values - median) / median
    return finite_positive & (relative_error <= relative_tolerance)


def _stable_fraction(lengths: np.ndarray) -> float:
    """Compatibility helper for recording-level length diagnostics."""
    values = np.asarray(lengths, dtype=np.float64)
    stable = _stable_mask(values, np.ones(len(values), dtype=bool))
    return float(np.mean(stable))


def _vector_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dots = np.sum(first * second, axis=-1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def _temporal_continuity_fraction(vectors: np.ndarray) -> float:
    values = np.asarray(vectors, dtype=np.float64)
    if len(values) < 2:
        return 1.0
    changes = _vector_angles(values[1:], values[:-1])
    return float(np.mean(changes <= MAX_FRAME_ANGLE_CHANGE_DEGREES))


def _temporal_outlier_mask(
    vectors: np.ndarray,
    max_outlier_frames: int = MAX_INTERPOLATED_GAP_FRAMES,
) -> np.ndarray:
    """Identify brief excursions that jump away and then return.

    A lone large transition may be real motion. Paired jump boundaries with
    mutually consistent outer frames identify an isolated excursion. Edge
    frames are handled only when the two following/preceding frames agree.
    """
    values = np.asarray(vectors, dtype=np.float64)
    count = len(values)
    invalid = np.zeros(count, dtype=bool)
    if count < 2:
        return invalid

    jumps = _vector_angles(values[1:], values[:-1]) > MAX_FRAME_ANGLE_CHANGE_DEGREES
    boundaries = np.flatnonzero(jumps)
    boundary_index = 0
    while boundary_index + 1 < len(boundaries):
        first = int(boundaries[boundary_index])
        second = int(boundaries[boundary_index + 1])
        outer_angle = float(
            _vector_angles(
                values[first][np.newaxis, :],
                values[second + 1][np.newaxis, :],
            )[0]
        )
        excursion_length = second - first
        if (
            excursion_length <= max_outlier_frames
            and outer_angle <= MAX_FRAME_ANGLE_CHANGE_DEGREES
        ):
            invalid[first + 1 : second + 1] = True
            boundary_index += 2
        else:
            boundary_index += 1

    if count >= 3:
        if jumps[0] and not jumps[1]:
            invalid[0] = True
        if jumps[-1] and not jumps[-2]:
            invalid[-1] = True
    return invalid


def _body_frame_masks(
    sequence: JointSequence,
    body_forward: np.ndarray,
    exercise: str,
    max_temporal_outlier_frames: int = MAX_INTERPOLATED_GAP_FRAMES,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    positions = sequence.positions
    tracking_states = sequence.tracking_states
    body_indices = [JOINT_INDEX[name] for name in BODY_FRAME_JOINTS]
    fully_tracked = np.all(tracking_states[:, body_indices] == 2, axis=1)
    observed = np.all(tracking_states[:, body_indices] != 0, axis=1)

    shoulder_left = positions[:, JOINT_INDEX["ShoulderLeft"]]
    shoulder_right = positions[:, JOINT_INDEX["ShoulderRight"]]
    hip_left = positions[:, JOINT_INDEX["HipLeft"]]
    hip_right = positions[:, JOINT_INDEX["HipRight"]]
    shoulder_midpoint = (shoulder_left + shoulder_right) / 2.0
    hip_midpoint = (hip_left + hip_right) / 2.0
    source_lengths = (
        np.linalg.norm(shoulder_midpoint - hip_midpoint, axis=1),
        np.linalg.norm(shoulder_left - shoulder_right, axis=1),
        np.linalg.norm(hip_left - hip_right, axis=1),
    )
    stable = np.ones(len(positions), dtype=bool)
    diagnostically_stable = np.ones(len(positions), dtype=bool)
    for lengths in source_lengths:
        stable &= _stable_mask(
            lengths,
            observed,
            relative_tolerance=MAX_FRAME_SOURCE_LENGTH_RELATIVE_ERROR,
        )
        diagnostically_stable &= _stable_mask(lengths, observed)

    temporal_outlier = _temporal_outlier_mask(
        body_forward,
        max_outlier_frames=max_temporal_outlier_frames,
    )
    anatomical = np.ones(len(positions), dtype=bool)
    if exercise.casefold() == "es3":
        anatomical = (shoulder_midpoint - hip_midpoint)[:, 1] > 0.0
    return (
        fully_tracked,
        observed,
        stable,
        diagnostically_stable,
        temporal_outlier,
        anatomical,
    )


def _body_frame_stability(sequence: JointSequence) -> float:
    dummy_forward = np.zeros((len(sequence.positions), 3), dtype=np.float64)
    dummy_forward[:, 0] = 1.0
    _, _, _, stable, _, _ = _body_frame_masks(sequence, dummy_forward, "")
    return float(np.mean(stable))


def _spherical_interpolation(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
) -> np.ndarray:
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle <= 1e-8:
        value = (1.0 - fraction) * start + fraction * end
    else:
        denominator = float(np.sin(angle))
        value = (
            np.sin((1.0 - fraction) * angle) / denominator * start
            + np.sin(fraction * angle) / denominator * end
        )
    length = float(np.linalg.norm(value))
    if length <= np.finfo(np.float64).eps:
        raise ValueError("Cannot interpolate antipodal unit vectors")
    return value / length


def _interpolate_short_runs(
    vectors: np.ndarray,
    invalid_mask: np.ndarray,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    repaired = np.asarray(vectors, dtype=np.float64).copy()
    invalid = np.asarray(invalid_mask, dtype=bool)
    interpolated = np.zeros(len(repaired), dtype=bool)
    for start, end in _true_runs(invalid):
        run_length = end - start
        if (
            run_length > max_gap_frames
            or start == 0
            or end == len(repaired)
            or invalid[start - 1]
            or invalid[end]
        ):
            continue
        endpoint_angle = float(
            _vector_angles(
                repaired[start - 1][np.newaxis, :],
                repaired[end][np.newaxis, :],
            )[0]
        )
        if endpoint_angle > MAX_INTERPOLATION_ENDPOINT_ANGLE_DEGREES:
            continue
        for frame in range(start, end):
            fraction = (frame - start + 1) / (run_length + 1)
            repaired[frame] = _spherical_interpolation(
                repaired[start - 1], repaired[end], fraction
            )
            interpolated[frame] = True
    return repaired, interpolated


def _component_status(
    tracked_fraction: float,
    stable_length_fraction: float,
    temporal_continuity_fraction: float,
    anatomical_plausibility_fraction: float | None,
    interpolated_frames: int,
    unrepaired_mask: np.ndarray,
) -> tuple[str, tuple[str, ...]]:
    failures: list[str] = []
    warnings: list[str] = []
    total_frames = len(unrepaired_mask)
    unrepaired_frames = int(np.sum(unrepaired_mask))
    unrepaired_fraction = unrepaired_frames / total_frames
    longest_unrepaired = _longest_true_run(unrepaired_mask)

    if tracked_fraction < MIN_USABLE_TRACKED_FRACTION:
        failures.append("tracking_below_10_percent")
    elif tracked_fraction < MIN_PASS_TRACKED_FRACTION:
        warnings.append("tracking_below_50_percent")

    if unrepaired_fraction > 1.0 - MIN_RETAINED_FRAME_FRACTION:
        failures.append("too_many_unrepairable_frames")
    if longest_unrepaired / total_frames > MAX_UNREPAIRED_RUN_FRACTION:
        failures.append("long_unrepairable_frame_run")

    if stable_length_fraction < MIN_PASS_STABLE_LENGTH_FRACTION:
        warnings.append("source_length_warning")
    if temporal_continuity_fraction < MIN_PASS_TEMPORAL_CONTINUITY_FRACTION:
        warnings.append("frame_to_frame_angle_jump_warning")
    if (
        anatomical_plausibility_fraction is not None
        and anatomical_plausibility_fraction
        < MIN_PASS_ANATOMICAL_PLAUSIBILITY_FRACTION
    ):
        warnings.append("anatomical_direction_warning")

    if unrepaired_frames:
        warnings.append("unrepairable_frames_removed")
        if (
            anatomical_plausibility_fraction is not None
            and anatomical_plausibility_fraction < 0.50
        ):
            failures.append("anatomically_implausible_direction")
    if interpolated_frames:
        warnings.append("short_frame_gaps_interpolated")

    reasons = tuple(dict.fromkeys(failures + warnings))
    if failures:
        return "fail", reasons
    if warnings:
        return "warning", reasons
    return "pass", ()


def apply_frame_quality_control(
    sequence: JointSequence,
    vectors: np.ndarray,
    exercise: str,
    max_interpolation_gap_frames: int = MAX_INTERPOLATED_GAP_FRAMES,
) -> FrameQualityResult:
    """Detect, repair, and remove invalid feature frames for one recording."""
    positions = np.asarray(sequence.positions, dtype=np.float64)
    tracking_states = np.asarray(sequence.tracking_states)
    feature_vectors = np.asarray(vectors, dtype=np.float64)
    expected_vector_shape = (len(positions), FEATURE_DIMENSIONS, 3)
    if positions.ndim != 3 or positions.shape[1:] != (len(JOINT_INDEX), 3):
        raise ValueError("Joint positions have an invalid shape")
    if tracking_states.shape != positions.shape[:2]:
        raise ValueError("Tracking states do not match joint positions")
    if feature_vectors.shape != expected_vector_shape:
        raise ValueError(
            f"Feature vectors must have shape {expected_vector_shape}; "
            f"got {feature_vectors.shape}"
        )
    if len(positions) == 0:
        raise ValueError("Cannot quality-check an empty recording")
    if max_interpolation_gap_frames < 0:
        raise ValueError("Maximum interpolation gap cannot be negative")
    if not np.isfinite(positions).all() or not np.isfinite(feature_vectors).all():
        raise ValueError("Quality inputs must be finite")

    (
        body_fully_tracked,
        body_observed,
        body_stable,
        body_diagnostic_stable,
        body_temporal_outlier,
        body_anatomical,
    ) = _body_frame_masks(
        sequence,
        feature_vectors[:, 8],
        exercise,
        max_temporal_outlier_frames=max_interpolation_gap_frames,
    )
    body_invalid = (
        ~body_observed
        | ~body_stable
        | body_temporal_outlier
        | ~body_anatomical
    )

    component_inputs: list[dict[str, object]] = []
    body_indices = [JOINT_INDEX[name] for name in BODY_FRAME_JOINTS]
    for component_index, bone in enumerate(BONE_JOINTS):
        start, end = bone
        bone_indices = [JOINT_INDEX[start], JOINT_INDEX[end]]
        required_indices = sorted(set(bone_indices + body_indices))
        # Report tracking confidence for the bone's own endpoints. The body
        # frame has its own component-level tracking diagnostic; requiring the
        # intersection of both here would double-penalize otherwise plausible
        # limb vectors. State-0 body joints still invalidate the frame below.
        fully_tracked = np.all(
            tracking_states[:, bone_indices] == 2, axis=1
        )
        observed = np.all(tracking_states[:, required_indices] != 0, axis=1)
        source_lengths = np.linalg.norm(
            positions[:, bone_indices[1]] - positions[:, bone_indices[0]],
            axis=1,
        )
        source_stable = _stable_mask(
            source_lengths,
            observed,
            relative_tolerance=MAX_FRAME_SOURCE_LENGTH_RELATIVE_ERROR,
        )
        diagnostic_source_stable = _stable_mask(source_lengths, observed)
        temporal_outlier = _temporal_outlier_mask(
            feature_vectors[:, component_index],
            max_outlier_frames=max_interpolation_gap_frames,
        )
        anatomical = np.ones(len(positions), dtype=bool)
        if exercise.casefold() == "es3" and component_index in LEG_COMPONENT_INDICES:
            anatomical = feature_vectors[:, component_index, 1] < 0.0
        invalid = (
            ~observed
            | ~source_stable
            | temporal_outlier
            | ~anatomical
            | body_invalid
        )
        component_inputs.append(
            {
                "source_joints": bone,
                "fully_tracked": fully_tracked,
                "observed": observed,
                "source_stable": source_stable,
                "diagnostic_source_stable": diagnostic_source_stable,
                "temporal_outlier": temporal_outlier,
                "anatomical": anatomical,
                "body_invalid": body_invalid,
                "invalid": invalid,
            }
        )

    component_inputs.append(
        {
            "source_joints": BODY_FRAME_JOINTS,
            "fully_tracked": body_fully_tracked,
            "observed": body_observed,
            "source_stable": body_stable,
            "diagnostic_source_stable": body_diagnostic_stable,
            "temporal_outlier": body_temporal_outlier,
            "anatomical": body_anatomical,
            "body_invalid": body_invalid,
            "invalid": body_invalid,
        }
    )

    repaired_vectors = feature_vectors.copy()
    interpolated_masks = np.zeros(
        (len(positions), FEATURE_DIMENSIONS), dtype=bool
    )
    unrepaired_masks = np.zeros_like(interpolated_masks)
    summaries: list[ComponentQualitySummary] = []
    for component_index, inputs in enumerate(component_inputs):
        invalid = np.asarray(inputs["invalid"], dtype=bool)
        repaired_component, interpolated = _interpolate_short_runs(
            feature_vectors[:, component_index],
            invalid,
            max_interpolation_gap_frames,
        )
        repaired_vectors[:, component_index] = repaired_component
        interpolated_masks[:, component_index] = interpolated
        unrepaired = invalid & ~interpolated
        unrepaired_masks[:, component_index] = unrepaired

        fully_tracked = np.asarray(inputs["fully_tracked"], dtype=bool)
        observed = np.asarray(inputs["observed"], dtype=bool)
        source_stable = np.asarray(inputs["source_stable"], dtype=bool)
        diagnostic_source_stable = np.asarray(
            inputs["diagnostic_source_stable"], dtype=bool
        )
        temporal_outlier = np.asarray(inputs["temporal_outlier"], dtype=bool)
        anatomical = np.asarray(inputs["anatomical"], dtype=bool)
        body_component_invalid = np.asarray(inputs["body_invalid"], dtype=bool)
        tracked_fraction = float(np.mean(fully_tracked))
        stable_length_fraction = float(np.mean(diagnostic_source_stable))
        temporal_continuity_fraction = _temporal_continuity_fraction(
            feature_vectors[:, component_index]
        )
        anatomical_plausibility_fraction = (
            float(np.mean(anatomical))
            if exercise.casefold() == "es3"
            and (component_index in LEG_COMPONENT_INDICES or component_index == 8)
            else None
        )
        quality_status, quality_reasons = _component_status(
            tracked_fraction=tracked_fraction,
            stable_length_fraction=stable_length_fraction,
            temporal_continuity_fraction=temporal_continuity_fraction,
            anatomical_plausibility_fraction=anatomical_plausibility_fraction,
            interpolated_frames=int(np.sum(interpolated)),
            unrepaired_mask=unrepaired,
        )
        summaries.append(
            ComponentQualitySummary(
                component_index=component_index,
                component_name=FEATURE_NAMES[component_index],
                source_joints=tuple(inputs["source_joints"]),
                tracked_fraction=tracked_fraction,
                stable_length_fraction=stable_length_fraction,
                temporal_continuity_fraction=temporal_continuity_fraction,
                anatomical_plausibility_fraction=anatomical_plausibility_fraction,
                not_fully_tracked_frames=int(np.sum(~fully_tracked)),
                untracked_frames=int(np.sum(~observed)),
                source_length_invalid_frames=int(np.sum(~source_stable)),
                temporal_outlier_frames=int(np.sum(temporal_outlier)),
                anatomical_invalid_frames=int(np.sum(~anatomical)),
                body_frame_invalid_frames=int(np.sum(body_component_invalid)),
                invalid_frames=int(np.sum(invalid)),
                invalid_fraction=float(np.mean(invalid)),
                invalid_run_count=len(_true_runs(invalid)),
                longest_invalid_run=_longest_true_run(invalid),
                interpolated_frames=int(np.sum(interpolated)),
                unrepaired_invalid_frames=int(np.sum(unrepaired)),
                unrepaired_invalid_fraction=float(np.mean(unrepaired)),
                longest_unrepaired_run=_longest_true_run(unrepaired),
                quality_status=quality_status,
                quality_reasons=quality_reasons,
            )
        )

    dropped_frame_mask = np.any(unrepaired_masks, axis=1)
    retained_frame_indices = np.flatnonzero(~dropped_frame_mask)
    cleaned_vectors = repaired_vectors[~dropped_frame_mask].copy()
    total_frames = len(positions)
    dropped_frames = int(np.sum(dropped_frame_mask))
    retained_frames = total_frames - dropped_frames
    retained_fraction = retained_frames / total_frames
    longest_dropped_run = _longest_true_run(dropped_frame_mask)

    recording_failures: list[str] = []
    recording_warnings: list[str] = []
    component_status = overall_quality_status(tuple(summaries))
    if component_status == "fail":
        recording_failures.append("component_quality_failure")
    elif component_status == "warning":
        recording_warnings.append("component_quality_warning")
    if retained_frames < 2:
        recording_failures.append("fewer_than_two_usable_frames")
    if retained_fraction < MIN_RETAINED_FRAME_FRACTION:
        recording_failures.append("retained_frame_fraction_below_80_percent")
    if longest_dropped_run / total_frames > MAX_UNREPAIRED_RUN_FRACTION:
        recording_failures.append("long_unrepairable_frame_run")
    if dropped_frames:
        recording_warnings.append("unrepairable_frames_removed")
    unique_interpolated = int(np.sum(np.any(interpolated_masks, axis=1)))
    if unique_interpolated:
        recording_warnings.append("short_frame_gaps_interpolated")

    quality_reasons = tuple(
        dict.fromkeys(recording_failures + recording_warnings)
    )
    if recording_failures:
        quality_status = "fail"
    elif recording_warnings:
        quality_status = "warning"
    else:
        quality_status = "pass"

    return FrameQualityResult(
        repaired_vectors=repaired_vectors,
        cleaned_vectors=cleaned_vectors,
        component_summaries=tuple(summaries),
        retained_frame_indices=retained_frame_indices,
        interpolated_component_mask=interpolated_masks,
        dropped_frame_mask=dropped_frame_mask,
        total_frames=total_frames,
        interpolated_frames=unique_interpolated,
        interpolated_component_frames=int(np.sum(interpolated_masks)),
        dropped_frames=dropped_frames,
        retained_frames=retained_frames,
        retained_fraction=retained_fraction,
        longest_dropped_run=longest_dropped_run,
        quality_status=quality_status,
        quality_reasons=quality_reasons,
    )


def assess_component_quality(
    sequence: JointSequence,
    vectors: np.ndarray,
    exercise: str,
) -> tuple[ComponentQualitySummary, ...]:
    """Return component summaries while preserving the original public helper."""
    return apply_frame_quality_control(
        sequence, vectors, exercise
    ).component_summaries


def overall_quality_status(
    summaries: tuple[ComponentQualitySummary, ...],
) -> str:
    """Return the most severe component status for one recording."""
    if len(summaries) != FEATURE_DIMENSIONS:
        raise ValueError(f"Expected {FEATURE_DIMENSIONS} quality summaries")
    statuses = {summary.quality_status for summary in summaries}
    if not statuses.issubset({"pass", "warning", "fail"}):
        raise ValueError(f"Unexpected quality status: {sorted(statuses)}")
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    return "pass"
