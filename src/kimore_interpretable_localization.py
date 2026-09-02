"""Localize QC-usable interpretable-DTW deviations in original frame time."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from kimore_interpretable_dtw import InterpretableDtwAlignment
from kimore_interpretable_quality import FrameQualityResult
from kimore_yu_xiong_dtw import FEATURE_DIMENSIONS, FEATURE_NAMES, YuXiongPreparedSample


DEVIATION_WINDOW_COUNT = 20
TOP_DEVIATION_INTERVALS_PER_SAMPLE = 5
CANDIDATE_INTERPRETATION = (
    "candidate_deviation_from_reference_not_validated_as_execution_error"
)


def _progress_percent(frame_index: int, total_frames: int) -> float:
    if total_frames < 1:
        raise ValueError("Total frames must be positive")
    if total_frames == 1:
        return 0.0
    return 100.0 * frame_index / (total_frames - 1)


def _validate_inputs(
    prepared: YuXiongPreparedSample,
    reference: YuXiongPreparedSample,
    alignment: InterpretableDtwAlignment,
    sample_quality: FrameQualityResult,
    reference_quality: FrameQualityResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_original = np.asarray(
        sample_quality.retained_frame_indices, dtype=np.int64
    )
    reference_original = np.asarray(
        reference_quality.retained_frame_indices, dtype=np.int64
    )
    path = np.asarray(alignment.path, dtype=np.int64)
    errors = np.asarray(alignment.component_errors_degrees, dtype=np.float64)

    if sample_quality.quality_status == "fail":
        raise ValueError("Cannot localize a QC-failed sample")
    if reference_quality.quality_status == "fail":
        raise ValueError("Cannot localize against a QC-failed reference")
    if sample_original.shape != (len(prepared.vectors),):
        raise ValueError("Sample retained-frame map does not match QC vectors")
    if reference_original.shape != (len(reference.vectors),):
        raise ValueError("Reference retained-frame map does not match QC vectors")
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) == 0:
        raise ValueError("Alignment path must have shape (steps, 2)")
    if errors.shape != (len(path), FEATURE_DIMENSIONS):
        raise ValueError("Component errors do not match the alignment path")
    if np.any(path[:, 0] < 0) or np.any(path[:, 0] >= len(sample_original)):
        raise IndexError("Alignment contains an invalid sample frame")
    if np.any(path[:, 1] < 0) or np.any(path[:, 1] >= len(reference_original)):
        raise IndexError("Alignment contains an invalid reference frame")
    if sample_quality.interpolated_component_mask.shape != (
        sample_quality.total_frames,
        FEATURE_DIMENSIONS,
    ):
        raise ValueError("Sample interpolation mask has an invalid shape")
    if reference_quality.interpolated_component_mask.shape != (
        reference_quality.total_frames,
        FEATURE_DIMENSIONS,
    ):
        raise ValueError("Reference interpolation mask has an invalid shape")
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise ValueError("Timeline errors must be finite and non-negative")
    return sample_original, reference_original, path, errors


def iter_frame_timeline_rows(
    fold_number: int,
    prepared: YuXiongPreparedSample,
    reference: YuXiongPreparedSample,
    alignment: InterpretableDtwAlignment,
    sample_quality: FrameQualityResult,
    reference_quality: FrameQualityResult,
) -> Iterator[dict[str, object]]:
    """Yield one QC error row per retained sample frame and body component."""
    sample_original, reference_original, path, errors = _validate_inputs(
        prepared,
        reference,
        alignment,
        sample_quality,
        reference_quality,
    )
    sample_path = path[:, 0]
    aligned_reference_original = reference_original[path[:, 1]]
    sample_frame_count = len(sample_original)

    path_counts = np.bincount(sample_path, minlength=sample_frame_count)
    if np.any(path_counts == 0):
        raise AssertionError("DTW path does not cover every retained sample frame")
    error_sums = np.zeros((sample_frame_count, FEATURE_DIMENSIONS), dtype=float)
    error_maxima = np.zeros_like(error_sums)
    np.add.at(error_sums, sample_path, errors)
    np.maximum.at(error_maxima, sample_path, errors)
    error_means = error_sums / path_counts[:, np.newaxis]

    reference_minima = np.full(
        sample_frame_count, reference_quality.total_frames, dtype=np.int64
    )
    reference_maxima = np.full(sample_frame_count, -1, dtype=np.int64)
    np.minimum.at(reference_minima, sample_path, aligned_reference_original)
    np.maximum.at(reference_maxima, sample_path, aligned_reference_original)

    reference_interpolated = reference_quality.interpolated_component_mask[
        aligned_reference_original
    ].astype(float)
    reference_interpolated_sums = np.zeros_like(error_sums)
    np.add.at(
        reference_interpolated_sums,
        sample_path,
        reference_interpolated,
    )
    reference_interpolated_fractions = (
        reference_interpolated_sums / path_counts[:, np.newaxis]
    )

    for qc_frame_index, original_frame_index_value in enumerate(sample_original):
        original_frame_index = int(original_frame_index_value)
        reference_start = int(reference_minima[qc_frame_index])
        reference_end = int(reference_maxima[qc_frame_index])
        for component_index, component_name in enumerate(FEATURE_NAMES):
            yield {
                "fold": fold_number,
                "input_variant": "frame_qc",
                "sample_id": prepared.sample.sample_id,
                "subject_id": prepared.sample.subject_id,
                "cohort": prepared.sample.cohort,
                "actual_ts": prepared.sample.score,
                "reference_sample_id": reference.sample.sample_id,
                "reference_subject_id": reference.sample.subject_id,
                "qc_frame_index": qc_frame_index,
                "original_frame_index": original_frame_index,
                "sample_progress_percent": _progress_percent(
                    original_frame_index, sample_quality.total_frames
                ),
                "reference_original_frame_start": reference_start,
                "reference_original_frame_end": reference_end,
                "reference_progress_start_percent": _progress_percent(
                    reference_start, reference_quality.total_frames
                ),
                "reference_progress_end_percent": _progress_percent(
                    reference_end, reference_quality.total_frames
                ),
                "component_index": component_index,
                "component_name": component_name,
                "mean_angular_deviation_degrees": float(
                    error_means[qc_frame_index, component_index]
                ),
                "maximum_angular_deviation_degrees": float(
                    error_maxima[qc_frame_index, component_index]
                ),
                "aligned_path_steps": int(path_counts[qc_frame_index]),
                "sample_component_interpolated": bool(
                    sample_quality.interpolated_component_mask[
                        original_frame_index, component_index
                    ]
                ),
                "reference_component_interpolated_fraction": float(
                    reference_interpolated_fractions[
                        qc_frame_index, component_index
                    ]
                ),
                "interpretation": CANDIDATE_INTERPRETATION,
            }


def top_deviation_interval_rows(
    fold_number: int,
    prepared: YuXiongPreparedSample,
    reference: YuXiongPreparedSample,
    alignment: InterpretableDtwAlignment,
    sample_quality: FrameQualityResult,
    reference_quality: FrameQualityResult,
    window_count: int = DEVIATION_WINDOW_COUNT,
    limit: int = TOP_DEVIATION_INTERVALS_PER_SAMPLE,
) -> list[dict[str, object]]:
    """Return the largest fixed-progress QC deviations for annotation review."""
    if window_count < 1:
        raise ValueError("Deviation window count must be positive")
    if limit < 1:
        raise ValueError("Deviation interval limit must be positive")
    sample_original, reference_original, path, errors = _validate_inputs(
        prepared,
        reference,
        alignment,
        sample_quality,
        reference_quality,
    )
    aligned_sample_original = sample_original[path[:, 0]]
    aligned_reference_original = reference_original[path[:, 1]]
    sample_denominator = max(sample_quality.total_frames - 1, 1)
    progress = aligned_sample_original / sample_denominator
    window_indices = np.minimum(
        (progress * window_count).astype(int), window_count - 1
    )

    path_counts = np.bincount(window_indices, minlength=window_count)
    error_sums = np.zeros((window_count, FEATURE_DIMENSIONS), dtype=float)
    error_maxima = np.zeros_like(error_sums)
    np.add.at(error_sums, window_indices, errors)
    np.maximum.at(error_maxima, window_indices, errors)

    sample_minima = np.full(window_count, sample_quality.total_frames, dtype=np.int64)
    sample_maxima = np.full(window_count, -1, dtype=np.int64)
    reference_minima = np.full(
        window_count, reference_quality.total_frames, dtype=np.int64
    )
    reference_maxima = np.full(window_count, -1, dtype=np.int64)
    np.minimum.at(sample_minima, window_indices, aligned_sample_original)
    np.maximum.at(sample_maxima, window_indices, aligned_sample_original)
    np.minimum.at(reference_minima, window_indices, aligned_reference_original)
    np.maximum.at(reference_maxima, window_indices, aligned_reference_original)

    sample_interpolated = sample_quality.interpolated_component_mask[
        aligned_sample_original
    ].astype(float)
    reference_interpolated = reference_quality.interpolated_component_mask[
        aligned_reference_original
    ].astype(float)
    sample_interpolated_sums = np.zeros_like(error_sums)
    reference_interpolated_sums = np.zeros_like(error_sums)
    np.add.at(sample_interpolated_sums, window_indices, sample_interpolated)
    np.add.at(reference_interpolated_sums, window_indices, reference_interpolated)

    component_means = np.mean(errors, axis=0)
    candidates: list[dict[str, object]] = []
    for window_index in np.flatnonzero(path_counts):
        count = int(path_counts[window_index])
        for component_index, component_name in enumerate(FEATURE_NAMES):
            mean_error = float(error_sums[window_index, component_index] / count)
            candidates.append(
                {
                    "fold": fold_number,
                    "sample_id": prepared.sample.sample_id,
                    "subject_id": prepared.sample.subject_id,
                    "cohort": prepared.sample.cohort,
                    "actual_ts": prepared.sample.score,
                    "sample_quality_status": sample_quality.quality_status,
                    "reference_sample_id": reference.sample.sample_id,
                    "reference_subject_id": reference.sample.subject_id,
                    "window_index": int(window_index),
                    "window_start_percent": 100.0 * window_index / window_count,
                    "window_end_percent": 100.0 * (window_index + 1) / window_count,
                    "original_frame_start": int(sample_minima[window_index]),
                    "original_frame_end": int(sample_maxima[window_index]),
                    "reference_original_frame_start": int(
                        reference_minima[window_index]
                    ),
                    "reference_original_frame_end": int(
                        reference_maxima[window_index]
                    ),
                    "component_index": component_index,
                    "component_name": component_name,
                    "mean_angular_deviation_degrees": mean_error,
                    "maximum_angular_deviation_degrees": float(
                        error_maxima[window_index, component_index]
                    ),
                    "whole_alignment_component_mean_degrees": float(
                        component_means[component_index]
                    ),
                    "excess_over_component_mean_degrees": float(
                        mean_error - component_means[component_index]
                    ),
                    "aligned_path_steps": count,
                    "sample_component_interpolated_fraction": float(
                        sample_interpolated_sums[
                            window_index, component_index
                        ]
                        / count
                    ),
                    "reference_component_interpolated_fraction": float(
                        reference_interpolated_sums[
                            window_index, component_index
                        ]
                        / count
                    ),
                    "interpretation": CANDIDATE_INTERPRETATION,
                }
            )

    candidates.sort(
        key=lambda row: (
            -float(row["mean_angular_deviation_degrees"]),
            -float(row["maximum_angular_deviation_degrees"]),
            int(row["component_index"]),
            int(row["window_index"]),
        )
    )
    selected = candidates[:limit]
    for rank, row in enumerate(selected, start=1):
        row["candidate_rank"] = rank
    return selected


def annotation_queue_rows(
    interval_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Add blank human-review fields without inventing execution-error labels."""
    return [
        {
            **row,
            "review_status": "unreviewed",
            "execution_label": "",
            "error_type": "",
            "severity": "",
            "reviewer_confidence": "",
            "annotator": "",
            "review_notes": "",
        }
        for row in interval_rows
    ]
