"""Interpretable explanation layer for the Yu--Xiong angular DTW baseline.

The original Yu--Xiong module remains an unchanged comparison baseline.  This
module reuses its optimal path and score, then preserves the nine individual
body-component errors that the baseline normally combines into one cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kimore_yu_xiong_dtw import (
    FEATURE_DIMENSIONS,
    FEATURE_NAMES,
    YuXiongAlignment,
    yu_xiong_dtw,
)


@dataclass(frozen=True)
class InterpretableDtwAlignment(YuXiongAlignment):
    component_errors_degrees: np.ndarray  # shape: (path steps, 9)


@dataclass(frozen=True)
class ComponentErrorSummary:
    name: str
    total_error_degrees: float
    mean_error_degrees: float
    maximum_error_degrees: float
    contribution_percent: float


def _vector_sequence(values: np.ndarray, name: str) -> np.ndarray:
    """Validate and normalize one sequence of nine XYZ unit vectors."""
    vectors = np.asarray(values, dtype=np.float64)
    expected_suffix = (FEATURE_DIMENSIONS, 3)
    if vectors.ndim != 3 or vectors.shape[1:] != expected_suffix or len(vectors) == 0:
        raise ValueError(
            f"{name} must have shape (frames, {FEATURE_DIMENSIONS}, 3)"
        )
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} contains non-finite values")

    lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
    if np.any(lengths[:, :, 0] <= np.finfo(np.float64).eps):
        raise ValueError(f"{name} contains a zero-length vector")
    return vectors / lengths


def _path_component_errors(
    sample: np.ndarray,
    reference: np.ndarray,
    path: np.ndarray,
) -> np.ndarray:
    """Return each feature's angular error at every aligned path step."""
    aligned_sample = sample[path[:, 0]]
    aligned_reference = reference[path[:, 1]]
    dots = np.einsum("pkd,pkd->pk", aligned_sample, aligned_reference)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def interpretable_dtw(
    sample: np.ndarray,
    reference: np.ndarray,
) -> InterpretableDtwAlignment:
    """Run Yu--Xiong DTW and retain its per-component path errors."""
    sample_vectors = _vector_sequence(sample, "sample")
    reference_vectors = _vector_sequence(reference, "reference")
    # Run the baseline on the original inputs so that it performs exactly the
    # same single normalization as a direct yu_xiong_dtw call. Passing the
    # already-normalized arrays would normalize twice and can create tiny,
    # but accumulated, self-alignment costs on long recordings.
    base = yu_xiong_dtw(sample, reference)
    component_errors = _path_component_errors(
        sample_vectors,
        reference_vectors,
        base.path,
    )

    if not np.isclose(
        np.sum(component_errors),
        base.total_angular_cost_degrees,
        rtol=1e-12,
        atol=1e-9,
    ):
        raise RuntimeError("Per-component DTW errors do not sum to the total cost")

    return InterpretableDtwAlignment(
        path=base.path,
        total_angular_cost_degrees=base.total_angular_cost_degrees,
        mean_angle_degrees=base.mean_angle_degrees,
        paper_score_unclipped=base.paper_score_unclipped,
        paper_score=base.paper_score,
        component_errors_degrees=component_errors,
    )


def summarize_component_errors(
    alignment: InterpretableDtwAlignment,
) -> tuple[ComponentErrorSummary, ...]:
    """Summarize each named component's contribution to one DTW alignment."""
    errors = np.asarray(alignment.component_errors_degrees, dtype=np.float64)
    expected_shape = (len(alignment.path), FEATURE_DIMENSIONS)
    if errors.shape != expected_shape:
        raise ValueError(
            f"Component errors must have shape {expected_shape}; got {errors.shape}"
        )
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise ValueError("Component errors must be finite and non-negative")

    totals = np.sum(errors, axis=0)
    means = np.mean(errors, axis=0)
    maxima = np.max(errors, axis=0)
    overall_total = float(np.sum(totals))
    if overall_total == 0.0:
        contributions = np.zeros(FEATURE_DIMENSIONS, dtype=np.float64)
    else:
        contributions = 100.0 * totals / overall_total

    return tuple(
        ComponentErrorSummary(
            name=name,
            total_error_degrees=float(totals[index]),
            mean_error_degrees=float(means[index]),
            maximum_error_degrees=float(maxima[index]),
            contribution_percent=float(contributions[index]),
        )
        for index, name in enumerate(FEATURE_NAMES)
    )
