"""Small, exact dynamic-time-warping utilities for the KIMORE baseline.

The baseline uses squared Euclidean local costs and chooses the path with the
smallest accumulated cost.  The reported distance is the root mean squared
local cost along that path, so it is not automatically larger just because an
alignment contains more steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DtwAlignment:
    path: np.ndarray  # shape: (path steps, 2), [sample index, reference index]
    total_squared_cost: float
    aligned_rmse: float


def _feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 1D or 2D sequence")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite feature values")
    return matrix


def exact_dtw(sample: np.ndarray, reference: np.ndarray) -> DtwAlignment:
    """Align two feature sequences and return their exact, unconstrained DTW path."""
    sample_matrix = _feature_matrix(sample, "sample")
    reference_matrix = _feature_matrix(reference, "reference")
    if sample_matrix.shape[1] != reference_matrix.shape[1]:
        raise ValueError(
            "Sample and reference must have the same number of features; "
            f"got {sample_matrix.shape[1]} and {reference_matrix.shape[1]}"
        )

    sample_length = len(sample_matrix)
    reference_length = len(reference_matrix)
    accumulated = np.empty((sample_length, reference_length), dtype=np.float64)

    first_row_costs = np.sum(
        (reference_matrix - sample_matrix[0]) ** 2,
        axis=1,
    )
    accumulated[0, 0] = first_row_costs[0]
    for reference_index in range(1, reference_length):
        accumulated[0, reference_index] = (
            accumulated[0, reference_index - 1]
            + first_row_costs[reference_index]
        )

    for sample_index in range(1, sample_length):
        row_costs = np.sum(
            (reference_matrix - sample_matrix[sample_index]) ** 2,
            axis=1,
        )
        accumulated[sample_index, 0] = (
            accumulated[sample_index - 1, 0] + row_costs[0]
        )
        for reference_index in range(1, reference_length):
            predecessor = min(
                accumulated[sample_index - 1, reference_index - 1],
                accumulated[sample_index - 1, reference_index],
                accumulated[sample_index, reference_index - 1],
            )
            accumulated[sample_index, reference_index] = (
                row_costs[reference_index] + predecessor
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
            # The tuple order makes diagonal the deterministic choice on ties.
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
    differences = (
        sample_matrix[path[:, 0]]
        - reference_matrix[path[:, 1]]
    )
    total_squared_cost = float(np.sum(differences ** 2))
    aligned_rmse = float(np.sqrt(total_squared_cost / len(path)))
    return DtwAlignment(
        path=path,
        total_squared_cost=total_squared_cost,
        aligned_rmse=aligned_rmse,
    )
