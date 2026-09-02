"""Five-fold subject-wise evaluation for the interpretable plain-DTW baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
import numpy as np

from kimore_dataset import KimoreSample, read_manifest
from kimore_grouping import assert_no_subject_leakage, make_subject_folds, subject_groups
from kimore_plain_dtw import (
    FEATURE_NAME,
    MIN_REFERENCE_SHOULDER_TRACKED_FRACTION,
    SUPPORTED_EXERCISE,
    PreparedSample,
    fit_linear_calibration,
    prepare_sample,
    select_training_reference,
)
from kimore_dtw import DtwAlignment, exact_dtw


METRIC_NAMES = ("mae", "rmse", "spearman", "pearson")
BOOTSTRAP_SEED = 20260825
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _package_version(distribution_name: str) -> str:
    """Read package metadata without importing optional compiled libraries."""
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not-installed"


def _pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson's r using NumPy, or NaN when it is undefined."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(
        np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    )
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Ranks require a one-dimensional array")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return Spearman's rho as Pearson correlation of average ranks."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        return float("nan")
    return _pearson_correlation(_average_ranks(x), _average_ranks(y))


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the four fixed evaluation metrics."""
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    if (
        y_true.ndim != 1
        or y_pred.ndim != 1
        or len(y_true) != len(y_pred)
        or len(y_true) == 0
    ):
        raise ValueError("Metrics need equally sized, non-empty 1D arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metric inputs must be finite")

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        spearman = float("nan")
        pearson = float("nan")
    else:
        spearman = _spearman_correlation(y_true, y_pred)
        pearson = _pearson_correlation(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "pearson": pearson,
    }


def training_constant_values(
    scores: np.ndarray,
    train_indices: Sequence[int],
) -> tuple[float, float]:
    """Return training-only constants suited to MAE and RMSE, respectively."""
    values = np.asarray(scores, dtype=np.float64)
    indices = np.asarray(train_indices, dtype=int)
    if values.ndim != 1 or len(indices) == 0:
        raise ValueError("Constant baselines need scores and non-empty training indices")
    if np.any(indices < 0) or np.any(indices >= len(values)):
        raise IndexError("Training index is outside the score array")
    training_scores = values[indices]
    return float(np.median(training_scores)), float(np.mean(training_scores))


def training_cohort_constant_values(
    scores: np.ndarray,
    cohorts: Sequence[str],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return training-only cohort constants for each requested test row."""
    values = np.asarray(scores, dtype=np.float64)
    cohort_values = np.asarray(cohorts, dtype=object)
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    if values.ndim != 1 or len(values) != len(cohort_values):
        raise ValueError("Cohort baselines need one cohort for every score")
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Cohort baselines need non-empty training and test indices")
    if np.any(train < 0) or np.any(test < 0) or np.any(train >= len(values)) or np.any(
        test >= len(values)
    ):
        raise IndexError("Cohort baseline index is outside the score array")

    medians: list[float] = []
    means: list[float] = []
    for index in test:
        matching_train = train[cohort_values[train] == cohort_values[index]]
        if len(matching_train) == 0:
            raise ValueError(
                f"No training subject from test cohort {cohort_values[index]!r}"
            )
        cohort_scores = values[matching_train]
        medians.append(float(np.median(cohort_scores)))
        means.append(float(np.mean(cohort_scores)))
    return np.asarray(medians), np.asarray(means)


def validate_oof_indices(test_indices_by_fold: Sequence[Sequence[int]], sample_count: int) -> None:
    """Require every sample index to occur in exactly one outer test fold."""
    flattened = [int(index) for fold_indices in test_indices_by_fold for index in fold_indices]
    counts = Counter(flattened)
    expected = set(range(sample_count))
    observed = set(counts)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    repeated = sorted(index for index, count in counts.items() if count != 1)
    if missing or unexpected or repeated:
        raise AssertionError(
            "Invalid out-of-fold coverage; "
            f"missing={missing}, unexpected={unexpected}, repeated={repeated}"
        )


def _group_bootstrap_indices(
    groups: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    parts = [np.flatnonzero(groups == group) for group in sampled_groups]
    return np.concatenate(parts)


def bootstrap_metric_intervals(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: Sequence[str],
    resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float | int | None]]:
    """Return conditional fixed-prediction subject-bootstrap intervals."""
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be at least 1")
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    if len(y_true) != len(y_pred) or len(y_true) != len(group_array):
        raise ValueError("Bootstrap arrays and groups must have equal length")

    rng = np.random.RandomState(seed)
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(resamples):
        indices = _group_bootstrap_indices(group_array, rng)
        metrics = regression_metrics(y_true[indices], y_pred[indices])
        for name, value in metrics.items():
            if np.isfinite(value):
                values[name].append(value)

    intervals: dict[str, dict[str, float | int | None]] = {}
    for name, samples in values.items():
        if samples:
            low, high = np.percentile(np.asarray(samples), (2.5, 97.5))
            intervals[name] = {
                "low": float(low),
                "high": float(high),
                "valid_resamples": len(samples),
            }
        else:
            intervals[name] = {
                "low": None,
                "high": None,
                "valid_resamples": 0,
            }
    return intervals


def bootstrap_improvement_intervals(
    actual: np.ndarray,
    model_predictions: np.ndarray,
    median_predictions: np.ndarray,
    mean_predictions: np.ndarray,
    groups: Sequence[str],
    resamples: int,
    seed: int = BOOTSTRAP_SEED + 1,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap fixed-prediction error reduction versus constant baselines."""
    y_true = np.asarray(actual, dtype=np.float64)
    model = np.asarray(model_predictions, dtype=np.float64)
    median_baseline = np.asarray(median_predictions, dtype=np.float64)
    mean_baseline = np.asarray(mean_predictions, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    arrays = (model, median_baseline, mean_baseline, group_array)
    if any(len(values) != len(y_true) for values in arrays):
        raise ValueError("Improvement bootstrap arrays and groups must have equal length")

    rng = np.random.RandomState(seed)
    mae_improvements: list[float] = []
    rmse_improvements: list[float] = []
    for _ in range(resamples):
        indices = _group_bootstrap_indices(group_array, rng)
        actual_sample = y_true[indices]
        model_errors = actual_sample - model[indices]
        median_errors = actual_sample - median_baseline[indices]
        mean_errors = actual_sample - mean_baseline[indices]
        mae_improvements.append(
            float(np.mean(np.abs(median_errors)) - np.mean(np.abs(model_errors)))
        )
        rmse_improvements.append(
            float(
                np.sqrt(np.mean(mean_errors ** 2))
                - np.sqrt(np.mean(model_errors ** 2))
            )
        )

    model_errors = y_true - model
    median_errors = y_true - median_baseline
    mean_errors = y_true - mean_baseline

    return {
        "mae_reduction_vs_training_median": {
            "estimate": float(
                np.mean(np.abs(median_errors)) - np.mean(np.abs(model_errors))
            ),
            "low": float(np.percentile(mae_improvements, 2.5)),
            "high": float(np.percentile(mae_improvements, 97.5)),
            "valid_resamples": resamples,
        },
        "rmse_reduction_vs_training_mean": {
            "estimate": float(
                np.sqrt(np.mean(mean_errors ** 2))
                - np.sqrt(np.mean(model_errors ** 2))
            ),
            "low": float(np.percentile(rmse_improvements, 2.5)),
            "high": float(np.percentile(rmse_improvements, 97.5)),
            "valid_resamples": resamples,
        },
    }


def bootstrap_paired_metric_improvements(
    actual: np.ndarray,
    candidate_predictions: np.ndarray,
    comparator_predictions: np.ndarray,
    groups: Sequence[str],
    resamples: int,
    seed: int = BOOTSTRAP_SEED + 2,
) -> dict[str, dict[str, float | int | None]]:
    """Bootstrap paired metric gains for a candidate over a comparator.

    Positive values always favor the candidate: error metrics are reported as
    comparator minus candidate, while correlations are candidate minus
    comparator.
    """
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be at least 1")

    y_true = np.asarray(actual, dtype=np.float64)
    candidate = np.asarray(candidate_predictions, dtype=np.float64)
    comparator = np.asarray(comparator_predictions, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    arrays = (candidate, comparator, group_array)
    if any(len(values) != len(y_true) for values in arrays):
        raise ValueError("Paired bootstrap arrays and groups must have equal length")
    if len(y_true) == 0:
        raise ValueError("Paired bootstrap needs at least one prediction row")

    def improvements(indices: np.ndarray) -> dict[str, float]:
        candidate_metrics = regression_metrics(y_true[indices], candidate[indices])
        comparator_metrics = regression_metrics(y_true[indices], comparator[indices])
        return {
            "mae_reduction": (
                comparator_metrics["mae"] - candidate_metrics["mae"]
            ),
            "rmse_reduction": (
                comparator_metrics["rmse"] - candidate_metrics["rmse"]
            ),
            "spearman_increase": (
                candidate_metrics["spearman"] - comparator_metrics["spearman"]
            ),
            "pearson_increase": (
                candidate_metrics["pearson"] - comparator_metrics["pearson"]
            ),
        }

    rng = np.random.RandomState(seed)
    values: dict[str, list[float]] = {
        "mae_reduction": [],
        "rmse_reduction": [],
        "spearman_increase": [],
        "pearson_increase": [],
    }
    for _ in range(resamples):
        indices = _group_bootstrap_indices(group_array, rng)
        for name, value in improvements(indices).items():
            if np.isfinite(value):
                values[name].append(float(value))

    estimates = improvements(np.arange(len(y_true), dtype=int))
    intervals: dict[str, dict[str, float | int | None]] = {}
    for name, samples in values.items():
        if samples:
            low, high = np.percentile(np.asarray(samples), (2.5, 97.5))
            intervals[name] = {
                "estimate": float(estimates[name]),
                "low": float(low),
                "high": float(high),
                "valid_resamples": len(samples),
            }
        else:
            estimate = estimates[name]
            intervals[name] = {
                "estimate": float(estimate) if np.isfinite(estimate) else None,
                "low": None,
                "high": None,
                "valid_resamples": 0,
            }
    return intervals


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if isinstance(value, float) and not np.isfinite(value) else value
                    for key, value in row.items()
                }
            )


def save_prediction_plot(
    rows: list[dict[str, object]],
    metrics: dict[str, float],
    intervals: dict[str, dict[str, float | int | None]],
    output_path: Path,
    title: str = "Plain DTW: five-fold subject-wise out-of-fold predictions",
) -> None:
    import matplotlib.pyplot as plt

    actual = np.asarray([row["actual_ts"] for row in rows], dtype=float)
    predicted = np.asarray([row["predicted_ts"] for row in rows], dtype=float)
    residuals = predicted - actual
    cohorts = np.asarray([row["cohort"] for row in rows], dtype=object)
    cohort_names = sorted(set(cohorts))
    colors = plt.get_cmap("tab10")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    lower = float(min(actual.min(), predicted.min()) - 2.0)
    upper = float(max(actual.max(), predicted.max()) + 2.0)
    for color_index, cohort in enumerate(cohort_names):
        mask = cohorts == cohort
        label = cohort.replace("_", " ")
        axes[0].scatter(
            actual[mask],
            predicted[mask],
            s=38,
            alpha=0.8,
            color=colors(color_index),
            label=label,
        )
        axes[1].scatter(
            actual[mask],
            residuals[mask],
            s=38,
            alpha=0.8,
            color=colors(color_index),
        )

    axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Actual clinical TS")
    axes[0].set_ylabel("Out-of-fold predicted TS")
    axes[0].set_title("Actual versus predicted")
    axes[0].grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()

    mae_interval = intervals["mae"]
    spearman_interval = intervals["spearman"]
    annotation = (
        f"MAE {metrics['mae']:.2f} "
        f"[{mae_interval['low']:.2f}, {mae_interval['high']:.2f}]\n"
        f"Spearman {metrics['spearman']:.2f} "
        f"[{spearman_interval['low']:.2f}, {spearman_interval['high']:.2f}]"
    )
    axes[0].text(
        0.03,
        0.97,
        annotation,
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        fontsize=9,
    )

    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Actual clinical TS")
    axes[1].set_ylabel("Prediction error (predicted - actual)")
    axes[1].set_title("Residuals")
    axes[1].grid(alpha=0.2)

    figure.suptitle(title, fontsize=14)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(cohort_names),
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _metrics_row(
    scope: str,
    model: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    include_correlations: bool,
) -> dict[str, object]:
    metrics = regression_metrics(actual, predicted)
    if not include_correlations:
        metrics["spearman"] = float("nan")
        metrics["pearson"] = float("nan")
    return {
        "scope": scope,
        "model": model,
        "subjects": len(actual),
        **metrics,
    }


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_inputs_sha256(samples: Sequence[KimoreSample]) -> str:
    """Hash the target metadata and raw position bytes used by the experiment."""
    digest = hashlib.sha256()
    for sample in samples:
        position_digest = hashlib.sha256()
        with sample.position_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                position_digest.update(chunk)
        metadata = {
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "cohort": sample.cohort,
            "exercise": sample.exercise,
            "score": format(sample.score, ".17g"),
            "frames": sample.frames,
            "position_sha256": position_digest.hexdigest(),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _source_code_sha256() -> str:
    """Hash the executable project source even when the Git tree is dirty."""
    paths = [PROJECT_ROOT / "run_experiment.py", PROJECT_ROOT / "requirements.txt"]
    paths.extend(sorted((PROJECT_ROOT / "src").glob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(
        paths,
        key=lambda item: item.relative_to(PROJECT_ROOT).as_posix(),
    ):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative}\0{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("+dirty" if dirty else "")


def alignment_warp_diagnostics(
    sample_frames: int,
    reference_frames: int,
    path_length: int,
) -> dict[str, float]:
    """Summarize how much an exact DTW path departs from diagonal matching."""
    if sample_frames < 1 or reference_frames < 1 or path_length < 1:
        raise ValueError("Frame counts and path length must be positive")
    minimum_path = max(sample_frames, reference_frames)
    maximum_path = sample_frames + reference_frames - 1
    if not minimum_path <= path_length <= maximum_path:
        raise ValueError("DTW path length is inconsistent with the sequence lengths")

    moves = path_length - 1
    non_diagonal_moves = 2 * path_length - sample_frames - reference_frames
    longer_sequence_moves = max(sample_frames, reference_frames) - 1
    return {
        "non_diagonal_step_fraction": (
            float(non_diagonal_moves / moves) if moves else 0.0
        ),
        "minimum_required_non_diagonal_step_fraction": (
            float(abs(sample_frames - reference_frames) / longer_sequence_moves)
            if longer_sequence_moves
            else 0.0
        ),
        "path_length_over_longer_sequence": float(path_length / minimum_path),
    }


def post_hoc_cohort_diagnostics(
    actual: np.ndarray,
    predicted: np.ndarray,
    cohorts: Sequence[str],
) -> dict[str, object]:
    """Describe pooled cohort structure without changing the frozen model."""
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    cohort_values = np.asarray(cohorts, dtype=object)
    if len(y_true) != len(y_pred) or len(y_true) != len(cohort_values):
        raise ValueError("Cohort diagnostics need equally sized arrays")

    centered_actual = np.empty_like(y_true)
    centered_predicted = np.empty_like(y_pred)
    per_cohort: dict[str, object] = {}
    for cohort in sorted(set(cohort_values)):
        mask = cohort_values == cohort
        cohort_actual = y_true[mask]
        cohort_predicted = y_pred[mask]
        metrics = regression_metrics(cohort_actual, cohort_predicted)
        per_cohort[str(cohort)] = {
            "subjects": int(mask.sum()),
            "mean_bias_predicted_minus_actual": float(
                np.mean(cohort_predicted - cohort_actual)
            ),
            "mae": metrics["mae"],
            "spearman": metrics["spearman"],
            "pearson": metrics["pearson"],
        }
        centered_actual[mask] = cohort_actual - np.mean(cohort_actual)
        centered_predicted[mask] = cohort_predicted - np.mean(cohort_predicted)

    centered_metrics = regression_metrics(centered_actual, centered_predicted)
    return {
        "status": "post_hoc_descriptive_diagnostic",
        "per_cohort": per_cohort,
        "within_cohort_mean_centered_spearman": centered_metrics["spearman"],
        "within_cohort_mean_centered_pearson": centered_metrics["pearson"],
    }


def run_cross_validated_evaluation(
    manifest_path: Path,
    exercise: str,
    output_dir: Path,
    figure_dir: Path,
    bootstrap_resamples: int = 5000,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run the frozen five-fold baseline and save complete OOF evaluation artifacts."""
    if exercise.casefold() != SUPPORTED_EXERCISE.casefold():
        raise ValueError(
            f"The shoulder-yaw plain-DTW method is defined only for "
            f"{SUPPORTED_EXERCISE}, not {exercise}"
        )
    if bootstrap_resamples < 1:
        raise ValueError("Bootstrap resamples must be at least 1")
    samples, excluded = read_manifest(manifest_path, exercise)
    folds = make_subject_folds(samples, n_splits=5)
    groups = subject_groups(samples)
    validate_oof_indices([fold.test_indices for fold in folds], len(samples))

    progress(f"Loading and preprocessing {len(samples)} usable {exercise} recordings...")
    prepared_samples: list[PreparedSample] = []
    for number, sample in enumerate(samples, start=1):
        prepared_samples.append(prepare_sample(sample))
        if number % 10 == 0 or number == len(samples):
            progress(f"  prepared {number}/{len(samples)}")

    scores = np.asarray([sample.score for sample in samples], dtype=np.float64)
    cohorts = np.asarray([sample.cohort for sample in samples], dtype=object)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fold_metadata: list[dict[str, object]] = []
    alignment_cache: dict[int, list[DtwAlignment]] = {}

    for fold in folds:
        assert_no_subject_leakage(groups, fold.train_indices, fold.test_indices)
        reference_index = select_training_reference(prepared_samples, fold.train_indices)
        reference = prepared_samples[reference_index]
        progress(
            f"Fold {fold.number}: reference {reference.sample.sample_id} "
            f"(TS={reference.sample.score:.2f}, shoulders tracked="
            f"{reference.shoulder_tracked_fraction:.2%})"
        )

        if reference_index not in alignment_cache:
            progress("  calculating exact DTW alignments for this reference...")
            reference_alignments: list[DtwAlignment] = []
            for number, prepared in enumerate(prepared_samples, start=1):
                reference_alignments.append(exact_dtw(prepared.feature, reference.feature))
                if number % 10 == 0 or number == len(prepared_samples):
                    progress(f"    aligned {number}/{len(prepared_samples)}")
            alignment_cache[reference_index] = reference_alignments
        else:
            progress("  reusing exact DTW alignments for the same training reference")

        alignments = alignment_cache[reference_index]
        distances = np.asarray(
            [alignment.aligned_rmse for alignment in alignments],
            dtype=np.float64,
        )
        calibration = fit_linear_calibration(
            distances[fold.train_indices],
            scores[fold.train_indices],
        )
        fold_predictions = calibration.predict(distances[fold.test_indices])
        median_value, mean_value = training_constant_values(scores, fold.train_indices)
        median_predictions = np.full(len(fold.test_indices), median_value)
        mean_predictions = np.full(len(fold.test_indices), mean_value)
        cohort_median_predictions, cohort_mean_predictions = (
            training_cohort_constant_values(
                scores,
                cohorts,
                fold.train_indices,
                fold.test_indices,
            )
        )
        fold_actual = scores[fold.test_indices]

        metric_rows.extend(
            (
                _metrics_row(
                    f"fold_{fold.number}",
                    "plain_dtw",
                    fold_actual,
                    fold_predictions,
                    include_correlations=True,
                ),
                _metrics_row(
                    f"fold_{fold.number}",
                    "training_median_constant",
                    fold_actual,
                    median_predictions,
                    include_correlations=False,
                ),
                _metrics_row(
                    f"fold_{fold.number}",
                    "training_mean_constant",
                    fold_actual,
                    mean_predictions,
                    include_correlations=False,
                ),
                _metrics_row(
                    f"fold_{fold.number}",
                    "post_hoc_training_cohort_median_constant",
                    fold_actual,
                    cohort_median_predictions,
                    include_correlations=False,
                ),
                _metrics_row(
                    f"fold_{fold.number}",
                    "post_hoc_training_cohort_mean_constant",
                    fold_actual,
                    cohort_mean_predictions,
                    include_correlations=False,
                ),
            )
        )

        train_subjects = set(groups[fold.train_indices])
        test_subjects = set(groups[fold.test_indices])
        fold_metadata.append(
            {
                "fold": fold.number,
                "training_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "subject_overlap": len(train_subjects.intersection(test_subjects)),
                "test_cohort_counts": dict(
                    Counter(samples[int(index)].cohort for index in fold.test_indices)
                ),
                "reference_sample_id": reference.sample.sample_id,
                "reference_subject_id": reference.sample.subject_id,
                "reference_actual_ts": reference.sample.score,
                "reference_shoulder_tracked_fraction": reference.shoulder_tracked_fraction,
                "reference_threshold_candidate_count": sum(
                    prepared_samples[int(index)].shoulder_tracked_fraction
                    >= MIN_REFERENCE_SHOULDER_TRACKED_FRACTION
                    for index in fold.train_indices
                ),
                "calibration_intercept": calibration.intercept,
                "calibration_slope": calibration.slope,
                "training_median_ts": median_value,
                "training_mean_ts": mean_value,
                "test_metrics": regression_metrics(fold_actual, fold_predictions),
            }
        )

        for position, sample_index_value in enumerate(fold.test_indices):
            sample_index = int(sample_index_value)
            prepared = prepared_samples[sample_index]
            alignment = alignments[sample_index]
            warp = alignment_warp_diagnostics(
                len(prepared.feature),
                len(reference.feature),
                len(alignment.path),
            )
            sample_initial_yaw = float(np.median(prepared.feature[:30, 0]))
            reference_initial_yaw = float(np.median(reference.feature[:30, 0]))
            prediction_rows.append(
                {
                    "fold": fold.number,
                    "sample_id": prepared.sample.sample_id,
                    "subject_id": prepared.sample.subject_id,
                    "cohort": prepared.sample.cohort,
                    "actual_ts": prepared.sample.score,
                    "predicted_ts": float(fold_predictions[position]),
                    "training_median_baseline_ts": median_value,
                    "training_mean_baseline_ts": mean_value,
                    "post_hoc_training_cohort_median_baseline_ts": float(
                        cohort_median_predictions[position]
                    ),
                    "post_hoc_training_cohort_mean_baseline_ts": float(
                        cohort_mean_predictions[position]
                    ),
                    "dtw_aligned_rmse_degrees": alignment.aligned_rmse,
                    "frames": len(prepared.feature),
                    "reference_frames": len(reference.feature),
                    "alignment_path_length": len(alignment.path),
                    "alignment_non_diagonal_step_fraction": warp[
                        "non_diagonal_step_fraction"
                    ],
                    "minimum_required_non_diagonal_step_fraction": warp[
                        "minimum_required_non_diagonal_step_fraction"
                    ],
                    "alignment_path_length_over_longer_sequence": warp[
                        "path_length_over_longer_sequence"
                    ],
                    "initial_30_frame_yaw_median_degrees": sample_initial_yaw,
                    "reference_initial_30_frame_yaw_median_degrees": reference_initial_yaw,
                    "initial_window_absolute_yaw_offset_degrees": abs(
                        sample_initial_yaw - reference_initial_yaw
                    ),
                    "feature": FEATURE_NAME,
                    "reference_sample_id": reference.sample.sample_id,
                    "reference_subject_id": reference.sample.subject_id,
                    "reference_actual_ts": reference.sample.score,
                    "reference_shoulder_tracked_fraction": reference.shoulder_tracked_fraction,
                    "calibration_intercept": calibration.intercept,
                    "calibration_slope": calibration.slope,
                }
            )

    sample_order = {sample.sample_id: index for index, sample in enumerate(samples)}
    prediction_rows.sort(key=lambda row: sample_order[str(row["sample_id"])])
    if len(prediction_rows) != len(samples):
        raise AssertionError(
            f"Expected {len(samples)} OOF rows but created {len(prediction_rows)}"
        )
    if len({row["sample_id"] for row in prediction_rows}) != len(samples):
        raise AssertionError("Each sample must have exactly one OOF prediction row")

    actual = np.asarray([row["actual_ts"] for row in prediction_rows], dtype=float)
    predicted = np.asarray([row["predicted_ts"] for row in prediction_rows], dtype=float)
    median_baseline = np.asarray(
        [row["training_median_baseline_ts"] for row in prediction_rows],
        dtype=float,
    )
    mean_baseline = np.asarray(
        [row["training_mean_baseline_ts"] for row in prediction_rows],
        dtype=float,
    )
    cohort_median_baseline = np.asarray(
        [
            row["post_hoc_training_cohort_median_baseline_ts"]
            for row in prediction_rows
        ],
        dtype=float,
    )
    cohort_mean_baseline = np.asarray(
        [
            row["post_hoc_training_cohort_mean_baseline_ts"]
            for row in prediction_rows
        ],
        dtype=float,
    )
    oof_groups = [str(row["subject_id"]) for row in prediction_rows]
    oof_cohorts = [str(row["cohort"]) for row in prediction_rows]
    overall_metrics = regression_metrics(actual, predicted)
    median_metrics = regression_metrics(actual, median_baseline)
    mean_metrics = regression_metrics(actual, mean_baseline)
    cohort_median_metrics = regression_metrics(actual, cohort_median_baseline)
    cohort_mean_metrics = regression_metrics(actual, cohort_mean_baseline)
    metric_rows.extend(
        (
            _metrics_row("overall", "plain_dtw", actual, predicted, True),
            _metrics_row(
                "overall",
                "training_median_constant",
                actual,
                median_baseline,
                False,
            ),
            _metrics_row(
                "overall",
                "training_mean_constant",
                actual,
                mean_baseline,
                False,
            ),
            _metrics_row(
                "overall",
                "post_hoc_training_cohort_median_constant",
                actual,
                cohort_median_baseline,
                False,
            ),
            _metrics_row(
                "overall",
                "post_hoc_training_cohort_mean_constant",
                actual,
                cohort_mean_baseline,
                False,
            ),
        )
    )

    progress(
        f"Bootstrapping {bootstrap_resamples} fixed-OOF subject-level resamples..."
    )
    intervals = bootstrap_metric_intervals(
        actual,
        predicted,
        oof_groups,
        bootstrap_resamples,
    )
    improvements = bootstrap_improvement_intervals(
        actual,
        predicted,
        median_baseline,
        mean_baseline,
        oof_groups,
        bootstrap_resamples,
    )
    cohort_improvements_raw = bootstrap_improvement_intervals(
        actual,
        predicted,
        cohort_median_baseline,
        cohort_mean_baseline,
        oof_groups,
        bootstrap_resamples,
    )
    cohort_improvements = {
        "mae_reduction_vs_training_cohort_median": cohort_improvements_raw[
            "mae_reduction_vs_training_median"
        ],
        "rmse_reduction_vs_training_cohort_mean": cohort_improvements_raw[
            "rmse_reduction_vs_training_mean"
        ],
    }

    non_diagonal_fractions = np.asarray(
        [row["alignment_non_diagonal_step_fraction"] for row in prediction_rows],
        dtype=float,
    )
    minimum_non_diagonal_fractions = np.asarray(
        [
            row["minimum_required_non_diagonal_step_fraction"]
            for row in prediction_rows
        ],
        dtype=float,
    )
    path_length_ratios = np.asarray(
        [
            row["alignment_path_length_over_longer_sequence"]
            for row in prediction_rows
        ],
        dtype=float,
    )
    initial_yaw = np.asarray(
        [row["initial_30_frame_yaw_median_degrees"] for row in prediction_rows],
        dtype=float,
    )
    initial_offsets = np.asarray(
        [row["initial_window_absolute_yaw_offset_degrees"] for row in prediction_rows],
        dtype=float,
    )
    distances = np.asarray(
        [row["dtw_aligned_rmse_degrees"] for row in prediction_rows],
        dtype=float,
    )
    absolute_errors = np.abs(predicted - actual)

    predictions_path = output_dir / "oof_predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    fold_metadata_path = output_dir / "fold_metadata.json"
    summary_path = output_dir / "evaluation_summary.json"
    plot_path = figure_dir / "actual_vs_predicted.png"
    _write_csv(prediction_rows, predictions_path)
    _write_csv(metric_rows, metrics_path)
    _write_json({"folds": fold_metadata}, fold_metadata_path)
    progress("Hashing the exact Es3 targets and JointPosition inputs...")
    experiment_inputs_sha256 = _experiment_inputs_sha256(samples)

    summary: dict[str, object] = {
        "method": "plain_dtw",
        "exercise": exercise,
        "feature": FEATURE_NAME,
        "samples": len(samples),
        "unique_subjects": len(set(groups)),
        "folds": len(folds),
        "excluded_samples": excluded,
        "subject_overlap_in_every_fold": 0,
        "oof_prediction_rows": len(prediction_rows),
        "reference_tracking_threshold": MIN_REFERENCE_SHOULDER_TRACKED_FRACTION,
        "reference_rule": (
            "training recordings with at least 99% fully tracked shoulder frames, "
            "then highest clinical TS; if none qualify, best tracked training "
            "recording(s), then highest TS; final tie by sample ID"
        ),
        "distance": "aligned RMSE in shoulder-axis yaw degrees along the exact DTW path",
        "calibration": "ordinary least-squares line fitted separately in each outer training fold",
        "overall_plain_dtw": overall_metrics,
        "overall_training_median_constant": {
            "mae": median_metrics["mae"],
            "rmse": median_metrics["rmse"],
        },
        "overall_training_mean_constant": {
            "mae": mean_metrics["mae"],
            "rmse": mean_metrics["rmse"],
        },
        "post_hoc_training_cohort_constants": {
            "status": (
                "post hoc comparator added after inspecting the frozen OOF result; "
                "constants still use outer-training rows only"
            ),
            "cohort_median": {
                "mae": cohort_median_metrics["mae"],
                "rmse": cohort_median_metrics["rmse"],
            },
            "cohort_mean": {
                "mae": cohort_mean_metrics["mae"],
                "rmse": cohort_mean_metrics["rmse"],
            },
            "paired_fixed_prediction_bootstrap_improvements": cohort_improvements,
        },
        "post_hoc_cohort_diagnostics": post_hoc_cohort_diagnostics(
            actual, predicted, oof_cohorts
        ),
        "post_hoc_dtw_warp_diagnostics": {
            "status": "post_hoc_descriptive_diagnostic",
            "median_non_diagonal_step_fraction": float(
                np.median(non_diagonal_fractions)
            ),
            "maximum_non_diagonal_step_fraction": float(
                np.max(non_diagonal_fractions)
            ),
            "median_minimum_required_non_diagonal_step_fraction": float(
                np.median(minimum_non_diagonal_fractions)
            ),
            "median_excess_non_diagonal_step_fraction": float(
                np.median(
                    non_diagonal_fractions - minimum_non_diagonal_fractions
                )
            ),
            "median_path_length_over_longer_sequence": float(
                np.median(path_length_ratios)
            ),
            "minimum_path_length_over_longer_sequence": float(
                np.min(path_length_ratios)
            ),
            "maximum_path_length_over_longer_sequence": float(
                np.max(path_length_ratios)
            ),
            "interpretation": (
                "Unconstrained paths use many horizontal/vertical steps and may "
                "hide missing repetitions, extra repetitions, idle periods or "
                "irregular timing. This is a limitation, not an implementation error."
            ),
        },
        "post_hoc_initial_window_diagnostic": {
            "status": "post_hoc_descriptive_diagnostic",
            "window_frames": 30,
            "sample_yaw_median_min_degrees": float(np.min(initial_yaw)),
            "sample_yaw_median_max_degrees": float(np.max(initial_yaw)),
            "sample_reference_absolute_offset_median_degrees": float(
                np.median(initial_offsets)
            ),
            "sample_reference_absolute_offset_95th_percentile_degrees": float(
                np.percentile(initial_offsets, 95.0)
            ),
            "sample_reference_absolute_offset_max_degrees": float(
                np.max(initial_offsets)
            ),
            "offset_spearman_with_dtw_distance": float(
                _spearman_correlation(initial_offsets, distances)
            ),
            "offset_spearman_with_absolute_prediction_error": float(
                _spearman_correlation(initial_offsets, absolute_errors)
            ),
            "interpretation": (
                "The untrimmed absolute yaw feature may mix movement phase, resting "
                "orientation and acquisition setup; this diagnostic cannot identify "
                "which cause dominates."
            ),
        },
        "plain_dtw_bootstrap_95_percent_intervals": intervals,
        "paired_bootstrap_improvements": improvements,
        "bootstrap_scope": (
            "conditional fixed-OOF-prediction subject bootstrap; folds, reference "
            "selection, calibration and method choices are not refitted"
        ),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "manifest_path": _portable_path(manifest_path),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "manifest_hash_note": (
            "The manifest includes local absolute paths and is not portable; use "
            "experiment_inputs_sha256 to compare the actual targets and position data."
        ),
        "experiment_inputs_sha256": experiment_inputs_sha256,
        "source_git_revision": _git_revision(),
        "source_code_sha256": _source_code_sha256(),
        "outputs": {
            "predictions": _portable_path(predictions_path),
            "metrics": _portable_path(metrics_path),
            "fold_metadata": _portable_path(fold_metadata_path),
            "diagnostic_plot": _portable_path(plot_path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "matplotlib": matplotlib.__version__,
            "scikit_learn": _package_version("scikit-learn"),
        },
        "interpretation": (
            "Subject-disjoint internal development cross-validation on KIMORE, "
            "with fold-specific fitting restricted to outer-training rows. Fold 1 "
            "informed pipeline inspection and the feature, exercise and preprocessing "
            "were developed on this dataset; these predictions are neither untouched "
            "confirmatory evaluation nor external validation."
        ),
    }
    _write_json(summary, summary_path)
    save_prediction_plot(prediction_rows, overall_metrics, intervals, plot_path)

    progress(
        "Overall plain-DTW metrics: "
        f"MAE={overall_metrics['mae']:.3f}, "
        f"RMSE={overall_metrics['rmse']:.3f}, "
        f"Spearman={overall_metrics['spearman']:.3f}, "
        f"Pearson={overall_metrics['pearson']:.3f}"
    )
    progress(
        "Constant baselines: "
        f"training-median MAE={median_metrics['mae']:.3f}, "
        f"training-mean RMSE={mean_metrics['rmse']:.3f}"
    )
    progress(f"Predictions: {predictions_path.resolve()}")
    progress(f"Metrics: {metrics_path.resolve()}")
    progress(f"Summary: {summary_path.resolve()}")
    progress(f"Plot: {plot_path.resolve()}")
    return summary
