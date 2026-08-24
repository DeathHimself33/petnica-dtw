"""Five-fold subject-wise evaluation for the interpretable plain-DTW baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr

from kimore_dataset import read_manifest
from kimore_grouping import assert_no_subject_leakage, make_subject_folds, subject_groups
from kimore_plain_dtw import (
    FEATURE_NAME,
    MIN_REFERENCE_SHOULDER_TRACKED_FRACTION,
    PreparedSample,
    fit_linear_calibration,
    prepare_sample,
    select_training_reference,
)
from kimore_dtw import DtwAlignment, exact_dtw


METRIC_NAMES = ("mae", "rmse", "spearman", "pearson")
BOOTSTRAP_SEED = 20260825


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
        spearman = float(spearmanr(y_true, y_pred).correlation)
        pearson = float(pearsonr(y_true, y_pred)[0])
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
    """Return paired subject-bootstrap percentile intervals for all metrics."""
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
    """Bootstrap error reduction versus the appropriate constant baseline."""
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

    figure.suptitle(
        "Plain DTW: five-fold subject-wise out-of-fold predictions",
        fontsize=14,
    )
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


def run_cross_validated_evaluation(
    manifest_path: Path,
    exercise: str,
    output_dir: Path,
    figure_dir: Path,
    bootstrap_resamples: int = 5000,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run the frozen five-fold baseline and save complete OOF evaluation artifacts."""
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
                    "dtw_aligned_rmse_degrees": alignment.aligned_rmse,
                    "frames": len(prepared.feature),
                    "alignment_path_length": len(alignment.path),
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
    oof_groups = [str(row["subject_id"]) for row in prediction_rows]
    overall_metrics = regression_metrics(actual, predicted)
    median_metrics = regression_metrics(actual, median_baseline)
    mean_metrics = regression_metrics(actual, mean_baseline)
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
        )
    )

    progress(f"Bootstrapping {bootstrap_resamples} subject-level resamples...")
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

    predictions_path = output_dir / "oof_predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    fold_metadata_path = output_dir / "fold_metadata.json"
    summary_path = output_dir / "evaluation_summary.json"
    plot_path = figure_dir / "actual_vs_predicted.png"
    _write_csv(prediction_rows, predictions_path)
    _write_csv(metric_rows, metrics_path)
    _write_json({"folds": fold_metadata}, fold_metadata_path)

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
        "plain_dtw_bootstrap_95_percent_intervals": intervals,
        "paired_bootstrap_improvements": improvements,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "outputs": {
            "predictions": str(predictions_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "fold_metadata": str(fold_metadata_path.resolve()),
            "diagnostic_plot": str(plot_path.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "interpretation": (
            "Internal development cross-validation on KIMORE. Fold 1 informed "
            "pipeline inspection before the rule was frozen; this is not external validation."
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
