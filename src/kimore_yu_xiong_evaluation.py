"""Subject-disjoint KIMORE evaluation of the Yu--Xiong DTW baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import scipy
import sklearn

from kimore_dataset import read_manifest
from kimore_evaluation import (
    BOOTSTRAP_SEED,
    _experiment_inputs_sha256,
    _git_revision,
    _manifest_sha256,
    _metrics_row,
    _portable_path,
    _source_code_sha256,
    _write_csv,
    _write_json,
    alignment_warp_diagnostics,
    bootstrap_improvement_intervals,
    bootstrap_metric_intervals,
    post_hoc_cohort_diagnostics,
    regression_metrics,
    save_prediction_plot,
    training_cohort_constant_values,
    training_constant_values,
    validate_oof_indices,
)
from kimore_grouping import assert_no_subject_leakage, make_subject_folds, subject_groups
from kimore_plain_dtw import fit_linear_calibration
from kimore_yu_xiong_dtw import (
    FEATURE_DIMENSIONS,
    FEATURE_NAME,
    YuXiongAlignment,
    YuXiongPreparedSample,
    prepare_yu_xiong_sample,
    select_yu_xiong_reference,
    yu_xiong_dtw,
)


METHOD_NAME = "yu_xiong_dtw"
PAPER_DOI = "10.3390/s19132882"


def run_yu_xiong_evaluation(
    manifest_path: Path,
    exercise: str,
    output_dir: Path,
    figure_dir: Path,
    bootstrap_resamples: int = 5000,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run five-fold Yu--Xiong DTW with a training-only KIMORE coach."""
    if bootstrap_resamples < 1:
        raise ValueError("Bootstrap resamples must be at least 1")

    samples, excluded = read_manifest(manifest_path, exercise)
    folds = make_subject_folds(samples, n_splits=5)
    groups = subject_groups(samples)
    validate_oof_indices([fold.test_indices for fold in folds], len(samples))

    progress(
        f"Loading and extracting Yu--Xiong vectors for {len(samples)} usable "
        f"{exercise} recordings..."
    )
    prepared_samples: list[YuXiongPreparedSample] = []
    for number, sample in enumerate(samples, start=1):
        prepared_samples.append(prepare_yu_xiong_sample(sample))
        if number % 10 == 0 or number == len(samples):
            progress(f"  prepared {number}/{len(samples)}")

    scores = np.asarray([sample.score for sample in samples], dtype=np.float64)
    cohorts = np.asarray([sample.cohort for sample in samples], dtype=object)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fold_metadata: list[dict[str, object]] = []
    alignment_cache: dict[int, list[YuXiongAlignment]] = {}

    for fold in folds:
        assert_no_subject_leakage(groups, fold.train_indices, fold.test_indices)
        reference_index = select_yu_xiong_reference(
            prepared_samples, fold.train_indices
        )
        reference = prepared_samples[reference_index]
        progress(
            f"Fold {fold.number}: coach {reference.sample.sample_id} "
            f"(TS={reference.sample.score:.2f}, required joints tracked="
            f"{reference.required_joints_tracked_fraction:.2%})"
        )

        if reference_index not in alignment_cache:
            progress("  calculating exact angular-DTW alignments for this coach...")
            reference_alignments: list[YuXiongAlignment] = []
            for number, prepared in enumerate(prepared_samples, start=1):
                reference_alignments.append(
                    yu_xiong_dtw(prepared.vectors, reference.vectors)
                )
                if number % 10 == 0 or number == len(prepared_samples):
                    progress(f"    aligned {number}/{len(prepared_samples)}")
            alignment_cache[reference_index] = reference_alignments
        else:
            progress("  reusing angular-DTW alignments for the same coach")

        alignments = alignment_cache[reference_index]
        paper_scores = np.asarray(
            [alignment.paper_score for alignment in alignments], dtype=np.float64
        )
        # Yu and Xiong calibrate their percentage score against expert ratings.
        # Here the same linear step is fitted only on the current outer training
        # fold, with KIMORE clinical Total Score as the target.
        calibration = fit_linear_calibration(
            paper_scores[fold.train_indices], scores[fold.train_indices]
        )
        fold_predictions = calibration.predict(paper_scores[fold.test_indices])
        median_value, mean_value = training_constant_values(scores, fold.train_indices)
        median_predictions = np.full(len(fold.test_indices), median_value)
        mean_predictions = np.full(len(fold.test_indices), mean_value)
        cohort_median_predictions, cohort_mean_predictions = (
            training_cohort_constant_values(
                scores, cohorts, fold.train_indices, fold.test_indices
            )
        )
        fold_actual = scores[fold.test_indices]

        metric_rows.extend(
            (
                _metrics_row(
                    f"fold_{fold.number}",
                    METHOD_NAME,
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
                    "training_cohort_median_constant",
                    fold_actual,
                    cohort_median_predictions,
                    include_correlations=False,
                ),
                _metrics_row(
                    f"fold_{fold.number}",
                    "training_cohort_mean_constant",
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
                "reference_required_joints_tracked_fraction": (
                    reference.required_joints_tracked_fraction
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
                len(prepared.vectors),
                len(reference.vectors),
                len(alignment.path),
            )
            prediction_rows.append(
                {
                    "fold": fold.number,
                    "sample_id": prepared.sample.sample_id,
                    "subject_id": prepared.sample.subject_id,
                    "cohort": prepared.sample.cohort,
                    "actual_ts": prepared.sample.score,
                    "predicted_ts": float(fold_predictions[position]),
                    "yu_xiong_paper_score_0_100": alignment.paper_score,
                    "yu_xiong_paper_score_unclipped": alignment.paper_score_unclipped,
                    "mean_aligned_vector_angle_degrees": alignment.mean_angle_degrees,
                    "total_angular_dtw_cost_degrees": (
                        alignment.total_angular_cost_degrees
                    ),
                    "required_joints_tracked_fraction": (
                        prepared.required_joints_tracked_fraction
                    ),
                    "training_median_baseline_ts": median_value,
                    "training_mean_baseline_ts": mean_value,
                    "training_cohort_median_baseline_ts": float(
                        cohort_median_predictions[position]
                    ),
                    "training_cohort_mean_baseline_ts": float(
                        cohort_mean_predictions[position]
                    ),
                    "frames": len(prepared.vectors),
                    "reference_frames": len(reference.vectors),
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
                    "feature": FEATURE_NAME,
                    "reference_sample_id": reference.sample.sample_id,
                    "reference_subject_id": reference.sample.subject_id,
                    "reference_actual_ts": reference.sample.score,
                    "reference_required_joints_tracked_fraction": (
                        reference.required_joints_tracked_fraction
                    ),
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
    predicted = np.asarray(
        [row["predicted_ts"] for row in prediction_rows], dtype=float
    )
    median_baseline = np.asarray(
        [row["training_median_baseline_ts"] for row in prediction_rows], dtype=float
    )
    mean_baseline = np.asarray(
        [row["training_mean_baseline_ts"] for row in prediction_rows], dtype=float
    )
    cohort_median_baseline = np.asarray(
        [row["training_cohort_median_baseline_ts"] for row in prediction_rows],
        dtype=float,
    )
    cohort_mean_baseline = np.asarray(
        [row["training_cohort_mean_baseline_ts"] for row in prediction_rows],
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
            _metrics_row("overall", METHOD_NAME, actual, predicted, True),
            _metrics_row(
                "overall", "training_median_constant", actual, median_baseline, False
            ),
            _metrics_row(
                "overall", "training_mean_constant", actual, mean_baseline, False
            ),
            _metrics_row(
                "overall",
                "training_cohort_median_constant",
                actual,
                cohort_median_baseline,
                False,
            ),
            _metrics_row(
                "overall",
                "training_cohort_mean_constant",
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
        actual, predicted, oof_groups, bootstrap_resamples
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
    progress(f"Hashing the exact {exercise} targets and JointPosition inputs...")
    experiment_inputs_sha256 = _experiment_inputs_sha256(samples)

    non_diagonal_fractions = np.asarray(
        [row["alignment_non_diagonal_step_fraction"] for row in prediction_rows],
        dtype=float,
    )
    tracking_fractions = np.asarray(
        [
            row["required_joints_tracked_fraction"]
            for row in prediction_rows
        ],
        dtype=float,
    )
    summary: dict[str, object] = {
        "method": METHOD_NAME,
        "exercise": exercise,
        "paper": {
            "authors": "Xiaoqun Yu and Shuping Xiong",
            "title": (
                "A Dynamic Time Warping Based Algorithm to Evaluate Kinect-Enabled "
                "Home-Based Physical Rehabilitation Exercises for Older People"
            ),
            "doi": PAPER_DOI,
        },
        "feature": FEATURE_NAME,
        "feature_dimensions": FEATURE_DIMENSIONS,
        "samples": len(samples),
        "unique_subjects": len(set(groups)),
        "folds": len(folds),
        "excluded_samples": excluded,
        "subject_overlap_in_every_fold": 0,
        "oof_prediction_rows": len(prediction_rows),
        "reference_rule_kimore_adaptation": (
            "highest clinical TS among outer-training recordings, then highest "
            "fraction of frames with every required limb joint fully tracked; "
            "final tie by sample ID"
        ),
        "required_joint_tracking_diagnostic": {
            "definition": (
                "fraction of frames in which all 12 joints required by the "
                "eight limb vectors have Kinect tracking state 2"
            ),
            "minimum": float(np.min(tracking_fractions)),
            "median": float(np.median(tracking_fractions)),
            "maximum": float(np.max(tracking_fractions)),
        },
        "local_cost": (
            "sum of angular differences in degrees between eight corresponding "
            "body-local limb vectors and the body forward vector"
        ),
        "paper_score": (
            "100 * (1 - angular_DTW_cost / (90 * 9 * path_length)); raw score "
            "retained and public score clipped to [0, 100] because KIMORE can "
            "violate the paper's <=90-degree assumption"
        ),
        "calibration": (
            "ordinary least-squares KIMORE TS = intercept + slope * paper score, "
            "fitted separately in each outer training fold"
        ),
        "overall_yu_xiong_dtw": overall_metrics,
        "overall_training_median_constant": {
            "mae": median_metrics["mae"],
            "rmse": median_metrics["rmse"],
        },
        "overall_training_mean_constant": {
            "mae": mean_metrics["mae"],
            "rmse": mean_metrics["rmse"],
        },
        "overall_training_cohort_constants": {
            "cohort_median": {
                "mae": cohort_median_metrics["mae"],
                "rmse": cohort_median_metrics["rmse"],
            },
            "cohort_mean": {
                "mae": cohort_mean_metrics["mae"],
                "rmse": cohort_mean_metrics["rmse"],
            },
        },
        "per_cohort_diagnostics": post_hoc_cohort_diagnostics(
            actual, predicted, oof_cohorts
        ),
        "dtw_warp_diagnostics": {
            "median_non_diagonal_step_fraction": float(
                np.median(non_diagonal_fractions)
            ),
            "maximum_non_diagonal_step_fraction": float(
                np.max(non_diagonal_fractions)
            ),
        },
        "bootstrap_95_percent_intervals": intervals,
        "paired_bootstrap_improvements": improvements,
        "bootstrap_scope": (
            "conditional fixed-OOF-prediction subject bootstrap; folds, coach "
            "selection and calibration are not refitted"
        ),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "manifest_path": _portable_path(manifest_path),
        "manifest_sha256": _manifest_sha256(manifest_path),
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
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation": (
            "Subject-disjoint internal development cross-validation. The original "
            "paper uses a separately recorded virtual coach; choosing a KIMORE "
            "training-fold coach and fitting KIMORE TS calibration are explicit "
            "dataset adaptations, not claims of an exact experimental replication."
        ),
    }
    _write_json(summary, summary_path)
    save_prediction_plot(
        prediction_rows,
        overall_metrics,
        intervals,
        plot_path,
        title="Yu--Xiong DTW: five-fold subject-wise OOF predictions",
    )

    progress(
        "Overall Yu--Xiong DTW metrics: "
        f"MAE={overall_metrics['mae']:.3f}, "
        f"RMSE={overall_metrics['rmse']:.3f}, "
        f"Spearman={overall_metrics['spearman']:.3f}, "
        f"Pearson={overall_metrics['pearson']:.3f}"
    )
    progress(f"Predictions: {predictions_path.resolve()}")
    progress(f"Metrics: {metrics_path.resolve()}")
    progress(f"Summary: {summary_path.resolve()}")
    progress(f"Plot: {plot_path.resolve()}")
    return summary
