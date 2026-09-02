"""Subject-disjoint CSV export for interpretable KIMORE DTW alignments."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

from kimore_dataset import load_joint_positions, read_manifest
from kimore_evaluation import (
    BOOTSTRAP_SEED,
    _experiment_inputs_sha256,
    _git_revision,
    _manifest_sha256,
    _metrics_row,
    _package_version,
    _portable_path,
    _source_code_sha256,
    _write_json,
    alignment_warp_diagnostics,
    bootstrap_improvement_intervals,
    bootstrap_metric_intervals,
    bootstrap_paired_metric_improvements,
    post_hoc_cohort_diagnostics,
    regression_metrics,
    save_prediction_plot,
    training_constant_values,
    validate_oof_indices,
)
from kimore_grouping import assert_no_subject_leakage, make_subject_folds, subject_groups
from kimore_interpretable_dtw import (
    InterpretableDtwAlignment,
    interpretable_dtw,
    summarize_component_errors,
)
from kimore_interpretable_quality import (
    FrameQualityResult,
    apply_frame_quality_control,
)
from kimore_interpretable_localization import (
    TOP_DEVIATION_INTERVALS_PER_SAMPLE,
    annotation_queue_rows,
    iter_frame_timeline_rows,
    top_deviation_interval_rows,
)
from kimore_plain_dtw import fit_linear_calibration
from kimore_yu_xiong_dtw import (
    FEATURE_DIMENSIONS,
    FEATURE_NAME,
    YuXiongPreparedSample,
    prepare_yu_xiong_sample,
    select_yu_xiong_reference,
)


METHOD_NAME = "interpretable_dtw"


def save_component_distribution_plot(
    rows: list[dict[str, object]],
    value_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot the held-out subject distribution for every body component."""
    if not rows:
        raise ValueError("Cannot plot empty component summaries")

    component_indices = sorted({int(row["component_index"]) for row in rows})
    if component_indices != list(range(FEATURE_DIMENSIONS)):
        raise ValueError(
            f"Expected component indices 0--{FEATURE_DIMENSIONS - 1}; "
            f"got {component_indices}"
        )

    labels: list[str] = []
    distributions: list[np.ndarray] = []
    for component_index in component_indices:
        component_rows = [
            row for row in rows if int(row["component_index"]) == component_index
        ]
        names = {str(row["component_name"]) for row in component_rows}
        if len(names) != 1:
            raise ValueError(
                f"Component {component_index} has inconsistent names: {sorted(names)}"
            )
        values = np.asarray(
            [float(row[value_key]) for row in component_rows],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"{value_key} must contain finite non-negative values")
        labels.append(next(iter(names)).replace("_", " "))
        distributions.append(values)

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13.0, 6.2))
    boxes = axis.boxplot(
        distributions,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    axis.set_xticks(np.arange(1, len(labels) + 1))
    axis.set_xticklabels(labels)
    colors = plt.get_cmap("tab10")
    for index, (box, values) in enumerate(zip(boxes["boxes"], distributions)):
        color = colors(index)
        box.set_facecolor(color)
        box.set_alpha(0.35)
        offsets = (
            np.linspace(-0.16, 0.16, len(values))
            if len(values) > 1
            else np.zeros(1, dtype=np.float64)
        )
        axis.scatter(
            np.full(len(values), index + 1, dtype=np.float64) + offsets,
            values,
            s=14,
            alpha=0.45,
            color=color,
            edgecolors="none",
        )

    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.22)
    axis.tick_params(axis="x", labelrotation=32)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_stream(
    rows: Iterable[dict[str, object]],
    path: Path,
) -> int:
    """Write a potentially large CSV without retaining every row in memory."""
    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError(f"Cannot write an empty CSV: {path}") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        for row in iterator:
            writer.writerow(row)
            count += 1
    return count


def _alignment_rows(
    fold_number: int,
    prepared: YuXiongPreparedSample,
    reference: YuXiongPreparedSample,
    alignment: InterpretableDtwAlignment,
    sample_quality: FrameQualityResult,
    reference_quality: FrameQualityResult,
    input_variant: str,
) -> list[dict[str, object]]:
    """Build the nine component rows for one raw or frame-QC alignment."""
    summaries = summarize_component_errors(alignment)
    rows: list[dict[str, object]] = []
    for component_index, component in enumerate(summaries):
        quality = sample_quality.component_summaries[component_index]
        rows.append(
            {
                "fold": fold_number,
                "input_variant": input_variant,
                "sample_id": prepared.sample.sample_id,
                "subject_id": prepared.sample.subject_id,
                "cohort": prepared.sample.cohort,
                "actual_ts": prepared.sample.score,
                "reference_sample_id": reference.sample.sample_id,
                "reference_subject_id": reference.sample.subject_id,
                "reference_actual_ts": reference.sample.score,
                "sample_frames_used": len(prepared.vectors),
                "reference_frames_used": len(reference.vectors),
                "component_index": component_index,
                "component_name": component.name,
                "sample_quality_status": sample_quality.quality_status,
                "sample_quality_reasons": "|".join(
                    sample_quality.quality_reasons
                ),
                "sample_total_frames": sample_quality.total_frames,
                "sample_interpolated_frames": (
                    sample_quality.interpolated_frames
                ),
                "sample_interpolated_component_frames": (
                    sample_quality.interpolated_component_frames
                ),
                "sample_dropped_frames": sample_quality.dropped_frames,
                "sample_retained_frames": sample_quality.retained_frames,
                "sample_retained_fraction": sample_quality.retained_fraction,
                "sample_longest_dropped_run": (
                    sample_quality.longest_dropped_run
                ),
                "component_quality_status": quality.quality_status,
                "component_quality_reasons": "|".join(
                    quality.quality_reasons
                ),
                "component_tracked_fraction": quality.tracked_fraction,
                "component_stable_length_fraction": (
                    quality.stable_length_fraction
                ),
                "component_temporal_continuity_fraction": (
                    quality.temporal_continuity_fraction
                ),
                "component_anatomical_plausibility_fraction": (
                    quality.anatomical_plausibility_fraction
                ),
                "component_invalid_frames": quality.invalid_frames,
                "component_invalid_fraction": quality.invalid_fraction,
                "component_invalid_run_count": quality.invalid_run_count,
                "component_longest_invalid_run": quality.longest_invalid_run,
                "component_interpolated_frames": quality.interpolated_frames,
                "component_unrepaired_invalid_frames": (
                    quality.unrepaired_invalid_frames
                ),
                "component_unrepaired_invalid_fraction": (
                    quality.unrepaired_invalid_fraction
                ),
                "component_longest_unrepaired_run": (
                    quality.longest_unrepaired_run
                ),
                "total_error_degrees": component.total_error_degrees,
                "mean_error_degrees": component.mean_error_degrees,
                "maximum_error_degrees": component.maximum_error_degrees,
                "contribution_percent": component.contribution_percent,
                "alignment_path_length": len(alignment.path),
                "total_angular_dtw_cost_degrees": (
                    alignment.total_angular_cost_degrees
                ),
                "mean_aligned_vector_angle_degrees": (
                    alignment.mean_angle_degrees
                ),
                "yu_xiong_paper_score_0_100": alignment.paper_score,
                "required_joints_tracked_fraction": (
                    prepared.required_joints_tracked_fraction
                ),
                "reference_required_joints_tracked_fraction": (
                    reference.required_joints_tracked_fraction
                ),
                "reference_quality_status": reference_quality.quality_status,
                "reference_total_frames": reference_quality.total_frames,
                "reference_interpolated_frames": (
                    reference_quality.interpolated_frames
                ),
                "reference_dropped_frames": reference_quality.dropped_frames,
                "reference_retained_fraction": (
                    reference_quality.retained_fraction
                ),
            }
        )
    return rows


def _quality_aware_reference(
    prepared_samples: list[YuXiongPreparedSample],
    train_indices: np.ndarray,
    sample_quality_statuses: list[str],
) -> int:
    """Select a training-only reference that has not failed full-body QC."""
    usable_indices = [
        int(index)
        for index in train_indices
        if sample_quality_statuses[int(index)] != "fail"
    ]
    if not usable_indices:
        raise ValueError("No full-body-QC-usable reference in the training fold")
    return select_yu_xiong_reference(prepared_samples, usable_indices)


def run_interpretable_evaluation(
    manifest_path: Path,
    exercise: str,
    output_dir: Path,
    figure_dir: Path,
    bootstrap_resamples: int = 5000,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Evaluate frame QC and export interpretable held-out alignments."""
    if bootstrap_resamples < 1:
        raise ValueError("Bootstrap resamples must be at least 1")

    samples, excluded = read_manifest(manifest_path, exercise)
    folds = make_subject_folds(samples, n_splits=5)
    groups = subject_groups(samples)
    validate_oof_indices([fold.test_indices for fold in folds], len(samples))

    progress(
        f"Loading, extracting, and quality-checking interpretable DTW vectors "
        f"for {len(samples)} usable {exercise} recordings..."
    )
    prepared_samples: list[YuXiongPreparedSample] = []
    qc_prepared_samples: list[YuXiongPreparedSample] = []
    frame_quality: list[FrameQualityResult] = []
    sample_quality_statuses: list[str] = []
    quality_rows: list[dict[str, object]] = []
    for number, sample in enumerate(samples, start=1):
        prepared = prepare_yu_xiong_sample(sample)
        frame_result = apply_frame_quality_control(
            load_joint_positions(sample.position_path),
            prepared.vectors,
            exercise,
        )
        quality = frame_result.component_summaries
        sample_status = frame_result.quality_status
        prepared_samples.append(prepared)
        qc_prepared_samples.append(
            YuXiongPreparedSample(
                sample=prepared.sample,
                vectors=frame_result.cleaned_vectors,
                required_joints_tracked_fraction=(
                    prepared.required_joints_tracked_fraction
                ),
            )
        )
        frame_quality.append(frame_result)
        sample_quality_statuses.append(sample_status)
        for component in quality:
            quality_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "cohort": sample.cohort,
                    "actual_ts": sample.score,
                    "sample_quality_status": sample_status,
                    "sample_quality_reasons": "|".join(
                        frame_result.quality_reasons
                    ),
                    "total_frames": frame_result.total_frames,
                    "interpolated_frames": frame_result.interpolated_frames,
                    "interpolated_component_frames": (
                        frame_result.interpolated_component_frames
                    ),
                    "dropped_frames": frame_result.dropped_frames,
                    "retained_frames": frame_result.retained_frames,
                    "retained_fraction": frame_result.retained_fraction,
                    "longest_dropped_run": frame_result.longest_dropped_run,
                    "component_index": component.component_index,
                    "component_name": component.component_name,
                    "source_joints": "|".join(component.source_joints),
                    "tracked_fraction": component.tracked_fraction,
                    "stable_length_fraction": component.stable_length_fraction,
                    "temporal_continuity_fraction": (
                        component.temporal_continuity_fraction
                    ),
                    "anatomical_plausibility_fraction": (
                        component.anatomical_plausibility_fraction
                    ),
                    "not_fully_tracked_frames": (
                        component.not_fully_tracked_frames
                    ),
                    "untracked_frames": component.untracked_frames,
                    "source_length_invalid_frames": (
                        component.source_length_invalid_frames
                    ),
                    "temporal_outlier_frames": (
                        component.temporal_outlier_frames
                    ),
                    "anatomical_invalid_frames": (
                        component.anatomical_invalid_frames
                    ),
                    "body_frame_invalid_frames": (
                        component.body_frame_invalid_frames
                    ),
                    "invalid_frames": component.invalid_frames,
                    "invalid_fraction": component.invalid_fraction,
                    "invalid_run_count": component.invalid_run_count,
                    "longest_invalid_run": component.longest_invalid_run,
                    "component_interpolated_frames": (
                        component.interpolated_frames
                    ),
                    "unrepaired_invalid_frames": (
                        component.unrepaired_invalid_frames
                    ),
                    "unrepaired_invalid_fraction": (
                        component.unrepaired_invalid_fraction
                    ),
                    "longest_unrepaired_run": (
                        component.longest_unrepaired_run
                    ),
                    "component_quality_status": component.quality_status,
                    "quality_reasons": "|".join(component.quality_reasons),
                }
            )
        if number % 10 == 0 or number == len(samples):
            progress(f"  prepared {number}/{len(samples)}")

    quality_path = output_dir / "component_quality.csv"
    _write_csv(quality_rows, quality_path)
    quality_counts = {
        status: sample_quality_statuses.count(status)
        for status in ("pass", "warning", "fail")
    }
    progress(
        "Full-body QC: "
        f"{quality_counts['pass']} pass, {quality_counts['warning']} warning, "
        f"{quality_counts['fail']} fail"
    )
    progress(
        "Frame QC actions: "
        f"{sum(result.interpolated_frames for result in frame_quality)} "
        "unique frames interpolated, "
        f"{sum(result.dropped_frames for result in frame_quality)} frames removed"
    )

    scores = np.asarray([sample.score for sample in samples], dtype=np.float64)
    rows: list[dict[str, object]] = []
    qc_usable_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fold_metadata: list[dict[str, object]] = []
    localization_records: list[
        tuple[
            int,
            YuXiongPreparedSample,
            YuXiongPreparedSample,
            InterpretableDtwAlignment,
            FrameQualityResult,
            FrameQualityResult,
        ]
    ] = []
    deviation_interval_rows: list[dict[str, object]] = []
    exported_sample_ids: list[str] = []
    raw_alignment_cache: dict[int, list[InterpretableDtwAlignment]] = {}
    qc_alignment_cache: dict[
        int, list[InterpretableDtwAlignment | None]
    ] = {}

    for fold in folds:
        assert_no_subject_leakage(groups, fold.train_indices, fold.test_indices)
        usable_train_indices = np.asarray(
            [
                int(index)
                for index in fold.train_indices
                if sample_quality_statuses[int(index)] != "fail"
            ],
            dtype=int,
        )
        usable_test_indices = np.asarray(
            [
                int(index)
                for index in fold.test_indices
                if sample_quality_statuses[int(index)] != "fail"
            ],
            dtype=int,
        )
        if len(usable_train_indices) < 2:
            raise ValueError(
                f"Fold {fold.number} has fewer than two QC-usable training rows"
            )

        reference_index = _quality_aware_reference(
            prepared_samples,
            fold.train_indices,
            sample_quality_statuses,
        )
        reference = prepared_samples[reference_index]
        qc_reference = qc_prepared_samples[reference_index]
        reference_frame_quality = frame_quality[reference_index]
        progress(
            f"Fold {fold.number}: reference {reference.sample.sample_id}; "
            f"{len(usable_train_indices)} QC-usable train and "
            f"{len(usable_test_indices)} QC-usable test subjects"
        )

        if reference_index not in raw_alignment_cache:
            progress("  calculating raw and frame-QC alignments for calibration...")
            raw_alignments: list[InterpretableDtwAlignment] = []
            qc_alignments: list[InterpretableDtwAlignment | None] = []
            for number, prepared in enumerate(prepared_samples, start=1):
                raw_alignments.append(
                    interpretable_dtw(prepared.vectors, reference.vectors)
                )
                sample_index = number - 1
                if sample_quality_statuses[sample_index] == "fail":
                    qc_alignments.append(None)
                else:
                    qc_alignments.append(
                        interpretable_dtw(
                            qc_prepared_samples[sample_index].vectors,
                            qc_reference.vectors,
                        )
                    )
                if number % 10 == 0 or number == len(prepared_samples):
                    progress(f"    aligned {number}/{len(prepared_samples)}")
            raw_alignment_cache[reference_index] = raw_alignments
            qc_alignment_cache[reference_index] = qc_alignments
        else:
            progress("  reusing raw and frame-QC alignments for this reference")

        raw_alignments = raw_alignment_cache[reference_index]
        qc_alignments = qc_alignment_cache[reference_index]
        raw_train_scores = np.asarray(
            [raw_alignments[index].paper_score for index in usable_train_indices],
            dtype=np.float64,
        )
        qc_train_scores = np.asarray(
            [
                qc_alignments[index].paper_score
                for index in usable_train_indices
                if qc_alignments[index] is not None
            ],
            dtype=np.float64,
        )
        if len(qc_train_scores) != len(usable_train_indices):
            raise AssertionError("QC training scores do not match usable rows")

        raw_calibration = fit_linear_calibration(
            raw_train_scores,
            scores[usable_train_indices],
        )
        qc_calibration = fit_linear_calibration(
            qc_train_scores,
            scores[usable_train_indices],
        )
        median_value, mean_value = training_constant_values(
            scores,
            usable_train_indices,
        )

        raw_test_scores = np.asarray(
            [raw_alignments[index].paper_score for index in usable_test_indices],
            dtype=np.float64,
        )
        qc_test_scores = np.asarray(
            [
                qc_alignments[index].paper_score
                for index in usable_test_indices
                if qc_alignments[index] is not None
            ],
            dtype=np.float64,
        )
        if len(qc_test_scores) != len(usable_test_indices):
            raise AssertionError("QC test scores do not match usable rows")
        raw_fold_predictions = raw_calibration.predict(raw_test_scores)
        qc_fold_predictions = qc_calibration.predict(qc_test_scores)
        fold_actual = scores[usable_test_indices]
        median_predictions = np.full(len(usable_test_indices), median_value)
        mean_predictions = np.full(len(usable_test_indices), mean_value)

        if len(usable_test_indices) > 0:
            metric_rows.extend(
                (
                    _metrics_row(
                        f"fold_{fold.number}",
                        "frame_qc_yu_xiong_dtw",
                        fold_actual,
                        qc_fold_predictions,
                        include_correlations=True,
                    ),
                    _metrics_row(
                        f"fold_{fold.number}",
                        "raw_yu_xiong_dtw_qc_eligible",
                        fold_actual,
                        raw_fold_predictions,
                        include_correlations=True,
                    ),
                    _metrics_row(
                        f"fold_{fold.number}",
                        "training_median_constant_qc_eligible",
                        fold_actual,
                        median_predictions,
                        include_correlations=False,
                    ),
                    _metrics_row(
                        f"fold_{fold.number}",
                        "training_mean_constant_qc_eligible",
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
                "qc_usable_training_subjects": len(usable_train_indices),
                "test_subjects": len(test_subjects),
                "qc_usable_test_subjects": len(usable_test_indices),
                "subject_overlap": len(train_subjects.intersection(test_subjects)),
                "qc_usable_test_cohort_counts": dict(
                    Counter(samples[index].cohort for index in usable_test_indices)
                ),
                "reference_sample_id": reference.sample.sample_id,
                "reference_subject_id": reference.sample.subject_id,
                "reference_actual_ts": reference.sample.score,
                "reference_quality_status": reference_frame_quality.quality_status,
                "raw_calibration_intercept": raw_calibration.intercept,
                "raw_calibration_slope": raw_calibration.slope,
                "qc_calibration_intercept": qc_calibration.intercept,
                "qc_calibration_slope": qc_calibration.slope,
                "training_median_ts": median_value,
                "training_mean_ts": mean_value,
                "raw_test_metrics": (
                    regression_metrics(fold_actual, raw_fold_predictions)
                    if len(usable_test_indices) > 0
                    else None
                ),
                "qc_test_metrics": (
                    regression_metrics(fold_actual, qc_fold_predictions)
                    if len(usable_test_indices) > 0
                    else None
                ),
            }
        )

        usable_test_positions = {
            int(sample_index): position
            for position, sample_index in enumerate(usable_test_indices)
        }
        for sample_index_value in fold.test_indices:
            sample_index = int(sample_index_value)
            prepared = prepared_samples[sample_index]
            alignment = raw_alignments[sample_index]
            sample_frame_quality = frame_quality[sample_index]
            exported_sample_ids.append(prepared.sample.sample_id)
            rows.extend(
                _alignment_rows(
                    fold_number=fold.number,
                    prepared=prepared,
                    reference=reference,
                    alignment=alignment,
                    sample_quality=sample_frame_quality,
                    reference_quality=reference_frame_quality,
                    input_variant="raw",
                )
            )

            if sample_frame_quality.quality_status == "fail":
                continue
            qc_alignment = qc_alignments[sample_index]
            if qc_alignment is None:
                raise AssertionError("QC-usable sample is missing its alignment")
            qc_prepared = qc_prepared_samples[sample_index]
            qc_usable_rows.extend(
                _alignment_rows(
                    fold_number=fold.number,
                    prepared=qc_prepared,
                    reference=qc_reference,
                    alignment=qc_alignment,
                    sample_quality=sample_frame_quality,
                    reference_quality=reference_frame_quality,
                    input_variant="frame_qc",
                )
            )
            localization_records.append(
                (
                    fold.number,
                    qc_prepared,
                    qc_reference,
                    qc_alignment,
                    sample_frame_quality,
                    reference_frame_quality,
                )
            )
            deviation_interval_rows.extend(
                top_deviation_interval_rows(
                    fold.number,
                    qc_prepared,
                    qc_reference,
                    qc_alignment,
                    sample_frame_quality,
                    reference_frame_quality,
                )
            )

            position = usable_test_positions[sample_index]
            raw_warp = alignment_warp_diagnostics(
                len(prepared.vectors),
                len(reference.vectors),
                len(alignment.path),
            )
            qc_warp = alignment_warp_diagnostics(
                len(qc_prepared.vectors),
                len(qc_reference.vectors),
                len(qc_alignment.path),
            )
            prediction_rows.append(
                {
                    "fold": fold.number,
                    "sample_id": prepared.sample.sample_id,
                    "subject_id": prepared.sample.subject_id,
                    "cohort": prepared.sample.cohort,
                    "actual_ts": prepared.sample.score,
                    "predicted_ts": float(qc_fold_predictions[position]),
                    "qc_predicted_ts": float(qc_fold_predictions[position]),
                    "raw_predicted_ts": float(raw_fold_predictions[position]),
                    "qc_paper_score_0_100": qc_alignment.paper_score,
                    "raw_paper_score_0_100": alignment.paper_score,
                    "qc_mean_aligned_vector_angle_degrees": (
                        qc_alignment.mean_angle_degrees
                    ),
                    "raw_mean_aligned_vector_angle_degrees": (
                        alignment.mean_angle_degrees
                    ),
                    "sample_quality_status": sample_frame_quality.quality_status,
                    "sample_quality_reasons": "|".join(
                        sample_frame_quality.quality_reasons
                    ),
                    "sample_total_frames": sample_frame_quality.total_frames,
                    "sample_interpolated_frames": (
                        sample_frame_quality.interpolated_frames
                    ),
                    "sample_dropped_frames": sample_frame_quality.dropped_frames,
                    "sample_retained_frames": sample_frame_quality.retained_frames,
                    "sample_retained_fraction": (
                        sample_frame_quality.retained_fraction
                    ),
                    "raw_frames": len(prepared.vectors),
                    "qc_frames": len(qc_prepared.vectors),
                    "raw_reference_frames": len(reference.vectors),
                    "qc_reference_frames": len(qc_reference.vectors),
                    "raw_alignment_path_length": len(alignment.path),
                    "qc_alignment_path_length": len(qc_alignment.path),
                    "raw_alignment_non_diagonal_step_fraction": raw_warp[
                        "non_diagonal_step_fraction"
                    ],
                    "qc_alignment_non_diagonal_step_fraction": qc_warp[
                        "non_diagonal_step_fraction"
                    ],
                    "training_median_baseline_ts": median_value,
                    "training_mean_baseline_ts": mean_value,
                    "qc_usable_training_subjects": len(usable_train_indices),
                    "feature": FEATURE_NAME,
                    "reference_sample_id": reference.sample.sample_id,
                    "reference_subject_id": reference.sample.subject_id,
                    "reference_actual_ts": reference.sample.score,
                    "reference_quality_status": reference_frame_quality.quality_status,
                    "raw_calibration_intercept": raw_calibration.intercept,
                    "raw_calibration_slope": raw_calibration.slope,
                    "qc_calibration_intercept": qc_calibration.intercept,
                    "qc_calibration_slope": qc_calibration.slope,
                }
            )

    if len(exported_sample_ids) != len(samples):
        raise AssertionError(
            f"Expected {len(samples)} held-out samples; got {len(exported_sample_ids)}"
        )
    if len(set(exported_sample_ids)) != len(samples):
        raise AssertionError("Each sample must be exported from exactly one test fold")
    expected_rows = len(samples) * FEATURE_DIMENSIONS
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} component rows; got {len(rows)}")
    qc_usable_samples = sum(status != "fail" for status in sample_quality_statuses)
    expected_qc_rows = qc_usable_samples * FEATURE_DIMENSIONS
    if len(qc_usable_rows) != expected_qc_rows:
        raise AssertionError(
            f"Expected {expected_qc_rows} frame-QC component rows; "
            f"got {len(qc_usable_rows)}"
        )
    if len(prediction_rows) != qc_usable_samples:
        raise AssertionError(
            f"Expected {qc_usable_samples} QC OOF rows; got {len(prediction_rows)}"
        )
    if len({row["sample_id"] for row in prediction_rows}) != qc_usable_samples:
        raise AssertionError("Each QC-usable sample must have one OOF prediction")
    expected_interval_rows = (
        qc_usable_samples * TOP_DEVIATION_INTERVALS_PER_SAMPLE
    )
    if len(deviation_interval_rows) != expected_interval_rows:
        raise AssertionError(
            f"Expected {expected_interval_rows} deviation intervals; "
            f"got {len(deviation_interval_rows)}"
        )

    sample_order = {sample.sample_id: index for index, sample in enumerate(samples)}
    rows.sort(
        key=lambda row: (
            sample_order[str(row["sample_id"])],
            int(row["component_index"]),
        )
    )
    qc_usable_rows.sort(
        key=lambda row: (
            sample_order[str(row["sample_id"])],
            int(row["component_index"]),
        )
    )
    prediction_rows.sort(key=lambda row: sample_order[str(row["sample_id"])])
    deviation_interval_rows.sort(
        key=lambda row: (
            sample_order[str(row["sample_id"])],
            int(row["candidate_rank"]),
        )
    )
    review_queue_rows = annotation_queue_rows(deviation_interval_rows)

    output_path = output_dir / "component_summaries.csv"
    qc_usable_output_path = output_dir / "component_summaries_qc_usable.csv"
    timeline_path = output_dir / "error_timeline.csv"
    deviation_intervals_path = output_dir / "top_deviation_intervals.csv"
    annotation_queue_path = output_dir / "annotation_queue.csv"
    predictions_path = output_dir / "oof_predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    fold_metadata_path = output_dir / "fold_metadata.json"
    summary_path = output_dir / "evaluation_summary.json"
    _write_csv(rows, output_path)
    _write_csv(qc_usable_rows, qc_usable_output_path)
    _write_csv(deviation_interval_rows, deviation_intervals_path)
    _write_csv(review_queue_rows, annotation_queue_path)

    def timeline_rows() -> Iterable[dict[str, object]]:
        for record in localization_records:
            yield from iter_frame_timeline_rows(*record)

    timeline_row_count = _write_csv_stream(timeline_rows(), timeline_path)
    expected_timeline_rows = sum(
        len(prepared.vectors) * FEATURE_DIMENSIONS
        for _, prepared, _, _, _, _ in localization_records
    )
    if timeline_row_count != expected_timeline_rows:
        raise AssertionError(
            f"Expected {expected_timeline_rows} timeline rows; "
            f"got {timeline_row_count}"
        )
    _write_csv(prediction_rows, predictions_path)

    actual = np.asarray([row["actual_ts"] for row in prediction_rows], dtype=float)
    qc_predicted = np.asarray(
        [row["qc_predicted_ts"] for row in prediction_rows], dtype=float
    )
    raw_predicted = np.asarray(
        [row["raw_predicted_ts"] for row in prediction_rows], dtype=float
    )
    median_baseline = np.asarray(
        [row["training_median_baseline_ts"] for row in prediction_rows], dtype=float
    )
    mean_baseline = np.asarray(
        [row["training_mean_baseline_ts"] for row in prediction_rows], dtype=float
    )
    oof_groups = [str(row["subject_id"]) for row in prediction_rows]
    oof_cohorts = [str(row["cohort"]) for row in prediction_rows]
    qc_metrics = regression_metrics(actual, qc_predicted)
    raw_metrics = regression_metrics(actual, raw_predicted)
    median_metrics = regression_metrics(actual, median_baseline)
    mean_metrics = regression_metrics(actual, mean_baseline)
    metric_rows.extend(
        (
            _metrics_row(
                "overall", "frame_qc_yu_xiong_dtw", actual, qc_predicted, True
            ),
            _metrics_row(
                "overall",
                "raw_yu_xiong_dtw_qc_eligible",
                actual,
                raw_predicted,
                True,
            ),
            _metrics_row(
                "overall",
                "training_median_constant_qc_eligible",
                actual,
                median_baseline,
                False,
            ),
            _metrics_row(
                "overall",
                "training_mean_constant_qc_eligible",
                actual,
                mean_baseline,
                False,
            ),
        )
    )
    _write_csv(metric_rows, metrics_path)
    _write_json({"folds": fold_metadata}, fold_metadata_path)

    progress(
        f"Bootstrapping {bootstrap_resamples} paired fixed-OOF resamples..."
    )
    qc_intervals = bootstrap_metric_intervals(
        actual, qc_predicted, oof_groups, bootstrap_resamples
    )
    raw_intervals = bootstrap_metric_intervals(
        actual, raw_predicted, oof_groups, bootstrap_resamples
    )
    paired_improvements = bootstrap_paired_metric_improvements(
        actual,
        qc_predicted,
        raw_predicted,
        oof_groups,
        bootstrap_resamples,
    )
    qc_constant_improvements = bootstrap_improvement_intervals(
        actual,
        qc_predicted,
        median_baseline,
        mean_baseline,
        oof_groups,
        bootstrap_resamples,
    )

    error_figure_path = figure_dir / "component_error_distributions.png"
    contribution_figure_path = figure_dir / "component_contribution_distributions.png"
    qc_error_figure_path = figure_dir / "component_error_distributions_qc_usable.png"
    qc_contribution_figure_path = (
        figure_dir / "component_contribution_distributions_qc_usable.png"
    )
    prediction_figure_path = figure_dir / "actual_vs_predicted_qc.png"
    save_component_distribution_plot(
        rows,
        value_key="mean_error_degrees",
        ylabel="Mean aligned angular error (degrees)",
        title=(
            f"{exercise} interpretable DTW: component error distributions "
            "across held-out subjects"
        ),
        output_path=error_figure_path,
    )
    save_component_distribution_plot(
        rows,
        value_key="contribution_percent",
        ylabel="Contribution to total angular DTW cost (%)",
        title=(
            f"{exercise} interpretable DTW: component contribution distributions "
            "across held-out subjects"
        ),
        output_path=contribution_figure_path,
    )
    save_component_distribution_plot(
        qc_usable_rows,
        value_key="mean_error_degrees",
        ylabel="Mean aligned angular error (degrees)",
        title=(
            f"{exercise} interpretable DTW: component errors for "
            "full-body-QC-usable subjects"
        ),
        output_path=qc_error_figure_path,
    )
    save_component_distribution_plot(
        qc_usable_rows,
        value_key="contribution_percent",
        ylabel="Contribution to total angular DTW cost (%)",
        title=(
            f"{exercise} interpretable DTW: component contributions for "
            "full-body-QC-usable subjects"
        ),
        output_path=qc_contribution_figure_path,
    )
    save_prediction_plot(
        prediction_rows,
        qc_metrics,
        qc_intervals,
        prediction_figure_path,
        title="Frame-QC Yu--Xiong DTW: subject-wise OOF predictions",
    )

    progress(f"Hashing the exact {exercise} targets and JointPosition inputs...")
    experiment_inputs_sha256 = _experiment_inputs_sha256(samples)
    summary: dict[str, object] = {
        "method": METHOD_NAME,
        "evaluation_variant": "frame_qc_yu_xiong_dtw",
        "exercise": exercise,
        "samples": len(samples),
        "unique_subjects": len(set(groups)),
        "folds": len(folds),
        "excluded_samples": excluded,
        "subject_overlap_in_every_fold": 0,
        "component_rows": len(rows),
        "qc_usable_samples": qc_usable_samples,
        "qc_failed_samples": len(samples) - qc_usable_samples,
        "qc_coverage_fraction": qc_usable_samples / len(samples),
        "qc_usable_component_rows": len(qc_usable_rows),
        "error_timeline_rows": timeline_row_count,
        "top_deviation_interval_rows": len(deviation_interval_rows),
        "annotation_queue_rows": len(review_queue_rows),
        "localization_interpretation": (
            "angular deviations from the training-only reference are candidates "
            "for human review, not validated execution-error labels"
        ),
        "oof_prediction_rows": len(prediction_rows),
        "interpolated_frames": sum(
            result.interpolated_frames for result in frame_quality
        ),
        "interpolated_component_frames": sum(
            result.interpolated_component_frames for result in frame_quality
        ),
        "dropped_frames": sum(result.dropped_frames for result in frame_quality),
        "components_per_sample": FEATURE_DIMENSIONS,
        "quality_counts": quality_counts,
        "calibration": (
            "separate ordinary least-squares TS calibrations for raw and frame-QC "
            "paper scores, each fitted only on QC-usable outer-training rows"
        ),
        "paired_comparison_population": (
            "identical QC-usable held-out subjects; raw and frame-QC variants use "
            "the same QC-usable training rows and the same training-only reference"
        ),
        "overall_frame_qc_yu_xiong_dtw": qc_metrics,
        "overall_raw_yu_xiong_dtw_qc_eligible": raw_metrics,
        "overall_training_median_constant_qc_eligible": {
            "mae": median_metrics["mae"],
            "rmse": median_metrics["rmse"],
        },
        "overall_training_mean_constant_qc_eligible": {
            "mae": mean_metrics["mae"],
            "rmse": mean_metrics["rmse"],
        },
        "frame_qc_bootstrap_95_percent_intervals": qc_intervals,
        "raw_qc_eligible_bootstrap_95_percent_intervals": raw_intervals,
        "paired_frame_qc_improvements_over_raw": paired_improvements,
        "frame_qc_improvements_over_training_constants": (
            qc_constant_improvements
        ),
        "per_cohort_frame_qc_diagnostics": post_hoc_cohort_diagnostics(
            actual, qc_predicted, oof_cohorts
        ),
        "bootstrap_scope": (
            "conditional paired fixed-OOF-prediction subject bootstrap; folds, "
            "QC decisions, reference selection and calibrations are not refitted"
        ),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "manifest_path": _portable_path(manifest_path),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "experiment_inputs_sha256": experiment_inputs_sha256,
        "source_git_revision": _git_revision(),
        "source_code_sha256": _source_code_sha256(),
        "outputs": {
            "component_summaries": _portable_path(output_path),
            "qc_component_summaries": _portable_path(qc_usable_output_path),
            "component_quality": _portable_path(quality_path),
            "error_timeline": _portable_path(timeline_path),
            "top_deviation_intervals": _portable_path(
                deviation_intervals_path
            ),
            "annotation_queue": _portable_path(annotation_queue_path),
            "predictions": _portable_path(predictions_path),
            "metrics": _portable_path(metrics_path),
            "fold_metadata": _portable_path(fold_metadata_path),
            "summary": _portable_path(summary_path),
            "prediction_plot": _portable_path(prediction_figure_path),
        },
        "environment": {
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "matplotlib": matplotlib.__version__,
            "scikit_learn": _package_version("scikit-learn"),
        },
        "interpretation": (
            "Subject-disjoint internal development cross-validation. QC-failed "
            "recordings are abstentions and remain outside prediction metrics; "
            "coverage must therefore be reported with accuracy."
        ),
        "output": str(output_path.resolve()),
        "qc_usable_output": str(qc_usable_output_path.resolve()),
        "component_quality_output": str(quality_path.resolve()),
        "figures": [
            str(error_figure_path.resolve()),
            str(contribution_figure_path.resolve()),
            str(qc_error_figure_path.resolve()),
            str(qc_contribution_figure_path.resolve()),
            str(prediction_figure_path.resolve()),
        ],
    }
    _write_json(summary, summary_path)

    progress(
        "Overall frame-QC metrics: "
        f"MAE={qc_metrics['mae']:.3f}, RMSE={qc_metrics['rmse']:.3f}, "
        f"Spearman={qc_metrics['spearman']:.3f}, "
        f"Pearson={qc_metrics['pearson']:.3f}"
    )
    progress(
        "Paired raw metrics: "
        f"MAE={raw_metrics['mae']:.3f}, RMSE={raw_metrics['rmse']:.3f}, "
        f"Spearman={raw_metrics['spearman']:.3f}, "
        f"Pearson={raw_metrics['pearson']:.3f}"
    )
    progress(
        f"QC coverage: {qc_usable_samples}/{len(samples)} "
        f"({qc_usable_samples / len(samples):.1%})"
    )
    progress(f"Predictions: {predictions_path.resolve()}")
    progress(f"Metrics: {metrics_path.resolve()}")
    progress(f"Error timeline: {timeline_path.resolve()}")
    progress(f"Annotation queue: {annotation_queue_path.resolve()}")
    progress(f"Summary: {summary_path.resolve()}")
    return summary
