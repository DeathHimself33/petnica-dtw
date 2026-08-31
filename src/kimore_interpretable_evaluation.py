"""Subject-disjoint CSV export for interpretable KIMORE DTW alignments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from kimore_dataset import load_joint_positions, read_manifest
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
from kimore_yu_xiong_dtw import (
    FEATURE_DIMENSIONS,
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
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Export and plot nine component summaries for every held-out subject."""
    samples, excluded = read_manifest(manifest_path, exercise)
    folds = make_subject_folds(samples, n_splits=5)
    groups = subject_groups(samples)

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

    rows: list[dict[str, object]] = []
    qc_usable_rows: list[dict[str, object]] = []
    exported_sample_ids: list[str] = []
    for fold in folds:
        assert_no_subject_leakage(groups, fold.train_indices, fold.test_indices)
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
            f"explaining {len(fold.test_indices)} held-out subjects..."
        )

        for sample_index_value in fold.test_indices:
            sample_index = int(sample_index_value)
            prepared = prepared_samples[sample_index]
            alignment = interpretable_dtw(prepared.vectors, reference.vectors)
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

            if sample_frame_quality.quality_status != "fail":
                qc_prepared = qc_prepared_samples[sample_index]
                qc_alignment = interpretable_dtw(
                    qc_prepared.vectors,
                    qc_reference.vectors,
                )
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

    if len(exported_sample_ids) != len(samples):
        raise AssertionError(
            f"Expected {len(samples)} held-out samples; got {len(exported_sample_ids)}"
        )
    if len(set(exported_sample_ids)) != len(samples):
        raise AssertionError("Each sample must be exported from exactly one test fold")
    expected_rows = len(samples) * FEATURE_DIMENSIONS
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} component rows; got {len(rows)}")
    qc_usable_samples = sum(
        status != "fail" for status in sample_quality_statuses
    )
    expected_qc_rows = qc_usable_samples * FEATURE_DIMENSIONS
    if len(qc_usable_rows) != expected_qc_rows:
        raise AssertionError(
            f"Expected {expected_qc_rows} frame-QC component rows; "
            f"got {len(qc_usable_rows)}"
        )

    sample_order = {sample.sample_id: index for index, sample in enumerate(samples)}
    rows.sort(
        key=lambda row: (
            sample_order[str(row["sample_id"])],
            int(row["component_index"]),
        )
    )
    output_path = output_dir / "component_summaries.csv"
    _write_csv(rows, output_path)
    progress(f"Component summaries: {output_path.resolve()}")

    if not qc_usable_rows:
        raise ValueError("Full-body QC rejected every interpretable DTW sample")
    qc_usable_rows.sort(
        key=lambda row: (
            sample_order[str(row["sample_id"])],
            int(row["component_index"]),
        )
    )
    qc_usable_output_path = output_dir / "component_summaries_qc_usable.csv"
    _write_csv(qc_usable_rows, qc_usable_output_path)
    progress(f"QC-usable component summaries: {qc_usable_output_path.resolve()}")

    error_figure_path = figure_dir / "component_error_distributions.png"
    contribution_figure_path = figure_dir / "component_contribution_distributions.png"
    qc_error_figure_path = figure_dir / "component_error_distributions_qc_usable.png"
    qc_contribution_figure_path = (
        figure_dir / "component_contribution_distributions_qc_usable.png"
    )
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
    progress(f"Component error figure: {error_figure_path.resolve()}")
    progress(f"Component contribution figure: {contribution_figure_path.resolve()}")
    progress(f"QC-usable component error figure: {qc_error_figure_path.resolve()}")
    progress(
        "QC-usable component contribution figure: "
        f"{qc_contribution_figure_path.resolve()}"
    )

    return {
        "method": METHOD_NAME,
        "exercise": exercise,
        "samples": len(samples),
        "unique_subjects": len(set(groups)),
        "folds": len(folds),
        "excluded_samples": excluded,
        "subject_overlap_in_every_fold": 0,
        "component_rows": len(rows),
        "qc_usable_samples": qc_usable_samples,
        "qc_usable_component_rows": len(qc_usable_rows),
        "interpolated_frames": sum(
            result.interpolated_frames for result in frame_quality
        ),
        "interpolated_component_frames": sum(
            result.interpolated_component_frames for result in frame_quality
        ),
        "dropped_frames": sum(
            result.dropped_frames for result in frame_quality
        ),
        "components_per_sample": FEATURE_DIMENSIONS,
        "output": str(output_path.resolve()),
        "qc_usable_output": str(qc_usable_output_path.resolve()),
        "component_quality_output": str(quality_path.resolve()),
        "quality_counts": quality_counts,
        "figures": [
            str(error_figure_path.resolve()),
            str(contribution_figure_path.resolve()),
            str(qc_error_figure_path.resolve()),
            str(qc_contribution_figure_path.resolve()),
        ],
    }
