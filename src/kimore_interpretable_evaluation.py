"""Subject-disjoint CSV export for interpretable KIMORE DTW alignments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from kimore_dataset import read_manifest
from kimore_grouping import assert_no_subject_leakage, make_subject_folds, subject_groups
from kimore_interpretable_dtw import (
    interpretable_dtw,
    summarize_component_errors,
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
        f"Loading and extracting interpretable DTW vectors for {len(samples)} "
        f"usable {exercise} recordings..."
    )
    prepared_samples: list[YuXiongPreparedSample] = []
    for number, sample in enumerate(samples, start=1):
        prepared_samples.append(prepare_yu_xiong_sample(sample))
        if number % 10 == 0 or number == len(samples):
            progress(f"  prepared {number}/{len(samples)}")

    rows: list[dict[str, object]] = []
    exported_sample_ids: list[str] = []
    for fold in folds:
        assert_no_subject_leakage(groups, fold.train_indices, fold.test_indices)
        reference_index = select_yu_xiong_reference(
            prepared_samples,
            fold.train_indices,
        )
        reference = prepared_samples[reference_index]
        progress(
            f"Fold {fold.number}: reference {reference.sample.sample_id}; "
            f"explaining {len(fold.test_indices)} held-out subjects..."
        )

        for sample_index_value in fold.test_indices:
            sample_index = int(sample_index_value)
            prepared = prepared_samples[sample_index]
            alignment = interpretable_dtw(prepared.vectors, reference.vectors)
            summaries = summarize_component_errors(alignment)
            exported_sample_ids.append(prepared.sample.sample_id)

            for component_index, component in enumerate(summaries):
                rows.append(
                    {
                        "fold": fold.number,
                        "sample_id": prepared.sample.sample_id,
                        "subject_id": prepared.sample.subject_id,
                        "cohort": prepared.sample.cohort,
                        "actual_ts": prepared.sample.score,
                        "reference_sample_id": reference.sample.sample_id,
                        "reference_subject_id": reference.sample.subject_id,
                        "reference_actual_ts": reference.sample.score,
                        "component_index": component_index,
                        "component_name": component.name,
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

    error_figure_path = figure_dir / "component_error_distributions.png"
    contribution_figure_path = figure_dir / "component_contribution_distributions.png"
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
    progress(f"Component error figure: {error_figure_path.resolve()}")
    progress(f"Component contribution figure: {contribution_figure_path.resolve()}")

    return {
        "method": METHOD_NAME,
        "exercise": exercise,
        "samples": len(samples),
        "unique_subjects": len(set(groups)),
        "folds": len(folds),
        "excluded_samples": excluded,
        "subject_overlap_in_every_fold": 0,
        "component_rows": len(rows),
        "components_per_sample": FEATURE_DIMENSIONS,
        "output": str(output_path.resolve()),
        "figures": [
            str(error_figure_path.resolve()),
            str(contribution_figure_path.resolve()),
        ],
    }
