"""Run reproducible single- or all-exercise KIMORE experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SUPPORTED_EXERCISES = tuple(f"Es{index}" for index in range(1, 6))
ALL_EXERCISES = "all"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "kimore_audit_output" / "kimore_manifest.csv",
        help="Audit manifest (default: ./kimore_audit_output/kimore_manifest.csv)",
    )
    parser.add_argument(
        "--exercise",
        default="Es3",
        choices=[*SUPPORTED_EXERCISES, ALL_EXERCISES],
        help=(
            "KIMORE exercise, or 'all' for the QC/interpretable pipeline; "
            "plain_dtw is defined only for Es3"
        ),
    )
    parser.add_argument(
        "--method",
        default="plain_dtw",
        choices=["plain_dtw", "yu_xiong_dtw", "interpretable_dtw"],
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=5000,
        help=(
            "Conditional fixed-OOF-prediction subject-bootstrap resamples "
            "(default: 5000)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Single-run output directory, or all-exercise output root "
            "(default: ./results/<method>/<exercise>)"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help=(
            "Single-run figure directory, or all-exercise figure root "
            "(default: ./figures/<method>/<exercise>)"
        ),
    )
    args = parser.parse_args(argv)
    if args.exercise == ALL_EXERCISES and args.method != "interpretable_dtw":
        parser.error("--exercise all requires interpretable_dtw")
    return args


def _evaluator(method: str) -> Callable[..., dict[str, object]]:
    if method == "interpretable_dtw":
        from kimore_interpretable_evaluation import run_interpretable_evaluation

        return run_interpretable_evaluation
    if method == "plain_dtw":
        from kimore_evaluation import run_cross_validated_evaluation

        return run_cross_validated_evaluation
    from kimore_yu_xiong_evaluation import run_yu_xiong_evaluation

    return run_yu_xiong_evaluation


def _shared_fold_assignments(manifest_path: Path) -> dict[str, int]:
    """Build one subject split from every usable recording in Es1--Es5."""
    from kimore_dataset import read_manifest
    from kimore_grouping import make_subject_fold_assignments

    samples = []
    for exercise in SUPPORTED_EXERCISES:
        exercise_samples, _ = read_manifest(manifest_path, exercise)
        samples.extend(exercise_samples)
    return make_subject_fold_assignments(samples, n_splits=5)


def _run_root(
    requested: Path | None,
    default_parent: Path,
    exercise: str,
) -> Path:
    """Resolve an isolated default root while preserving explicit custom paths."""
    if requested is not None:
        return requested.expanduser().resolve()
    suffix = "all_exercises" if exercise == ALL_EXERCISES else exercise
    return (default_parent / suffix).resolve()


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _combine_csv_files(paths: list[Path], output_path: Path) -> int:
    """Combine exercise CSVs after verifying that their schemas are identical."""
    fieldnames: list[str] | None = None
    combined_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = None
        for path in paths:
            with path.open("r", encoding="utf-8-sig", newline="") as input_handle:
                reader = csv.DictReader(input_handle)
                current_fields = list(reader.fieldnames or ())
                if fieldnames is None:
                    fieldnames = current_fields
                    if "exercise" not in fieldnames:
                        raise ValueError(f"Combined CSV has no exercise column: {path}")
                    writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
                    writer.writeheader()
                elif current_fields != fieldnames:
                    raise ValueError(
                        f"Cannot combine different CSV schemas: {paths[0]} and {path}"
                    )
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    combined_rows += 1
    if fieldnames is None:
        raise ValueError("No CSV inputs were provided")
    return combined_rows


def _fold_payload(assignments: Mapping[str, int]) -> dict[str, object]:
    return {
        "fold_numbering": "one_based",
        "strategy": "shared_subject_assignment_across_exercises",
        "folds": [
            {
                "fold": fold_index + 1,
                "subjects": sorted(
                    subject
                    for subject, assigned_fold in assignments.items()
                    if assigned_fold == fold_index
                ),
            }
            for fold_index in range(5)
        ],
    }


def _batch_summary_entry(
    summary: Mapping[str, object],
    summary_path: Path,
) -> dict[str, object]:
    metrics = summary.get("overall_frame_qc_yu_xiong_dtw")
    if metrics is None:
        metrics = summary.get("overall_yu_xiong_dtw")
    entry: dict[str, object] = {
        "samples": summary["samples"],
        "unique_subjects": summary["unique_subjects"],
        "excluded_samples": summary["excluded_samples"],
        "metrics": metrics,
        "summary": str(summary_path.resolve()),
    }
    if "qc_coverage_fraction" in summary:
        entry["qc_coverage_fraction"] = summary["qc_coverage_fraction"]
    return entry


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    batch = args.exercise == ALL_EXERCISES
    exercises = SUPPORTED_EXERCISES if batch else (args.exercise,)
    evaluator = _evaluator(args.method)
    output_root = _run_root(
        args.output_dir,
        PROJECT_ROOT / "results" / args.method,
        args.exercise,
    )
    figure_root = _run_root(
        args.figure_dir,
        PROJECT_ROOT / "figures" / args.method,
        args.exercise,
    )
    shared_assignments = _shared_fold_assignments(manifest_path) if batch else None
    summaries: dict[str, dict[str, object]] = {}

    if shared_assignments is not None:
        _write_json(
            _fold_payload(shared_assignments),
            output_root / "subject_folds.json",
        )

    for exercise in exercises:
        output_dir = output_root / exercise if batch else output_root
        figure_dir = figure_root / exercise if batch else figure_root
        summaries[exercise] = evaluator(
            manifest_path=manifest_path,
            exercise=exercise,
            output_dir=output_dir,
            figure_dir=figure_dir,
            bootstrap_resamples=args.bootstrap_resamples,
            subject_fold_assignments=shared_assignments,
        )

    if not batch:
        return 0

    combined_outputs: dict[str, object] = {}
    csv_names = ["oof_predictions.csv"]
    if args.method == "interpretable_dtw":
        csv_names.extend(
            (
                "annotation_queue.csv",
                "top_deviation_intervals.csv",
                "component_quality.csv",
            )
        )
    for csv_name in csv_names:
        stem = Path(csv_name).stem
        combined_path = output_root / f"{stem}_all_exercises.csv"
        row_count = _combine_csv_files(
            [output_root / exercise / csv_name for exercise in exercises],
            combined_path,
        )
        combined_outputs[stem] = {
            "path": str(combined_path.resolve()),
            "rows": row_count,
        }

    index_path = output_root / "all_exercises_summary.json"
    _write_json(
        {
            "method": args.method,
            "exercises": list(exercises),
            "fold_strategy": "shared_subject_assignment_across_exercises",
            "unique_subjects": len(shared_assignments or ()),
            "subject_folds": str((output_root / "subject_folds.json").resolve()),
            "exercise_results": {
                exercise: _batch_summary_entry(
                    summaries[exercise],
                    output_root / exercise / "evaluation_summary.json",
                )
                for exercise in exercises
            },
            "combined_outputs": combined_outputs,
        },
        index_path,
    )
    print(f"All-exercise summary: {index_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
