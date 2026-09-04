"""Aggregate repeated KIMORE ML runs and compare them on identical OOF rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kimore_evaluation import (  # noqa: E402
    bootstrap_paired_metric_improvements,
    regression_metrics,
)


IDENTITY_FIELDS = (
    "sample_id",
    "subject_id",
    "cohort",
    "exercise",
    "fold",
    "actual_ts",
    "training_exercise_mean_ts",
    "quality_status",
    "retained_fraction",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Completed training output directory; repeat once per seed.",
    )
    parser.add_argument(
        "--dtw-predictions",
        type=Path,
        default=Path(
            "results/interpretable_dtw/all_exercises/"
            "oof_predictions_all_exercises.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ml_baseline/multiseed_analysis"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--score-max", type=float, default=50.0)
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(document: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_dict(
    rows: Sequence[dict[str, object]], prediction_field: str
) -> dict[str, float | None]:
    actual = np.asarray([float(row["actual_ts"]) for row in rows])
    predicted = np.asarray([float(row[prediction_field]) for row in rows])
    values = regression_metrics(actual, predicted)
    return {
        name: float(value) if np.isfinite(value) else None
        for name, value in values.items()
    }


def load_run(run_dir: Path) -> tuple[int, dict[str, object], list[dict[str, str]]]:
    resolved = run_dir.expanduser().resolve()
    summary_path = resolved / "summary.json"
    predictions_path = resolved / "oof_predictions.csv"
    if not summary_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(
            f"Run needs summary.json and oof_predictions.csv: {resolved}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completed = summary.get("completed_folds")
    if completed != [1, 2, 3, 4, 5]:
        raise ValueError(f"Run is not a complete five-fold result: {resolved}")
    seed = int(summary["configuration"]["seed"])
    rows = read_csv(predictions_path)
    if len(rows) != int(summary["oof_samples"]):
        raise ValueError(f"OOF row count disagrees with summary: {resolved}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Run repeats OOF sample IDs: {resolved}")
    return seed, summary, rows


def rows_by_sample(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in rows}


def validate_compatible_runs(
    reference: Sequence[dict[str, str]],
    candidate: Sequence[dict[str, str]],
    candidate_name: str,
) -> None:
    reference_map = rows_by_sample(reference)
    candidate_map = rows_by_sample(candidate)
    if set(reference_map) != set(candidate_map):
        raise ValueError(f"OOF sample population differs for {candidate_name}")
    for sample_id, expected in reference_map.items():
        observed = candidate_map[sample_id]
        for field in IDENTITY_FIELDS:
            if field not in expected or field not in observed:
                raise ValueError(f"Missing comparison field {field!r}")
            if field in {"actual_ts", "training_exercise_mean_ts", "retained_fraction"}:
                matches = np.isclose(
                    float(expected[field]), float(observed[field]), rtol=0.0, atol=1e-6
                )
            else:
                matches = expected[field] == observed[field]
            if not matches:
                raise ValueError(
                    f"OOF field {field!r} differs for {sample_id} in {candidate_name}"
                )


def build_ensemble_rows(
    runs: Sequence[tuple[int, list[dict[str, str]]]],
    dtw_rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    reference = runs[0][1]
    run_maps = [(seed, rows_by_sample(rows)) for seed, rows in runs]
    qc_dtw = {
        row["sample_id"]: row
        for row in dtw_rows
        if row.get("qc_predicted_ts", "") != ""
    }
    qc_dtw_count = sum(
        row.get("qc_predicted_ts", "") != "" for row in dtw_rows
    )
    if len(qc_dtw) != qc_dtw_count:
        raise ValueError("QC-DTW predictions repeat sample IDs")
    expected_ids = {row["sample_id"] for row in reference}
    if set(qc_dtw) != expected_ids:
        raise ValueError("QC-DTW and ML OOF sample populations differ")

    ensemble: list[dict[str, object]] = []
    ordered_reference = sorted(
        reference,
        key=lambda row: (int(row["fold"]), row["exercise"], row["sample_id"]),
    )
    for row in ordered_reference:
        sample_id = row["sample_id"]
        dtw_row = qc_dtw[sample_id]
        for field in ("subject_id", "exercise", "fold"):
            if dtw_row.get(field) != row[field]:
                raise ValueError(
                    f"QC-DTW field {field!r} differs for {sample_id}"
                )
        if not np.isclose(
            float(dtw_row["actual_ts"]),
            float(row["actual_ts"]),
            rtol=0.0,
            atol=1e-4,
        ):
            raise ValueError(f"QC-DTW target differs for {sample_id}")
        predictions = np.asarray(
            [float(run_map[sample_id]["predicted_ts"]) for _, run_map in run_maps]
        )
        predicted = float(np.mean(predictions))
        actual = float(row["actual_ts"])
        output: dict[str, object] = {
            field: row[field] for field in IDENTITY_FIELDS
        }
        for seed, run_map in run_maps:
            output[f"prediction_seed_{seed}"] = float(
                run_map[sample_id]["predicted_ts"]
            )
        output.update(
            {
                "ensemble_predicted_ts": predicted,
                "prediction_seed_std": float(np.std(predictions, ddof=1)),
                "ensemble_absolute_error": abs(predicted - actual),
                "qc_dtw_predicted_ts": float(dtw_row["qc_predicted_ts"]),
            }
        )
        ensemble.append(output)
    return ensemble


def scope_rows(
    rows: Sequence[dict[str, object]], scope: str
) -> list[dict[str, object]]:
    if scope == "overall":
        return list(rows)
    return [row for row in rows if row["exercise"] == scope]


def bootstrap_rows(
    rows: Sequence[dict[str, object]],
    scopes: Sequence[str],
    resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    comparators = {
        "qc_dtw": "qc_dtw_predicted_ts",
        "training_exercise_mean": "training_exercise_mean_ts",
    }
    for scope_index, scope in enumerate(scopes):
        selected = scope_rows(rows, scope)
        actual = np.asarray([float(row["actual_ts"]) for row in selected])
        candidate = np.asarray(
            [float(row["ensemble_predicted_ts"]) for row in selected]
        )
        groups = [str(row["subject_id"]) for row in selected]
        for comparator_index, (name, field) in enumerate(comparators.items()):
            comparison = np.asarray([float(row[field]) for row in selected])
            intervals = bootstrap_paired_metric_improvements(
                actual,
                candidate,
                comparison,
                groups,
                resamples=resamples,
                seed=seed + scope_index * 10 + comparator_index,
            )
            for metric, interval in intervals.items():
                output.append(
                    {
                        "scope": scope,
                        "comparator": name,
                        "metric": metric,
                        **interval,
                    }
                )
    return output


def seed_metric_rows(
    runs: Sequence[tuple[int, list[dict[str, str]]]], scopes: Sequence[str]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for seed, rows in runs:
        for scope in scopes:
            selected = scope_rows(rows, scope)
            output.append({"seed": seed, "scope": scope, **metric_dict(selected, "predicted_ts")})
    return output


def aggregate_seed_metrics(
    rows: Sequence[dict[str, object]], scopes: Sequence[str]
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for scope in scopes:
        selected = [row for row in rows if row["scope"] == scope]
        output[scope] = {}
        for metric in ("mae", "rmse", "spearman", "pearson"):
            values = np.asarray(
                [float(row[metric]) for row in selected if row[metric] is not None]
            )
            output[scope][metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return output


def population_coverage(
    dtw_path: Path, comparison_rows: int
) -> dict[str, int | float | None]:
    summary_path = dtw_path.with_name("all_exercises_summary.json")
    if not summary_path.is_file():
        return {
            "qc_usable_samples": comparison_rows,
            "full_samples": None,
            "qc_coverage_fraction": None,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    exercise_results = summary.get("exercise_results", {})
    full_samples = sum(
        int(result["samples"]) for result in exercise_results.values()
    )
    if full_samples < comparison_rows:
        raise ValueError("DTW summary population is smaller than its OOF CSV")
    return {
        "qc_usable_samples": comparison_rows,
        "full_samples": full_samples,
        "qc_coverage_fraction": comparison_rows / full_samples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.run_dir) < 2:
        raise ValueError("Multi-seed analysis needs at least two completed runs")
    if args.bootstrap_resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    if args.score_min >= args.score_max:
        raise ValueError("Score minimum must be smaller than score maximum")

    loaded = [load_run(path) for path in args.run_dir]
    seeds = [seed for seed, _, _ in loaded]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Every run must use a distinct random seed")
    reference_rows = loaded[0][2]
    for path, (_, _, rows) in zip(args.run_dir[1:], loaded[1:]):
        validate_compatible_runs(reference_rows, rows, str(path))

    dtw_path = args.dtw_predictions.expanduser().resolve()
    dtw_rows = read_csv(dtw_path)
    runs = [(seed, rows) for seed, _, rows in loaded]
    ensemble_rows = build_ensemble_rows(runs, dtw_rows)
    exercises = sorted({str(row["exercise"]) for row in ensemble_rows})
    scopes = ["overall", *exercises]
    seed_metrics = seed_metric_rows(runs, scopes)
    bootstrap = bootstrap_rows(
        ensemble_rows,
        scopes,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    output_dir = args.output_dir.expanduser().resolve()
    ensemble_path = output_dir / "ensemble_oof_predictions.csv"
    seed_metrics_path = output_dir / "seed_metrics.csv"
    bootstrap_path = output_dir / "bootstrap_intervals.csv"
    write_csv(ensemble_rows, ensemble_path)
    write_csv(seed_metrics, seed_metrics_path)
    write_csv(bootstrap, bootstrap_path)

    qc_rows = [row for row in dtw_rows if row.get("qc_predicted_ts", "") != ""]
    coverage = population_coverage(dtw_path, len(qc_rows))
    out_of_range = [
        row
        for row in ensemble_rows
        if not args.score_min
        <= float(row["ensemble_predicted_ts"])
        <= args.score_max
    ]
    disagreement = sorted(
        ensemble_rows,
        key=lambda row: float(row["prediction_seed_std"]),
        reverse=True,
    )
    errors = sorted(
        ensemble_rows,
        key=lambda row: float(row["ensemble_absolute_error"]),
        reverse=True,
    )
    summary: dict[str, object] = {
        "interpretation": (
            "Post-hoc multi-seed sensitivity analysis on fixed subject-disjoint "
            "OOF folds; it is not an independent external validation."
        ),
        "seeds": seeds,
        "runs": [
            {
                "path": str(path.expanduser().resolve()),
                "predictions_sha256": sha256(
                    path.expanduser().resolve() / "oof_predictions.csv"
                ),
            }
            for path in args.run_dir
        ],
        "dtw_predictions": str(dtw_path),
        "dtw_predictions_sha256": sha256(dtw_path),
        "samples": len(ensemble_rows),
        "unique_subjects": len({row["subject_id"] for row in ensemble_rows}),
        "population": coverage,
        "seed_metric_distribution": aggregate_seed_metrics(seed_metrics, scopes),
        "ensemble_metrics": {
            scope: metric_dict(scope_rows(ensemble_rows, scope), "ensemble_predicted_ts")
            for scope in scopes
        },
        "comparator_metrics": {
            scope: {
                "qc_dtw": metric_dict(scope_rows(ensemble_rows, scope), "qc_dtw_predicted_ts"),
                "training_exercise_mean": metric_dict(
                    scope_rows(ensemble_rows, scope), "training_exercise_mean_ts"
                ),
            }
            for scope in scopes
        },
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "group": "subject_id",
            "positive_values_favor": "ensemble",
        },
        "score_range": {
            "minimum": args.score_min,
            "maximum": args.score_max,
            "ensemble_predictions_outside": len(out_of_range),
        },
        "largest_ensemble_errors": [
            {
                "sample_id": row["sample_id"],
                "exercise": row["exercise"],
                "actual_ts": row["actual_ts"],
                "ensemble_predicted_ts": row["ensemble_predicted_ts"],
                "absolute_error": row["ensemble_absolute_error"],
            }
            for row in errors[:10]
        ],
        "largest_seed_disagreements": [
            {
                "sample_id": row["sample_id"],
                "exercise": row["exercise"],
                "prediction_seed_std": row["prediction_seed_std"],
            }
            for row in disagreement[:10]
        ],
        "outputs": {
            "ensemble_predictions": str(ensemble_path),
            "seed_metrics": str(seed_metrics_path),
            "bootstrap_intervals": str(bootstrap_path),
        },
    }
    summary_path = output_dir / "multiseed_summary.json"
    write_json(summary, summary_path)
    overall = summary["ensemble_metrics"]["overall"]
    print(
        "Ensemble: "
        f"MAE={overall['mae']:.3f}, RMSE={overall['rmse']:.3f}, "
        f"Pearson={overall['pearson']:.3f}, Spearman={overall['spearman']:.3f}"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
