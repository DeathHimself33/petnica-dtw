from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset import JOINT_INDEX, JOINT_NAMES  # noqa: E402
from kimore_evaluation import run_cross_validated_evaluation  # noqa: E402
from kimore_interpretable_evaluation import run_interpretable_evaluation  # noqa: E402
from kimore_yu_xiong_evaluation import run_yu_xiong_evaluation  # noqa: E402
from run_experiment import main as run_experiment_main  # noqa: E402


def write_synthetic_recording(path: Path, yaw_offset: float) -> int:
    yaw_values = np.asarray([0.0, 12.0, 0.0, -12.0, 0.0]) + yaw_offset
    rows: list[list[float]] = []
    for yaw in yaw_values:
        positions = np.zeros((len(JOINT_NAMES), 3), dtype=float)
        positions[:, 1] = 0.5
        positions[JOINT_INDEX["SpineBase"]] = [0.0, 0.0, 0.0]
        positions[JOINT_INDEX["SpineShoulder"]] = [0.0, 1.0, 0.0]
        radians = math.radians(float(yaw))
        shoulder_axis = np.asarray([math.cos(radians), 0.0, math.sin(radians)])
        positions[JOINT_INDEX["ShoulderLeft"]] = (
            np.asarray([0.0, 1.0, 0.0]) - shoulder_axis / 2.0
        )
        positions[JOINT_INDEX["ShoulderRight"]] = (
            np.asarray([0.0, 1.0, 0.0]) + shoulder_axis / 2.0
        )
        states = np.full((len(JOINT_NAMES), 1), 2.0)
        rows.append(np.concatenate((positions, states), axis=1).reshape(-1).tolist())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return len(rows)


def write_synthetic_full_body_recording(path: Path, yaw_offset: float) -> int:
    yaw_values = np.asarray([0.0, 8.0, 0.0, -8.0, 0.0]) + yaw_offset
    rows: list[list[float]] = []
    for yaw in yaw_values:
        positions = np.zeros((len(JOINT_NAMES), 3), dtype=float)
        positions[JOINT_INDEX["SpineBase"]] = [0.0, 0.75, 0.0]
        positions[JOINT_INDEX["SpineShoulder"]] = [0.0, 1.35, 0.0]
        positions[JOINT_INDEX["ShoulderLeft"]] = [-0.4, 1.35, 0.0]
        positions[JOINT_INDEX["ElbowLeft"]] = [-0.72, 1.02, 0.08]
        positions[JOINT_INDEX["WristLeft"]] = [-0.86, 0.68, 0.18]
        positions[JOINT_INDEX["ShoulderRight"]] = [0.4, 1.35, 0.0]
        positions[JOINT_INDEX["ElbowRight"]] = [0.72, 1.02, -0.08]
        positions[JOINT_INDEX["WristRight"]] = [0.86, 0.68, -0.18]
        positions[JOINT_INDEX["HipLeft"]] = [-0.24, 0.75, 0.0]
        positions[JOINT_INDEX["KneeLeft"]] = [-0.25, 0.38, 0.06]
        positions[JOINT_INDEX["AnkleLeft"]] = [-0.25, 0.04, 0.11]
        positions[JOINT_INDEX["HipRight"]] = [0.24, 0.75, 0.0]
        positions[JOINT_INDEX["KneeRight"]] = [0.25, 0.38, -0.06]
        positions[JOINT_INDEX["AnkleRight"]] = [0.25, 0.04, -0.11]

        radians = math.radians(float(yaw))
        rotation = np.asarray(
            [
                [math.cos(radians), 0.0, math.sin(radians)],
                [0.0, 1.0, 0.0],
                [-math.sin(radians), 0.0, math.cos(radians)],
            ]
        )
        positions = positions @ rotation.T
        states = np.full((len(JOINT_NAMES), 1), 2.0)
        rows.append(np.concatenate((positions, states), axis=1).reshape(-1).tolist())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return len(rows)


class EndToEndExperimentTests(unittest.TestCase):
    def test_all_exercise_run_uses_shared_folds_and_combines_queues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rows: list[dict[str, object]] = []
            for exercise in ("Es1", "Es2", "Es3", "Es4", "Es5"):
                for number in range(5):
                    subject_id = f"SYN_ID{number + 1}"
                    sample_id = f"{subject_id}_{exercise}"
                    position_path = root / f"{sample_id}.csv"
                    frames = write_synthetic_full_body_recording(
                        position_path,
                        yaw_offset=float(number * 4),
                    )
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "cohort": "synthetic",
                            "exercise": exercise,
                            "clinical_ts": 30.0 + number,
                            "position_frames": frames,
                            "position_columns": 100,
                            "position_path": str(position_path),
                            "issues": "",
                        }
                    )

            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output_root = root / "results"
            figure_root = root / "figures"
            exit_code = run_experiment_main(
                [
                    "--manifest",
                    str(manifest),
                    "--exercise",
                    "all",
                    "--method",
                    "interpretable_dtw",
                    "--bootstrap-resamples",
                    "2",
                    "--output-dir",
                    str(output_root),
                    "--figure-dir",
                    str(figure_root),
                ]
            )

            self.assertEqual(exit_code, 0)
            batch_summary = json.loads(
                (output_root / "all_exercises_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(batch_summary["unique_subjects"], 5)
            self.assertEqual(
                batch_summary["fold_strategy"],
                "shared_subject_assignment_across_exercises",
            )
            self.assertEqual(
                set(batch_summary["exercise_results"]),
                {"Es1", "Es2", "Es3", "Es4", "Es5"},
            )

            with (
                output_root / "annotation_queue_all_exercises.csv"
            ).open(encoding="utf-8", newline="") as handle:
                queue_rows = list(csv.DictReader(handle))
            self.assertEqual(len(queue_rows), 125)
            self.assertEqual(
                {row["exercise"] for row in queue_rows},
                {"Es1", "Es2", "Es3", "Es4", "Es5"},
            )
            self.assertEqual(len({row["candidate_id"] for row in queue_rows}), 125)

            for exercise in ("Es1", "Es2", "Es3", "Es4", "Es5"):
                summary = json.loads(
                    (
                        output_root / exercise / "evaluation_summary.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    summary["fold_strategy"],
                    "shared_subject_assignment_across_exercises",
                )

    def test_small_manifest_runs_through_all_five_folds_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rows: list[dict[str, object]] = []
            for number in range(5):
                sample_id = f"SYN_ID{number + 1}_Es3"
                position_path = root / f"{sample_id}.csv"
                frames = write_synthetic_recording(
                    position_path,
                    yaw_offset=float(number * 3),
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": f"SYN_ID{number + 1}",
                        "cohort": "synthetic",
                        "exercise": "Es3",
                        "clinical_ts": 30.0 + number,
                        "position_frames": frames,
                        "position_columns": 100,
                        "position_path": str(position_path),
                        "issues": "",
                    }
                )

            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output_dir = root / "results"
            figure_dir = root / "figures"
            summary = run_cross_validated_evaluation(
                manifest_path=manifest,
                exercise="Es3",
                output_dir=output_dir,
                figure_dir=figure_dir,
                bootstrap_resamples=10,
                progress=lambda _: None,
            )

            self.assertEqual(summary["samples"], 5)
            self.assertEqual(summary["oof_prediction_rows"], 5)
            self.assertEqual(summary["subject_overlap_in_every_fold"], 0)
            self.assertTrue((output_dir / "oof_predictions.csv").is_file())
            self.assertTrue((output_dir / "metrics.csv").is_file())
            self.assertTrue((output_dir / "fold_metadata.json").is_file())
            self.assertTrue((figure_dir / "actual_vs_predicted.png").is_file())
            saved = json.loads(
                (output_dir / "evaluation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["experiment_inputs_sha256"], summary["experiment_inputs_sha256"])

    def test_yu_xiong_baseline_runs_through_all_folds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rows: list[dict[str, object]] = []
            for number in range(5):
                sample_id = f"SYN_ID{number + 1}_Es3"
                position_path = root / f"{sample_id}.csv"
                frames = write_synthetic_full_body_recording(
                    position_path, yaw_offset=float(number * 4)
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": f"SYN_ID{number + 1}",
                        "cohort": "synthetic",
                        "exercise": "Es3",
                        "clinical_ts": 30.0 + number,
                        "position_frames": frames,
                        "position_columns": 100,
                        "position_path": str(position_path),
                        "issues": "",
                    }
                )

            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output_dir = root / "results"
            figure_dir = root / "figures"
            summary = run_yu_xiong_evaluation(
                manifest_path=manifest,
                exercise="Es3",
                output_dir=output_dir,
                figure_dir=figure_dir,
                bootstrap_resamples=10,
                progress=lambda _: None,
            )

            self.assertEqual(summary["method"], "yu_xiong_dtw")
            self.assertEqual(summary["oof_prediction_rows"], 5)
            self.assertEqual(summary["subject_overlap_in_every_fold"], 0)
            self.assertTrue((output_dir / "oof_predictions.csv").is_file())
            self.assertTrue((output_dir / "metrics.csv").is_file())
            self.assertTrue((output_dir / "fold_metadata.json").is_file())
            self.assertTrue((figure_dir / "actual_vs_predicted.png").is_file())

    def test_interpretable_dtw_exports_nine_rows_per_held_out_subject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rows: list[dict[str, object]] = []
            for number in range(5):
                sample_id = f"SYN_ID{number + 1}_Es3"
                position_path = root / f"{sample_id}.csv"
                frames = write_synthetic_full_body_recording(
                    position_path, yaw_offset=float(number * 4)
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": f"SYN_ID{number + 1}",
                        "cohort": "synthetic",
                        "exercise": "Es3",
                        "clinical_ts": 30.0 + number,
                        "position_frames": frames,
                        "position_columns": 100,
                        "position_path": str(position_path),
                        "issues": "",
                    }
                )

            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output_dir = root / "results"
            figure_dir = root / "figures"
            summary = run_interpretable_evaluation(
                manifest_path=manifest,
                exercise="Es3",
                output_dir=output_dir,
                figure_dir=figure_dir,
                bootstrap_resamples=10,
                progress=lambda _: None,
            )

            self.assertEqual(summary["method"], "interpretable_dtw")
            self.assertEqual(summary["exercise"], "Es3")
            self.assertEqual(
                summary["fold_strategy"],
                "exercise_specific_subject_assignment",
            )
            self.assertEqual(summary["samples"], 5)
            self.assertEqual(summary["component_rows"], 45)
            self.assertEqual(summary["qc_usable_samples"], 5)
            self.assertEqual(summary["qc_usable_component_rows"], 45)
            self.assertEqual(summary["oof_prediction_rows"], 5)
            self.assertEqual(summary["qc_coverage_fraction"], 1.0)
            self.assertEqual(summary["error_timeline_rows"], 225)
            self.assertEqual(summary["top_deviation_interval_rows"], 25)
            self.assertEqual(summary["annotation_queue_rows"], 25)
            self.assertEqual(summary["interpolated_frames"], 0)
            self.assertEqual(summary["interpolated_component_frames"], 0)
            self.assertEqual(summary["dropped_frames"], 0)
            self.assertEqual(
                summary["quality_counts"],
                {"pass": 5, "warning": 0, "fail": 0},
            )
            self.assertEqual(summary["subject_overlap_in_every_fold"], 0)
            self.assertEqual(len(summary["figures"]), 5)
            self.assertTrue((output_dir / "component_quality.csv").is_file())
            self.assertTrue((output_dir / "oof_predictions.csv").is_file())
            self.assertTrue((output_dir / "metrics.csv").is_file())
            self.assertTrue((output_dir / "fold_metadata.json").is_file())
            self.assertTrue((output_dir / "evaluation_summary.json").is_file())
            self.assertTrue((output_dir / "error_timeline.csv").is_file())
            self.assertTrue(
                (output_dir / "top_deviation_intervals.csv").is_file()
            )
            self.assertTrue((output_dir / "annotation_queue.csv").is_file())
            self.assertTrue(
                (output_dir / "component_summaries_qc_usable.csv").is_file()
            )
            self.assertTrue(
                (figure_dir / "component_error_distributions.png").is_file()
            )
            self.assertTrue(
                (figure_dir / "component_contribution_distributions.png").is_file()
            )
            self.assertTrue(
                (
                    figure_dir
                    / "component_error_distributions_qc_usable.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    figure_dir
                    / "component_contribution_distributions_qc_usable.png"
                ).is_file()
            )
            self.assertTrue(
                (figure_dir / "actual_vs_predicted_qc.png").is_file()
            )

            with (output_dir / "oof_predictions.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                prediction_rows = list(csv.DictReader(handle))
            self.assertEqual(len(prediction_rows), 5)
            self.assertEqual({row["exercise"] for row in prediction_rows}, {"Es3"})
            self.assertEqual(
                {row["sample_quality_status"] for row in prediction_rows},
                {"pass"},
            )
            self.assertTrue(
                all(row["qc_predicted_ts"] for row in prediction_rows)
            )
            self.assertTrue(
                all(row["raw_predicted_ts"] for row in prediction_rows)
            )

            with (output_dir / "error_timeline.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                timeline_rows = list(csv.DictReader(handle))
            self.assertEqual(len(timeline_rows), 225)
            self.assertEqual({row["exercise"] for row in timeline_rows}, {"Es3"})
            self.assertEqual(
                {row["interpretation"] for row in timeline_rows},
                {
                    "candidate_deviation_from_reference_not_validated_as_"
                    "execution_error"
                },
            )

            with (output_dir / "annotation_queue.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                annotation_rows = list(csv.DictReader(handle))
            self.assertEqual(len(annotation_rows), 25)
            self.assertEqual({row["exercise"] for row in annotation_rows}, {"Es3"})
            self.assertEqual(len({row["candidate_id"] for row in annotation_rows}), 25)
            self.assertEqual(
                {row["review_status"] for row in annotation_rows},
                {"unreviewed"},
            )
            self.assertEqual(
                {row["execution_label"] for row in annotation_rows},
                {""},
            )

            with (output_dir / "component_summaries.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                component_rows = list(csv.DictReader(handle))
            self.assertEqual(len(component_rows), 45)
            self.assertEqual(
                {row["input_variant"] for row in component_rows},
                {"raw"},
            )
            self.assertEqual(
                {row["sample_id"] for row in component_rows},
                {row["sample_id"] for row in rows},
            )
            for sample_id in {row["sample_id"] for row in rows}:
                sample_rows = [
                    row for row in component_rows if row["sample_id"] == sample_id
                ]
                self.assertEqual(len(sample_rows), 9)
                self.assertEqual(
                    [int(row["component_index"]) for row in sample_rows],
                    list(range(9)),
                )

            with (output_dir / "component_summaries_qc_usable.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                qc_component_rows = list(csv.DictReader(handle))
            self.assertEqual(len(qc_component_rows), 45)
            self.assertEqual(
                {row["input_variant"] for row in qc_component_rows},
                {"frame_qc"},
            )
            self.assertEqual(
                {row["sample_frames_used"] for row in qc_component_rows},
                {"5"},
            )

            with (output_dir / "component_quality.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                quality_rows = list(csv.DictReader(handle))
            self.assertEqual(len(quality_rows), 45)
            self.assertEqual(
                {row["component_quality_status"] for row in quality_rows},
                {"pass"},
            )


if __name__ == "__main__":
    unittest.main()
