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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset import JOINT_INDEX, JOINT_NAMES  # noqa: E402
from kimore_evaluation import run_cross_validated_evaluation  # noqa: E402


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


class EndToEndExperimentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
