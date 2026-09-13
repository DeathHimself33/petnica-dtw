from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kimore_expert_review import create_app  # noqa: E402


QUEUE_FIELDS = [
    "fold",
    "sample_id",
    "subject_id",
    "cohort",
    "actual_ts",
    "sample_quality_status",
    "reference_sample_id",
    "reference_subject_id",
    "window_index",
    "window_start_percent",
    "window_end_percent",
    "original_frame_start",
    "original_frame_end",
    "reference_original_frame_start",
    "reference_original_frame_end",
    "component_index",
    "component_name",
    "mean_angular_deviation_degrees",
    "maximum_angular_deviation_degrees",
    "whole_alignment_component_mean_degrees",
    "excess_over_component_mean_degrees",
    "aligned_path_steps",
    "sample_component_interpolated_fraction",
    "reference_component_interpolated_fraction",
    "interpretation",
    "candidate_rank",
    "review_status",
    "execution_label",
    "error_type",
    "severity",
    "reviewer_confidence",
    "annotator",
    "review_notes",
    "second_review_selection",
]


class ExpertReviewAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sheets = self.root / "sheets"
        self.sheets.mkdir()
        self.queue = self.root / "queue.csv"
        self.primary = self.root / "primary.csv"
        self.database = self.root / "review.sqlite3"
        queue_rows = []
        primary_rows = []
        for index in range(1, 3):
            sample_id = f"S{index}_Es3"
            row = {field: "" for field in QUEUE_FIELDS}
            row.update(
                {
                    "fold": str(index),
                    "sample_id": sample_id,
                    "subject_id": f"S{index}",
                    "cohort": "hidden_cohort",
                    "actual_ts": "42.0",
                    "reference_sample_id": "REF_Es3",
                    "window_start_percent": str(index * 5),
                    "window_end_percent": str(index * 5 + 5),
                    "component_name": "right_upper_arm",
                    "mean_angular_deviation_degrees": "99.9",
                    "candidate_rank": str(index),
                    "second_review_selection": "synthetic_test",
                }
            )
            queue_rows.append(row)
            primary_rows.append(
                {
                    **row,
                    "sample_id": sample_id,
                    "candidate_rank": str(index),
                    "review_status": "reviewed",
                    "execution_label": "correct",
                    "error_type": "",
                    "severity": "",
                    "reviewer_confidence": "high",
                    "annotator": "primary_reviewer",
                    "review_notes": "SECRET PRIMARY NOTE",
                }
            )
            image = Image.new("RGB", (1916, 1320), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1915, 77), fill=(220, 0, 0))
            draw.rectangle((0, 78, 439, 1319), fill=(0, 0, 220))
            draw.rectangle((440, 78, 1915, 409), fill=(0, 180, 0))
            for rank in range(1, 6):
                top = 410 + (rank - 1) * 178
                draw.rectangle(
                    (440, top, 1915, top + 177),
                    fill=(40 * rank, 100, 150),
                )
            image.save(self.sheets / f"{sample_id}.jpg")
        self._write_csv(self.queue, QUEUE_FIELDS, queue_rows)
        self._write_csv(self.primary, list(primary_rows[0]), primary_rows)
        self.app = create_app(
            queue_path=self.queue,
            primary_labels_path=self.primary,
            sheets_dir=self.sheets,
            database_path=self.database,
            testing=True,
            secret_key="test-secret",
        )
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _csrf(self) -> str:
        with self.client.session_transaction() as state:
            token = state.get("csrf_token", "test-csrf")
            state["csrf_token"] = token
        return token

    def _start(self, reviewer_id: str = "expert_02") -> None:
        response = self.client.post(
            "/session",
            data={"csrf_token": self._csrf(), "reviewer_id": reviewer_id},
        )
        self.assertEqual(response.status_code, 302)

    def _save_label(
        self,
        position: int,
        label: str = "correct",
        *,
        error_type: str = "",
        severity: str = "",
        notes: str = "Visible evidence checked.",
    ) -> None:
        response = self.client.post(
            f"/review/{position}",
            data={
                "csrf_token": self._csrf(),
                "execution_label": label,
                "error_type": error_type,
                "severity": severity,
                "reviewer_confidence": "high",
                "review_notes": notes,
            },
        )
        self.assertEqual(response.status_code, 302)

    def test_review_page_and_evidence_hide_primary_and_score_fields(self) -> None:
        self.assertEqual(self.client.get("/api/progress").status_code, 401)
        self._start()
        progress = self.client.get("/api/progress")
        self.assertEqual(progress.status_code, 200)
        self.assertEqual(progress.get_json()["total"], 2)
        self.assertEqual(progress.get_json()["reviewed"], 0)
        response = self.client.get("/review/1")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"42.0", response.data)
        self.assertNotIn(b"hidden_cohort", response.data)
        self.assertNotIn(b"99.9", response.data)
        self.assertNotIn(b"SECRET PRIMARY NOTE", response.data)
        self.assertNotIn(b"primary_reviewer", response.data)

        evidence = self.client.get("/evidence/1.jpg")
        self.assertEqual(evidence.status_code, 200)
        with Image.open(io.BytesIO(evidence.data)) as image:
            self.assertEqual(image.size, (1476, 526))
            overview_pixel = image.getpixel((10, 10))
            self.assertLess(overview_pixel[0], 20)
            self.assertGreater(overview_pixel[1], 160)
            self.assertLess(overview_pixel[2], 20)

    def test_complete_review_is_sealed_before_agreement_and_exports(self) -> None:
        self._start()
        self._save_label(1)
        self._save_label(2)
        self.assertEqual(self.client.get("/agreement").status_code, 403)

        finished = self.client.post("/finish", data={"csrf_token": self._csrf()})
        self.assertEqual(finished.status_code, 302)
        agreement = self.client.get("/agreement")
        self.assertEqual(agreement.status_code, 200)
        self.assertIn("100%".encode(), agreement.data)

        edit = self.client.post(
            "/review/1",
            data={
                "csrf_token": self._csrf(),
                "execution_label": "correct",
                "reviewer_confidence": "high",
                "review_notes": "Changed after seal.",
            },
        )
        self.assertEqual(edit.status_code, 409)

        exported = self.client.get("/export/second-review.csv")
        rows = list(csv.DictReader(io.StringIO(exported.get_data(as_text=True))))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["annotator"] for row in rows}, {"expert_02"})

    def test_collect_only_keeps_primary_hidden_after_sealing(self) -> None:
        self.app = create_app(
            queue_path=self.queue,
            primary_labels_path=self.primary,
            sheets_dir=self.sheets,
            database_path=self.root / "collect-only.sqlite3",
            testing=True,
            secret_key="test-secret",
            collect_only=True,
        )
        self.client = self.app.test_client()
        self._start()
        self._save_label(1)
        self._save_label(2)

        finished = self.client.post(
            "/finish", data={"csrf_token": self._csrf()}
        )
        self.assertIn("/progress", finished.location)
        progress = self.client.get("/progress")
        self.assertIn("Preuzmi zaključani pregled".encode(), progress.data)
        self.assertNotIn(b"SECRET PRIMARY NOTE", progress.data)
        self.assertNotIn(b"primary_reviewer", progress.data)
        self.assertEqual(self.client.get("/agreement").status_code, 403)
        self.assertEqual(self.client.get("/adjudication").status_code, 403)
        self.assertEqual(self.client.get("/export/agreement.json").status_code, 403)
        self.assertEqual(self.client.get("/export/adjudicated.csv").status_code, 403)
        self.assertEqual(self.client.get("/export/second-review.csv").status_code, 200)

    def test_error_requires_type_and_severity(self) -> None:
        self._start()
        response = self.client.post(
            "/review/1",
            data={
                "csrf_token": self._csrf(),
                "execution_label": "error",
                "reviewer_confidence": "medium",
                "review_notes": "An error is visible.",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Izaberite tip greške".encode(), response.data)

    def test_disagreement_can_be_adjudicated_and_exported(self) -> None:
        self._start()
        self._save_label(
            1,
            "error",
            error_type="posture",
            severity="moderate",
        )
        self._save_label(2)
        self.client.post("/finish", data={"csrf_token": self._csrf()})
        agreement = self.client.get("/agreement")
        self.assertIn(b"Za adjudikaciju", agreement.data)

        start = self.client.post(
            "/adjudication",
            data={"csrf_token": self._csrf(), "adjudicator_id": "expert_03"},
        )
        self.assertEqual(start.status_code, 302)
        saved = self.client.post(
            "/adjudication/1",
            data={
                "csrf_token": self._csrf(),
                "execution_label": "correct",
                "error_type": "",
                "severity": "",
                "reviewer_confidence": "high",
                "review_notes": "Consensus review found no component error.",
            },
        )
        self.assertEqual(saved.status_code, 302)
        exported = self.client.get("/export/adjudicated.csv")
        rows = list(csv.DictReader(io.StringIO(exported.get_data(as_text=True))))
        completed = [row for row in rows if row["adjudication_required"] == "true"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["adjudication_complete"], "true")
        self.assertEqual(completed[0]["final_annotator"], "expert_03")


if __name__ == "__main__":
    unittest.main()
