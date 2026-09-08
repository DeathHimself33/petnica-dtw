"""Regression coverage for the reproduced project audit failures."""
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kimore_run_provenance import prepare_run, validate_population, complete_fold, verify_fold
from kimore_candidate_identity import validate_identity
from kimore_ml_data import build_ml_dataset
from kimore_expert_review import create_app
from kimore_expert_review.model import validate_adjudication, connect_database
from test_expert_review_app import ExpertReviewAppTests


class ProvenanceTests(unittest.TestCase):
    def test_supplemental_queue_contains_every_unresolved_pilot_case(self):
        def read(name):
            with (ROOT / 'annotations' / name).open(encoding='utf-8', newline='') as handle:
                return list(csv.DictReader(handle))
        labels = read('kimore_es3_pilot_labels.csv')
        supplemental = read('unresolved_pilot_queue.csv')
        expected = {(r['sample_id'], r['candidate_rank']) for r in labels
                    if r['execution_label'] in {'uncertain', 'ungradable'}}
        self.assertEqual(len(expected), 4)
        self.assertEqual(expected, {(r['sample_id'], r['candidate_rank']) for r in supplemental})
        self.assertTrue(all(not row['execution_label'] for row in supplemental))

    def test_workbook_identity_and_score_anomalies_are_reported_without_relabeling(self):
        from openpyxl import Workbook
        from kimore_dataset_audit import read_subject_scores
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary) / 'NE_ID2'
            path = subject / 'Es1/Label/ClinicalAssessment_NE_ID2.xlsx'
            path.parent.mkdir(parents=True)
            workbook = Workbook()
            workbook.active.append(['Subject ID', 'Clinical TS Ex #1', 'Clinical PO Ex #1', 'Clinical CF Ex #1'])
            workbook.active.append(['E_ID1', 44.6667, 15, 29])
            workbook.save(path)
            workbook.close()
            scores, _, warnings = read_subject_scores(subject)
            self.assertEqual(scores, {})
            self.assertTrue(any('Subject ID' in message for message in warnings))
            self.assertTrue(any('PO+CF' in message for message in warnings))

    def test_skeleton_joints_fit_below_tile_header_for_entire_clip(self):
        from kimore_pilot_review import render_skeleton, SKELETON_EDGES
        from kimore_dataset import JOINT_INDEX
        from types import SimpleNamespace
        points = np.random.default_rng(1).normal(size=(4, 25, 3))
        sequence = SimpleNamespace(positions=points)
        circles = []
        with patch('kimore_pilot_review.cv2.circle', side_effect=lambda canvas, point, *args: circles.append(point)):
            for frame in range(4):
                render_skeleton(sequence, frame)
        self.assertTrue(all(120 <= y <= 480 for x, y in circles))
        self.assertTrue(all(30 <= x <= 450 or 510 <= x <= 930 for x, y in circles))

    def test_resume_rejects_changed_seed_model_and_data(self):
        for field in ('seed', 'channels', 'data_sha256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                manifest = {'seed': 1, 'channels': 64, 'data_sha256': 'original'}
                prepare_run(folder, manifest, False)
                prepare_run(folder, manifest, True)
                with self.assertRaises(ValueError):
                    prepare_run(folder, {**manifest, field: 'changed'}, True)

    def test_legacy_predictions_are_not_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'predictions.csv').write_text('old predictions')
            with self.assertRaises(ValueError):
                prepare_run(folder, {}, True)

    def test_fold_requires_checkpoint_and_intact_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ('checkpoint.pt', 'history.csv', 'metrics.json'):
                (folder / name).write_text('fixture')
            (folder / 'predictions.csv').write_text('sample_id,fold\ns1,1\n')
            complete_fold(folder, 'run')
            self.assertEqual(len(verify_fold(folder, 'run')), 1)
            (folder / 'predictions.csv').write_text('sample_id,fold\ns2,1\n')
            with self.assertRaises(ValueError):
                verify_fold(folder, 'run')

    def test_six_fold_export_rejected_before_reading_data(self):
        with self.assertRaisesRegex(ValueError, 'five folds'):
            build_ml_dataset(Path('does-not-exist'), n_splits=6)

    def test_oof_requires_exact_coverage_and_correct_fold(self):
        for rows in ([], [{'sample_id': 's', 'fold': 2}],
                     [{'sample_id': 's', 'fold': 1}] * 2):
            with self.assertRaises(ValueError):
                validate_population(rows, [('s', 1)])

    def test_interval_changes_reject_labels(self):
        row = {'sample_id': 's', 'candidate_rank': '1', 'component_name': 'arm', 'original_frame_start': '10'}
        for field in ('component_name', 'original_frame_start', 'reference_sample_id', 'run_id'):
            with self.assertRaises(ValueError):
                validate_identity({**row, field: 'changed'}, row)

    def test_ungradable_and_uncertain_adjudication(self):
        for label, status in [('ungradable', 'excluded_ungradable'), ('uncertain', 'adjudication_needed')]:
            result = validate_adjudication({'execution_label': label, 'reviewer_confidence': 'high', 'review_notes': 'More evidence needed'})
            self.assertEqual(result['review_status'], status)


class ReviewRegressionTests(ExpertReviewAppTests):
    def restart(self):
        return create_app(queue_path=self.queue, primary_labels_path=self.primary, sheets_dir=self.sheets,
                          database_path=self.database, testing=True, secret_key='test-secret')

    def test_changed_primary_rejected_after_seal(self):
        self._start()
        self._save_label(1)
        self._save_label(2)
        self.client.post('/finish', data={'csrf_token': self._csrf()})
        self.primary.write_text(self.primary.read_text().replace('SECRET PRIMARY NOTE', 'changed'))
        with self.assertRaisesRegex(RuntimeError, 'Round labels'):
            self.restart()

    def test_original_evidence_is_frozen_and_restart_rejects_change(self):
        self._start()
        before = self.client.get('/evidence/1.jpg').data
        (self.sheets / 'S1_Es3.jpg').write_bytes(b'changed')
        self.assertEqual(before, self.client.get('/evidence/1.jpg').data)
        with self.assertRaisesRegex(RuntimeError, 'Round labels'):
            self.restart()

    def test_adjudication_does_not_leak_and_history_is_preserved(self):
        from kimore_expert_review.model import save_adjudication, load_adjudications
        for reviewer in ('expert_02', 'expert_04'):
            self._start(reviewer)
            self._save_label(1, 'uncertain')
            self._save_label(2, 'uncertain')
            self.client.post('/finish', data={'csrf_token': self._csrf()})
        candidate = self.app.extensions['review_candidates'][0]
        decision = validate_adjudication({'execution_label': 'ungradable', 'reviewer_confidence': 'high', 'review_notes': 'Insufficient evidence'})
        save_adjudication(self.database, 'expert_03', candidate, decision, 'expert_02')
        self.assertFalse(load_adjudications(self.database, 'expert_04'))
        save_adjudication(self.database, 'expert_03', candidate, decision, 'expert_02')
        with connect_database(self.database) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM adjudication_history').fetchone()[0], 2)

    def test_agreement_displays_numeric_item_count(self):
        self._start()
        self._save_label(1)
        self._save_label(2)
        self.client.post('/finish', data={'csrf_token': self._csrf()})
        html = self.client.get('/agreement').data
        self.assertIn(b'2 stavki', html)
        self.assertNotIn(b'built-in method', html)


if __name__ == '__main__':
    unittest.main()
