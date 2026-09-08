"""Holdout boundaries and reference-first workflow, using synthetic video metadata."""
from unittest.mock import patch, MagicMock

from test_expert_review_app import ExpertReviewAppTests
from kimore_expert_review import create_app
from kimore_expert_review.video import load_video_round


class ExpertVideoTests(ExpertReviewAppTests):
    def setUp(self):
        super().setUp()
        self.manifest = self.root / 'videos.csv'
        self.training = self.root / 'training.csv'
        self._write_csv(self.training, ['sample_id', 'subject_id'], [{'sample_id': 'TRAIN_Es3', 'subject_id': 'TRAIN'}])
        self.video_rows = []
        for sid, subject, role in [('S1_Es3', 'S1', 'validation'), ('S2_Es3', 'S2', 'validation'), ('REF_Es3', 'REF', 'reference')]:
            (self.root / f'{sid}.mp4').write_bytes(b'synthetic-video')
            self.video_rows.append(dict(sample_id=sid, subject_id=subject, role=role,
                                       rgb_path=f'{sid}.mp4', start_seconds='1', end_seconds='9'))
        self._write_csv(self.manifest, list(self.video_rows[0]), self.video_rows)
        capture = MagicMock()
        capture.get.side_effect = lambda prop: 10 if prop == 5 else 100
        capture.read.return_value = (True, None)
        self.capture_patch = patch('kimore_expert_review.video.cv2.VideoCapture', return_value=capture)
        self.capture_patch.start()
        self.addCleanup(self.capture_patch.stop)

    def test_training_subject_leakage_rejected(self):
        self._write_csv(self.training, ['sample_id', 'subject_id'], [{'sample_id': 'OTHER_EXERCISE', 'subject_id': 'S1'}])
        with self.assertRaisesRegex(ValueError, 'leakage'):
            load_video_round(self.manifest, self.training, self.app.extensions['review_candidates'])

    def test_missing_rgb_excluded(self):
        (self.root / 'S1_Es3.mp4').unlink()
        selected, _, audit = load_video_round(self.manifest, self.training, self.app.extensions['review_candidates'])
        self.assertEqual([c.row['sample_id'] for c in selected], ['S2_Es3'])
        self.assertTrue(audit['excluded'])

    def test_invalid_boundaries_rejected(self):
        self.video_rows[0]['end_seconds'] = '100'
        self._write_csv(self.manifest, list(self.video_rows[0]), self.video_rows)
        with self.assertRaisesRegex(ValueError, 'boundaries'):
            load_video_round(self.manifest, self.training, self.app.extensions['review_candidates'])

    def test_reference_gate_and_video_delivery(self):
        self.app = create_app(queue_path=self.queue, primary_labels_path=self.primary,
                              sheets_dir=self.sheets, database_path=self.root / 'video.sqlite3',
                              testing=True, secret_key='test', video_manifest_path=self.manifest,
                              training_inventory_path=self.training)
        self.client = self.app.test_client()
        self._start()
        self.assertIn('/reference/1', self.client.get('/review/1').location)
        self.assertIn('/reference/1', self.client.post('/review/1', data={}).location)
        self.assertEqual(self.client.get('/video/1/sample').status_code, 403)
        self.assertEqual(self.client.get('/reference/1').status_code, 200)
        self.assertEqual(self.client.post('/reference/1', data={'csrf_token': self._csrf()}).status_code, 400)
        self.client.post('/reference/1', data={'csrf_token': self._csrf(), 'understood': 'yes'})
        page = self.client.get('/review/1')
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'data-video-comparison', page.data)
        response = self.client.get('/video/1/sample', headers={'Range': 'bytes=0-3'})
        self.assertEqual(response.status_code, 206)
        response.close()
