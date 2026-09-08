"""Verify duplicate recording bundles without modifying source files."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from kimore_run_provenance import sha256


def select_identical_bundle(folder: Path) -> dict:
    """Select only when every stream has one unique byte content and recording ID.

    Equal RGB/skeleton counts are a structural check, not event synchronization.
    RGB is optional for the numeric pipeline; ambiguous video bundles are refused.
    """
    from kimore_dataset import load_joint_positions
    from kimore_dataset_audit import read_numeric_csv

    selected = {}
    inventory = {}
    recording_ids = set()
    counts = []
    for stream, width in [('JointPosition', 100), ('JointOrientation', 100), ('TimeStamp', 1)]:
        paths = sorted(folder.glob(f'{stream}*.csv'))
        if not paths:
            raise ValueError(f'Missing {stream} stream')
        hashes = {path: sha256(path) for path in paths}
        if len(set(hashes.values())) != 1:
            raise ValueError(f'{stream} copies differ; cannot select automatically')
        path = min(paths, key=lambda p: ('(' in p.stem, p.name))
        for copy in paths:
            recording_ids.add(re.sub(r'\(\d+\)$', '', copy.stem[len(stream):]))
        values = read_numeric_csv(path)
        if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
            raise ValueError(f'Invalid {stream} numeric schema')
        if stream == 'TimeStamp' and (len(values) < 2 or not np.all(np.diff(values[:, 0]) > 0)):
            raise ValueError('Timestamps must increase strictly')
        counts.append(len(values))
        selected[stream] = str(path.resolve())
        inventory[stream] = [{'path': str(p.resolve()), 'sha256': h} for p, h in hashes.items()]
    if len(recording_ids) != 1:
        raise ValueError('Streams refer to different recording IDs')
    if len(set(counts)) != 1:
        raise ValueError('Position/orientation/timestamp frame counts differ')
    load_joint_positions(Path(selected['JointPosition']))

    rgb_paths = sorted(folder.parent.glob('rgb/*.mp4'))
    rgb = {'available': bool(rgb_paths), 'event_synchronization': 'not inferred from frame counts'}
    if rgb_paths:
        import cv2
        rgb_hashes = {p: sha256(p) for p in rgb_paths}
        if len(set(rgb_hashes.values())) != 1:
            raise ValueError('Multiple different RGB recordings')
        path = min(rgb_paths, key=lambda p: ('(' in p.stem, p.name))
        if not any(recording_id in path.stem for recording_id in recording_ids):
            raise ValueError('RGB filename does not match numeric recording ID')
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError('Cannot open RGB recording')
            frames = 0
            while capture.grab():
                frames += 1
        finally:
            capture.release()
        rgb.update(frames=frames, frame_count_matches=frames == counts[0],
                   timing_evidence_usable=False,
                   note=('matching counts do not establish synchronization' if frames == counts[0]
                         else f'RGB has {frames} frames, numeric streams have {counts[0]}; numeric selection is independent of RGB alignment'),
                   files=[{'path': str(p.resolve()), 'sha256': h} for p, h in rgb_hashes.items()])
    return {'selected': selected, 'files': inventory, 'frames': counts[0], 'rgb': rgb,
            'policy': 'identical-content bundles only; no source files modified'}


def build_selection_registry(root: Path) -> dict:
    bundles, rejected = {}, {}
    for folder in sorted(root.glob('**/Raw')):
        if len(list(folder.glob('JointPosition*.csv'))) < 2:
            continue
        sample_id = f'{folder.parent.parent.name}_{folder.parent.name}'
        try:
            bundles[sample_id] = select_identical_bundle(folder)
        except ValueError as error:
            rejected[sample_id] = str(error)
    return {'version': 1, 'bundles': bundles, 'rejected': rejected}
