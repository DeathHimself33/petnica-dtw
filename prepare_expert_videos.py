"""Prepare actual RGB videos for an explicitly non-validation review preview.

Existing candidate identities and primary labels are preserved. Recording
endpoints are measured, never presented as verified movement boundaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import cv2
from kimore_dataset import read_manifest
from kimore_pilot_review import video_path, read_rows, write_rows
from kimore_expert_review.model import load_candidates, load_primary_labels, file_sha256


def prepare_packet(manifest, queue, primary, sheets, output):
    manifest, queue, primary, sheets, output = [Path(p).resolve() for p in (manifest, queue, primary, sheets, output)]
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty packet directory; existing review rounds must be preserved')
    candidates, _ = load_candidates(queue, sheets)
    load_primary_labels(primary, candidates)
    samples = {}
    for exercise in ('Es1', 'Es2', 'Es3', 'Es4', 'Es5'):
        selected, _ = read_manifest(manifest, exercise)
        samples.update({sample.sample_id: sample for sample in selected})
    sample_ids = {c.row['sample_id'] for c in candidates}
    reference_ids = {c.row['reference_sample_id'] for c in candidates}
    videos, exclusions = {}, []
    for sid in sorted(sample_ids | reference_ids):
        if sid not in samples:
            raise ValueError(f'Queue identity not in usable dataset manifest: {sid}')
        sample = samples[sid]
        path = video_path(sample)
        if path is None:
            exclusions.append({'sample_id': sid, 'reason': 'missing_rgb'})
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frames / fps if fps > 0 else 0
            readable, _ = capture.read()
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = ''.join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
        finally:
            capture.release()
        if not readable or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'RGB cannot be decoded: {sid}')
        if codec.lower() not in {'h264', 'avc1'}:
            raise ValueError(f'{sid}: convert {codec} video to browser-compatible H.264 before packaging')
        if abs(frames - sample.frames) / sample.frames > 0.10:
            raise ValueError(f'{sid}: RGB/skeleton frame counts differ by more than 10%')
        role = 'both' if sid in sample_ids & reference_ids else 'reference' if sid in reference_ids else 'validation'
        videos[sid] = dict(sample_id=sid, subject_id=sample.subject_id, role=role,
                           rgb_path=str(path.resolve()), start_seconds=0, end_seconds=duration,
                           boundary_basis='recording', codec=codec)
    kept = []
    for candidate in candidates:
        sid, ref = candidate.row['sample_id'], candidate.row['reference_sample_id']
        if sid not in videos or ref not in videos:
            exclusions.append({'sample_id': sid, 'reason': 'sample_or_reference_missing_rgb'})
            continue
        if candidate.row['subject_id'] != samples[sid].subject_id:
            raise ValueError(f'Subject mismatch: {sid}')
        kept.append(candidate)
    if not kept:
        raise ValueError('No RGB pairs remain')
    used = {c.row[k] for c in kept for k in ('sample_id', 'reference_sample_id')}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'sheets').mkdir(exist_ok=True)
    fields, _ = read_rows(queue)
    write_rows(output / 'queue.csv', fields, [c.row for c in kept])
    shutil.copyfile(primary, output / 'primary.csv')
    for candidate in kept:
        shutil.copyfile(candidate.sheet_path, output / 'sheets' / candidate.sheet_path.name)
    write_rows(output / 'videos.csv', list(next(iter(videos.values()))), [videos[sid] for sid in sorted(used)])
    metadata = dict(purpose='preview_not_validation', boundary_basis='recording',
                    items=len(kept), recordings=len({c.row['sample_id'] for c in kept}),
                    video_files=len(used), exclusions=exclusions,
                    sources={str(p): file_sha256(p) for p in (manifest, queue, primary)})
    (output / 'packet.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / 'README.txt').write_text(
        'Probni pregled stvarnih RGB videa. Nije nezavisna validacija.\n'
        '0-100% predstavlja ceo snimak, ne potvrdjene granice pokreta.\n'
        'Pokretanje iz foldera projekta:\n'
        f'.\\.venv\\Scripts\\python.exe expert_review_app.py --video-packet "{output}"\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'kimore_audit_output/kimore_manifest.csv')
    parser.add_argument('--queue', type=Path, default=ROOT / 'results/audit_fixes/pilot_review_v3/second_review_queue.csv')
    parser.add_argument('--primary-labels', type=Path, default=ROOT / 'annotations/kimore_es3_pilot_labels.csv')
    parser.add_argument('--sheets', type=Path, default=ROOT / 'results/audit_fixes/pilot_review_v3/sheets')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/video_preview')
    args = parser.parse_args()
    print(json.dumps(prepare_packet(args.manifest, args.queue, args.primary_labels, args.sheets, args.output), indent=2))


if __name__ == '__main__':
    main()
