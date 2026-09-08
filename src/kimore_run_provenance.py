"""Immutable experiment identities and verified fold completion records."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_population(rows, expected) -> None:
    observed = [(str(row['sample_id']), int(row['fold'])) for row in rows]
    wanted = [(str(sample), int(fold)) for sample, fold in expected]
    if len(observed) != len(set(observed)) or sorted(observed) != sorted(wanted):
        raise ValueError('Predictions must cover every expected sample exactly once in its test fold')
    if len({sample for sample, _ in observed}) != len(observed):
        raise ValueError('Duplicate OOF sample IDs')


def prepare_run(output: Path, manifest: dict, resume: bool) -> str:
    path = output / 'run_manifest.json'
    if path.exists():
        if not resume:
            raise ValueError('Output already contains a run; use --resume or a new directory')
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Resume provenance mismatch: configuration, data, code or runtime changed')
    elif any(output.iterdir()):
        raise ValueError('Existing output has no verified run manifest; use a new directory')
    else:
        atomic_json(path, manifest)
    return digest(manifest)


FOLD_FILES = ('checkpoint.pt', 'history.csv', 'predictions.csv', 'metrics.json')


def complete_fold(folder: Path, run_id: str) -> None:
    atomic_json(folder / 'complete.json', {
        'run_id': run_id, 'files': {name: sha256(folder / name) for name in FOLD_FILES}
    })


def verify_fold(folder: Path, run_id: str) -> list[dict]:
    record = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
    if record != {'run_id': run_id, 'files': {name: sha256(folder / name) for name in FOLD_FILES}}:
        raise ValueError(f'Fold artifacts changed or belong to another run: {folder}')
    with (folder / 'predictions.csv').open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
