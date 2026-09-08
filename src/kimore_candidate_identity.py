"""Content identity for review intervals, including explicitly bound legacy labels."""
import hashlib
import json
from pathlib import Path

IDENTITY_FIELDS = (
    'sample_id', 'candidate_rank', 'fold', 'reference_sample_id',
    'component_name', 'component_index', 'window_start_percent', 'window_end_percent',
    'original_frame_start', 'original_frame_end',
    'reference_original_frame_start', 'reference_original_frame_end',
)


def candidate_id(row):
    value = {key: str(row.get(key, '')) for key in IDENTITY_FIELDS}
    value['run_id'] = row.get('run_id', '')
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_identity(queue, label):
    if 'component_name' in label:
        if candidate_id(queue) != candidate_id(label):
            raise ValueError('Label interval identity differs from review queue')
    else:
        # This registry binds the original sparse pilot labels to audited intervals.
        registry = Path(__file__).resolve().parents[1] / 'annotations' / 'pilot_identity_binding.json'
        bindings = json.loads(registry.read_text(encoding='utf-8'))['labels']
        label_hash = hashlib.sha256(json.dumps(label, sort_keys=True).encode()).hexdigest()
        if bindings.get(label_hash) != candidate_id(queue):
            raise ValueError('Label has no verified interval identity; supply immutable interval fields')
    if label.get('candidate_id') and label['candidate_id'] != candidate_id(queue):
        raise ValueError('Candidate ID mismatch')
