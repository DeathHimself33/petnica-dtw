"""Explicit, auditable holdout selection and RGB evidence for a review round."""
import csv
import math
from pathlib import Path

import cv2

from .model import file_sha256


def load_video_round(manifest_path, training_path, candidates):
    """Training CSV must enumerate the actual training run, not fold membership."""
    manifest_path, training_path = Path(manifest_path), Path(training_path)
    with training_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "subject_id"} <= set(reader.fieldnames or []):
            raise ValueError("Training inventory requires sample_id and subject_id")
        training = list(reader)
    if not training or any(not r["sample_id"] or not r["subject_id"] for r in training):
        raise ValueError("Supply the complete, nonempty training inventory")
    training_samples = {r["sample_id"] for r in training}
    training_subjects = {r["subject_id"] for r in training}
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "subject_id", "role", "rgb_path", "start_seconds", "end_seconds"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Video manifest requires {sorted(required)}")
        rows = list(reader)
    videos, excluded, seen = {}, [], set()
    for row in rows:
        sid = row["sample_id"]
        if not sid or not row["subject_id"] or sid in seen:
            raise ValueError("Video manifest has missing or duplicate identities")
        seen.add(sid)
        if row["role"] not in {"validation", "reference"}:
            raise ValueError("Video role must be validation or reference")
        if row["role"] == "validation" and (sid in training_samples or row["subject_id"] in training_subjects):
            raise ValueError(f"Training/validation leakage: {sid}")
        path = (manifest_path.parent / row["rgb_path"]).resolve()
        if not row["rgb_path"] or not path.is_file():
            excluded.append({"sample_id": sid, "reason": "missing_rgb"})
            continue
        if path.suffix.lower() not in {".mp4", ".webm"}:
            raise ValueError(f"Browser video must be MP4 or WebM: {sid}")
        start, end = float(row["start_seconds"]), float(row["end_seconds"])
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
            readable, _ = capture.read()
        finally:
            capture.release()
        if not readable or not all(math.isfinite(v) for v in (start, end, duration)) or not 0 <= start < end <= duration:
            raise ValueError(f"Invalid video or movement boundaries: {sid}")
        videos[sid] = {**row, "path": path, "start": start, "end": end, "sha256": file_sha256(path)}
    selected = []
    for candidate in candidates:
        sid, ref = candidate.row["sample_id"], candidate.row.get("reference_sample_id", "")
        if sid not in seen or ref not in seen:
            raise ValueError(f"Missing manifest identity for {sid} or its reference")
        if sid not in videos or ref not in videos:
            excluded.append({"sample_id": sid, "reason": "sample_or_reference_missing_rgb"})
            continue
        if videos[sid]["role"] != "validation" or videos[ref]["role"] != "reference":
            raise ValueError("Candidates must be validation recordings with a reference-role video")
        if videos[sid]["subject_id"] != candidate.row.get("subject_id"):
            raise ValueError(f"Subject identity mismatch: {sid}")
        selected.append(candidate)
    if not selected:
        raise ValueError("No eligible RGB validation candidates remain")
    validation_subjects = {videos[c.row["sample_id"]]["subject_id"] for c in selected}
    if any(v["role"] == "reference" and v["subject_id"] in validation_subjects for v in videos.values()):
        raise ValueError("Reference and validation subjects must be disjoint")
    audit = {"manifest_sha256": file_sha256(manifest_path), "training_sha256": file_sha256(training_path),
             "videos": {k: {field: value for field, value in v.items() if field != "path"} for k, v in videos.items()},
             "excluded": excluded, "normalization": "linear movement start/end; no phase alignment"}
    return selected, videos, audit
