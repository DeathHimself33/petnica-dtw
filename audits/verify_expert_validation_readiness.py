"""Verify the frozen expert-pilot package before a reviewer starts."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kimore_expert_review.model import (  # noqa: E402
    REVIEW_COLUMNS,
    file_sha256,
    load_candidates,
    load_primary_labels,
)
from kimore_expert_review.video import load_video_round  # noqa: E402


FREEZE = ROOT / "annotations/expert_validation_freeze_v1.json"
CORE = ROOT / "annotations/expert_validation_core_queue_v1.csv"
SUPPLEMENTAL = ROOT / "annotations/expert_validation_supplemental_queue_v1.csv"
ALL_UNRESOLVED = ROOT / "annotations/unresolved_pilot_queue.csv"
PRIMARY = ROOT / "annotations/kimore_es3_pilot_labels.csv"
SHEETS = ROOT / "results/audit_fixes/pilot_review_v3/sheets"
VIDEO_PACKETS = {
    "core": ROOT / "results/expert_validation_20260913/core_video_preview",
    "supplemental": ROOT / "results/expert_validation_20260913/supplemental_round",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def key(row: dict[str, str]) -> tuple[str, int]:
    return row["sample_id"], int(row["candidate_rank"])


def frozen_hash(path: Path, mode: str = "raw") -> str:
    if mode == "raw":
        return file_sha256(path)
    if mode == "text_lf":
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    raise ValueError(f"unsupported hash mode: {mode}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for artifact in freeze["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file():
            errors.append(f"missing frozen artifact: {artifact['path']}")
        elif frozen_hash(path, artifact.get("hash_mode", "raw")) != artifact["sha256"]:
            errors.append(f"hash mismatch: {artifact['path']}")

    queues = {
        "core": (CORE, 20, "one_blinded_candidate_per_recording_balanced_across_ranks"),
        "supplemental": (
            SUPPLEMENTAL,
            3,
            "supplemental_unresolved_pilot_case_excluded_from_original_agreement",
        ),
    }
    queue_rows: dict[str, list[dict[str, str]]] = {}
    for name, (path, expected, selection) in queues.items():
        rows = read_rows(path)
        queue_rows[name] = rows
        if len(rows) != expected:
            errors.append(f"{name}: expected {expected} rows, found {len(rows)}")
        if len({key(row) for row in rows}) != len(rows):
            errors.append(f"{name}: duplicate candidate identity")
        if any(row.get(field, "") for row in rows for field in REVIEW_COLUMNS):
            errors.append(f"{name}: blinded queue contains a review value")
        if any(row.get("second_review_selection") != selection for row in rows):
            errors.append(f"{name}: unexpected selection scope")
        try:
            candidates, _ = load_candidates(path, SHEETS)
            load_primary_labels(PRIMARY, candidates)
        except (FileNotFoundError, ValueError) as error:
            errors.append(f"{name}: {error}")

    core = queue_rows["core"]
    supplemental = queue_rows["supplemental"]
    if Counter(row["cohort"] for row in core) != Counter(
        {
            "back_pain": 4,
            "expert_control": 4,
            "nonexpert_control": 4,
            "parkinson": 4,
            "stroke": 4,
        }
    ):
        errors.append("core: cohort balance changed")
    if Counter(int(row["candidate_rank"]) for row in core) != Counter(
        {rank: 4 for rank in range(1, 6)}
    ):
        errors.append("core: candidate-rank balance changed")

    core_keys = {key(row) for row in core}
    supplemental_keys = {key(row) for row in supplemental}
    unresolved_keys = {key(row) for row in read_rows(ALL_UNRESOLVED)}
    if core_keys & supplemental_keys:
        errors.append("core and supplemental queues overlap")
    if (core_keys & unresolved_keys) | supplemental_keys != unresolved_keys:
        errors.append("the four unresolved pilot items are not covered exactly once")

    video_report: dict[str, object] = {}
    active_keys: dict[str, set[tuple[str, int]]] = {}
    for name, packet_dir in VIDEO_PACKETS.items():
        packet_path = packet_dir / "packet.json"
        if not packet_path.is_file():
            errors.append(f"required RGB packet missing: {packet_dir.relative_to(ROOT)}")
            continue
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        if packet.get("purpose") != "preview_not_validation":
            errors.append(f"{name}: RGB packet has an invalid purpose")
        try:
            candidates, _ = load_candidates(packet_dir / "queue.csv", packet_dir / "sheets")
            load_primary_labels(packet_dir / "primary.csv", candidates)
            if any(c.row.get(field, "") for c in candidates for field in REVIEW_COLUMNS):
                errors.append(f"{name}: RGB queue contains review values")
            selected, _, audit = load_video_round(
                packet_dir / "videos.csv", None, candidates, preview=True
            )
            active_keys[name] = {c.key for c in selected}
            expected_rows = {
                key(row): row for row in queue_rows[name]
                if row["sample_id"] not in freeze["review_design"]["rgb_excluded_sample_ids"]
            }
            if {c.key: c.row for c in selected} != expected_rows:
                errors.append(f"{name}: active RGB selection differs from frozen source rows")
            if audit["excluded"] or len(selected) != freeze["review_design"][f"{name}_items"]:
                errors.append(f"{name}: frozen RGB evidence is incomplete")
        except (FileNotFoundError, ValueError) as error:
            errors.append(f"{name}: {error}")
        video_report[name] = {
            "items": packet.get("items"),
            "excluded": packet.get("exclusions", []),
            "purpose": packet.get("purpose"),
        }

    active_core = active_keys.get("core", set())
    active_supplemental = active_keys.get("supplemental", set())
    active_union = active_core | active_supplemental
    if not unresolved_keys <= active_union:
        errors.append("RGB selection does not cover all four unresolved pilot items")
    report = {
        "pilot_gate_ready": not errors,
        "strict_external_validation_ready": False,
        "primary_evidence": "RGB videos",
        "candidate_core_items_before_rgb_filter": len(core),
        "core_items": len(active_core),
        "supplemental_items": len(active_supplemental),
        "unique_items": len(active_union),
        "unresolved_items_covered": len(unresolved_keys & active_union),
        "video_packets": video_report,
        "errors": errors,
        "warnings": warnings
        + [
            "RGB is the primary review evidence. Progress uses whole-recording endpoints; movement boundaries and phase synchronization are not verified.",
            "Strict external validation still requires new held-out subjects, verified movement boundaries, synchronization checks, and the evaluated model's complete training inventory.",
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
