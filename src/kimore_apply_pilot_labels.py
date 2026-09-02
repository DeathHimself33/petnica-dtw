"""Validate and apply the preliminary Es3 pilot labels to the review queue."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


KEY_FIELDS = ("sample_id", "candidate_rank")
REVIEW_FIELDS = (
    "review_status",
    "execution_label",
    "error_type",
    "severity",
    "reviewer_confidence",
    "annotator",
    "review_notes",
)
ALLOWED_STATUS = {"reviewed", "adjudication_needed"}
ALLOWED_LABELS = {"correct", "error", "uncertain", "ungradable"}
ALLOWED_ERROR_TYPES = {
    "range_of_motion",
    "direction",
    "timing",
    "asymmetry",
    "compensation",
    "posture",
    "other",
}
ALLOWED_SEVERITIES = {"mild", "moderate", "severe"}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row["sample_id"], row["candidate_rank"]


def validate_label(row: dict[str, str]) -> None:
    key = row_key(row)
    status = row["review_status"]
    label = row["execution_label"]
    error_type = row["error_type"]
    severity = row["severity"]
    confidence = row["reviewer_confidence"]
    if status not in ALLOWED_STATUS:
        raise ValueError(f"{key}: invalid review_status {status!r}")
    if label not in ALLOWED_LABELS:
        raise ValueError(f"{key}: invalid execution_label {label!r}")
    if confidence not in ALLOWED_CONFIDENCE:
        raise ValueError(f"{key}: invalid reviewer_confidence {confidence!r}")
    if not row["annotator"] or not row["review_notes"]:
        raise ValueError(f"{key}: annotator and review_notes are required")
    if label == "error":
        if error_type not in ALLOWED_ERROR_TYPES:
            raise ValueError(f"{key}: error row needs a valid error_type")
        if severity not in ALLOWED_SEVERITIES:
            raise ValueError(f"{key}: error row needs a valid severity")
    elif error_type or severity:
        raise ValueError(f"{key}: non-error row must not have error_type/severity")
    if label in {"uncertain", "ungradable"} and status != "adjudication_needed":
        raise ValueError(f"{key}: {label} must have adjudication_needed status")


def merge_labels(
    queue_rows: list[dict[str, str]], label_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    labels_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for label in label_rows:
        validate_label(label)
        key = row_key(label)
        if key in labels_by_key:
            raise ValueError(f"Duplicate label key: {key}")
        labels_by_key[key] = label

    queue_keys = [row_key(row) for row in queue_rows]
    if len(queue_keys) != len(set(queue_keys)):
        raise ValueError("Pilot queue contains duplicate sample/rank keys")
    missing = set(queue_keys).difference(labels_by_key)
    extra = set(labels_by_key).difference(queue_keys)
    if missing or extra:
        raise ValueError(
            f"Queue/label key mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    merged: list[dict[str, str]] = []
    for row in queue_rows:
        label = labels_by_key[row_key(row)]
        merged.append({**row, **{field: label[field] for field in REVIEW_FIELDS}})
    return merged


def summary_group(
    rows: list[dict[str, str]], group_type: str, group_value: str
) -> dict[str, object]:
    counts = Counter(row["execution_label"] for row in rows)
    decided = counts["correct"] + counts["error"]
    return {
        "group_type": group_type,
        "group_value": group_value,
        "candidates": len(rows),
        "correct": counts["correct"],
        "error": counts["error"],
        "uncertain": counts["uncertain"],
        "ungradable": counts["ungradable"],
        "decided_candidates": decided,
        "preliminary_candidate_precision": (
            counts["error"] / decided if decided else ""
        ),
    }


def build_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = [summary_group(rows, "overall", "all")]
    for field, group_type in (
        ("cohort", "cohort"),
        ("component_name", "component"),
        ("candidate_rank", "candidate_rank"),
    ):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        for value in sorted(grouped, key=lambda item: (len(item), item)):
            output.append(summary_group(grouped[value], group_type, value))
    boundary_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        position = (
            "boundary_0_to_5_or_95_to_100_percent"
            if row["window_start_percent"] == "0.0"
            or row["window_end_percent"] == "100.0"
            else "interior_5_to_95_percent"
        )
        boundary_groups[position].append(row)
    for value in sorted(boundary_groups):
        output.append(summary_group(boundary_groups[value], "window_position", value))
    return output


def build_breakdown_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    dimensions = (
        ("review_status", rows),
        ("execution_label", rows),
        ("reviewer_confidence", rows),
        ("error_type", [row for row in rows if row["execution_label"] == "error"]),
        ("severity", [row for row in rows if row["execution_label"] == "error"]),
    )
    for dimension, eligible in dimensions:
        for value, count in sorted(Counter(row[dimension] for row in eligible).items()):
            output.append(
                {"dimension": dimension, "value": value, "count": count}
            )
    return output


def second_review_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select one blinded candidate per recording, balanced across ranks."""
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
    selected: list[dict[str, str]] = []
    for sample_index, sample_id in enumerate(sorted(by_sample)):
        wanted_rank = str(sample_index % 5 + 1)
        candidates = [
            row for row in by_sample[sample_id] if row["candidate_rank"] == wanted_rank
        ]
        if len(candidates) != 1:
            raise ValueError(f"{sample_id}: expected candidate rank {wanted_rank}")
        blinded = dict(candidates[0])
        for field in REVIEW_FIELDS:
            blinded[field] = ""
        blinded["second_review_selection"] = (
            "one_blinded_candidate_per_recording_balanced_across_ranks"
        )
        selected.append(blinded)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("results/interpretable_dtw/pilot_review/pilot_annotation_queue.csv"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("annotations/kimore_es3_pilot_labels.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/interpretable_dtw/pilot_review"),
    )
    args = parser.parse_args()

    queue_fields, queue_rows = read_rows(args.queue)
    _, label_rows = read_rows(args.labels)
    merged = merge_labels(queue_rows, label_rows)
    write_rows(
        args.output / "pilot_annotation_queue_preliminary.csv",
        queue_fields,
        merged,
    )

    summary_rows = build_summary_rows(merged)
    write_rows(
        args.output / "pilot_label_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    breakdown_rows = build_breakdown_rows(merged)
    write_rows(
        args.output / "pilot_label_breakdown.csv",
        list(breakdown_rows[0]),
        breakdown_rows,
    )
    overall = summary_rows[0]
    payload = {
        "label_scope": "preliminary_nonclinical_first_pass",
        "annotator": "codex_preliminary_v1",
        "recordings": len({row["sample_id"] for row in merged}),
        **overall,
        "precision_definition": (
            "error / (error + correct), excluding uncertain and ungradable"
        ),
        "independent_second_review_required": True,
    }
    with (args.output / "pilot_label_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    blinded = second_review_rows(queue_rows)
    write_rows(
        args.output / "second_review_queue.csv",
        [*queue_fields, "second_review_selection"],
        blinded,
    )
    print(
        f"Applied {len(merged)} labels: {overall['error']} error, "
        f"{overall['correct']} correct, {overall['uncertain']} uncertain; "
        f"wrote {len(blinded)} blinded second-review rows."
    )


if __name__ == "__main__":
    main()
