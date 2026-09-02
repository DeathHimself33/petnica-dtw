"""Build a reproducible RGB/skeleton review packet for an Es3 pilot.

The packet is deliberately separate from the full annotation queue.  It gives a
reviewer the original RGB frames, the fold-specific reference frames, and a few
simple posture measurements; it does not assign execution-error labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from kimore_dataset import JOINT_INDEX, JointSequence, KimoreSample, load_joint_positions, read_manifest


PILOT_SAMPLE_IDS = (
    "B_ID1_Es3",
    "B_ID5_Es3",
    "B_ID6_Es3",
    "B_ID8_Es3",
    "E_ID14_Es3",
    "E_ID8_Es3",
    "E_ID9_Es3",
    "E_ID12_Es3",
    "NE_ID10_Es3",
    "NE_ID6_Es3",
    "NE_ID20_Es3",
    "NE_ID24_Es3",
    "P_ID4_Es3",
    "P_ID5_Es3",
    "P_ID12_Es3",
    "P_ID6_Es3",
    "S_ID9_Es3",
    "S_ID10_Es3",
    "S_ID8_Es3",
    "S_ID3_Es3",
)

REVIEW_COLUMNS = (
    "review_status",
    "execution_label",
    "error_type",
    "severity",
    "reviewer_confidence",
    "annotator",
    "review_notes",
)


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


def video_path(sample: KimoreSample) -> Path | None:
    candidates = sorted(sample.position_path.parent.parent.joinpath("rgb").glob("*.mp4"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"{sample.sample_id}: expected one RGB MP4, found {len(candidates)}"
        )
    return candidates[0]


def frame_triplet(start: int, end: int) -> tuple[int, int, int]:
    return start, (start + end) // 2, end


def progress_frames(frame_count: int) -> tuple[int, ...]:
    return tuple(int(round(value)) for value in np.linspace(0, frame_count - 1, 7))


def open_video(path: Path | None, expected_frames: int) -> cv2.VideoCapture | None:
    if path is None:
        return None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(video_frames - expected_frames) / expected_frames > 0.10:
        capture.release()
        raise ValueError(
            f"{path}: RGB has {video_frames} frames, skeleton has {expected_frames}"
        )
    return capture


SKELETON_EDGES = (
    ("Head", "Neck"),
    ("Neck", "SpineShoulder"),
    ("SpineShoulder", "SpineMid"),
    ("SpineMid", "SpineBase"),
    ("SpineShoulder", "ShoulderLeft"),
    ("ShoulderLeft", "ElbowLeft"),
    ("ElbowLeft", "WristLeft"),
    ("SpineShoulder", "ShoulderRight"),
    ("ShoulderRight", "ElbowRight"),
    ("ElbowRight", "WristRight"),
    ("SpineBase", "HipLeft"),
    ("HipLeft", "KneeLeft"),
    ("KneeLeft", "AnkleLeft"),
    ("AnkleLeft", "FootLeft"),
    ("SpineBase", "HipRight"),
    ("HipRight", "KneeRight"),
    ("KneeRight", "AnkleRight"),
    ("AnkleRight", "FootRight"),
)


def render_skeleton(sequence: JointSequence, frame_index: int) -> np.ndarray:
    canvas = np.full((540, 960, 3), 245, dtype=np.uint8)
    pose = sequence.positions[frame_index]
    center = pose[JOINT_INDEX["SpineBase"]]
    body_height = max(
        float(
            pose[JOINT_INDEX["Head"], 1]
            - min(pose[JOINT_INDEX["AnkleLeft"], 1], pose[JOINT_INDEX["AnkleRight"], 1])
        ),
        0.5,
    )
    scale = 360.0 / body_height

    for title, x_axis, panel_center in (("FRONT x/y", 0, 240), ("SIDE z/y", 2, 720)):
        cv2.putText(canvas, title, (panel_center - 75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)

        def point(joint_name: str) -> tuple[int, int]:
            value = pose[JOINT_INDEX[joint_name]]
            return (
                int(round(panel_center + (value[x_axis] - center[x_axis]) * scale)),
                int(round(470 - (value[1] - center[1]) * scale)),
            )

        for first, second in SKELETON_EDGES:
            cv2.line(canvas, point(first), point(second), (60, 105, 165), 5, cv2.LINE_AA)
        for joint_name in {name for edge in SKELETON_EDGES for name in edge}:
            cv2.circle(canvas, point(joint_name), 5, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.putText(canvas, "NO RGB - skeleton rendering", (337, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 180), 2, cv2.LINE_AA)
    return canvas


def read_frame(
    capture: cv2.VideoCapture | None,
    frame_index: int,
    skeleton_frames: int,
    sequence: JointSequence,
) -> np.ndarray:
    if capture is None:
        return render_skeleton(sequence, frame_index)
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if skeleton_frames <= 1:
        video_frame_index = 0
    else:
        video_frame_index = int(
            round(frame_index * (video_frames - 1) / (skeleton_frames - 1))
        )
    capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame_index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(
            f"Could not decode RGB frame {video_frame_index} mapped from skeleton frame {frame_index}"
        )
    return frame


def tile(frame: np.ndarray, label: str, width: int = 236, height: int = 133) -> np.ndarray:
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (width, 25), (0, 0, 0), thickness=-1)
    cv2.putText(
        resized,
        label,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return resized


def angle_degrees(first: np.ndarray, vertex: np.ndarray, third: np.ndarray) -> np.ndarray:
    left = first - vertex
    right = third - vertex
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def interval_measurements(sequence: JointSequence, start: int, end: int) -> dict[str, float]:
    positions = sequence.positions[start : end + 1]
    joint = lambda name: positions[:, JOINT_INDEX[name]]
    spine_base = joint("SpineBase")
    spine_shoulder = joint("SpineShoulder")
    torso = spine_shoulder - spine_base
    torso_length = np.maximum(np.linalg.norm(torso, axis=1), 1e-12)
    torso_up = torso / torso_length[:, None]
    vertical = np.zeros_like(torso_up)
    vertical[:, 1] = 1.0

    left_upper = joint("ElbowLeft") - joint("ShoulderLeft")
    right_upper = joint("ElbowRight") - joint("ShoulderRight")
    left_upper /= np.maximum(np.linalg.norm(left_upper, axis=1, keepdims=True), 1e-12)
    right_upper /= np.maximum(np.linalg.norm(right_upper, axis=1, keepdims=True), 1e-12)

    def median(values: np.ndarray) -> float:
        return float(np.median(values))

    return {
        "left_shoulder_angle_deg": median(
            np.degrees(np.arccos(np.clip(np.abs(np.sum(left_upper * torso_up, axis=1)), -1, 1)))
        ),
        "right_shoulder_angle_deg": median(
            np.degrees(np.arccos(np.clip(np.abs(np.sum(right_upper * torso_up, axis=1)), -1, 1)))
        ),
        "left_elbow_angle_deg": median(
            angle_degrees(joint("ShoulderLeft"), joint("ElbowLeft"), joint("WristLeft"))
        ),
        "right_elbow_angle_deg": median(
            angle_degrees(joint("ShoulderRight"), joint("ElbowRight"), joint("WristRight"))
        ),
        "left_knee_angle_deg": median(
            angle_degrees(joint("HipLeft"), joint("KneeLeft"), joint("AnkleLeft"))
        ),
        "right_knee_angle_deg": median(
            angle_degrees(joint("HipRight"), joint("KneeRight"), joint("AnkleRight"))
        ),
        "left_wrist_height_torso": median(
            (joint("WristLeft")[:, 1] - joint("ShoulderLeft")[:, 1]) / torso_length
        ),
        "right_wrist_height_torso": median(
            (joint("WristRight")[:, 1] - joint("ShoulderRight")[:, 1]) / torso_length
        ),
        "torso_tilt_deg": median(
            np.degrees(np.arccos(np.clip(np.sum(torso_up * vertical, axis=1), -1, 1)))
        ),
    }


def metric_lines(metrics: dict[str, float], prefix: str) -> list[str]:
    return [
        f"{prefix} shoulder L/R {metrics['left_shoulder_angle_deg']:.0f}/{metrics['right_shoulder_angle_deg']:.0f} deg",
        f"{prefix} elbow L/R {metrics['left_elbow_angle_deg']:.0f}/{metrics['right_elbow_angle_deg']:.0f} deg",
        f"{prefix} wrist-h L/R {metrics['left_wrist_height_torso']:+.2f}/{metrics['right_wrist_height_torso']:+.2f}",
        f"{prefix} knee L/R {metrics['left_knee_angle_deg']:.0f}/{metrics['right_knee_angle_deg']:.0f}; tilt {metrics['torso_tilt_deg']:.1f}",
    ]


def put_lines(
    canvas: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    color: tuple[int, int, int] = (35, 35, 35),
    scale: float = 0.43,
    spacing: int = 18,
) -> None:
    for offset, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + offset * spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def build_sheet(
    sample: KimoreSample,
    reference: KimoreSample,
    candidates: list[dict[str, str]],
    output_path: Path,
) -> list[dict[str, object]]:
    sample_sequence = load_joint_positions(sample.position_path)
    reference_sequence = load_joint_positions(reference.position_path)
    sample_video = open_video(video_path(sample), sample.frames)
    reference_video = open_video(video_path(reference), reference.frames)

    margin = 20
    label_width = 420
    tile_width = 236
    tile_height = 133
    gap = 8
    columns = 6
    width = margin * 2 + label_width + columns * tile_width + (columns - 1) * gap
    overview_height = 166
    candidate_height = 178
    height = 78 + overview_height * 2 + candidate_height * len(candidates) + margin
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)

    cv2.putText(
        canvas,
        f"{sample.sample_id} | {sample.cohort} | TS {sample.score:.2f} | ref {reference.sample_id}",
        (margin, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "RGB is blurred by KIMORE. Skeleton frames are mapped to RGB by normalized recording progress.",
        (margin, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    y = 78
    for label, capture, frame_count in (
        ("SAMPLE overview", sample_video, sample.frames),
        ("REFERENCE overview", reference_video, reference.frames),
    ):
        cv2.rectangle(canvas, (0, y), (width, y + overview_height - 1), (232, 238, 243), -1)
        put_lines(canvas, [label, "0 / 17 / 33 / 50 / 67 / 83 / 100%"], margin, y + 35, scale=0.55, spacing=27)
        frames = progress_frames(frame_count)
        overview_region_width = columns * tile_width + (columns - 1) * gap
        overview_width = int((overview_region_width - 6 * gap) / 7)
        for index, frame_index in enumerate(frames):
            image = tile(
                read_frame(
                    capture,
                    frame_index,
                    frame_count,
                    sample_sequence if label.startswith("SAMPLE") else reference_sequence,
                ),
                f"{100 * index / 6:.0f}% f{frame_index}",
                width=overview_width,
                height=tile_height,
            )
            x = margin + label_width + index * (overview_width + gap)
            canvas[y + 16 : y + 16 + tile_height, x : x + overview_width] = image
        y += overview_height

    summary_rows: list[dict[str, object]] = []
    for candidate in candidates:
        start = int(candidate["original_frame_start"])
        end = int(candidate["original_frame_end"])
        ref_start = int(candidate["reference_original_frame_start"])
        ref_end = int(candidate["reference_original_frame_end"])
        sample_metrics = interval_measurements(sample_sequence, start, end)
        reference_metrics = interval_measurements(reference_sequence, ref_start, ref_end)
        rank = int(candidate["candidate_rank"])

        background = (244, 244, 244) if rank % 2 else (235, 241, 246)
        cv2.rectangle(canvas, (0, y), (width, y + candidate_height - 1), background, -1)
        heading = (
            f"#{rank} {candidate['component_name']} | "
            f"{float(candidate['window_start_percent']):.0f}-{float(candidate['window_end_percent']):.0f}% | "
            f"DTW {float(candidate['mean_angular_deviation_degrees']):.1f} deg"
        )
        put_lines(
            canvas,
            [heading, *metric_lines(sample_metrics, "S"), *metric_lines(reference_metrics, "R")],
            margin,
            y + 22,
            scale=0.42,
            spacing=17,
        )

        frames = frame_triplet(start, end)
        reference_frames = frame_triplet(ref_start, ref_end)
        for column, (capture, frame_index, prefix) in enumerate(
            [(sample_video, value, "S") for value in frames]
            + [(reference_video, value, "R") for value in reference_frames]
        ):
            image = tile(
                read_frame(
                    capture,
                    frame_index,
                    sample.frames if prefix == "S" else reference.frames,
                    sample_sequence if prefix == "S" else reference_sequence,
                ),
                f"{prefix} f{frame_index}",
            )
            x = margin + label_width + column * (tile_width + gap)
            canvas[y + 22 : y + 22 + tile_height, x : x + tile_width] = image

        summary_rows.append(
            {
                "sample_id": sample.sample_id,
                "reference_sample_id": reference.sample_id,
                "candidate_rank": rank,
                "component_name": candidate["component_name"],
                "window_start_percent": candidate["window_start_percent"],
                "window_end_percent": candidate["window_end_percent"],
                **{f"sample_{key}": value for key, value in sample_metrics.items()},
                **{f"reference_{key}": value for key, value in reference_metrics.items()},
            }
        )
        y += candidate_height

    if sample_video is not None:
        sample_video.release()
    if reference_video is not None:
        reference_video.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise ValueError(f"Could not write {output_path}")
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("kimore_audit_output/kimore_manifest.csv"))
    parser.add_argument("--queue", type=Path, default=Path("results/interpretable_dtw/annotation_queue.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/interpretable_dtw/pilot_review"))
    args = parser.parse_args()

    samples, _ = read_manifest(args.manifest, exercise="Es3")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    missing_samples = set(PILOT_SAMPLE_IDS).difference(sample_by_id)
    if missing_samples:
        raise ValueError(f"Pilot samples missing from manifest: {sorted(missing_samples)}")

    queue_fields, queue_rows = read_rows(args.queue)
    rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in queue_rows:
        if row["sample_id"] in PILOT_SAMPLE_IDS:
            rows_by_sample[row["sample_id"]].append(row)
    for sample_id in PILOT_SAMPLE_IDS:
        rows_by_sample[sample_id].sort(key=lambda row: int(row["candidate_rank"]))
        if len(rows_by_sample[sample_id]) != 5:
            raise ValueError(f"{sample_id}: expected five pilot candidates")

    quality_fields, quality_rows = read_rows(args.queue.parent / "component_quality.csv")
    del quality_fields
    quality_by_sample = {
        row["sample_id"]: row
        for row in quality_rows
        if row["component_index"] == "0"
    }

    selection_rows: list[dict[str, object]] = []
    pilot_queue: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for sample_id in PILOT_SAMPLE_IDS:
        sample = sample_by_id[sample_id]
        candidates = rows_by_sample[sample_id]
        reference = sample_by_id[candidates[0]["reference_sample_id"]]
        quality = quality_by_sample[sample_id]
        selection_rows.append(
            {
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "cohort": sample.cohort,
                "actual_ts": sample.score,
                "sample_quality_status": quality["sample_quality_status"],
                "retained_fraction": quality["retained_fraction"],
                "reference_sample_id": reference.sample_id,
                "selection_reason": "balanced_cohort_score_qc_and_component_pilot",
                "review_sheet": f"sheets/{sample.sample_id}.jpg",
            }
        )
        pilot_queue.extend(candidates)
        print(f"Building {sample.sample_id} -> {reference.sample_id}", flush=True)
        summary_rows.extend(
            build_sheet(
                sample,
                reference,
                candidates,
                args.output / "sheets" / f"{sample.sample_id}.jpg",
            )
        )

    write_rows(
        args.output / "pilot_selection.csv",
        list(selection_rows[0]),
        selection_rows,
    )
    write_rows(args.output / "pilot_annotation_queue.csv", queue_fields, pilot_queue)
    write_rows(
        args.output / "pilot_feature_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    print(f"Wrote {len(selection_rows)} recordings and {len(pilot_queue)} candidates to {args.output}")


if __name__ == "__main__":
    main()
