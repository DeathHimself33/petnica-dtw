# Manual review of DTW deviation candidates

Use `results/interpretable_dtw/annotation_queue.csv` as the review queue. Each
row is one automatically selected 5%-of-recording interval and body component.
The automatic fields describe a deviation from the training-only reference;
they are not ground-truth execution-error labels.

Review the original recording or an independently rendered skeleton sequence.
Do not assign a label from the DTW angle alone. Keep the automatic candidate
columns unchanged and fill only the review columns.

## Review fields

- `review_status`: `reviewed` or `adjudication_needed`.
- `execution_label`: `correct`, `error`, `uncertain`, or `ungradable`.
- `error_type`: for an `error`, use `range_of_motion`, `direction`, `timing`,
  `asymmetry`, `compensation`, `posture`, or `other`.
- `severity`: for an `error`, use `mild`, `moderate`, or `severe`.
- `reviewer_confidence`: `low`, `medium`, or `high`.
- `annotator`: stable reviewer identifier, not a personal note.
- `review_notes`: concise visible evidence and any corrected interval/component.

Use `ungradable` when tracking, visibility, or recording quality prevents a
judgment. This is different from `correct`. Use `uncertain` when the movement is
visible but the reviewer cannot confidently decide whether it is an execution
error.

## Pilot procedure

1. Select 15--20 recordings balanced across cohort, clinical score, and QC
   status (`pass`/`warning`).
2. Review all five candidates for each selected recording.
3. Have a second reviewer independently label at least 20% of the pilot rows.
4. Resolve disagreements without seeing the model's predicted clinical score.
5. Only after the protocol is frozen, measure component and interval
   localization precision/recall on the reviewed rows.

Do not tune QC or deviation thresholds on the same rows used for the final
localization evaluation.

## Current preliminary pilot

The reproducible first-pass pilot is documented in `PILOT_LABEL_REPORT.md`.
Its labels are stored in `annotations/kimore_es3_pilot_labels.csv`, while the
generated review sheets, merged queue, summaries, and blinded 20-row
second-review set are under `results/interpretable_dtw/pilot_review/`.

These labels are explicitly non-clinical and preliminary. Do not call them
ground truth until the independent review and adjudication gate in the report
has been completed.
