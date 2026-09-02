# KIMORE Es3 preliminary localization pilot

This report records the first-pass review of the interpretable-DTW deviation
candidates. These are technical, non-clinical annotations and are not yet
ground truth.

## Scope and evidence

- 20 Es3 recordings: four from each of `back_pain`, `expert_control`,
  `nonexpert_control`, `parkinson`, and `stroke`.
- Five DTW candidates per recording, for 100 reviewed component/interval rows.
- Selection covers low, middle, and high clinical TS values plus both QC `pass`
  and `warning` recordings.
- Each review sheet contains seven whole-recording sample/reference snapshots,
  three sample and three reference frames per candidate, and interval summaries
  of shoulder, elbow, wrist-height, knee, and torso measurements.
- Skeleton frames are mapped to RGB by normalized recording progress because
  some KIMORE streams differ by a small number of terminal frames. When a
  reference RGB stream is absent, the packet renders front/side skeleton views.
  Every selected sample recording itself has RGB evidence.

Component-specific review was used: an elbow error visible during an
`upper_arm` candidate does not make that upper-arm candidate a true positive.
Setup and completion frames were treated as correct unless the same defect was
visible persistently in the recording overview.

## First-pass result

| Label | Count |
|---|---:|
| Visible component/interval error | 40 |
| Correct / candidate false positive | 56 |
| Needs adjudication | 4 |
| Ungradable | 0 |

Among the 96 decided rows, preliminary candidate precision is `40 / 96 =
41.7%`. This number describes the top-five localization queue, not the clinical
score predictor and not QC's regression metrics.

By cohort, the decided candidate precision is:

| Cohort | Precision |
|---|---:|
| back pain | 30.0% |
| expert control | 15.8% |
| non-expert control | 55.0% |
| Parkinson | 63.2% |
| stroke | 44.4% |

The strongest localized component was `left_lower_arm` (14 errors in 16
decided candidates, 87.5%). `body_forward` produced no confirmed errors in this
pilot: 12 were judged correct and two require adjudication. The first/last 5%
of recordings also had lower yield than interior windows:

| Window position | Candidates | Decided | Errors | Precision |
|---|---:|---:|---:|---:|
| first/last 5% | 29 | 28 | 7 | 25.0% |
| interior 5--95% | 71 | 68 | 33 | 48.5% |

These observations are development hypotheses only. Boundary suppression or a
different `body_forward` rule must be validated on newly reviewed recordings,
not reported as an improvement on this same pilot.

The 40 visible errors were categorized as 17 `range_of_motion`, 15 `posture`,
three `asymmetry`, three `direction`, and two `timing`; 27 were severe and 13
moderate.

## Files and reproducibility

- Tracked first-pass labels: `annotations/kimore_es3_pilot_labels.csv`
- Review sheets and pilot selection:
  `results/interpretable_dtw/pilot_review/`
- Merged queue: `pilot_annotation_queue_preliminary.csv`
- Machine-readable summaries: `pilot_label_summary.csv`,
  `pilot_label_summary.json`, and `pilot_label_breakdown.csv`
- Blinded 20-row independent review set: `second_review_queue.csv`

Regenerate the review packet and apply the labels with:

```powershell
.\.venv\Scripts\python.exe .\src\kimore_pilot_review.py
.\.venv\Scripts\python.exe .\src\kimore_apply_pilot_labels.py
```

## Required gate before a ground-truth claim

An independent human reviewer should label `second_review_queue.csv` without
seeing the first-pass labels. After calculating agreement and adjudicating all
disagreements plus the four uncertain rows, freeze the protocol. Then annotate
a separate held-out set and report final component/interval precision and
recall there. Until that gate is complete, these labels support development and
error analysis only.
