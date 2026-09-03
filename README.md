# Petnica interpretable DTW baselines

This repository contains a small, interpretable baseline for predicting the
KIMORE Exercise 3 clinical Total Score from Kinect skeleton recordings. It uses
subject-disjoint cross-validation, shoulder-axis yaw, exact dynamic time
warping, and a training-only linear calibration. It also includes a Yu--Xiong
angular-DTW comparison baseline based on nine skeleton vectors.

The project is an internal development experiment, not a clinically validated
model. Raw KIMORE data and generated experiment outputs are not included.

## Repository structure

- `src/kimore_dataset_audit.py`: audits the local KIMORE dataset and creates a
  manifest.
- `src/kimore_dataset.py`: selects manifest rows and loads JointPosition files.
- `src/kimore_preprocessing.py`: SpineBase centring and torso-length scaling.
- `src/kimore_grouping.py`: subject-wise folds and leakage assertions.
- `src/kimore_dtw.py`: exact DTW and path-normalized aligned RMSE.
- `src/kimore_plain_dtw.py`: shoulder-yaw feature, reference rule, and linear
  calibration.
- `src/kimore_yu_xiong_dtw.py`: Yu--Xiong body-local bone vectors, angular DTW,
  and the paper's percentage-score equation.
- `src/kimore_yu_xiong_evaluation.py`: subject-disjoint KIMORE adaptation and
  evaluation of the Yu--Xiong baseline.
- `src/kimore_interpretable_dtw.py`: per-component explanation layer over the
  unchanged Yu--Xiong alignment and score.
- `src/kimore_interpretable_quality.py`: per-vector tracking, source-length,
  temporal-continuity, and anatomical plausibility checks.
- `src/kimore_interpretable_evaluation.py`: held-out component CSV export and
  component-distribution plots.
- `src/kimore_pilot_review.py`: balanced Es3 pilot selection and synchronized
  RGB/reference review-sheet generation.
- `src/kimore_apply_pilot_labels.py`: strict label validation, queue merge,
  summary export, and blinded second-review sampling.
- `src/kimore_evaluation.py`: five-fold evaluation, metrics, diagnostics, and
  output generation.
- `annotations/kimore_es3_pilot_labels.csv`: versioned preliminary first-pass
  labels for the 100-row localization pilot.
- `run_experiment.py`: command-line entry point.
- `tests/`: automated tests for the dataset, preprocessing, grouping, DTW, and
  evaluation code.

## Setup

The tested environment is Python 3.13.14 on Windows.

```powershell
py -V:3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset

The raw KIMORE dataset is deliberately excluded from Git. The official dataset
paper is:

> Capecci, M. et al. (2019), "The KIMORE Dataset: KInematic Assessment of
> MOvement and Clinical Scores for Remote Monitoring of Physical
> REhabilitation," *IEEE TNSRE*, 27(7), 1436-1448.
> <https://doi.org/10.1109/TNSRE.2019.2923060>

Create a local manifest from the raw dataset:

```powershell
.\.venv\Scripts\python.exe src\kimore_dataset_audit.py data\raw\KIMORE `
    --output-dir kimore_audit_output
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Run the experiment

```powershell
.\.venv\Scripts\python.exe run_experiment.py --exercise Es3 --method plain_dtw
```

Run the Yu--Xiong comparison baseline on the same exercise:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --exercise Es3 --method yu_xiong_dtw
```

Run the full interpretable pipeline for all five exercises:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --exercise all --method interpretable_dtw
```

The all-exercise command is intentionally the QC/interpretable variant. The
strict Yu--Xiong comparator can reject a whole recording when even one Kinect
frame contains a zero-length body axis or limb, whereas the interpretable
pipeline records, repairs, or removes those frames transparently.

The all-exercise run creates one shared subject-to-fold assignment and reuses it
for Es1--Es5. This is required for later multi-exercise learning: a person can
never be in the training side for one exercise and the corresponding test side
for another. Exercise-specific artifacts are written to
`results/interpretable_dtw/all_exercises/Es1/` through `Es5/`; combined
prediction, candidate, interval, and quality CSVs plus `subject_folds.json` and
`all_exercises_summary.json` are written one directory above them. Default
single-exercise outputs remain separately exercise-scoped, for example
`results/interpretable_dtw/Es3/`. A standalone run therefore cannot overwrite
an Es3 result produced under the shared all-exercise fold protocol.

Export and plot the interpretable per-component analysis:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --exercise Es3 --method interpretable_dtw
```

This produces one CSV row per held-out sample and body component under
`results/interpretable_dtw/`, plus component error and contribution
distribution plots under `figures/interpretable_dtw/`. The explanation layer
reuses the Yu--Xiong path and score without changing the comparison baseline.
It also writes `component_quality.csv`, a frame-QC component subset, separate
QC-usable plots, and a leakage-safe QC prediction evaluation. For every outer
fold, both raw and QC paper-score calibrations are fitted only on the same
QC-usable training subjects and evaluated on the same QC-usable held-out
subjects. The paired outputs are `oof_predictions.csv`, `metrics.csv`,
`fold_metadata.json`, and `evaluation_summary.json`; accuracy must be reported
together with the retained-subject coverage.

The QC run also exports `error_timeline.csv`, with one row for every retained
original sample frame and body component. Each row records the mean and maximum
aligned angular deviation, the original sample frame, and the aligned reference
frame range. `top_deviation_intervals.csv` selects the five largest fixed 5%
progress windows per recording. These are explicitly candidate deviations from
the reference, not validated execution errors. `annotation_queue.csv` copies
those candidates and adds blank human-review fields (`execution_label`,
`error_type`, `severity`, confidence, annotator, and notes) for a manual pilot.
The allowed values and review procedure are defined in `ANNOTATION_GUIDE.md`.

Generate the balanced 20-recording visual pilot and apply the versioned
first-pass labels:

```powershell
.\.venv\Scripts\python.exe .\src\kimore_pilot_review.py
.\.venv\Scripts\python.exe .\src\kimore_apply_pilot_labels.py
```

The result and its limitations are documented in `PILOT_LABEL_REPORT.md`.
The first pass contains 100 candidates and is explicitly non-clinical; an
independent reviewer must complete the generated blinded
`second_review_queue.csv` before any ground-truth claim.

Frame QC checks tracking, source-bone length, and isolated angular jumps for all
exercises. Es3 additionally uses its existing torso-up and leg-direction
plausibility checks. The other exercises deliberately do not reuse the Es3
anatomical rule until exercise-specific rules are defined. Internal invalid runs
of at most five frames are interpolated between valid unit-vector endpoints;
unresolved invalid frames are removed from the QC sequence. A recording is
rejected only when less than 80% remains, one removed run exceeds 10% of the
recording, fewer than two usable frames remain, or a component is almost never
fully tracked. Torso-frame failures propagate to the eight body-local limb
features that depend on that frame. Raw KIMORE files are never changed, and
`component_summaries.csv` retains raw-vector results while
`component_summaries_qc_usable.csv` contains the repaired/trimmed results.

The method follows Yu and Xiong's eight body-local limb vectors plus body
forward vector, angular multidimensional DTW, and Equation (5) score. KIMORE
does not include the virtual-coach recording assumed by the original method,
so each outer fold selects a high-score, well-tracked coach from training only
and fits the paper's expert-score calibration using that fold's training rows.
The highest training TS is the primary coach criterion; required-joint tracking
and sample ID resolve ties.
This is a leakage-safe dataset adaptation, not an exact replication of the
original Tai Chi experiment.

> Yu, X. and Xiong, S. (2019), "A Dynamic Time Warping Based Algorithm to
> Evaluate Kinect-Enabled Home-Based Physical Rehabilitation Exercises for
> Older People," *Sensors*, 19(13), 2882.
> <https://doi.org/10.3390/s19132882>

The experiment expects `kimore_audit_output/kimore_manifest.csv`. Generated
results and figures are written under `results/` and `figures/`; both directories
are ignored by Git.

## ML-ready sequence export

The first temporal-ML data layer reuses the same nine Yu--Xiong unit vectors,
frame QC, and shared subject folds across Es1--Es5. Export 128-step tensors with:

```powershell
.\.venv\Scripts\python.exe .\src\kimore_ml_data.py
```

The generated NPZ contains `(sample, time, 9, 3)` features, frame and component
observation masks, targets, exercise indices, and one-based outer-fold numbers.
It is deliberately not globally standardized. During evaluation,
`FeatureStandardizer` must be fitted separately on each fold's training rows so
no held-out subject influences preprocessing.
