# Petnica interpretable DTW baseline

This project builds a small, interpretable baseline for predicting the KIMORE
Exercise 3 clinical Total Score from Kinect skeleton recordings. Its folds are
subject-disjoint and its fold-specific fitting uses training rows only. It is an
internal development experiment, not an untouched validation or a claim of
clinical readiness.

## Current experiment

The pipeline:

1. selects the 76 usable Es3 recordings from the audit manifest;
2. centres each skeleton on `SpineBase` and normalizes it by median torso
   length (correct generic preprocessing, but mathematically invariant for the
   final shoulder-yaw feature and therefore irrelevant to its predictions);
3. extracts shoulder-axis yaw as an interpretable trunk-rotation signal;
4. creates five subject-disjoint outer folds;
5. selects one reliable reference from each training fold only;
6. computes exact DTW and path-normalized aligned RMSE in degrees;
7. fits a training-only linear calibration from distance to clinical TS; and
8. saves one out-of-fold prediction for every subject.

MAE and Spearman are the primary metrics. RMSE and Pearson are secondary. The
error metrics are compared with training-only constant baselines, and pooled
metrics receive conditional fixed-prediction subject-bootstrap 95% intervals.

## Dataset and citation

The raw KIMORE data are not redistributed by this repository. The official
paper describes 78 heterogeneous subjects, five rehabilitation exercises,
Kinect skeleton/video inputs and clinician questionnaire scores, and links the
depth/skeleton download:

> Capecci, M. et al. (2019), “The KIMORE Dataset: KInematic Assessment of
> MOvement and Clinical Scores for Remote Monitoring of Physical
> REhabilitation,” *IEEE TNSRE*, 27(7), 1436–1448.
> <https://doi.org/10.1109/TNSRE.2019.2923060>

The paper calls the dataset free, but this repository does not contain a
dataset license or grant redistribution rights. Check the official terms before
sharing raw files. This code repository also currently has no `LICENSE` file.

## Reproduce the run

The tested environment is Python 3.7.4 on Windows. From the project root:

```powershell
py -3.7 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\kimore_dataset_audit.py data\raw\KIMORE `
    --output-dir kimore_audit_output
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_experiment.py --exercise Es3 --method plain_dtw
```

The raw KIMORE data are deliberately not versioned. The default command
expects `kimore_audit_output/kimore_manifest.csv`, whose position paths must
point to the local read-only KIMORE files. Rerun the dataset audit if the raw
data have moved.

The experiment uses 5,000 bootstrap resamples by default. A smaller number can
be used for a quick smoke run:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --bootstrap-resamples 100
```

## Saved outputs

- `results/plain_dtw/oof_predictions.csv`: one out-of-fold prediction per
  subject, plus post-hoc cohort constants and warp/initial-window diagnostics.
- `results/plain_dtw/metrics.csv`: fold-wise and pooled model/baseline metrics.
- `results/plain_dtw/fold_metadata.json`: references, calibration coefficients,
  cohort counts and leakage checks for every fold.
- `results/plain_dtw/evaluation_summary.json`: overall metrics, conditional
  bootstrap intervals, post-hoc diagnostics, input/source-content hashes,
  environment and interpretation warning.
- `figures/plain_dtw/actual_vs_predicted.png`: prediction and residual
  diagnostic.

## Result of the frozen baseline

| Metric | Plain DTW | 95% bootstrap interval |
|---|---:|---:|
| MAE | 6.981 TS points | 5.825 to 8.283 |
| RMSE | 8.932 TS points | 7.157 to 10.804 |
| Spearman | 0.301 | 0.090 to 0.492 |
| Pearson | 0.345 | 0.129 to 0.553 |

The training-median constant has MAE 7.342, and the training-mean constant has
RMSE 9.521. Plain DTW improves those errors by only 0.360 and 0.589 points. The
paired 95% intervals for both improvements cross zero, so this run does not
show a reliable advantage over constant prediction.

That comparison is too weak on its own because cohort labels already carry
substantial score information. A post-hoc training-fold cohort-specific
constant comparison gives:

| Error target | Plain DTW | Global training constant | Post-hoc cohort constant |
|---|---:|---:|---:|
| MAE | 6.981 | 7.342 (median) | 5.846 (cohort median) |
| RMSE | 8.932 | 9.521 (mean) | 7.525 (cohort mean) |

Plain DTW is worse than the cohort constants by 1.136 MAE points and 1.407 RMSE
points. The fixed-prediction bootstrap interval for the RMSE reduction is
entirely below zero (-2.640 to -0.030); the MAE interval narrowly crosses zero
(-2.367 to 0.084). Because this comparator was added after inspecting the frozen
predictions, it is labeled post hoc rather than presented as a preregistered
test.

## Known limitations

- This is internal development cross-validation on KIMORE, not untouched or
  external validation. Fold 1 was inspected while the 99% template-quality
  threshold was chosen, and the feature/exercise were developed on KIMORE.
- The 76-subject dataset is small and combines heterogeneous cohorts.
- Pooled correlations partly reflect between-cohort structure: Spearman falls
  from 0.301 pooled to 0.217 after mean-centring within cohort. Cohort-specific
  biases range from -8.295 to +6.229 TS points.
- One shoulder-yaw signal describes rotation amplitude and timing but not every
  posture/control component in clinical TS.
- Absolute shoulder yaw is used without repetition segmentation, trimming or
  per-recording offset normalization. The first-30-frame sample/reference
  offset has a 12.79-degree median and a 56.71-degree 95th percentile, so the
  feature may mix movement phase, resting orientation and acquisition setup.
- Exact DTW is unconstrained. A median 70.34% of path moves are non-diagonal and
  median path length is 1.365 times the longer sequence, so the aligner can hide
  extra/missing repetitions, idle periods and irregular timing.
- The 99% reference threshold admits only 8 of 76 recordings in the full
  cohort. It controls template selection, but all feature frames—including
  inferred shoulder coordinates—are still used for other recordings.
- SpineBase centring and torso scaling are useful for coordinate features but
  change the final shoulder-yaw feature by less than numerical precision; they
  do not improve these predictions.
- The method is sensitive to the chosen reference. Four folds used
  `E_ID7_Es3`; when that subject was held out, fold 4 used `NE_ID13_Es3` and its
  calibration relationship almost disappeared.
- Predictions are compressed toward the middle of the score range, producing
  especially large overestimates for some low-scoring Parkinson's recordings.
- Bootstrap intervals condition on the 76 saved out-of-fold prediction pairs;
  they do not refit folds, references or calibrations and do not include method-
  selection uncertainty.
- Exact unconstrained DTW is quadratic in the two sequence lengths.

These limitations should guide later work, but they should not be tuned away
using the same outer test predictions.

## Project map

- `src/kimore_dataset.py`: manifest selection and skeleton loading.
- `src/kimore_preprocessing.py`: centring and body-size normalization.
- `src/kimore_grouping.py`: subject-wise folds and leakage assertion.
- `src/kimore_dtw.py`: exact DTW and aligned RMSE.
- `src/kimore_plain_dtw.py`: feature, reference rule and linear calibration.
- `src/kimore_evaluation.py`: cross-validation, metrics, bootstrap and outputs.
- `run_experiment.py`: reproducible command-line entry point.
- `EVALUATION.md`: detailed evaluation rationale.
- `notes.md`: daily decisions, evidence and results.
