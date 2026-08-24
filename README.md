# Petnica interpretable DTW baseline

This project builds a small, leakage-safe baseline for predicting the KIMORE
Exercise 3 clinical Total Score from Kinect skeleton recordings. The goal is an
understandable reference implementation, not a claim of clinical readiness.

## Current experiment

The pipeline:

1. selects the 76 usable Es3 recordings from the audit manifest;
2. centres each skeleton on `SpineBase` and normalizes it by median torso
   length;
3. extracts shoulder-axis yaw as an interpretable trunk-rotation signal;
4. creates five subject-disjoint outer folds;
5. selects one reliable reference from each training fold only;
6. computes exact DTW and path-normalized aligned RMSE in degrees;
7. fits a training-only linear calibration from distance to clinical TS; and
8. saves one out-of-fold prediction for every subject.

MAE and Spearman are the primary metrics. RMSE and Pearson are secondary. The
error metrics are compared with training-only constant baselines, and pooled
metrics receive paired subject-bootstrap 95% intervals.

## Reproduce the run

The tested environment is Python 3.7.4 on Windows. From the project root:

```powershell
py -3.7 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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

- `results/plain_dtw/oof_predictions.csv`: one held-out prediction per subject.
- `results/plain_dtw/metrics.csv`: fold-wise and pooled model/baseline metrics.
- `results/plain_dtw/fold_metadata.json`: references, calibration coefficients,
  cohort counts and leakage checks for every fold.
- `results/plain_dtw/evaluation_summary.json`: overall metrics, bootstrap
  intervals, manifest hash, environment and interpretation warning.
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

This is still a useful baseline result: the end-to-end method works without
subject leakage, but one shoulder-yaw signal and one fold-specific template do
not capture clinical quality robustly enough.

## Known limitations

- This is internal development cross-validation on KIMORE, not external
  validation. Fold 1 was inspected while the template-quality rule was being
  frozen.
- The 76-subject dataset is small and combines heterogeneous cohorts.
- One shoulder-yaw signal describes rotation amplitude and timing but not every
  posture/control component in clinical TS.
- The method is sensitive to the chosen reference. Four folds used
  `E_ID7_Es3`; when that subject was held out, fold 4 used `NE_ID13_Es3` and its
  calibration relationship almost disappeared.
- Predictions are compressed toward the middle of the score range, producing
  especially large overestimates for some low-scoring Parkinson's recordings.
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
