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
- `src/kimore_evaluation.py`: five-fold evaluation, metrics, diagnostics, and
  output generation.
- `run_experiment.py`: command-line entry point.
- `tests/`: automated tests for the dataset, preprocessing, grouping, DTW, and
  evaluation code.

## Setup

The tested environment is Python 3.7.4 on Windows.

```powershell
py -3.7 -m venv .venv
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

Export and plot the interpretable per-component analysis:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --exercise Es3 --method interpretable_dtw
```

This produces one CSV row per held-out sample and body component under
`results/interpretable_dtw/`, plus component error and contribution
distribution plots under `figures/interpretable_dtw/`. The explanation layer
reuses the Yu--Xiong path and score without changing the comparison baseline.
It also writes `component_quality.csv`, a frame-QC component subset, and
separate QC-usable plots. Frame QC checks tracking, source-bone length,
isolated angular jumps, and Es3 anatomical direction for each of the nine
features. Internal invalid runs of at most five frames are interpolated between
valid unit-vector endpoints; unresolved invalid frames are removed from the QC
sequence. A recording is rejected only when less than 80% remains, one removed
run exceeds 10% of the recording, fewer than two usable frames remain, or a
component is almost never fully tracked. Torso-frame failures propagate to the
eight body-local limb features that depend on that frame. Raw KIMORE files are
never changed, and `component_summaries.csv` retains raw-vector results while
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
