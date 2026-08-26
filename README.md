# Petnica interpretable DTW baseline

This repository contains a small, interpretable baseline for predicting the
KIMORE Exercise 3 clinical Total Score from Kinect skeleton recordings. It uses
subject-disjoint cross-validation, shoulder-axis yaw, exact dynamic time
warping, and a training-only linear calibration.

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

The experiment expects `kimore_audit_output/kimore_manifest.csv`. Generated
results and figures are written under `results/` and `figures/`; both directories
are ignored by Git.
