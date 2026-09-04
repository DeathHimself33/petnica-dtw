# Multi-seed temporal ML baseline report

## Scope

This is an internal, post-hoc sensitivity analysis of the compact
TCN--BiGRU--attention model. It uses five random seeds with the same five
subject-disjoint outer folds. For every outer test fold, the next shared fold
is used for early stopping and the remaining three folds are used for fitting.
The test fold is never used for fitting, standardization, or checkpoint
selection.

The evaluation contains 338 QC-usable recordings from 76 subjects across
Es1--Es5. This is 92.9% of the 364 usable-target recordings in the
all-exercise DTW experiment. All ML runs and QC-DTW comparisons contain the
same sample IDs, targets, and fold assignments.

## Repeated-seed results

| Seed | MAE | RMSE | Pearson | Spearman |
| ---: | ---: | ---: | ---: | ---: |
| 20260903 | 6.071 | 7.758 | 0.694 | 0.622 |
| 20260904 | 6.087 | 7.836 | 0.693 | 0.641 |
| 20260905 | 5.673 | 7.361 | 0.741 | 0.679 |
| 20260906 | 6.130 | 7.832 | 0.681 | 0.645 |
| 20260907 | 5.927 | 7.789 | 0.688 | 0.647 |
| Mean | 5.977 | 7.715 | 0.700 | 0.647 |
| Sample SD | 0.186 | 0.200 | 0.024 | 0.020 |

The small across-seed dispersion supports the claim that the result is not
driven by one favorable initialization. It does not replace evaluation on an
independent external dataset.

## Five-seed ensemble

The ensemble is the unweighted mean of the five OOF predictions for each
recording.

| Method | MAE | RMSE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Five-seed temporal ensemble | **5.410** | **6.953** | **0.754** | **0.700** |
| QC-DTW | 7.943 | 10.015 | 0.298 | 0.349 |
| Training-only exercise mean | 8.339 | 10.161 | 0.234 | 0.219 |

Relative to QC-DTW, the ensemble reduces MAE by 31.9% and RMSE by 30.6%.
Relative to the training-only exercise mean, it reduces MAE by 35.1% and RMSE
by 31.6%.

Ten thousand paired bootstrap resamples were grouped by subject so that a
person's exercises remain together. Positive values favor the ensemble:

| Comparator | MAE reduction (95% CI) | RMSE reduction (95% CI) |
| --- | ---: | ---: |
| QC-DTW | 2.533 (1.603, 3.509) | 3.062 (1.916, 4.225) |
| Training-only exercise mean | 2.930 (1.981, 3.939) | 3.209 (2.037, 4.400) |

## Per-exercise ensemble result

| Exercise | N | MAE | RMSE | Pearson | Spearman | MAE gain over QC-DTW (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Es1 | 65 | 4.367 | 5.615 | 0.771 | 0.636 | 2.464 (1.095, 3.887) |
| Es2 | 70 | 5.947 | 7.349 | 0.799 | 0.712 | 4.042 (2.219, 5.939) |
| Es3 | 68 | 4.609 | 6.173 | 0.762 | 0.645 | 1.756 (0.577, 2.957) |
| Es4 | 66 | 6.754 | 8.235 | 0.565 | 0.559 | 1.731 (0.247, 3.195) |
| Es5 | 69 | 5.351 | 7.062 | 0.730 | 0.722 | 2.602 (1.255, 4.020) |

Es4 remains the weakest and most seed-sensitive exercise, although averaging
the seeds improves it materially and its paired MAE interval over QC-DTW no
longer crosses zero. The ensemble produces no scores outside the clinical
0--50 range; individual runs produced two such predictions.

## Reproduce the aggregation

```powershell
.\.venv\Scripts\python.exe .\analyze_ml_results.py `
    --run-dir results\ml_baseline\tcn_bigru_attention `
    --run-dir results\ml_baseline\tcn_bigru_attention_seed_20260904 `
    --run-dir results\ml_baseline\tcn_bigru_attention_seed_20260905 `
    --run-dir results\ml_baseline\tcn_bigru_attention_seed_20260906 `
    --run-dir results\ml_baseline\tcn_bigru_attention_seed_20260907
```

The analysis validates identical OOF populations, rejects duplicate seeds,
records input hashes, and writes ensemble predictions, seed metrics, and
bootstrap intervals to `results/ml_baseline/multiseed_analysis/`.

## Interpretation limits

- This analysis was designed after inspecting the original single-seed result.
- The ensemble uses the same fixed outer folds and is not external validation.
- Results apply to the 338 QC-usable recordings, not the 26 QC-failed ones.
- Hyperparameter changes motivated by these OOF results need a nested or newly
  held-out evaluation before they can support a confirmatory claim.
- The model predicts existing KIMORE clinical Total Scores; the pending expert
  review of localized candidate intervals is a separate validation task.
