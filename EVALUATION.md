# Result evaluation plan

## Short answer for the mentor

We will evaluate the model with five-fold subject-wise cross-validation. In
each fold, the template and the regression from DTW distance to clinical Total
Score are fitted using training subjects only. The held-out subjects are then
predicted once. After all five folds, the out-of-fold predictions are combined
and evaluated primarily with MAE and Spearman correlation, with RMSE and
Pearson correlation as secondary metrics.

This is subject-disjoint internal development cross-validation on KIMORE. It is
not an untouched confirmatory estimate and does not show generalization to a
new clinic, population or sensor setup.

## What happens inside one fold

1. Split by `subject_id`, and assert that the training and test subject sets do
   not overlap.
2. Load and preprocess each recording. Per-recording SpineBase centring and
   torso scaling use no clinical target and no other subject. For the final
   shoulder-yaw feature, however, these operations are exactly invariant and
   have no effect beyond floating-point noise.
3. Select the reference from the training indices only. The frozen plain-DTW
   rule prefers training recordings with at least 99% fully tracked shoulder
   frames, then selects the highest clinical TS within that reliable pool.
4. Extract the unwrapped shoulder-axis yaw signal, an interpretable Es3 trunk-
   rotation measurement.
5. Align every recording to the reference with exact DTW and calculate path-
   normalized aligned RMSE in degrees.
6. Fit `predicted TS = intercept + slope * DTW distance` on training rows only.
7. Apply that fixed calibration to the held-out subjects and save their sample
   IDs, actual scores, predictions, distances, fold and reference metadata.

In a future confirmatory experiment, any model choice must use subject-wise
validation inside the outer training fold or be fixed before collecting new
test data. That standard was not fully met here: fold 1 was inspected while the
99% reference threshold was selected, and the feature/exercise were developed
using KIMORE.

## Reported results

Every usable subject receives exactly one out-of-fold prediction. From the 76
pooled out-of-fold predictions we will report:

- **MAE (primary):** typical absolute error in clinical TS points.
- **Spearman correlation (primary):** whether predicted and actual clinical
  quality are ordered similarly, without requiring a perfectly linear
  relationship.
- **RMSE (secondary):** gives more weight to large prediction errors.
- **Pearson correlation (secondary):** measures linear association between
  predicted and actual TS.

We will save overall metrics, the same metrics for each fold, and an actual-
versus-predicted plot. Fold variation matters because 76 subjects is a small
sample. The pooled metrics have subject-level bootstrap 95% intervals. These
are conditional fixed-prediction intervals: they resample the saved `(actual,
predicted)` pairs but do not recreate folds, reselect references, refit
calibrations or represent method-selection uncertainty.

## Comparisons needed to interpret the numbers

A model error has no meaning without a simple comparator. Each test fold will
therefore also receive constant predictions calculated from its training fold:

- the training-score median as the natural constant baseline for MAE;
- the training-score mean as the natural constant baseline for RMSE.

The DTW model is useful only if it improves meaningfully over these no-signal
baselines and shows a stable positive rank relationship. Correlation is not
defined for constant predictions, so the constant baselines are compared on
error metrics only.

Those global constants are necessary but not sufficient because KIMORE combines
cohorts with different score distributions. After the frozen OOF result was
inspected, a stronger descriptive comparator was added: a separate median/mean
for each test subject's cohort, still calculated from outer-training rows only.
It is explicitly post hoc.

## Leakage and reproducibility checks

- Assert zero subject overlap in every outer and inner split.
- Select a separate reference inside every outer training fold.
- Fit a separate calibration inside every outer training fold.
- Keep every held-out score out of fold-specific template selection and
  calibration. This code-level property does not make the overall development
  process untouched.
- Save one row per prediction plus the exact fold, feature, reference,
  calibration coefficients and DTW path length.
- Recreate all predictions, metrics and plots with one command.

For Es3 there is currently one recording per subject, so grouping and ordinary
sample-wise splitting happen to be equivalent for this narrow experiment.
Keeping the grouping explicit is still important: it makes the safety rule
testable and remains correct when more exercises or repeated recordings are
added.

## Interpretation limits

The current feature measures shoulder-axis yaw only. It can describe rotation
timing and amplitude, but it cannot represent every posture and control factor
contained in the clinical Total Score. A weak result would therefore not prove
that DTW is generally unsuitable; it would show that this particular
single-template, single-feature baseline is insufficient.

The centring and torso scaling code is correct and useful for coordinate-based
features, but shoulder-axis yaw is invariant to translation and uniform scale.
Across all 76 Es3 recordings, raw and preprocessed yaw differed by less than
`1e-12` degrees. It is therefore wrong to imply that this preprocessing improves
the final model.

The 99% shoulder-tracking threshold applies only to reference candidates. Only
8 of 76 recordings meet it, and the selected feature still uses all frames of
every other sample, including inferred shoulder coordinates. The threshold was
chosen after fold-1 inspection, so it is both restrictive and development-set
informed.

The feature uses absolute yaw without repetition segmentation, start/end
trimming or per-recording offset normalization. First-30-frame yaw medians range
from -63.60 to +58.16 degrees; the median absolute sample/reference offset is
12.79 degrees. That may mix movement phase, resting orientation and acquisition
setup. The available diagnostic cannot prove which source dominates.

The fold-1 numbers produced on August 24 were used while inspecting and freezing
the template-quality rule. Because fold 1 remains in the pooled result, the
five-fold result is an internal development evaluation. An independently held-
out or external dataset would be required for a strong generalization claim.

## Frozen five-fold result

The complete run produced one out-of-fold prediction for each of the 76
subjects and confirmed zero subject overlap in every fold.

| Metric | Plain DTW | Subject-bootstrap 95% interval |
|---|---:|---:|
| MAE | 6.981 TS points | 5.825 to 8.283 |
| RMSE | 8.932 TS points | 7.157 to 10.804 |
| Spearman | 0.301 | 0.090 to 0.492 |
| Pearson | 0.345 | 0.129 to 0.553 |

The training-median baseline MAE was 7.342, so plain DTW reduced MAE by 0.360
points. The paired 95% interval for this reduction was -0.547 to 1.334. The
training-mean baseline RMSE was 9.521, so plain DTW reduced RMSE by 0.589
points, with a paired interval of -0.192 to 1.501. Both intervals include zero;
the apparent improvements are too uncertain to claim a reliable advantage.

The stronger post-hoc cohort constants reverse the comparison. Cohort-median
MAE is 5.846 versus 6.981 for DTW, and cohort-mean RMSE is 7.525 versus 8.932.
The DTW reductions are therefore -1.136 MAE and -1.407 RMSE points. This shows
that a substantial part of the pooled signal is cohort structure, not a robust
within-cohort movement-quality relationship. Consistently, Spearman drops from
0.301 pooled to 0.217 after mean-centring actual and predicted scores within
cohort. Conditional fixed-prediction intervals are -2.367 to 0.084 for the MAE
reduction and -2.640 to -0.030 for the RMSE reduction.

Results also varied substantially between folds. Fold-5 Spearman was 0.686,
but fold-4 Spearman was -0.018. Four folds used `E_ID7_Es3` as the reference.
Fold 4 held that subject out, selected `NE_ID13_Es3`, and produced a calibration
slope of only -0.090 rather than approximately -0.8 to -0.9. This exposes the
single-template baseline's sensitivity to reference choice.

The alignment paths are also too permissive for a clinical interpretation of
repetition consistency. Across the 76 OOF alignments, the median path has 70.34%
non-diagonal moves and is 1.365 times as long as the longer input sequence.
Unconstrained DTW can align away extra or missing repetitions, long idle periods
and irregular timing. This is intended behavior for the implemented algorithm,
but a serious limitation of the chosen methodology.

The correct conclusion is not that DTW has succeeded or failed in general. The
software is reproducible, the folds are subject-disjoint, and fold-specific
fitting uses training rows only. The predictive method is reference-sensitive,
heavily warped, partly cohort-confounded, and worse than a post-hoc cohort-aware
constant. It should not be called generally “leakage-safe” or externally
validated.
