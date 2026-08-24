# Result evaluation plan

## Short answer for the mentor

We will evaluate the model with five-fold subject-wise cross-validation. In
each fold, the template and the regression from DTW distance to clinical Total
Score are fitted using training subjects only. The held-out subjects are then
predicted once. After all five folds, the out-of-fold predictions are combined
and evaluated primarily with MAE and Spearman correlation, with RMSE and
Pearson correlation as secondary metrics.

This is an internal cross-validation estimate on KIMORE, not proof that the
method generalizes to a new clinic or sensor setup.

## What happens inside one fold

1. Split by `subject_id`, and assert that the training and test subject sets do
   not overlap.
2. Load and preprocess each recording. Per-recording SpineBase centring and
   torso scaling use no clinical target and no other subject.
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

If a later model choice is tuned rather than fixed in advance, that choice
must use subject-wise validation inside the outer training fold. The outer test
subjects must not choose a signal, filter, threshold, template rule or model.

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
sample. Subject-level bootstrap 95% confidence intervals should be added for
the pooled metrics rather than reporting point estimates alone.

## Comparisons needed to interpret the numbers

A model error has no meaning without a simple comparator. Each test fold will
therefore also receive constant predictions calculated from its training fold:

- the training-score median as the natural constant baseline for MAE;
- the training-score mean as the natural constant baseline for RMSE.

The DTW model is useful only if it improves meaningfully over these no-signal
baselines and shows a stable positive rank relationship. Correlation is not
defined for constant predictions, so the constant baselines are compared on
error metrics only.

## Leakage and reproducibility checks

- Assert zero subject overlap in every outer and inner split.
- Select a separate reference inside every outer training fold.
- Fit a separate calibration inside every outer training fold.
- Keep every held-out score out of template selection and calibration.
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

The fold-1 numbers produced on August 24 are a pipeline preview. They were used
while inspecting and freezing the template-quality rule, so they must not be
presented as an untouched performance estimate. The frozen five-fold run is an
internal development evaluation, and an independent external dataset would
still be required for a strong generalization claim.

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

Results also varied substantially between folds. Fold-5 Spearman was 0.686,
but fold-4 Spearman was -0.018. Four folds used `E_ID7_Es3` as the reference.
Fold 4 held that subject out, selected `NE_ID13_Es3`, and produced a calibration
slope of only -0.090 rather than approximately -0.8 to -0.9. This exposes the
single-template baseline's sensitivity to reference choice.

The correct conclusion is not that DTW has succeeded or failed in general. The
thin pipeline is reproducible and leakage-safe, but this single-feature,
single-template version does not outperform constant prediction reliably.
