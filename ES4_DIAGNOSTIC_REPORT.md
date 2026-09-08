# Es4: post-hoc analysis of weaker prediction performance

Follow-up: `ES2_ES4_DATA_AUDIT.md` identifies internal workbook-ID
inconsistencies, retained discontinuities and a separate pelvis-translation
feature gap. Its findings extend the body-up-only follow-up proposed here.

## Conclusion

The clearest observed problem is compression of predicted scores toward the
middle, together with sensitivity to the subject split and random seed.
Current QC indicators and measured 128-frame reconstruction distortion do not
explain larger errors within the retained Es4 population. The feature mapping
also has a verified blind spot for rigid body roll about the forward axis;
its contribution to Es4 prediction error remains an untested hypothesis.

No model, checkpoint, fold assignment, QC rule or prediction was changed.
This is exploratory analysis of previously inspected out-of-fold results,
not a new independent evaluation or a causal explanation.

## Data and reproducibility

Run `python analyze_es4_diagnostics.py` from the repository root with numpy,
pandas and Pillow installed. The script reads the frozen five-seed ensemble,
the component QC exports and the raw manifest/JointPosition recordings.
It writes tables, a JSON summary and skeleton contact sheets to the ignored
`results/es4_diagnostics/` directory. It verifies unique sample IDs, ensemble
arithmetic, QC merge coverage and the reconstructed training-score means.

The analysis used 66 Es4 recordings from 66 subjects, drawn from 71 recordings
in the Es4 QC export (five failed QC). Across all exercises there are 338
retained recordings. The bundled Python 3.12 runtime was used because the
existing Python 3.13 virtual environment could not launch; it was not modified.

## 1. Performance and score compression

| Exercise | N | Ensemble MAE | Pearson | Mean prediction SD across seeds |
|---|---:|---:|---:|---:|
| Es1 | 65 | 4.367 | 0.771 | 2.666 |
| Es2 | 70 | 5.947 | 0.799 | 3.036 |
| Es3 | 68 | 4.609 | 0.762 | 2.984 |
| Es4 | 66 | 6.754 | 0.565 | 4.201 |
| Es5 | 69 | 5.351 | 0.730 | 3.566 |

Es4 target SD is 9.94, versus prediction SD 6.61. The descriptive slope of
prediction against actual score is 0.376. Overall signed bias is only +0.78,
but opposite errors at the two ends cancel:

| Actual TS band | N | MAE | Mean prediction minus actual |
|---|---:|---:|---:|
| 0–20 inclusive | 5 | 12.37 | +12.37 |
| >20–30 | 22 | 6.32 | +6.16 |
| >30–40 | 17 | 5.93 | -0.28 |
| >40–50 | 22 | 6.55 | -6.42 |

Only 2–4 Es4 examples with TS <=20 are available for fitting in each split.
Each split fits on 36–42 Es4 recordings; a separate shared fold supplies early
stopping. This sparse low-score coverage is consistent with difficulty at
the low end, but does not alone prove its cause.

## 2. Splits and seeds

| Test fold | N | MAE | Pearson | Es4 fitting examples | Fitting TS <=20 |
|---|---:|---:|---:|---:|---:|
| 1 | 14 | 4.34 | 0.800 | 40 | 4 |
| 2 | 12 | 4.75 | 0.757 | 42 | 3 |
| 3 | 12 | 8.33 | 0.506 | 42 | 2 |
| 4 | 12 | 9.00 | 0.435 | 38 | 2 |
| 5 | 16 | 7.50 | 0.434 | 36 | 4 |

Test-fold target means are similar (31.97–34.56), so a large shift in mean
score alone does not explain this pattern. Fold membership also changes
fitting and validation subjects; these effects cannot be separated here.
Individual seed Es4 MAEs range from 6.68 to 8.16. Es4 has the largest mean
per-recording prediction SD across seeds (4.20). That SD is not a calibrated
uncertainty estimate: its correlation with absolute error is -0.10.

Code inspection shows early stopping uses pooled validation MAE across all
five exercises, not Es4-specific validation MAE. This is a plausible source
of compromise between tasks, not evidence that a different checkpoint would
generalize better. Saved histories do not provide per-exercise validation
trajectories, so this cannot be resolved from the aggregate history alone.

## 3. Cohorts and individual errors

| Cohort | N | MAE | Signed bias |
|---|---:|---:|---:|
| Back pain | 5 | 3.44 | +0.08 |
| Expert control | 13 | 6.28 | -4.41 |
| Nonexpert control | 23 | 7.20 | +0.10 |
| Parkinson | 16 | 7.33 | +4.56 |
| Stroke | 9 | 7.10 | +3.69 |

Errors occur across cohorts. These small groups have different score
distributions, so this table does not establish a cohort-specific mechanism.

| Example | Actual TS | ML | QC-DTW | QC |
|---|---:|---:|---:|---|
| P_ID9_Es4 | 20.00 | 41.06 | 33.75 | pass |
| P_ID11_Es4 | 15.33 | 34.13 | 33.78 | warning |
| NE_ID16_Es4 | 50.00 | 34.16 | 30.90 | warning |
| S_ID6_Es4 | 33.24 | 17.88 | 33.60 | warning |
| NE_ID12_Es4 | 27.00 | 42.05 | 34.46 | warning |

The ten worst recordings contribute 34.8% of total absolute error. Removing
only the worst descriptively changes MAE from 6.75 to 6.53, and removing the
worst three to 6.19. This is not one isolated outlier. Such removals are
diagnostic only and must not replace the reported full-population metrics.

## 4. QC and comparison with DTW

QC-pass recordings (N=32) have MAE 6.755; warnings (N=34) have MAE 6.753.
Pearson correlations of absolute ML error with mean tracking fraction,
dropped-frame fraction and interpolated-frame fraction are +0.037, -0.147
and -0.042 respectively. Rank correlations are also small. This provides
no clear evidence that the measured QC problems explain retained-set errors.
It does not establish that all retained skeletons are accurate or tell us
how the model would perform on the five excluded recordings.

Es4 QC-DTW MAE is 8.485; the fitting-only exercise-mean baseline is 8.346.
ML improves over both, but DTW provides almost no gain over predicting a mean.
ML and DTW signed residuals correlate at 0.763, whereas their absolute errors
correlate at only 0.295; just three of their ten worst examples overlap.
Shared actual targets and compression toward the middle contribute to signed
residual correlation, so it is not independent proof of a shared data defect.
S_ID6 is a concrete example where ML fails and DTW is close to the target.

## 5. Visual inspection and representation

The contact sheet shows seven equally spaced raw skeleton frames for the three
worst examples and two lowest-error examples (P_ID7, NE_ID22). Blue bones have
fully tracked endpoints; orange bones have at least one endpoint not fully
tracked. This is a technical frontal skeleton inspection, not clinical RGB
review or a new expert annotation.

P_ID11 and the accurately predicted P_ID7 both show inferred limb positions
and unusual limb configurations. P_ID9 has a large error despite passing QC
and showing substantial visible trunk movement. Good predictions also occur
with warnings. These selected snapshots support inspecting tracking locally,
but do not identify a unique visual defect that explains the worst scores.
Seven snapshots cannot assess all repetitions or timing.

The code represents eight limb directions in a body-local coordinate frame,
plus one world-space forward vector. A numerical check applies a rigid
20-degree rotation to an actual skeleton frame about its own forward axis:
all nine features remain unchanged to floating-point precision (maximum
absolute difference 3.89e-16). This verifies that absolute roll is not fully
identifiable from these features. Real trunk bending can still change relative
limb directions, so the representation does not erase all bending information.
Explicit body-up/torso orientation is a justified candidate for a future
ablation; this check alone does not prove it would improve Es4 scores.

## 6. Resampling

The model takes the nearest 128 uniformly spaced progress frames and masks
unusable frames; it does not receive absolute duration as an explicit input.
For each retained Es4 sequence, the diagnostic reconstructs repaired unit
vectors from the usable sampled grid by linear interpolation and normalization,
then measures angular difference on QC-retained original frames.

The median recording mean difference is 1.29 degrees; the median recording
95th-percentile difference is 4.51 degrees. Mean reconstruction difference
correlates with absolute prediction error at -0.009. Original frame count
correlates with absolute error at +0.038. Thus these diagnostics do not point
to resampling distortion as the main explanation. They do not test the effect
of lost absolute duration, brief events, aliasing or model sensitivity, and
are not a 128-versus-256 training ablation.

## Recommended follow-up

1. Retain the frozen reported result; use these findings in the discussion.
2. Ask the expert specifically to inspect the worst examples and matched good
   examples, especially P_ID9 and the high-score NE_ID16. Do not alter clinical
   labels on the basis of model disagreement.
3. If further modeling is in scope, predefine an ablation adding explicit
   torso/body-up orientation; investigate per-exercise validation diagnostics.
   Treat a duration feature or denser sampling as secondary hypotheses.
4. Any selection motivated by this analysis needs nested or newly held-out
   evaluation for a confirmatory improvement claim. Reusing these OOF errors
   for selection and reporting the same folds as independent evidence is invalid.
