# Two-day revision and explanation guide

The goal is not to memorize the files. The goal is to be able to reconstruct
the logic: what problem each step solves, what information it is allowed to
use, what it outputs and what its limitation is.

## Day 1: understand the complete pipeline

### First pass: data flow without code

Explain this sequence aloud from memory:

`manifest -> loader -> preprocessing -> shoulder yaw -> subject folds -> training reference -> DTW distance -> training calibration -> held-out prediction -> metrics`

For every arrow, answer:

1. What is the input shape or object?
2. What transformation happens?
3. Why is it needed?
4. Could it leak test information?
5. What is saved or passed to the next step?

### Second pass: walk through the files

1. `src/kimore_dataset.py`
   - Explain why one CSV frame has 100 values.
   - Explain the reshape to `(frames, 25, 4)` and separation into XYZ plus
     tracking state.
   - Explain why 76 Es3 samples are usable and why two are excluded.
   - Explain why the audit has 74 fully clean Es3 rows but the position-based
     loader has 76 usable rows: missing orientation alone does not block this
     model.
   - Explain how correctly named repeated clinical workbooks are compared and
     why E_ID3's misfiled E_ID1 workbooks are ignored with a warning.
2. `src/kimore_preprocessing.py`
   - Demonstrate NumPy broadcasting when subtracting `SpineBase`.
   - Explain median torso length, MAD and one scale per recording.
   - Explain why translation and scale do not remove trunk rotation.
3. `src/kimore_grouping.py`
   - Trace one fold's training and test indices.
   - Explain the subject-set intersection assertion.
   - Be ready to admit that Es3 currently has one sample per subject, so GroupKFold
     and sample-wise splitting are equivalent here; explicit grouping protects
     later multi-exercise/repeated-recording work.
4. `src/kimore_dtw.py`
   - Draw the accumulated-cost grid for two tiny sequences.
   - Explain diagonal, vertical and horizontal predecessor choices.
   - Explain why the path begins at both first frames and ends at both last
     frames.
   - Reproduce `sqrt(total squared aligned error / path length)`.
5. `src/kimore_plain_dtw.py`
   - Explain shoulder-axis yaw and angle unwrapping.
   - Explain the 99% tracking eligibility rule and its fallback.
   - Explain the line `predicted TS = intercept + slope * distance`.
6. `src/kimore_evaluation.py`
   - Follow one fold: reference, distances, calibration, test predictions and
     constants.
   - Explain why every subject must occur in exactly one outer test fold.
   - Explain paired subject bootstrap resampling.
7. `run_experiment.py`
   - Run it and point to each saved output.

End Day 1 by running all 33 tests and explaining what failure each test would
catch. Do not just state that they pass.

## Day 2: explain and defend the result

### A 90-second project explanation

> I built an interpretable baseline for KIMORE Exercise 3 using 76 usable
> subjects. Each skeleton recording is centred on SpineBase and normalized by
> median torso length. I extract shoulder-axis yaw because Es3 measures trunk
> rotation, then use exact DTW to align recordings performed at different
> speeds. Distance is path-normalized aligned RMSE rather than raw accumulated
> cost. Centring and scaling are useful generic preprocessing, but shoulder yaw
> is invariant to both, so they do not change this model's signal.
>
> Evaluation uses five subject-wise folds. In every fold, both the reference
> execution and the linear mapping from distance to clinical Total Score use
> training subjects only. Every subject receives one out-of-fold prediction.
>
> The baseline achieved MAE 6.98 TS points and pooled Spearman 0.30. It was only
> 0.36 MAE points better than a global training-median constant, with an
> interval crossing zero. More importantly, a post-hoc training-cohort median
> achieved MAE 5.85, so the DTW model was worse than a cohort-aware constant.
> Within-cohort mean-centred Spearman fell to 0.22. Paths were also very
> permissive: a median 70% of moves were non-diagonal. Therefore the software is
> reproducible and subject-disjoint, but this predictive method is unstable,
> overwarped and partly cohort-confounded.

### Likely mentor questions

**Why Exercise 3?**

It has 76 usable subjects across all cohorts, a useful TS range, few exclusions
and an interpretable repeated trunk-rotation movement that DTW can align.

**Why centre on SpineBase?**

It removes camera-relative translation. Subtracting the same origin from every
joint preserves distances and directions between joints. For shoulder-axis yaw,
this operation makes no numerical difference, so it is pipeline consistency
rather than a source of predictive improvement.

**Why median torso length?**

The diagnostic showed lower relative MAD and better tracking availability than
shoulder width. One median scale per recording reduces body-size differences
for coordinate features without making the skeleton expand and contract frame
by frame. Uniform scaling does not change shoulder yaw, so it has no effect on
this model's final signal.

**Why no frame-wise rotation?**

Es3 is trunk rotation. Rotating every frame into a common body orientation
could remove the movement being measured.

**Why no resampling to one length?**

Different duration is the main reason to use DTW. Resampling would add another
transformation. However, the path diagnostic shows that unconstrained DTW went
too far in the other direction: 70.34% of path moves were non-diagonal at the
median.
So “DTW handles duration” is not a complete defence; a future constrained,
penalized or repetition-segmented method needs genuinely new evaluation.

**Why no global smoothing?**

Most measurements were clean and some inferred runs were long. Blanket
smoothing could alter valid movement and cannot reconstruct long unreliable
segments. The baseline keeps this as a documented limitation.

**Why shoulder-axis yaw?**

It directly describes horizontal trunk/shoulder rotation and is easy to plot
and explain. Its limitation is equally important: it does not encode every
posture and control factor included in clinical TS.

**Why path-normalized RMSE?**

Raw DTW accumulated cost generally grows with path length. Dividing squared
error by the number of aligned pairs and taking the square root expresses a
typical aligned error in degrees.

**How is leakage prevented?**

Training and test subject sets are asserted disjoint. Each fold selects its
reference from training indices only, fits calibration on training distances
and scores only, and gives each subject one out-of-fold prediction. Do not call
the predictions untouched: fold 1 informed the reference threshold, and the
method was developed on KIMORE.

**Why MAE and Spearman?**

MAE is directly interpretable as typical TS-point error. Spearman checks score
ordering without assuming a perfect linear relationship. RMSE emphasizes large
mistakes, and Pearson describes linear association.

**Why compare with constants?**

An error value alone does not show whether the movement signal contributes
anything. Training median is the optimal constant target for MAE; training
mean is the optimal constant target for squared error/RMSE.

**What does the bootstrap interval mean?**

Subjects are sampled with replacement, keeping the 76 saved actual/predicted
pairs fixed. Repeating this 5,000 times gives a conditional fixed-prediction
interval. It does not rerun folds, reference selection, calibration or feature
selection, so it understates the full development and model-selection
uncertainty.

**Does the 99% threshold solve shoulder-tracking quality?**

No. It only filters reference candidates, and only 8 of 76 recordings meet it.
All frames from non-reference samples still enter DTW, including inferred
shoulder coordinates. The threshold was also chosen after inspecting fold 1,
so it is a documented development choice rather than independently validated.

**Why did fold 4 fail?**

`E_ID7`, the reference used in four folds, was held out. The next eligible
reference was `NE_ID13`, and training distance barely related to score: the
calibration slope was -0.09 instead of about -0.8 to -0.9. This exposes
single-template sensitivity.

**Why is the cohort-aware baseline important?**

KIMORE combines groups with different score distributions. A global constant
ignores information already present in the cohort label. Training-fold cohort
constants obtain MAE 5.85 and RMSE 7.53, both better than DTW. This comparator
was added post hoc, so it is diagnostic rather than preregistered, but it makes
the weakness of the original comparison impossible to ignore.

**Does the method work?**

The software runs reproducibly, prevents subject overlap and keeps fold-specific
fitting on training rows. The predictive method does not work well: it is worse
than a cohort-aware constant, unstable across folds, excessively warped and
poorly calibrated for low Parkinson scores. Do not collapse those two meanings
of "works."

**What would you try next?**

Possible hypotheses are several training templates, a training medoid, or a
small set of additional interpretable posture/control features. They must be
chosen with nested training validation or tested on new external data, not
tuned against these same outer predictions.

### Claims to avoid

- Do not call this external validation or a clinical model.
- Do not call the entire development evaluation untouched or simply
  “leakage-safe.” State the narrower verified properties.
- Do not say DTW definitely beats the baseline; the improvement intervals
  include zero.
- Do not use only the global constants while hiding the stronger post-hoc
  cohort constants.
- Do not say DTW is useless; this experiment tests one feature and one template.
- Do not call 0.30 a strong correlation.
- Do not hide fold 4 or the low-score overprediction pattern.
- Do not say preprocessing improves the shoulder-yaw model; the feature is
  translation- and scale-invariant.
- Do not say bootstrap intervals include split, refitting or selection
  uncertainty.
- Do not claim GroupKFold changes leakage for the current one-sample-per-subject
  Es3 subset; explain why the explicit rule is still correct engineering.

Finish Day 2 by explaining one good alignment, one poor alignment, the
actual-versus-predicted plot, fold 4 and the constant comparison without reading
from the notes.
