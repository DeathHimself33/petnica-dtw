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

End Day 1 by running all 24 tests and explaining what failure each test would
catch. Do not just state that they pass.

## Day 2: explain and defend the result

### A 90-second project explanation

> I built an interpretable baseline for KIMORE Exercise 3 using 76 usable
> subjects. Each skeleton recording is centred on SpineBase and normalized by
> median torso length. I extract shoulder-axis yaw because Es3 measures trunk
> rotation, then use exact DTW to align recordings performed at different
> speeds. Distance is path-normalized aligned RMSE rather than raw accumulated
> cost.
>
> Evaluation uses five subject-wise folds. In every fold, both the reference
> execution and the linear mapping from distance to clinical Total Score use
> training subjects only. Every subject receives one held-out prediction.
>
> The baseline achieved MAE 6.98 TS points and Spearman 0.30. It was only 0.36
> MAE points better than a training-median constant, and the paired bootstrap
> interval for that improvement crossed zero. Results also varied strongly by
> fold because the method depends on one reference. Therefore the pipeline is
> reproducible and leakage-safe, and shoulder yaw contains some score-related
> information, but this version does not reliably beat constant prediction.

### Likely mentor questions

**Why Exercise 3?**

It has 76 usable subjects across all cohorts, a useful TS range, few exclusions
and an interpretable repeated trunk-rotation movement that DTW can align.

**Why centre on SpineBase?**

It removes camera-relative translation. Subtracting the same origin from every
joint preserves distances and directions between joints.

**Why median torso length?**

The diagnostic showed lower relative MAD and better tracking availability than
shoulder width. One median scale per recording reduces body-size differences
without making the skeleton expand and contract frame by frame.

**Why no frame-wise rotation?**

Es3 is trunk rotation. Rotating every frame into a common body orientation
could remove the movement being measured.

**Why no resampling to one length?**

Different duration is the main reason to use DTW. Resampling would add another
transformation before an observed need justified it.

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
and scores only, and evaluates untouched fold predictions. Each subject appears
in exactly one outer test fold.

**Why MAE and Spearman?**

MAE is directly interpretable as typical TS-point error. Spearman checks score
ordering without assuming a perfect linear relationship. RMSE emphasizes large
mistakes, and Pearson describes linear association.

**Why compare with constants?**

An error value alone does not show whether the movement signal contributes
anything. Training median is the optimal constant target for MAE; training
mean is the optimal constant target for squared error/RMSE.

**What does the bootstrap interval mean?**

Subjects are sampled with replacement, keeping actual and predicted values
paired. Repeating this 5,000 times shows how much the metric could vary from
the particular 76-subject sample. The error-improvement intervals cross zero,
so a reliable advantage over constants is not established.

**Why did fold 4 fail?**

`E_ID7`, the reference used in four folds, was held out. The next eligible
reference was `NE_ID13`, and training distance barely related to score: the
calibration slope was -0.09 instead of about -0.8 to -0.9. This exposes
single-template sensitivity.

**Does the method work?**

The software and leakage-safe evaluation work. The predictive method shows a
modest positive association, but it does not reliably beat constant prediction.
Do not collapse those two different meanings of "works."

**What would you try next?**

Possible hypotheses are several training templates, a training medoid, or a
small set of additional interpretable posture/control features. They must be
chosen with nested training validation or tested on new external data, not
tuned against these same outer predictions.

### Claims to avoid

- Do not call this external validation or a clinical model.
- Do not say DTW definitely beats the baseline; the improvement intervals
  include zero.
- Do not say DTW is useless; this experiment tests one feature and one template.
- Do not call 0.30 a strong correlation.
- Do not hide fold 4 or the low-score overprediction pattern.
- Do not claim GroupKFold changes leakage for the current one-sample-per-subject
  Es3 subset; explain why the explicit rule is still correct engineering.

Finish Day 2 by explaining one good alignment, one poor alignment, the
actual-versus-predicted plot, fold 4 and the constant comparison without reading
from the notes.
