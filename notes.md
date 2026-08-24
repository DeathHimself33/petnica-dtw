# Petnica project notes

## 20 August 2026

Today I started properly working with the KIMORE dataset instead of thinking about DTW or the final model straight away.

First I generated the full KIMORE file structure so we could see how the dataset is organised. There are 78 subjects split into expert controls, non-expert controls and several patient groups. Each subject can have five exercises (`Es1`–`Es5`), along with skeleton recordings and clinical scores.

I ran `kimore_dataset_audit.py` on the full dataset. It created a manifest with one row for each subject/exercise combination, checked the files and connected the recordings with their clinical scores. The first result said 354 out of 390 samples were OK, but this turned out to be too strict because it sometimes marked an exercise as problematic due to a missing score for a different exercise. It also treated missing orientation data as fatal even though we currently only need joint positions, so I should not delete the rows it marked as problematic.

Based on the usable samples and score distributions, I chose **Exercise 3 (trunk rotation)** as the first exercise. It has 76 usable samples, a clinical Total Score range of 10–50 and data from all groups. Two samples are excluded for now: `E_ID17_Es3` has no clinical TS score, while `S_ID5_Es3` has no joint-position recording.

I then ran `kimore_dataset.py` for Es3. It selected the expected 76 samples and successfully loaded `B_ID1_Es3` with the shape `(1031, 25, 3)`, meaning 1031 frames, 25 joints and XYZ coordinates for each joint. The original CSV also contains a tracking-state value for every joint. In this example about 80.7% of the joint observations were fully tracked and 19.3% were inferred, with none completely untracked.

I plotted a SpineShoulder signal from the sample. The graph showed a neutral section followed by about five repeated movements, and there did not seem to be any major drift or obvious missing tracking.

The loader is now working and I moved `kimore_dataset.py` into `src/`. I understand what the output represents, but I do not yet understand the code well enough to explain exactly how it selects samples and reshapes a CSV row into 25 joints. Before moving on to preprocessing, I need to go through the loader line by line and write down how each part works.

### Next step

Tomorrow I should first understand the manifest and loader properly. After that I can start preprocessing by centring the skeleton around `SpineBase` and normalising for body size without removing the trunk rotation that the exercise is supposed to measure.

##21 August 2026

Today I went through both kimore_dataset_audit.py and kimore_dataset.py in detail instead of immediately adding more code. I now understand the main path from the original KIMORE folders to the NumPy arrays that will eventually be used for DTW.

The audit recursively finds subject folders using regular expressions, reads the clinical workbook, checks the position, orientation and timestamp files for each exercise, and writes one manifest row per subject/exercise. I also clarified what the clinical scores mean. PO is the Primary Outcome score for achieving the main goal of the exercise, CF covers posture and control factors, and TS is the total of the two. For this project TS is currently the value that will be predicted.

I also understand why the audit status and loader usability are separate. The audit marks a row as problem if it finds any issue at all, but the position loader only rejects problems that matter to the current approach. It needs a TS score, one valid JointPosition file, a non-empty recording, 100 values per frame and consistent row widths. Missing orientation or timestamp data does not automatically make the position recording unusable.

The loader reads every 100-value frame and reshapes the data from (frames, 100) to (frames, 25, 4). The four values are X, Y, Z and tracking state for each joint. It then separates this into positions shaped (frames, 25, 3) and tracking states shaped (frames, 25).

After understanding the existing code, I started preprocessing. Each frame is centred by subtracting its SpineBase position from every joint. This removes the subject's position relative to the camera without rotating the skeleton, which is important because Exercise 3 is trunk rotation.

Before choosing a body-size scale, I compared torso length (SpineBase to SpineShoulder) with shoulder width (ShoulderLeft to ShoulderRight) over all 76 usable Es3 recordings. Torso length was clearly more reliable: its median relative MAD was 3.16% compared with 6.46% for shoulder width, both torso joints were usable in 100% of frames, and the torso was more stable in 70 of the 76 recordings. Because of this I used the median torso length as the scale rather than assuming it was the best option without testing it.

The preprocessing now centres the positions and divides every coordinate in the recording by one torso-length scale. It keeps the original array unchanged and preserves the tracking states. I first checked B_ID1_Es3, where the shape stayed (1031, 25, 3), SpineBase became exactly zero and the median normalised torso length became 1. I then ran the same validation across all 76 usable recordings and got zero preprocessing failures. The two expected exclusions are still E_ID17_Es3 because of the missing TS score and S_ID5_Es3 because of the missing JointPosition file.

###Next step

Tomorrow I want to go through the preprocessing again from the beginning before adding anything new. I mostly understand it, but I want to be able to explain the centring, broadcasting, scale diagnostic, MAD and normalisation properly rather than only knowing that the checks pass.

## 22 August 2026

### Why I chose Exercise 3

I chose **Exercise 3 (trunk rotation)** for the first DTW baseline. This is a practical starting choice, not a claim that Es3 is universally the best KIMORE exercise.

Es3 has 76 usable recordings with clinical Total Scores ranging from 10 to 50. The usable set includes subjects from all five cohorts: expert controls, non-expert controls, people with back pain, people with Parkinson's disease and people recovering from stroke. Only two Es3 samples are currently excluded: `E_ID17_Es3` has no clinical TS target, and `S_ID5_Es3` has no JointPosition recording.

The movement is also suitable for an interpretable DTW baseline. A recording contains repeated trunk rotations, so DTW can align similar movements that were performed at different speeds. The relationship between the signal and the exercise is understandable: the model should compare how the trunk-rotation pattern develops over time, rather than relying on an opaque feature.

The preprocessing choices are designed to preserve that movement. Centring on `SpineBase` removes camera-relative translation, and normalization by one median torso length reduces body-size differences. I will not rotate each frame into a common body orientation, because doing that could remove the trunk rotation that Es3 is intended to measure.

### What I understand about the scale diagnostic

For each candidate bone, the diagnostic first keeps only frames in which both endpoint joints are fully tracked. It subtracts the first joint position from the second and uses the Euclidean norm of that XYZ vector as the bone length in each usable frame.

The median of those lengths estimates the typical bone length while resisting occasional Kinect outliers. The median absolute deviation (MAD) measures the typical absolute difference from that median. Dividing MAD by the median length produces relative MAD, which lets me compare the stability of measurements with different sizes. The usable fraction separately records how often both joints were fully tracked.

The torso was more suitable than shoulder width because it had lower relative variation and much better tracking availability across Es3. `preprocess_sequence()` therefore uses the median `SpineBase`-to-`SpineShoulder` length as one scale value for the entire recording. It does not choose a different scale candidate for every sample or resize every frame independently.

### What I understand about preprocessing

The input position array has shape `(frames, 25, 3)`. For every frame, centring subtracts that frame's `SpineBase` XYZ position from all 25 joints. NumPy broadcasts the origin from shape `(frames, 1, 3)` across the joint dimension. This makes `SpineBase` equal to zero and removes camera-relative translation, while distances and directions between joints remain unchanged.

Body-size normalization then divides every centred coordinate in the recording by the recording's single median torso length. The resulting coordinates are expressed in torso-length units, which makes subjects of different sizes more comparable. Using one scale for the entire recording avoids stretching and shrinking the skeleton from frame to frame. Tracking states are copied unchanged, and the original input array is not modified.

The preprocessing intentionally does not rotate each frame, force every recording to the same length, smooth signals automatically or discard inferred coordinates. These decisions preserve the Es3 trunk rotation and avoid adding transformations before the data show that they are necessary.

### Subject-wise evaluation groups

I created subject groups from `subject_id` and used five-fold `GroupKFold` splitting. Samples from the same subject are treated as one indivisible group, so that subject cannot appear in both training and testing in a fold. A separate assertion calculates the intersection of the training and testing subject sets and raises an error if it is not empty.

The grouping checks passed on the 76 usable Es3 recordings. Every fold had zero overlapping subjects: the first fold contained 60 training and 16 testing subjects, while the other four contained 61 training and 15 testing subjects. Synthetic tests also confirmed that repeated recordings from one subject stay together and that a deliberately leaking split is rejected.

### End-of-day checks

I reran the preprocessing check on `B_ID1_Es3`. It preserved the `(1031, 25, 3)` shape and tracking states, did not modify the input, produced finite coordinates, made every `SpineBase` coordinate zero and made the median normalized torso length exactly 1.0.

I also reran the scale diagnostic across all 76 usable Es3 recordings. There were zero preprocessing failures. The torso was more stable than shoulder width in 70 recordings, while shoulder width was more stable in six. Torso relative MAD had a dataset median of 3.16% and a worst value of 8.14%; it was usable in 100% of frames in every recording. Shoulder relative MAD had a median of 6.46%, and its usable fraction ranged from 29.24% to a median of 69.65%.

All four subject-grouping tests passed, and the real five-fold split had zero overlapping subjects in every fold. This completes the planned August 22 work: I can explain the core preprocessing, the choice of Es3, the body-scale diagnostic and the reason for subject-wise grouping.

## 23 August 2026

### Tracking quality and missing-data decision

I added a dataset-wide tracking-quality diagnostic and ran it on all 76 usable Es3 recordings. It inspected 61,567 frames and 1,539,175 joint observations with zero diagnostic failures. Overall, 92.43% of joint observations were fully tracked and 7.57% were inferred. There were no untracked states, non-finite coordinates or all-zero XYZ joint positions.

The tracking quality is not uniform across joints. `SpineBase`, `SpineMid`, `SpineShoulder` and `Neck` were fully tracked in every frame, which supports the current centring and torso-scaling operations. The least reliable joint was `ElbowRight`, with 77.19% fully tracked observations. `ShoulderLeft` was fully tracked in 85.73% of frames and `ShoulderRight` in 84.54%. Their longest continuous inferred runs were 141 and 154 frames respectively.

Because there are no missing or untracked coordinates, I will not delete frames or recordings and I will not add a missing-value imputation step. I will also not interpolate inferred shoulder or limb coordinates: some inferred runs are long enough that interpolation would invent a substantial part of the movement. The loader now rejects non-finite coordinates and tracking-state values other than 0, 1 or 2 instead of silently accepting malformed data. Tracking states remain available so a later feature or DTW distance can use reliability information if the baseline shows that it is needed.

### Noise and smoothing decision

I measured short-lived noise in a shoulder-axis yaw signal by comparing it with a five-frame moving median. The moving median was used only as a diagnostic, not as preprocessing. Across recordings, the median per-sample 95th-percentile deviation was 0.291 degrees. The worst per-sample 95th percentile was 4.072 degrees, and the largest isolated deviation was 53.555 degrees in `P_ID9_Es3`.

This means a few localized tracking jumps are real, so it would be wrong to claim that the recordings are perfectly clean. It does not justify applying one blanket smoother to all coordinates: most values are clean, different joints have different tracking quality, and long inferred runs cannot be repaired by light smoothing. The baseline will therefore use no automatic global smoothing. If a chosen DTW signal is visibly damaged by these jumps, a fixed feature-level filter can be compared as an explicit model choice using training subjects only.

### Rotation, duration and visual verification

I kept the decision not to rotate each frame, because that could remove the Es3 trunk rotation. As a rough camera-alignment check, the per-recording median pelvis-axis yaw ranged from -5.42 to 9.87 degrees between the dataset's 5th and 95th percentiles. This does not show a clear camera-orientation problem, and pelvis direction also changes with the subject's movement, so I will not introduce a fixed camera correction from this evidence.

Sequence lengths range from 364 to 1,517 frames. I will not resample them to one length because duration variation is the reason for using DTW. Any learned dataset-level feature scaling or parameter selection must be fitted using training subjects only; the per-recording torso scale is computed from that recording itself and does not use other subjects or clinical scores.

I generated two before/after figures for `B_ID1_Es3`. The first compares the camera-relative `SpineShoulder` X coordinate with the SpineBase-relative, torso-normalized signal. The second compares shoulder-axis yaw before and after preprocessing. The yaw curves have a maximum absolute difference of only `1.42e-14` degrees, confirming numerically and visually that translation and uniform body-size normalization preserve this trunk-rotation measurement.

### End-of-day validation

The final scale and preprocessing diagnostic again completed all 76 usable recordings with zero failures. It checked that preprocessing preserves shape and tracking states, does not modify the input arrays, produces finite coordinates, centres `SpineBase` at zero and makes median torso length equal to one. All 11 automated tests passed, including new tests for loader validation, centring, scaling, input preservation, tracking-state counts, moving-median behaviour, run-length measurement and yaw invariance to translation and scale.

## 24 August 2026

### Plain DTW baseline

I implemented the first end-to-end plain-DTW prediction pipeline on one
subject-wise fold. It loads and preprocesses all 76 usable Es3 recordings,
extracts shoulder-axis yaw as a one-dimensional and interpretable trunk-
rotation signal, selects one reference using training subjects only, aligns
every signal with exact unconstrained DTW and reports the path-normalized
aligned RMSE in degrees.

The clinical calibration is a simple least-squares line from DTW distance to
Total Score. It is fitted on the 60 training subjects only and then applied to
the 16 held-out subjects. The saved prediction table includes the fold and
train/test role, sample and subject IDs, cohort, actual and predicted TS, DTW
distance, frame count, alignment-path length, feature, reference metadata and
calibration coefficients. The fold has zero overlapping subjects.

I initially selected the highest-score training execution and used shoulder
tracking fraction only to break score ties. The resulting reference,
`E_ID12_Es3`, contained obvious abrupt jumps in its yaw trace even though its
aggregate shoulder tracking fraction was high. The alignment figures caught
this problem. I changed and froze the reference rule to first require at least
99% fully tracked shoulder frames when that is possible, and then choose the
highest clinical score from that reliable training pool. If no training sample
meets the threshold, the rule falls back to the best-tracked training sample or
samples. This selected `E_ID7_Es3`, with TS 47.33 and 99.89% fully tracked
shoulder frames. Its reference trace is visibly much cleaner.

The final fold-1 preview produced a test MAE of 5.907 TS points, RMSE of 7.293,
Spearman correlation of 0.308 and Pearson correlation of 0.378 on 16 test
subjects. These values are not the final model result. The fold is small, the
correlations are weak, and this preliminary split was used while checking and
freezing the pipeline. A slightly earlier, less defensible reference happened
to give a lower MAE, which is a useful reminder not to choose methodology just
because it improves one test number.

### How result evaluation will work

The final internal evaluation will use all five outer `GroupKFold` splits. A
new reference and calibration will be fitted independently inside each
training fold, every subject will receive exactly one out-of-fold prediction,
and the pooled predictions will be evaluated with MAE and Spearman as primary
metrics plus RMSE and Pearson as secondary metrics. Fold-wise metrics and an
actual-versus-predicted plot will also be saved.

I added an evaluation plan that makes two important points more explicit than
the original schedule. First, the DTW errors must be compared with constant
training-fold baselines: the training median for MAE and the training mean for
RMSE. Second, the pooled metrics should receive subject-level bootstrap 95%
confidence intervals because 76 subjects is too small for point estimates to
look more certain than they are.

This is internal KIMORE cross-validation, not external validation. The current
shoulder-yaw feature captures rotation timing and amplitude but not every
posture and control factor in the clinical score. A weak result would show the
limits of this thin baseline rather than prove that all DTW-based approaches
are unsuitable.

### End-of-day validation

All 17 automated tests pass. The new tests cover an identical zero-cost
alignment, repetition handling, rejection of non-finite features,
training-only reference selection, the reference-quality fallback and recovery
of a known linear calibration. The real run saved 76 inspectable prediction
rows, fold/reference metadata, and closest and most-distant held-out alignment
figures.

## 25 August 2026 (completed early on 24 August)

### Five-fold subject-wise evaluation

I extended the frozen plain-DTW method from one preview split to all five outer
subject-wise folds. Every fold programmatically confirmed zero overlap between
training and test subjects. Each fold selected its reference and fitted its
linear calibration using training subjects only. Every one of the 76 usable
Es3 subjects now has exactly one out-of-fold prediction.

The reproducible runner saves the 76 prediction rows, fold-wise and overall
metrics, per-fold references and calibration coefficients, constant-baseline
predictions, the manifest SHA-256 hash, package versions, bootstrap settings
and an actual-versus-predicted diagnostic plot. Exact DTW alignments are cached
when multiple folds select the same training reference. One project-root
command recreates the experiment:

`python run_experiment.py --exercise Es3 --method plain_dtw`

### Final internal baseline result

Across all 76 out-of-fold predictions, plain DTW obtained MAE 6.981 TS points,
RMSE 8.932, Spearman 0.301 and Pearson 0.345. The subject-bootstrap 95%
intervals were 5.825-8.283 for MAE, 7.157-10.804 for RMSE, 0.090-0.492 for
Spearman and 0.129-0.553 for Pearson.

The comparison with constant predictions changes how these values should be
interpreted. The training-median baseline had MAE 7.342, only 0.360 points
worse than DTW. Its paired bootstrap improvement interval was -0.547 to 1.334.
The training-mean baseline had RMSE 9.521, only 0.589 points worse, with an
improvement interval of -0.192 to 1.501. Because both intervals include zero,
I cannot claim that this DTW baseline reliably beats constant prediction.

### What the fold differences show

The result was unstable across folds. Fold 5 had Spearman 0.686 and MAE 5.882,
while fold 4 had Spearman -0.018 and MAE 10.070. Four folds used `E_ID7_Es3`
as their training-only reference and learned calibration slopes between about
-0.79 and -0.90. Fold 4 held `E_ID7` out, selected the next eligible reference,
`NE_ID13_Es3`, and learned a slope of only -0.090. This shows that the
single-template method depends heavily on which suitable reference is
available in training.

The prediction plot shows strong regression toward the middle of the score
range. Several low-scoring Parkinson's recordings were predicted near 40, and
many high-scoring expert/non-expert recordings were underpredicted. Shoulder
yaw contains some relationship with score, but it does not capture all the
posture and control information included in clinical TS.

I will not change the template rule or feature after seeing these outer-fold
results. Possible later work includes multiple templates or additional
interpretable features, but those ideas require a new nested or external
evaluation rather than tuning against these same test predictions.

### Packaging and checks

The project now has a root experiment runner, locked direct dependencies, a
README with setup and outputs, a detailed evaluation rationale and backed-up
compact final results/figure. All 24 automated tests pass. New checks cover
metric calculations, undefined constant correlations, training-only constant
baselines, exact once-only out-of-fold coverage, deterministic subject-level
bootstrap intervals and paired baseline improvements.
