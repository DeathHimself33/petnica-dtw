# Es2 / Es4 source-data audit

## Findings and decision

The audit found source identity inconsistencies and some retained skeletal
discontinuities worth addressing. It did not find a numerical score-column
mapping error, duplicated position recordings among the uniquely selected manifest paths, or a general case for trimming
quiet recording boundaries. Feature coverage is also a separate concern:
the supplied Es4 reference scripts measure pelvis translation, which the
current nine-vector representation is mathematically unable to preserve.

The subsequent whole-project inventory also inspected multiple-recording folders
that this focused audit could not load through the manifest. It found identical
Es4 position-file pairs for E_ID14 and E_ID15 (plus two Es5 pairs). See
`PROJECT_AUDIT_REPORT.md`; the earlier duplicate check did not cover those folders.

No raw file, clinical label, manifest, QC rule, model or prediction was changed.
The existing results remain a frozen experiment with the additional provenance
limitations documented below. This audit is exploratory and does not establish
that cleaning or adding features will improve predictive performance.

## Scope and reproduction

Run `python audit_es2_es4_data.py`; dependencies are numpy, pandas, openpyxl
and Pillow. Generated evidence is under `results/es2_es4_audit/` (Git-ignored).
The script verifies prediction targets against the manifest, independently
reads source score columns and internal subject IDs, and reads raw coordinates,
orientations and timestamps. It uses the existing QC implementation to separate
raw-coordinate flags from changes remaining in repaired features.

There are 156 manifest rows (78 per exercise). Coordinates could be loaded for
74 Es2 and 72 Es4 rows; some have missing labels or fail QC. Error comparisons
use exactly the frozen 70 Es2 and 66 Es4 out-of-fold recordings. The workbook
check covers all 398 clinical workbook copies below the 78 subject folders,
because each exercise folder may contain the same subject-level score table.

Numerical/source checks passed for the available scores, folder identities,
manifest frame counts, and all 136 evaluated prediction targets. These checks
establish pipeline consistency, not that every original clinical label is valid.

## 1. Two newly identified internal-ID inconsistencies

The original loader checks workbook filenames and agreement of score copies,
but does not validate the `Subject ID` cell inside the workbook.

| Source | Observation | Interpretation/action |
|---|---|---|
| E_ID16 / Es2 / ClinicalAssessment_E_ID16.xlsx | Internal ID is E_ID17 | Four other E_ID16 copies have the correct ID and identical scores. A source metadata typo is plausible; use identity-consistent copies in a future validated loader and retain a provenance warning. |
| NE_ID2 / all five copies | Internal ID is E_ID1 | All five correctly named copies share the same scores, but those scores differ from actual E_ID1's scores. Identity remains unresolved; do not substitute E_ID1's scores or silently rewrite the ID. |
| E_ID3 / Es3–Es5 | Wrongly named E_ID1 workbooks | Already handled by the existing loader, which ignores these copies and uses correctly named consistent copies. |

NE_ID2 is present in both audited evaluations: Es2 ML absolute error is 7.48,
Es4 is 2.07. E_ID16 is present in Es2 (error 2.02); its Es4 recording is not in
the retained evaluation. These are not the largest errors, but target provenance
matters regardless of predictive error. Resolving NE_ID2 requires independent
source confirmation; the local copies alone cannot settle it. No messages were
sent to dataset authors or reviewers.

All 156 manifest rows agree numerically with the applicable source TS/PO/CF
values or their missingness. No conflicting score copies were found among
correctly named copies. There were no byte-identical position files across
the loaded audit recordings, and no identical adjacent coordinate frames.
This does not rule out approximate duplicates or duplication elsewhere.

## 2. Timestamp gaps exist, but do not explain the general error pattern

Position and timestamp counts match, and timestamps increase strictly, in all
loaded Es2/Es4 recordings. In the evaluated population:

| Check | Es2 (70) | Es4 (66) |
|---|---:|---:|
| Recordings with a step >2 times their median positive timestamp step | 49 | 39 |
| Recordings with a step >5 times their median step | 11 | 6 |
| Median largest step / normal step | 2.03 | 2.03 |

Gaps are usually sparse: the median fraction of >2x steps is about 0.11–0.12%.
However, S_ID2_Es2 has a maximum gap of 270.68 median steps, S_ID3_Es2 44.42,
P_ID4_Es2 19.14 and E_ID7_Es4 13.12. These should be inspected before treating
frame index as uniformly spaced physical time. No timestamp unit is assumed.

Largest-gap ratio has Spearman correlation -0.177 with Es2 ML absolute error
and -0.100 with Es4 ML error. There is no simple larger-gap/larger-error pattern.
The current ML sampler uses frame progress, not timestamps. A future time-aware
sampler should not interpolate long acquisition gaps as if movement were known.

## 3. QC misses some discontinuities; many raw flags are not final-feature defects

For this diagnostic, a large raw jump means that at least one major joint,
fully tracked at both endpoints, moves >0.25 median torso lengths between
adjacent root-centered frames. This is an inspection threshold, not a new QC
rule or a clinical error label. Removing root translation can itself expose
root-tracking problems. Rapid valid movement can also exceed this threshold.

We distinguish a raw jump at retained frames from a discontinuity still present
after interpolation. A normal timestamp step is positive and <=1.5 times the
recording's median step.

| Candidate transition count | Es2 | Es4 |
|---|---:|---:|
| Raw jumps at QC-retained endpoints | 901 | 266 |
| Above, also at normal timestamp spacing | 894 | 264 |
| Above, with a repaired-feature angular step >45 degrees | 130 | 76 |

These are transition counts, not independent events or affected patients.
Large transitions at the same recording may be related.

In Es2, normal-timestep raw-jump fraction correlates with ML absolute error
at Spearman 0.394, versus 0.010 for DTW. Es4 associations are 0.086 for ML
and -0.064 for DTW. This makes local tracking diagnostics worth following up
for Es2, without proving causality; associations are unadjusted, exploratory,
and potentially confounded by cohort, movement amplitude and score.

Four sequences around the largest normal-timestep jumps were visually inspected:

- NE_ID21_Es2, frames 8–9: major abrupt configuration change, no interpolation;
  repaired feature step reaches 113.87 degrees.
- NE_ID9_Es2, frames 3–4: abrupt configuration change, no interpolation;
  repaired feature step reaches 101.08 degrees.
- NE_ID17_Es2, frames 383–384: severe raw-coordinate change, but interpolation
  reduces the largest repaired-feature step to 31.72 degrees.
- NE_ID6_Es2, frames 70–71: raw limb jump; interpolation reduces the largest
  repaired-feature step to 12.79 degrees.

The last two demonstrate why raw jumps alone must not be counted as unhandled
feature defects. Evidence and frame indices are in `jump_examples.png`,
`visual_jump_selection.csv` and `retained_jump_candidates.csv`.

## 4. No evidence for blanket trimming of quiet starts/ends

Motion is measured as the 75th percentile of fully tracked major-joint
root-centered displacement in torso lengths per frame, smoothed with a 15-frame
rolling median. At least six tracked joint pairs and eight finite window values
are required. Missing tracking is not classified as inactivity.

At a threshold of 0.002 torso lengths/frame, terminal low-motion runs occupy
>10% of just one Es2 and zero Es4 evaluated recordings. At 0.001 the counts
are zero/zero; at 0.005 they are four/seven. Results depend on the arbitrary
diagnostic threshold, so there is no validated segmentation rule here.
Slow pathological movement, pauses within an exercise and moving preparation
cannot be separated by this simple motion measure. No frames were trimmed.

Matched technical contact sheets compare the two largest ML errors per exercise
with lower-error same-cohort examples selected by nearest clinical score:
NE_ID5 vs NE_ID12 and P_ID6 vs P_ID9 for Es2; P_ID9 vs P_ID13 and P_ID11 vs
P_ID8 for Es4. The first pair has an 8.33-point target difference, so matching
is imperfect. Some Es4 controls still have 5–6 point errors.
The sheets show movement across the recording and tracking artifacts in both
error groups, without a common long inactive boundary explaining the worst cases.
Seven frontal raw skeleton snapshots are not an RGB/video or clinical review.

## 5. Important revision to the representation hypothesis

The dataset's supplied MATLAB files under
`data/raw/KIMORE/GPP/BackPain/B_ID1/Es2/Script/feat_extract_Ex2.m` use left/right
shoulder-to-hip inclination as primary-outcome signals. The corresponding
`Es4/Script/feat_extract_Ex4.m` uses centered SpineBase X/Z trajectories.
These local scripts are evidence of intended exercise-specific signals, not
proof of the clinical scoring process or an exact replicated algorithm.

The current nine vectors discard global translation, including time-varying
translation. An actual sequence translated laterally with a sinusoidal
0.2-coordinate-unit offset produces identical vectors to numerical precision
(max difference 5.55e-16). Thus explicit pelvis displacement is a separate
candidate missing signal for Es4. The previous Es4 report's body-up-only
recommendation was too narrow: trunk orientation is directly relevant to the
supplied Es2 signals, while pelvis trajectory deserves its own Es4 ablation.

Adding pelvis movement should remove initial camera position and normalize body
scale while retaining displacement over time. Camera-coordinate axes and root
tracking quality require care. Representation invariance is proven; predictive
benefit of either extra signal has not been tested.

## Next work justified by this audit

1. Add internal workbook identity validation to the main loader with explicit
   provenance handling; resolve or explicitly quarantine NE_ID2 before a new
   confirmatory dataset release. Do not invent replacement labels.
2. Review flagged uncorrected transitions and major time gaps; design a separate,
   auditable QC variant if warranted. Preserve the frozen raw/QC comparison.
3. Run controlled feature ablations: torso orientation and pelvis displacement
   separately, then combined if justified. Keep current results as the baseline.
4. Do not apply broad denoising, score-driven exclusion or blanket start/end cuts.
   Any improvement selected after this OOF inspection is exploratory unless
   assessed with nested or newly held-out evaluation.
