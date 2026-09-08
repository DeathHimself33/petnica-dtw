# Full project audit — work log

User request: comprehensive, unhurried audit of the dataset, algorithms, code,
evaluation, expert-review workflow, tests and reporting. Find and reproduce
issues first so they can be resolved systematically later.

## Scope and rules

- Preserve source data, existing predictions/checkpoints and production behavior.
- New audit scripts, synthetic fixtures and reports may be created.
- Separate confirmed defects, methodological limitations and open hypotheses.
- Record file/line evidence, reproducer, impact, severity and proposed resolution.
- Cover every maintained source/test module and the generated artifacts they claim
  to validate; avoid claiming that a test suite proves absence of defects.
- No deployment, publishing, messaging, data relabeling or model retraining.

## Starting state

- HEAD: 4050628 (expert review application).
- Previous-turn audit files are present and uncommitted; preserve them.
- Existing Python 3.13 environment is functional outside the sandbox.
  Earlier `Access is denied` was an execution restriction, not a broken venv.
- Runtime: Python 3.13.14, torch 2.7.1+cu128, numpy 2.5.2, matplotlib 3.11.1.

## Coverage checklist

- [x] Baseline test suite and existing environment (89 passing tests).
- [x] Dataset inventory, all source workbooks, internal IDs, duplicate sources,
      excluded/multiple recordings, timestamps, frame schemas and provenance.
- [x] Skeleton loading, preprocessing, tracking, scaling and frame QC.
- [x] Exact DTW implementation, reference selection and calibration. Full original
      paper equation verification remains open because full text was unavailable.
- [x] Subject folds, all-exercise consistency, OOF metrics and uncertainty.
- [x] Interpretable components, localization, RGB alignment code and pilot labels.
      Frame-by-frame human RGB review and clinical validation remain open.
- [x] ML tensor export, masks, model, optimization, resume and checkpoints.
- [x] Multi-seed aggregation, comparator populations and bootstrap statistics.
- [x] Expert-review identity, blinding, persistence, adjudication and exports.
- [x] CLI behavior, output collisions, error handling and documentation claims.
- [x] Existing test coverage reviewed; independent audit reproducers added.
- [x] Consolidated findings and prioritized repair plan with explicit unknowns.

## Activity

Audit initiated. Inventory contains approximately 12k source lines plus tests,
raw data and generated artifacts. Existing focused Es2/Es4 reports are leads,
not substitutes for checking the complete project.

## Completed evidence

- Rebuilt the 390-row manifest; exact CSV-equivalent match.
- Inspected all 402 JointPosition files including multiple-recording folders and
  two out-of-scope Es6 recordings; four identical pairs and four malformed files.
- Rebuilt all 12 ML arrays exactly; verified 25 training-only normalizers and
  subject separation; repeated all 25 checkpoint inferences on the original GPU
  with zero difference from saved predictions.
- Verified 13 DTW result sets against raw source hashes and independently
  recomputed metrics; verified ML/DTW common population and ensemble arithmetic.
- Reproduced unsafe resume, six-fold exporter/trainer mismatch, cross-reviewer
  adjudication reuse, mutable primary labels after sealing, identity reentry,
  ungradable-final rejection, unstable label joins and Jinja item-count rendering.
- Verified missing uncertain cases in default expert queue and visually inspected
  the cropped lower-limb skeleton fallback.
- Independently read both anomalous Es5 target workbooks across all five copies;
  preserved original targets and recorded PO+CF discrepancies as source questions.
- Consolidated results in PROJECT_AUDIT_REPORT.md, including severity, scope,
  repair sequence, evidence paths and limits. No production fixes or retraining.

## Reproduction environment notes

The project venv has the production dependencies but not pandas. The independent
artifact arithmetic checker and prior focused data audit used bundled Python
with pandas. No packages were installed. Original GPU inference is reproducible;
CPU inference differs slightly and is not used to label the artifacts corrupt.

Remaining external/experimental work is enumerated in the report: clinical and
source-identity confirmation, full-video synchronization review, full-paper
equation verification, new ablations and clean-machine reproduction. These are
not represented as completed by the code audit.
