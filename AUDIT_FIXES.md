# Audit remediation — 2026-09-06

The confirmed software failures A01–A11 have been addressed. Existing raw data,
clinical targets, trained models, saved predictions, preliminary labels, and
the historical expert database were preserved. This does not close the audit's
clinical, source-identity, synchronization, or experimental research questions.

## Changes and compatibility

| Finding | Resolution |
|---|---|
| A01 | Training writes an immutable run manifest containing configuration, dataset hash, exact sample/fold population, code hashes and runtime. Resume rejects mismatches and unverified legacy outputs. Fold artifacts are staged together and atomically published, with checkpoint/history/prediction/metric hashes. Aggregation validates provenance, exact OOF coverage and fold contents, and permits only the seed to differ between runs. |
| A02 | Export and training require five folds. Every expected sample must have exactly one prediction in the correct test fold. Resuming a subset preserves other completed folds in the summary. |
| A03 | Adjudication is explicitly for a particular sealed second review and frozen primary round. Decisions append to history with input hashes and decision IDs; they cannot appear in another reviewer's export. Legacy unscoped decisions are retained in the old table but never reused. |
| A04 | A round binds the queue, primary labels, evidence, protocol, annotation guide and application hashes. Source files are copied into a sibling `.sqlite3.round` package and evidence is served from that snapshot. Changed inputs require a new database; damaged snapshots fail closed. Exports identify the round. Databases containing old reviews without a package are refused, not silently migrated. |
| A05 | Newly generated DTW review queues include a run ID and content-derived candidate ID. Merging checks immutable interval/reference/component fields. The original sparse labels remain unchanged and are explicitly bound to the audited historical intervals in `annotations/pilot_identity_binding.json`; arbitrary sparse labels are rejected. |
| A06 | Adjudication supports `ungradable` (final exclusion) and `uncertain` (still unresolved), both requiring a reason. Exports state binary-metric eligibility and completion. Agreement reports its decided-item denominator separately. |
| A07 | Label application also writes `unresolved_pilot_queue.csv` containing all four unresolved cases. These form a separate supplemental round; the original 20-item agreement sample is unchanged. |
| A08 | The supported workflow remains an organizer-controlled local pilot. The entry page states that IDs are not authentication and independence is the organizer's responsibility. The launcher refuses non-loopback hosts. Account/token authentication remains necessary before any shared deployment. |
| A09 | Skeleton rendering uses whole-clip bounds, stable scale, and margins for both front/side views and the contact-sheet heading. All 20 sheets were regenerated in a new directory and the affected reference sheet visually checked. |
| A10 | Pilot generation requires explicit queue and output paths and refuses a nonempty destination. Documentation distinguishes new runs from the historical pilot. |
| A11 | Agreement HTML uses the dictionary's numeric item count. |

The posture display now calls the absolute-dot-product shoulder measure an
arm/torso axis angle. Its legacy CSV column names remain for compatibility;
it is not an oriented clinical shoulder range-of-motion measurement.

## Prepared review packet

The corrected packet is `results/audit_fixes/pilot_review_v3/` (20 sheets,
100 unchanged candidate intervals). It uses the historical audited queue
explicitly, with updated rendering. Primary labels remain preliminary,
nonclinical historical judgments; regeneration does not validate them.

Use separate new databases for the agreement sample and supplemental cases:

```powershell
.\.venv\Scripts\python.exe expert_review_app.py --legacy-sheets --queue results/audit_fixes/pilot_review_v3/second_review_queue.csv --sheets results/audit_fixes/pilot_review_v3/sheets --database results/audit_fixes/agreement_v3.sqlite3
.\.venv\Scripts\python.exe expert_review_app.py --legacy-sheets --queue results/audit_fixes/pilot_review_v3/unresolved_pilot_queue.csv --sheets results/audit_fixes/pilot_review_v3/sheets --database results/audit_fixes/supplemental_v3.sqlite3
```

Run those commands separately. Neither server was published or left running.
The four supplemental cases are also listed in
`annotations/unresolved_pilot_queue.csv`. Do not pool their agreement statistic
with the predefined 20-item sample. One case belongs to both sets; a supplemental
decision does not automatically modify the original round.

New ML experiments need a new output directory. Pre-fix training runs remain
historical results and cannot be resumed or passed off as verified new runs by
the strict aggregator. No provenance manifests were fabricated for them.

## Source anomalies and remaining work

### Follow-up software fixes (completed before human review)

Human review and clinical adjudication are deferred until after September 10,
as requested. They are not a prerequisite for these software fixes.

- The label loader now uses only workbook copies whose internal Subject ID
  matches the folder. E_ID16 uses its valid copies. With no matching copy,
  all five NE_ID2 targets are excluded from new manifests; raw scores are not edited.
- `prepare_dataset_version.py` verifies and selects canonical byte-identical
  position/orientation/timestamp bundles. It checks recording IDs, schemas,
  tracking states, finite coordinates, frame counts and increasing timestamps.
  It hashes and decodes RGB separately; video alignment is not inferred.
- All four identical numeric bundles passed verification and existing frame QC:
  E_ID14_Es4, E_ID15_Es4, NE_ID24_Es5 and P_ID3_Es5. Nine genuinely different
  multi-recording bundles remain excluded automatically. Malformed source rows
  are still rejected.
- ML exports and DTW summaries now report coverage by exercise and cohort,
  separating source exclusions from processing/QC exclusions.
- ML export and DTW runs accept `--folds-from` with the complete saved
  `subject_folds.json` or an NPZ. The full JSON also preserves assignments for
  subjects absent from the old NPZ because all their recordings failed QC.
- Training history records per-exercise validation MAE and sample counts at
  every epoch. Early stopping retains the original pooled-MAE rule.

The completed version is `results/dataset_v2_verified/`, containing the
manifest, hashed source-selection registry, source inventory, exported NPZ,
coverage metadata, and `verification.json`. It has **337 retained samples**:
333 common samples, four recovered samples and five excluded NE_ID2 rows.
All eleven exported arrays are exactly unchanged on the 333 common samples,
including targets and fold assignments. No existing model was retrained.

Reproduction (choose fresh output paths on subsequent runs):

```powershell
.\.venv\Scripts\python.exe prepare_dataset_version.py --output-dir results/dataset_v2_verified
.\.venv\Scripts\python.exe src/kimore_ml_data.py --manifest results/dataset_v2_verified/kimore_manifest.csv --output results/dataset_v2_verified/kimore_all_exercises_128.npz --folds-from results/interpretable_dtw/all_exercises/subject_folds.json
.\.venv\Scripts\python.exe audits/verify_dataset_version.py
```

For future training, explicitly pass
`--data results/dataset_v2_verified/kimore_all_exercises_128.npz` and a new
`--output-dir`. Historical default paths remain available for baseline
reproduction; the old artifacts were not overwritten. Direct comparisons of
old and new results must use the common sample population.

### Preserved source evidence

`annotations/source_anomaly_registry.json` records hashes and unresolved
decisions for 19 anomalous workbook copies and four identical position-file
groups. Each duplicate group includes a proposed canonical file and hashes,
row counts and parse checks for companion streams, plus RGB hashes when present.
This original sidecar is preserved as an audit snapshot. The completed numeric
bundle selections and their exact hashes are in the new dataset version.

The dataset audit reports internal subject-ID mismatches and TS versus PO+CF
discrepancies. Identity-mismatched copies cannot supply training targets.
Identity-consistent source TS values are preserved even when PO+CF differs.
Three recovered recordings have 2–4 extra RGB frames; their numeric data is
usable, while timing evidence remains explicitly unverified.

The audit's D04/D05 and methodological limitations still require the specified
work: source confirmation, clinical review, video
synchronization, full-paper verification, movement/time features and controlled
ablations with the same population and folds. No production retraining or
clinical labeling was performed in this repair pass.

## Validation

The final follow-up suite passed **122 tests**, including source-ID selection,
duplicate-stream validation, coverage accounting and preserved fold references.
The real 337-sample export passed exact comparison of all eleven arrays on its
333 common samples. `git diff --check` also passed.

The first repair pass passed **112 tests**. This includes actual
five-fold training on a tiny synthetic dataset, partial-run resume, repeated
resume, and verified aggregation. Added regressions cover provenance tampering,
missing checkpoint artifacts, six-fold rejection, OOF coverage, interval identity,
cross-reviewer adjudication, preserved decision history, changed primary labels,
frozen evidence, ungradable/unresolved final decisions, numeric HTML counts and
contact-sheet skeleton margins. Both real prepared review queues also passed
startup checks with temporary databases, and `git diff --check` passed.

Raw KIMORE models were not retrained. Generated review packets and test logs are
under Git-ignored `results/` and `tmp/`; preserve the corrected packet and each
database's adjacent frozen package when archiving a review round.
