# RGB validation review

The video workflow shows the reference first and requires acknowledgment of the
component/interval task before opening examples. A changed reference triggers
the introduction again. Reviewers judge a visible execution error in the named
body component and source interval, not overall instructor or clinical quality.
The reference remains available during review. Acknowledgment is self-reported;
the application does not claim to verify that the entire video was watched.

## Preparing a round

Provide a CSV video manifest with these columns (example identities only):

```csv
sample_id,subject_id,role,rgb_path,start_seconds,end_seconds
HOLDOUT_Es3,HOLDOUT,validation,videos/holdout.mp4,1.2,28.4
REFERENCE_Es3,REFERENCE,reference,videos/reference.mp4,0.8,25.1
```

Paths are relative to the manifest directory. Use browser-playable MP4 or WebM.
Start/end times must be independently checked movement boundaries in seconds,
within the source video. They are not inferred from silence or the file length.
Every queue sample and its existing reference must be declared. Missing RGB
files exclude the affected items and are recorded in the round manifest.
Invalid/undecodable videos or invalid boundaries stop setup for correction.
Existing review sheets and identity-matched primary labels are still required.

Provide a separate CSV containing `sample_id,subject_id` for **every recording
actually used to train the model being evaluated**. Export it from that run's
training inventory. The app rejects validation samples or subjects found there;
it also rejects reference/validation subject overlap. It cannot establish that a
manually supplied training inventory is complete. A fold number alone is not an
inventory, and samples used in other cross-validation folds are not a globally
untouched holdout. Do not reuse the final holdout for tuning or model selection.

```powershell
.\.venv\Scripts\python.exe .\expert_review_app.py --video-manifest annotations/validation_videos.csv --training-inventory annotations/evaluated_model_training.csv --queue annotations/validation_queue.csv --primary-labels annotations/validation_primary.csv --sheets results/validation/sheets --database results/validation/review.sqlite3
```

These files must be prepared from the selected model and recordings; the example
command does not create them or claim that the existing pilot is a holdout.
Use a new database for a new protocol/round. Video bytes are frozen with hashes;
the round records manifest/training-inventory hashes, movement boundaries,
included media and exclusion reasons. `--legacy-sheets` explicitly opens the
historical image-only pilot without the new holdout guarantees.

## Time normalization

The shared slider pauses both videos and maps progress `p` to
`start_seconds + p * (end_seconds - start_seconds)`. This aligns movement
endpoints, not intermediate phases. Individual play controls use original speed.
The pre-existing candidate interval and review sheet retain their original
recording coordinates and are explicitly distinguished from this new scale.
The transformation does not change DTW, skeleton coordinates, or model features.

## Choosing recordings and reviewers

Freeze the selected recordings before collecting ratings. Each reviewer can
independently review the same queue under a separate organizer-assigned ID.
For fewer recordings, seek more independent ratings per recording; with fewer
available reviewers, use more distinct recordings while retaining shared items
for measuring agreement. These are separate design dimensions, not an exact
exchange rate: repeated ratings cannot replace coverage of different people and
movement errors. Multiple intervals from one recording are not independent
recordings. Record unique subjects, recordings, intervals, and ratings per item
separately. The existing agreement screen compares each reviewer to the primary
reviewer; it is not a pooled multi-rater agreement analysis.
