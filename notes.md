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
