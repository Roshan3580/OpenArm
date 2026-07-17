# Visual success-detector prototype (bonus)

## Critical limitation

`lerobot/svla_so100_pickplace` does **not** provide verified positive/negative task-success labels.

This prototype is a **terminal completion-state proxy classifier**:

- It distinguishes likely **terminal** wrist frames from **early** task frames.
- It is **not** a failure-aware success detector.
- It does **not** prove that final frames are successful placements.
- True validation requires Task 2 success / partial-success / failure labels.

## Final-frame inspection

A deterministic contact sheet of final frames from 10 episodes was written to `artifacts/task_04_policy_evaluation/final_frame_contact_sheet.png` with notes in `final_frame_inspection.json`.

Terminal frames are visually consistent with end-of-episode task context in this corpus, but **episode success is not verified**.

## Proxy-label methodology

| Class | Definition |
|-------|------------|
| Proxy positive | Frames from the final 15% of each episode |
| Proxy negative | Frames from the first 40% of each episode |
| Excluded | Middle 45% (ambiguous) |

Sampling: 4 positives + 4 negatives per episode, deterministic linspace indices.  
Manifest: `artifacts/task_04_policy_evaluation/proxy_label_manifest.json` (400 rows; 200/200).

**Not used as model features:** frame index, timestamp, % through episode, episode length, episode ID.

## Leakage controls

1. Split **episodes** before sampling frames (70% / 10% / 20%, seed 42).
2. Assert no episode overlap (`episode_split.json`).
3. Fit standardizer and models on train only.
4. Select frame threshold on validation only; report test once.
5. Select temporal threshold on validation episodes only.

## Feature / model design

| Item | Value |
|------|-------|
| Input | Wrist RGB only |
| Resize | 96×96 |
| HSV hist bins | 8×8×8 (512-D) |
| HOG | 9 orientations, 16×16 cells, 2×2 blocks |
| Full feature dim | 1412 (512 HSV + 900 HOG) |
| HSV-only dim | 512 |
| Model | Class-weighted L2 logistic regression (C=1.0, max_iter=200) |
| Baselines | Majority class; HSV-only logistic regression |
| Seed | 42 |

Models/features cached under Git-ignored `data/models/` and `data/evaluation_cache/`.

## Episode split

| Split | Episodes | Frames (pos/neg balanced) |
|-------|---------:|---------------------------|
| Train | 35 | 280 (140 pos) |
| Val | 5 | 40 (20 pos) |
| Test | 10 | 80 (40 pos) |

## Frame-level test metrics (proxy labels)

Main (HSV+HOG), threshold 0.225 (val F1):

| Metric | Value |
|--------|------:|
| Accuracy | 0.725 |
| Balanced accuracy | 0.725 |
| Precision | 0.680 |
| Recall | 0.850 |
| F1 | 0.756 |
| Specificity | 0.600 |
| AUROC | 0.833 |
| AP | 0.845 |
| Brier | 0.213 |
| ECE | 0.190 |
| Confusion | TP34 TN24 FP16 FN6 |

Episode-grouped bootstrap 95% CI (200 resamples):

| Metric | Point | Low | High |
|--------|------:|----:|-----:|
| F1 | 0.756 | 0.659 | 0.845 |
| AUROC | 0.833 | 0.747 | 0.916 |
| Accuracy | 0.725 | 0.613 | 0.813 |

Baselines (same test set):

| Model | Accuracy | F1 | AUROC |
|-------|---------:|---:|------:|
| Majority | 0.500 | 0.667 | 0.500 |
| HSV-only logistic | 0.688 | 0.719 | 0.788 |

**Do not overstate:** high scores are expected when early vs late frames differ systematically.

## Temporal aggregation (proxy)

Hysteresis: 4-of-5 votes; temporal threshold selected on validation to minimize early triggers (0.35).

Test episodes (n=10):

| Proxy temporal metric | Value |
|-----------------------|------:|
| Fraction first-trigger in terminal region | 0.00 |
| False early-trigger rate | 1.00 |
| Trigger still active in terminal (sticky) | 1.00 |
| Median detection latency | n/a (no clean terminal-first triggers) |

Interpretation: the proxy model fires before the terminal band on held-out episodes, so sticky hysteresis is on by the end without a clean “completion” onset. This is a **failure mode of the proxy approach**, not evidence of a working success detector.

Plots: `temporal_detection_examples.png`, `confusion_matrix.png`, `precision_recall_curve.png`, `calibration_plot.png`.

## Requirements for a genuine success detector

1. Task 2 labels: success / partial_success / failure / aborted.
2. Negatives from true failures, not only early frames.
3. Hard negatives near success (almost-placed).
4. Separate calibration set with adjudicated labels.
5. Primary rollout success still from simulator / human — never from this detector alone.

## What this bonus demonstrates

- Reproducible wrist-only feature pipeline.
- Episode-grouped evaluation without identity leakage.
- Honest separation of proxy metrics vs task success.
- Temporal hysteresis machinery ready for true labels later.
