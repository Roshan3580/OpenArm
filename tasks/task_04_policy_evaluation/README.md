# Task 4 : Policy Evaluation Design

**Dataset:** `lerobot/svla_so100_pickplace` @ `728583b5eaf9e739a7f119e2def466fa1d552402`  
**Task:** Pick up the cube and place it in the box.  
**Policy example:** ACT (protocol generalizes to Diffusion Policy).  
**Status:** Teleoperation protocol **complete** (executable, validated). Egocentric protocol + offline wrist detector prototype **complete**.

**Honesty constraints**

- No ACT / Diffusion Policy was trained in this repository.
- No policy rollouts were executed; `rollout_protocol.yaml` is an evaluation design matrix, not result rows.
- No hardware evaluation was performed.
- The wrist “success detector” is a **terminal-completion proxy classifier**, not a verified success/failure detector.

## 1. Scope

Design an executable evaluation plan for an ACT-style policy trained on Task 3 curated windows, covering:

- Teleoperation rollout matrix, metrics, acceptance criteria, and sim-to-real diagnosis.
- Egocentric / wrist-camera evaluation aligned to robot telemetry and Task 2 taxonomies.
- Bonus: offline wrist-only completion-proxy prototype on real frames.

## 2. Assumed policy (ACT)

| Item | Assumption |
|------|------------|
| Inputs | Current (or short-history) robot state; wrist image; optional top image; optional language instruction |
| Outputs | Action chunk of length 100; execute first 50 with temporal ensembling |
| Action dim | 6 (SO-100) |
| Control rate | 30 Hz |
| Images | Resize/crop to 224×224 in policy stack (detector prototype uses 96×96) |
| Normalization | Dataset statistics from training view; gripper scaled separately |
| Training views | Conservative (default) vs strict Task 3 windows |
| Checkpoint | **None claimed** |

## 3. Evaluation architecture

```text
Task 3 curated view → (hypothetical) ACT train
                         ↓
              Fixed 100-rollout sim matrix
                         ↓
         Metrics + Task 2 labels + visual detector
                         ↓
              Acceptance / diagnosis ladder
```

Primary success labels come from simulator state, trusted external evaluator, or adjudicated humans : **never** from the wrist detector under test.

## 4. Rollout matrix

| Slice | N | Purpose |
|-------|--:|---------|
| Nominal held-out | 40 | In-distribution |
| Object/target shift | 20 | Spatial generalization |
| Lighting/camera | 15 | Visual robustness |
| Latency/dynamics | 15 | Dynamics robustness |
| Occlusion/distractor | 10 | Perception robustness |
| **Total** | **100** | |

Paired comparison templates fix seeds across conservative/strict, wrist/top/both, raw/smoothed. See `rollout_protocol.yaml` and `teleop_protocol.md`.

## 5. Teleoperation metrics

Primary task, efficiency, motion quality (gripper separate), safety, systems, and visual metrics : definitions in `teleop_protocol.md`.

## 6. Egocentric metrics

Wrist-specific failure modes, phase-level evaluation, camera-quality flags, and cross-modal adjudication : `egocentric_protocol.md`.

## 7. Acceptance criteria (proposed project thresholds)

- Overall success ≥ 80%; Wilson 95% lower ≥ 70%
- Nominal ≥ 90%; no slice < 60%
- Zero severe safety events; missed deadlines ≤ 2%
- Median completion ≤ 1.5× demo median
- Visual detector early FPR ≤ 5%; episode recall ≥ 80% **on genuinely labeled data** (not claimed here)

100 rollouts leave wide uncertainty (e.g. 80/100 → Wilson ~71–87%).

## 8. Sim-to-real diagnosis

12-stage ladder from interface/schema parity through contact dynamics and OOD behavior : `teleop_protocol.md`. No hardware tests claimed.

## 9. Visual success-detector prototype

HSV+HOG → class-weighted logistic regression on wrist pixels only; episode-grouped 70/10/20 split; validation-only thresholds; temporal 4-of-5 hysteresis. Details and results: `visual_success_detector.md`.

## 10. Results and limitations

- Protocol validates: 100 rollouts, correct slices, no GT leakage into policy inputs.
- Detector test metrics are **proxy** metrics (see artifacts). High AUROC does not imply failure-aware success detection.
- Temporal proxy metrics remain brittle; early triggers illustrate the gap to real success labels.

## 11. Simplifying assumptions

- Simulation evaluation environment exists matching SO-100 pick-and-place.
- ACT hyperparameters above are representative, not measured from a trained run.
- Proxy labels use episode progress bands because verified success labels are absent.

## 12. What would change with hardware

- Replace simulator GT with adjudicated human / instrumented success.
- Run shadow-mode inference and open-loop replay stages of the diagnosis ladder.
- Collect true success/partial/failure labels (Task 2) before claiming a success detector.

## 13. README-ready summary

Task 4 defines an executable 100-rollout ACT evaluation protocol for teleoperation and egocentric streams on `svla_so100_pickplace`, with explicit metrics, acceptance thresholds, and a sim-to-real diagnosis ladder. No policy was trained and no rollouts were run. A wrist-only terminal-completion **proxy** classifier was prototyped on real frames with episode-grouped splits; it is not a verified success detector.

## Commands

```bash
python scripts/validate_rollout_protocol.py
python scripts/evaluate_success_detector.py
pytest tests/test_evaluation.py -q
```
