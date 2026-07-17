# Egocentric adaptation for OpenVLA

## Wrist-camera preprocessing

| Step | Setting |
|------|---------|
| Source | `observation.images.wrist` (typically 640×480) |
| Color | BGR→RGB |
| Resize | 224×224 bilinear |
| Range at export | uint8 [0,255] |
| Mean/std norm | Deferred to official `PrismaticImageProcessor` |
| Augmentation | Optional wrist-safe photometric + mild center-preserving scale |

**Disabled:** horizontal/vertical flip, large rotation, time reversal, arbitrary reorder, crops that remove the gripper.

## Alignment to state/action targets

Default training pair:

`(wrist_image_t, language [, unused_state_t]) → action_t`

| Field | Value |
|-------|-------|
| Action offset | 0 frames (0 ms @ 30 FPS) |
| Max alignment error | ±0.5 frame |
| Episode end | drop examples lacking target |
| Invalid visual | excluded via Task 3 hard-valid + window membership |
| Strict vs conservative | policy selects which windows contribute frames |
| Chunks | if used later, all targets must stay in-episode, consecutive, no invalid gaps |

Source frames are **never renumbered**.

## How wrist changes evaluation

Wrist views reveal contact, enclosure, slip, wrong-object grasp, receptacle misalignment, loss-from-view, near-gripper collision, occlusion, blur, exposure, freeze, and failed release : failures joints alone miss (Task 4 egocentric protocol).

## Third-person pretraining → egocentric fine-tuning failures

| Failure mode | Symptom | Diagnostic | Mitigation | Task 4 slice |
|--------------|---------|------------|------------|--------------|
| Viewpoint distribution shift | Policy looks for tabletop overview cues | Compare top vs wrist success | Wrist-heavy fine-tune; view token (future) | nominal + occlusion |
| Object scale / FOV change | Misses close objects / overshoots | Scale histogram of object bbox proxies | Safe scale aug; progressive unfreeze | object_target_shift |
| Gripper dominance | Attends only to gripper pixels | Attention/saliency on gripper | Occlusion aug; mix third-person if available | occlusion_distractor |
| Self-occlusion | Fails when cube hidden by fingers | Phase-conditioned errors at grasp | Hard negatives; recovery demos | nominal grasp phases |
| Ego-motion blur | Unstable during fast wrist moves | Correlate fail with Task 1 blur flags | Mild blur aug; slower exec | lighting_camera / dynamics |
| Background motion | Treats background flow as object motion | Freeze robot, move camera in replay | Augment with background clutter | occlusion_distractor |
| Close-range lighting/exposure | Fail under glare | Exposure flags vs outcome | Brightness/contrast aug | lighting_camera_perturbation |
| Partial observability / target loss | Searches blindly | “target_lost_from_view” events | Teach recovery phases; top ablation | occlusion_distractor |
| Wrong spatial priors | Moves as if third-person camera-fixed | Open-loop action vs wrist motion | LoRA-first; small LR | nominal_held_out |
| Third-person positional shortcuts | Relies on abs table location | Shift object/target | Spatial shift slice | object_target_shift |
| Action–viewpoint coupling | Actions inconsistent with wrist ego-motion | Compare Δjoint to optical flow | Align offset; check latency | control_latency_dynamics |
| Rolling shutter / latency | Systematic lag | Timestamp skew metrics | Delay compensation (Task 4 ladder) | control_latency_dynamics |
| Embodiment / action-space mismatch | Implausible joint commands | Action histogram vs demos | Masked 6-D joint encoding (this project) | all slices |
| Catastrophic forgetting | Loses general vision after narrow FT | Probe Bridge-like prompts (if any) | LoRA before full FT; mix data | nominal + lighting |

**Not all mitigations are implemented.** Implemented here: wrist-safe aug, LoRA-first config, masked joint encoding, episode-grouped split, Task 4 linkage.

## Ablations

1. **Wrist-only (primary)**  
2. **Top-only** (separate run; same labels)  
3. **Wrist+top composite** (optional diagnostic; aspect-ratio / distribution-shift risk; **not** claimed as native multi-view)

True multi-view fusion requires model changes beyond stock OpenVLA-7B.

## Evaluation linkage

Use Task 4 metrics and slices; primary success from simulator/human : not the Task 4 wrist completion-proxy.
