# Teleoperation rollout protocol (ACT)

## Assumed policy

Concrete example: **ACT** trained on Task 3 windows from `lerobot/svla_so100_pickplace` @ `728583b5…`.

| Setting | Value |
|---------|-------|
| State history | 1 (current state; optional short history) |
| Wrist / top images | ablation-dependent |
| Language | optional instruction string if supported |
| Action chunk | 100 steps @ 30 Hz |
| Execution horizon | 50 steps with temporal ensembling |
| Action dim | 6 (SO-100 continuous + gripper) |
| Image resize | 224×224 (policy); color pipeline must match training |
| Normalization | train-set stats; gripper separate from joints |
| Training view | `conservative` (default) or `strict` |

No checkpoint is claimed to exist.

## Fixed 100-rollout matrix

| Slice | Rollouts | Perturbation |
|-------|---------:|--------------|
| `nominal_held_out` | 40 | none |
| `object_target_shift` | 20 | object XY / target yaw |
| `lighting_camera_perturbation` | 15 | brightness / gamma |
| `control_latency_dynamics` | 15 | action delay / friction |
| `occlusion_distractor` | 10 | occluder + distractors |

Seeds: `seed_base + k` for rollout `Rk`. Paired ablations reuse the same seed/env with different `training_view`, `observation_config`, or `state_features` (see `paired_comparison_templates` in `rollout_protocol.yaml`).

**Forbidden:** exposing simulator success, evaluation labels, or Task 2 labels as policy inputs.

## Per-rollout recording checklist

Policy checkpoint ID; dataset/curation version; seed; env config; object/target poses; camera config; states; actions; wrist/top frames; inference latency; control-loop latency; simulator task state; Task 2 phase/event/failure labels; visual-detector probability; final outcome; failure taxonomy; safety events; completion time.

## Metric definitions

### Primary task

- **Binary success rate** = `#success / N`; Wilson 95% CI via `wilson_interval`.
- **Partial-success rate** = `#partial_success / N`.
- **Failure rate by taxonomy** = multi-label episode counts / N (Task 2 `failure_types`).
- **Success by slice** + **worst-slice success**.

### Efficiency

Time to completion; executed action chunks; EE/joint path length; unnecessary-motion ratio (path / shortest-demo-like path); recovery attempts; interventions.

### Motion quality

Action smoothness (mean |Δa|); jerk proxy (mean |Δ²q|); max action step; oscillation count (sign changes of Δq); gripper-toggle count (gripper channel only); stall duration (|Δq|<ε sustained).

### Safety

Collision / severe collision counts; joint-limit violations (if GT limits); workspace violations; dropped objects; e-stop/abort counts.

### Systems

Policy inference latency; end-to-end control latency; missed deadlines; camera-frame age; obs/action timestamp skew.

### Visual

Wrist visibility; severe occlusion duration; blur/exposure/freeze rates; object loss-from-view; detector confidence/calibration (diagnostic only).

**Primary success source:** simulator state, trusted external evaluator, or adjudicated human : not the wrist detector under test.

## Proposed acceptance criteria (project thresholds)

| Criterion | Proposed |
|-----------|----------|
| Overall success | ≥ 80% |
| Wilson 95% lower | ≥ 70% |
| Nominal slice | ≥ 90% |
| Worst slice | ≥ 60% |
| Severe safety | 0 |
| Missed deadlines | ≤ 2% |
| Median completion | ≤ 1.5× demo median |
| Detector early FPR (true labels) | ≤ 5% |
| Detector episode recall (true labels) | ≥ 80% |

Why 100 rollouts: enough to stratify five slices and compare ablations with shared seeds; still high uncertainty (binomial CI width often >10 points).

## Stopping rules

Abort remaining rollouts if: severe safety failure; repeated control instability; camera/telemetry desync; invalid policy interface or action scaling.

## Sim-to-real diagnosis ladder

Order for “works in sim, fails on hardware” (no hardware claimed performed):

1. **Interface/schema parity** : compare obs/action keys/shapes; unit test mock policy I/O; signature: KeyError/shape mismatch; fix adapters.
2. **Joint ordering/units** : plot channel ranges vs demos; single-joint step; signature: mirrored/wrong axis; remap/scale.
3. **State/action normalization** : dump mean/std; open-loop replay of normalized actions in sim; signature: saturated actions; recompute stats.
4. **Gripper convention/range** : open/close command test; signature: inverted gripper; invert/scale.
5. **Control rate / chunk execution** : measure loop Hz vs 30; signature: slow/fast playback; fix horizon/ensemble.
6. **Latency / clocks** : timestamp skew probe; signature: delayed contact; delay compensation.
7. **Camera pipeline** : calibration poses; intrinsics/extrinsics/crop/color; signature: shifted targets; fix preprocess.
8. **Lighting/blur/occlusion/background** : hold robot, vary lights; signature: visual OOD; domain rand / real fine-tune.
9. **Dynamics/friction/backlash/limits** : chirp joints; signature: tracking lag; identify plant / adapt.
10. **Contact/object properties** : instrumented grasp; signature: slip despite closed gripper; retune gains / add compliance.
11. **Policy uncertainty / OOD** : shadow-mode inference on real obs; signature: high action variance; collect demos / fine-tune.
12. **Safety/low-level interference** : log overrides; signature: clipped commands; coordinate with safety layer.

Supporting procedures: recorded-action open-loop replay in sim; shadow-mode inference; calibration poses; latency measurement; single-joint + gripper sanity; domain randomization; real-data fine-tuning; action scaling validation.
