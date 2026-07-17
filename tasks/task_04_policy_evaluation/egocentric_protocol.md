# Egocentric evaluation protocol

## How wrist observations change evaluation

Joint/action streams describe **commanded and measured kinematics**. The wrist camera observes **contact geometry, object identity, occlusion, and photometric failure**. Evaluation must therefore couple visual evidence to the same `(episode_index, frame_index, timestamp)` identity used in Tasks 2–3.

## Failures wrist video reveals that joints miss

- Contact actually occurred (or not) despite gripper angle.
- Gripper enclosed the object vs pinched air / wrong object.
- Object slip with nominally closed gripper.
- Misalignment with the receptacle.
- Object or target loss from view.
- Near-gripper collisions.
- Occlusion during critical phases.
- Motion blur on approach/transport.
- Exposure failure; frozen/stale frames.
- Placement motion without successful release.
- “Successful-looking” joint trajectories with failed physical interaction.

## Alignment with robot telemetry

Every visual failure label retains:

- Episode, frame, timestamp
- Teleoperation phase (Task 2)
- State/action context (Δ-norms, gripper)
- Camera-quality context (blur/exposure/freeze flags from Task 1/3)

Visual events may overlap phases; primary phase remains one-per-frame when exhaustive.

## Phase-level evaluation

Score failures and recoveries within:

`approach → pregrasp_alignment → grasp → lift → transport → place → release → recovery`

Report rates conditioned on phase occupancy (frames or segments).

## Camera-specific failures

Evaluate separately:

| Family | Examples |
|--------|----------|
| Geometric interaction | miss, slip, wrong object, incomplete release |
| Visibility | object/target lost, severe occlusion, self-occlusion vs unusable occlusion |
| Photometric / stream | blur, exposure, freeze/stale |
| Placement appearance | over-target vs seated; release visible |

Top camera is contextual / adjudication aid; primary egocentric metrics use wrist.

## Cross-modal disagreement and adjudication

| Conflict | Precedence |
|----------|------------|
| Simulator success, wrist ambiguous | Keep sim success; flag `visual_uncertain`; human review sample |
| Wrist detector success before sim completion | **Ignore for primary label**; log early trigger diagnostic |
| Top shows placement, wrist occluded | Prefer top + sim; mark wrist `not_visible` |
| Joints nominal, visual contact failed | Outcome `failure` or `partial_success` per Task 2; joints alone insufficient |

Outcome becomes `uncertain` when simulator/external GT unavailable **and** cameras disagree without adjudicator.

## Visual metrics in rollouts

Record wrist visibility; occlusion duration; blur/exposure/freeze rates; object-loss events; detector probabilities (diagnostic). Primary success remains non-detector.

## Relation to Task 3 views

Conservative/strict windows change training distribution, not evaluation identity. Rollouts always log source coordinates; curated-view ID is metadata on the policy under test.
