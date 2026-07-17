# Task 2 : Synchronized Multimodal Labeling Design

**Dataset:** `lerobot/svla_so100_pickplace` @ `728583b5eaf9e739a7f119e2def466fa1d552402`  
**Task:** “Pick up the cube and place it in the box.”  
**Status:** Teleoperation + egocentric labeling design **complete** (schemas, alignment, agreement, validator, synthetic sample, tests).  
**No real ground-truth labels were produced in this take-home.** The sample annotation is synthetic / illustrative only.

## 1. Objective

Define a practical, versioned annotation contract for synchronized teleoperation (state/action + dual cameras) and egocentric (wrist-camera) labels so that:

- Episode outcomes, phase segments, and motion events can be labeled jointly.
- Wrist-specific visibility, interaction state, failure moments, and attention-proxy ROIs can be recorded without inventing a second timeline.
- Labels remain valid across conservative and strict curated views because they use **immutable source coordinates**, not reindexed curated rows.

## 2. Tool choice

**MVP tool: Label Studio.**

### Why Label Studio

- Strong for **episode-level** classifications, **temporal segments**, and **timestamped events** via timeline / video labeling templates.
- Easy export to JSON that we validate against `annotation_schema.json`.
- Low ops cost for a take-home / pilot corpus (5–10 pilot episodes, then stratified double-annotation).

### Synchronized review asset

Annotators review a **preprocessed synchronized composite review video** containing:

1. Wrist-camera video as the **primary** view.
2. Top-camera video as a **contextual** inset / side panel.
3. A compact rendered robot-state / gripper timeline strip.
4. On-screen **canonical** `episode_index`, `frame_index`, and `timestamp`.

**Honest limitation:** Label Studio does **not** natively guarantee frame-perfect synchronized multi-video playback for arbitrary dual-stream robotics layouts in the configuration intended here. We therefore preprocess a composite review MP4 while **retaining all annotations in the original canonical coordinate system** `(dataset_revision, episode_index, frame_index, timestamp)`. The composite is a review aid only; it is not a new identity space.

The supplied `label_studio_config.xml` is a **design template**. It has **not** been production-tested against a running Label Studio server in this take-home.

### Limitations (Label Studio)

- Weak native support for dense joint time-series overlays and robotics-specific phase logic.
- Dense multi-object tracking / polygons are awkward compared with CVAT.
- Multi-stream sync requires preprocessing (as above).

### When a custom synchronized robotics viewer becomes worthwhile

- Continuous labeling of hundreds of episodes with joint/gripper scrubbing.
- Tight coupling to curation manifests and OpenArm teleop playback.
- Need for live pre-label proposals from robot signals inside the same UI.

### CVAT comparison

- **Stronger** for dense boxes/tracks and polygon QA.
- **Weaker** as the *sole* interface for joint-state-aware phase labeling and episode outcome + recovery taxonomy.
- Practical split: Label Studio (or custom viewer) for phases/events/outcomes; optional CVAT pass only when dense wrist ROIs are required at scale.

## 3. Label hierarchy

| Level | Teleoperation | Egocentric (wrist) |
|-------|---------------|--------------------|
| Episode | outcome, quality, failure types, recoveries, intervention | (inherits episode context; visual quality flags optional) |
| Segment | primary phase timeline + motion quality | visual interaction state aligned to phase segments |
| Event / frame | motion and contact events | failure moments, visibility, sparse attention-proxy ROIs |

Prefer **one primary phase per frame**. Separate **event** labels may overlap phases. Dense boxes are optional (sparse keyframes + interpolation preferred).

## 4. Teleoperation schema

See `teleop_schema.yaml`.

**Episode:** `task_outcome`, `demonstration_quality`, multi-label `failure_types`, recovery count, intervention flag, notes, confidence.

**Phases (primary, mutually exclusive per frame when exhaustive):**  
`idle_or_reset`, `approach`, `pregrasp_alignment`, `gripper_close`, `grasp_verification`, `lift`, `transport`, `place_alignment`, `lower`, `release`, `retract`, `recovery`, `terminal_hold`, `uncertain`.

**Events:** `motion_start`, `gripper_close_start`, `first_contact`, `stable_grasp`, `object_liftoff`, `transport_start`, `object_over_target`, `placement_contact`, `release_start`, `release_complete`, `task_success_visible`, `failure_onset`, `recovery_start`, `collision`, `object_slip`, `operator_intervention`.  
Repeatable: recovery/failure/collision/slip/intervention family; most success-path events once per attempt.

## 5. Egocentric schema

See `egocentric_schema.yaml`.

- Object-interaction flags with explicit `unknown` / `not_visible` (never silent occlusion).
- Visual interaction states aligned to teleop phases (not a second independent phase taxonomy).
- Failure moments (visual).
- Ordinal visibility; self-occlusion of arm/gripper kept separate from unusable camera occlusion.
- `attention_proxy_roi`: **not human gaze / eye tracking** : task-relevant visual region proxy only.

## 6. Temporal alignment

Canonical identity: `(dataset_revision, episode_index, frame_index, timestamp)`.

- Segments use half-open frame intervals `[start_frame, end_frame)`.
- Label times snap to nearest canonical robot frame; original unsnapped time retained; alignment error recorded.
- Max acceptable alignment error: **±0.5 frame** (≈ ±16.7 ms @ 30 FPS). Beyond tolerance → `status=needs_review` (not silently accepted).
- Annotations **must never be reindexed** after frame filtering; curated views are metadata filters over the same coordinates.
- See `alignment_and_agreement.md`.

## 7. Agreement protocol

Pilot 5–10 episodes; double-annotate stratified 20%; adjudicator resolves disagreements; adjudicated labels stored separately. Metrics: Cohen’s/Fleiss’/Krippendorff for categories; framewise kappa + temporal IoU + boundary error for phases; event F1 in ±3 frames / ±100 ms; ROI IoU. Project targets (not universal standards) documented in `alignment_and_agreement.md`.

## 8. Pre-labeling workflow

1. Robot-signal pre-labels (motion magnitude, gripper transitions).  
2. Optional visual tracking suggestions.  
3. Humans correct pre-labels (**never** presented as ground truth).  
4. Schema validation (`scripts/validate_annotations.py`).  
5. Temporal-consistency checks.  
6. Double-annotate agreement subset.  
7. Adjudicate.  
8. Freeze versioned label release.

## 9. Annotation QA

Automatic checks: required fields, overlapping primary phases, exhaustive gaps, invalid ranges, frame/timestamp disagreement, events outside bounds / incompatible phases, ROI outside `[0,1]`, unknown enums, duplicate IDs, missing source frames, invalid revision.

## 10. Simplifying assumptions

- 30 FPS canonical clock for this dataset.
- One primary phase per frame when exhaustive mode is required.
- No real eye-tracking hardware; attention ROIs are proxies.
- Composite review video may differ in pixel layout from raw wrist frames; boxes are always in **wrist image normalized coordinates**, not composite pixels.
- Label Studio XML is illustrative, not server-verified.

## 11. Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| Label Studio MVP | Fast episode/segment/event labeling | Weak dense tracking; sync via preprocess |
| Immutable source coords | Labels survive curation policy changes | Annotators must not “renumber” after filters |
| Sparse ROIs | Practical labeling cost | Less dense supervision for detectors |
| Shared phase taxonomy | Cross-modal agreement | Wrist states are projections, not a second ontology |

## 12. README-ready summary

Task 2 designs a synchronized labeling contract for teleoperation and egocentric streams on `svla_so100_pickplace`, using Label Studio as the MVP tool with a preprocessed multi-view review asset, immutable `(revision, episode, frame, timestamp)` identities, half-open phase segments, explicit agreement metrics, and a bounded validator. No real ground-truth labels were collected; `sample_annotation.json` is synthetic. Attention ROIs are gaze proxies, not measured gaze. The scheme targets this pick-and-place corpus and generalizes to OpenArm; curated-view policy is metadata, not a new annotation coordinate system.

## Files

| File | Role |
|------|------|
| `teleop_schema.yaml` | Teleoperation label taxonomy |
| `egocentric_schema.yaml` | Wrist / egocentric taxonomy |
| `annotation_schema.json` | Unified export JSON Schema |
| `alignment_and_agreement.md` | Alignment + IAA protocol |
| `label_studio_config.xml` | Illustrative Label Studio template |
| `sample_annotation.json` | Synthetic example (`illustrative_not_ground_truth`) |
| `../../scripts/validate_annotations.py` | Bounded QA validator |
| `../../tests/test_labeling_schema.py` | Schema / QA tests |
| `../../artifacts/task_02_labeling_design/schema_validation.json` | Validator output on sample |
