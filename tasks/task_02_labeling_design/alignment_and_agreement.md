# Alignment contract and inter-annotator agreement

Dataset: `lerobot/svla_so100_pickplace`  
Pinned revision: `728583b5eaf9e739a7f119e2def466fa1d552402`  
Canonical FPS: 30 (`dt = 1/30 s`)

## 1. Canonical annotation identity

Every annotation references:

| Field | Required |
|-------|----------|
| `dataset_repo_id` | yes |
| `dataset_revision` | yes (immutable pin) |
| `schema_version` | yes |
| `episode_index` | yes |
| `start_frame` / `end_frame` | segments (half-open) |
| `frame_index` | events / frame labels |
| `start_timestamp` / `end_timestamp` / `timestamp` | yes where applicable |
| `annotator_id` | yes (pseudonymous OK) |
| `created_at` | yes (ISO-8601 UTC) |
| `status` | yes (`draft`, `submitted`, `adjudicated`, `needs_review`, `rejected`) |
| `confidence` | yes in `[0, 1]` |
| `review_status` | yes (`unreviewed`, `accepted`, `rejected`, `adjudicated`) |
| `source_modality` | yes (`teleoperation`, `egocentric`, `joint`) |
| `curation_policy_context` | recommended (`raw`, `conservative`, `strict`) |

**Canonical identity tuple:**

`(dataset_revision, episode_index, frame_index, timestamp)`

Temporal segments use **half-open** frame intervals:

`[start_frame, end_frame)`

Example: frames `{10,11,12}` → `start_frame=10`, `end_frame=13`.

## 2. Timestamp snap and alignment error

1. Annotator marks a time `t_raw` on the review timeline (may be continuous).
2. Snap to nearest canonical robot frame:

   `frame_index = round(t_raw * fps)` clamped to episode bounds  
   `timestamp = frame_index / fps` (or the stored tabular timestamp for that frame when available)

3. Retain:
   - `timestamp_raw` / `start_timestamp_raw` / `end_timestamp_raw` (unsnapped)
   - `alignment_error_s = timestamp_snapped - timestamp_raw` (signed)
   - `alignment_error_frames = alignment_error_s * fps`

4. **Maximum acceptable alignment error:** `|alignment_error_frames| ≤ 0.5` (≈ 16.7 ms @ 30 FPS).

5. If error exceeds tolerance:
   - Set `status = needs_review`
   - Do **not** silently accept
   - QA validator emits `alignment_error_exceeds_tolerance`

## 3. Labels across curated views

- Conservative / strict curated views are **filters over source timesteps**, not new indices.
- Annotations always point at source `(episode_index, frame_index, timestamp)`.
- A label remains valid if its referenced frames still exist in the source revision, even if a curated policy excludes them from training windows.
- Downstream consumers apply `curation_policy_context` to decide whether to *use* a label for a given training view.
- **Never independently reindex** annotations after frame filtering — that silently breaks cross-modal joins and agreement metrics.

## 4. Sampling protocol (IAA)

1. **Pilot:** 5–10 episodes; refine guidelines and examples.
2. **Double-annotate** a stratified **20%** of the final corpus.
3. Stratify to include: successes, failures, short/long episodes, high-motion episodes, recoveries, and visual-quality flags (overexposure / frozen candidates from Task 1/3).
4. **Adjudicator** resolves disagreements.
5. Store adjudicated labels distinctly (`review_status=adjudicated`, `adjudicator_id`) — never overwrite raw annotator rows.

## 5. Metrics

### Episode categorical labels

- Cohen’s kappa (two annotators).
- Fleiss’ kappa or Krippendorff’s alpha when >2 annotators or missing labels.
- Jaccard (or multi-label F1) for `failure_types`.

### Temporal phases

- Framewise Cohen’s kappa (primary phase per frame).
- Segment temporal intersection-over-union (IoU).
- Boundary error in frames and milliseconds.
- Boundary-tolerant F1 (match if boundaries within tolerance).

### Motion / event labels

- Precision, recall, F1 within a tolerance window (±3 frames or ±100 ms).
- Median absolute event-time difference.
- Exact-frame agreement is **too strict** for motion boundaries because human perception and teleop latency smear contact/onset by several frames.

### Visual spatial labels

- Bounding-box IoU; polygon IoU where used.
- Visibility-state kappa.
- Normalized center-distance when useful.

## 6. Project acceptance targets

These are **project targets for this take-home corpus**, not universal standards:

| Metric | Target |
|--------|--------|
| Episode-label kappa | ≥ 0.75 |
| Phase framewise kappa | ≥ 0.70 |
| Mean segment temporal IoU | ≥ 0.70 |
| Event F1 within ±3 frames or ±100 ms | ≥ 0.80 |
| Median boundary error | ≤ 3 frames |
| Median ROI IoU | ≥ 0.60 |

## 7. When targets are missed

1. Clarify written guidelines with borderline examples.
2. Merge ambiguous categories (e.g. `uncertain` policy).
3. Add annotated examples / gold snippets from the pilot.
4. Retrain annotators.
5. Repeat the pilot.
6. Increase adjudication rate on weak categories.

## 8. Exhaustive vs non-exhaustive phase timelines

- **Exhaustive (default for training releases):** every frame in `[0, N)` covered by exactly one primary phase; gaps and overlaps are QA failures.
- **Non-exhaustive (pilot / sparse):** only labeled spans; gaps allowed; overlaps of primary phases still forbidden.
- Short transitions (<3 frames): prefer absorbing into neighboring dominant phase **or** label `uncertain` — document the chosen rule in the release notes; this project’s default is absorb if <3 frames unless a named event fires inside.
- Ambiguous boundaries: mark `uncertain` segment or lower confidence; adjudicator decides.
- Recovery: insert `recovery` segment; after recovery, resume the nominal phase that best describes the restarted attempt (often `approach` or `pregrasp_alignment`). Events `recovery_start` / `failure_onset` should fall inside or at the boundary of recovery-compatible phases.
