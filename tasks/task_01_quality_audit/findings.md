# Task 1 Findings — Quality Audit (Teleoperation + Egocentric)

## 1. Scope and dataset-selection decision

Task 1 audits **two** corpora:

| Role | Dataset | Artifact dir |
|------|---------|--------------|
| Original teleop baseline | `lerobot/aloha_sim_insertion_human` | `artifacts/task_01_quality_audit/aloha_sim_insertion_human/` |
| **Primary paired dataset (Tasks 2–5)** | `lerobot/svla_so100_pickplace` | `artifacts/task_01_quality_audit/svla_so100_pickplace/` |

Three candidates were compared (`tasks/task_01_quality_audit/dataset_selection.md`, `artifacts/task_01_quality_audit/dataset_comparison.json`). **Selected:** `svla_so100_pickplace` for explicit wrist+top cameras, sync’d teleop fields, pick-and-place task, and ~470 MB size.

## 2. Original ALOHA audit

| Field | Value |
|-------|-------|
| Revision | `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` |
| Scope | Full tabular: 50 episodes × 500 frames = 25,000 @ 50 Hz |
| Cameras | `observation.images.top` only → `verified_external` |
| NaN/Inf | 0 |
| Timestamp / frame integrity | 0 gaps, duplicates, or mismatches |
| Trajectory length | All episodes length **500** |
| State discontinuities | 0.221% of within-episode Δ-elements (`\|Δ\| > max(8·MAD, 0.05)`) |
| Action “outliers” (all dims) | 2.17% overall; **left_gripper 13.4%, right_gripper 17.0%** under modified-z>10 |

**Gripper reinterpretation:** histogram analysis shows **bimodal / open–closed concentration** on both ALOHA grippers (`bimodal_or_open_closed_concentrated=true`). The 13–17% gripper outlier rates are **not treated as corruption**; they are expected grasp-mode behavior. Do **not** delete normal grasp transitions. Joint channels should be filtered separately from grippers.

## 3. Why ALOHA could not satisfy the egocentric requirement

ALOHA meta exposes a single video feature, `observation.images.top`. That name and ALOHA hardware convention identify an overhead/workspace camera (`verified_external`). No wrist/ego feature exists. Empirical wrist metrics were correctly **blocked** on that corpus.

## 4. Selected paired dataset and revision

| Field | Value |
|-------|-------|
| Repo | `lerobot/svla_so100_pickplace` |
| Revision | `728583b5eaf9e739a7f119e2def466fa1d552402` |
| Robot | SO-100 |
| Task | “Pick up the cube and place it in the box.” |
| Episodes / frames | 50 / 19,631 |
| FPS | 30 |
| License | Apache-2.0 |
| Approx size | ~470 MB |
| Access | `huggingface_hub` + parquet + OpenCV (LeRobot package not required) |
| Tabular scope | **Full dataset** |
| Wrist decode scope | **Every wrist frame** (19,631 / 19,631; coverage 100%; all 50 episodes) |
| Audit runtime | **279.7 s** |

## 5. Verified schema and camera viewpoints

**Teleop fields:** `observation.state` [6], `action` [6], `timestamp`, `frame_index`, `episode_index`, `task_index`, `index`.

Motor names: `main_shoulder_pan`, `main_shoulder_lift`, `main_elbow_flex`, `main_wrist_flex`, `main_wrist_roll`, `main_gripper`.

| Camera key | Viewpoint | Evidence |
|------------|-----------|----------|
| `observation.images.wrist` | **`verified_egocentric`** | Feature token `wrist`; contact sheet (eps 0/24/49 × begin/mid/end) shows gripper fingers and close-up cube grasp |
| `observation.images.top` | **`verified_external`** | Feature token `top`; workspace overview of arm, cube, and bin |

## 6. Teleoperation findings on the paired dataset

| Metric | Result |
|--------|--------|
| Episode lengths | min **326**, max **575**, mean **392.6**, median **380**, std **42.9** |
| State/action NaN/Inf | **0 / 0** of 117,786 elements each |
| Non-monotonic / duplicate timestamps | **0** |
| Timestamp gaps (>1.5×Δt) | **0** |
| Frame-index gaps | **0** |
| Length mismatches | **0** |
| Near-stuck dims | **0** |
| Robust outliers (all dims, joints-only) | **0%** state and action |
| Boundary saturation (empirical) | 1 dim soft-flagged: `main_elbow_flex` (diagnostic; bounds are empirical, not hardware limits) |
| Within-episode state discontinuities | **8,734** flags, rate **7.43%** of Δ-elements |
| Within-episode action discontinuities | **9,068** flags, rate **7.72%** |
| Largest \|Δstate\|_∞ examples | ~**5.6°** (ep 39), ~**5.4°** (ep 34), ~**4.9°** (ep 17) on arm joints with smooth neighbors — **review candidates**, not auto-corrupt |

Discontinuity rule: `|Δ| > max(8·MAD, abs_floor=1.0, 0.01·dim_range)` **within episodes only**. Degree-scale SO-100 joints make small absolute floors inappropriate without `range_frac`.

**Gripper:** `main_gripper` is analyzed separately. State gripper is heavily massed near the low end (~49% near low 15% of range) with grasp dynamics; **not labeled corruption**. Action gripper shows multiple histogram peaks. Outlier rate on grippers under MAD-z is **0%** here.

## 7. Wrist-camera quality findings

Full-stream decode of `observation.images.wrist` (AV1, 640×480 @ 30 FPS).

| Metric | Distribution / rate | Threshold type |
|--------|---------------------|----------------|
| Laplacian variance | mean 33.8, median **9.6**, p05 3.7, p95 139.1, max 293 | Absolute blur cut **50** = **exploratory heuristic** (miscalibrated for this camera: flags **78.5%**) |
| Mean luma | mean 137.0, min 107.4, max 191.3 | — |
| Underexposed | **0** frames | Hard engineering (mean<40 or sat≥15%) |
| Overexposed | **267** frames (**1.36%**) | Hard engineering |
| Entropy (bits) | mean 7.51, min 6.22, max 7.73 | Low-H<3.5 = exploratory; **0%** flagged — does **not** prove occlusion |
| Exact adjacent duplicates (array-equal) | **3,227** (**16.48%** of **19,581** within-episode adjacent pairs) | Hard (identical pixels) |
| Near-lossless duplicates (0 < MSE ≤ 1) | **991** (**5.06%**) | Hard engineering band |
| Near-duplicates (1 < MSE ≤ 25) | **7,405** (**37.82%**) | Screening (mutually exclusive) |
| Decode failures / missing | **0 / 0** | — |

**Within-episode duplicate denominator (corrected):** adjacent pairs are counted only inside an episode after sorting by `frame_index`. With 19,631 frames and 50 episodes the correct denominator is **19,631 − 50 = 19,581**. An earlier recount used 19,630 (global stream order), which incorrectly included the 49 last→first transitions between episodes; 19 of those boundary pairs had been classified as near-duplicates and are now excluded (`near` 7,424 → 7,405). Exact and near-lossless counts were unchanged.

**Duplicate terminology (corrected):** categories are mutually exclusive. Prior Task 1 wording that labeled MSE ≤ 1 as “exact” was wrong; those pairs are split into true exact (array equality) vs near-lossless.

**Interpretation:** High “blur” flag rate means the absolute Laplacian threshold is too aggressive for this wrist stream’s texture/optics — treat as **diagnostic / recalibrate**, not a dataset defect claim. Overexposure at 1.36% is a **soft review** flag. Exact duplicates at 16.4% and near-lossless at 5.0% are **soft review** (possible slow motion, encode reuse, or freeze) — not automatic drops. Low entropy never fired; close-up grasps remain high-entropy useful views (expected self-occlusion ≠ unusable occlusion).

Montages: `wrist_montage_{normal,low_sharpness,exposure,duplicate}.png` with episode/frame/timestamp/metric labels.

## 8. Video and temporal-alignment findings

Per-episode report: `artifacts/task_01_quality_audit/svla_so100_pickplace/video_alignment.json`.

| Check | Result |
|-------|--------|
| Videos missing / unreadable / zero-length | **0 / 0 / 0** |
| Container frames (top & wrist) | **19,631** each (= tabular rows) |
| Codec / resolution / FPS | AV01 / 640×480 / 30 |
| Material frame-count mismatches | **0** episodes |
| Material duration mismatches | **0** episodes |
| `frame_index/fps` vs timestamp | max \|mismatch\| ≈ **9e-7 s**; **0** episodes exceed 1-frame tolerance |
| Distinctions recorded | expected tabular count vs container count vs successfully decoded count (full wrist decode = 19,631) |

No structural or timing mismatches were detected by the implemented checks.

## 9. Cross-modal findings

- **Low-sharpness vs motion:** flagged frames have much higher mean state Δ-norm (**1.47** vs **0.07** unflagged) → blur heuristic correlates with fast motion (soft, expected).
- **Near-duplicates vs motion:** near-dups occur at **lower** mean motion than non-dups → often static/slow segments.
- Exact dup while state moves: see corrected duplicate categories; exact+near-lossless frozen-while-moving is a soft review signal (not automatic corruption).
- **Blur runs:** 319 runs; 175 sustained (≥5 frames); 41 isolated singles.
- **Alignment failures:** none affecting either modality. No structural or timing mismatches were detected by the implemented checks.

## 10. Observed quality issues

Separating **defects** from **screening flags**:

1. **Soft review — adjacent exact duplicate wrist frames (16.48% of within-episode pairs)**, plus near-lossless 5.06%; a subset coincides with above-median state motion (soft freeze/stutter review).  
2. **Soft review — overexposed wrist frames (1.36%).**  
3. **Diagnostic — absolute blur threshold miscalibration** (78.5% flag rate; median sharpness 9.6 ≪ 50). Recalibrate with distribution-derived cut (e.g. p05) before using as a filter.  
4. **Soft review — within-episode joint discontinuity candidates (~7.4–7.7%)**; largest jumps ~5–6° are plausible teleop jerks after temporal context review.  
5. **ALOHA-only historical note:** uniform episode length 500; gripper “outliers” reclassified as normal bimodal grasp behavior.

No hard evidence of NaN corruption, missing videos, or timeline tears on the paired dataset.

## 11. Audited risks with zero observed incidence

| Risk | Incidence (paired dataset) |
|------|----------------------------|
| State/action NaN or Inf | 0 |
| Timestamp gaps / duplicates / non-monotonic | 0 |
| Frame-index skips / length mismatches | 0 |
| Near-stuck channels | 0 |
| Robust statistical outliers (joints & grippers) | 0 |
| Underexposure / low-entropy wrist flags | 0 |
| Missing / undecodable wrist frames | 0 |
| Material video–tabular mismatches | 0 |
| Timing tolerance exceeded | 0 |

## 12. Recommended pre-training filters

| Recommendation | Category |
|----------------|----------|
| Reject episodes with NaN/Inf, non-monotonic time, or video open failure | **Hard rejection** |
| Shared timestep validity mask for overexposed / exact-dup-while-moving / extreme joint Δ | **Soft review flag** → Task 3 mask |
| Recalibrate blur threshold from wrist sharpness distribution before filtering | **Diagnostic only** |
| Keep expected grasp self-occlusion and normal gripper mode switches | **Do not filter** |
| Short-gap hold-last-frame only if a future dataset shows brief decode holes | **Soft**, justified only then |
| Episode-level reject if sustained unusable occlusion (not observed here) | **Hard** when criteria met |
| **Never delete a wrist frame alone** — drop/replace the aligned `(state, action, images.*)` timestep | Alignment contract |

## 13. Teleoperation-versus-egocentric filtering differences

Teleop filters act on low-dimensional series (Δ, z-scores, length). Egocentric filters act on appearance proxies (sharpness, exposure, entropy, duplicates) that are scene-dependent and easily miscalibrated. Both share a timeline: independently deleting video frames breaks imitation pairs. Task 3 must use **episode rejection**, **shared masks**, or **paired replace**, never unpaired frame deletion.

## 14. Simplifying assumptions and limitations

- No hardware joint limits → no “physically invalid” labels.
- Laplacian blur cut of 50 is an exploratory heuristic; distribution shows it is too strict for this wrist camera.
- Low entropy is an occlusion **proxy**, not proof.
- Near-lossless MSE≤1 and near-duplicate 1<MSE≤25 are mutually exclusive screening bands; exact requires array equality.
- Duplicate comparisons never cross episode boundaries; non-consecutive frame indices are refused outside sampled diagnostic mode.
- Discontinuity abs_floor=1.0° + range_frac is scale-aware but still heuristic.
- LeRobot Python package not used; Hub+parquet+OpenCV fallback.
- ALOHA baseline artifacts preserved unchanged in content (path relocated only).

## 15. README-ready conclusion

Task 1 is complete for **both** modalities on `lerobot/svla_so100_pickplace` @ `728583b5…` (wrist+top, full 19,631-frame tabular + full wrist decode). No structural or timing mismatches were detected by the implemented checks. The original ALOHA audit is preserved as a teleop-only baseline; its gripper “outliers” are reclassified as normal bimodal grasp behavior. Paired-dataset issues are mostly soft review flags (exact/near-lossless/near duplicate bands, mild overexposure, motion-correlated low sharpness under a miscalibrated absolute blur cut)—not hard corruption. Future curation must use shared timestep masks so egocentric cleaning cannot desynchronize robot state and action.
