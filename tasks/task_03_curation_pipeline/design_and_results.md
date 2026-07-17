# Task 3 : Data Curation Pipeline: Design and Results

## 1. Objective

Build a conservative, evidence-driven curation pipeline for both teleoperation and egocentric modalities on the primary paired dataset, producing a reproducible **manifest-backed curated training view** that preserves temporal alignment. Do not force arbitrary removals on a structurally clean corpus.

## 2. Source dataset and pinned revision

| Field | Value |
|-------|-------|
| Repo | `lerobot/svla_so100_pickplace` |
| Revision | `728583b5eaf9e739a7f119e2def466fa1d552402` |
| Modalities | `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `observation.images.top`, `observation.images.wrist` |
| ALOHA baseline | Preserved under Task 1 artifacts only (not curated here) |

## 3. Simplifying assumptions

1. Source Hub videos remain immutable; curation references them rather than copying ~470 MB of media.
2. No hardware joint limits → discontinuity flags are statistical, not physical invalidity.
3. Absolute Laplacian threshold 50 is never a hard filter (Task 1 over-flagged 78.5%).
4. Low entropy is not confirmed occlusion.
5. Gripper channels are excluded from smoothing and from joint-outlier style filtering.
6. Fabricating interpolated wrist frames is disallowed in the default pipeline.
7. Task 1 found no hard structural corruption; zero hard episode rejections on the real run are expected and acceptable.

## 4. Pipeline architecture

```text
Hub source (pinned revision)
        │
        ▼
Hard episode validation ──reject──► episode_decisions.csv
        │ accept
        ▼
Within-episode joint smoothing (gripper excluded)  → state_smoothed
        │
        ▼
Aligned visual quality flags (wrist decode stream)
        │
        ▼
Per-timestep table (flags + hard_valid / soft_exclude)
        │
        ├── conservative windows (hard failures only)
        └── strict windows (hard + sustained soft exclusions)
        │
        ▼
data/curated/svla_so100_pickplace/  (Git-ignored local view)
artifacts/task_03_curation_pipeline/ (small trackable summaries)
```

This is a curated training view over an immutable source, **not** a republished standalone LeRobot corpus.

## 5. Teleoperation filtering

### Hard episode rejection (reason codes)

NaN/Inf; non-monotonic or duplicate timestamps; frame-index gaps; state/action/length mismatch; missing/unreadable wrist video; material video/tabular mismatch; episode shorter than `min_episode_frames=64`.

**Justification for 64 frames:** 30 FPS × ~2 s ≈ 60 frames for a short pick segment; also ≥ 4× `horizon=16` so multiple windows fit. Task 1 minimum episode length is 326, so this threshold does **not** force deletions.

### Diagnostic discontinuities

Element-level Task 1-style flags (~7.4–7.7% historically) remain **diagnostic**. Ordinary discontinuities do not exclude timesteps. Severe discontinuities (`mad_k=20`, `abs_floor=8`) can exclude windows under the **strict** policy only.

## 6. Teleoperation smoothing

- Method: Savitzky–Golay (`window_length=5`, `polyorder=2`), within episode only.
- Gripper excluded; original state/action preserved; smoothed written as `state_smoothed`.
- Safety: skip applying a dimension if RMSE/std > 0.35.
- Real-run: applied on 250 joint-dimension episode rows; mean RMSE ≈ 0.039; mean first-difference variance reduction ≈ 3.4%; 50 gripper rows unchanged.

## 7. Egocentric quality detection

Corrected duplicate bands (mutually exclusive):

| Category | Definition | Real count (timestep flags) | Rate (of 19,581 within-episode adjacent pairs) |
|----------|------------|-----------------------------|----------------|
| Exact | array equality | 3,227 | 16.48% |
| Near-lossless | 0 < MSE ≤ 1 | 991 | 5.06% |
| Near | 1 < MSE ≤ 25 | 7,405 | 37.82% |

Comparisons are within-episode only (sorted by `frame_index`); cross-episode boundaries are never compared.

Additional flags: distribution-relative low sharpness (p05 ≈ 3.75 from Task 1), over/underexposure, low-entropy heuristic, frozen-while-moving (exact/near-lossless while motion ≥ median), sustained overexposure runs (≥5 frames).

**Policy:** missing/undecodable → hard invalid; isolated low sharpness → diagnostic; stationary duplicates → keep; expected self-occlusion not auto-labeled unusable; no visual frame interpolation.

## 8. Alignment-preserving curated view

`timesteps.parquet` retains original episode/frame/timestamp/global index, state, `state_smoothed`, action, camera timestamp refs, and all flags.

`training_windows.parquet` stores consecutive within-episode horizons of length 16. Invalid gaps are never bridged. Videos are referenced via Hub paths + coordinates, not copied.

## 9. Conservative versus strict policies

| Policy | Excludes |
|--------|----------|
| **conservative (default reported)** | Hard-invalid timesteps only |
| **strict** | Hard-invalid ∪ sustained overexposure ∪ sustained frozen-while-moving ∪ severe discontinuity |

Strict windows are a subset of conservative windows (validated).

## 10. Real-dataset results

| Quantity | Before | After |
|----------|--------|-------|
| Episodes | 50 | 50 accepted, 0 rejected |
| Timesteps hard-valid | 19,631 | 19,631 (0 hard-invalid) |
| Conservative windows | : | **18,881** |
| Strict windows | : | **18,386** |

Honest conclusion: the real corpus needed **no hard episode/timestep rejection**. Curation still applied smoothing + aligned visual flagging + dual-policy window construction. Strict policy removed 495 overlapping windows intersecting sustained soft-exclusion timesteps (240 timesteps with `soft_exclude_strict`, primarily sustained overexposure).

## 11. Synthetic-corruption validation

`tests/test_curation.py::test_synthetic_corruption_integration` injects short episode, NaN state, timestamp tear, missing wrist frame, frozen-while-moving, stationary duplicates, and gripper transitions. Verified: appropriate rejects; stationary dups retained; gripper unsmoothed; windows never include hard-invalid frames; ≥2 distinct rules activate. Synthetic results are **not** mixed into real artifacts.

## 12. Filtering-choice justification

| Choice | Why |
|--------|-----|
| Manifest view, not video rewrite | Avoid duplicating hundreds of MB; keep source immutable |
| Shared timestep / windows | Arbitrary image deletion breaks imitation pairs |
| Gripper excluded from smoothing | Preserve open/close semantics |
| No absolute blur=50 hard filter | Task 1 showed 78.5% false-positive rate |
| Conservative default | Clean sim/real teleop should not be shrunk for show |

## 13. Trade-offs and limitations

- Soft sharpness uses a global p05; scene-dependent blur remains imperfect.
- Frozen-while-moving uses distributional motion thresholds; teleop pauses vs freezes can confuse.
- Strict exclusions are sustained-run based; isolated overexposure stays review-only.
- Curated view requires Hub/cache access to resolve pixels at train time.

## 14. What would change with hardware access

Calibrate joint limits, exposure, and blur thresholds on the real OpenArm wrist camera; add contact/force channels; consider learned occlusion detectors with labels; stress-test freeze detection under Wi-Fi camera dropouts.

## 15. README-ready summary

Task 3 curates `lerobot/svla_so100_pickplace` @ `728583b5…` into a Git-ignored manifest-backed view with joint smoothing (gripper excluded), aligned wrist quality flags (corrected exact / near-lossless / near duplicates), and conservative (18,881) vs strict (18,386) training windows of horizon 16. No real episodes were hard-rejected:consistent with Task 1’s clean structural audit:while synthetic tests prove the filters fire when corruption is present.
