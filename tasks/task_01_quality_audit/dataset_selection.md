# Task 1 : Dataset selection

## Candidates inspected (3)

| Repository ID | Task | Episodes | Frames | State/Action dim | Camera keys | FPS | Approx size | License | Selected / rejected | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| `lerobot/svla_so100_pickplace` | Pick up the cube and place it in the box | 50 | 19,631 | 6 / 6 | `observation.images.top`, `observation.images.wrist` | 30 | ~470 MB | Apache-2.0 | **Selected** | Explicit wrist + external cameras; synchronized state/action/time indices; pick-and-place; manageable size; public |
| `lerobot/aloha_static_coffee` | ALOHA static coffee (manipulation) | 50 | 55,000 | 14 / 14 | `cam_high`, `cam_left_wrist`, `cam_right_wrist`, `cam_low` | 50 | ~1.57 GB | MIT | Rejected | Strong wrist coverage but exceeds ~1 GB size preference; heavier multi-camera decode |
| `lerobot/imperialcollege_sawyer_wrist_cam` | Sawyer wrist-cam demos | 170 | 7,148 | present | `observation.images.image`, `observation.images.wrist_image` | 5 | ~3.8 MB | Apache-2.0 | Rejected | Has wrist key, but 64×64 @ 5 FPS is too weak for egocentric quality audit; not pick-and-place focused |

## Selection decision

**Primary paired dataset:** `lerobot/svla_so100_pickplace`  
**Revision:** `728583b5eaf9e739a7f119e2def466fa1d552402`

### Wrist verification evidence

1. **Feature name:** `observation.images.wrist` → classified `verified_egocentric` (placement token `wrist`).
2. **Paired external:** `observation.images.top` → `verified_external`.
3. **Visual inspection** (episodes 0, 24, 49; positions 0.1 / 0.5 / 0.9): wrist frames show gripper fingers (orange/white) and close-up of the red cube entering the grasp; top frames show a third-person workspace overview. Appearance supports, but does not solely determine, the egocentric label.
4. **Sync:** tabular `timestamp` / `frame_index` / `episode_index` present; video segment timestamps in `meta/episodes` align with tabular lengths; both camera containers report 19,631 frames matching the tabular corpus.

### Preserved baseline

`lerobot/aloha_sim_insertion_human` remains the original teleoperation baseline under `artifacts/task_01_quality_audit/aloha_sim_insertion_human/` (no wrist stream; egocentric blocked on that corpus).
