# Progress tracker

| # | Item | Status |
|---|------|--------|
| 1 | Repository scaffolding | Done |
| 2 | Task 1 teleoperation | Done |
| 3 | Task 1 egocentric | Done |
| 4 | Task 2 teleoperation | Done (design + schemas + validator; no real GT labels) |
| 5 | Task 2 egocentric | Done (design + schemas + validator; no real GT labels) |
| 6 | Task 3 teleoperation | Done |
| 7 | Task 3 egocentric | Done |
| 8 | Task 4 teleoperation | Done (100-rollout ACT protocol validated; no rollouts executed) |
| 9 | Task 4 egocentric | Done (visual protocol + failed wrist completion-proxy prototype) |
| 10 | Task 5 teleoperation | Done (OpenVLA adapter, action mapping, LoRA config, smoke test) |
| 11 | Task 5 egocentric | Done (wrist preprocess, alignment, viewpoint-shift analysis) |
| 12 | Final README polish | Done (submission README, checklist, Task 5 clarifications) |

## Notes

- Primary paired dataset: `lerobot/svla_so100_pickplace` @ `728583b5eaf9e739a7f119e2def466fa1d552402`.
- ALOHA baseline preserved under `artifacts/task_01_quality_audit/aloha_sim_insertion_human/`.
- Task 1 duplicate accounting: within-episode adjacent pairs only (denominator **19,581**). Counts: exact 3,227 / near-lossless 991 / near 7,405.
- Task 2: no real ground-truth labels; sample annotation is synthetic.
- Task 3: hard rejections 0/0; conservative windows 18,881; strict 18,386 (horizon-16 units).
- Task 4: no policy trained; no fabricated rollouts; wrist proxy temporal false-early-trigger = 1.00.
- Task 5: no OpenVLA-7B download/fine-tune; single-timestep examples (19,631/19,391) vs Task 3 windows; NHWC uint8 is processor input, not final model tensors; masked 7th action dim requires loss integration for production training.
- No hardware evaluation.
