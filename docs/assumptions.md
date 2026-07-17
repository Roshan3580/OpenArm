# Global assumptions

1. **No hardware.** Public Hugging Face / LeRobot datasets and simulation only.
2. **Two Task 1 corpora:** ALOHA sim insertion = teleop baseline; `lerobot/svla_so100_pickplace` = primary paired teleop+egocentric dataset for Tasks 2–5.
3. **Egocentric requires `verified_egocentric`.** Feature-name tokens (e.g. `wrist`) plus visual confirmation; appearance alone is insufficient when placement is uncertain.
4. **Statistical ≠ physical invalidity** without documented joint limits.
5. **Gripper channels are special:** open/closed or multimodal distributions are normal; do not treat grasp transitions as corruption.
6. **Imports are side-effect free** (no download/audit on import).
7. **Raw data stays out of Git** (`data/`, caches, parquet, mp4 ignored). Assignment PDF is gitignored locally.
8. **Task 1 measures; Task 3 curates.**
9. **Alignment contract:** shared episode/`frame_index` (or timestamp) coupling; filters use shared masks / paired drop/replace.
10. **Access fallback:** Hub metadata + parquet + OpenCV when `lerobot` is unavailable.
11. **Thresholds live in `configs/audit.yaml` / `configs/curation.yaml`** and are typed as hard / screening / exploratory in reports.
12. **Task 3 default policy is conservative:** hard failures only; strict policy additionally drops sustained soft visual/motion failures from training windows.
13. **Curated output is a manifest-backed view** under `data/curated/` (Git-ignored), not a duplicated video corpus.
14. **Task 2 labels use immutable source coordinates** `(dataset_revision, episode_index, frame_index, timestamp)`; curated policies are metadata filters, not a new annotation index space.
15. **Attention-proxy ROIs are not eye tracking.** No real ground-truth labels were produced in this take-home; `sample_annotation.json` is synthetic.
16. **Duplicate comparisons are within-episode only** (sorted by `frame_index`); cross-episode boundaries are never compared.
17. **Task 4 does not claim a trained ACT/Diffusion checkpoint or executed rollouts.** The 100-rollout matrix is an evaluation design.
18. **The wrist “success detector” is a terminal-completion proxy** on position-derived labels, not a verified task-success detector. Primary rollout success must come from simulator state, trusted external evaluation, or adjudicated humans.
19. **Task 5 does not download or fine-tune OpenVLA-7B.** The adapter maps SO-100 6-D joint actions into OpenVLA’s 7-D token slots with a masked pad; hyperparameters are proposed, not trained.
20. **OpenVLA baseline is single-image.** Wrist-only is the supported egocentric fine-tune view; true multi-view fusion needs model changes.
21. **Task 3 window counts ≠ Task 5 example counts.** Horizon-16 windows (18,881/18,386) vs single-timestep OpenVLA examples (19,631/19,391) are different units.
22. **Smoke-test images are processor-ready NHWC uint8 RGB**, not the final normalized CHW OpenVLA tensors.
23. **Masked padding is adapter-side only until the reference loss consumes `action_mask`.** Stock OpenVLA will not ignore the padded token automatically.
