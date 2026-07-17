# README notes (verified Task 1 + Task 3)

## Primary paired dataset

- `lerobot/svla_so100_pickplace` @ `728583b5eaf9e739a7f119e2def466fa1d552402`
- Cameras: `observation.images.wrist` (verified egocentric), `observation.images.top` (verified external)
- Full tabular audit: 50 episodes, 19,631 frames @ 30 FPS
- Full wrist decode: 19,631 frames; no structural or timing mismatches detected by implemented checks

## Corrected duplicate terminology (Task 1)

Within-episode adjacent pairs only (denominator **19,581** = 19,631 frames − 50 episodes). Cross-episode boundaries are never compared.

| Category | Rule | Rate (within-episode adjacent pairs) |
|----------|------|------------------------|
| Exact | array equality | 16.48% (3,227) |
| Near-lossless | 0 < MSE ≤ 1 | 5.06% (991) |
| Near | 1 < MSE ≤ 25 | 37.82% (7,405) |

## Task 3 curation

- Manifest-backed view: `data/curated/svla_so100_pickplace/` (ignored by Git)
- Cleaning steps: hard validation + Savitzky–Golay joint smoothing (gripper excluded) + aligned visual flags + training windows
- Real hard rejections: 0 episodes / 0 timesteps
- Windows: conservative 18,881; strict 18,386 (horizon 16)

## Task 2 labeling design

- Schemas + agreement protocol under `tasks/task_02_labeling_design/`
- MVP tool: Label Studio (XML is illustrative; not server-tested here)
- Validator: `python scripts/validate_annotations.py`
- No real GT labels; sample is `illustrative_not_ground_truth`

## Task 4 policy evaluation

- 100-rollout ACT protocol: `tasks/task_04_policy_evaluation/rollout_protocol.yaml`
- Validator: `python scripts/validate_rollout_protocol.py`
- Wrist completion-proxy detector: `python scripts/evaluate_success_detector.py` (not true success labels)
- Models/caches under `data/models/` and `data/evaluation_cache/` (Git-ignored)

## Task 5 OpenVLA adaptation

- Interface pins + LoRA config: `tasks/task_05_vla_adaptation/`, `configs/openvla_lora.yaml`
- Export / validate / smoke: `scripts/export_openvla_dataset.py`, `validate_openvla_dataset.py`, `smoke_test_openvla_batch.py`
- Local exports under `data/vla/` (Git-ignored); **no 7B weights downloaded**
- Count units: Task 3 horizon-16 windows vs Task 5 single-timestep examples
- Smoke batch emits NHWC uint8 processor input; masked pad needs loss integration for real training

## Commands

```bash
uv pip install -e ".[dev]"
pytest -q
python scripts/curate_dataset.py --force
python scripts/validate_annotations.py --require-exhaustive
python scripts/validate_rollout_protocol.py
python scripts/evaluate_success_detector.py
python scripts/export_openvla_dataset.py
python scripts/validate_openvla_dataset.py
python scripts/smoke_test_openvla_batch.py
```
