# Task 5 : OpenVLA Adaptation (Option A)

**Dataset:** `lerobot/svla_so100_pickplace` @ `728583b5eaf9e739a7f119e2def466fa1d552402`  
**Model family:** OpenVLA (`openvla/openvla-7b`)  
**Status:** Teleoperation mapping **complete**; egocentric preprocessing + viewpoint-shift analysis **complete**.

## Honesty constraints

- **No OpenVLA checkpoint was downloaded.**
- **No VLA fine-tuning was performed.**
- **No policy rollout was performed.**
- The adapter and batch construction were tested on **real** wrist frames.
- Hyperparameters in `configs/openvla_lora.yaml` are a **proposed** starting configuration.
- Rollout performance must use Task 4 protocols after real training.
- SO-100 action-space adaptation is **project-specific**.

## 1. Scope

Design and implement a reproducible path from Task 3 curated windows to an OpenVLA-compatible training representation, including LoRA config, action-space handling, wrist preprocessing, alignment, and egocentric failure analysis : without training the 7B model.

## 2. Model choice

OpenVLA-7B with parameter-efficient **LoRA** fine-tuning (`vla-scripts/finetune.py`). See `model_reference.md`.

## 3. Verified OpenVLA interface

- Inputs: one RGB image (224) + language instruction.
- Outputs: 7-DoF actions, 256-bin tokenized, trained with causal LM loss.
- Pretrained semantics: end-effector deltas + gripper (Open-X).
- License: MIT (model card); Llama Community License for LLM.

## 4. Dataset mapping and count units

Task 3 conservative windows → episode-grouped 80/10/10 split → train-only q01/q99 stats → JSONL/parquet metadata under `data/vla/` (gitignored) referencing source frames by identity.

**These counts use different units and are not contradictory:**

| Source | Unit | Count |
|--------|------|------:|
| Task 3 conservative | **horizon-16 training windows** | 18,881 |
| Task 3 strict | **horizon-16 training windows** | 18,386 |
| Task 5 conservative | **single-timestep OpenVLA examples** | 19,631 |
| Task 5 strict | **single-timestep examples after soft exclusions** | 19,391 |
| Task 5 train / val / test | single-timestep (conservative) | 15,594 / 1,979 / 2,058 |

Stock OpenVLA predicts **single-step** actions, so Task 5 exports one example per valid curated timestep rather than ACT-style horizon-16 chunks.

## 5. Teleoperation action representation

Primary: **absolute 6-D joint actions** (high agreement with state targets in this corpus).  
Encoded to OpenVLA’s 7 slots via robust normalization to [-1, 1] + **masked zero pad** on dim 6.

**Masked-padding integration (explicit):**

- Adapter-side masking is tested (pad weight = 0 in loss metadata).
- Actual training requires the reference fine-tuning loss/collator to **consume** `action_mask`.
- Stock OpenVLA must **not** be assumed to ignore the padded token automatically.
- A production fine-tune must either integrate the mask into the loss **or** implement a native 6-D tokenizer/head.
- No real training validated this modification.

Details: `teleop_finetuning.md`.

## 6. Egocentric preprocessing boundary

Pipeline steps implemented here:

1. Dataset adapter decodes **BGR** video frames (OpenCV).
2. Adapter converts them to **RGB**.
3. Adapter resizes them to **224×224** (bilinear).
4. Adapter emits **processor-ready NHWC uint8** images.
5. The verified official OpenVLA / Prismatic processor would then produce **normalized CHW model tensors**.
6. The model-free smoke test **stops before** loading that processor or any checkpoint.
7. Therefore the smoke test validates **alignment, encoding, masking, and batching** : not end-to-end model compatibility.

Safe augs only (brightness/contrast/mild blur/center-preserving scale). No flips / large rotations / time reversal. See `egocentric_adaptation.md` and `sample_grid.png`.

## 7. Temporal alignment

Default: `(wrist_image_t, language) → action_t` (offset 0 frames / 0 ms @ 30 FPS). Configurable offset; same-episode only; invalid/missing future targets dropped; hard-invalid frames excluded via Task 3 windows.

## 8. LoRA strategy

PEFT LoRA on `all-linear`, rank 32, alpha 16, dropout 0, bf16, optional 4-bit discouraged. Effective batch 16 via batch 8 × accum 2. LR 5e-4 (official). Max steps 20k (project).

## 9. Key hyperparameters

See `configs/openvla_lora.yaml` (official vs project tags). Highest impact: LR, LoRA rank, effective batch, action normalization, image aug, action offset, training duration.

## 10. Model-free smoke-test results

`artifacts/task_05_vla_adaptation/batch_smoke_test.json`:

- Diverse batch across train/val/test with early/middle/late frames.
- Images: NHWC uint8 RGB processor input (not final model tensors).
- Actions encoded `[B,7]`; mask dim-6 all zero.
- Round-trip MAE ~0 (within float tolerance).
- Normalization stats: train-only q01/q99 (frozen for val/test rows in the batch).
- `model_checkpoint_downloaded: false`.
- `stock_loss_mask_integration_verified: false`.

## 11. Third-person → egocentric failures

Camera FOV/scale, gripper dominance, self-occlusion, blur, background motion, lighting, partial observability, spatial-prior mismatch, action–viewpoint coupling, embodiment mismatch, forgetting : with diagnostics and Task 4 slices in `egocentric_adaptation.md`.

## 12. Evaluation plan

After real training: Task 4 100-rollout matrix; compare conservative/strict and wrist/top ablations; never use the wrist completion-proxy as primary success.

## 13. Simplifying assumptions

- Single-image OpenVLA; wrist-only baseline.
- Padded 7th dim is an adapter compatibility choice, not stock-compatible without loss changes.
- Absolute joints as imitation targets.

## 14. Limitations

- No trained policy; no sim/real rollouts.
- Padding requires training-loop changes vs a native 6-D head.
- Composite wrist+top is diagnostic only.

## 15. What requires real training / hardware

Load pinned checkpoint; attach LoRA; integrate action mask into loss; overfit/sanity; full fine-tune; Task 4 rollouts; optional real-robot diagnosis ladder from Task 4.

## 16. README-ready summary

Task 5 defines an OpenVLA LoRA adaptation path for SO-100 teleoperation and wrist egocentric data: verified interface pins, Task 3→OpenVLA export with train-only normalization, 6-D→7-D masked action encoding, wrist preprocessing, alignment contract, and egocentric shift analysis. No 7B weights were downloaded and no fine-tuning was run; a model-free real-frame batch smoke test passed on processor-ready uint8 batches.

## Commands

```bash
python scripts/export_openvla_dataset.py
python scripts/validate_openvla_dataset.py
python scripts/smoke_test_openvla_batch.py
pytest tests/test_vla_adapter.py -q
```
