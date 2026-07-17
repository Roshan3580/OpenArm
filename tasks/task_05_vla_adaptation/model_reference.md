# OpenVLA model reference (verified primary sources)

Inspected without downloading the 7B weights.

## Sources inspected

| Source | Identifier |
|--------|------------|
| Official repository | https://github.com/openvla/openvla |
| Code commit inspected | `c8f03f48af69` (GitHub `main` tip at inspection time, 2025-03-23) |
| Hugging Face model | `openvla/openvla-7b` |
| HF model revision (sha) | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` |
| Model card | https://huggingface.co/openvla/openvla-7b |
| Fine-tune script | `vla-scripts/finetune.py` |
| Action tokenizer | `prismatic/vla/action_tokenizer.py` |

## Verified facts (official)

- **License:** MIT (model card). Llama-2 backbone also subject to the Llama Community License.
- **I/O:** language instruction + **one** camera image → robot action.
- **Image:** Prismatic `prism-dinosiglip-224px` family (224px); processor applies transforms.
- **Language prompt pattern:** `In: What action should the robot take to {<INSTRUCTION>}?\nOut:`
- **Action representation (pretrained):** **7-DoF end-effector deltas** `(x, y, z, roll, pitch, yaw, gripper)` after un-normalization with a dataset key.
- **Action dimensionality:** treated as **fixed 7** in the pretrained / HF predict path; fine-tuning uses the same ActionTokenizer discretization.
- **Discretization:** continuous actions clipped to `[-1, 1]`, digitized into **256 bins**, mapped onto the least-used LLM tokens (`ActionTokenizer`).
- **Training loss:** autoregressive **causal LM cross-entropy** on action tokens (`output.loss` in `finetune.py`); logged continuous L1 is diagnostic after detokenization.
- **Single-step vs chunks:** reference OpenVLA predicts **single actions** (not ACT-style chunks). FAST tokenizer / chunking is a later related line of work, not the baseline OpenVLA-7B contract.
- **Normalization:** per-dataset statistics saved during fine-tune (`save_dataset_statistics`); inference uses `unnorm_key`.
- **LoRA/QLoRA:** supported via PEFT in `finetune.py` (`use_lora`, optional 4-bit with caution).
- **Official LoRA defaults (script):** `lora_rank=32`, `lora_alpha=min(rank,16)`, `lora_dropout=0`, `target_modules="all-linear"`, `learning_rate=5e-4`, `batch_size=16`, `max_steps=200000`, `image_aug=True`.
- **Hardware (official docs):** LoRA `batch_size=16` ≈ **~72GB** GPU memory; reduce batch and raise grad accumulation otherwise.
- **Data format expectation:** RLDS / Open-X style loaders by default; custom PyTorch datasets possible with comments in `finetune.py`.

## Project-specific design decisions

- Map SO-100 **6-D joint absolute actions** into OpenVLA’s **7-D [-1,1] token slots** by **zero-padding one masked dimension** with an adapter-side `action_mask`.
- Primary visual stream: **wrist-only**.
- Primary curation: Task 3 **conservative** windows; strict as ablation.
- Do **not** download or fine-tune the 7B checkpoint in this take-home.
- Adapter emits **NHWC uint8 RGB 224** as processor input; official PrismaticProcessor tensors are out of scope for the smoke test.

## Assumptions

- Absolute joint targets are a valid imitation supervision signal for this LeRobot SO-100 corpus (actions ≈ joint targets).
- Official processor mean/std will be applied at real training time; export/smoke store uint8 RGB 224.
- A future true multi-view OpenVLA variant is out of scope.

## Features requiring reference-implementation changes

- Loss/collator changes to honor the padded-dimension `action_mask` (or a native 6-D joint head without a padded slot).
- Native dual-camera fusion.
- Native ACT-style action chunking inside OpenVLA-7B (would need OFT/FAST-style extensions).
- Proprio state tokens (not in baseline OpenVLA forward path).
