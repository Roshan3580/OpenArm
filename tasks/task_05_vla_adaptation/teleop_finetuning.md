# Teleoperation fine-tuning plan (OpenVLA + SO-100)

## How a pretrained VLA is fine-tuned on OpenArm teleop data

1. Export Task 3 **conservative** windows to OpenVLA-style examples (image ref + language + normalized 7-D action with mask).
2. Load `openvla/openvla-7b` with `trust_remote_code=True` (not done here).
3. Wrap with PEFT LoRA (`all-linear`, rank 32).
4. Train with causal LM loss on action tokens using `vla-scripts/finetune.py` or an equivalent custom Dataset that yields the same tensors.
5. Save dataset statistics for action un-normalization at inference.
6. Evaluate with Task 4 rollouts (not training loss alone).

## Expected data format

OpenVLA’s reference stack prefers RLDS/Open-X. This project exports an intermediate JSONL/parquet with:

- Immutable source identity `(revision, episode, frame, timestamp)`
- Wrist image **reference** (no video duplication)
- Instruction string
- `action_raw` (6), `action_encoded` (7 in [-1,1]), `action_mask` (pad=0)
- Curation policy + split

At true training time, a thin loader would decode the referenced wrist frame, run `PrismaticProcessor`, tokenize the prompt, and discretize `action_encoded` with `ActionTokenizer`.

## Supervision signal

Verified OpenVLA objective: **next-token cross-entropy** over discrete action tokens conditioned on image patch tokens + language tokens. Gradients flow through LoRA-adapted linear modules; base weights remain frozen except adapted subspaces. Continuous L1 after detokenization is a log metric, not the optimized loss.

Do not conflate with ACT chunk regression or pure language modeling without action tokens.

## Count units (Task 3 windows vs Task 5 examples)

Task 3’s conservative/strict figures (18,881 / 18,386) count **horizon-16 training windows**.
Task 5’s conservative/strict figures (19,631 / 19,391) count **single-timestep OpenVLA examples**, because verified stock OpenVLA predicts single-step actions. These are different units, not a contradiction.

## Action representation decision

| Topic | Decision |
|-------|----------|
| Source | 6-D SO-100 joint commands |
| Official pretrained space | 7-D EEF delta + gripper |
| Compatibility | Normalize 6-D with train-only q01/q99 → [-1,1]; **pad 7th dim with 0**; **mask pad out of loss** |
| Primary mode | **Absolute** joints (actions ≈ targets; mean \|a−s\| small) |
| Alternate | Delta joints within episode; gripper absolute |
| Why not silent reshape | Would invent fake EEF semantics |

**Masked padding is not fully stock-compatible.** Adapter-side masks are tested. Production training must wire `action_mask` into the reference loss/collator (or replace the 7-D tokenizer/head with a native 6-D path). Stock OpenVLA will not ignore the padded token automatically. No real training validated this change.

## Operational plan

1. Validate dataset + revision pin.  
2. Select conservative curated windows.  
3. Episode-grouped 80/10/10 split.  
4. Compute train-only normalization; freeze for val/test.  
5. Export OpenVLA examples (`scripts/export_openvla_dataset.py`).  
6. Run model-free smoke test (`scripts/smoke_test_openvla_batch.py`).  
7. Load pinned pretrained checkpoint (**not performed**).  
8. Attach LoRA adapters.  
9. Tiny overfit on 1–2 episodes.  
10. Short validation training.  
11. Planned full fine-tune (`configs/openvla_lora.yaml`).  
12. Select checkpoint by val action-token accuracy / detokenized L1.  
13. Task 4 rollouts.  
14. Ablate strict/conservative and wrist/top.

## Failure gates

| Gate | Meaning |
|------|---------|
| Loss flat on tiny overfit | Bug in labels/tokenizer/mask |
| Decoded action range wrong | Norm stats / inverse broken |
| Gripper collapses | Gripper scaling/mask bug |
| Val improves, rollouts fail | Distribution shift / offset / camera preprocess |
| Ignores language | Prompt formatting / overfitting to vision |
| Ignores wrist frames | Processor / pixel pipeline bug |
| Split leakage | Episode IDs overlap |
| Camera preprocess mismatch | Train vs infer resize/color |
| Wrong action timestamp offset | Misaligned imitation |

## Key hyperparameters (proposed)

See `configs/openvla_lora.yaml`. Official: LR `5e-4`, LoRA r32/α16, image_aug on, batch 16 @ ~72GB. Project: batch 8 × accum 2, max_steps 20k, wrist-safe aug, masked 7th dim.

**No claim these were trained or tuned.**
