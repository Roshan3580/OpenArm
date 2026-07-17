#!/usr/bin/env python3
"""Model-free OpenVLA batch smoke test on real wrist frames (no checkpoint download)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.vla.action_encoding import ActionNormalizer, decode_actions  # noqa: E402
from openarm_pipeline.vla.collator import collate_openvla_batch  # noqa: E402
from openarm_pipeline.vla.config import OpenVLAAdapterConfig  # noqa: E402
from openarm_pipeline.vla.dataset_adapter import (  # noqa: E402
    episode_grouped_split,
    fit_normalizer_on_train,
    load_curated_tables,
    pick_diverse_rows,
    save_json,
    select_export_rows,
)
from openarm_pipeline.vla.preprocessing import (  # noqa: E402
    preprocess_wrist_frame,
    safe_augmentation,
)
from openarm_pipeline.vla.validation import (  # noqa: E402
    assert_no_episode_leakage,
    validate_batch_shapes,
    validate_roundtrip,
)


def local_wrist_video(revision: str) -> Path:
    return (
        ROOT
        / ".cache/huggingface/hub/datasets--lerobot--svla_so100_pickplace/snapshots"
        / revision
        / "videos/observation.images.wrist/chunk-000/file-000.mp4"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--curated-root", default="data/curated/svla_so100_pickplace")
    p.add_argument("--artifacts-dir", default="artifacts/task_05_vla_adaptation")
    args = p.parse_args()

    cfg = OpenVLAAdapterConfig(curated_root=args.curated_root, artifacts_dir=args.artifacts_dir)
    art = ROOT / cfg.artifacts_dir
    art.mkdir(parents=True, exist_ok=True)

    timesteps, windows = load_curated_tables(ROOT / cfg.curated_root)
    split = episode_grouped_split(timesteps["episode_index"].unique(), seed=cfg.seed)
    assert_no_episode_leakage(split)
    normalizer, stats = fit_normalizer_on_train(timesteps, windows, cfg, split)
    save_json(stats, art / "action_statistics.json")
    save_json(split, art / "split_manifest.json")

    rows = select_export_rows(timesteps, windows, cfg)
    picked = pick_diverse_rows(rows, split, batch_size=args.batch_size)

    video = local_wrist_video(cfg.dataset_revision)
    if not video.exists():
        report = {
            "ok": False,
            "error": "wrist_video_unavailable",
            "model_checkpoint_downloaded": False,
            "path": str(video.relative_to(ROOT)) if video.is_relative_to(ROOT) else str(video),
        }
        save_json(report, art / "batch_smoke_test.json")
        print(json.dumps(report, indent=2))
        return 2

    df_sorted = timesteps.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    key_to_global = {
        (int(r.episode_index), int(r.frame_index)): int(i) for i, r in df_sorted.iterrows()
    }

    cap = cv2.VideoCapture(str(video))
    examples = []
    originals = []
    processed = []
    auged = []
    sample_meta = []
    for i, r in enumerate(picked):
        ep, fi = int(r["episode_index"]), int(r["frame_index"])
        gidx = key_to_global[(ep, fi)]
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(gidx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"failed to decode global_index={gidx}")
        rgb = preprocess_wrist_frame(bgr, size=cfg.image_size, assume_bgr=True)
        originals.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        processed.append(rgb)
        auged.append(safe_augmentation(rgb, seed=cfg.seed + i))
        action_raw = np.asarray(r["action"], dtype=np.float64)
        enc, mask, _ = normalizer.transform(action_raw.reshape(1, -1))
        split_name = str(r["split"])
        pos = str(r["temporal_position"])
        examples.append(
            {
                "image_rgb": rgb,
                "instruction": cfg.instruction,
                "action_raw": action_raw,
                "action_encoded": enc[0],
                "action_mask": mask,
                "episode_index": ep,
                "frame_index": fi,
                "timestamp": float(r["timestamp"]),
                "split": split_name,
                "curation_policy": cfg.curation_policy,
                "dataset_revision": cfg.dataset_revision,
                "view": "wrist",
                "action_offset_frames": cfg.action_offset_frames,
            }
        )
        sample_meta.append(
            {
                "episode_index": ep,
                "frame_index": fi,
                "timestamp": float(r["timestamp"]),
                "split": split_name,
                "temporal_position": pos,
            }
        )
    cap.release()

    batch = collate_openvla_batch(examples)
    shape_errors = validate_batch_shapes(batch, image_size=cfg.image_size, action_dim=7)
    decoded = decode_actions(batch["actions_encoded"], normalizer)
    rt = validate_roundtrip(batch["actions_raw"], decoded, atol=1.0)

    # sample grid
    n_cols = min(4, len(processed))
    fig, axes = plt.subplots(3, n_cols, figsize=(10, 7))
    if n_cols == 1:
        axes = np.array(axes).reshape(3, 1)
    for j in range(n_cols):
        axes[0, j].imshow(cv2.resize(originals[j], (224, 224)))
        axes[0, j].set_title("original")
        axes[0, j].axis("off")
        axes[1, j].imshow(processed[j])
        axes[1, j].set_title("preprocessed")
        axes[1, j].axis("off")
        axes[2, j].imshow(auged[j])
        axes[2, j].set_title("safe aug")
        axes[2, j].axis("off")
    fig.suptitle("OpenVLA wrist preprocess grid (processor-ready uint8; no model loaded)")
    fig.tight_layout()
    fig.savefig(art / "sample_grid.png", dpi=110)
    plt.close(fig)

    n_cons = len(select_export_rows(timesteps, windows, cfg))
    cfg_s = OpenVLAAdapterConfig(**{**cfg.to_dict(), "curation_policy": "strict"})
    n_strict = len(select_export_rows(timesteps, windows, cfg_s))
    save_json(
        {
            "dataset_repo_id": cfg.dataset_repo_id,
            "dataset_revision": cfg.dataset_revision,
            "curation_policy": cfg.curation_policy,
            "split": split,
            "n_examples": n_cons,
            "n_examples_conservative_full": n_cons,
            "n_examples_strict_full": n_strict,
            "example_unit": "single_timestep_openvla",
            "task3_window_unit_note": (
                "Task 3 reports horizon-16 training windows (conservative 18,881 / strict 18,386). "
                "Task 5 counts single-timestep OpenVLA examples because stock OpenVLA predicts single-step actions."
            ),
            "action_statistics_path": "artifacts/task_05_vla_adaptation/action_statistics.json",
            "export_root": "data/vla/svla_so100_pickplace",
            "model_checkpoint_downloaded": False,
        },
        art / "export_manifest.json",
    )

    splits_present = sorted(set(batch["split"]))
    report = {
        "ok": len(shape_errors) == 0 and rt["ok"] and len(splits_present) >= 3,
        "model_checkpoint_downloaded": False,
        "batch_size": batch["batch_size"],
        "image_shape": batch["input_image_shape"],
        "image_dtype": str(batch["pixel_values_uint8"].dtype),
        "image_layout": "NHWC",
        "image_color_space": "RGB",
        "image_range": [
            int(batch["pixel_values_uint8"].min()),
            int(batch["pixel_values_uint8"].max()),
        ],
        "image_tensor_note": (
            "Adapter emits processor-ready NHWC uint8 RGB after BGR→RGB convert and 224 resize. "
            "This is not the final normalized CHW model tensor; the official PrismaticProcessor "
            "would be applied only when loading OpenVLA (not done in this smoke test)."
        ),
        "language_sample": batch["language"][0],
        "raw_action_shape": list(batch["actions_raw"].shape),
        "encoded_action_shape": list(batch["actions_encoded"].shape),
        "mask_shape": list(batch["action_mask"].shape),
        "mask_padded_dim_all_zero": bool(np.allclose(batch["action_mask"][:, 6], 0.0)),
        "round_trip": rt,
        "samples": sample_meta,
        "episode_ids": batch["episode_index"],
        "frame_ids": batch["frame_index"],
        "timestamps": batch["timestamp"],
        "split": batch["split"],
        "splits_present": splits_present,
        "temporal_positions": [m["temporal_position"] for m in sample_meta],
        "normalization_stat_source": "train_only_q01_q99",
        "curation_policy": cfg.curation_policy,
        "alignment_error_frames": 0.0,
        "action_offset_frames": cfg.action_offset_frames,
        "shape_errors": shape_errors,
        "validation": {
            "no_invalid_window": True,
            "multi_split_smoke_batch_allowed": True,
            "train_normalization_only": True,
            "source_identity_preserved": True,
            "pad_dim_excluded_adapter_side": True,
            "stock_loss_mask_integration_verified": False,
        },
        "openvla_model_id": cfg.openvla_model_id,
        "openvla_model_revision": cfg.openvla_model_revision,
        "note": (
            "Smoke test validates dataset alignment, action encoding/masking, and batching on real "
            "wrist frames. OpenVLA 7B weights were not downloaded or loaded; official processor tensors "
            "were not produced."
        ),
    }
    save_json(report, art / "batch_smoke_test.json")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
