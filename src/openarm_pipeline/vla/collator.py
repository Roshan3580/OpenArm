"""Collate OpenVLA-style training examples without loading the model."""

from __future__ import annotations

from typing import Any

import numpy as np


def collate_openvla_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack images/actions/masks; keep language as list; preserve source IDs."""
    if not examples:
        raise ValueError("empty batch")
    images = np.stack([np.asarray(e["image_rgb"], dtype=np.uint8) for e in examples], axis=0)
    actions = np.stack([np.asarray(e["action_encoded"], dtype=np.float64) for e in examples], axis=0)
    masks = np.stack([np.asarray(e["action_mask"], dtype=np.float64) for e in examples], axis=0)
    raw = np.stack([np.asarray(e["action_raw"], dtype=np.float64) for e in examples], axis=0)
    return {
        "pixel_values_uint8": images,  # [B,H,W,3]
        "input_image_shape": list(images.shape),
        "language": [e["instruction"] for e in examples],
        "actions_encoded": actions,  # [B,7]
        "actions_raw": raw,  # [B,6]
        "action_mask": masks,  # [B,7]
        "episode_index": [int(e["episode_index"]) for e in examples],
        "frame_index": [int(e["frame_index"]) for e in examples],
        "timestamp": [float(e["timestamp"]) for e in examples],
        "split": [e["split"] for e in examples],
        "curation_policy": [e["curation_policy"] for e in examples],
        "dataset_revision": examples[0]["dataset_revision"],
        "view": examples[0].get("view", "wrist"),
        "action_offset_frames": int(examples[0].get("action_offset_frames", 0)),
        "batch_size": len(examples),
    }
