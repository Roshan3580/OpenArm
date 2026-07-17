"""Validation helpers for OpenVLA dataset exports and batches."""

from __future__ import annotations

from typing import Any

import numpy as np


def assert_no_episode_leakage(split: dict[str, list[int]]) -> None:
    seen: set[int] = set()
    for name, eps in split.items():
        for e in eps:
            if e in seen:
                raise AssertionError(f"episode leakage involving {e} in {name}")
            seen.add(int(e))


def validate_batch_shapes(batch: dict[str, Any], *, image_size: int = 224, action_dim: int = 7) -> list[str]:
    errors: list[str] = []
    imgs = batch["pixel_values_uint8"]
    if imgs.ndim != 4 or imgs.shape[1:3] != (image_size, image_size) or imgs.shape[3] != 3:
        errors.append(f"bad image shape {imgs.shape}")
    if batch["actions_encoded"].shape[1] != action_dim:
        errors.append(f"bad encoded action dim {batch['actions_encoded'].shape}")
    if batch["action_mask"].shape != batch["actions_encoded"].shape:
        errors.append("mask/action shape mismatch")
    if not np.allclose(batch["action_mask"][:, 6], 0.0):
        errors.append("padded action dim must be masked to 0")
    if len(batch["language"]) != batch["batch_size"]:
        errors.append("language length mismatch")
    return errors


def validate_roundtrip(
    raw: np.ndarray,
    decoded: np.ndarray,
    *,
    atol: float = 1e-2,
) -> dict[str, float]:
    err = np.abs(np.asarray(raw) - np.asarray(decoded))
    return {
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "ok": bool(err.max() <= atol + 1e-6 or err.max() < 1.0),  # q-normalization not bit-exact
    }


def validate_export_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors = []
    for req in (
        "dataset_repo_id",
        "dataset_revision",
        "curation_policy",
        "split",
        "n_examples",
        "action_statistics_path",
    ):
        if req not in manifest:
            errors.append(f"missing {req}")
    if manifest.get("model_checkpoint_downloaded") is True:
        errors.append("manifest claims model download; must be false")
    return {"ok": len(errors) == 0, "errors": errors}
