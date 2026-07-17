"""Deterministic wrist-image preprocessing for OpenVLA-style 224px inputs."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


UNSAFE_AUGMENTATIONS = {
    "horizontal_flip",
    "vertical_flip",
    "large_rotation",
    "time_reversal",
    "frame_reorder",
}


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def resize_square(rgb: np.ndarray, size: int = 224, method: str = "bilinear") -> np.ndarray:
    interp = cv2.INTER_LINEAR if method == "bilinear" else cv2.INTER_AREA
    return cv2.resize(rgb, (size, size), interpolation=interp)


def preprocess_wrist_frame(
    frame_bgr_or_rgb: np.ndarray,
    *,
    size: int = 224,
    assume_bgr: bool = True,
    method: str = "bilinear",
) -> np.ndarray:
    """Return uint8 RGB HxWx3 at `size`, matching OpenVLA's 224px family input scale.

    Numeric mean/std normalization is deferred to the official PrismaticImageProcessor
    at training time; this adapter produces a deterministic resized RGB uint8 image.
    """
    arr = np.asarray(frame_bgr_or_rgb)
    if assume_bgr and arr.ndim == 3 and arr.shape[2] == 3:
        rgb = bgr_to_rgb(arr)
    else:
        rgb = arr[..., :3].copy()
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb = rgb.clip(0, 255).astype(np.uint8)
    return resize_square(rgb, size=size, method=method)


def safe_augmentation(
    rgb: np.ndarray,
    *,
    seed: int,
    brightness: float = 0.15,
    contrast: float = 0.15,
    blur_prob: float = 0.2,
) -> np.ndarray:
    """Wrist-safe photometric augmentation (no flips/large rotations)."""
    rng = np.random.default_rng(seed)
    out = rgb.astype(np.float32)
    # brightness / contrast
    b = 1.0 + float(rng.uniform(-brightness, brightness))
    c = 1.0 + float(rng.uniform(-contrast, contrast))
    mean = out.mean()
    out = (out - mean) * c + mean
    out = out * b
    out = np.clip(out, 0, 255)
    if rng.random() < blur_prob:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out.astype(np.uint8), (k, k), 0).astype(np.float32)
    # mild center-preserving scale crop
    h, w = out.shape[:2]
    scale = float(rng.uniform(0.9, 1.0))
    nh, nw = int(h * scale), int(w * scale)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    crop = out[y0 : y0 + nh, x0 : x0 + nw]
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return crop.clip(0, 255).astype(np.uint8)


def assert_augmentation_safe(name: str) -> None:
    if name in UNSAFE_AUGMENTATIONS:
        raise ValueError(f"unsafe augmentation disabled: {name}")


def preprocessing_summary(size: int = 224) -> dict[str, Any]:
    return {
        "source_typical_resolution": [480, 640],
        "color_space": "RGB",
        "resize_method": "bilinear",
        "crop_behavior": "none_at_export_center_crop_optional_in_safe_aug",
        "output_resolution": [size, size],
        "numeric_range_at_export": "uint8_[0,255]",
        "normalization": "deferred_to_OpenVLA_PrismaticImageProcessor",
        "augmentation_policy": "wrist_safe_photometric_optional",
        "unsafe_disabled": sorted(UNSAFE_AUGMENTATIONS),
        "ownership": "dataset_export_produces_uint8_RGB; model_processor_applies_train_norm",
    }
