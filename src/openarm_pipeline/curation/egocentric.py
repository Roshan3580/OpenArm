"""Egocentric visual-quality flags aligned to robot timesteps."""

from __future__ import annotations

from typing import Any

import numpy as np

from openarm_pipeline.audit.egocentric import (
    classify_frame_duplicate,
    score_frame,
    should_compare_adjacent_frames,
    to_gray_uint8,
)


def sustained_runs(flags: np.ndarray, min_run: int) -> np.ndarray:
    """Return boolean mask True for frames inside runs of length >= min_run."""
    f = np.asarray(flags, dtype=bool)
    out = np.zeros_like(f)
    i = 0
    n = len(f)
    while i < n:
        if not f[i]:
            i += 1
            continue
        j = i
        while j < n and f[j]:
            j += 1
        if j - i >= min_run:
            out[i:j] = True
        i = j
    return out


def compute_visual_timestep_flags(
    frames_rgb: list[np.ndarray | None],
    state_delta_norm: np.ndarray,
    action_delta_norm: np.ndarray,
    config: dict[str, Any],
    sharpness_soft_threshold: float | None = None,
    frame_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-timestep visual flags for one episode's wrist frames (aligned list).

    frames_rgb[i] is HxWx3 or None if missing/undecodable.
    state_delta_norm length n (0 at t=0; |Δ| into t for t>0).
    """
    ego = config.get("egocentric", {})
    n = len(frames_rgb)
    available = np.array([f is not None for f in frames_rgb], dtype=bool)
    decode_fail = ~available

    sharpness = np.full(n, np.nan)
    mean_luma = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    under = np.zeros(n, dtype=bool)
    over = np.zeros(n, dtype=bool)
    low_info = np.zeros(n, dtype=bool)
    low_sharp = np.zeros(n, dtype=bool)

    for i, fr in enumerate(frames_rgb):
        if fr is None:
            continue
        s = score_frame(fr, {"egocentric": ego})
        sharpness[i] = s["laplacian_var"]
        mean_luma[i] = s["mean_luma"]
        entropy[i] = s["entropy_bits"]
        under[i] = s["flags"]["underexposed"]
        over[i] = s["flags"]["overexposed"]
        low_info[i] = s["flags"]["low_entropy_possible_occlusion"]

    finite_sharp = sharpness[np.isfinite(sharpness)]
    if sharpness_soft_threshold is None and finite_sharp.size:
        pct = float(ego.get("soft_sharpness_percentile", 5.0))
        sharpness_soft_threshold = float(np.percentile(finite_sharp, pct))
    elif sharpness_soft_threshold is None:
        sharpness_soft_threshold = 0.0
    low_sharp = np.isfinite(sharpness) & (sharpness < sharpness_soft_threshold)

    nl_thr = float(ego.get("near_lossless_mse_threshold", 1.0))
    near_thr = float(ego.get("near_duplicate_mse_threshold", 25.0))

    exact = np.zeros(n, dtype=bool)
    near_lossless = np.zeros(n, dtype=bool)
    near = np.zeros(n, dtype=bool)
    if frame_index is None:
        fi = np.arange(n)
    else:
        fi = np.asarray(frame_index)
    # Per-episode callers pass one episode; still refuse non-consecutive indices.
    for i in range(1, n):
        if frames_rgb[i] is None or frames_rgb[i - 1] is None:
            continue
        if not should_compare_adjacent_frames(0, int(fi[i - 1]), 0, int(fi[i])):
            continue
        d = classify_frame_duplicate(
            frames_rgb[i - 1], frames_rgb[i], near_lossless_mse=nl_thr, near_mse=near_thr
        )
        exact[i] = d["exact_duplicate"]
        near_lossless[i] = d["near_lossless_duplicate"]
        near[i] = d["near_duplicate"]

    sd = np.asarray(state_delta_norm, dtype=np.float64)
    if len(sd) != n:
        sd = np.resize(sd, n)
    motion_pct = float(ego.get("motion_percentile", 50.0))
    moving_thr = float(np.percentile(sd[1:], motion_pct)) if n > 1 else 0.0
    robot_moving = sd >= moving_thr if moving_thr > 0 else sd > 0

    frozen_source = exact | near_lossless if ego.get("frozen_uses_near_lossless", True) else exact
    frozen_candidate = frozen_source & robot_moving
    frozen_sustained = sustained_runs(
        frozen_candidate, int(ego.get("sustained_frozen_run", 5))
    )
    over_sustained = sustained_runs(over, int(ego.get("sustained_overexposure_run", 5)))

    # Hard / soft validity
    hard_invalid = decode_fail.copy()
    soft_review = low_sharp | over | low_info | frozen_candidate
    # Soft exclusion for strict policy
    soft_exclude = over_sustained | frozen_sustained

    reason_codes = []
    for i in range(n):
        codes = []
        if decode_fail[i]:
            codes.append("missing_or_undecodable_wrist")
        if low_sharp[i]:
            codes.append("low_sharpness_soft")
        if under[i]:
            codes.append("underexposed")
        if over[i]:
            codes.append("overexposed")
        if over_sustained[i]:
            codes.append("sustained_overexposure")
        if low_info[i]:
            codes.append("low_entropy_heuristic")
        if exact[i]:
            codes.append("exact_duplicate")
        if near_lossless[i]:
            codes.append("near_lossless_duplicate")
        if near[i]:
            codes.append("near_duplicate")
        if frozen_candidate[i]:
            codes.append("frozen_while_moving_candidate")
        if frozen_sustained[i]:
            codes.append("sustained_frozen_while_moving")
        if robot_moving[i] and (exact[i] or near_lossless[i]):
            pass
        elif (exact[i] or near_lossless[i] or near[i]) and not robot_moving[i]:
            codes.append("stationary_duplicate_keep")
        reason_codes.append("|".join(codes) if codes else "")

    return {
        "wrist_available": available,
        "decode_failure": decode_fail,
        "sharpness": sharpness,
        "low_sharpness_candidate": low_sharp,
        "sharpness_soft_threshold": sharpness_soft_threshold,
        "mean_luma": mean_luma,
        "underexposed": under,
        "overexposed": over,
        "entropy": entropy,
        "low_information_candidate": low_info,
        "exact_duplicate": exact,
        "near_lossless_duplicate": near_lossless,
        "near_duplicate": near,
        "state_motion": sd,
        "action_motion": np.asarray(action_delta_norm, dtype=np.float64)
        if len(action_delta_norm) == n
        else np.resize(np.asarray(action_delta_norm, dtype=np.float64), n),
        "robot_moving": robot_moving,
        "motion_threshold": moving_thr,
        "frozen_while_moving_candidate": frozen_candidate,
        "frozen_sustained": frozen_sustained,
        "overexposure_sustained": over_sustained,
        "hard_invalid": hard_invalid,
        "soft_review": soft_review,
        "soft_exclude_strict": soft_exclude,
        "visual_hard_valid": ~hard_invalid,
        "visual_soft_valid": ~hard_invalid,  # soft cases still soft-valid
        "reason_codes": reason_codes,
        "notes": [
            "Absolute Laplacian threshold 50 is never used as a hard filter.",
            "Low entropy alone is not confirmed occlusion.",
            "Stationary duplicates are retained.",
            "Expected self-occlusion is not auto-inferred as unusable.",
        ],
    }


def load_episode_wrist_frames(
    video_path: str,
    from_timestamp: float,
    to_timestamp: float,
    n_frames: int,
    fps: float,
) -> list[np.ndarray | None]:
    """Decode n_frames from a concatenated v3 video between timestamps."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [None] * n_frames
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps) or fps
    start_idx = int(round(from_timestamp * vid_fps))
    out: list[np.ndarray | None] = []
    for i in range(n_frames):
        idx = start_idx + i
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            out.append(None)
        else:
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out
