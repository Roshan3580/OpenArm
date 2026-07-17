"""Cross-modal analysis linking visual flags to robot motion."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_cross_modal(
    sharpness: np.ndarray,
    low_info_flag: np.ndarray,
    exact_dup_flag: np.ndarray,
    near_dup_flag: np.ndarray,
    state_delta_norm: np.ndarray,
    action_delta_norm: np.ndarray,
    gripper_abs_delta: np.ndarray | None,
    blur_threshold: float,
) -> dict[str, Any]:
    """Relate visual quality flags to motion magnitude (aligned arrays, length N or N-1)."""
    sharp = np.asarray(sharpness, dtype=np.float64)
    n = sharp.size
    # Align delta arrays (typically length n-1) to frame pairs ending at t
    sd = np.asarray(state_delta_norm, dtype=np.float64)
    ad = np.asarray(action_delta_norm, dtype=np.float64)

    def _pair(flag: np.ndarray) -> dict[str, Any]:
        f = np.asarray(flag, dtype=bool)
        if f.size == n and sd.size == n - 1:
            # flag on frame t compared with motion into t (delta[t-1])
            f_m = f[1:]
            motion = sd
            action_m = ad if ad.size == n - 1 else ad
        elif f.size == sd.size:
            f_m = f
            motion = sd
            action_m = ad
        else:
            m = min(f.size, sd.size)
            f_m = f[:m]
            motion = sd[:m]
            action_m = ad[:m]
        if f_m.size == 0:
            return {"n_flagged": 0}
        flagged = f_m
        return {
            "n_flagged": int(flagged.sum()),
            "mean_state_delta_norm_flagged": float(np.mean(motion[flagged])) if flagged.any() else None,
            "mean_state_delta_norm_unflagged": float(np.mean(motion[~flagged])) if (~flagged).any() else None,
            "mean_action_delta_norm_flagged": float(np.mean(action_m[flagged])) if flagged.any() else None,
            "mean_action_delta_norm_unflagged": float(np.mean(action_m[~flagged])) if (~flagged).any() else None,
            "fraction_flagged_in_top_quartile_state_motion": (
                float(np.mean(motion[flagged] >= np.percentile(motion, 75))) if flagged.any() else None
            ),
        }

    blur_flag = sharp < blur_threshold
    # sustained intervals: runs of blur
    sustained = _run_length_stats(blur_flag)

    dup_while_moving = None
    if exact_dup_flag is not None and sd.size:
        d = np.asarray(exact_dup_flag, dtype=bool)
        m = min(d.size, sd.size)
        if m:
            moving = sd[:m] > np.percentile(sd[:m], 50)
            dup_while_moving = {
                "n_exact_dup_and_above_median_state_motion": int(np.sum(d[:m] & moving)),
                "n_exact_dup": int(np.sum(d[:m])),
                "rate_among_dups": float(np.mean(moving[d[:m]])) if np.any(d[:m]) else None,
                "note": (
                    "Duplicate visuals while state changes suggest frozen/stream stutter "
                    "or encode reuse — soft review flag, not automatic corruption."
                ),
            }

    grasp_proxy = None
    if gripper_abs_delta is not None:
        g = np.asarray(gripper_abs_delta, dtype=np.float64)
        m = min(low_info_flag.size, g.size)
        if m:
            g75 = np.percentile(g[:m], 75)
            li = np.asarray(low_info_flag[:m], dtype=bool)
            grasp_proxy = {
                "low_info_rate_during_large_gripper_delta": float(np.mean(li[g[:m] >= g75]))
                if np.any(g[:m] >= g75)
                else None,
                "low_info_rate_otherwise": float(np.mean(li[g[:m] < g75])) if np.any(g[:m] < g75) else None,
                "note": (
                    "Elevated low-information rates near large gripper deltas may reflect "
                    "expected close-up grasp / self-occlusion, not unusable camera failure."
                ),
            }

    return {
        "blur_vs_motion": _pair(blur_flag),
        "low_info_vs_motion": _pair(np.asarray(low_info_flag, dtype=bool)),
        "near_dup_vs_motion": _pair(np.asarray(near_dup_flag, dtype=bool)),
        "duplicate_while_state_changes": dup_while_moving,
        "low_info_near_grasp_proxy": grasp_proxy,
        "blur_run_length_stats": sustained,
        "interpretation_rules": [
            "Do not treat expected gripper self-occlusion during grasp as corruption.",
            "Isolated single-frame blur spikes coincident with fast motion are soft review flags.",
            "Sustained blur/low-info intervals are stronger soft flags for shared-timestep masking.",
            "Never delete a wrist frame independently of its state/action timestep.",
        ],
    }


def _run_length_stats(flags: np.ndarray) -> dict[str, Any]:
    f = np.asarray(flags, dtype=bool)
    if f.size == 0:
        return {"n_runs": 0, "max_run": 0, "mean_run": None, "isolated_single_frame_runs": 0}
    runs = []
    i = 0
    while i < len(f):
        if not f[i]:
            i += 1
            continue
        j = i
        while j < len(f) and f[j]:
            j += 1
        runs.append(j - i)
        i = j
    if not runs:
        return {"n_runs": 0, "max_run": 0, "mean_run": None, "isolated_single_frame_runs": 0}
    return {
        "n_runs": len(runs),
        "max_run": int(max(runs)),
        "mean_run": float(np.mean(runs)),
        "isolated_single_frame_runs": int(sum(1 for r in runs if r == 1)),
        "sustained_runs_ge_5": int(sum(1 for r in runs if r >= 5)),
    }


def per_frame_state_action_deltas(
    state: np.ndarray,
    action: np.ndarray,
    gripper_dims: list[int] | None = None,
) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    if action.ndim == 1:
        action = action.reshape(-1, 1)
    sd = np.linalg.norm(np.diff(state, axis=0), axis=1) if len(state) > 1 else np.zeros(0)
    ad = np.linalg.norm(np.diff(action, axis=0), axis=1) if len(action) > 1 else np.zeros(0)
    gd = None
    if gripper_dims:
        cols = [d for d in gripper_dims if d < state.shape[1]]
        if cols and len(state) > 1:
            gd = np.linalg.norm(np.diff(state[:, cols], axis=0), axis=1)
    return {"state_delta_norm": sd, "action_delta_norm": ad, "gripper_abs_delta": gd}
