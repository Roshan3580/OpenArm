"""Teleoperation hard validation, smoothing, and discontinuity diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from openarm_pipeline.audit.gripper import identify_gripper_dims
from openarm_pipeline.audit.teleop import detect_discontinuities, robust_mad
from openarm_pipeline.data.alignment import check_episode_alignment


def validate_episode_hard(
    episode_index: int,
    timestamps: np.ndarray,
    frame_index: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
    config: dict[str, Any],
    wrist_video_ok: bool = True,
    material_video_mismatch: bool = False,
) -> dict[str, Any]:
    """Return accept/reject decision with reason codes for one episode."""
    cfg = config.get("episode_validation", {})
    reasons: list[str] = []
    n = len(timestamps)
    min_frames = int(cfg.get("min_episode_frames", 64))
    horizon = int(config.get("horizon", 16))

    if n < min_frames:
        reasons.append("too_short_min_episode_frames")
    if n < horizon:
        reasons.append("too_few_frames_for_horizon")

    st = np.asarray(state, dtype=np.float64)
    ac = np.asarray(action, dtype=np.float64)
    if cfg.get("reject_nan_inf", True):
        if np.isnan(st).any() or np.isinf(st).any():
            reasons.append("state_nan_inf")
        if np.isnan(ac).any() or np.isinf(ac).any():
            reasons.append("action_nan_inf")

    align = check_episode_alignment(
        timestamps=timestamps,
        frame_index=frame_index,
        state_len=len(st),
        action_len=len(ac),
        expected_dt=1.0 / fps if fps else None,
        gap_factor=1.5,
    )
    if cfg.get("reject_non_monotonic_timestamps", True) and align["non_monotonic"]:
        reasons.append("non_monotonic_timestamps")
    if cfg.get("reject_duplicate_timestamps", True) and align["n_duplicate_timestamps"] > 0:
        reasons.append("duplicate_timestamps")
    if cfg.get("reject_frame_index_gaps", True) and align["n_frame_index_gaps"] > 0:
        reasons.append("frame_index_discontinuity")
    if cfg.get("reject_length_mismatch", True) and align["length_mismatch"]:
        reasons.append("state_action_frame_length_mismatch")
    if cfg.get("reject_missing_wrist_video", True) and not wrist_video_ok:
        reasons.append("missing_or_unreadable_wrist_video")
    if cfg.get("reject_material_video_mismatch", True) and material_video_mismatch:
        reasons.append("material_video_tabular_mismatch")

    return {
        "episode_index": int(episode_index),
        "n_frames": int(n),
        "accepted": len(reasons) == 0,
        "reasons": reasons,
        "alignment": align,
    }


def smooth_continuous_joints(
    state: np.ndarray,
    names: list[str] | None,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Savitzky–Golay smooth continuous joints; gripper excluded by default.

    Returns (smoothed_state, metrics). Original values are not mutated.
    """
    sm_cfg = config.get("smoothing", {})
    out = np.array(state, dtype=np.float64, copy=True)
    metrics: dict[str, Any] = {
        "enabled": bool(sm_cfg.get("enabled", True)),
        "method": sm_cfg.get("method", "savgol"),
        "per_dim": [],
        "applied": False,
    }
    if not sm_cfg.get("enabled", True) or out.ndim != 2 or out.shape[0] < 3:
        return out, metrics

    grip = identify_gripper_dims(names, out.shape[1]) if sm_cfg.get("exclude_gripper", True) else []
    joint_dims = [d for d in range(out.shape[1]) if d not in grip]
    window = int(sm_cfg.get("window_length", 5))
    poly = int(sm_cfg.get("polyorder", 2))
    if window % 2 == 0:
        window += 1
    if window > out.shape[0]:
        window = out.shape[0] if out.shape[0] % 2 == 1 else out.shape[0] - 1
    if window < poly + 2 or window < 3:
        metrics["note"] = "episode_too_short_for_savgol"
        return out, metrics

    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy required for Savitzky–Golay smoothing") from exc

    max_ratio = float(sm_cfg.get("max_rmse_over_std", 0.35))
    smoothed = out.copy()
    for d in joint_dims:
        raw = out[:, d]
        sm = savgol_filter(raw, window_length=window, polyorder=poly, mode="interp")
        diff = sm - raw
        rmse = float(np.sqrt(np.mean(diff**2)))
        std = float(np.std(raw)) if np.std(raw) > 0 else 1.0
        max_abs = float(np.max(np.abs(diff)))
        d1_raw = np.diff(raw)
        d1_sm = np.diff(sm)
        var_raw = float(np.var(d1_raw)) if d1_raw.size else 0.0
        var_sm = float(np.var(d1_sm)) if d1_sm.size else 0.0
        var_reduction = float((var_raw - var_sm) / var_raw) if var_raw > 0 else 0.0
        applied = rmse / std <= max_ratio
        if applied:
            smoothed[:, d] = sm
        metrics["per_dim"].append(
            {
                "dim": d,
                "name": names[d] if names and d < len(names) else f"dim_{d}",
                "rmse": rmse,
                "max_abs_change": max_abs,
                "first_diff_var_before": var_raw,
                "first_diff_var_after": var_sm,
                "first_diff_var_reduction": var_reduction,
                "rmse_over_std": rmse / std,
                "applied": applied,
            }
        )
    # Gripper dims unchanged by construction
    for d in grip:
        metrics["per_dim"].append(
            {
                "dim": d,
                "name": names[d] if names and d < len(names) else f"dim_{d}",
                "applied": False,
                "note": "gripper_excluded",
                "max_abs_change": 0.0,
                "rmse": 0.0,
            }
        )
    metrics["applied"] = any(p.get("applied") for p in metrics["per_dim"])
    metrics["window_length"] = window
    metrics["polyorder"] = poly
    return smoothed, metrics


def discontinuity_timestep_flags(
    state: np.ndarray,
    action: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Element-level and timestep-level discontinuity diagnostics (within episode)."""
    dcfg = config.get("discontinuity", {})
    kwargs = dict(
        mad_k=float(dcfg.get("mad_k", 8.0)),
        mad_floor=1e-6,
        abs_floor=float(dcfg.get("abs_floor", 1.0)),
        range_frac=float(dcfg.get("range_frac", 0.01)),
    )
    severe_kwargs = dict(
        mad_k=float(dcfg.get("severe_mad_k", 20.0)),
        mad_floor=1e-6,
        abs_floor=float(dcfg.get("severe_abs_floor", 8.0)),
        range_frac=float(dcfg.get("range_frac", 0.01)),
    )
    n = len(state)
    elem_state = detect_discontinuities(state, **kwargs)
    elem_action = detect_discontinuities(action, **kwargs)
    sev_state = detect_discontinuities(state, **severe_kwargs)
    sev_action = detect_discontinuities(action, **severe_kwargs)

    # Timestep flags: any dimension exceeded at transition into frame t (t>=1)
    def _timestep_any(arr: np.ndarray, mad_k: float, abs_floor: float, range_frac: float) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        flags = np.zeros(len(a), dtype=bool)
        if len(a) < 2:
            return flags
        deltas = np.diff(a, axis=0)
        for d in range(a.shape[1]):
            col = deltas[:, d]
            finite = col[np.isfinite(col)]
            vals = a[:, d]
            vfinite = vals[np.isfinite(vals)]
            vrange = float(np.ptp(vfinite)) if vfinite.size else 0.0
            mad = max(robust_mad(finite), 1e-6)
            thr = max(mad_k * mad, abs_floor, range_frac * vrange)
            hit = np.abs(col) > thr
            flags[1:] |= hit
        return flags

    ts_ord = _timestep_any(
        state, kwargs["mad_k"], kwargs["abs_floor"], kwargs["range_frac"]
    ) | _timestep_any(action, kwargs["mad_k"], kwargs["abs_floor"], kwargs["range_frac"])
    ts_sev = _timestep_any(
        state, severe_kwargs["mad_k"], severe_kwargs["abs_floor"], severe_kwargs["range_frac"]
    ) | _timestep_any(
        action, severe_kwargs["mad_k"], severe_kwargs["abs_floor"], severe_kwargs["range_frac"]
    )

    exclude_ord = bool(dcfg.get("exclude_ordinary_from_windows", False))
    exclude_sev_strict = bool(dcfg.get("exclude_severe_from_strict_windows", True))
    window_exclude_conservative = np.zeros(n, dtype=bool)  # ordinary never excludes by default
    window_exclude_strict = ts_sev if exclude_sev_strict else np.zeros(n, dtype=bool)
    if exclude_ord:
        window_exclude_conservative = ts_ord
        window_exclude_strict = window_exclude_strict | ts_ord

    return {
        "element_state": elem_state,
        "element_action": elem_action,
        "element_state_severe": sev_state,
        "element_action_severe": sev_action,
        "timestep_any_discontinuity": ts_ord,
        "timestep_severe_discontinuity": ts_sev,
        "window_exclude_conservative": window_exclude_conservative,
        "window_exclude_strict": window_exclude_strict,
        "note": (
            "Element-level rates are diagnostic; ordinary discontinuities do not "
            "auto-exclude timesteps unless configured."
        ),
    }
