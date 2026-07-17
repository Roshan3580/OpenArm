"""Teleoperation quality audit: states, actions, timestamps, dropouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from openarm_pipeline.data.alignment import (
    aggregate_alignment_reports,
    check_episode_alignment,
    infer_dt,
)
from openarm_pipeline.data.lerobot_adapter import stack_vector_column


def describe_lengths(lengths: np.ndarray) -> dict[str, Any]:
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p5": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    return {
        "count": int(lengths.size),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "p5": float(np.percentile(lengths, 5)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p95": float(np.percentile(lengths, 95)),
    }


def nan_inf_counts(arr: np.ndarray) -> dict[str, int]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
        "total_elements": int(a.size),
    }


def per_dim_stats(arr: np.ndarray, names: list[str] | None = None) -> list[dict[str, Any]]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    n_dims = a.shape[1] if a.size else 0
    out: list[dict[str, Any]] = []
    for d in range(n_dims):
        col = a[:, d]
        finite = col[np.isfinite(col)]
        name = names[d] if names and d < len(names) else f"dim_{d}"
        if finite.size == 0:
            out.append(
                {
                    "name": name,
                    "dim": d,
                    "count": int(col.size),
                    "nan": int(np.isnan(col).sum()),
                    "inf": int(np.isinf(col).sum()),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "median": None,
                }
            )
            continue
        out.append(
            {
                "name": name,
                "dim": d,
                "count": int(col.size),
                "nan": int(np.isnan(col).sum()),
                "inf": int(np.isinf(col).sum()),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "median": float(np.median(finite)),
            }
        )
    return out


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def detect_discontinuities(
    arr: np.ndarray,
    mad_k: float = 8.0,
    mad_floor: float = 1e-6,
    abs_floor: float = 0.05,
    range_frac: float = 0.01,
) -> dict[str, Any]:
    """Flag large first-differences using max(k*MAD, abs_floor, range_frac*value_range)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[0] < 2:
        return {
            "n_frames": int(a.shape[0]),
            "n_dims": int(a.shape[1]) if a.size else 0,
            "total_flags": 0,
            "flag_rate": 0.0,
            "per_dim": [],
            "max_abs_delta_global": None,
        }

    deltas = np.diff(a, axis=0)
    per_dim = []
    total_flags = 0
    n_delta = deltas.shape[0]
    for d in range(deltas.shape[1]):
        col = deltas[:, d]
        finite = col[np.isfinite(col)]
        vals = a[:, d]
        vfinite = vals[np.isfinite(vals)]
        vrange = float(np.max(vfinite) - np.min(vfinite)) if vfinite.size else 0.0
        mad = max(robust_mad(finite), mad_floor)
        thr = max(mad_k * mad, abs_floor, range_frac * vrange)
        flags = np.abs(col) > thr
        n_flags = int(np.sum(flags & np.isfinite(col)))
        total_flags += n_flags
        # top candidate indices (within-series)
        abs_col = np.abs(col)
        top_idx = np.argsort(-abs_col)[:5]
        per_dim.append(
            {
                "dim": d,
                "mad": mad,
                "value_range": vrange,
                "threshold": float(thr),
                "n_flags": n_flags,
                "flag_rate": float(n_flags / n_delta) if n_delta else 0.0,
                "max_abs_delta": float(np.nanmax(abs_col)) if col.size else None,
                "top_delta_frame_indices": [int(i + 1) for i in top_idx if np.isfinite(abs_col[i])],
                "top_abs_deltas": [float(abs_col[i]) for i in top_idx if np.isfinite(abs_col[i])],
            }
        )

    return {
        "n_frames": int(a.shape[0]),
        "n_dims": int(a.shape[1]),
        "n_deltas": int(n_delta),
        "total_flags": int(total_flags),
        "flag_rate": float(total_flags / (n_delta * a.shape[1])) if n_delta and a.shape[1] else 0.0,
        "per_dim": per_dim,
        "max_abs_delta_global": float(np.nanmax(np.abs(deltas))) if deltas.size else None,
        "rule": (
            f"|delta| > max({mad_k}*MAD, abs_floor={abs_floor}, "
            f"range_frac={range_frac}*dim_range) (MAD floor={mad_floor})"
        ),
    }


def detect_near_stuck(
    arr: np.ndarray,
    std_thresh: float = 1e-5,
    range_thresh: float = 1e-4,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    out = []
    for d in range(a.shape[1] if a.size else 0):
        col = a[:, d]
        finite = col[np.isfinite(col)]
        name = names[d] if names and d < len(names) else f"dim_{d}"
        if finite.size == 0:
            out.append({"dim": d, "name": name, "near_stuck": True, "reason": "no_finite_values"})
            continue
        std = float(np.std(finite))
        rng = float(np.max(finite) - np.min(finite))
        stuck = std < std_thresh or rng < range_thresh
        out.append(
            {
                "dim": d,
                "name": name,
                "near_stuck": bool(stuck),
                "std": std,
                "range": rng,
                "std_thresh": std_thresh,
                "range_thresh": range_thresh,
            }
        )
    return out


def detect_robust_outliers(
    arr: np.ndarray,
    mad_k: float = 10.0,
    mad_floor: float = 1e-6,
) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    total = 0
    n_elem = int(a.size)
    per_dim = []
    for d in range(a.shape[1] if a.size else 0):
        col = a[:, d]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            per_dim.append({"dim": d, "n_outliers": 0, "rate": 0.0})
            continue
        med = float(np.median(finite))
        mad = max(robust_mad(finite), mad_floor)
        # modified z-score
        scores = np.abs(col - med) / (1.4826 * mad)
        flags = (scores > mad_k) & np.isfinite(col)
        n = int(np.sum(flags))
        total += n
        per_dim.append({"dim": d, "n_outliers": n, "rate": float(n / len(col)), "mad": mad, "median": med})
    return {
        "total_outliers": total,
        "total_elements": n_elem,
        "rate": float(total / n_elem) if n_elem else 0.0,
        "rule": f"modified_z = |x-median|/(1.4826*MAD) > {mad_k}",
        "per_dim": per_dim,
    }


def detect_boundary_saturation(
    arr: np.ndarray,
    edge_eps_frac: float = 0.01,
    rate_flag: float = 0.05,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Heuristic: fraction of frames near observed min/max (not known joint limits)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    out = []
    for d in range(a.shape[1] if a.size else 0):
        col = a[:, d]
        finite = col[np.isfinite(col)]
        name = names[d] if names and d < len(names) else f"dim_{d}"
        if finite.size == 0:
            continue
        lo, hi = float(np.min(finite)), float(np.max(finite))
        rng = hi - lo
        if rng <= 0:
            out.append(
                {
                    "dim": d,
                    "name": name,
                    "saturation_flag": False,
                    "low_rate": 0.0,
                    "high_rate": 0.0,
                    "note": "zero_range",
                }
            )
            continue
        eps = edge_eps_frac * rng
        low_rate = float(np.mean(finite <= (lo + eps)))
        high_rate = float(np.mean(finite >= (hi - eps)))
        out.append(
            {
                "dim": d,
                "name": name,
                "observed_min": lo,
                "observed_max": hi,
                "low_rate": low_rate,
                "high_rate": high_rate,
                "saturation_flag": bool(low_rate >= rate_flag or high_rate >= rate_flag),
                "caveat": "bounds are empirical dataset min/max, not known hardware joint limits",
            }
        )
    return out


def _motor_names_from_schema(feature: dict[str, Any] | None) -> list[str] | None:
    if not feature:
        return None
    names = feature.get("names")
    if isinstance(names, dict) and "motors" in names:
        return list(names["motors"])
    if isinstance(names, list):
        return [str(x) for x in names]
    return None


def audit_teleop(
    df: pd.DataFrame,
    feature_schema: dict[str, Any],
    state_key: str,
    action_key: str,
    fps: float | None,
    config: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any]:
    """Run full tabular teleoperation audit."""
    tele_cfg = config.get("teleop", {})
    episode_key = "episode_index" if "episode_index" in df.columns else None
    frame_key = "frame_index" if "frame_index" in df.columns else None
    ts_key = "timestamp" if "timestamp" in df.columns else None

    state = stack_vector_column(df[state_key]) if state_key in df.columns else np.zeros((0, 0))
    action = stack_vector_column(df[action_key]) if action_key in df.columns else np.zeros((0, 0))
    state_names = _motor_names_from_schema(feature_schema.get(state_key))
    action_names = _motor_names_from_schema(feature_schema.get(action_key))

    if episode_key:
        lengths = df.groupby(episode_key).size().to_numpy()
        n_episodes = int(df[episode_key].nunique())
    else:
        lengths = np.asarray([len(df)], dtype=np.int64)
        n_episodes = 1 if len(df) else 0

    expected_dt = infer_dt(
        df[ts_key].to_numpy() if ts_key else np.asarray([]),
        fps=fps,
    )

    episode_reports = []
    if episode_key and ts_key and frame_key:
        for ep, g in df.groupby(episode_key, sort=True):
            g = g.sort_values(frame_key)
            st = stack_vector_column(g[state_key]) if state_key in g.columns else np.zeros((0, 0))
            ac = stack_vector_column(g[action_key]) if action_key in g.columns else np.zeros((0, 0))
            episode_reports.append(
                check_episode_alignment(
                    timestamps=g[ts_key].to_numpy(),
                    frame_index=g[frame_key].to_numpy(),
                    state_len=len(st),
                    action_len=len(ac),
                    expected_dt=expected_dt,
                    gap_factor=float(tele_cfg.get("timestamp_gap_factor", 1.5)),
                    duplicate_tol=float(tele_cfg.get("duplicate_timestamp_tol_s", 1e-6)),
                )
            )
    alignment = aggregate_alignment_reports(episode_reports, expected_dt)

    # Global inferred dt from all positive diffs
    inferred_dt = expected_dt
    if ts_key and episode_key:
        dts = []
        for _, g in df.groupby(episode_key):
            ts = np.asarray(g[ts_key].to_numpy(), dtype=np.float64).reshape(-1)
            if len(ts) > 1:
                d = np.diff(np.sort(ts))
                dts.extend(d[d > 0].tolist())
        if dts:
            inferred_dt = float(np.median(dts))

    disc_kwargs = dict(
        mad_k=float(tele_cfg.get("discontinuity_mad_k", 8.0)),
        mad_floor=float(tele_cfg.get("discontinuity_mad_floor", 1e-6)),
        abs_floor=float(tele_cfg.get("discontinuity_abs_floor", 0.05)),
        range_frac=float(tele_cfg.get("discontinuity_range_frac", 0.01)),
    )
    disc_state = detect_discontinuities(state, **disc_kwargs)
    disc_action = detect_discontinuities(action, **disc_kwargs)

    # Per-episode discontinuity aggregation + anomaly counts
    ep_disc_flags = 0
    ep_disc_deltas = 0
    ep_action_disc_flags = 0
    ep_action_disc_deltas = 0
    per_episode_anomalies: list[dict[str, Any]] = []
    largest_state_disc_examples: list[dict[str, Any]] = []

    from openarm_pipeline.audit.gripper import analyze_gripper_channel, identify_gripper_dims

    grip_dims = identify_gripper_dims(state_names, state.shape[1] if state.size else 0)
    joint_dims = [i for i in range(state.shape[1] if state.size else 0) if i not in grip_dims]

    if episode_key and state_key in df.columns:
        for ep, g in df.groupby(episode_key):
            g = g.sort_values(frame_key or g.columns[0])
            st = stack_vector_column(g[state_key])
            ac = stack_vector_column(g[action_key]) if action_key in g.columns else np.zeros((0, 0))
            r_s = detect_discontinuities(st, **disc_kwargs)
            r_a = detect_discontinuities(ac, **disc_kwargs) if ac.size else {
                "total_flags": 0,
                "n_deltas": 0,
                "n_dims": 0,
                "max_abs_delta_global": None,
            }
            ep_disc_flags += r_s["total_flags"]
            ep_disc_deltas += r_s.get("n_deltas", 0) * max(r_s.get("n_dims", 0), 1)
            ep_action_disc_flags += r_a["total_flags"]
            ep_action_disc_deltas += r_a.get("n_deltas", 0) * max(r_a.get("n_dims", 0), 1)
            nan_s = int(np.isnan(st).sum()) if st.size else 0
            nan_a = int(np.isnan(ac).sum()) if ac.size else 0
            per_episode_anomalies.append(
                {
                    "episode_index": int(ep),
                    "n_frames": int(len(g)),
                    "state_discontinuity_flags": r_s["total_flags"],
                    "action_discontinuity_flags": r_a["total_flags"],
                    "state_max_abs_delta": r_s.get("max_abs_delta_global"),
                    "action_max_abs_delta": r_a.get("max_abs_delta_global"),
                    "state_nan": nan_s,
                    "action_nan": nan_a,
                }
            )
            # contextualize largest state jump in this episode
            if st.shape[0] >= 2:
                deltas = np.diff(st, axis=0)
                flat = np.nanmax(np.abs(deltas), axis=1)
                j = int(np.nanargmax(flat))
                largest_state_disc_examples.append(
                    {
                        "episode_index": int(ep),
                        "delta_at_frame_index": j + 1,
                        "max_abs_delta": float(flat[j]),
                        "state_before": st[j].tolist(),
                        "state_after": st[j + 1].tolist(),
                        "note": "largest within-episode |Δstate|_inf; review candidate, not auto-corrupt",
                    }
                )

    # Keep only global top-10 largest discontinuity contexts
    largest_state_disc_examples = sorted(
        largest_state_disc_examples, key=lambda x: -(x["max_abs_delta"] or 0)
    )[:10]

    gripper_analyses = []
    for d in grip_dims:
        name = state_names[d] if state_names else f"dim_{d}"
        gripper_analyses.append(analyze_gripper_channel(state[:, d], name=f"state.{name}"))
        if action.size and d < action.shape[1]:
            aname = action_names[d] if action_names else f"dim_{d}"
            gripper_analyses.append(analyze_gripper_channel(action[:, d], name=f"action.{aname}"))

    # Joint-only outliers (exclude gripper dims)
    def _outliers_subset(arr: np.ndarray, dims: list[int], names: list[str] | None) -> dict[str, Any]:
        if not dims or not arr.size:
            return {"total_outliers": 0, "total_elements": 0, "rate": 0.0, "dims": dims}
        sub = arr[:, dims]
        out = detect_robust_outliers(
            sub,
            mad_k=float(tele_cfg.get("outlier_mad_k", 10.0)),
            mad_floor=float(tele_cfg.get("discontinuity_mad_floor", 1e-6)),
        )
        out["dims"] = dims
        out["names"] = [names[i] for i in dims] if names else dims
        return out

    return {
        "scope": scope,
        "n_episodes": n_episodes,
        "n_frames": int(len(df)),
        "trajectory_length": describe_lengths(lengths),
        "trajectory_lengths_raw": lengths.astype(int).tolist(),
        "state_key": state_key,
        "action_key": action_key,
        "state_dim": int(state.shape[1]) if state.size else 0,
        "action_dim": int(action.shape[1]) if action.size else 0,
        "state_names": state_names,
        "action_names": action_names,
        "gripper_dims": grip_dims,
        "joint_dims": joint_dims,
        "state_nan_inf": nan_inf_counts(state),
        "action_nan_inf": nan_inf_counts(action),
        "state_per_dim": per_dim_stats(state, state_names),
        "action_per_dim": per_dim_stats(action, action_names),
        "fps_metadata": fps,
        "expected_dt_s": expected_dt,
        "inferred_dt_s": inferred_dt,
        "alignment": alignment.to_dict(),
        "state_discontinuities_global_concat": disc_state,
        "action_discontinuities_global_concat": disc_action,
        "state_discontinuities_per_episode": {
            "total_flags": ep_disc_flags,
            "total_delta_elements": ep_disc_deltas,
            "flag_rate": float(ep_disc_flags / ep_disc_deltas) if ep_disc_deltas else 0.0,
            "note": "computed within episodes to avoid cross-episode boundary artifacts",
        },
        "action_discontinuities_per_episode": {
            "total_flags": ep_action_disc_flags,
            "total_delta_elements": ep_action_disc_deltas,
            "flag_rate": float(ep_action_disc_flags / ep_action_disc_deltas)
            if ep_action_disc_deltas
            else 0.0,
        },
        "per_episode_anomaly_counts": per_episode_anomalies,
        "largest_state_discontinuity_contexts": largest_state_disc_examples,
        "gripper_channel_analysis": gripper_analyses,
        "state_near_stuck": detect_near_stuck(
            state,
            std_thresh=float(tele_cfg.get("near_stuck_std", 1e-5)),
            range_thresh=float(tele_cfg.get("near_stuck_range", 1e-4)),
            names=state_names,
        ),
        "action_near_stuck": detect_near_stuck(
            action,
            std_thresh=float(tele_cfg.get("near_stuck_std", 1e-5)),
            range_thresh=float(tele_cfg.get("near_stuck_range", 1e-4)),
            names=action_names,
        ),
        "state_outliers_all_dims": detect_robust_outliers(
            state,
            mad_k=float(tele_cfg.get("outlier_mad_k", 10.0)),
            mad_floor=float(tele_cfg.get("discontinuity_mad_floor", 1e-6)),
        ),
        "action_outliers_all_dims": detect_robust_outliers(
            action,
            mad_k=float(tele_cfg.get("outlier_mad_k", 10.0)),
            mad_floor=float(tele_cfg.get("discontinuity_mad_floor", 1e-6)),
        ),
        "state_outliers_joints_only": _outliers_subset(state, joint_dims, state_names),
        "action_outliers_joints_only": _outliers_subset(action, joint_dims, action_names),
        "state_outliers": detect_robust_outliers(
            state,
            mad_k=float(tele_cfg.get("outlier_mad_k", 10.0)),
            mad_floor=float(tele_cfg.get("discontinuity_mad_floor", 1e-6)),
        ),
        "action_outliers": detect_robust_outliers(
            action,
            mad_k=float(tele_cfg.get("outlier_mad_k", 10.0)),
            mad_floor=float(tele_cfg.get("discontinuity_mad_floor", 1e-6)),
        ),
        "state_boundary_saturation": detect_boundary_saturation(
            state,
            edge_eps_frac=float(tele_cfg.get("saturation_edge_eps_frac", 0.01)),
            rate_flag=float(tele_cfg.get("saturation_rate_flag", 0.05)),
            names=state_names,
        ),
        "missing_value_note": (
            "parquet float columns: NaN counted explicitly; "
            "no separate null mask beyond NaN for these arrays"
        ),
    }


def plot_trajectory_lengths(lengths: list[int] | np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt

    lengths = np.asarray(lengths)
    fig, ax = plt.subplots(figsize=(8, 4))
    if lengths.size == 0:
        ax.text(0.5, 0.5, "no episodes", ha="center", va="center")
    else:
        bins = min(30, max(5, int(np.sqrt(lengths.size) * 2)))
        ax.hist(lengths, bins=bins, color="#2c5f7c", edgecolor="white")
        ax.axvline(np.median(lengths), color="#c45c26", linestyle="--", label=f"median={np.median(lengths):.0f}")
        ax.legend()
    ax.set_xlabel("Trajectory length (frames)")
    ax.set_ylabel("Episode count")
    ax.set_title("Episode trajectory length distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_gap_distribution(gaps_s: list[float], expected_dt: float | None, out_path: str) -> None:
    import matplotlib.pyplot as plt

    gaps = np.asarray(gaps_s, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4))
    if gaps.size == 0:
        ax.text(0.5, 0.5, "no timestamp gaps flagged", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(gaps, bins=min(40, max(5, gaps.size)), color="#5b7c5a", edgecolor="white")
    if expected_dt:
        ax.axvline(expected_dt, color="#333", linestyle=":", label=f"expected_dt={expected_dt:.4f}s")
        ax.legend()
    ax.set_xlabel("Timestamp gap (s)")
    ax.set_ylabel("Count")
    ax.set_title("Flagged timestamp gaps within episodes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
