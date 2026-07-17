"""Video integrity and tabular–video temporal alignment checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def probe_video_file(path: str) -> dict[str, Any]:
    """Open a video and record container-level properties."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {
            "path": path,
            "opened": False,
            "error": "unreadable_or_missing",
            "n_frames_container": 0,
            "fps": None,
            "width": None,
            "height": None,
            "duration_s": None,
            "fourcc": None,
        }
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    duration = (n / fps) if fps and n else None
    # sample-decode first and last frame
    ok_first, _ = cap.read()
    decoded = 0
    if ok_first:
        decoded += 1
    if n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok_last, _ = cap.read()
        if ok_last:
            decoded += 1
    cap.release()
    return {
        "path": path,
        "opened": True,
        "error": None,
        "n_frames_container": n,
        "fps": fps,
        "width": w,
        "height": h,
        "duration_s": duration,
        "fourcc": fourcc,
        "sample_decoded_endpoint_frames": decoded,
        "zero_length": n == 0,
    }


def episode_video_alignment_row(
    episode_index: int,
    tabular_length: int,
    tabular_timestamps: np.ndarray,
    fps: float,
    from_timestamp: float | None,
    to_timestamp: float | None,
    container_fps: float | None,
    timing_tol_s: float = 1.0 / 30.0,
    material_frame_mismatch: int = 2,
    material_duration_mismatch_s: float = 0.1,
) -> dict[str, Any]:
    """Compare one episode's tabular timeline to its video segment metadata."""
    ts = np.asarray(tabular_timestamps, dtype=np.float64).reshape(-1)
    expected_frames = int(tabular_length)
    expected_duration_from_frames = (expected_frames - 1) / fps if expected_frames > 1 and fps else None
    ts_span = float(ts[-1] - ts[0]) if ts.size > 1 else (0.0 if ts.size == 1 else None)

    video_duration = None
    video_frames_from_timestamps = None
    if from_timestamp is not None and to_timestamp is not None:
        video_duration = float(to_timestamp - from_timestamp)
        use_fps = container_fps or fps
        if use_fps:
            video_frames_from_timestamps = int(round(video_duration * use_fps))

    # frame_index/fps vs timestamp mismatch within episode
    max_timing_mismatch = None
    if ts.size and fps:
        frame_idx = np.arange(ts.size, dtype=np.float64)
        pred = frame_idx / float(fps)
        # align pred to start at first timestamp
        pred = pred + float(ts[0])
        max_timing_mismatch = float(np.max(np.abs(pred - ts)))

    frame_mismatch = None
    if video_frames_from_timestamps is not None:
        frame_mismatch = int(video_frames_from_timestamps - expected_frames)

    duration_mismatch = None
    if video_duration is not None and expected_duration_from_frames is not None:
        duration_mismatch = float(video_duration - expected_duration_from_frames)

    material_frames = frame_mismatch is not None and abs(frame_mismatch) >= material_frame_mismatch
    material_duration = (
        duration_mismatch is not None and abs(duration_mismatch) >= material_duration_mismatch_s
    )
    timing_ok = max_timing_mismatch is None or max_timing_mismatch <= timing_tol_s

    return {
        "episode_index": int(episode_index),
        "expected_tabular_frame_count": expected_frames,
        "container_reported_segment_frame_count_est": video_frames_from_timestamps,
        "video_segment_from_timestamp": from_timestamp,
        "video_segment_to_timestamp": to_timestamp,
        "video_segment_duration_s": video_duration,
        "expected_duration_from_tabular_frames_s": expected_duration_from_frames,
        "tabular_timestamp_span_s": ts_span,
        "duration_mismatch_s": duration_mismatch,
        "frame_count_delta_est": frame_mismatch,
        "max_frame_index_over_fps_vs_timestamp_s": max_timing_mismatch,
        "timing_within_tolerance": bool(timing_ok),
        "timing_tol_s": timing_tol_s,
        "material_frame_count_mismatch": bool(material_frames),
        "material_duration_mismatch": bool(material_duration),
        "flags": {
            "material_frame_count_mismatch": bool(material_frames),
            "material_duration_mismatch": bool(material_duration),
            "timing_tolerance_exceeded": not timing_ok,
        },
    }


def build_video_alignment_report(
    episodes_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
    camera_keys: list[str],
    video_probes: dict[str, dict[str, Any]],
    fps: float,
    timing_tol_s: float = 1.0 / 30.0,
    material_frame_mismatch: int = 2,
    material_duration_mismatch_s: float = 0.1,
) -> dict[str, Any]:
    """Per-episode, per-camera alignment report for a LeRobot v3-style dataset."""
    per_episode: list[dict[str, Any]] = []
    summary_flags = {
        "missing_videos": 0,
        "unreadable_videos": 0,
        "zero_length_videos": 0,
        "episodes_material_frame_mismatch": 0,
        "episodes_material_duration_mismatch": 0,
        "episodes_timing_tolerance_exceeded": 0,
    }

    for cam in camera_keys:
        probe = video_probes.get(cam) or {}
        if not probe.get("opened"):
            if probe.get("error") == "unreadable_or_missing":
                summary_flags["unreadable_videos"] += 1
            else:
                summary_flags["missing_videos"] += 1
        if probe.get("zero_length"):
            summary_flags["zero_length_videos"] += 1

    for _, erow in episodes_df.sort_values("episode_index").iterrows():
        ep = int(erow["episode_index"])
        length = int(erow["length"]) if "length" in erow else int(
            (tabular_df["episode_index"] == ep).sum()
        )
        g = tabular_df[tabular_df["episode_index"] == ep].sort_values("frame_index")
        ts = g["timestamp"].to_numpy() if "timestamp" in g.columns else np.asarray([])
        cam_rows = {}
        ep_material_frame = False
        ep_material_dur = False
        ep_timing = False
        for cam in camera_keys:
            from_key = f"videos/{cam}/from_timestamp"
            to_key = f"videos/{cam}/to_timestamp"
            from_ts = float(erow[from_key]) if from_key in erow.index else None
            to_ts = float(erow[to_key]) if to_key in erow.index else None
            probe = video_probes.get(cam) or {}
            row = episode_video_alignment_row(
                episode_index=ep,
                tabular_length=length,
                tabular_timestamps=ts,
                fps=fps,
                from_timestamp=from_ts,
                to_timestamp=to_ts,
                container_fps=probe.get("fps"),
                timing_tol_s=timing_tol_s,
                material_frame_mismatch=material_frame_mismatch,
                material_duration_mismatch_s=material_duration_mismatch_s,
            )
            row["camera_key"] = cam
            row["video_file_opened"] = bool(probe.get("opened"))
            row["container_n_frames_file"] = probe.get("n_frames_container")
            row["successfully_inspected_endpoint_frames"] = probe.get(
                "sample_decoded_endpoint_frames"
            )
            cam_rows[cam] = row
            ep_material_frame |= row["material_frame_count_mismatch"]
            ep_material_dur |= row["material_duration_mismatch"]
            ep_timing |= bool(row["flags"]["timing_tolerance_exceeded"])

        if ep_material_frame:
            summary_flags["episodes_material_frame_mismatch"] += 1
        if ep_material_dur:
            summary_flags["episodes_material_duration_mismatch"] += 1
        if ep_timing:
            summary_flags["episodes_timing_tolerance_exceeded"] += 1

        per_episode.append(
            {
                "episode_index": ep,
                "tabular_length": length,
                "cameras": cam_rows,
            }
        )

    return {
        "fps_metadata": fps,
        "timing_tol_s": timing_tol_s,
        "material_frame_mismatch_threshold": material_frame_mismatch,
        "material_duration_mismatch_s": material_duration_mismatch_s,
        "video_file_probes": video_probes,
        "n_episodes": len(per_episode),
        "summary_flags": summary_flags,
        "episodes": per_episode,
        "definitions": {
            "expected_tabular_frame_count": "rows in parquet for the episode (or episodes.length)",
            "container_reported_segment_frame_count_est": "round((to_ts-from_ts)*fps) from episode video timestamps",
            "successfully_inspected_or_decoded": (
                "endpoint decode probe at file level; full wrist decode counts appear in egocentric audit"
            ),
        },
    }


def contiguous_windows_for_episode(
    episode_length: int,
    n_windows: int,
    window_size: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Deterministic contiguous windows spanning an episode.

    Returns list of (start, end) half-open frame_index ranges within [0, episode_length).
    Guarantees coverage attempt on every episode; windows may shrink for short episodes.
    """
    if episode_length <= 0 or n_windows <= 0 or window_size <= 0:
        return []
    ws = min(window_size, episode_length)
    if n_windows == 1:
        start = max(0, (episode_length - ws) // 2)
        return [(start, start + ws)]

    # Place window starts evenly from 0 to length-ws
    max_start = max(episode_length - ws, 0)
    starts = np.linspace(0, max_start, n_windows)
    # Deterministic micro-jitter from seed without overlapping chaos
    rng = np.random.default_rng(seed + episode_length)
    jitter = rng.integers(0, max(1, ws // 4), size=n_windows) if max_start > 0 else np.zeros(n_windows, dtype=int)
    out: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, s in enumerate(starts):
        st = int(np.clip(int(round(s)) + int(jitter[i]), 0, max_start))
        # de-dup identical starts
        while st in used and st < max_start:
            st += 1
        used.add(st)
        out.append((st, st + ws))
    return out


def plan_episode_windows(
    episode_lengths: dict[int, int],
    n_windows: int,
    window_size: int,
    seed: int = 42,
    min_total_frames: int = 2000,
) -> dict[str, Any]:
    """Build per-episode contiguous windows covering every episode."""
    plan = {}
    total = 0
    for ep, length in sorted(episode_lengths.items()):
        wins = contiguous_windows_for_episode(length, n_windows, window_size, seed=seed + int(ep))
        plan[int(ep)] = wins
        total += sum(e - s for s, e in wins)
    # If under min_total_frames, grow window_size proportionally
    grown = False
    if total < min_total_frames and episode_lengths:
        scale = int(np.ceil(min_total_frames / max(total, 1)))
        new_ws = window_size * max(scale, 2)
        plan = {}
        total = 0
        for ep, length in sorted(episode_lengths.items()):
            wins = contiguous_windows_for_episode(length, n_windows, new_ws, seed=seed + int(ep))
            plan[int(ep)] = wins
            total += sum(e - s for s, e in wins)
        window_size = new_ws
        grown = True
    return {
        "seed": seed,
        "n_windows": n_windows,
        "window_size": window_size,
        "window_size_grown": grown,
        "min_total_frames_target": min_total_frames,
        "total_frames_planned": total,
        "n_episodes_covered": len(plan),
        "episodes_covered": sorted(plan.keys()),
        "covers_every_episode": len(plan) == len(episode_lengths),
        "windows": {str(k): v for k, v in plan.items()},
    }
