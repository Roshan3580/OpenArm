"""Temporal alignment helpers between camera frames and robot observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AlignmentReport:
    """Summary of frame/state/action temporal consistency checks."""

    expected_dt: float | None
    n_episodes: int
    episodes_with_non_monotonic_timestamps: int
    episodes_with_duplicate_timestamps: int
    episodes_with_frame_gaps: int
    episodes_with_length_mismatch: int
    total_timestamp_gaps: int
    total_duplicate_timestamps: int
    total_frame_index_gaps: int
    gap_magnitudes_s: list[float]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        gaps = np.asarray(self.gap_magnitudes_s, dtype=np.float64)
        gap_stats: dict[str, Any]
        if gaps.size == 0:
            gap_stats = {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p95": None,
            }
        else:
            gap_stats = {
                "count": int(gaps.size),
                "min": float(np.min(gaps)),
                "max": float(np.max(gaps)),
                "mean": float(np.mean(gaps)),
                "median": float(np.median(gaps)),
                "p95": float(np.percentile(gaps, 95)),
            }
        return {
            "expected_dt": self.expected_dt,
            "n_episodes": self.n_episodes,
            "episodes_with_non_monotonic_timestamps": self.episodes_with_non_monotonic_timestamps,
            "episodes_with_duplicate_timestamps": self.episodes_with_duplicate_timestamps,
            "episodes_with_frame_gaps": self.episodes_with_frame_gaps,
            "episodes_with_length_mismatch": self.episodes_with_length_mismatch,
            "total_timestamp_gaps": self.total_timestamp_gaps,
            "total_duplicate_timestamps": self.total_duplicate_timestamps,
            "total_frame_index_gaps": self.total_frame_index_gaps,
            "gap_magnitude_stats_s": gap_stats,
            "gap_magnitudes_s_sample": self.gap_magnitudes_s[:500],
            "notes": self.notes,
        }


def infer_dt(timestamps: np.ndarray, fps: float | None = None) -> float | None:
    """Infer sampling interval from timestamps or fps metadata."""
    if fps is not None and fps > 0:
        return 1.0 / float(fps)
    if timestamps is None or len(timestamps) < 2:
        return None
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64).reshape(-1))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return None
    return float(np.median(positive))


def check_episode_alignment(
    timestamps: np.ndarray,
    frame_index: np.ndarray,
    state_len: int,
    action_len: int,
    expected_dt: float | None,
    gap_factor: float = 1.5,
    duplicate_tol: float = 1e-6,
) -> dict[str, Any]:
    """Check monotonicity, duplicates, gaps, and length consistency for one episode."""
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    fi = np.asarray(frame_index, dtype=np.int64).reshape(-1)
    n = len(ts)
    notes: list[str] = []

    length_mismatch = not (n == len(fi) == state_len == action_len)
    if length_mismatch:
        notes.append(
            f"length_mismatch: ts={n}, frame_index={len(fi)}, state={state_len}, action={action_len}"
        )

    non_monotonic = bool(np.any(np.diff(ts) < -duplicate_tol)) if n > 1 else False
    diffs = np.diff(ts) if n > 1 else np.asarray([], dtype=np.float64)
    duplicate_mask = np.abs(diffs) <= duplicate_tol if diffs.size else np.asarray([], dtype=bool)
    n_duplicates = int(np.sum(duplicate_mask))

    gaps: list[float] = []
    if expected_dt is not None and diffs.size:
        threshold = expected_dt * gap_factor
        gap_mask = diffs > threshold
        gaps = [float(x) for x in diffs[gap_mask]]

    frame_gaps = 0
    if len(fi) > 1:
        fdiffs = np.diff(fi)
        frame_gaps = int(np.sum(fdiffs != 1))

    return {
        "n_frames": n,
        "non_monotonic": non_monotonic,
        "n_duplicate_timestamps": n_duplicates,
        "n_timestamp_gaps": len(gaps),
        "gap_magnitudes_s": gaps,
        "n_frame_index_gaps": frame_gaps,
        "length_mismatch": length_mismatch,
        "notes": notes,
    }


def aggregate_alignment_reports(
    episode_reports: list[dict[str, Any]],
    expected_dt: float | None,
) -> AlignmentReport:
    """Aggregate per-episode alignment checks."""
    all_gaps: list[float] = []
    notes: list[str] = []
    n_non_mono = 0
    n_dup_eps = 0
    n_frame_gap_eps = 0
    n_len_mismatch = 0
    total_dups = 0
    total_gaps = 0
    total_frame_gaps = 0

    for r in episode_reports:
        if r.get("non_monotonic"):
            n_non_mono += 1
        if r.get("n_duplicate_timestamps", 0) > 0:
            n_dup_eps += 1
            total_dups += int(r["n_duplicate_timestamps"])
        if r.get("n_frame_index_gaps", 0) > 0:
            n_frame_gap_eps += 1
            total_frame_gaps += int(r["n_frame_index_gaps"])
        if r.get("length_mismatch"):
            n_len_mismatch += 1
        gaps = r.get("gap_magnitudes_s", [])
        total_gaps += len(gaps)
        all_gaps.extend(gaps)
        notes.extend(r.get("notes", []))

    return AlignmentReport(
        expected_dt=expected_dt,
        n_episodes=len(episode_reports),
        episodes_with_non_monotonic_timestamps=n_non_mono,
        episodes_with_duplicate_timestamps=n_dup_eps,
        episodes_with_frame_gaps=n_frame_gap_eps,
        episodes_with_length_mismatch=n_len_mismatch,
        total_timestamp_gaps=total_gaps,
        total_duplicate_timestamps=total_dups,
        total_frame_index_gaps=total_frame_gaps,
        gap_magnitudes_s=all_gaps,
        notes=notes[:50],
    )
