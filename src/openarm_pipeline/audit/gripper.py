"""Gripper-channel analysis: bimodality vs corruption."""

from __future__ import annotations

from typing import Any

import numpy as np


def analyze_gripper_channel(
    values: np.ndarray,
    name: str = "gripper",
    n_bins: int = 40,
) -> dict[str, Any]:
    """Characterize gripper value distribution without labeling modes as corruption.

    Uses histogram peak counting and simple two-mode separation heuristics.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "name": name,
            "n": 0,
            "bimodal_likely": False,
            "note": "no_finite_values",
        }

    lo, hi = float(np.min(finite)), float(np.max(finite))
    hist, edges = np.histogram(finite, bins=n_bins, range=(lo, hi) if hi > lo else None)
    # peak = bin taller than both neighbors
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 0.05 * hist.max():
            center = 0.5 * (edges[i] + edges[i + 1])
            peaks.append({"bin": i, "count": int(hist[i]), "center": float(center)})

    # End bins as candidate modes (fully open/closed concentrations)
    end_left = float(hist[0] / finite.size)
    end_right = float(hist[-1] / finite.size)
    q10, q50, q90 = np.percentile(finite, [10, 50, 90])
    # mass near extremes
    span = hi - lo if hi > lo else 1.0
    near_low = float(np.mean(finite <= lo + 0.15 * span))
    near_high = float(np.mean(finite >= hi - 0.15 * span))
    bimodal_likely = (len(peaks) >= 2) or (near_low > 0.15 and near_high > 0.15)

    return {
        "name": name,
        "n": int(finite.size),
        "min": lo,
        "max": hi,
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "q10": float(q10),
        "q50": float(q50),
        "q90": float(q90),
        "n_histogram_peaks": len(peaks),
        "peaks": peaks[:5],
        "mass_near_low_15pct_range": near_low,
        "mass_near_high_15pct_range": near_high,
        "end_bin_mass_left": end_left,
        "end_bin_mass_right": end_right,
        "bimodal_or_open_closed_concentrated": bool(bimodal_likely),
        "interpretation": (
            "Gripper appears concentrated near open/closed (or multi-modal). "
            "Statistical outlier rates on this channel often reflect normal grasp "
            "transitions rather than sensor corruption — do not delete normal grasp transitions."
            if bimodal_likely
            else "Gripper distribution not clearly bimodal under this heuristic; review before filtering."
        ),
        "recommendation": "diagnostic_only_separate_from_joint_outlier_filters",
    }


def identify_gripper_dims(names: list[str] | None, n_dims: int) -> list[int]:
    if not names:
        return []
    out = []
    for i, n in enumerate(names):
        if i >= n_dims:
            break
        if "gripper" in str(n).lower() or "grip" in str(n).lower():
            out.append(i)
    return out
