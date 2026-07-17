"""Temporal aggregation of frame-level success-detector probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def hysteresis_triggers(
    probs: np.ndarray,
    threshold: float,
    *,
    window: int = 5,
    votes_required: int = 4,
    reset_threshold: float | None = None,
) -> dict[str, Any]:
    """Four-of-five style hysteresis over a probability sequence.

    Returns per-frame trigger state and summary stats.
    """
    p = np.asarray(probs, dtype=float)
    n = len(p)
    if reset_threshold is None:
        reset_threshold = float(threshold) - 0.10
    triggered = np.zeros(n, dtype=bool)
    state = False
    changes = 0
    for i in range(n):
        start = max(0, i - window + 1)
        votes = int(np.sum(p[start : i + 1] >= threshold))
        need = min(votes_required, i - start + 1)
        if not state:
            if votes >= need and (i - start + 1) >= min(window, votes_required):
                # require full window once enough history exists
                if (i + 1) >= window and votes >= votes_required:
                    state = True
                    changes += 1
        else:
            if p[i] < reset_threshold:
                # stay triggered unless we want optional reset; for completion
                # detectors we keep sticky completion once triggered.
                pass
        triggered[i] = state
    return {
        "triggered": triggered,
        "n_state_changes": int(changes),
        "first_trigger_index": int(np.argmax(triggered)) if triggered.any() else None,
        "threshold": float(threshold),
        "reset_threshold": float(reset_threshold),
        "window": int(window),
        "votes_required": int(votes_required),
    }


def evaluate_proxy_temporal(
    probs: np.ndarray,
    *,
    threshold: float,
    proxy_positive_onset: int,
    n_frames: int | None = None,
    window: int = 5,
    votes_required: int = 4,
) -> dict[str, Any]:
    """Proxy temporal metrics (not genuine task-success metrics)."""
    n = int(n_frames if n_frames is not None else len(probs))
    p = np.asarray(probs, dtype=float)[:n]
    hyst = hysteresis_triggers(p, threshold, window=window, votes_required=votes_required)
    first = hyst["first_trigger_index"]
    early_false = first is not None and first < int(proxy_positive_onset)
    in_terminal = first is not None and first >= int(proxy_positive_onset)
    latency = None
    if in_terminal:
        latency = int(first - proxy_positive_onset)
    # flicker: rising edges in raw thresholded signal
    raw = (p >= threshold).astype(int)
    flicker = int(np.sum(np.diff(raw) != 0))
    return {
        "trigger_occurred": first is not None,
        "trigger_in_proxy_positive_region": bool(in_terminal),
        "false_early_trigger": bool(early_false),
        "detection_latency_frames": latency,
        "first_trigger_index": first,
        "proxy_positive_onset": int(proxy_positive_onset),
        "probability_mean": float(np.mean(p)) if len(p) else float("nan"),
        "probability_std": float(np.std(p)) if len(p) else float("nan"),
        "flicker_events": flicker,
        "n_state_changes": hyst["n_state_changes"],
        "triggered": hyst["triggered"],
    }


def aggregate_episode_temporal(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "n_episodes": 0,
            "fraction_detected": float("nan"),
            "false_early_trigger_rate": float("nan"),
            "median_detection_latency_frames": float("nan"),
        }
    n = len(results)
    detected = sum(1 for r in results if r.get("trigger_in_proxy_positive_region"))
    early = sum(1 for r in results if r.get("false_early_trigger"))
    lats = [r["detection_latency_frames"] for r in results if r.get("detection_latency_frames") is not None]
    active_term = sum(1 for r in results if r.get("trigger_active_in_terminal"))
    return {
        "n_episodes": n,
        "fraction_detected": float(detected / n),
        "fraction_trigger_active_in_terminal": float(active_term / n),
        "false_early_trigger_rate": float(early / n),
        "median_detection_latency_frames": float(np.median(lats)) if lats else float("nan"),
        "mean_flicker_events": float(np.mean([r.get("flicker_events", 0) for r in results])),
        "note": "Proxy temporal metrics only : not genuine task-success metrics.",
    }
