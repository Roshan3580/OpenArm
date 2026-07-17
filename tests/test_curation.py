"""Tests for Task 3 curation (no HF downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openarm_pipeline.audit.egocentric import classify_frame_duplicate
from openarm_pipeline.curation.curated_view import CuratedView, build_training_windows, config_hash
from openarm_pipeline.curation.egocentric import compute_visual_timestep_flags, sustained_runs
from openarm_pipeline.curation.teleop import (
    discontinuity_timestep_flags,
    smooth_continuous_joints,
    validate_episode_hard,
)


def _cfg(**overrides):
    base = {
        "horizon": 4,
        "episode_validation": {
            "reject_nan_inf": True,
            "reject_non_monotonic_timestamps": True,
            "reject_duplicate_timestamps": True,
            "reject_frame_index_gaps": True,
            "reject_length_mismatch": True,
            "reject_missing_wrist_video": True,
            "reject_material_video_mismatch": True,
            "min_episode_frames": 6,
        },
        "smoothing": {
            "enabled": True,
            "method": "savgol",
            "window_length": 5,
            "polyorder": 2,
            "exclude_gripper": True,
            "max_rmse_over_std": 0.5,
        },
        "discontinuity": {
            "mad_k": 8.0,
            "abs_floor": 1.0,
            "range_frac": 0.01,
            "severe_mad_k": 20.0,
            "severe_abs_floor": 8.0,
            "exclude_ordinary_from_windows": False,
            "exclude_severe_from_strict_windows": True,
        },
        "egocentric": {
            "near_lossless_mse_threshold": 1.0,
            "near_duplicate_mse_threshold": 25.0,
            "soft_sharpness_percentile": 5.0,
            "underexposure_mean_threshold": 40.0,
            "overexposure_mean_threshold": 220.0,
            "overexposure_sat_frac": 0.15,
            "underexposure_sat_frac": 0.15,
            "low_entropy_threshold": 3.5,
            "sustained_overexposure_run": 3,
            "sustained_frozen_run": 3,
            "motion_percentile": 50.0,
            "frozen_uses_near_lossless": True,
        },
    }
    base.update(overrides)
    return base


def test_hard_episode_rejection_nan():
    n = 20
    state = np.zeros((n, 3))
    state[5, 0] = np.nan
    r = validate_episode_hard(
        0, np.arange(n) / 30.0, np.arange(n), state, np.zeros((n, 3)), 30.0, _cfg()
    )
    assert r["accepted"] is False
    assert "state_nan_inf" in r["reasons"]


def test_min_episode_length():
    n = 4
    r = validate_episode_hard(
        0, np.arange(n) / 30.0, np.arange(n), np.zeros((n, 2)), np.zeros((n, 2)), 30.0, _cfg()
    )
    assert r["accepted"] is False
    assert "too_short_min_episode_frames" in r["reasons"]


def test_timestamp_and_frame_discontinuity():
    n = 20
    ts = np.arange(n) / 30.0
    ts[10] = ts[9] - 0.01
    fi = np.arange(n)
    fi[12] = 14
    r = validate_episode_hard(
        0, ts, fi, np.zeros((n, 2)), np.zeros((n, 2)), 30.0, _cfg()
    )
    assert "non_monotonic_timestamps" in r["reasons"]
    assert "frame_index_discontinuity" in r["reasons"]


def test_smoothing_excludes_gripper_and_isolates_episodes():
    names = ["j1", "j2", "main_gripper"]
    # noisy joints + step gripper
    rng = np.random.default_rng(0)
    state = np.cumsum(rng.normal(0, 0.2, size=(40, 3)), axis=0)
    state[:, 2] = 0.0
    state[20:, 2] = 30.0  # gripper mode transition
    sm, metrics = smooth_continuous_joints(state, names, _cfg())
    assert sm.shape == state.shape
    assert np.allclose(sm[:, 2], state[:, 2])
    assert any(p["name"] == "main_gripper" and p.get("note") == "gripper_excluded" for p in metrics["per_dim"])
    # joint dims should change somewhat
    assert not np.allclose(sm[:, 0], state[:, 0])


def test_exact_vs_near_lossless_mutually_exclusive():
    a = np.full((8, 8, 3), 50, dtype=np.uint8)
    b = a.copy()
    c = a.copy()
    c[0:3, 0:3, :] = 51
    d = a.copy()
    d[:] = 60
    e = classify_frame_duplicate(a, b)
    nl = classify_frame_duplicate(a, c)
    near = classify_frame_duplicate(a, d, near_lossless_mse=1.0, near_mse=200.0)
    assert e["exact_duplicate"] and not e["near_lossless_duplicate"] and not e["near_duplicate"]
    assert nl["mse"] > 0 and nl["mse"] <= 1.0
    assert nl["near_lossless_duplicate"] and not nl["exact_duplicate"] and not nl["near_duplicate"]
    assert near["near_duplicate"] and not near["exact_duplicate"] and not near["near_lossless_duplicate"]


def test_frozen_while_moving_and_stationary_dup_retention():
    n = 12
    frames = [np.full((16, 16, 3), 100, dtype=np.uint8) for _ in range(n)]
    sd = np.zeros(n)
    sd[:3] = 0.0
    sd[3:] = 10.0  # clearly above median of mixed series
    ad = sd.copy()
    cfg = _cfg()
    cfg["egocentric"]["motion_percentile"] = 40.0
    vis = compute_visual_timestep_flags(frames, sd, ad, cfg, sharpness_soft_threshold=1.0)
    assert bool(vis["exact_duplicate"][5]) is True
    assert bool(vis["robot_moving"][5]) is True
    assert bool(vis["frozen_while_moving_candidate"][5]) is True
    assert bool(vis["robot_moving"][1]) is False
    assert bool(vis["frozen_while_moving_candidate"][1]) is False
    assert bool(vis["visual_hard_valid"][1]) is True


def test_sustained_run_detection():
    f = np.array([0, 1, 1, 1, 0, 1, 0], dtype=bool)
    s = sustained_runs(f, 3)
    assert s.tolist() == [False, True, True, True, False, False, False]


def test_training_windows_policies_and_alignment():
    n = 20
    rows = []
    for i in range(n):
        rows.append(
            {
                "episode_index": 0,
                "frame_index": i,
                "timestamp": i / 30.0,
                "global_index": i,
                "hard_valid": True,
                "soft_exclude_strict": i in (10, 11, 12),
            }
        )
    # second episode
    for i in range(10):
        rows.append(
            {
                "episode_index": 1,
                "frame_index": i,
                "timestamp": i / 30.0,
                "global_index": 100 + i,
                "hard_valid": True,
                "soft_exclude_strict": False,
            }
        )
    ts = pd.DataFrame(rows)
    cons = build_training_windows(ts, horizon=4, stride=1, policy="conservative")
    strict = build_training_windows(ts, horizon=4, stride=1, policy="strict")
    assert len(cons) > 0
    cons_keys = set(zip(cons.episode_index, cons.start_frame_index))
    strict_keys = set(zip(strict.episode_index, strict.start_frame_index))
    assert strict_keys.issubset(cons_keys)
    # no cross-episode: end within episode
    for _, w in cons.iterrows():
        assert w["end_frame_index_exclusive"] - w["start_frame_index"] == 4


def test_discontinuity_flags_diagnostic():
    x = np.cumsum(np.ones((30, 2)) * 0.1, axis=0)
    x[15] = x[14] + 20.0
    d = discontinuity_timestep_flags(x, x, _cfg())
    assert d["timestep_any_discontinuity"].any() or d["timestep_severe_discontinuity"].any()
    assert not d["window_exclude_conservative"].any()  # ordinary not excluded


def test_manifest_and_deterministic_hash():
    c1 = {"a": 1, "b": [2, 3]}
    c2 = {"b": [2, 3], "a": 1}
    assert config_hash(c1) == config_hash(c2)


def test_curated_view_loader(tmp_path):
    # minimal fake curated view
    root = tmp_path / "view"
    root.mkdir()
    episodes = pd.DataFrame([{"episode_index": 0, "n_frames": 8, "accepted": True, "reasons": ""}])
    timesteps = []
    for i in range(8):
        timesteps.append(
            {
                "episode_index": 0,
                "frame_index": i,
                "timestamp": i / 30.0,
                "global_index": i,
                "state": [float(i), 0.0, 0.0],
                "state_smoothed": [float(i), 0.0, 0.0],
                "action": [0.0, 0.0, 0.0],
                "hard_valid": True,
                "soft_exclude_strict": False,
                "wrist_from_timestamp": 0.0,
            }
        )
    timesteps = pd.DataFrame(timesteps)
    windows = build_training_windows(timesteps, horizon=4, stride=2, policy="conservative")
    windows2 = build_training_windows(timesteps, horizon=4, stride=2, policy="strict")
    windows = pd.concat([windows, windows2], ignore_index=True)
    episodes.to_parquet(root / "episodes.parquet", index=False)
    timesteps.to_parquet(root / "timesteps.parquet", index=False)
    windows.to_parquet(root / "training_windows.parquet", index=False)
    manifest = {
        "source_repo_id": "test/repo",
        "source_revision": "abc",
        "camera_keys": {"wrist": "observation.images.wrist", "top": "observation.images.top"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    view = CuratedView.load(root)
    got = list(view.iter_windows("conservative"))
    assert len(got) > 0
    w0 = got[0]
    assert w0["state"].shape[0] == 4
    assert w0["state"].shape == w0["state_smoothed"].shape


def test_synthetic_corruption_integration(tmp_path):
    """Synthetic fixture proving filters activate without real HF data."""
    cfg = _cfg()
    cfg["episode_validation"]["min_episode_frames"] = 8
    cfg["horizon"] = 4

    # Episode 0: short -> reject
    # Episode 1: NaN -> reject
    # Episode 2: timestamp tear -> reject
    # Episode 3: good with frozen-while-moving sustained + stationary dups + gripper transition
    results = []

    # short
    r = validate_episode_hard(
        0, np.arange(5) / 30.0, np.arange(5), np.zeros((5, 3)), np.zeros((5, 3)), 30.0, cfg
    )
    results.append(("short", r))
    assert not r["accepted"]

    # nan
    st = np.zeros((20, 3))
    st[3, 1] = np.nan
    r = validate_episode_hard(1, np.arange(20) / 30.0, np.arange(20), st, np.zeros((20, 3)), 30.0, cfg)
    results.append(("nan", r))
    assert "state_nan_inf" in r["reasons"]

    # timestamp discontinuity
    ts = np.arange(20) / 30.0
    ts[8] = ts[7] - 0.05
    r = validate_episode_hard(2, ts, np.arange(20), np.zeros((20, 3)), np.zeros((20, 3)), 30.0, cfg)
    results.append(("ts", r))
    assert "non_monotonic_timestamps" in r["reasons"]

    # good episode visual behaviors
    n = 24
    names = ["j1", "j2", "main_gripper"]
    state = np.zeros((n, 3))
    state[:, 0] = np.linspace(0, 10, n)
    state[12:, 2] = 30.0  # gripper transition
    sm, met = smooth_continuous_joints(state, names, cfg)
    assert np.allclose(sm[:, 2], state[:, 2])

    frames = []
    for i in range(n):
        if i == 5:
            frames.append(None)  # missing wrist
        else:
            frames.append(np.full((12, 12, 3), 80, dtype=np.uint8))
    # make frames 15-20 identical while motion high
    sd = np.linspace(0, 5, n)
    ad = sd.copy()
    vis = compute_visual_timestep_flags(frames, sd, ad, cfg, sharpness_soft_threshold=1.0)
    assert vis["decode_failure"][5]
    assert vis["hard_invalid"][5]
    # stationary early: frame 1 vs 0 identical but low motion at start : keep
    assert vis["visual_soft_valid"][1] or vis["exact_duplicate"][1]

    # Build windows: mark hard invalid at missing frame
    rows = []
    for i in range(n):
        rows.append(
            {
                "episode_index": 3,
                "frame_index": i,
                "timestamp": i / 30.0,
                "global_index": i,
                "hard_valid": not vis["hard_invalid"][i],
                "soft_exclude_strict": bool(vis["soft_exclude_strict"][i]),
            }
        )
    tsdf = pd.DataFrame(rows)
    cons = build_training_windows(tsdf, horizon=4, stride=1, policy="conservative")
    # no window may include frame 5
    for _, w in cons.iterrows():
        assert not (w.start_frame_index <= 5 < w.end_frame_index_exclusive)

    assert len(results) >= 3
    assert any(not r[1]["accepted"] for r in results)
