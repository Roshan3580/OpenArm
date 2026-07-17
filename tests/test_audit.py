"""Synthetic unit tests for Task 1 audit helpers (no HF downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from openarm_pipeline.audit.egocentric import (
    exposure_stats,
    frame_mse,
    is_exact_or_near_duplicate,
    laplacian_variance,
    score_frame,
)
from openarm_pipeline.audit.teleop import (
    describe_lengths,
    detect_discontinuities,
    detect_near_stuck,
    nan_inf_counts,
)
from openarm_pipeline.data.alignment import check_episode_alignment, infer_dt
from openarm_pipeline.data.lerobot_adapter import classify_camera_viewpoint, discover_keys


def test_describe_lengths_basic():
    stats = describe_lengths(np.array([10, 20, 30, 40]))
    assert stats["count"] == 4
    assert stats["min"] == 10
    assert stats["max"] == 40
    assert stats["median"] == 25.0


def test_describe_lengths_empty():
    stats = describe_lengths(np.array([]))
    assert stats["count"] == 0
    assert stats["mean"] is None


def test_nan_inf_counts():
    arr = np.array([[1.0, np.nan], [np.inf, -np.inf], [0.0, 2.0]])
    c = nan_inf_counts(arr)
    assert c["nan"] == 1
    assert c["inf"] == 2
    assert c["total_elements"] == 6


def test_timestamp_discontinuities():
    ts = np.array([0.0, 0.02, 0.04, 0.20, 0.22])  # gap between 0.04 and 0.20
    fi = np.arange(len(ts))
    report = check_episode_alignment(
        timestamps=ts,
        frame_index=fi,
        state_len=5,
        action_len=5,
        expected_dt=0.02,
        gap_factor=1.5,
    )
    assert report["n_timestamp_gaps"] == 1
    assert report["non_monotonic"] is False
    assert report["length_mismatch"] is False


def test_non_monotonic_and_duplicates():
    ts = np.array([0.0, 0.02, 0.02, 0.01])
    fi = np.array([0, 1, 2, 4])  # frame gap
    report = check_episode_alignment(
        timestamps=ts,
        frame_index=fi,
        state_len=4,
        action_len=3,
        expected_dt=0.02,
    )
    assert report["n_duplicate_timestamps"] >= 1
    assert report["non_monotonic"] is True
    assert report["n_frame_index_gaps"] >= 1
    assert report["length_mismatch"] is True


def test_infer_dt_from_fps():
    assert infer_dt(np.array([]), fps=50) == pytest.approx(0.02)


def test_robust_discontinuity_detection():
    # Smooth increments then one huge jump
    y = np.cumsum(np.ones((100, 1)) * 0.01, axis=0)
    y[50] = y[49] + 5.0
    r = detect_discontinuities(y, mad_k=8.0, abs_floor=0.05, range_frac=0.01)
    assert r["total_flags"] >= 1
    assert r["flag_rate"] > 0
    # Tiny noise below abs_floor should not flag when MAD is small
    z = np.cumsum(np.ones((100, 1)) * 0.001, axis=0)
    r2 = detect_discontinuities(z, mad_k=8.0, abs_floor=0.05, range_frac=0.01)
    assert r2["total_flags"] == 0


def test_video_tabular_frame_alignment_and_timing():
    from openarm_pipeline.audit.video_alignment import episode_video_alignment_row

    ts = np.arange(10, dtype=np.float64) / 30.0
    ok = episode_video_alignment_row(
        episode_index=0,
        tabular_length=10,
        tabular_timestamps=ts,
        fps=30.0,
        from_timestamp=0.0,
        to_timestamp=10 / 30.0,
        container_fps=30.0,
        timing_tol_s=1.0 / 30.0,
        material_frame_mismatch=2,
        material_duration_mismatch_s=0.1,
    )
    assert ok["timing_within_tolerance"] is True
    assert ok["material_frame_count_mismatch"] is False

    # Acceptable small timing tolerance
    ts_jitter = ts.copy()
    ts_jitter[-1] += 0.5 / 30.0
    mild = episode_video_alignment_row(
        0, 10, ts_jitter, 30.0, 0.0, 10 / 30.0, 30.0, timing_tol_s=1.0 / 30.0
    )
    assert mild["timing_within_tolerance"] is True

    # Material mismatch
    bad = episode_video_alignment_row(
        0,
        10,
        ts,
        30.0,
        from_timestamp=0.0,
        to_timestamp=20 / 30.0,  # estimates ~20 frames vs 10 tabular
        container_fps=30.0,
        material_frame_mismatch=2,
    )
    assert bad["material_frame_count_mismatch"] is True


def test_exact_and_near_duplicate_adjacent():
    from openarm_pipeline.audit.egocentric import classify_frame_duplicate

    a = np.full((16, 16, 3), 100, dtype=np.uint8)
    b = a.copy()
    exact = classify_frame_duplicate(a, b, near_lossless_mse=1.0, near_mse=25.0)
    assert exact["exact_duplicate"] is True
    assert exact["near_lossless_duplicate"] is False
    assert exact["near_duplicate"] is False
    # near-lossless: change enough pixels for 0 < gray-MSE <= 1
    e = a.copy()
    e[0:2, 0:2, :] = 101
    nl = classify_frame_duplicate(a, e, near_lossless_mse=1.0, near_mse=25.0)
    assert nl["mse"] > 0
    assert nl["mse"] <= 1.0
    assert nl["exact_duplicate"] is False
    assert nl["near_lossless_duplicate"] is True
    assert nl["near_duplicate"] is False
    d = a.copy()
    d[:] = 110
    near = classify_frame_duplicate(a, d, near_lossless_mse=1.0, near_mse=200.0)
    assert near["exact_duplicate"] is False
    assert near["near_lossless_duplicate"] is False
    assert near["near_duplicate"] is True
    assert frame_mse(a, d) > frame_mse(a, b)


def test_contiguous_windows_cover_every_episode():
    from openarm_pipeline.audit.video_alignment import plan_episode_windows

    lengths = {i: 100 + i for i in range(10)}
    plan = plan_episode_windows(lengths, n_windows=3, window_size=10, seed=42, min_total_frames=200)
    assert plan["covers_every_episode"] is True
    assert plan["n_episodes_covered"] == 10
    assert plan["total_frames_planned"] >= 200
    # windows are contiguous
    for ep, wins in plan["windows"].items():
        for s, e in wins:
            assert e > s
            assert e - s >= 1


def test_multi_dataset_artifact_isolation(tmp_path):
    from openarm_pipeline.data.lerobot_adapter import dataset_slug, save_json

    a = tmp_path / dataset_slug("lerobot/aloha_sim_insertion_human")
    b = tmp_path / dataset_slug("lerobot/svla_so100_pickplace")
    a.mkdir()
    b.mkdir()
    save_json({"dataset": "aloha"}, a / "audit_summary.json")
    save_json({"dataset": "svla"}, b / "audit_summary.json")
    assert (a / "audit_summary.json").exists()
    assert (b / "audit_summary.json").exists()
    assert a.name != b.name


def test_bimodal_gripper_separate_from_joints():
    from openarm_pipeline.audit.gripper import analyze_gripper_channel, identify_gripper_dims

    names = ["joint1", "joint2", "main_gripper"]
    assert identify_gripper_dims(names, 3) == [2]
    # Construct open/closed-like values
    closed = np.zeros(500)
    opened = np.full(500, 30.0)
    vals = np.concatenate([closed, opened])
    analysis = analyze_gripper_channel(vals, name="main_gripper")
    assert analysis["bimodal_or_open_closed_concentrated"] is True
    assert analysis["recommendation"] == "diagnostic_only_separate_from_joint_outlier_filters"


def test_near_stuck_channel():
    arr = np.zeros((50, 2))
    arr[:, 0] = 1.234567  # constant
    arr[:, 1] = np.linspace(0, 1, 50)
    stuck = detect_near_stuck(arr, std_thresh=1e-5, range_thresh=1e-4)
    assert stuck[0]["near_stuck"] is True
    assert stuck[1]["near_stuck"] is False


def test_blur_metric_sharp_vs_blurred():
    rng = np.random.default_rng(0)
    # High-frequency checkerboard = sharp
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[::2, ::2] = 255
    sharp[1::2, 1::2] = 255
    # Heavy box blur approximation
    blurred = sharp.astype(np.float64)
    for _ in range(8):
        padded = np.pad(blurred, 1, mode="edge")
        blurred = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[1:-1, 1:-1]
        ) / 5.0
    blurred = blurred.astype(np.uint8)
    assert laplacian_variance(sharp) > laplacian_variance(blurred)


def test_exposure_metric_dark_normal_bright():
    dark = np.full((32, 32, 3), 5, dtype=np.uint8)
    normal = np.full((32, 32, 3), 120, dtype=np.uint8)
    bright = np.full((32, 32, 3), 250, dtype=np.uint8)
    assert exposure_stats(dark)["mean_luma"] < exposure_stats(normal)["mean_luma"]
    assert exposure_stats(bright)["mean_luma"] > exposure_stats(normal)["mean_luma"]
    assert exposure_stats(dark)["underexposed_pixel_frac"] > 0.5
    assert exposure_stats(bright)["overexposed_pixel_frac"] > 0.5


def test_duplicate_frame_detection():
    from openarm_pipeline.audit.egocentric import classify_frame_duplicate

    a = np.full((16, 16, 3), 100, dtype=np.uint8)
    b = a.copy()
    c = a.copy()
    c[0, 0] = 200
    d = classify_frame_duplicate(a, b, near_lossless_mse=1.0, near_mse=25.0)
    assert d["exact_duplicate"] is True
    assert frame_mse(a, c) > frame_mse(a, b)


def test_within_episode_adjacent_pair_count():
    from openarm_pipeline.audit.egocentric import within_episode_adjacent_pair_count

    assert within_episode_adjacent_pair_count(19631, 50) == 19581
    assert within_episode_adjacent_pair_count(100, 1) == 99


def test_should_compare_adjacent_frames_refuses_cross_episode():
    from openarm_pipeline.audit.egocentric import should_compare_adjacent_frames

    # Same episode, consecutive frames — OK
    assert should_compare_adjacent_frames(0, 10, 0, 11) is True
    # Cross-episode boundary with identical-looking frame indices reset — refuse
    assert should_compare_adjacent_frames(0, 399, 1, 0) is False
    assert should_compare_adjacent_frames(0, 100, 1, 101) is False
    # Non-consecutive within episode — refuse (unless diagnostic mode disables check)
    assert should_compare_adjacent_frames(2, 5, 2, 7) is False
    assert (
        should_compare_adjacent_frames(
            2, 5, 2, 7, require_consecutive_frame_index=False
        )
        is True
    )


def test_cross_episode_boundary_identical_frames_not_compared():
    """Regression: identical pixels at episode boundary must not count as duplicates."""
    from openarm_pipeline.audit.egocentric import (
        classify_frame_duplicate,
        should_compare_adjacent_frames,
    )

    frame_a = np.full((8, 8, 3), 42, dtype=np.uint8)
    frame_b = frame_a.copy()  # identical-looking across boundary
    assert classify_frame_duplicate(frame_a, frame_b)["exact_duplicate"] is True
    # Stream order: last of ep0 then first of ep1 — must not compare
    assert should_compare_adjacent_frames(0, 199, 1, 0) is False
    # Simulate accounting: only count when should_compare is True
    pairs = [
        (0, 198, 0, 199, frame_a, frame_a),
        (0, 199, 1, 0, frame_a, frame_b),  # boundary — excluded
        (1, 0, 1, 1, frame_b, frame_b),
    ]
    n_compared = 0
    n_exact = 0
    for ep_a, fi_a, ep_b, fi_b, fa, fb in pairs:
        if not should_compare_adjacent_frames(ep_a, fi_a, ep_b, fi_b):
            continue
        n_compared += 1
        if classify_frame_duplicate(fa, fb)["exact_duplicate"]:
            n_exact += 1
    assert n_compared == 2
    assert n_exact == 2  # both within-episode pairs are exact; boundary excluded


def test_score_frame_flags_dark():
    dark = np.zeros((24, 24, 3), dtype=np.uint8)
    s = score_frame(dark, {"egocentric": {}})
    assert s["flags"]["underexposed"] is True
    assert s["flags"]["low_entropy_possible_occlusion"] is True


def test_empty_and_short_edge_cases():
    assert describe_lengths(np.array([])).get("count") == 0
    r = detect_discontinuities(np.zeros((1, 3)))
    assert r["total_flags"] == 0
    r0 = detect_discontinuities(np.zeros((0, 0)))
    assert r0["total_flags"] == 0
    stuck = detect_near_stuck(np.zeros((0, 0)))
    assert stuck == []


def test_discover_and_classify_cameras():
    features = {
        "observation.state": {"dtype": "float32", "shape": [14]},
        "action": {"dtype": "float32", "shape": [14]},
        "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    d = discover_keys(features)
    assert "observation.state" in d["state_keys"]
    assert "action" in d["action_keys"]
    assert "observation.images.top" in d["image_video_features"]
    top = classify_camera_viewpoint("observation.images.top", features["observation.images.top"])
    wrist = classify_camera_viewpoint("observation.images.wrist", features["observation.images.wrist"])
    assert top.viewpoint == "verified_external"
    assert wrist.viewpoint == "verified_egocentric"
    amb = classify_camera_viewpoint("observation.images.cam0", {"dtype": "video", "shape": [48, 64, 3]})
    assert amb.viewpoint == "ambiguous"
