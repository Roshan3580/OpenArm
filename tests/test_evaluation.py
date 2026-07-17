"""Tests for Task 4 policy evaluation design and success-detector helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from openarm_pipeline.evaluation.metrics import (
    binary_success_rate,
    bootstrap_metric_ci,
    expected_calibration_error,
    failure_taxonomy_rates,
    precision_recall_f1,
    specificity,
    success_by_slice,
    wilson_interval,
)
from openarm_pipeline.evaluation.rollout_protocol import (
    generate_rollout_matrix,
    validate_rollout_protocol,
    validate_rollout_result_record,
)
from openarm_pipeline.evaluation.success_detector import (
    PROHIBITED_FEATURE_NAMES,
    DetectorConfig,
    assert_no_episode_leakage,
    build_proxy_labels_for_episode,
    episode_grouped_split,
    extract_features,
    select_threshold_on_validation,
)
from openarm_pipeline.evaluation.temporal_detection import (
    evaluate_proxy_temporal,
    hysteresis_triggers,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_interval_known_values():
    w = wilson_interval(80, 100)
    assert 0.7 < w["low"] < w["p"] < w["high"] < 1.0
    assert w["p"] == pytest.approx(0.8)


def test_success_rate_aggregation():
    r = binary_success_rate(["success", "failure", "success", "success"])
    assert r["successes"] == 3
    assert r["rate"] == pytest.approx(0.75)


def test_failure_taxonomy_aggregation():
    r = failure_taxonomy_rates([["object_slip"], ["collision", "object_slip"], []])
    assert r["counts"]["object_slip"] == 2
    assert r["rates"]["collision"] == pytest.approx(1 / 3)


def test_success_by_slice_worst():
    outcomes = ["success"] * 9 + ["failure"] + ["success"] * 2 + ["failure"] * 3
    slices = ["nominal_held_out"] * 10 + ["occlusion_distractor"] * 5
    s = success_by_slice(outcomes, slices)
    assert s["worst_slice"] == "occlusion_distractor"


def test_grouped_split_no_leakage_and_deterministic():
    eps = list(range(50))
    a = episode_grouped_split(eps, seed=42)
    b = episode_grouped_split(eps, seed=42)
    assert a == b
    assert_no_episode_leakage(a)
    c = episode_grouped_split(eps, seed=7)
    assert a["test"] != c["test"]


def test_proxy_label_construction_excludes_middle():
    fi = np.arange(100)
    ts = fi / 30.0
    rows = build_proxy_labels_for_episode(3, fi, ts, cfg=DetectorConfig())
    labels = {r["proxy_label"] for r in rows}
    assert labels == {0, 1}
    for r in rows:
        if r["proxy_label"] == 0:
            assert r["global_pos_in_episode"] < 40
        else:
            assert r["global_pos_in_episode"] >= 85


def test_prohibited_metadata_not_in_features():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    feat = extract_features(rgb, DetectorConfig(), mode="full")
    assert feat.ndim == 1 and feat.size > 10
    for name in PROHIBITED_FEATURE_NAMES:
        assert name not in {"hsv_0", "hog_block"}


def test_feature_shape_consistency():
    cfg = DetectorConfig()
    a = np.zeros((96, 96, 3), dtype=np.uint8)
    b = np.full((96, 96, 3), 200, dtype=np.uint8)
    assert extract_features(a, cfg).shape == extract_features(b, cfg).shape


def test_validation_only_threshold_selection():
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    thr = select_threshold_on_validation(y, probs, metric="f1")
    assert 0.05 <= thr <= 0.95
    pred = (probs >= thr).astype(int)
    assert precision_recall_f1(y, pred)["f1"] >= 0.5


def test_specificity_and_calibration():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    assert specificity(y, pred) == pytest.approx(0.5)
    ece = expected_calibration_error(y, np.array([0.1, 0.9, 0.8, 0.7]), n_bins=2)
    assert ece >= 0.0


def test_deterministic_bootstrap_intervals():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    a = bootstrap_metric_ci(y, pred, lambda yt, yp: float(np.mean(yt == yp)), groups=groups, seed=1)
    b = bootstrap_metric_ci(y, pred, lambda yt, yp: float(np.mean(yt == yp)), groups=groups, seed=1)
    assert a == b
    assert a["low"] <= a["point"] <= a["high"]


def test_four_of_five_hysteresis():
    probs = np.array([0.1, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2])
    h = hysteresis_triggers(probs, 0.5, window=5, votes_required=4)
    assert h["first_trigger_index"] == 5
    assert h["triggered"][5]
    assert not h["triggered"][3]


def test_false_early_trigger_and_latency():
    probs = np.zeros(20)
    probs[2:7] = 0.9
    r = evaluate_proxy_temporal(probs, threshold=0.5, proxy_positive_onset=15, window=5, votes_required=4)
    assert r["false_early_trigger"] is True
    probs2 = np.zeros(20)
    probs2[15:20] = 0.95
    r2 = evaluate_proxy_temporal(probs2, threshold=0.5, proxy_positive_onset=15, window=5, votes_required=4)
    assert r2["trigger_in_proxy_positive_region"] is True
    assert r2["detection_latency_frames"] == 3


def test_rollout_count_and_slice_validation():
    m = generate_rollout_matrix()
    report = validate_rollout_protocol(m)
    assert report["ok"] is True
    assert report["n_rollouts"] == 100
    assert report["slice_counts"]["nominal_held_out"] == 40


def test_duplicate_rollout_id_rejected():
    m = generate_rollout_matrix()
    m["rollouts"][1]["rollout_id"] = m["rollouts"][0]["rollout_id"]
    report = validate_rollout_protocol(m)
    assert report["ok"] is False
    assert any(e["code"] == "duplicate_rollout_id" for e in report["errors"])


def test_simulator_gt_leakage_rejected():
    m = generate_rollout_matrix()
    m["rollouts"][0]["policy_inputs"] = {"simulator_success": True}
    report = validate_rollout_protocol(m)
    assert any(e["code"] == "simulator_gt_leakage" for e in report["errors"])


def test_rollout_result_schema_validation():
    schema_path = ROOT / "tasks/task_04_policy_evaluation/rollout_result_schema.json"
    schema = json.loads(schema_path.read_text())
    assert "ground_truth_success_source" in schema["required"]
    good = {
        "rollout_id": "R000",
        "policy_checkpoint_id": "none",
        "dataset_repo_id": "lerobot/svla_so100_pickplace",
        "dataset_revision": "728583b5eaf9e739a7f119e2def466fa1d552402",
        "curation_view": "conservative",
        "seed": 42000,
        "evaluation_slice": "nominal_held_out",
        "observation_config": "wrist_plus_top",
        "task_outcome": "success",
        "ground_truth_success_source": "simulator_state",
    }
    assert validate_rollout_result_record(good)["ok"] is True
    bad = dict(good)
    bad["ground_truth_success_source"] = "wrist_success_detector_under_test"
    assert validate_rollout_result_record(bad)["ok"] is False


def test_checked_in_protocol_file_valid():
    path = ROOT / "tasks/task_04_policy_evaluation/rollout_protocol.yaml"
    protocol = yaml.safe_load(path.read_text())
    assert validate_rollout_protocol(protocol)["ok"] is True
