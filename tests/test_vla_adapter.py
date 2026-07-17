"""Tests for Task 5 OpenVLA data adapter (no model download)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openarm_pipeline.vla.action_encoding import (
    ActionNormalizer,
    compute_delta_actions,
    decode_actions,
    encode_actions,
    masked_action_loss_weights,
)
from openarm_pipeline.vla.collator import collate_openvla_batch
from openarm_pipeline.vla.config import OpenVLAAdapterConfig
from openarm_pipeline.vla.dataset_adapter import (
    episode_grouped_split,
    frames_covered_by_policy,
    select_export_rows,
)
from openarm_pipeline.vla.preprocessing import (
    UNSAFE_AUGMENTATIONS,
    assert_augmentation_safe,
    preprocess_wrist_frame,
    safe_augmentation,
)
from openarm_pipeline.vla.validation import (
    assert_no_episode_leakage,
    validate_batch_shapes,
    validate_export_manifest,
    validate_roundtrip,
)

ROOT = Path(__file__).resolve().parents[1]


def test_episode_grouped_split_no_leakage_deterministic():
    a = episode_grouped_split(range(50), seed=42)
    b = episode_grouped_split(range(50), seed=42)
    assert a == b
    assert_no_episode_leakage(a)


def test_train_only_normalization_frozen():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 6))
    val = rng.normal(loc=5.0, size=(50, 6))
    norm = ActionNormalizer.fit(train)
    enc_val, _, _ = norm.transform(val)
    # using train quantiles: val mean shifted => many clips, but transform uses frozen stats
    enc_train2, _, _ = ActionNormalizer.fit(train).transform(train)
    assert enc_train2.shape[1] == 7
    assert np.allclose(norm.q_low, ActionNormalizer.fit(train).q_low)


def test_absolute_and_delta_encoding():
    a = np.array([[0.0, 0, 0, 0, 0, 1.0], [1.0, 0, 0, 0, 0, 2.0], [2.0, 0, 0, 0, 0, 2.0]])
    ep = np.array([0, 0, 0])
    fi = np.array([0, 1, 2])
    d = compute_delta_actions(a, episode_index=ep, frame_index=fi, gripper_index=5)
    assert d[0, 0] == 0.0
    assert d[1, 0] == pytest.approx(1.0)
    assert d[1, 5] == pytest.approx(2.0)  # gripper absolute


def test_no_cross_episode_delta():
    a = np.array([[10.0, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]])
    ep = np.array([0, 1])
    fi = np.array([5, 0])
    d = compute_delta_actions(a, episode_index=ep, frame_index=fi)
    assert d[1, 0] == 0.0


def test_zero_range_normalization():
    a = np.zeros((20, 6))
    a[:, 0] = 3.0
    norm = ActionNormalizer.fit(a)
    enc, mask, _ = encode_actions(a, norm)
    assert enc.shape == (20, 7)
    assert mask[6] == 0.0
    dec = decode_actions(enc, norm)
    assert np.allclose(dec[:, 0], 3.0, atol=1e-5)


def test_clipping_and_roundtrip():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(100, 6))
    norm = ActionNormalizer.fit(a)
    enc, _, stats = encode_actions(a, norm)
    assert "clip_fraction" in stats
    dec = decode_actions(enc, norm)
    rt = validate_roundtrip(a, dec, atol=1.0)
    assert rt["mean_abs_error"] < 1.0


def test_padded_dim_excluded_from_loss_metadata():
    w = masked_action_loss_weights(np.array([1, 1, 1, 1, 1, 1, 0], dtype=float))
    assert w[-1] == 0.0
    assert w[:6].sum() == 6.0


def test_view_selection_config():
    cfg = OpenVLAAdapterConfig(primary_view="wrist")
    assert cfg.primary_view == "wrist"
    assert cfg.wrist_key.endswith("wrist")


def test_rgb_and_deterministic_preprocess():
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue channel in BGR
    a = preprocess_wrist_frame(bgr, size=224, assume_bgr=True)
    b = preprocess_wrist_frame(bgr, size=224, assume_bgr=True)
    assert a.shape == (224, 224, 3)
    assert np.array_equal(a, b)
    # BGR blue -> RGB should put energy in channel 2
    assert a[..., 2].mean() > a[..., 0].mean()


def test_unsafe_augmentations_disabled():
    for name in UNSAFE_AUGMENTATIONS:
        with pytest.raises(ValueError):
            assert_augmentation_safe(name)
    rgb = np.full((224, 224, 3), 120, dtype=np.uint8)
    out = safe_augmentation(rgb, seed=0)
    assert out.shape == rgb.shape


def test_action_offset_and_episode_end():
    cfg = OpenVLAAdapterConfig(action_offset_frames=1)
    assert cfg.action_offset_frames == 1
    # select_export_rows with offset drops last frames : unit-level via tiny synthetic
    ts = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "timestamp": [0.0, 0.03, 0.06],
            "global_index": [0, 1, 2],
            "state": [np.zeros(6)] * 3,
            "state_smoothed": [np.zeros(6)] * 3,
            "action": [np.zeros(6)] * 3,
            "hard_valid": [True, True, True],
        }
    )
    win = pd.DataFrame(
        {
            "policy": ["conservative"],
            "episode_index": [0],
            "start_frame_index": [0],
            "end_frame_index_exclusive": [3],
            "start_global_index": [0],
            "end_global_index_exclusive": [3],
            "start_timestamp": [0.0],
            "end_timestamp": [0.1],
            "horizon": [3],
        }
    )
    out0 = select_export_rows(ts, win, OpenVLAAdapterConfig(action_offset_frames=0))
    out1 = select_export_rows(ts, win, OpenVLAAdapterConfig(action_offset_frames=1))
    assert len(out0) == 3
    assert len(out1) == 2


def test_no_invalid_window_export_real_if_present():
    curated = ROOT / "data/curated/svla_so100_pickplace"
    if not (curated / "training_windows.parquet").exists():
        pytest.skip("curated view not present")
    ts = pd.read_parquet(curated / "timesteps.parquet")
    win = pd.read_parquet(curated / "training_windows.parquet")
    covered = frames_covered_by_policy(win, "conservative")
    rows = select_export_rows(ts, win, OpenVLAAdapterConfig())
    for ep, fi in zip(rows["episode_index"], rows["frame_index"]):
        assert (int(ep), int(fi)) in covered


def test_collator_batch_shapes_and_identity():
    ex = []
    for i in range(3):
        ex.append(
            {
                "image_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
                "instruction": "Pick up the cube and place it in the box.",
                "action_raw": np.zeros(6),
                "action_encoded": np.zeros(7),
                "action_mask": np.array([1, 1, 1, 1, 1, 1, 0], dtype=float),
                "episode_index": i,
                "frame_index": 10 + i,
                "timestamp": 0.1 * i,
                "split": "train",
                "curation_policy": "conservative",
                "dataset_revision": "728583b5eaf9e739a7f119e2def466fa1d552402",
            }
        )
    batch = collate_openvla_batch(ex)
    assert validate_batch_shapes(batch) == []
    assert batch["episode_index"] == [0, 1, 2]


def test_smoke_manifest_schema_if_present():
    path = ROOT / "artifacts/task_05_vla_adaptation/batch_smoke_test.json"
    if not path.exists():
        pytest.skip("smoke artifact not generated yet")
    report = json.loads(path.read_text())
    assert report.get("model_checkpoint_downloaded") is False
    assert "encoded_action_shape" in report
    assert report.get("image_layout") == "NHWC"
    assert report.get("normalization_stat_source") == "train_only_q01_q99"
    splits = set(report.get("splits_present") or report.get("split") or [])
    assert {"train", "val", "test"} <= splits
    samples = report.get("samples") or []
    assert len(samples) >= 3
    positions = {s.get("temporal_position") for s in samples}
    assert positions & {"early", "middle", "late"}
    assert report.get("validation", {}).get("stock_loss_mask_integration_verified") is False


def test_pick_diverse_rows_covers_splits():
    from openarm_pipeline.vla.dataset_adapter import pick_diverse_rows

    rows = []
    for ep in (0, 1, 2):
        for fi in (0, 50, 100):
            rows.append(
                {
                    "episode_index": ep,
                    "frame_index": fi,
                    "timestamp": fi * 0.033,
                    "action": np.zeros(6),
                }
            )
    df = pd.DataFrame(rows)
    split = {"train": [0], "val": [1], "test": [2]}
    picked = pick_diverse_rows(df, split, batch_size=6)
    assert len(picked) == 6
    assert {str(r["split"]) for r in picked} == {"train", "val", "test"}


def test_export_manifest_validator():
    m = {
        "dataset_repo_id": "x",
        "dataset_revision": "y",
        "curation_policy": "conservative",
        "split": {},
        "n_examples": 1,
        "action_statistics_path": "z",
        "model_checkpoint_downloaded": False,
    }
    assert validate_export_manifest(m)["ok"] is True
    m2 = dict(m)
    m2["model_checkpoint_downloaded"] = True
    assert validate_export_manifest(m2)["ok"] is False
