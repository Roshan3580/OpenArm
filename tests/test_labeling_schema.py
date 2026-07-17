"""Tests for Task 2 labeling schema validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openarm_pipeline.labeling.validate import validate_annotation

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tasks/task_02_labeling_design/sample_annotation.json"


@pytest.fixture
def sample() -> dict:
    with open(SAMPLE) as f:
        return json.load(f)


def test_valid_illustrative_annotation(sample):
    report = validate_annotation(sample)
    assert report["ok"] is True
    assert report["n_errors"] == 0
    assert sample["illustrative_not_ground_truth"] is True


def test_invalid_schema_version(sample):
    sample["schema_version"] = "0.0.0"
    report = validate_annotation(sample)
    assert report["ok"] is False
    assert any(e["code"] == "invalid_schema_version" for e in report["errors"])


def test_invalid_frame_range(sample):
    sample["phase_segments"][0]["end_frame"] = sample["phase_segments"][0]["start_frame"]
    report = validate_annotation(sample)
    assert any(e["code"] == "invalid_frame_range" for e in report["errors"])


def test_frame_timestamp_mismatch(sample):
    sample["events"][0]["timestamp"] = 9.99  # not matching frame 30 @ 30fps
    report = validate_annotation(sample)
    assert any(e["code"] == "frame_timestamp_mismatch" for e in report["errors"])


def test_overlapping_primary_phases(sample):
    a = sample["phase_segments"][0]
    b = copy.deepcopy(a)
    b["annotation_id"] = "overlap-bad"
    b["start_frame"] = a["start_frame"] + 1
    b["end_frame"] = a["end_frame"] + 5
    b["start_timestamp"] = b["start_frame"] / sample["fps"]
    b["end_timestamp"] = b["end_frame"] / sample["fps"]
    sample["phase_segments"].append(b)
    report = validate_annotation(sample)
    assert any(e["code"] == "overlapping_primary_phases" for e in report["errors"])


def test_exhaustive_timeline_gap(sample):
    sample["phase_segments"] = [s for s in sample["phase_segments"] if s["annotation_id"] != "seg-007"]
    report = validate_annotation(sample, require_exhaustive=True)
    assert any(e["code"] == "exhaustive_timeline_gap" for e in report["errors"])


def test_invalid_roi_coordinates(sample):
    sample["attention_proxy_rois"][0]["coordinates"] = [0.1, 0.2, 1.5, 0.4]
    report = validate_annotation(sample)
    assert any(e["code"] == "invalid_roi_coordinates" for e in report["errors"])


def test_unknown_enum_value(sample):
    sample["phase_segments"][0]["phase"] = "flying"
    report = validate_annotation(sample)
    assert any(e["code"] == "unknown_enum_value" for e in report["errors"])


def test_event_outside_episode_bounds(sample):
    sample["events"][0]["frame_index"] = 10_000
    sample["events"][0]["timestamp"] = 10_000 / sample["fps"]
    report = validate_annotation(sample)
    assert any(e["code"] == "event_outside_episode_bounds" for e in report["errors"])


def test_dataset_revision_mismatch(sample):
    sample["dataset_revision"] = "deadbeef"
    report = validate_annotation(sample)
    assert any(e["code"] == "invalid_dataset_revision" for e in report["errors"])


def test_duplicate_annotation_ids(sample):
    sample["events"][1]["annotation_id"] = sample["events"][0]["annotation_id"]
    report = validate_annotation(sample)
    assert any(e["code"] == "duplicate_annotation_id" for e in report["errors"])


def test_half_open_segment_boundary_behavior(sample):
    """Frame end_frame is exclusive: segment [10,13) covers 10,11,12 only."""
    seg = {
        "annotation_id": "half-open-demo",
        "start_frame": 10,
        "end_frame": 13,
        "start_timestamp": 10 / 30,
        "end_timestamp": 13 / 30,
        "phase": "approach",
        "motion_quality": "smooth",
        "interaction_state": "none",
        "is_recovery_behavior": False,
        "confidence": 1.0,
    }
    covered = list(range(seg["start_frame"], seg["end_frame"]))
    assert covered == [10, 11, 12]
    a = (0, 10)
    b = (10, 13)
    assert a[1] == b[0]
    doc = {
        "schema_version": "1.0.0",
        "dataset_repo_id": "lerobot/svla_so100_pickplace",
        "dataset_revision": "728583b5eaf9e739a7f119e2def466fa1d552402",
        "illustrative_not_ground_truth": True,
        "episode_index": 0,
        "fps": 30,
        "n_frames": 13,
        "annotator_id": "t",
        "created_at": "2026-07-17T00:00:00Z",
        "status": "draft",
        "confidence": 1.0,
        "review_status": "unreviewed",
        "source_modality": "teleoperation",
        "exhaustive_phase_timeline": True,
        "episode_label": {
            "annotation_id": "e",
            "task_outcome": "uncertain",
            "demonstration_quality": "review_required",
            "failure_types": [],
            "n_recovery_attempts": 0,
            "intervention_occurred": False,
            "confidence": 1.0,
        },
        "phase_segments": [
            {
                "annotation_id": "s0",
                "start_frame": 0,
                "end_frame": 10,
                "start_timestamp": 0.0,
                "end_timestamp": 10 / 30,
                "phase": "idle_or_reset",
                "motion_quality": "smooth",
                "interaction_state": "none",
                "is_recovery_behavior": False,
                "confidence": 1.0,
            },
            seg,
        ],
        "events": [],
    }
    report = validate_annotation(doc, require_exhaustive=True)
    assert report["ok"] is True
