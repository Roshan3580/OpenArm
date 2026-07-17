"""Bounded annotation QA for Task 2 labeling exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
SCHEMA_VERSION = "1.0.0"
MAX_ALIGNMENT_ERROR_FRAMES = 0.5

PHASE_ENUM = {
    "idle_or_reset",
    "approach",
    "pregrasp_alignment",
    "gripper_close",
    "grasp_verification",
    "lift",
    "transport",
    "place_alignment",
    "lower",
    "release",
    "retract",
    "recovery",
    "terminal_hold",
    "uncertain",
}

EVENT_ENUM = {
    "motion_start",
    "gripper_close_start",
    "first_contact",
    "stable_grasp",
    "object_liftoff",
    "transport_start",
    "object_over_target",
    "placement_contact",
    "release_start",
    "release_complete",
    "task_success_visible",
    "failure_onset",
    "recovery_start",
    "collision",
    "object_slip",
    "operator_intervention",
}

FAILURE_MOMENT_ENUM = {
    "missed_grasp",
    "visual_slip",
    "collision_visible",
    "object_lost_from_view",
    "target_lost_from_view",
    "severe_occlusion",
    "exposure_failure",
    "frozen_camera",
    "ambiguous_visual_failure",
}

VISIBILITY_ENUM = {
    "fully_visible",
    "partially_visible",
    "heavily_occluded",
    "not_visible",
    "uncertain",
}


def _err(code: str, message: str, **ctx: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **ctx}


def _collect_ids(doc: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    el = doc.get("episode_label") or {}
    if el.get("annotation_id"):
        ids.append(str(el["annotation_id"]))
    for key in (
        "phase_segments",
        "events",
        "wrist_interaction_states",
        "visibility_labels",
        "failure_moments",
        "attention_proxy_rois",
        "object_interaction_flags",
    ):
        for item in doc.get(key) or []:
            if item.get("annotation_id"):
                ids.append(str(item["annotation_id"]))
    return ids


def _frame_ts_ok(frame: int, ts: float, fps: float, tol: float = 1e-6) -> bool:
    expected = frame / fps
    return abs(expected - float(ts)) <= max(tol, 0.5 / fps)


def validate_annotation(
    doc: dict[str, Any],
    *,
    expected_revision: str = EXPECTED_REVISION,
    require_exhaustive: bool | None = None,
) -> dict[str, Any]:
    """Validate a multimodal annotation export. Returns a report dict."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    required_top = [
        "schema_version",
        "dataset_repo_id",
        "dataset_revision",
        "episode_index",
        "fps",
        "annotator_id",
        "created_at",
        "status",
        "confidence",
        "review_status",
        "source_modality",
        "episode_label",
        "phase_segments",
        "events",
    ]
    for k in required_top:
        if k not in doc:
            errors.append(_err("missing_required_field", f"missing top-level field: {k}", field=k))

    if doc.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            _err(
                "invalid_schema_version",
                f"schema_version must be {SCHEMA_VERSION}",
                got=doc.get("schema_version"),
            )
        )

    if doc.get("dataset_revision") != expected_revision:
        errors.append(
            _err(
                "invalid_dataset_revision",
                "dataset_revision does not match pinned corpus revision",
                expected=expected_revision,
                got=doc.get("dataset_revision"),
            )
        )

    fps = float(doc.get("fps") or 30.0)
    n_frames = doc.get("n_frames")
    if n_frames is not None:
        n_frames = int(n_frames)
    exhaustive = (
        require_exhaustive
        if require_exhaustive is not None
        else bool(doc.get("exhaustive_phase_timeline", False))
    )

    # Duplicate annotation IDs
    ids = _collect_ids(doc)
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            errors.append(_err("duplicate_annotation_id", f"duplicate annotation_id: {i}", annotation_id=i))
        seen.add(i)

    # Episode label
    ep = doc.get("episode_label") or {}
    for k in (
        "task_outcome",
        "demonstration_quality",
        "failure_types",
        "n_recovery_attempts",
        "intervention_occurred",
        "confidence",
    ):
        if k not in ep:
            errors.append(_err("missing_required_field", f"episode_label missing {k}", field=k))

    # Phase segments
    segments = list(doc.get("phase_segments") or [])
    segments_sorted = sorted(segments, key=lambda s: (s.get("start_frame", -1), s.get("end_frame", -1)))
    covered: list[tuple[int, int]] = []
    for seg in segments_sorted:
        sf = int(seg.get("start_frame", -1))
        ef = int(seg.get("end_frame", -1))
        if sf < 0 or ef < 0 or ef <= sf:
            errors.append(
                _err(
                    "invalid_frame_range",
                    "phase segment requires start_frame < end_frame with non-negative indices",
                    annotation_id=seg.get("annotation_id"),
                    start_frame=sf,
                    end_frame=ef,
                )
            )
            continue
        if n_frames is not None and (sf >= n_frames or ef > n_frames):
            errors.append(
                _err(
                    "invalid_frame_range",
                    "phase segment outside episode bounds",
                    annotation_id=seg.get("annotation_id"),
                    n_frames=n_frames,
                )
            )
        phase = seg.get("phase")
        if phase not in PHASE_ENUM:
            errors.append(
                _err("unknown_enum_value", f"unknown phase: {phase}", field="phase", value=phase)
            )
        st = seg.get("start_timestamp")
        et = seg.get("end_timestamp")
        if st is not None and not _frame_ts_ok(sf, float(st), fps):
            errors.append(
                _err(
                    "frame_timestamp_mismatch",
                    "start_timestamp does not match start_frame/fps",
                    annotation_id=seg.get("annotation_id"),
                )
            )
        if et is not None and not _frame_ts_ok(ef, float(et), fps):
            errors.append(
                _err(
                    "frame_timestamp_mismatch",
                    "end_timestamp does not match end_frame/fps (half-open end)",
                    annotation_id=seg.get("annotation_id"),
                )
            )
        # Overlap check against previous covered intervals (primary phases)
        for a, b in covered:
            if sf < b and ef > a:
                errors.append(
                    _err(
                        "overlapping_primary_phases",
                        "primary phase segments overlap",
                        annotation_id=seg.get("annotation_id"),
                        other_interval=[a, b],
                    )
                )
                break
        covered.append((sf, ef))
        ae = seg.get("alignment_error_s")
        if ae is not None and abs(float(ae) * fps) > MAX_ALIGNMENT_ERROR_FRAMES:
            errors.append(
                _err(
                    "alignment_error_exceeds_tolerance",
                    "segment alignment error exceeds ±0.5 frame",
                    annotation_id=seg.get("annotation_id"),
                )
            )

    if exhaustive and n_frames is not None and covered:
        covered_sorted = sorted(covered)
        if covered_sorted[0][0] != 0:
            errors.append(
                _err(
                    "exhaustive_timeline_gap",
                    "exhaustive timeline does not start at frame 0",
                    start=covered_sorted[0][0],
                )
            )
        cursor = covered_sorted[0][0]
        for a, b in covered_sorted:
            if a > cursor:
                errors.append(
                    _err(
                        "exhaustive_timeline_gap",
                        f"gap in exhaustive timeline [{cursor}, {a})",
                        gap_start=cursor,
                        gap_end=a,
                    )
                )
            cursor = max(cursor, b)
        if cursor < n_frames:
            errors.append(
                _err(
                    "exhaustive_timeline_gap",
                    f"gap at end of exhaustive timeline [{cursor}, {n_frames})",
                    gap_start=cursor,
                    gap_end=n_frames,
                )
            )

    # Events
    for ev in doc.get("events") or []:
        fi = int(ev.get("frame_index", -1))
        if n_frames is not None and (fi < 0 or fi >= n_frames):
            errors.append(
                _err(
                    "event_outside_episode_bounds",
                    "event frame_index outside episode",
                    annotation_id=ev.get("annotation_id"),
                    frame_index=fi,
                )
            )
        et = ev.get("event_type")
        if et not in EVENT_ENUM:
            errors.append(
                _err("unknown_enum_value", f"unknown event_type: {et}", field="event_type", value=et)
            )
        ts = ev.get("timestamp")
        if ts is not None and fi >= 0 and not _frame_ts_ok(fi, float(ts), fps):
            errors.append(
                _err(
                    "frame_timestamp_mismatch",
                    "event timestamp does not match frame_index/fps",
                    annotation_id=ev.get("annotation_id"),
                )
            )
        ae = ev.get("alignment_error_s")
        if ae is not None and abs(float(ae) * fps) > MAX_ALIGNMENT_ERROR_FRAMES:
            errors.append(
                _err(
                    "alignment_error_exceeds_tolerance",
                    "event alignment error exceeds ±0.5 frame",
                    annotation_id=ev.get("annotation_id"),
                )
            )

    # Failure moments
    for fm in doc.get("failure_moments") or []:
        ft = fm.get("failure_type")
        if ft not in FAILURE_MOMENT_ENUM:
            errors.append(
                _err(
                    "unknown_enum_value",
                    f"unknown failure_type: {ft}",
                    field="failure_type",
                    value=ft,
                )
            )
        fi = int(fm.get("frame_index", -1))
        if n_frames is not None and (fi < 0 or fi >= n_frames):
            errors.append(
                _err(
                    "event_outside_episode_bounds",
                    "failure moment outside episode",
                    annotation_id=fm.get("annotation_id"),
                )
            )

    # Visibility
    for vis in doc.get("visibility_labels") or []:
        v = vis.get("visibility")
        if v not in VISIBILITY_ENUM:
            errors.append(
                _err("unknown_enum_value", f"unknown visibility: {v}", field="visibility", value=v)
            )

    # ROIs
    for roi in doc.get("attention_proxy_rois") or []:
        coords = roi.get("coordinates") or []
        if not isinstance(coords, list) or not coords:
            errors.append(
                _err("invalid_roi_coordinates", "ROI coordinates missing", annotation_id=roi.get("annotation_id"))
            )
            continue
        for c in coords:
            if not isinstance(c, (int, float)) or c < 0 or c > 1:
                errors.append(
                    _err(
                        "invalid_roi_coordinates",
                        "ROI coordinates must be in [0,1]",
                        annotation_id=roi.get("annotation_id"),
                        value=c,
                    )
                )
                break
        shape = roi.get("shape")
        if shape == "bbox_xyxy" and len(coords) == 4:
            x0, y0, x1, y1 = coords
            if x1 < x0 or y1 < y0:
                errors.append(
                    _err(
                        "invalid_roi_coordinates",
                        "bbox_xyxy requires x1>=x0 and y1>=y0",
                        annotation_id=roi.get("annotation_id"),
                    )
                )

    ok = len(errors) == 0
    return {
        "ok": ok,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "schema_version": SCHEMA_VERSION,
        "expected_revision": expected_revision,
    }


def validate_annotation_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    from openarm_pipeline.paths import sanitize_path_string

    path = Path(path)
    with open(path) as f:
        doc = json.load(f)
    report = validate_annotation(doc, **kwargs)
    report["path"] = sanitize_path_string(str(path))
    return report
