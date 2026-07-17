"""Rollout protocol loading and validation (no fabricated rollout results)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SLICE_COUNTS = {
    "nominal_held_out": 40,
    "object_target_shift": 20,
    "lighting_camera_perturbation": 15,
    "control_latency_dynamics": 15,
    "occlusion_distractor": 10,
}

REQUIRED_ROLLOUT_FIELDS = {
    "rollout_id",
    "slice",
    "seed",
    "observation_config",
    "training_view",
    "perturbations",
    "paired_comparison_group",
}


FORBIDDEN_POLICY_INPUT_KEYS = {
    "simulator_success",
    "ground_truth_success",
    "task_success_label",
    "evaluation_labels",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def generate_rollout_matrix(
    *,
    seed_base: int = 42000,
    training_views: list[str] | None = None,
    observation_configs: list[str] | None = None,
) -> dict[str, Any]:
    """Build the fixed 100-rollout evaluation matrix (design artifact)."""
    training_views = training_views or ["conservative"]
    observation_configs = observation_configs or ["wrist_plus_top"]
    # Primary matrix uses one default training view / obs config; paired
    # comparison groups document fixed seeds for ablations.
    slices = [
        ("nominal_held_out", 40),
        ("object_target_shift", 20),
        ("lighting_camera_perturbation", 15),
        ("control_latency_dynamics", 15),
        ("occlusion_distractor", 10),
    ]
    rollouts = []
    k = 0
    for slice_name, count in slices:
        for i in range(count):
            seed = seed_base + k
            rid = f"R{k:03d}"
            pert: dict[str, Any] = {"enabled": slice_name != "nominal_held_out", "type": slice_name}
            if slice_name == "object_target_shift":
                pert.update({"object_xy_offset_m": [0.02 * ((i % 5) - 2), 0.015 * ((i % 3) - 1)], "target_yaw_deg": 5 * (i % 4)})
            elif slice_name == "lighting_camera_perturbation":
                pert.update({"brightness_scale": 0.7 + 0.1 * (i % 5), "gamma": 0.9 + 0.05 * (i % 3)})
            elif slice_name == "control_latency_dynamics":
                pert.update({"action_delay_ms": 20 + 10 * (i % 4), "friction_scale": 0.9 + 0.05 * (i % 3)})
            elif slice_name == "occlusion_distractor":
                pert.update({"occluder": True, "distractor_count": 1 + (i % 2)})
            rollouts.append(
                {
                    "rollout_id": rid,
                    "slice": slice_name,
                    "seed": seed,
                    "observation_config": observation_configs[0],
                    "training_view": training_views[0],
                    "state_features": "raw",
                    "perturbations": pert,
                    "paired_comparison_group": f"PAIR_{seed}",
                    "policy_inputs_forbid_simulator_success": True,
                    "record_fields": [
                        "policy_checkpoint_id",
                        "dataset_curation_version",
                        "random_seed",
                        "environment_configuration",
                        "object_target_initial_poses",
                        "camera_configuration",
                        "state_observations",
                        "actions",
                        "wrist_frames",
                        "top_frames",
                        "policy_inference_latency",
                        "control_loop_latency",
                        "simulator_task_state",
                        "task2_phase_event_failure_labels",
                        "visual_detector_probability",
                        "final_outcome",
                        "failure_taxonomy",
                        "safety_events",
                        "completion_time",
                    ],
                }
            )
            k += 1
    assert k == 100
    # Document paired comparison templates (same seeds, different configs)
    paired_templates = []
    for seed in [seed_base, seed_base + 1, seed_base + 40]:
        for tv in ["conservative", "strict"]:
            for obs in ["wrist_only", "top_only", "wrist_plus_top"]:
                for state in ["raw", "smoothed"]:
                    paired_templates.append(
                        {
                            "paired_comparison_group": f"PAIR_{seed}",
                            "seed": seed,
                            "training_view": tv,
                            "observation_config": obs,
                            "state_features": state,
                            "note": "Use identical seed/env; do not expose simulator success to policy.",
                        }
                    )
    return {
        "version": "1.0.0",
        "policy_family": "ACT",
        "total_rollouts": 100,
        "slice_counts": dict(REQUIRED_SLICE_COUNTS),
        "seed_base": seed_base,
        "rollouts": rollouts,
        "paired_comparison_templates": paired_templates,
        "notes": [
            "This file defines the evaluation matrix only; it does not contain fabricated outcomes.",
            "No ACT checkpoint is claimed to exist in this repository.",
        ],
    }


def validate_rollout_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    rollouts = protocol.get("rollouts") or []
    if len(rollouts) != 100:
        errors.append({"code": "rollout_count", "message": f"expected 100 rollouts, got {len(rollouts)}"})

    counts = Counter(r.get("slice") for r in rollouts)
    for name, expected in REQUIRED_SLICE_COUNTS.items():
        if counts.get(name, 0) != expected:
            errors.append(
                {
                    "code": "slice_count",
                    "message": f"slice {name}: expected {expected}, got {counts.get(name, 0)}",
                }
            )

    ids = [r.get("rollout_id") for r in rollouts]
    if len(ids) != len(set(ids)):
        errors.append({"code": "duplicate_rollout_id", "message": "duplicate rollout_id detected"})

    seeds = [r.get("seed") for r in rollouts]
    if len(seeds) != len(set(seeds)):
        # allow intentional pairing only via templates; primary matrix seeds must be unique
        errors.append({"code": "duplicate_seed", "message": "primary matrix seeds must be unique"})

    for r in rollouts:
        missing = REQUIRED_ROLLOUT_FIELDS - set(r.keys())
        if missing:
            errors.append(
                {
                    "code": "missing_field",
                    "message": f"{r.get('rollout_id')}: missing {sorted(missing)}",
                }
            )
        if not r.get("observation_config"):
            errors.append({"code": "missing_observation_config", "message": r.get("rollout_id")})
        if not isinstance(r.get("perturbations"), dict):
            errors.append({"code": "missing_perturbations", "message": r.get("rollout_id")})
        # leakage: policy input block must not include simulator success
        policy_inputs = r.get("policy_inputs") or {}
        for bad in FORBIDDEN_POLICY_INPUT_KEYS:
            if bad in policy_inputs:
                errors.append(
                    {
                        "code": "simulator_gt_leakage",
                        "message": f"{r.get('rollout_id')} exposes {bad} as policy input",
                    }
                )
        if r.get("policy_inputs_forbid_simulator_success") is not True:
            errors.append(
                {
                    "code": "simulator_gt_leakage",
                    "message": f"{r.get('rollout_id')} missing forbid flag",
                }
            )

    # paired templates present
    templates = protocol.get("paired_comparison_templates") or []
    if not templates:
        errors.append({"code": "paired_comparisons", "message": "missing paired_comparison_templates"})

    return {
        "ok": len(errors) == 0,
        "n_errors": len(errors),
        "errors": errors,
        "slice_counts": dict(counts),
        "n_rollouts": len(rollouts),
        "n_paired_templates": len(templates),
    }


def validate_rollout_result_record(record: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Lightweight required-field validation for a future rollout result row."""
    required = [
        "rollout_id",
        "policy_checkpoint_id",
        "dataset_repo_id",
        "dataset_revision",
        "curation_view",
        "seed",
        "evaluation_slice",
        "observation_config",
        "task_outcome",
        "ground_truth_success_source",
    ]
    errors = []
    for k in required:
        if k not in record:
            errors.append({"code": "missing_required_field", "field": k})
    # leakage check
    for bad in FORBIDDEN_POLICY_INPUT_KEYS:
        if bad in (record.get("policy_inputs") or {}):
            errors.append({"code": "simulator_gt_leakage", "field": bad})
    if record.get("ground_truth_success_source") == "wrist_success_detector_under_test":
        errors.append(
            {
                "code": "invalid_gt_source",
                "message": "primary success must not come from the detector under test",
            }
        )
    return {"ok": len(errors) == 0, "errors": errors}
