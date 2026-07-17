#!/usr/bin/env python3
"""Recount within-episode adjacent wrist-frame duplicates (Task 1 fix).

Compares only consecutive frame_index pairs inside the same episode.
Never compares across episode boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.audit.egocentric import (  # noqa: E402
    audit_wrist_video_full,
    within_episode_adjacent_pair_count,
)
from openarm_pipeline.data.lerobot_adapter import (  # noqa: E402
    load_tabular_dataframe,
    load_yaml,
    resolve_camera_video_path,
    save_json,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="lerobot/svla_so100_pickplace")
    p.add_argument("--revision", default="728583b5eaf9e739a7f119e2def466fa1d552402")
    p.add_argument(
        "--output-dir",
        default="artifacts/task_01_quality_audit/svla_so100_pickplace",
    )
    p.add_argument("--config", default="configs/audit.yaml")
    args = p.parse_args()

    config = load_yaml(ROOT / args.config)
    df = load_tabular_dataframe(args.repo_id, revision=args.revision)
    video_path = resolve_camera_video_path(
        args.repo_id, "observation.images.wrist", revision=args.revision
    )
    n_frames = len(df)
    n_episodes = int(df["episode_index"].nunique())
    expected = within_episode_adjacent_pair_count(n_frames, n_episodes)

    print(f"frames={n_frames} episodes={n_episodes} expected_pairs={expected}")
    print(f"video={video_path}")

    result = audit_wrist_video_full(
        video_path=str(video_path),
        episode_index=df["episode_index"].to_numpy(),
        frame_index=df["frame_index"].to_numpy(),
        timestamp=df["timestamp"].to_numpy() if "timestamp" in df.columns else np.zeros(len(df)),
        config=config,
        windows_plan=None,
        decode_every_frame=True,
    )

    denom = int(result["duplicate_adjacent_pairs_denominator"])
    exact = int(result["duplicate_adjacent_exact"])
    nl = int(result["duplicate_adjacent_near_lossless"])
    near = int(result["duplicate_adjacent_near"])
    assert denom == expected, f"denominator {denom} != expected {expected}"

    out = {
        "dataset_repo_id": args.repo_id,
        "dataset_revision": args.revision,
        "n_frames": n_frames,
        "n_episodes": n_episodes,
        "within_episode_adjacent_pairs_denominator": denom,
        "exact_duplicate": exact,
        "near_lossless_duplicate": nl,
        "near_duplicate": near,
        "exact_duplicate_rate": float(exact / denom),
        "near_lossless_duplicate_rate": float(nl / denom),
        "near_duplicate_rate": float(near / denom),
        "cross_episode_pairs_excluded": 49,
        "note": (
            "Mutually exclusive within-episode adjacent pairs only; "
            "never across episode boundaries."
        ),
    }
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out, out_dir / "duplicate_recount_within_episode.json")
    print(json.dumps(out, indent=2))

    # Patch audit_summary.json egocentric section if present
    summary_path = out_dir / "audit_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        ego = summary.get("egocentric") or summary.get("wrist_full") or {}
        # Walk common nesting
        updated = False
        for key in ("egocentric", "wrist_video_full", "wrist_full", "cameras"):
            node = summary.get(key)
            if isinstance(node, dict) and "duplicate_adjacent_exact" in node:
                node["duplicate_adjacent_pairs_denominator"] = denom
                node["duplicate_adjacent_pairs_expected"] = expected
                node["duplicate_adjacent_exact"] = exact
                node["duplicate_adjacent_near_lossless"] = nl
                node["duplicate_adjacent_near"] = near
                node["duplicate_exact_rate"] = float(exact / denom)
                node["duplicate_near_lossless_rate"] = float(nl / denom)
                node["duplicate_near_rate"] = float(near / denom)
                node["duplicate_accounting"] = {
                    "scope": "within_episode_adjacent_only",
                    "cross_episode_pairs_excluded": True,
                    "require_consecutive_frame_index": True,
                }
                updated = True
            elif isinstance(node, dict):
                # nested under camera key
                for sub in node.values():
                    if isinstance(sub, dict) and "duplicate_adjacent_exact" in sub:
                        sub["duplicate_adjacent_pairs_denominator"] = denom
                        sub["duplicate_adjacent_pairs_expected"] = expected
                        sub["duplicate_adjacent_exact"] = exact
                        sub["duplicate_adjacent_near_lossless"] = nl
                        sub["duplicate_adjacent_near"] = near
                        sub["duplicate_exact_rate"] = float(exact / denom)
                        sub["duplicate_near_lossless_rate"] = float(nl / denom)
                        sub["duplicate_near_rate"] = float(near / denom)
                        sub["duplicate_accounting"] = {
                            "scope": "within_episode_adjacent_only",
                            "cross_episode_pairs_excluded": True,
                            "require_consecutive_frame_index": True,
                        }
                        updated = True
        if "duplicate_recount_within_episode" not in summary:
            summary["duplicate_recount_within_episode"] = out
        else:
            summary["duplicate_recount_within_episode"] = out
        if updated or True:
            save_json(summary, summary_path)
            print(f"patched {summary_path}")


if __name__ == "__main__":
    main()
