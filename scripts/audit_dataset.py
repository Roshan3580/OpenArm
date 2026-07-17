#!/usr/bin/env python3
"""CLI: Dataset exploration & quality audit (Task 1).

Examples:
  # Paired primary dataset (teleop + wrist + video alignment)
  python scripts/audit_dataset.py \\
    --repo-id lerobot/svla_so100_pickplace \\
    --output-dir artifacts/task_01_quality_audit/svla_so100_pickplace

  # Original ALOHA baseline (teleop; egocentric blocked)
  python scripts/audit_dataset.py \\
    --repo-id lerobot/aloha_sim_insertion_human \\
    --output-dir artifacts/task_01_quality_audit/aloha_sim_insertion_human
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.audit.cross_modal import (  # noqa: E402
    per_frame_state_action_deltas,
    summarize_cross_modal,
)
from openarm_pipeline.audit.egocentric import (  # noqa: E402
    audit_wrist_video_full,
    make_contact_sheet,
)
from openarm_pipeline.audit.teleop import (  # noqa: E402
    audit_teleop,
    plot_gap_distribution,
    plot_trajectory_lengths,
)
from openarm_pipeline.audit.video_alignment import (  # noqa: E402
    build_video_alignment_report,
    plan_episode_windows,
    probe_video_file,
)
from openarm_pipeline.data.lerobot_adapter import (  # noqa: E402
    build_manifest,
    dataset_slug,
    load_episodes_dataframe,
    load_tabular_dataframe,
    load_yaml,
    resolve_camera_video_path,
    sample_video_frames,
    save_json,
    stack_vector_column,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenArm Task 1 dataset quality audit")
    p.add_argument("--repo-id", default="lerobot/svla_so100_pickplace")
    p.add_argument("--revision", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default=None, help="Defaults to artifacts/.../<slug>/")
    p.add_argument("--config", default="configs/audit.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-video-frames", type=int, default=None)
    p.add_argument(
        "--windowed-wrist",
        action="store_true",
        help="Use contiguous windows instead of decoding every wrist frame",
    )
    return p.parse_args()


def print_camera_inspection(manifest_dict: dict) -> None:
    print("\n=== CAMERA INSPECTION ===")
    cams = manifest_dict.get("cameras") or []
    if not cams:
        print("No image/video features found.")
    for cam in cams:
        print(f"  key: {cam['key']}")
        print(f"  viewpoint: {cam['viewpoint']}")
        print(f"  dtype/shape: {cam.get('dtype')} {cam.get('shape')}")
        for e in cam.get("evidence") or []:
            print(f"    - {e}")
    has_ego = any(c.get("viewpoint") == "verified_egocentric" for c in cams)
    print(
        f"\nUsable wrist/egocentric stream: {'YES' if has_ego else 'NO — egocentric empirical audit BLOCKED'}"
    )
    print("=== END CAMERA INSPECTION ===\n")


def stratified_episode_sample(n_episodes: int, k: int, rng: np.random.Generator) -> list[int]:
    if n_episodes <= 0:
        return []
    k = min(k, n_episodes)
    base = np.linspace(0, n_episodes - 1, k)
    idxs = sorted(set(int(i) for i in np.clip(np.round(base).astype(int), 0, n_episodes - 1)))
    while len(idxs) < k:
        cand = int(rng.integers(0, n_episodes))
        if cand not in idxs:
            idxs.append(cand)
    return sorted(idxs)[:k]


def main() -> int:
    args = parse_args()
    slug = dataset_slug(args.repo_id)
    out_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/task_01_quality_audit") / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    config = load_yaml(cfg_path)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    print(f"Inspecting dataset {args.repo_id} (revision={args.revision}) ...")
    manifest = build_manifest(args.repo_id, revision=args.revision, split=args.split)
    manifest_dict = manifest.to_dict()
    save_json(manifest_dict, out_dir / "dataset_manifest.json")
    print_camera_inspection(manifest_dict)

    if not manifest.state_keys or not manifest.action_keys:
        print("ERROR: missing state/action keys", file=sys.stderr)
        return 1
    state_key = manifest.state_keys[0]
    action_key = manifest.action_keys[0]
    fps = float(manifest.fps or 30.0)

    # Contact sheet
    samp_cfg = config.get("sampling", {})
    n_contact_eps = int(samp_cfg.get("contact_sheet_episodes", 6))
    frames_per = int(samp_cfg.get("frames_per_episode_contact", 3))
    positions = list(np.linspace(0.1, 0.9, frames_per))
    ep_count = int(manifest.episode_count or 0)
    contact_eps = stratified_episode_sample(ep_count, n_contact_eps, rng)
    all_samples = []
    for cam in manifest.cameras:
        print(f"Sampling contact frames: {cam.key}")
        samples = sample_video_frames(
            repo_id=args.repo_id,
            video_key=cam.key,
            episode_indices=contact_eps,
            frame_positions=positions,
            revision=args.revision,
        )
        for s in samples:
            s = dict(s)
            s["camera_key"] = cam.key
            all_samples.append(s)
    make_contact_sheet(
        all_samples,
        str(out_dir / "camera_samples.png"),
        title=" | ".join(f"{c.key}:{c.viewpoint}" for c in manifest.cameras),
        cols=frames_per,
    )

    # Tabular teleop
    print("Loading tabular parquet ...")
    df = load_tabular_dataframe(args.repo_id, revision=args.revision, max_episodes=args.max_episodes)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    audited_episodes = int(df["episode_index"].nunique())
    scope = {
        "repo_id": args.repo_id,
        "revision": manifest.revision,
        "split": args.split,
        "seed": seed,
        "tabular_scope": "full_dataset" if args.max_episodes is None else f"max_episodes={args.max_episodes}",
        "episodes_in_manifest": manifest.episode_count,
        "episodes_audited": audited_episodes,
        "frames_audited": int(len(df)),
        "frames_in_manifest": manifest.total_frames,
        "is_full_tabular_audit": bool(
            args.max_episodes is None
            and audited_episodes == manifest.episode_count
            and len(df) == manifest.total_frames
        ),
        "artifact_dir": str(out_dir),
    }
    teleop = audit_teleop(df, manifest.feature_schema, state_key, action_key, manifest.fps, config, scope)
    plot_trajectory_lengths(teleop["trajectory_lengths_raw"], str(out_dir / "trajectory_lengths.png"))
    plot_gap_distribution(
        teleop["alignment"].get("gap_magnitudes_s_sample", []),
        teleop.get("expected_dt_s"),
        str(out_dir / "timestamp_gaps.png"),
    )

    # Video alignment for every episode / camera
    print("Probing videos and building per-episode alignment ...")
    try:
        episodes_df = load_episodes_dataframe(args.repo_id, revision=args.revision)
    except Exception as exc:
        print(f"warning: episodes meta unavailable ({exc}); alignment limited")
        episodes_df = (
            df.groupby("episode_index")
            .agg(length=("frame_index", "count"))
            .reset_index()
        )

    video_probes = {}
    local_videos = {}
    for cam in manifest.cameras:
        try:
            path = resolve_camera_video_path(args.repo_id, cam.key, revision=args.revision)
            local_videos[cam.key] = path
            video_probes[cam.key] = probe_video_file(path)
            video_probes[cam.key]["camera_key"] = cam.key
        except Exception as exc:
            video_probes[cam.key] = {
                "path": None,
                "opened": False,
                "error": f"missing_or_unreadable:{exc}",
                "zero_length": True,
            }

    va_cfg = config.get("video_alignment", {})
    timing_tol_s = float(va_cfg.get("timing_tol_frames", 1.0)) / fps
    video_alignment = build_video_alignment_report(
        episodes_df=episodes_df,
        tabular_df=df,
        camera_keys=[c.key for c in manifest.cameras],
        video_probes=video_probes,
        fps=fps,
        timing_tol_s=timing_tol_s,
        material_frame_mismatch=int(va_cfg.get("material_frame_mismatch", 2)),
        material_duration_mismatch_s=float(va_cfg.get("material_duration_mismatch_s", 0.1)),
    )
    save_json(video_alignment, out_dir / "video_alignment.json")

    # Egocentric / wrist
    verified_ego = [c for c in manifest.cameras if c.viewpoint == "verified_egocentric"]
    ego_results = []
    cross_modal = None
    wrist_metric_arrays = None

    if not verified_ego:
        ego_results.append(
            {
                "status": "blocked",
                "reason": "No verified_egocentric camera in schema",
                "cameras": [c.to_dict() for c in manifest.cameras],
            }
        )
        print("Egocentric empirical audit BLOCKED.")
    else:
        wrist = verified_ego[0]
        wrist_path = local_videos.get(wrist.key)
        decode_every = bool(samp_cfg.get("wrist_decode_every_frame", True)) and not args.windowed_wrist
        lengths = {int(e): int(n) for e, n in df.groupby("episode_index").size().items()}
        windows_plan = plan_episode_windows(
            lengths,
            n_windows=int(samp_cfg.get("wrist_windows_per_episode", 4)),
            window_size=int(samp_cfg.get("wrist_window_size", 12)),
            seed=seed,
            min_total_frames=int(samp_cfg.get("wrist_min_total_frames", 2000)),
        )
        print(
            f"Wrist audit on {wrist.key}: decode_every_frame={decode_every} "
            f"(planned window frames={windows_plan['total_frames_planned']})"
        )
        wrist_audit = audit_wrist_video_full(
            video_path=wrist_path,
            episode_index=df["episode_index"].to_numpy(),
            frame_index=df["frame_index"].to_numpy(),
            timestamp=df["timestamp"].to_numpy(),
            config=config,
            windows_plan=None if decode_every else windows_plan,
            decode_every_frame=decode_every,
        )
        # Save montages
        frames_map = wrist_audit.pop("_montage_frames", {})
        for name, items in frames_map.items():
            if not items:
                continue
            make_contact_sheet(
                items[:8],
                str(out_dir / f"wrist_montage_{name}.png"),
                title=f"Wrist {name}",
                cols=4,
            )
        # Persist compact metric arrays separately (not huge)
        wrist_metric_arrays = wrist_audit.pop("arrays_for_cross_modal", None)
        dists = wrist_audit.get("metric_distributions", {})
        save_json(
            {k: v for k, v in dists.items()},
            out_dir / "wrist_metric_distributions.json",
        )
        # drop bulky per-frame compact if needed — keep
        ego_results.append(wrist_audit)

        # Cross-modal on scored frames
        if wrist_metric_arrays is not None:
            gidx = wrist_metric_arrays["global_index"]
            state = stack_vector_column(df[state_key])
            action = stack_vector_column(df[action_key])
            # compute deltas on full series then index
            full_deltas = per_frame_state_action_deltas(
                state, action, gripper_dims=teleop.get("gripper_dims")
            )
            # Map scored global indices to delta at transition into that frame
            sd = []
            ad = []
            gd = []
            for gi in gidx:
                if gi <= 0:
                    sd.append(0.0)
                    ad.append(0.0)
                    gd.append(0.0)
                else:
                    sd.append(float(full_deltas["state_delta_norm"][gi - 1]))
                    ad.append(float(full_deltas["action_delta_norm"][gi - 1]))
                    if full_deltas["gripper_abs_delta"] is not None:
                        gd.append(float(full_deltas["gripper_abs_delta"][gi - 1]))
            cross_modal = summarize_cross_modal(
                sharpness=np.asarray(wrist_metric_arrays["laplacian_var"]),
                low_info_flag=np.asarray(wrist_metric_arrays["low_info_flag"]),
                exact_dup_flag=np.asarray(wrist_metric_arrays["exact_dup_flag"]),
                near_dup_flag=np.asarray(wrist_metric_arrays["near_dup_flag"]),
                state_delta_norm=np.asarray(sd),
                action_delta_norm=np.asarray(ad),
                gripper_abs_delta=np.asarray(gd) if gd else None,
                blur_threshold=float(config.get("egocentric", {}).get("laplacian_var_blur_threshold", 50)),
            )
            # Plot sharpness distribution
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.hist(wrist_metric_arrays["laplacian_var"], bins=50, color="#2c5f7c", edgecolor="white")
                ax.axvline(
                    float(config.get("egocentric", {}).get("laplacian_var_blur_threshold", 50)),
                    color="#c45c26",
                    linestyle="--",
                    label="blur threshold",
                )
                ax.set_title("Wrist Laplacian variance distribution")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "wrist_sharpness_hist.png", dpi=140)
                plt.close(fig)
            except Exception as exc:
                print(f"warning: sharpness plot skipped: {exc}")

    elapsed = time.perf_counter() - t0
    summary = {
        "task": "task_01_quality_audit",
        "dataset_slug": slug,
        "runtime_s": elapsed,
        "scope": scope,
        "config_path": str(cfg_path),
        "thresholds": {
            "teleop": config.get("teleop"),
            "egocentric": config.get("egocentric"),
            "video_alignment": config.get("video_alignment"),
            "sampling": config.get("sampling"),
        },
        "camera_inspection": {
            "cameras": [c.to_dict() for c in manifest.cameras],
            "has_verified_egocentric": bool(verified_ego),
        },
        "teleoperation": teleop,
        "egocentric": ego_results,
        "video_alignment_summary": {
            "summary_flags": video_alignment.get("summary_flags"),
            "video_file_probes": {
                k: {kk: vv for kk, vv in v.items() if kk != "path"}
                for k, v in (video_alignment.get("video_file_probes") or {}).items()
            },
            "n_episodes": video_alignment.get("n_episodes"),
            "report_path": str(out_dir / "video_alignment.json"),
        },
        "cross_modal": cross_modal,
        "artifacts": {
            "dataset_manifest": str(out_dir / "dataset_manifest.json"),
            "audit_summary": str(out_dir / "audit_summary.json"),
            "video_alignment": str(out_dir / "video_alignment.json"),
            "trajectory_lengths": str(out_dir / "trajectory_lengths.png"),
            "camera_samples": str(out_dir / "camera_samples.png"),
        },
    }
    save_json(summary, out_dir / "audit_summary.json")
    print(f"\nAudit complete in {elapsed:.1f}s -> {out_dir / 'audit_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
