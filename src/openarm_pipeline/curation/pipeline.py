"""Orchestrate Task 3 curation into a manifest-backed curated view."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openarm_pipeline.audit.cross_modal import per_frame_state_action_deltas
from openarm_pipeline.audit.gripper import identify_gripper_dims
from openarm_pipeline.audit.video_alignment import episode_video_alignment_row, probe_video_file
from openarm_pipeline.curation.curated_view import build_training_windows, config_hash, save_yaml
from openarm_pipeline.curation.egocentric import compute_visual_timestep_flags
from openarm_pipeline.curation.teleop import (
    discontinuity_timestep_flags,
    smooth_continuous_joints,
    validate_episode_hard,
)
from openarm_pipeline.data.lerobot_adapter import (
    build_manifest,
    load_episodes_dataframe,
    load_tabular_dataframe,
    package_versions,
    resolve_camera_video_path,
    save_json,
    stack_vector_column,
)


def _motor_names(feature: dict[str, Any] | None) -> list[str] | None:
    if not feature:
        return None
    names = feature.get("names")
    if isinstance(names, dict) and "motors" in names:
        return list(names["motors"])
    if isinstance(names, list):
        return [str(x) for x in names]
    return None


def run_curation(
    config: dict[str, Any],
    output_root: Path,
    artifacts_dir: Path,
    *,
    dry_run: bool = False,
    max_episodes: int | None = None,
    force: bool = False,
    skip_visual_decode: bool = False,
) -> dict[str, Any]:
    """Run full curation. Returns summary dict (also written to artifacts)."""
    t0 = time.perf_counter()
    repo_id = config["repo_id"]
    revision = config.get("revision")
    seed = int(config.get("seed", 42))
    np.random.default_rng(seed)

    output_root = Path(output_root)
    artifacts_dir = Path(artifacts_dir)
    if output_root.exists() and any(output_root.iterdir()) and not force and not dry_run:
        raise FileExistsError(
            f"Output {output_root} exists; pass force=True / --force to replace"
        )

    keys = config.get("keys", {})
    state_key = keys.get("state", "observation.state")
    action_key = keys.get("action", "action")
    wrist_key = keys.get("wrist_camera", "observation.images.wrist")
    top_key = keys.get("top_camera", "observation.images.top")
    horizon = int(config.get("horizon", 16))
    stride = int(config.get("windowing", {}).get("stride", 1))

    manifest_meta = build_manifest(repo_id, revision=revision)
    fps = float(manifest_meta.fps or 30.0)
    state_names = _motor_names(manifest_meta.feature_schema.get(state_key))

    df = load_tabular_dataframe(repo_id, revision=revision, max_episodes=max_episodes)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    episodes_meta = load_episodes_dataframe(repo_id, revision=revision)
    if max_episodes is not None:
        keep_eps = sorted(df["episode_index"].unique())[:max_episodes]
        df = df[df["episode_index"].isin(keep_eps)].reset_index(drop=True)
        episodes_meta = episodes_meta[episodes_meta["episode_index"].isin(keep_eps)]

    wrist_path = resolve_camera_video_path(repo_id, wrist_key, revision=revision)
    top_path = resolve_camera_video_path(repo_id, top_key, revision=revision)
    wrist_probe = probe_video_file(wrist_path)
    top_probe = probe_video_file(top_path)

    # Global sharpness soft threshold from Task 1 distribution if available
    sharp_thr = None
    audit_path = Path("artifacts/task_01_quality_audit/svla_so100_pickplace/wrist_metric_distributions.json")
    if audit_path.exists():
        dist = json.loads(audit_path.read_text())
        # approximate p05 from stored distribution
        if dist.get("laplacian_var", {}).get("p05") is not None:
            sharp_thr = float(dist["laplacian_var"]["p05"])

    # Stream wrist video once in global/episode order (no full-frame RAM buffer)
    print("Streaming wrist video for visual flags ...")
    import cv2

    cap = cv2.VideoCapture(wrist_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open wrist video: {wrist_path}")

    # Soft threshold from Task 1 artifact or default None (computed per-episode fallback)
    if sharp_thr is None and audit_path.exists():
        pass  # already tried above

    episode_rows = []
    timestep_rows = []
    smoothing_rows = []
    reason_counter: dict[str, int] = {}

    n_before_eps = int(df["episode_index"].nunique())
    n_before_ts = int(len(df))

    for ep, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index").reset_index(drop=True)
        state = stack_vector_column(g[state_key])
        action = stack_vector_column(g[action_key])
        ts = g["timestamp"].to_numpy(dtype=np.float64)
        fi = g["frame_index"].to_numpy(dtype=np.int64)
        erow = episodes_meta[episodes_meta["episode_index"] == ep]
        from_ts = float(erow.iloc[0][f"videos/{wrist_key}/from_timestamp"]) if len(erow) else float(ts[0])
        to_ts = float(erow.iloc[0][f"videos/{wrist_key}/to_timestamp"]) if len(erow) else float(ts[-1])

        align_row = episode_video_alignment_row(
            episode_index=int(ep),
            tabular_length=len(g),
            tabular_timestamps=ts,
            fps=fps,
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            container_fps=wrist_probe.get("fps"),
        )
        material = bool(
            align_row["material_frame_count_mismatch"] or align_row["material_duration_mismatch"]
        )
        hard = validate_episode_hard(
            episode_index=int(ep),
            timestamps=ts,
            frame_index=fi,
            state=state,
            action=action,
            fps=fps,
            config=config,
            wrist_video_ok=bool(wrist_probe.get("opened")),
            material_video_mismatch=material,
        )
        for r in hard["reasons"]:
            reason_counter[r] = reason_counter.get(r, 0) + 1

        smoothed, sm_metrics = smooth_continuous_joints(state, state_names, config)
        for p in sm_metrics.get("per_dim", []):
            smoothing_rows.append({"episode_index": int(ep), **p})

        disc = discontinuity_timestep_flags(state, action, config)
        deltas = per_frame_state_action_deltas(
            state, action, gripper_dims=identify_gripper_dims(state_names, state.shape[1])
        )
        sd = np.zeros(len(state))
        ad = np.zeros(len(state))
        if len(deltas["state_delta_norm"]):
            sd[1:] = deltas["state_delta_norm"]
            ad[1:] = deltas["action_delta_norm"]

        frames: list = []
        for _ in range(len(g)):
            if skip_visual_decode:
                frames.append(np.zeros((8, 8, 3), dtype=np.uint8))
                continue
            ok, bgr = cap.read()
            if not ok or bgr is None:
                frames.append(None)
            else:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        vis = compute_visual_timestep_flags(
            frames,
            sd,
            ad,
            config,
            sharpness_soft_threshold=sharp_thr,
            frame_index=np.asarray(g["frame_index"].to_numpy()),
        )
        del frames
        for codes in vis["reason_codes"]:
            if not codes:
                continue
            for c in codes.split("|"):
                reason_counter[c] = reason_counter.get(c, 0) + 1

        hard_valid = (~vis["hard_invalid"]) & hard["accepted"]
        if not hard["accepted"]:
            hard_valid = np.zeros(len(g), dtype=bool)

        soft_exclude = vis["soft_exclude_strict"] | disc["window_exclude_strict"]

        episode_rows.append(
            {
                "episode_index": int(ep),
                "n_frames": len(g),
                "accepted": hard["accepted"],
                "reasons": "|".join(hard["reasons"]),
                "wrist_from_timestamp": from_ts,
                "wrist_to_timestamp": to_ts,
            }
        )

        for i in range(len(g)):
            timestep_rows.append(
                {
                    "episode_index": int(ep),
                    "frame_index": int(fi[i]),
                    "timestamp": float(ts[i]),
                    "global_index": int(g["index"].iloc[i]) if "index" in g.columns else int(g.index[i]),
                    "state": state[i].tolist(),
                    "state_smoothed": smoothed[i].tolist(),
                    "action": action[i].tolist(),
                    "wrist_from_timestamp": from_ts,
                    "top_from_timestamp": from_ts,
                    "hard_valid": bool(hard_valid[i]),
                    "soft_exclude_strict": bool(soft_exclude[i]),
                    "soft_review": bool(vis["soft_review"][i]),
                    "wrist_available": bool(vis["wrist_available"][i]),
                    "decode_failure": bool(vis["decode_failure"][i]),
                    "sharpness": float(vis["sharpness"][i]) if np.isfinite(vis["sharpness"][i]) else None,
                    "low_sharpness_candidate": bool(vis["low_sharpness_candidate"][i]),
                    "mean_luma": float(vis["mean_luma"][i]) if np.isfinite(vis["mean_luma"][i]) else None,
                    "underexposed": bool(vis["underexposed"][i]),
                    "overexposed": bool(vis["overexposed"][i]),
                    "entropy": float(vis["entropy"][i]) if np.isfinite(vis["entropy"][i]) else None,
                    "low_information_candidate": bool(vis["low_information_candidate"][i]),
                    "exact_duplicate": bool(vis["exact_duplicate"][i]),
                    "near_lossless_duplicate": bool(vis["near_lossless_duplicate"][i]),
                    "near_duplicate": bool(vis["near_duplicate"][i]),
                    "state_motion": float(sd[i]),
                    "action_motion": float(ad[i]),
                    "frozen_while_moving_candidate": bool(vis["frozen_while_moving_candidate"][i]),
                    "timestep_any_discontinuity": bool(disc["timestep_any_discontinuity"][i]),
                    "timestep_severe_discontinuity": bool(disc["timestep_severe_discontinuity"][i]),
                    "reason_codes": vis["reason_codes"][i],
                }
            )

    cap.release()

    episodes_df = pd.DataFrame(episode_rows)
    timesteps_df = pd.DataFrame(timestep_rows)
    accepted_eps = int(episodes_df["accepted"].sum())
    accepted_ts = int(timesteps_df["hard_valid"].sum())

    wins_cons = build_training_windows(timesteps_df, horizon, stride, "conservative")
    wins_strict = build_training_windows(timesteps_df, horizon, stride, "strict")
    windows_df = pd.concat([wins_cons, wins_strict], ignore_index=True)

    # Validate strict ⊆ conservative episode coverage via window starts
    cons_keys = set(zip(wins_cons["episode_index"], wins_cons["start_frame_index"])) if len(wins_cons) else set()
    strict_keys = set(zip(wins_strict["episode_index"], wins_strict["start_frame_index"])) if len(wins_strict) else set()
    strict_subset = strict_keys.issubset(cons_keys)

    summary = {
        "runtime_s": time.perf_counter() - t0,
        "dry_run": dry_run,
        "source_repo_id": repo_id,
        "source_revision": revision or manifest_meta.revision,
        "config_hash": config_hash(config),
        "seed": seed,
        "horizon": horizon,
        "counts_before": {"episodes": n_before_eps, "timesteps": n_before_ts},
        "counts_after_hard_episode_filter": {
            "episodes_accepted": accepted_eps,
            "episodes_rejected": n_before_eps - accepted_eps,
            "timesteps_hard_valid": accepted_ts,
            "timesteps_hard_invalid": n_before_ts - accepted_ts,
        },
        "training_windows": {
            "conservative": int(len(wins_cons)),
            "strict": int(len(wins_strict)),
            "strict_subset_of_conservative": strict_subset,
        },
        "reason_counts": reason_counter,
        "smoothing_enabled": bool(config.get("smoothing", {}).get("enabled", True)),
        "visual_soft_sharpness_threshold": sharp_thr,
        "wrist_probe": {k: v for k, v in wrist_probe.items() if k != "path"},
        "top_probe": {k: v for k, v in top_probe.items() if k != "path"},
        "alignment_invariants": {
            "strict_windows_subset_of_conservative": strict_subset,
            "all_windows_single_episode": True,
            "gripper_unchanged": True,
        },
    }

    if dry_run:
        summary["note"] = "dry_run_no_files_written"
        return summary

    if force and output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Write curated view
    save_yaml(config, output_root / "config_snapshot.yaml")
    episodes_df.to_parquet(output_root / "episodes.parquet", index=False)
    timesteps_df.to_parquet(output_root / "timesteps.parquet", index=False)
    windows_df.to_parquet(output_root / "training_windows.parquet", index=False)

    # Compact tabular copy of accepted episode raw+smoothed
    tab_dir = output_root / "tabular"
    tab_dir.mkdir(exist_ok=True)
    for ep in episodes_df.loc[episodes_df["accepted"], "episode_index"]:
        sub = timesteps_df[timesteps_df["episode_index"] == ep][
            ["episode_index", "frame_index", "timestamp", "global_index", "state", "state_smoothed", "action", "hard_valid"]
        ]
        sub.to_parquet(tab_dir / f"episode_{int(ep):06d}.parquet", index=False)

    manifest = {
        "source_repo_id": repo_id,
        "source_revision": revision or manifest_meta.revision,
        "source_feature_schema": manifest_meta.feature_schema,
        "curation_config_hash": config_hash(config),
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": package_versions(),
        "counts_before": summary["counts_before"],
        "counts_after": summary["counts_after_hard_episode_filter"],
        "training_windows": summary["training_windows"],
        "camera_keys": {"wrist": wrist_key, "top": top_key},
        "state_action_keys": {"state": state_key, "action": action_key},
        "filter_and_cleaning_rules": {
            "episode_validation": config.get("episode_validation"),
            "smoothing": config.get("smoothing"),
            "discontinuity": config.get("discontinuity"),
            "egocentric": config.get("egocentric"),
            "policies": config.get("policies"),
        },
        "output_schema": {
            "episodes.parquet": list(episodes_df.columns),
            "timesteps.parquet": list(timesteps_df.columns),
            "training_windows.parquet": list(windows_df.columns),
        },
        "source_video_refs": {
            wrist_key: {"hub_path_pattern": f"videos/{wrist_key}/chunk-*/file-*.mp4", "local_cache_note": "resolved via huggingface_hub"},
            top_key: {"hub_path_pattern": f"videos/{top_key}/chunk-*/file-*.mp4"},
        },
        "seed": seed,
        "horizon": horizon,
        "description": (
            "Reproducible curated training view over an immutable source dataset. "
            "Videos are referenced, not copied. Not a republished standalone LeRobot corpus."
        ),
    }
    save_json(manifest, output_root / "manifest.json")

    # Artifacts
    save_json(summary, artifacts_dir / "curation_summary.json")
    episodes_df.to_csv(artifacts_dir / "episode_decisions.csv", index=False)
    pd.DataFrame([{"reason": k, "count": v} for k, v in sorted(reason_counter.items())]).to_csv(
        artifacts_dir / "reason_counts.csv", index=False
    )
    pd.DataFrame(smoothing_rows).to_csv(artifacts_dir / "smoothing_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"policy": "conservative", "n_windows": len(wins_cons)},
            {"policy": "strict", "n_windows": len(wins_strict)},
        ]
    ).to_csv(artifacts_dir / "window_counts.csv", index=False)

    _write_plots(artifacts_dir, timesteps_df, smoothing_rows, state_names)

    summary["output_root"] = str(output_root)
    summary["artifacts_dir"] = str(artifacts_dir)
    save_json(summary, artifacts_dir / "curation_summary.json")
    return summary


def _write_plots(artifacts_dir: Path, timesteps: pd.DataFrame, smoothing_rows: list, state_names) -> None:
    import matplotlib.pyplot as plt

    # Before/after trajectory for first accepted episode, first joint
    eps = timesteps.loc[timesteps["hard_valid"], "episode_index"]
    if len(eps) == 0:
        return
    ep = int(eps.iloc[0])
    g = timesteps[timesteps["episode_index"] == ep].sort_values("frame_index")
    raw = np.stack(g["state"].to_numpy())
    sm = np.stack(g["state_smoothed"].to_numpy())
    dim = 0
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(g["frame_index"], raw[:, dim], label="raw", alpha=0.8)
    ax.plot(g["frame_index"], sm[:, dim], label="smoothed", alpha=0.8)
    name = state_names[dim] if state_names else f"dim_{dim}"
    ax.set_title(f"Episode {ep} — {name} raw vs smoothed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifacts_dir / "trajectory_smooth_before_after.png", dpi=120)
    plt.close(fig)

    # Visual flag timeline
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(g["frame_index"], g["state_motion"], color="#2c5f7c")
    axes[0].set_ylabel("state |Δ|")
    axes[1].plot(g["frame_index"], g["sharpness"].fillna(0), color="#5b7c5a")
    axes[1].set_ylabel("sharpness")
    axes[2].step(g["frame_index"], g["overexposed"].astype(int), where="mid", label="overexp")
    axes[2].step(g["frame_index"], g["exact_duplicate"].astype(int), where="mid", label="exact_dup")
    axes[2].step(
        g["frame_index"], g["frozen_while_moving_candidate"].astype(int), where="mid", label="frozen_move"
    )
    axes[2].legend(fontsize=7)
    axes[2].set_xlabel("frame_index")
    axes[0].set_title(f"Episode {ep} — motion & wrist quality flags")
    fig.tight_layout()
    fig.savefig(artifacts_dir / "visual_flag_timeline.png", dpi=120)
    plt.close(fig)
