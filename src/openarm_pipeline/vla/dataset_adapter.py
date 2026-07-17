"""Map Task 3 curated views into OpenVLA-compatible example metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from openarm_pipeline.vla.action_encoding import ActionNormalizer, compute_delta_actions
from openarm_pipeline.vla.config import OpenVLAAdapterConfig
from openarm_pipeline.vla.validation import assert_no_episode_leakage


def episode_grouped_split(
    episode_ids: Iterable[int],
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[int]]:
    eps = sorted({int(e) for e in episode_ids})
    rng = np.random.default_rng(seed)
    order = eps.copy()
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    while n_train + n_val >= n:
        n_val = max(0, n_val - 1)
    split = {
        "train": [int(x) for x in order[:n_train]],
        "val": [int(x) for x in order[n_train : n_train + n_val]],
        "test": [int(x) for x in order[n_train + n_val :]],
    }
    assert_no_episode_leakage(split)
    return split


def load_curated_tables(curated_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(curated_root)
    timesteps = pd.read_parquet(root / "timesteps.parquet")
    windows = pd.read_parquet(root / "training_windows.parquet")
    return timesteps, windows


def frames_covered_by_policy(windows: pd.DataFrame, policy: str) -> set[tuple[int, int]]:
    """Return {(episode_index, frame_index)} covered by valid training windows."""
    covered: set[tuple[int, int]] = set()
    sub = windows[windows["policy"] == policy]
    for row in sub.itertuples(index=False):
        ep = int(row.episode_index)
        for f in range(int(row.start_frame_index), int(row.end_frame_index_exclusive)):
            covered.add((ep, f))
    return covered


def select_export_rows(
    timesteps: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: OpenVLAAdapterConfig,
) -> pd.DataFrame:
    covered = frames_covered_by_policy(windows, cfg.curation_policy)
    df = timesteps.copy()
    keys = list(zip(df["episode_index"].astype(int), df["frame_index"].astype(int)))
    in_window = np.array([k in covered for k in keys], dtype=bool)
    hard_ok = df["hard_valid"].to_numpy().astype(bool) if "hard_valid" in df.columns else np.ones(len(df), bool)
    keep = in_window & hard_ok
    out = df.loc[keep].copy()
    # action target with offset
    offset = int(cfg.action_offset_frames)
    if offset != 0:
        # drop rows without same-episode future target
        rows = []
        for ep, g in out.groupby("episode_index"):
            g = g.sort_values("frame_index")
            fi = g["frame_index"].to_numpy()
            # map frame -> row
            for i, f in enumerate(fi):
                tgt = f + offset
                if tgt in set(fi):
                    rows.append(g.iloc[i])
        out = pd.DataFrame(rows)
    out["valid_for_training"] = True
    out["invalidity_reasons"] = [[] for _ in range(len(out))]
    return out.reset_index(drop=True)


def build_examples_dataframe(
    timesteps: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: OpenVLAAdapterConfig,
    split: dict[str, list[int]],
    normalizer: ActionNormalizer,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = select_export_rows(timesteps, windows, cfg)
    ep_to_split = {e: s for s, eps in split.items() for e in eps}
    rows = rows[rows["episode_index"].isin(ep_to_split)].copy()
    rows["split"] = rows["episode_index"].map(ep_to_split)

    actions = np.stack(rows["action"].to_numpy())
    if cfg.action_mode == "delta":
        # need full-episode context for deltas : recompute from timesteps then join
        full_a = np.stack(timesteps["action"].to_numpy())
        deltas = compute_delta_actions(
            full_a,
            episode_index=timesteps["episode_index"].to_numpy(),
            frame_index=timesteps["frame_index"].to_numpy(),
            gripper_index=cfg.gripper_index,
        )
        key_to_delta = {
            (int(e), int(f)): deltas[i]
            for i, (e, f) in enumerate(
                zip(timesteps["episode_index"].to_numpy(), timesteps["frame_index"].to_numpy())
            )
        }
        actions = np.stack(
            [key_to_delta[(int(e), int(f))] for e, f in zip(rows["episode_index"], rows["frame_index"])]
        )

    encoded, mask, _ = normalizer.transform(actions)
    # apply train stats already in normalizer; clip stats on this subset
    clip_frac = float(np.mean(np.abs(encoded[:, : cfg.source_action_dim]) >= 1.0 - 1e-12))

    records = []
    for i, (_, r) in enumerate(rows.iterrows()):
        records.append(
            {
                "dataset_repo_id": cfg.dataset_repo_id,
                "dataset_revision": cfg.dataset_revision,
                "curation_policy": cfg.curation_policy,
                "split": r["split"],
                "episode_index": int(r["episode_index"]),
                "frame_index": int(r["frame_index"]),
                "timestamp": float(r["timestamp"]),
                "global_index": int(r["global_index"]) if "global_index" in r and pd.notna(r["global_index"]) else None,
                "instruction": cfg.instruction,
                "view": cfg.primary_view,
                "wrist_image_ref": {
                    "camera_key": cfg.wrist_key,
                    "episode_index": int(r["episode_index"]),
                    "frame_index": int(r["frame_index"]),
                    "timestamp": float(r["timestamp"]),
                },
                "top_image_ref": {
                    "camera_key": cfg.top_key,
                    "episode_index": int(r["episode_index"]),
                    "frame_index": int(r["frame_index"]),
                    "timestamp": float(r["timestamp"]),
                },
                "state_raw": np.asarray(r["state"], dtype=np.float64).tolist(),
                "state_smoothed": np.asarray(r["state_smoothed"], dtype=np.float64).tolist()
                if "state_smoothed" in r
                else None,
                "action_raw": actions[i].tolist(),
                "action_encoded": encoded[i].tolist(),
                "action_mask": mask.tolist(),
                "action_offset_frames": cfg.action_offset_frames,
                "image_quality_flags": {
                    "hard_valid": bool(r.get("hard_valid", True)),
                    "decode_failure": bool(r.get("decode_failure", False)),
                    "overexposed": bool(r.get("overexposed", False)),
                    "exact_duplicate": bool(r.get("exact_duplicate", False)),
                },
                "source_identity": {
                    "dataset_revision": cfg.dataset_revision,
                    "episode_index": int(r["episode_index"]),
                    "frame_index": int(r["frame_index"]),
                    "timestamp": float(r["timestamp"]),
                },
                "valid_for_training": True,
                "invalidity_reasons": [],
            }
        )
    meta = {
        "n_examples": len(records),
        "n_by_split": {s: int((rows["split"] == s).sum()) for s in ("train", "val", "test")},
        "clip_fraction_on_export": clip_frac,
        "curation_policy": cfg.curation_policy,
        "n_windows": int((windows["policy"] == cfg.curation_policy).sum()),
    }
    return pd.DataFrame(records), meta


def fit_normalizer_on_train(
    timesteps: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: OpenVLAAdapterConfig,
    split: dict[str, list[int]],
) -> tuple[ActionNormalizer, dict[str, Any]]:
    rows = select_export_rows(timesteps, windows, cfg)
    train_eps = set(split["train"])
    train_rows = rows[rows["episode_index"].isin(train_eps)]
    actions = np.stack(train_rows["action"].to_numpy())
    if cfg.action_mode == "delta":
        full_a = np.stack(timesteps["action"].to_numpy())
        deltas = compute_delta_actions(
            full_a,
            episode_index=timesteps["episode_index"].to_numpy(),
            frame_index=timesteps["frame_index"].to_numpy(),
            gripper_index=cfg.gripper_index,
        )
        key_to_delta = {
            (int(e), int(f)): deltas[i]
            for i, (e, f) in enumerate(
                zip(timesteps["episode_index"].to_numpy(), timesteps["frame_index"].to_numpy())
            )
        }
        actions = np.stack(
            [
                key_to_delta[(int(e), int(f))]
                for e, f in zip(train_rows["episode_index"], train_rows["frame_index"])
            ]
        )
    normalizer = ActionNormalizer.fit(
        actions,
        q_low=cfg.q_low,
        q_high=cfg.q_high,
        gripper_index=cfg.gripper_index,
        pad_to_openvla_dim=cfg.pad_to_openvla_dim,
        openvla_dim=cfg.openvla_action_dim,
        action_mode=cfg.action_mode,
    )
    _, _, stats = normalizer.transform(actions)
    info = {
        **normalizer.to_dict(),
        "fit_on_split": "train",
        "n_train_frames": int(len(train_rows)),
        "train_clip_fraction": stats["clip_fraction"],
        "frozen_for_val_test": True,
    }
    return normalizer, info


def temporal_position(frame_index: int, max_frame: int) -> str:
    if max_frame <= 0:
        return "early"
    frac = frame_index / max_frame
    if frac <= 0.25:
        return "early"
    if frac >= 0.75:
        return "late"
    return "middle"


def pick_diverse_rows(
    rows: pd.DataFrame,
    split: dict[str, list[int]],
    *,
    batch_size: int = 6,
) -> list[pd.Series]:
    """Deterministic batch covering train/val/test and early/middle/late when possible."""
    ep_to_split = {e: s for s, eps in split.items() for e in eps}
    work = rows.copy()
    work["split"] = work["episode_index"].map(ep_to_split)
    work = work.dropna(subset=["split"]).sort_values(["episode_index", "frame_index"])

    ep_max = work.groupby("episode_index")["frame_index"].transform("max")
    work = work.assign(
        temporal_position=[
            temporal_position(int(f), int(m)) for f, m in zip(work["frame_index"], ep_max)
        ]
    )

    targets = [
        ("train", "early"),
        ("train", "middle"),
        ("train", "late"),
        ("val", "middle"),
        ("test", "early"),
        ("test", "late"),
    ]
    picked: list[pd.Series] = []
    used_keys: set[tuple[int, int]] = set()

    for split_name, pos in targets:
        if len(picked) >= batch_size:
            break
        cand = work[(work["split"] == split_name) & (work["temporal_position"] == pos)]
        for _, r in cand.iterrows():
            key = (int(r["episode_index"]), int(r["frame_index"]))
            if key in used_keys:
                continue
            picked.append(r)
            used_keys.add(key)
            break

    if len(picked) < batch_size:
        for _, r in work.iterrows():
            key = (int(r["episode_index"]), int(r["frame_index"]))
            if key in used_keys:
                continue
            picked.append(r)
            used_keys.add(key)
            if len(picked) >= batch_size:
                break

    return picked[:batch_size]


def save_json(obj: Any, path: str | Path) -> None:
    from openarm_pipeline.paths import sanitize_for_artifact

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sanitize_for_artifact(obj), f, indent=2)
        f.write("\n")


def export_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
