"""Training-window construction and curated-view loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml


def build_training_windows(
    timesteps: pd.DataFrame,
    horizon: int,
    stride: int,
    policy: str,
) -> pd.DataFrame:
    """Build consecutive within-episode training windows for a policy.

    Requires columns: episode_index, frame_index, timestamp, global_index,
    hard_valid, soft_exclude_strict (bool).
    """
    if policy not in ("conservative", "strict"):
        raise ValueError(f"unknown policy: {policy}")
    rows = []
    for ep, g in timesteps.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index").reset_index(drop=True)
        hard = g["hard_valid"].to_numpy()
        if policy == "strict":
            ok = hard & (~g["soft_exclude_strict"].to_numpy())
        else:
            ok = hard
        fi = g["frame_index"].to_numpy()
        ts = g["timestamp"].to_numpy()
        gi = g["global_index"].to_numpy()
        n = len(g)
        for start in range(0, n - horizon + 1, stride):
            end = start + horizon
            sl = slice(start, end)
            if not np.all(ok[sl]):
                continue
            # consecutive frame indices
            if not np.array_equal(fi[sl], np.arange(fi[start], fi[start] + horizon)):
                continue
            if not np.all(np.diff(ts[sl]) > 0):
                continue
            rows.append(
                {
                    "policy": policy,
                    "episode_index": int(ep),
                    "start_frame_index": int(fi[start]),
                    "end_frame_index_exclusive": int(fi[start] + horizon),
                    "start_global_index": int(gi[start]),
                    "end_global_index_exclusive": int(gi[start] + horizon),
                    "start_timestamp": float(ts[start]),
                    "end_timestamp": float(ts[end - 1]),
                    "horizon": int(horizon),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class CuratedView:
    root: Path
    manifest: dict[str, Any]
    timesteps: pd.DataFrame
    windows: pd.DataFrame
    episodes: pd.DataFrame

    @classmethod
    def load(cls, root: str | Path) -> "CuratedView":
        root = Path(root)
        with open(root / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        return cls(
            root=root,
            manifest=manifest,
            timesteps=pd.read_parquet(root / "timesteps.parquet"),
            windows=pd.read_parquet(root / "training_windows.parquet"),
            episodes=pd.read_parquet(root / "episodes.parquet"),
        )

    def windows_for_policy(self, policy: str) -> pd.DataFrame:
        return self.windows[self.windows["policy"] == policy].reset_index(drop=True)

    def iter_windows(self, policy: str = "conservative") -> Iterator[dict[str, Any]]:
        wins = self.windows_for_policy(policy)
        for _, w in wins.iterrows():
            yield self.resolve_window(w)

    def resolve_window(self, w: pd.Series | dict[str, Any]) -> dict[str, Any]:
        w = dict(w)
        ep = int(w["episode_index"])
        start = int(w["start_frame_index"])
        end = int(w["end_frame_index_exclusive"])
        g = self.timesteps[
            (self.timesteps["episode_index"] == ep)
            & (self.timesteps["frame_index"] >= start)
            & (self.timesteps["frame_index"] < end)
        ].sort_values("frame_index")
        state = np.stack(g["state"].to_numpy())
        state_s = np.stack(g["state_smoothed"].to_numpy())
        action = np.stack(g["action"].to_numpy())
        return {
            "policy": w["policy"],
            "episode_index": ep,
            "frame_index": g["frame_index"].to_list(),
            "timestamp": g["timestamp"].to_list(),
            "global_index": g["global_index"].to_list(),
            "state": state,
            "state_smoothed": state_s,
            "action": action,
            "wrist_video_key": self.manifest["camera_keys"]["wrist"],
            "top_video_key": self.manifest["camera_keys"]["top"],
            "wrist_from_timestamp": float(g["wrist_from_timestamp"].iloc[0]),
            "wrist_frame_coords": g[["episode_index", "frame_index", "timestamp"]].to_dict("records"),
            "top_frame_coords": g[["episode_index", "frame_index", "timestamp"]].to_dict("records"),
            "source_repo_id": self.manifest["source_repo_id"],
            "source_revision": self.manifest["source_revision"],
            "source_video_refs": self.manifest.get("source_video_refs", {}),
        }


def save_yaml(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def config_hash(config: dict[str, Any]) -> str:
    import hashlib

    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
