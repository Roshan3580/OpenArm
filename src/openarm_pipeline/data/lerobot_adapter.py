"""LeRobot / Hugging Face dataset adapter.

Imports are side-effect free with respect to downloads: callers must invoke
explicit load/inspect helpers. Prefer Hugging Face Hub metadata + parquet when
LeRobot APIs are unavailable or incompatible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

IMAGE_DTYPES = {"image", "video"}
STATE_CANDIDATES = (
    "observation.state",
    "observation.states",
    "state",
    "states",
)
ACTION_CANDIDATES = ("action", "actions")


@dataclass
class CameraStreamInfo:
    key: str
    dtype: str
    shape: list[int] | None
    names: Any
    video_info: dict[str, Any] | None
    viewpoint: str  # verified_egocentric | verified_external | ambiguous
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    repo_id: str
    revision: str | None
    split_names: list[str]
    size_info: dict[str, Any]
    episode_count: int | None
    total_frames: int | None
    fps: float | None
    feature_schema: dict[str, Any]
    state_keys: list[str]
    action_keys: list[str]
    timestamp_keys: list[str]
    frame_index_keys: list[str]
    episode_index_keys: list[str]
    task_keys: list[str]
    image_video_features: list[str]
    cameras: list[CameraStreamInfo]
    dataset_stats: dict[str, Any] | None
    inspection_timestamp: str
    package_versions: dict[str, str]
    access_method: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cameras"] = [c.to_dict() if isinstance(c, CameraStreamInfo) else c for c in self.cameras]
        return d


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "pyarrow",
        "datasets",
        "huggingface_hub",
        "cv2",
        "PIL",
        "yaml",
        "scipy",
        "matplotlib",
        "lerobot",
    ):
        try:
            if name == "cv2":
                import cv2

                versions["opencv-python-headless"] = getattr(cv2, "__version__", "unknown")
            elif name == "PIL":
                import PIL

                versions["Pillow"] = getattr(PIL, "__version__", "unknown")
            elif name == "yaml":
                import yaml as _yaml

                versions["PyYAML"] = getattr(_yaml, "__version__", "unknown")
            else:
                mod = __import__(name)
                versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name if name not in ("cv2", "PIL", "yaml") else name] = "not_installed"
    return versions


def _is_image_or_video_feature(feat: dict[str, Any]) -> bool:
    dtype = str(feat.get("dtype", "")).lower()
    return dtype in IMAGE_DTYPES or "image" in str(feat.get("dtype", "")).lower()


def discover_keys(features: dict[str, Any]) -> dict[str, list[str]]:
    """Discover modality keys from a LeRobot feature schema."""
    keys = list(features.keys())
    state_keys = [k for k in STATE_CANDIDATES if k in features]
    action_keys = [k for k in ACTION_CANDIDATES if k in features]
    image_video = [k for k, v in features.items() if isinstance(v, dict) and _is_image_or_video_feature(v)]
    # Also catch keys with images. in the name even if dtype missing
    for k in keys:
        if "images." in k or k.startswith("observation.images") or ".image" in k:
            if k not in image_video:
                image_video.append(k)

    return {
        "state_keys": state_keys,
        "action_keys": action_keys,
        "timestamp_keys": [k for k in ("timestamp", "timestamps", "time") if k in features],
        "frame_index_keys": [k for k in ("frame_index", "frame_idx", "observation.frame_index") if k in features],
        "episode_index_keys": [k for k in ("episode_index", "episode_idx", "episode") if k in features],
        "task_keys": [k for k in ("task_index", "task", "task_name") if k in features],
        "image_video_features": image_video,
        "all_keys": keys,
    }


def classify_camera_viewpoint(key: str, feature: dict[str, Any] | None = None) -> CameraStreamInfo:
    """Classify camera viewpoint from feature naming and metadata.

    Never labels a stream verified_egocentric from visual appearance alone.
    """
    feature = feature or {}
    key_l = key.lower()
    evidence: list[str] = [f"feature_key={key}"]
    dtype = str(feature.get("dtype", "unknown"))
    shape = feature.get("shape")
    names = feature.get("names")
    video_info = feature.get("video_info") or feature.get("info")
    evidence.append(f"dtype={dtype}")
    if shape is not None:
        evidence.append(f"shape={shape}")

    egocentric_tokens = (
        "wrist",
        "ego",
        "egocentric",
        "hand",
        "gripper_cam",
        "endeffector",
        "end_effector",
        "first_person",
        "fpv",
    )
    external_tokens = (
        "top",
        "front",
        "side",
        "external",
        "third_person",
        "overhead",
        "birdseye",
        "bird_eye",
        "agentview",
        "agent_view",
        "workspace",
        "cam_high",
        "cam_low",
    )

    hit_ego = [t for t in egocentric_tokens if t in key_l]
    hit_ext = [t for t in external_tokens if t in key_l]

    if hit_ego and not hit_ext:
        viewpoint = "verified_egocentric"
        evidence.append(f"key contains egocentric placement token(s): {hit_ego}")
    elif hit_ext and not hit_ego:
        viewpoint = "verified_external"
        evidence.append(f"key contains external placement token(s): {hit_ext}")
        if "top" in hit_ext:
            evidence.append(
                "ALOHA 'top' cameras are overhead/workspace views, not wrist-mounted egocentric streams"
            )
    elif hit_ego and hit_ext:
        viewpoint = "ambiguous"
        evidence.append(f"conflicting tokens ego={hit_ego} external={hit_ext}")
    else:
        viewpoint = "ambiguous"
        evidence.append(
            "no definitive placement token in feature key; visual appearance alone is insufficient"
        )

    return CameraStreamInfo(
        key=key,
        dtype=dtype,
        shape=list(shape) if shape is not None else None,
        names=names,
        video_info=video_info,
        viewpoint=viewpoint,
        evidence=evidence,
    )


def fetch_info_json(repo_id: str, revision: str | None = None) -> tuple[dict[str, Any], str | None, str]:
    """Fetch meta/info.json from the Hub. Returns (info, resolved_revision, access_method)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    resolved = revision
    try:
        info_meta = api.dataset_info(repo_id, revision=revision)
        resolved = getattr(info_meta, "sha", None) or revision
    except Exception:
        resolved = revision

    path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/info.json",
        repo_type="dataset",
        revision=revision,
    )
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    return info, resolved, "huggingface_hub:meta/info.json"


def _try_load_stats(repo_id: str, revision: str | None) -> dict[str, Any] | None:
    from huggingface_hub import hf_hub_download

    for candidate in ("meta/stats.json", "meta/episodes_stats.json", "meta/episodes.jsonl"):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=candidate,
                repo_type="dataset",
                revision=revision,
            )
            if candidate.endswith(".json"):
                with open(path, encoding="utf-8") as f:
                    return {"source": candidate, "data": json.load(f)}
            # jsonl: return light summary only
            rows = []
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    if line.strip():
                        rows.append(json.loads(line))
            return {"source": candidate, "sample_rows": rows, "note": "first_5_rows_only"}
        except Exception:
            continue
    return None


def build_manifest(
    repo_id: str,
    revision: str | None = None,
    split: str | None = None,
) -> DatasetManifest:
    """Inspect dataset metadata without loading full video payloads."""
    notes: list[str] = []
    info, resolved_rev, access = fetch_info_json(repo_id, revision=revision)
    features = info.get("features") or {}
    discovered = discover_keys(features)
    cameras = [
        classify_camera_viewpoint(k, features.get(k, {})) for k in discovered["image_video_features"]
    ]

    splits = info.get("splits") or {}
    split_names = list(splits.keys()) if isinstance(splits, dict) else list(splits or [])
    if split and split not in split_names and split_names:
        notes.append(f"requested split '{split}' not listed in meta splits={split_names}; proceeding")

    size_info = {
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_videos": info.get("total_videos"),
        "total_chunks": info.get("total_chunks"),
        "total_tasks": info.get("total_tasks"),
        "files_size_in_mb": info.get("files_size_in_mb"),
        "chunks_size": info.get("chunks_size"),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "data_path": info.get("data_path"),
        "video_path": info.get("video_path"),
        "splits": splits,
    }

    stats = _try_load_stats(repo_id, revision)
    if stats is None:
        notes.append("no dataset-level stats.json found on Hub")

    # Attempt LeRobot API for cross-check; never fail hard
    try:
        import lerobot  # noqa: F401

        notes.append(f"lerobot package available: {getattr(__import__('lerobot'), '__version__', 'unknown')}")
    except Exception:
        notes.append(
            "lerobot package not installed; using huggingface_hub + parquet fallback for tabular access"
        )

    return DatasetManifest(
        repo_id=repo_id,
        revision=resolved_rev,
        split_names=split_names,
        size_info=size_info,
        episode_count=info.get("total_episodes"),
        total_frames=info.get("total_frames"),
        fps=float(info["fps"]) if info.get("fps") is not None else None,
        feature_schema=features,
        state_keys=discovered["state_keys"],
        action_keys=discovered["action_keys"],
        timestamp_keys=discovered["timestamp_keys"],
        frame_index_keys=discovered["frame_index_keys"],
        episode_index_keys=discovered["episode_index_keys"],
        task_keys=discovered["task_keys"],
        image_video_features=discovered["image_video_features"],
        cameras=cameras,
        dataset_stats=stats,
        inspection_timestamp=datetime.now(timezone.utc).isoformat(),
        package_versions=package_versions(),
        access_method=access,
        notes=notes,
    )


def load_episodes_dataframe(repo_id: str, revision: str | None = None) -> pd.DataFrame:
    """Load LeRobot v3 meta/episodes parquet (concat all chunks)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    ep_files = sorted(f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No meta/episodes parquet for {repo_id}")
    frames = []
    for rel in ep_files:
        local = hf_hub_download(
            repo_id=repo_id, filename=rel, repo_type="dataset", revision=revision
        )
        frames.append(pd.read_parquet(local))
    return pd.concat(frames, ignore_index=True)


def resolve_camera_video_path(
    repo_id: str,
    camera_key: str,
    revision: str | None = None,
) -> str:
    """Download/resolve the primary video file for a camera key (v2 or v3 layout)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    # Prefer v3: videos/{key}/chunk-*/file-*.mp4
    key_files = sorted(
        f for f in files if f.endswith(".mp4") and f.startswith(f"videos/{camera_key}/")
    )
    if not key_files:
        key_files = sorted(f for f in files if f.endswith(".mp4") and camera_key in f)
    if not key_files:
        raise FileNotFoundError(f"No video for camera {camera_key} in {repo_id}")
    return hf_hub_download(
        repo_id=repo_id, filename=key_files[0], repo_type="dataset", revision=revision
    )


def dataset_slug(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def list_parquet_files(repo_id: str, revision: str | None = None) -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.endswith(".parquet") and f.startswith("data/"))


def load_tabular_dataframe(
    repo_id: str,
    revision: str | None = None,
    max_episodes: int | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load all (or episode-limited) parquet tables into a single DataFrame.

    Video/image columns are excluded by default to keep memory bounded.
    """
    from huggingface_hub import hf_hub_download

    parquet_files = list_parquet_files(repo_id, revision=revision)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under data/ for {repo_id}")

    frames: list[pd.DataFrame] = []
    kept_episodes: set[int] = set()

    for rel in parquet_files:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=rel,
            repo_type="dataset",
            revision=revision,
        )
        df = pd.read_parquet(local)
        # Drop heavy embedded image columns if present
        drop_cols = [c for c in df.columns if "image" in c.lower() or c.startswith("observation.images")]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            # Always keep episode_index for filtering
            if "episode_index" in df.columns and "episode_index" not in keep:
                keep = ["episode_index", *keep]
            df = df[keep]

        if max_episodes is not None and "episode_index" in df.columns:
            # Accumulate until we have enough distinct episodes
            if len(kept_episodes) >= max_episodes:
                # only keep episodes already selected
                df = df[df["episode_index"].isin(kept_episodes)]
            else:
                for ep in sorted(df["episode_index"].unique()):
                    if len(kept_episodes) < max_episodes:
                        kept_episodes.add(int(ep))
                df = df[df["episode_index"].isin(kept_episodes)]

        if len(df):
            frames.append(df)

        if max_episodes is not None and len(kept_episodes) >= max_episodes:
            # Continue scanning only if more files might contain same episodes; otherwise stop early
            # For chunked layouts, early stop when we already filtered enough episodes from seen files.
            pass

    if not frames:
        raise RuntimeError("No tabular rows loaded")

    out = pd.concat(frames, ignore_index=True)
    if max_episodes is not None and "episode_index" in out.columns:
        # Ensure hard cap on episode count (lowest indices)
        eps = sorted(out["episode_index"].unique())[:max_episodes]
        out = out[out["episode_index"].isin(eps)].reset_index(drop=True)
    return out


def stack_vector_column(series: pd.Series) -> np.ndarray:
    """Stack a parquet column of fixed-size vectors into (N, D)."""
    values = series.to_numpy()
    if len(values) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    first = values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=0)
    # scalar column
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def sample_video_frames(
    repo_id: str,
    video_key: str,
    episode_indices: list[int],
    frame_positions: list[float],
    revision: str | None = None,
    info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Sample decoded frames from Hub videos for contact sheets.

    frame_positions are fractions in [0, 1] within each episode.
    Supports LeRobot v2 per-episode mp4s and v3 concatenated videos with
    episode from_timestamp/to_timestamp metadata.
    """
    import cv2
    from huggingface_hub import hf_hub_download, list_repo_files

    if info is None:
        info, _, _ = fetch_info_json(repo_id, revision=revision)
    fps = float(info.get("fps") or 30.0)

    files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    key_files = sorted(f for f in files if f.endswith(".mp4") and f"/{video_key}/" in f)
    if not key_files:
        key_files = sorted(f for f in files if f.endswith(".mp4") and video_key in f)
    if not key_files:
        return [
            {
                "episode_index": ep,
                "position": pos,
                "frame": None,
                "error": f"no video files found for key={video_key}",
            }
            for ep in episode_indices
            for pos in frame_positions
        ]

    ep_to_file: dict[int, str] = {}
    for f in key_files:
        name = Path(f).stem
        if name.startswith("episode_"):
            try:
                ep_to_file[int(name.split("_")[-1])] = f
            except ValueError:
                continue

    # v3 episode timestamp map
    ep_meta = None
    try:
        ep_meta = load_episodes_dataframe(repo_id, revision=revision)
    except Exception:
        ep_meta = None

    results: list[dict[str, Any]] = []
    v3_mode = not ep_to_file and key_files
    # Cache opened captures by path
    local_cache: dict[str, str] = {}

    for ep in episode_indices:
        rel = ep_to_file.get(ep)
        if rel is None and v3_mode:
            # Prefer episode meta file_index if present
            if ep_meta is not None and f"videos/{video_key}/file_index" in ep_meta.columns:
                row = ep_meta[ep_meta["episode_index"] == ep]
                if len(row):
                    fi = int(row.iloc[0][f"videos/{video_key}/file_index"])
                    ci = int(row.iloc[0].get(f"videos/{video_key}/chunk_index", 0))
                    candidate = f"videos/{video_key}/chunk-{ci:03d}/file-{fi:03d}.mp4"
                    if candidate in files:
                        rel = candidate
            if rel is None:
                rel = key_files[0]
        if rel is None:
            for pos in frame_positions:
                results.append(
                    {
                        "episode_index": ep,
                        "position": pos,
                        "frame": None,
                        "error": "episode video not found",
                    }
                )
            continue

        if rel not in local_cache:
            local_cache[rel] = hf_hub_download(
                repo_id=repo_id, filename=rel, repo_type="dataset", revision=revision
            )
        local = local_cache[rel]
        cap = cv2.VideoCapture(local)
        if not cap.isOpened():
            for pos in frame_positions:
                results.append(
                    {
                        "episode_index": ep,
                        "position": pos,
                        "frame": None,
                        "error": "undecodable_video",
                        "path": rel,
                    }
                )
            continue

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps) or fps

        for pos in frame_positions:
            idx = None
            if v3_mode and ep_meta is not None:
                row = ep_meta[ep_meta["episode_index"] == ep]
                from_key = f"videos/{video_key}/from_timestamp"
                to_key = f"videos/{video_key}/to_timestamp"
                if len(row) and from_key in row.columns:
                    t0 = float(row.iloc[0][from_key])
                    t1 = float(row.iloc[0][to_key])
                    t = t0 + float(pos) * (t1 - t0)
                    idx = int(np.clip(round(t * vid_fps), 0, max(n_frames - 1, 0)))
            if idx is None:
                # v2 per-episode file: position within that file
                idx = int(np.clip(round(pos * max(n_frames - 1, 0)), 0, max(n_frames - 1, 0)))

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                results.append(
                    {
                        "episode_index": ep,
                        "position": pos,
                        "frame": None,
                        "error": "undecodable_frame",
                        "path": rel,
                        "frame_index": idx,
                    }
                )
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results.append(
                    {
                        "episode_index": ep,
                        "position": pos,
                        "frame": rgb,
                        "error": None,
                        "path": rel,
                        "frame_index": idx,
                        "n_frames": n_frames,
                    }
                )
        cap.release()

    return results


def save_json(obj: Any, path: str | Path) -> None:
    from openarm_pipeline.paths import sanitize_for_artifact, sanitize_path_string

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_for_artifact(obj)

    def _json_default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return sanitize_path_string(str(o))
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
        f.write("\n")
