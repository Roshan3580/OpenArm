"""Egocentric / camera quality metrics and audit helpers.

Empirical wrist-camera claims require viewpoint == verified_egocentric.
Metric functions are reusable and unit-tested on synthetic images.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_gray_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert HxWx3 or HxW array to uint8 grayscale."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # ITU-R BT.601 luma
        rgb = arr[..., :3].astype(np.float64)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError(f"unsupported frame shape: {arr.shape}")
    if gray.dtype != np.uint8:
        if np.nanmax(gray) <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def laplacian_variance(frame: np.ndarray) -> float:
    """Blur/sharpness proxy: variance of Laplacian (OpenCV if available, else numpy)."""
    gray = to_gray_uint8(frame).astype(np.float64)
    try:
        import cv2

        lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
        return float(lap.var())
    except Exception:
        # 3x3 Laplacian kernel fallback
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from numpy.lib.stride_tricks import sliding_window_view

        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0
        windows = sliding_window_view(gray, (3, 3))
        lap = np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))
        return float(lap.var())


def exposure_stats(frame: np.ndarray) -> dict[str, float]:
    gray = to_gray_uint8(frame).astype(np.float64)
    mean = float(np.mean(gray))
    underexposed_frac = float(np.mean(gray <= 5))
    overexposed_frac = float(np.mean(gray >= 250))
    return {
        "mean_luma": mean,
        "underexposed_pixel_frac": underexposed_frac,
        "overexposed_pixel_frac": overexposed_frac,
    }


def shannon_entropy(frame: np.ndarray, bins: int = 256) -> float:
    """Histogram entropy in bits — very low values may indicate blank/occluded frames."""
    gray = to_gray_uint8(frame)
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256), density=False)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def frame_mse(a: np.ndarray, b: np.ndarray) -> float:
    aa = to_gray_uint8(a).astype(np.float64)
    bb = to_gray_uint8(b).astype(np.float64)
    if aa.shape != bb.shape:
        # resize-free: compare center crop min shape
        h = min(aa.shape[0], bb.shape[0])
        w = min(aa.shape[1], bb.shape[1])
        aa = aa[:h, :w]
        bb = bb[:h, :w]
    return float(np.mean((aa - bb) ** 2))


def should_compare_adjacent_frames(
    episode_a: int,
    frame_a: int,
    episode_b: int,
    frame_b: int,
    *,
    require_consecutive_frame_index: bool = True,
) -> bool:
    """Return True only for same-episode consecutive frame pairs.

    Cross-episode boundaries must never be compared in full-stream audits.
    """
    if int(episode_a) != int(episode_b):
        return False
    if require_consecutive_frame_index and int(frame_b) != int(frame_a) + 1:
        return False
    return True


def within_episode_adjacent_pair_count(n_frames: int, n_episodes: int) -> int:
    """Denominator for within-episode adjacent duplicate rates: N_frames - N_episodes."""
    return max(int(n_frames) - int(n_episodes), 0)


def classify_frame_duplicate(
    a: np.ndarray,
    b: np.ndarray,
    near_lossless_mse: float = 1.0,
    near_mse: float = 25.0,
) -> dict[str, Any]:
    """Classify adjacent-frame similarity into mutually exclusive categories.

    Categories:
      - exact_duplicate: decoded pixel arrays identical (array equality)
      - near_lossless_duplicate: 0 < MSE <= near_lossless_mse (default 1.0)
      - near_duplicate: near_lossless_mse < MSE <= near_mse (default 25.0)
      - none: MSE > near_mse

    MSE is mean squared error over all decoded pixel channels (uint8 scale).
    """
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        h = min(aa.shape[0], bb.shape[0])
        w = min(aa.shape[1], bb.shape[1])
        if aa.ndim == 3:
            aa = aa[:h, :w]
            bb = bb[:h, :w]
        else:
            aa = aa[:h, :w]
            bb = bb[:h, :w]
    identical = bool(np.array_equal(aa, bb))
    if identical:
        mse = 0.0
        cat = "exact_duplicate"
    else:
        mse = float(np.mean((aa.astype(np.float64) - bb.astype(np.float64)) ** 2))
        if mse <= near_lossless_mse:
            cat = "near_lossless_duplicate"
        elif mse <= near_mse:
            cat = "near_duplicate"
        else:
            cat = "none"
    return {
        "mse": mse,
        "category": cat,
        "exact_duplicate": cat == "exact_duplicate",
        "near_lossless_duplicate": cat == "near_lossless_duplicate",
        "near_duplicate": cat == "near_duplicate",
    }


def is_exact_or_near_duplicate(
    a: np.ndarray,
    b: np.ndarray,
    exact_mse: float = 1.0,
    near_mse: float = 25.0,
) -> dict[str, Any]:
    """Backward-compatible wrapper; prefer classify_frame_duplicate.

    Note: ``exact_mse`` is treated as the near-lossless upper bound (MSE<=that,
    excluding true exact). True exact requires array equality / MSE==0.
    """
    return classify_frame_duplicate(
        a, b, near_lossless_mse=exact_mse, near_mse=near_mse
    )


def score_frame(frame: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Compute transparent quality proxies for a single RGB/gray frame."""
    ego = config.get("egocentric", config)
    sharp = laplacian_variance(frame)
    exp = exposure_stats(frame)
    entropy = shannon_entropy(frame)

    blur_thr = float(ego.get("laplacian_var_blur_threshold", 50.0))
    under_mean = float(ego.get("underexposure_mean_threshold", 40.0))
    over_mean = float(ego.get("overexposure_mean_threshold", 220.0))
    under_frac = float(ego.get("underexposure_sat_frac", 0.15))
    over_frac = float(ego.get("overexposure_sat_frac", 0.15))
    low_h = float(ego.get("low_entropy_threshold", 3.5))

    flags = {
        "blur_suspect": sharp < blur_thr,
        "underexposed": exp["mean_luma"] < under_mean or exp["underexposed_pixel_frac"] >= under_frac,
        "overexposed": exp["mean_luma"] > over_mean or exp["overexposed_pixel_frac"] >= over_frac,
        "low_entropy_possible_occlusion": entropy < low_h,
    }
    return {
        "laplacian_var": sharp,
        "entropy_bits": entropy,
        **exp,
        "flags": flags,
        "thresholds": {
            "laplacian_var_blur_threshold": blur_thr,
            "underexposure_mean_threshold": under_mean,
            "overexposure_mean_threshold": over_mean,
            "underexposure_sat_frac": under_frac,
            "overexposure_sat_frac": over_frac,
            "low_entropy_threshold": low_h,
        },
        "heuristic_notes": [
            "low_entropy is an imperfect occlusion/blank proxy, not a verified occlusion detector",
            "laplacian variance is a blur proxy and depends on scene texture",
        ],
    }


def summarize_metric_list(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "p05": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def audit_egocentric_frames(
    frames: list[dict[str, Any]],
    config: dict[str, Any],
    camera_key: str,
    viewpoint: str,
    alignment_notes: list[str] | None = None,
) -> dict[str, Any]:
    """Audit a list of sampled frames. Expects dicts with optional 'frame' ndarray."""
    if viewpoint != "verified_egocentric":
        return {
            "status": "blocked",
            "camera_key": camera_key,
            "viewpoint": viewpoint,
            "reason": (
                "Empirical egocentric metrics are only reported for cameras classified "
                "verified_egocentric. External/ambiguous streams are not treated as wrist POV."
            ),
            "n_frames_provided": len(frames),
            "supported_metrics": [
                "laplacian_variance (blur proxy)",
                "exposure mean / sat fractions",
                "shannon entropy (occlusion/blank heuristic)",
                "exact/near duplicate consecutive frames",
                "missing/undecodable frame counts",
                "camera–state temporal alignment checks",
            ],
            "alignment_notes": alignment_notes or [],
        }

    ego = config.get("egocentric", config)
    exact_mse = float(ego.get("duplicate_mse_threshold", 1.0))
    near_mse = float(ego.get("near_duplicate_mse_threshold", 25.0))

    scores = []
    missing = 0
    undecodable = 0
    valid_frames: list[np.ndarray] = []
    flag_counts = {
        "blur_suspect": 0,
        "underexposed": 0,
        "overexposed": 0,
        "low_entropy_possible_occlusion": 0,
    }

    for item in frames:
        if item.get("error"):
            if "undecodable" in str(item["error"]):
                undecodable += 1
            else:
                missing += 1
            continue
        frame = item.get("frame")
        if frame is None:
            missing += 1
            continue
        s = score_frame(frame, config)
        s["episode_index"] = item.get("episode_index")
        s["frame_index"] = item.get("frame_index")
        scores.append(s)
        valid_frames.append(frame)
        for k in flag_counts:
            if s["flags"].get(k):
                flag_counts[k] += 1

    dup_exact = 0
    dup_near_lossless = 0
    dup_near = 0
    n_pairs = 0
    # Sampled diagnostic mode: compare consecutive samples only when they share
    # an episode and have consecutive frame_index (refuse non-consecutive).
    for i in range(1, len(scores)):
        ep_a = scores[i - 1].get("episode_index")
        fi_a = scores[i - 1].get("frame_index")
        ep_b = scores[i].get("episode_index")
        fi_b = scores[i].get("frame_index")
        if ep_a is None or fi_a is None or ep_b is None or fi_b is None:
            continue
        if not should_compare_adjacent_frames(int(ep_a), int(fi_a), int(ep_b), int(fi_b)):
            continue
        n_pairs += 1
        d = classify_frame_duplicate(
            valid_frames[i - 1],
            valid_frames[i],
            near_lossless_mse=exact_mse,
            near_mse=near_mse,
        )
        if d["exact_duplicate"]:
            dup_exact += 1
        elif d["near_lossless_duplicate"]:
            dup_near_lossless += 1
        elif d["near_duplicate"]:
            dup_near += 1

    n_valid = len(scores)
    return {
        "status": "completed",
        "camera_key": camera_key,
        "viewpoint": viewpoint,
        "n_frames_sampled": len(frames),
        "n_valid": n_valid,
        "n_missing": missing,
        "n_undecodable": undecodable,
        "metric_distributions": {
            "laplacian_var": summarize_metric_list([s["laplacian_var"] for s in scores]),
            "mean_luma": summarize_metric_list([s["mean_luma"] for s in scores]),
            "entropy_bits": summarize_metric_list([s["entropy_bits"] for s in scores]),
        },
        "flag_counts": flag_counts,
        "flag_rates": {k: float(v / n_valid) if n_valid else 0.0 for k, v in flag_counts.items()},
        "duplicate_pairs_compared": n_pairs,
        "duplicate_pairs_exact": dup_exact,
        "duplicate_pairs_near_lossless": dup_near_lossless,
        "duplicate_pairs_near": dup_near,
        "duplicate_pair_rate_exact": float(dup_exact / max(n_pairs, 1)),
        "duplicate_pair_rate_near_lossless": float(dup_near_lossless / max(n_pairs, 1)),
        "duplicate_pair_rate_near": float(dup_near / max(n_pairs, 1)),
        "alignment_notes": alignment_notes or [],
        "per_frame_scores_sample": scores[:20],
    }


def make_contact_sheet(
    samples: list[dict[str, Any]],
    out_path: str,
    title: str,
    cols: int = 3,
) -> dict[str, Any]:
    """Save a labeled contact sheet; returns summary of what was drawn."""
    import matplotlib.pyplot as plt

    usable = [s for s in samples if s.get("frame") is not None]
    n = len(usable)
    meta = {"n_requested": len(samples), "n_drawn": n, "errors": [s.get("error") for s in samples if s.get("frame") is None]}
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "no decodable frames", ha="center", va="center")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return meta

    cols = max(1, cols)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, s in enumerate(usable):
        ax = axes[i]
        ax.imshow(s["frame"])
        metric = s.get("metric_label")
        base = f"ep={s.get('episode_index')} f={s.get('frame_index')} t={s.get('timestamp')}"
        if metric:
            base = f"{base}\n{metric}"
        ax.set_title(base, fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return meta


def edge_density(frame: np.ndarray) -> float:
    """Fraction of pixels with strong Sobel gradient magnitude."""
    gray = to_gray_uint8(frame).astype(np.float64)
    try:
        import cv2

        gx = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
    except Exception:
        mag = np.abs(np.diff(gray, axis=0, prepend=gray[:1])) + np.abs(
            np.diff(gray, axis=1, prepend=gray[:, :1])
        )
    return float(np.mean(mag > 40.0))


def audit_wrist_video_full(
    video_path: str,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    timestamp: np.ndarray,
    config: dict[str, Any],
    windows_plan: dict[str, Any] | None = None,
    decode_every_frame: bool = True,
) -> dict[str, Any]:
    """Stream-decode wrist video; score every frame or planned contiguous windows.

    Arrays must be aligned with the concatenated video frame order (global index).
    """
    import cv2

    ego = config.get("egocentric", config)
    blur_thr = float(ego.get("laplacian_var_blur_threshold", 50.0))
    under_mean = float(ego.get("underexposure_mean_threshold", 40.0))
    over_mean = float(ego.get("overexposure_mean_threshold", 220.0))
    under_frac = float(ego.get("underexposure_sat_frac", 0.15))
    over_frac = float(ego.get("overexposure_sat_frac", 0.15))
    low_h = float(ego.get("low_entropy_threshold", 3.5))
    exact_mse = float(ego.get("near_lossless_mse_threshold", ego.get("duplicate_mse_threshold", 1.0)))
    near_mse = float(ego.get("near_duplicate_mse_threshold", 25.0))

    ep = np.asarray(episode_index).reshape(-1)
    fi = np.asarray(frame_index).reshape(-1)
    ts = np.asarray(timestamp, dtype=np.float64).reshape(-1)
    n_tab = len(ep)

    # Build optional keep-mask for windowed mode
    keep = np.ones(n_tab, dtype=bool)
    if not decode_every_frame and windows_plan is not None:
        keep[:] = False
        for ep_s, wins in (windows_plan.get("windows") or {}).items():
            ep_i = int(ep_s)
            for start, end in wins:
                mask = (ep == ep_i) & (fi >= start) & (fi < end)
                keep |= mask

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "status": "failed",
            "error": "unreadable_video",
            "video_path": video_path,
        }

    container_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    records: list[dict[str, Any]] = []
    sharpness_all: list[float] = []
    luma_all: list[float] = []
    entropy_all: list[float] = []
    edge_all: list[float] = []
    under_pix: list[float] = []
    over_pix: list[float] = []

    flag_counts = {
        "blur_suspect": 0,
        "underexposed": 0,
        "overexposed": 0,
        "low_entropy_possible_occlusion": 0,
    }
    exact_dups = 0
    near_lossless_dups = 0
    near_dups = 0
    n_adjacent_pairs_compared = 0
    decode_failures = 0
    missing = 0
    prev_gray = None
    prev_global = None
    prev_ep = None
    prev_fi = None
    decoded_ok = 0

    # Candidate frames for montages
    candidates: dict[str, list[dict[str, Any]]] = {
        "normal": [],
        "low_sharpness": [],
        "exposure": [],
        "low_info": [],
        "duplicate": [],
    }

    global_i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if global_i >= n_tab:
            # extra container frames beyond tabular
            global_i += 1
            continue
        if not keep[global_i]:
            global_i += 1
            continue

        if bgr is None:
            decode_failures += 1
            missing += 1
            global_i += 1
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = score_frame(rgb, config)
        ed = edge_density(rgb)
        s["edge_density"] = ed
        decoded_ok += 1

        sharpness_all.append(s["laplacian_var"])
        luma_all.append(s["mean_luma"])
        entropy_all.append(s["entropy_bits"])
        edge_all.append(ed)
        under_pix.append(s["underexposed_pixel_frac"])
        over_pix.append(s["overexposed_pixel_frac"])

        for k in flag_counts:
            if s["flags"].get(k):
                flag_counts[k] += 1

        gray = to_gray_uint8(rgb)
        dup_info = None
        cur_ep = int(ep[global_i])
        cur_fi = int(fi[global_i])
        if (
            prev_gray is not None
            and prev_ep is not None
            and prev_fi is not None
            and should_compare_adjacent_frames(prev_ep, prev_fi, cur_ep, cur_fi)
        ):
            n_adjacent_pairs_compared += 1
            d = classify_frame_duplicate(
                prev_gray, gray, near_lossless_mse=exact_mse, near_mse=near_mse
            )
            if d["exact_duplicate"]:
                exact_dups += 1
            elif d["near_lossless_duplicate"]:
                near_lossless_dups += 1
            elif d["near_duplicate"]:
                near_dups += 1
            dup_info = d
            if d["exact_duplicate"] or d["near_lossless_duplicate"]:
                candidates["duplicate"].append(
                    {
                        "episode_index": cur_ep,
                        "frame_index": cur_fi,
                        "timestamp": float(ts[global_i]) if ts.size else None,
                        "frame": rgb,
                        "metric_label": f"{d['category']} mse={d['mse']:.2f}",
                        "laplacian_var": s["laplacian_var"],
                    }
                )

        rec = {
            "global_index": global_i,
            "episode_index": int(ep[global_i]),
            "frame_index": int(fi[global_i]),
            "timestamp": float(ts[global_i]) if ts.size else None,
            "laplacian_var": s["laplacian_var"],
            "mean_luma": s["mean_luma"],
            "entropy_bits": s["entropy_bits"],
            "edge_density": ed,
            "flags": s["flags"],
            "duplicate": dup_info,
        }
        records.append(rec)

        # montage pools
        item = {
            "episode_index": int(ep[global_i]),
            "frame_index": int(fi[global_i]),
            "timestamp": float(ts[global_i]) if ts.size else None,
            "frame": rgb,
            "laplacian_var": s["laplacian_var"],
            "mean_luma": s["mean_luma"],
            "entropy_bits": s["entropy_bits"],
        }
        if not any(s["flags"].values()):
            if len(candidates["normal"]) < 64:
                item["metric_label"] = f"sharp={s['laplacian_var']:.1f}"
                candidates["normal"].append(item)
        if s["flags"]["blur_suspect"]:
            item2 = dict(item)
            item2["metric_label"] = f"sharp={s['laplacian_var']:.1f}"
            candidates["low_sharpness"].append(item2)
        if s["flags"]["underexposed"] or s["flags"]["overexposed"]:
            item2 = dict(item)
            item2["metric_label"] = f"luma={s['mean_luma']:.1f}"
            candidates["exposure"].append(item2)
        if s["flags"]["low_entropy_possible_occlusion"]:
            item2 = dict(item)
            item2["metric_label"] = f"H={s['entropy_bits']:.2f}"
            candidates["low_info"].append(item2)

        prev_gray = gray
        prev_global = global_i
        prev_ep = cur_ep
        prev_fi = cur_fi
        global_i += 1

    # If video ended early
    if global_i < n_tab and decode_every_frame:
        missing += int(np.sum(keep[global_i:]))

    cap.release()
    n_valid = len(sharpness_all)

    # Sort montage candidates
    candidates["low_sharpness"] = sorted(candidates["low_sharpness"], key=lambda x: x["laplacian_var"])[:16]
    candidates["exposure"] = sorted(
        candidates["exposure"], key=lambda x: abs(x["mean_luma"] - 128), reverse=True
    )[:16]
    candidates["low_info"] = sorted(candidates["low_info"], key=lambda x: x["entropy_bits"])[:16]
    candidates["duplicate"] = candidates["duplicate"][:16]
    candidates["normal"] = candidates["normal"][:16]

    return {
        "status": "completed",
        "video_path": video_path,
        "decode_every_frame": decode_every_frame,
        "sampling": windows_plan,
        "n_tabular_rows": n_tab,
        "container_n_frames": container_n,
        "n_frames_scored": n_valid,
        "n_successfully_decoded": decoded_ok,
        "n_decode_failures": decode_failures,
        "n_missing": missing,
        "coverage_frac_of_tabular": float(n_valid / n_tab) if n_tab else 0.0,
        "episodes_represented": sorted({int(r["episode_index"]) for r in records}),
        "metric_distributions": {
            "laplacian_var": summarize_metric_list(sharpness_all),
            "mean_luma": summarize_metric_list(luma_all),
            "entropy_bits": summarize_metric_list(entropy_all),
            "edge_density": summarize_metric_list(edge_all),
            "underexposed_pixel_frac": summarize_metric_list(under_pix),
            "overexposed_pixel_frac": summarize_metric_list(over_pix),
        },
        "flag_counts": flag_counts,
        "flag_rates": {k: float(v / n_valid) if n_valid else 0.0 for k, v in flag_counts.items()},
        "duplicate_adjacent_pairs_denominator": n_adjacent_pairs_compared,
        "duplicate_adjacent_pairs_expected": within_episode_adjacent_pair_count(
            n_valid,
            int(len(np.unique(ep[keep]))) if n_valid else 0,
        ),
        "duplicate_accounting": {
            "scope": "within_episode_adjacent_only",
            "cross_episode_pairs_excluded": True,
            "require_consecutive_frame_index": True,
        },
        "duplicate_adjacent_exact": exact_dups,
        "duplicate_adjacent_near_lossless": near_lossless_dups,
        "duplicate_adjacent_near": near_dups,
        "duplicate_exact_rate": float(exact_dups / max(n_adjacent_pairs_compared, 1)),
        "duplicate_near_lossless_rate": float(near_lossless_dups / max(n_adjacent_pairs_compared, 1)),
        "duplicate_near_rate": float(near_dups / max(n_adjacent_pairs_compared, 1)),
        "duplicate_categories_note": (
            "Mutually exclusive within-episode pairs only "
            "(never across episode boundaries): exact (array-equal), "
            "near_lossless (0<MSE<=1), near (1<MSE<=25)"
        ),
        "threshold_types": {
            "laplacian_var_blur_threshold": {
                "value": blur_thr,
                "type": "exploratory_heuristic",
                "note": "absolute Laplacian-variance cutoff; scene-dependent",
            },
            "underexposure_mean_threshold": {"value": under_mean, "type": "hard_engineering_threshold"},
            "overexposure_mean_threshold": {"value": over_mean, "type": "hard_engineering_threshold"},
            "underexposure_sat_frac": {"value": under_frac, "type": "hard_engineering_threshold"},
            "overexposure_sat_frac": {"value": over_frac, "type": "hard_engineering_threshold"},
            "low_entropy_threshold": {
                "value": low_h,
                "type": "exploratory_heuristic",
                "note": "low entropy ≠ proven occlusion; may be blank, blur, or close-up self-occlusion",
            },
            "near_lossless_mse_threshold": {
                "value": exact_mse,
                "type": "hard_engineering_threshold",
                "note": "0 < MSE <= this => near_lossless_duplicate; MSE==0 => exact_duplicate",
            },
            "near_duplicate_mse_threshold": {
                "value": near_mse,
                "type": "distribution_derived_screening_rule",
            },
        },
        "occlusion_interpretation_guide": {
            "unusable_camera_occlusion": "near-zero entropy + near-black/white for sustained runs",
            "expected_self_occlusion": "gripper/object fills frame during grasp; still informative",
            "useful_closeup": "object/gripper dominate but edges/texture remain",
            "low_info_background": "empty table / motion away from workspace",
            "needs_human_review": "ambiguous mid-entropy dark frames",
        },
        "per_frame_records_compact": records[:: max(1, len(records) // 500)][:500],
        "arrays_for_cross_modal": {
            "laplacian_var": sharpness_all,
            "low_info_flag": [bool(r["flags"]["low_entropy_possible_occlusion"]) for r in records],
            "exact_dup_flag": [
                bool(r["duplicate"] and r["duplicate"]["exact_duplicate"]) if r.get("duplicate") else False
                for r in records
            ],
            "near_lossless_dup_flag": [
                bool(r["duplicate"] and r["duplicate"].get("near_lossless_duplicate"))
                if r.get("duplicate")
                else False
                for r in records
            ],
            "near_dup_flag": [
                bool(r["duplicate"] and r["duplicate"]["near_duplicate"]) if r.get("duplicate") else False
                for r in records
            ],
            "global_index": [r["global_index"] for r in records],
            "episode_index": [r["episode_index"] for r in records],
            "frame_index": [r["frame_index"] for r in records],
        },
        "montage_candidates": {
            k: [
                {
                    "episode_index": c["episode_index"],
                    "frame_index": c["frame_index"],
                    "timestamp": c["timestamp"],
                    "metric_label": c.get("metric_label"),
                    "laplacian_var": c.get("laplacian_var"),
                    "mean_luma": c.get("mean_luma"),
                    "entropy_bits": c.get("entropy_bits"),
                    "_has_frame": c.get("frame") is not None,
                }
                for c in v
            ]
            for k, v in candidates.items()
        },
        "_montage_frames": candidates,
    }
