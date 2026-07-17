#!/usr/bin/env python3
"""Run the offline wrist terminal-completion proxy detector (Task 4 bonus).

Does NOT claim genuine success/failure labels. Uses real wrist frames from the
pinned svla_so100_pickplace revision when available locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.evaluation.success_detector import (  # noqa: E402
    DetectorConfig,
    LogisticRegressionL2,
    MajorityBaseline,
    Standardizer,
    assert_no_episode_leakage,
    build_proxy_manifest,
    classification_report,
    decode_frame_at_global_index,
    episode_grouped_split,
    extract_features,
    inspect_final_frames,
    resize_rgb,
    save_json,
    select_threshold_on_validation,
)
from openarm_pipeline.evaluation.temporal_detection import (  # noqa: E402
    aggregate_episode_temporal,
    evaluate_proxy_temporal,
)


def local_snapshot_paths(revision: str) -> dict[str, Path]:
    snap = (
        ROOT
        / ".cache/huggingface/hub/datasets--lerobot--svla_so100_pickplace/snapshots"
        / revision
    )
    return {
        "snapshot": snap,
        "parquet": snap / "data/chunk-000/file-000.parquet",
        "wrist": snap / "videos/observation.images.wrist/chunk-000/file-000.mp4",
    }


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    cfg_yaml = load_config(ROOT / args.config)
    det = cfg_yaml["success_detector"]
    revision = cfg_yaml["dataset"]["revision"]
    paths = local_snapshot_paths(revision)
    art = ROOT / det["paths"]["artifacts_dir"]
    art.mkdir(parents=True, exist_ok=True)
    models_dir = ROOT / det["paths"]["models_dir"]
    cache_dir = ROOT / det["paths"]["cache_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not paths["parquet"].exists() or not paths["wrist"].exists():
        report = {
            "ok": False,
            "error": "cached_wrist_or_parquet_unavailable",
            "expected_parquet": str(paths["parquet"]),
            "expected_wrist": str(paths["wrist"]),
            "note": "Code/tests/protocol complete; egocentric detector run blocked without local cache.",
        }
        save_json(report, art / "success_detector_metrics.json")
        print(json.dumps(report, indent=2))
        return 2

    dcfg = DetectorConfig(
        seed=int(det["seed"]),
        resize=tuple(det["resize"]),
        hsv_bins=tuple(det["hsv_bins"]),
        hog_orientations=int(det["hog"]["orientations"]),
        hog_pixels_per_cell=tuple(det["hog"]["pixels_per_cell"]),
        hog_cells_per_block=tuple(det["hog"]["cells_per_block"]),
        positive_final_fraction=float(det["proxy"]["positive_final_fraction"]),
        negative_first_fraction=float(det["proxy"]["negative_first_fraction"]),
        frames_per_episode_pos=int(det["proxy"]["frames_per_episode_pos"]),
        frames_per_episode_neg=int(det["proxy"]["frames_per_episode_neg"]),
        train_frac=float(det["split"]["train"]),
        val_frac=float(det["split"]["val"]),
        test_frac=float(det["split"]["test"]),
        l2_C=float(det["logistic_regression"]["l2_C"]),
        max_iter=int(det["logistic_regression"]["max_iter"]),
    )

    cols = ["episode_index", "frame_index", "timestamp"]
    df = pd.read_parquet(paths["parquet"], columns=cols)
    if args.max_episodes is not None:
        keep = sorted(df["episode_index"].unique())[: args.max_episodes]
        df = df[df["episode_index"].isin(keep)].copy()

    # Final-frame inspection
    inspect = inspect_final_frames(
        str(paths["wrist"]),
        df,
        art / "final_frame_contact_sheet.png",
        n_episodes=10,
    )
    save_json(inspect, art / "final_frame_inspection.json")

    # Split episodes first
    split = episode_grouped_split(
        df["episode_index"].unique(),
        train_frac=dcfg.train_frac,
        val_frac=dcfg.val_frac,
        test_frac=dcfg.test_frac,
        seed=dcfg.seed,
    )
    assert_no_episode_leakage(split)
    save_json(split, art / "episode_split.json")

    # Proxy labels
    manifest = build_proxy_manifest(df, dcfg)
    # attach split
    ep_to_split = {e: s for s, eps in split.items() for e in eps}
    for row in manifest:
        row["split"] = ep_to_split[int(row["episode_index"])]
    save_json(
        {
            "limitation": (
                "Proxy labels are position-derived terminal-completion proxies, "
                "NOT verified task success/failure labels."
            ),
            "n_rows": len(manifest),
            "n_pos": sum(1 for r in manifest if r["proxy_label"] == 1),
            "n_neg": sum(1 for r in manifest if r["proxy_label"] == 0),
            "rows": manifest,
        },
        art / "proxy_label_manifest.json",
    )

    # Map (episode, frame) -> global video index
    df_sorted = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    key_to_global = {
        (int(r.episode_index), int(r.frame_index)): int(i)
        for i, r in df_sorted.iterrows()
    }

    cap = cv2.VideoCapture(str(paths["wrist"]))
    feats_full = []
    feats_hsv = []
    labels = []
    groups = []
    splits = []
    meta = []
    for row in manifest:
        gidx = key_to_global[(row["episode_index"], row["frame_index"])]
        bgr = decode_frame_at_global_index(cap, gidx)
        if bgr is None:
            continue
        rgb = resize_rgb(bgr, dcfg.resize)
        f_full = extract_features(rgb, dcfg, mode="full")
        f_hsv = extract_features(rgb, dcfg, mode="hsv")
        feats_full.append(f_full)
        feats_hsv.append(f_hsv)
        labels.append(row["proxy_label"])
        groups.append(row["episode_index"])
        splits.append(row["split"])
        meta.append(row)
    cap.release()

    Xf = np.vstack(feats_full)
    Xh = np.vstack(feats_hsv)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups, dtype=int)
    sp = np.asarray(splits)

    # cache features (gitignored)
    np.savez_compressed(
        cache_dir / "features.npz",
        X_full=Xf,
        X_hsv=Xh,
        y=y,
        groups=g,
        split=sp,
    )

    def mask(name: str) -> np.ndarray:
        return sp == name

    # Standardize on train only
    std_full = Standardizer().fit(Xf[mask("train")])
    std_hsv = Standardizer().fit(Xh[mask("train")])
    Xf_s = {k: std_full.transform(Xf[mask(k)]) for k in ("train", "val", "test")}
    Xh_s = {k: std_hsv.transform(Xh[mask(k)]) for k in ("train", "val", "test")}
    y_s = {k: y[mask(k)] for k in ("train", "val", "test")}
    g_s = {k: g[mask(k)] for k in ("train", "val", "test")}

    # Models
    main = LogisticRegressionL2(C=dcfg.l2_C, max_iter=dcfg.max_iter, class_weight="balanced")
    main.fit(Xf_s["train"], y_s["train"])
    hsv_base = LogisticRegressionL2(C=dcfg.l2_C, max_iter=dcfg.max_iter, class_weight="balanced")
    hsv_base.fit(Xh_s["train"], y_s["train"])
    maj = MajorityBaseline().fit(Xf_s["train"], y_s["train"])

    # Threshold from validation only (main model)
    val_probs = main.predict_proba(Xf_s["val"])[:, 1]
    thr = select_threshold_on_validation(y_s["val"], val_probs, metric="f1")

    # Test once
    test_probs = main.predict_proba(Xf_s["test"])[:, 1]
    hsv_probs = hsv_base.predict_proba(Xh_s["test"])[:, 1]
    maj_probs = maj.predict_proba(Xf_s["test"])[:, 1]
    # majority uses same thr 0.5 effectively
    main_rep = classification_report(
        y_s["test"],
        test_probs,
        thr,
        groups=g_s["test"],
        seed=dcfg.seed,
        n_bootstrap=int(det["bootstrap"]["n_resamples"]),
    )
    hsv_rep = classification_report(y_s["test"], hsv_probs, thr, groups=g_s["test"], seed=dcfg.seed)
    maj_rep = classification_report(y_s["test"], maj_probs, 0.5, groups=g_s["test"], seed=dcfg.seed)

    # Save model weights (gitignored)
    np.savez(
        models_dir / "logistic_full.npz",
        w=main.w_,
        mean=std_full.mean_,
        std=std_full.std_,
        threshold=thr,
        feature_dim=Xf.shape[1],
    )

    # Plots
    _plot_confusion(main_rep["confusion"], art / "confusion_matrix.png")
    _plot_pr(y_s["test"], test_probs, art / "precision_recall_curve.png")
    _plot_calibration(y_s["test"], test_probs, art / "calibration_plot.png")

    # Temporal evaluation on held-out test episodes (proxy).
    # Select a separate temporal threshold on validation episodes to control
    # early false triggers (frame F1 threshold is often too low for hysteresis).
    temporal_cfg = det["temporal"]

    def episode_prob_sequence(ep: int) -> tuple[np.ndarray, int]:
        g_ep = df_sorted[df_sorted["episode_index"] == ep]
        n = len(g_ep)
        onset = int(np.floor((1.0 - dcfg.positive_final_fraction) * n))
        idxs = list(range(0, n, 5))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        probs_seq = []
        cap_local = cv2.VideoCapture(str(paths["wrist"]))
        for local_i in idxs:
            global_i = int(g_ep.index[local_i])
            bgr = decode_frame_at_global_index(cap_local, global_i)
            if bgr is None:
                probs_seq.append(0.0)
                continue
            rgb = resize_rgb(bgr, dcfg.resize)
            feat = std_full.transform(extract_features(rgb, dcfg, mode="full").reshape(1, -1))
            probs_seq.append(float(main.predict_proba(feat)[0, 1]))
        cap_local.release()
        probs_full = np.interp(np.arange(n), idxs, probs_seq)
        return probs_full, onset

    val_eps = sorted(split["val"])
    val_seqs = [episode_prob_sequence(ep) for ep in val_eps]
    best_temp_thr = 0.8
    best_key = (1e9, -1.0)  # (early_rate, -detection_rate)
    for t_cand in np.linspace(0.35, 0.95, 25):
        early = 0
        detected = 0
        for probs_full, onset in val_seqs:
            r = evaluate_proxy_temporal(
                probs_full,
                threshold=float(t_cand),
                proxy_positive_onset=onset,
                window=int(temporal_cfg["window"]),
                votes_required=int(temporal_cfg["votes_required"]),
            )
            early += int(r["false_early_trigger"])
            detected += int(r["trigger_in_proxy_positive_region"])
        early_rate = early / max(len(val_seqs), 1)
        det_rate = detected / max(len(val_seqs), 1)
        key = (early_rate, -det_rate)
        if key < best_key:
            best_key = key
            best_temp_thr = float(t_cand)

    test_eps_all = sorted(split["test"])
    temporal_results = []
    seq_cache = {}
    for ep in test_eps_all:
        probs_full, onset = episode_prob_sequence(ep)
        seq_cache[ep] = (probs_full, onset)
        tres = evaluate_proxy_temporal(
            probs_full,
            threshold=best_temp_thr,
            proxy_positive_onset=onset,
            window=int(temporal_cfg["window"]),
            votes_required=int(temporal_cfg["votes_required"]),
        )
        tres["episode_index"] = int(ep)
        # sticky trigger still on during terminal region?
        tres["trigger_active_in_terminal"] = bool(np.any(tres["triggered"][onset:])) if len(tres["triggered"]) else False
        temporal_results.append(tres)

    example_eps = test_eps_all[:6]
    fig, axes = plt.subplots(len(example_eps), 1, figsize=(10, 2.2 * max(len(example_eps), 1)), squeeze=False)
    for ax_i, ep in enumerate(example_eps):
        probs_full, onset = seq_cache[ep]
        tres = next(r for r in temporal_results if r["episode_index"] == ep)
        ax = axes[ax_i, 0]
        ax.plot(probs_full, label="p(terminal_proxy)")
        ax.axhline(best_temp_thr, color="r", ls="--", label="temporal thr")
        ax.axvline(onset, color="g", ls=":", label="proxy+ onset")
        ax.plot(tres["triggered"].astype(float), label="trigger")
        ax.set_title(f"episode {ep} (proxy temporal)")
        ax.set_ylim(-0.05, 1.05)
        if ax_i == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(art / "temporal_detection_examples.png", dpi=120)
    plt.close(fig)

    temporal_agg = aggregate_episode_temporal(temporal_results)
    temporal_agg["frame_threshold"] = thr
    temporal_agg["temporal_threshold"] = best_temp_thr
    temporal_agg["temporal_threshold_selected_on"] = "validation_episodes"
    # JSON-safe
    if isinstance(temporal_agg.get("median_detection_latency_frames"), float) and np.isnan(
        temporal_agg["median_detection_latency_frames"]
    ):
        temporal_agg["median_detection_latency_frames"] = None

    metrics = {
        "ok": True,
        "disclaimer": (
            "Results are for a terminal-completion PROXY classifier on position-derived labels. "
            "They are NOT genuine task-success detection metrics. High scores may be easy because "
            "early vs late frames differ visually."
        ),
        "dataset_repo_id": cfg_yaml["dataset"]["repo_id"],
        "dataset_revision": revision,
        "config": {
            "resize": list(dcfg.resize),
            "hsv_bins": list(dcfg.hsv_bins),
            "hog": {
                "orientations": dcfg.hog_orientations,
                "pixels_per_cell": list(dcfg.hog_pixels_per_cell),
                "cells_per_block": list(dcfg.hog_cells_per_block),
            },
            "feature_dim_full": int(Xf.shape[1]),
            "feature_dim_hsv": int(Xh.shape[1]),
            "logistic_regression": {
                "l2_C": dcfg.l2_C,
                "max_iter": dcfg.max_iter,
                "class_weight": "balanced",
            },
            "seed": dcfg.seed,
            "threshold_selected_on": "validation",
            "threshold": thr,
        },
        "counts": {
            "train": {"n": int(mask("train").sum()), "n_pos": int(y[mask("train")].sum())},
            "val": {"n": int(mask("val").sum()), "n_pos": int(y[mask("val")].sum())},
            "test": {"n": int(mask("test").sum()), "n_pos": int(y[mask("test")].sum())},
            "episodes": {k: len(v) for k, v in split.items()},
            "proxy_manifest": {
                "n_rows": len(manifest),
                "n_pos": sum(1 for r in manifest if r["proxy_label"] == 1),
                "n_neg": sum(1 for r in manifest if r["proxy_label"] == 0),
            },
        },
        "baselines": {
            "majority": maj_rep,
            "hsv_logistic": hsv_rep,
        },
        "main_model": main_rep,
        "temporal_proxy": temporal_agg,
        "paths": {
            "models_dir": str(models_dir),
            "cache_dir": str(cache_dir),
            "artifacts_dir": str(art),
        },
    }
    save_json(metrics, art / "success_detector_metrics.json")
    print(json.dumps({k: metrics[k] for k in ("ok", "disclaimer", "counts", "main_model", "baselines", "temporal_proxy")}, indent=2, default=str))
    return 0


def _plot_confusion(conf: dict, path: Path) -> None:
    mat = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], ["pred0", "pred1"])
    ax.set_yticks([0, 1], ["true0", "true1"])
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set_title("Confusion (proxy labels)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pr(y: np.ndarray, probs: np.ndarray, path: Path) -> None:
    order = np.argsort(-probs)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve (proxy labels)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_calibration(y: np.ndarray, probs: np.ndarray, path: Path, n_bins: int = 10) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(m):
            continue
        xs.append(probs[m].mean())
        ys.append(y[m].mean())
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(xs, ys, "o-", label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Calibration (proxy labels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
