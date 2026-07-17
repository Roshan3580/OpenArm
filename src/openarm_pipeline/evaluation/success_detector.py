"""Offline wrist-camera terminal-completion proxy classifier (Task 4 bonus).

CRITICAL: The source dataset has no verified success/failure labels.
This trains a terminal-completion-state proxy classifier, NOT a true
failure-aware task-success detector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from openarm_pipeline.evaluation.metrics import (
    average_precision,
    balanced_accuracy,
    bootstrap_metric_ci,
    brier_score,
    confusion_counts,
    expected_calibration_error,
    precision_recall_f1,
    roc_auc,
    specificity,
)


PROHIBITED_FEATURE_NAMES = {
    "frame_index",
    "timestamp",
    "episode_index",
    "episode_id",
    "pct_through_episode",
    "episode_length",
    "n_frames",
}


@dataclass
class DetectorConfig:
    seed: int = 42
    resize: tuple[int, int] = (96, 96)
    hsv_bins: tuple[int, int, int] = (8, 8, 8)
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (16, 16)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    positive_final_fraction: float = 0.15
    negative_first_fraction: float = 0.40
    frames_per_episode_pos: int = 4
    frames_per_episode_neg: int = 4
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20
    l2_C: float = 1.0
    max_iter: int = 200


def episode_grouped_split(
    episode_ids: list[int] | np.ndarray,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    seed: int = 42,
) -> dict[str, list[int]]:
    eps = sorted({int(e) for e in episode_ids})
    rng = np.random.default_rng(seed)
    order = eps.copy()
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # ensure all assigned
    while n_train + n_val > n - 1:
        n_val = max(0, n_val - 1)
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    assert abs((len(train) / n) - train_frac) < 0.15 or n < 10
    sets = {"train": train, "val": val, "test": test}
    # no overlap
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    return {k: [int(x) for x in v] for k, v in sets.items()}


def assert_no_episode_leakage(split: dict[str, list[int]]) -> None:
    all_eps = []
    for k, v in split.items():
        all_eps.extend(v)
    if len(all_eps) != len(set(all_eps)):
        raise AssertionError("episode leakage across splits")


def build_proxy_labels_for_episode(
    episode_index: int,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    *,
    cfg: DetectorConfig,
) -> list[dict[str, Any]]:
    """Construct proxy +/- labels; excludes middle ambiguous band."""
    fi = np.asarray(frame_indices, dtype=int)
    ts = np.asarray(timestamps, dtype=float)
    order = np.argsort(fi)
    fi = fi[order]
    ts = ts[order]
    n = len(fi)
    if n < 8:
        return []
    neg_end = int(np.floor(cfg.negative_first_fraction * n))
    pos_start = int(np.floor((1.0 - cfg.positive_final_fraction) * n))
    neg_idx = np.linspace(0, max(neg_end - 1, 0), num=min(cfg.frames_per_episode_neg, max(neg_end, 1)), dtype=int)
    pos_idx = np.linspace(pos_start, n - 1, num=min(cfg.frames_per_episode_pos, n - pos_start), dtype=int)
    rows = []
    for i in neg_idx:
        rows.append(
            {
                "episode_index": int(episode_index),
                "frame_index": int(fi[i]),
                "timestamp": float(ts[i]),
                "proxy_label": 0,
                "proxy_label_name": "early_nonterminal",
                "global_pos_in_episode": int(i),
                "episode_length": int(n),
            }
        )
    for i in pos_idx:
        rows.append(
            {
                "episode_index": int(episode_index),
                "frame_index": int(fi[i]),
                "timestamp": float(ts[i]),
                "proxy_label": 1,
                "proxy_label_name": "terminal_completion_proxy",
                "global_pos_in_episode": int(i),
                "episode_length": int(n),
            }
        )
    return rows


def build_proxy_manifest(df: pd.DataFrame, cfg: DetectorConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep, g in df.groupby("episode_index"):
        rows.extend(
            build_proxy_labels_for_episode(
                int(ep),
                g["frame_index"].to_numpy(),
                g["timestamp"].to_numpy() if "timestamp" in g.columns else g["frame_index"].to_numpy() / 30.0,
                cfg=cfg,
            )
        )
    return rows


def resize_rgb(frame_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def hsv_histogram(rgb: np.ndarray, bins: tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def hog_descriptor(rgb: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    win = (cfg.resize[1], cfg.resize[0])
    cell = cfg.hog_pixels_per_cell
    block = (
        cfg.hog_cells_per_block[0] * cell[0],
        cfg.hog_cells_per_block[1] * cell[1],
    )
    hog = cv2.HOGDescriptor(
        _winSize=win,
        _blockSize=block,
        _blockStride=cell,
        _cellSize=cell,
        _nbins=cfg.hog_orientations,
    )
    feat = hog.compute(gray)
    if feat is None:
        return np.zeros(1, dtype=np.float64)
    return feat.flatten().astype(np.float64)


def extract_features(rgb: np.ndarray, cfg: DetectorConfig, *, mode: str = "full") -> np.ndarray:
    """Pixel-only features. mode: full (HSV+HOG) or hsv."""
    if mode == "hsv":
        return hsv_histogram(rgb, cfg.hsv_bins)
    return np.concatenate([hsv_histogram(rgb, cfg.hsv_bins), hog_descriptor(rgb, cfg)])


def feature_names(cfg: DetectorConfig, *, mode: str = "full") -> list[str]:
    n_hsv = int(np.prod(cfg.hsv_bins))
    names = [f"hsv_{i}" for i in range(n_hsv)]
    if mode == "full":
        # HOG length depends on geometry; filled after first extract
        names.append("hog_block")
    for banned in PROHIBITED_FEATURE_NAMES:
        assert banned not in names
    return names


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> Standardizer:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (X - self.mean_) / self.std_


class LogisticRegressionL2:
    """Binary logistic regression with L2 penalty and optional class weights."""

    def __init__(self, C: float = 1.0, max_iter: int = 200, class_weight: str | None = "balanced") -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.w_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionL2:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if self.class_weight == "balanced":
            n_pos = max((y == 1).sum(), 1)
            n_neg = max((y == 0).sum(), 1)
            w_pos = n / (2 * n_pos)
            w_neg = n / (2 * n_neg)
            sample_w = np.where(y == 1, w_pos, w_neg)
        else:
            sample_w = np.ones(n)

        def loss(theta: np.ndarray) -> float:
            logits = X @ theta
            # stable softplus-style NLL
            nll = sample_w * (np.logaddexp(0, logits) - y * logits)
            return float(nll.mean() + (0.5 / self.C) * np.dot(theta, theta) / n)

        def grad(theta: np.ndarray) -> np.ndarray:
            logits = X @ theta
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            g = (X.T @ (sample_w * (p - y))) / n + (theta / self.C) / n
            return g

        theta0 = np.zeros(d)
        res = minimize(loss, theta0, jac=grad, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.w_ = res.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.w_ is not None
        logits = np.asarray(X, dtype=float) @ self.w_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return np.vstack([1 - p1, p1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class MajorityBaseline:
    def __init__(self) -> None:
        self.majority_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MajorityBaseline:
        y = np.asarray(y).astype(int)
        self.majority_ = int(np.round(y.mean())) if len(y) else 0
        # prefer majority count
        self.majority_ = int(1 if y.sum() >= (len(y) - y.sum()) else 0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = float(self.majority_)
        n = len(X)
        return np.vstack([np.full(n, 1 - p), np.full(n, p)]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def select_threshold_on_validation(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    *,
    metric: str = "f1",
) -> float:
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (probs_val >= t).astype(int)
        if metric == "f1":
            score = precision_recall_f1(y_val, pred)["f1"]
        else:
            score = balanced_accuracy(y_val, pred)
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t


def classification_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    *,
    groups: np.ndarray | None = None,
    seed: int = 42,
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    pred = (probs >= threshold).astype(int)
    pr = precision_recall_f1(y_true, pred)
    conf = confusion_counts(y_true, pred)
    acc = float(np.mean(pred == y_true)) if len(y_true) else float("nan")
    report = {
        "threshold": float(threshold),
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy(y_true, pred),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "f1": pr["f1"],
        "specificity": specificity(y_true, pred),
        "auroc": roc_auc(y_true, probs),
        "average_precision": average_precision(y_true, probs),
        "confusion": conf,
        "brier": brier_score(y_true, probs),
        "ece": expected_calibration_error(y_true, probs),
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
    }
    if groups is not None and len(y_true):
        report["bootstrap_ci"] = {
            "f1": bootstrap_metric_ci(
                y_true,
                pred,
                lambda yt, yp: precision_recall_f1(yt, yp)["f1"],
                groups=groups,
                n_resamples=n_bootstrap,
                seed=seed,
            ),
            "auroc": bootstrap_metric_ci(
                y_true,
                probs,
                roc_auc,
                groups=groups,
                n_resamples=n_bootstrap,
                seed=seed + 1,
            ),
            "accuracy": bootstrap_metric_ci(
                y_true,
                pred,
                lambda yt, yp: float(np.mean(yt == yp)),
                groups=groups,
                n_resamples=n_bootstrap,
                seed=seed + 2,
            ),
        }
    return report


def decode_frame_at_global_index(cap: cv2.VideoCapture, global_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(global_index))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    return bgr


def save_json(obj: Any, path: str | Path) -> None:
    from openarm_pipeline.paths import sanitize_for_artifact

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sanitize_for_artifact(obj), f, indent=2)
        f.write("\n")


def inspect_final_frames(
    video_path: str,
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    n_episodes: int = 10,
) -> dict[str, Any]:
    """Deterministic final-frame contact sheet for proxy-label sanity check."""
    eps = sorted(df["episode_index"].unique())[:n_episodes]
    caps = cv2.VideoCapture(video_path)
    tiles = []
    notes = []
    # global index = row order in concatenated video
    df_sorted = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    for ep in eps:
        g = df_sorted[df_sorted["episode_index"] == ep]
        last_global = int(g.index[-1])
        bgr = decode_frame_at_global_index(caps, last_global)
        if bgr is None:
            notes.append({"episode_index": int(ep), "status": "decode_failed"})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (160, 120))
        tiles.append(small)
        notes.append(
            {
                "episode_index": int(ep),
                "frame_index": int(g["frame_index"].iloc[-1]),
                "timestamp": float(g["timestamp"].iloc[-1]) if "timestamp" in g.columns else None,
                "status": "ok",
                "visual_note": (
                    "Final frame captured for inspection. Terminal appearance is consistent with "
                    "end-of-episode task context in this dataset, but this does NOT prove each "
                    "episode succeeded (no verified success labels)."
                ),
            }
        )
    caps.release()
    if tiles:
        # 2x5 grid
        while len(tiles) < 10:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i : i + 5]) for i in range(0, 10, 5)]
        sheet = np.vstack(rows)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return {
        "n_inspected": len(notes),
        "episodes": notes,
        "limitation": (
            "Final-frame inspection cannot establish verified task success; "
            "proxy positives remain position-derived terminal-completion proxies."
        ),
    }
