"""Task 4 evaluation metrics and aggregations."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    """Wilson score 95% CI for a binomial proportion (default z≈1.96)."""
    if n <= 0:
        return {"p": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return {
        "p": float(phat),
        "low": float(max(0.0, center - margin)),
        "high": float(min(1.0, center + margin)),
        "n": int(n),
        "successes": int(successes),
    }


def binary_success_rate(outcomes: Iterable[str | bool], success_value: str | bool = "success") -> dict[str, Any]:
    vals = list(outcomes)
    n = len(vals)
    if n == 0:
        return {"rate": float("nan"), "n": 0, "wilson": wilson_interval(0, 0)}
    if isinstance(success_value, bool):
        succ = sum(1 for v in vals if bool(v) is success_value)
    else:
        succ = sum(1 for v in vals if v == success_value)
    w = wilson_interval(succ, n)
    return {"rate": w["p"], "n": n, "successes": succ, "wilson": w}


def failure_taxonomy_rates(failure_labels: Iterable[Iterable[str] | str]) -> dict[str, Any]:
    """Aggregate multi-label or single failure taxonomy counts and rates over episodes."""
    counts: Counter[str] = Counter()
    n = 0
    for item in failure_labels:
        n += 1
        if item is None:
            continue
        if isinstance(item, str):
            labels = [item] if item else []
        else:
            labels = list(item)
        for lab in labels:
            counts[lab] += 1
    rates = {k: float(v / n) if n else 0.0 for k, v in sorted(counts.items())}
    return {"n_episodes": n, "counts": dict(counts), "rates": rates}


def success_by_slice(
    outcomes: Iterable[str],
    slices: Iterable[str],
    success_value: str = "success",
) -> dict[str, Any]:
    by: dict[str, list[str]] = {}
    for o, s in zip(outcomes, slices):
        by.setdefault(s, []).append(o)
    out = {s: binary_success_rate(v, success_value=success_value) for s, v in by.items()}
    rates = {s: r["rate"] for s, r in out.items() if r["n"] > 0}
    worst = min(rates, key=rates.get) if rates else None
    return {
        "by_slice": out,
        "worst_slice": worst,
        "worst_slice_rate": rates.get(worst) if worst else None,
    }


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = tn + fp
    return float(tn / denom) if denom else float("nan")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = confusion_counts(y_true, y_pred)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else float("nan")
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan")
    if not np.isfinite(prec) or not np.isfinite(rec) or (prec + rec) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sens = precision_recall_f1(y_true, y_pred)["recall"]
    spec = specificity(y_true, y_pred)
    if not np.isfinite(sens) or not np.isfinite(spec):
        return float("nan")
    return float(0.5 * (sens + spec))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney / Wilcoxon form
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    # simple tie handling
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for i, c in enumerate(counts):
            if c > 1:
                idx = np.where(inv == i)[0]
                ranks[idx] = ranks[idx].mean()
    sum_pos = ranks[y == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    denom = tp + fp
    precision = np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    recall = tp / max(int(y.sum()), 1)
    # AP as sum of precision at each positive
    return float(np.sum(precision * (y_sorted == 1)) / max(int(y.sum()), 1))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    scores_or_pred: np.ndarray,
    metric_fn,
    *,
    groups: np.ndarray | None = None,
    n_resamples: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Episode-grouped bootstrap CI when groups provided; else sample-level."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    x = np.asarray(scores_or_pred)
    point = float(metric_fn(y, x))
    stats = []
    if groups is None:
        n = len(y)
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            stats.append(float(metric_fn(y[idx], x[idx])))
    else:
        g = np.asarray(groups)
        uniq = np.unique(g)
        for _ in range(n_resamples):
            sample_eps = rng.choice(uniq, size=len(uniq), replace=True)
            masks = [g == e for e in sample_eps]
            idx = np.concatenate([np.where(m)[0] for m in masks]) if masks else np.array([], dtype=int)
            if len(idx) == 0:
                continue
            stats.append(float(metric_fn(y[idx], x[idx])))
    arr = np.asarray(stats, dtype=float)
    arr = arr[np.isfinite(arr)]
    alpha = (1 - ci) / 2
    if len(arr) == 0:
        return {"point": point, "low": float("nan"), "high": float("nan")}
    return {
        "point": point,
        "low": float(np.quantile(arr, alpha)),
        "high": float(np.quantile(arr, 1 - alpha)),
        "n_resamples": int(len(arr)),
    }
