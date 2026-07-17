"""SO-100 action encoding compatible with OpenVLA's 7-slot discretized action space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ActionNormalizer:
    """Train-only robust per-dimension normalization into approximately [-1, 1]."""

    q_low: np.ndarray
    q_high: np.ndarray
    method: str = "q01_q99"
    eps: float = 1e-6
    source_dim: int = 6
    openvla_dim: int = 7
    pad_to_openvla_dim: bool = True
    gripper_index: int = 5
    action_mode: str = "absolute"

    @classmethod
    def fit(
        cls,
        actions: np.ndarray,
        *,
        q_low: float = 0.01,
        q_high: float = 0.99,
        gripper_index: int = 5,
        pad_to_openvla_dim: bool = True,
        openvla_dim: int = 7,
        action_mode: str = "absolute",
    ) -> ActionNormalizer:
        a = np.asarray(actions, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError("actions must be [N, D]")
        lo = np.quantile(a, q_low, axis=0)
        hi = np.quantile(a, q_high, axis=0)
        # zero-range guard
        span = hi - lo
        span[span < 1e-8] = 1.0
        hi = lo + span
        return cls(
            q_low=lo.astype(np.float64),
            q_high=hi.astype(np.float64),
            source_dim=a.shape[1],
            openvla_dim=openvla_dim,
            pad_to_openvla_dim=pad_to_openvla_dim,
            gripper_index=gripper_index,
            action_mode=action_mode,
        )

    def transform(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return (encoded [N, openvla_dim], mask [openvla_dim], stats)."""
        a = np.asarray(actions, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[1] != self.source_dim:
            raise ValueError(f"expected D={self.source_dim}, got {a.shape[1]}")
        # map each dim to [-1, 1] using train quantiles
        x = 2.0 * (a - self.q_low) / (self.q_high - self.q_low + self.eps) - 1.0
        clipped = np.clip(x, -1.0, 1.0)
        clip_frac = float(np.mean(np.abs(x) > 1.0))
        if self.pad_to_openvla_dim and self.openvla_dim > self.source_dim:
            pad = np.zeros((clipped.shape[0], self.openvla_dim - self.source_dim), dtype=np.float64)
            encoded = np.concatenate([clipped, pad], axis=1)
            mask = np.ones(self.openvla_dim, dtype=np.float64)
            mask[self.source_dim :] = 0.0  # padded dims excluded from loss
        else:
            encoded = clipped
            mask = np.ones(self.openvla_dim if not self.pad_to_openvla_dim else self.source_dim, dtype=np.float64)
        stats = {
            "clip_fraction": clip_frac,
            "n": int(a.shape[0]),
            "source_dim": self.source_dim,
            "encoded_dim": int(encoded.shape[1]),
            "pad_excluded_from_loss": bool(self.pad_to_openvla_dim),
        }
        return encoded, mask, stats

    def inverse(self, encoded: np.ndarray) -> np.ndarray:
        e = np.asarray(encoded, dtype=np.float64)
        if e.ndim == 1:
            e = e.reshape(1, -1)
        src = e[:, : self.source_dim]
        # [-1,1] -> original
        a = (src + 1.0) * 0.5 * (self.q_high - self.q_low + self.eps) + self.q_low
        return a

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "q_low": self.q_low.tolist(),
            "q_high": self.q_high.tolist(),
            "source_dim": self.source_dim,
            "openvla_dim": self.openvla_dim,
            "pad_to_openvla_dim": self.pad_to_openvla_dim,
            "gripper_index": self.gripper_index,
            "action_mode": self.action_mode,
            "note": (
                "OpenVLA pretrained ActionTokenizer expects 7 continuous dims in [-1,1] "
                "before 256-bin discretization. SO-100 provides 6 joint dims; dim 6 is "
                "zero-padded and masked out of the loss."
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionNormalizer:
        return cls(
            q_low=np.asarray(d["q_low"], dtype=np.float64),
            q_high=np.asarray(d["q_high"], dtype=np.float64),
            method=d.get("method", "q01_q99"),
            source_dim=int(d["source_dim"]),
            openvla_dim=int(d["openvla_dim"]),
            pad_to_openvla_dim=bool(d.get("pad_to_openvla_dim", True)),
            gripper_index=int(d.get("gripper_index", 5)),
            action_mode=str(d.get("action_mode", "absolute")),
        )


def compute_delta_actions(
    actions: np.ndarray,
    *,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    gripper_index: int = 5,
) -> np.ndarray:
    """Delta = a_t - a_{t-1} within episode; gripper kept absolute; first frame zeroed for arm."""
    a = np.asarray(actions, dtype=np.float64)
    ep = np.asarray(episode_index)
    fi = np.asarray(frame_index)
    out = np.zeros_like(a)
    order = np.lexsort((fi, ep))
    prev_ep = None
    prev_a = None
    prev_fi = None
    for idx in order:
        if prev_ep is None or ep[idx] != prev_ep or int(fi[idx]) != int(prev_fi) + 1:
            out[idx] = 0.0
            out[idx, gripper_index] = a[idx, gripper_index]  # absolute gripper
        else:
            out[idx] = a[idx] - prev_a
            out[idx, gripper_index] = a[idx, gripper_index]
        prev_ep = ep[idx]
        prev_a = a[idx]
        prev_fi = fi[idx]
    return out


def encode_actions(actions: np.ndarray, normalizer: ActionNormalizer) -> tuple[np.ndarray, np.ndarray, dict]:
    return normalizer.transform(actions)


def decode_actions(encoded: np.ndarray, normalizer: ActionNormalizer) -> np.ndarray:
    return normalizer.inverse(encoded)


def masked_action_loss_weights(mask: np.ndarray) -> np.ndarray:
    """Per-dimension loss weights; padded dims must be 0."""
    w = np.asarray(mask, dtype=np.float64)
    assert w.ndim == 1
    return w
