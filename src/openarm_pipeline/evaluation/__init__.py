"""Evaluation package public API."""

from openarm_pipeline.evaluation.metrics import binary_success_rate, wilson_interval
from openarm_pipeline.evaluation.rollout_protocol import (
    generate_rollout_matrix,
    validate_rollout_protocol,
)

__all__ = [
    "wilson_interval",
    "binary_success_rate",
    "generate_rollout_matrix",
    "validate_rollout_protocol",
]
