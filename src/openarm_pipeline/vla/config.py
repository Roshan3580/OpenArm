"""Configuration for OpenVLA data adaptation (Task 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OpenVLAAdapterConfig:
    # Dataset
    dataset_repo_id: str = "lerobot/svla_so100_pickplace"
    dataset_revision: str = "728583b5eaf9e739a7f119e2def466fa1d552402"
    curation_policy: str = "conservative"  # or strict
    instruction: str = "Pick up the cube and place it in the box."
    seed: int = 42

    # Split (episode-grouped)
    train_frac: float = 0.80
    val_frac: float = 0.10
    test_frac: float = 0.10

    # Cameras
    primary_view: str = "wrist"  # wrist | top | wrist_plus_top_composite_diag
    wrist_key: str = "observation.images.wrist"
    top_key: str = "observation.images.top"

    # Actions — project design for SO-100 vs OpenVLA 7-DoF tokenizer
    source_action_dim: int = 6
    openvla_action_dim: int = 7  # verified pretrained tokenizer slot count
    action_mode: str = "absolute"  # absolute | delta
    pad_to_openvla_dim: bool = True
    pad_dim_excluded_from_loss: bool = True
    gripper_index: int = 5
    normalize_method: str = "q01_q99"
    q_low: float = 0.01
    q_high: float = 0.99
    clip_to_unit: bool = True  # map into [-1, 1] for ActionTokenizer

    # Alignment
    action_offset_frames: int = 0
    max_alignment_error_frames: float = 0.5
    fps: float = 30.0

    # Images
    image_size: int = 224
    resize_method: str = "bilinear"
    color_space: str = "RGB"
    image_value_range: str = "uint8_0_255_then_processor"
    unsafe_augmentations_disabled: tuple[str, ...] = (
        "horizontal_flip",
        "vertical_flip",
        "large_rotation",
        "time_reversal",
        "frame_reorder",
    )

    # State
    use_state_as_conditioning: bool = False  # OpenVLA baseline is image+language
    state_field: str = "state"  # or state_smoothed

    # Paths
    curated_root: str = "data/curated/svla_so100_pickplace"
    export_root: str = "data/vla/svla_so100_pickplace"
    artifacts_dir: str = "artifacts/task_05_vla_adaptation"

    # Official pins (verified separately in model_reference.md)
    openvla_model_id: str = "openvla/openvla-7b"
    openvla_model_revision: str = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
    openvla_code_repo: str = "https://github.com/openvla/openvla"
    openvla_code_commit: str = "c8f03f48af69"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
