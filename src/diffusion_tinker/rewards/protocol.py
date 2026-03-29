from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Union

import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusion_tinker.rewards.base import BaseReward


@dataclass
class RewardContext:
    """All information available to a reward function."""

    images: list[Image.Image]
    prompts: list[str]
    device: torch.device = torch.device("cpu")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardOutput:
    """Structured output from a reward function."""

    scores: torch.Tensor  # shape (B,), float32
    metadata: dict[str, Any] = field(default_factory=dict)


RewardFunc = Union[str, Callable, "BaseReward", list]
