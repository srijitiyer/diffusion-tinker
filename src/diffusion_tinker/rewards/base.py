from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput


class BaseReward(ABC):
    """Base class for all reward models."""

    name: str = "base"

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

    @abstractmethod
    def _compute(self, ctx: RewardContext) -> RewardOutput:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, ctx: RewardContext) -> RewardOutput:
        return self._compute(ctx)

    def differentiable_forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor | None:
        """Score image tensors while keeping gradients attached.

        images is (B, 3, H, W) in [0, 1]. Returns (B,) scores, or None if this
        reward can't run differentiably (DRaFT then falls back to PIL scoring).
        """
        return None

    def to(self, device: str | torch.device) -> BaseReward:
        self.device = torch.device(device)
        return self
