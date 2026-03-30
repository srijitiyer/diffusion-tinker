from __future__ import annotations

from dataclasses import dataclass

from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


@dataclass
class DRaFTConfig(BaseDiffusionConfig):
    """Configuration for DRaFT (Direct Reward Fine-Tuning).

    DRaFT backpropagates reward gradients through truncated denoising chains.
    Requires a differentiable reward function (aesthetic, clip_score).

    DRaFT-1: backprop through only the last denoising step (fastest)
    DRaFT-K: backprop through last K steps (better but more memory)
    """

    # Number of denoising steps to backprop through (from the end)
    # 1 = DRaFT-1, K = DRaFT-K, None = full chain (AlignProp-style)
    truncation_steps: int = 1

    # Gradient accumulation across multiple samples before optimizer step
    gradient_accumulation_steps: int = 4

    # No PPO, no advantages - direct reward maximization
    # clip_range is unused for DRaFT
