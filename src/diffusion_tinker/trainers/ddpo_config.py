from __future__ import annotations

from dataclasses import dataclass

from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


@dataclass
class DDPOConfig(BaseDiffusionConfig):
    """Configuration for DDPO (Denoising Diffusion Policy Optimization).

    DDPO (Black et al., 2023) frames denoising as a multi-step MDP and
    applies PPO. This is the original RL method for diffusion models.

    On flow matching models (SD3, FLUX), DDPO reduces to FlowGRPO without
    the GRPO-specific enhancements (no group normalization, no guard).
    """

    # PPO clip range
    clip_range: float = 0.2

    # Per-step KL regularization (DPOK-style, 0 = disabled)
    kl_beta: float = 0.0

    # Number of PPO epochs per batch (re-iterate over the same trajectories)
    ppo_epochs: int = 1
