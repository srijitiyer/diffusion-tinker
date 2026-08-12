from __future__ import annotations

from dataclasses import dataclass

from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


@dataclass
class DDPOConfig(BaseDiffusionConfig):
    """Configuration for DDPO (Denoising Diffusion Policy Optimization).

    DDPO (Black et al., 2023) frames denoising as a multi-step MDP and
    applies PPO. This is the original RL method for diffusion models.

    On flow matching models (SD3, FLUX) the policy-gradient step is the same
    clipped-PPO term FlowGRPO uses. What actually distinguishes DDPO is its
    advantage estimator: a rolling per-prompt reward history (see
    BufferedPerPromptStatTracker) rather than statistics taken from the samples
    of the current batch. Normalizing DDPO within the batch instead would make it
    identical to FlowGRPO up to the clip range, so it would not be a test of
    DDPO at all.
    """

    # PPO clip range
    # 1e-3, not 0.2: the log-prob is mean-reduced, so the importance ratio stays
    # within 0.0025 of 1 and a 0.2 clip can never bind. See base_config for the
    # measurement. This redeclaration shadowed the base default, so fixing that
    # one alone left this trainer running unclipped.
    clip_range: float = 1e-3

    # Per-step KL regularization (DPOK-style, 0 = disabled)
    kl_beta: float = 0.0

    # Number of PPO epochs per batch (re-iterate over the same trajectories)
    ppo_epochs: int = 1

    # Per-prompt reward history used for the advantage baseline. Until a prompt
    # has stat_min_count samples buffered, the batch statistics are used instead.
    stat_buffer_size: int = 32
    stat_min_count: int = 16
