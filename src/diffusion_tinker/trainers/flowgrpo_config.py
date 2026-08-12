from __future__ import annotations

from dataclasses import dataclass

from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


@dataclass
class FlowGRPOConfig(BaseDiffusionConfig):
    """Configuration for FlowGRPO (Flow-based Group Relative Policy Optimization).

    FlowGRPO uses the same SDE sampling infrastructure as DDRL but with:
    - Standard GRPO advantages (no monotonic transform)
    - Optional reverse KL regularization via reference model
    - No data loss term
    """

    # KL regularization (0 = disabled, no reference model needed)
    kl_beta: float = 0.0

    # PPO clip range. DDRLConfig declares none and inherits the same 1e-3 from base since advantages aren't compressed)
    # 1e-3, not 0.2: the log-prob is mean-reduced, so the importance ratio stays
    # within 0.0025 of 1 and a 0.2 clip can never bind. See base_config for the
    # measurement. This redeclaration shadowed the base default, so fixing that
    # one alone left this trainer running unclipped.
    clip_range: float = 1e-3

    # GRPO-Guard: normalize importance ratios per group
    use_grpo_guard: bool = False

    # Denoising reduction: train on fewer timesteps than sampling (None = use all)
    num_train_timesteps: int | None = None
