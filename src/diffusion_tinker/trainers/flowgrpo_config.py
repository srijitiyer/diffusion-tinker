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

    # PPO clip range (larger than DDRL's 1e-4 since advantages aren't compressed)
    clip_range: float = 0.2

    # GRPO-Guard: normalize importance ratios per group
    use_grpo_guard: bool = False

    # Denoising reduction: train on fewer timesteps than sampling (None = use all)
    num_train_timesteps: int | None = None
