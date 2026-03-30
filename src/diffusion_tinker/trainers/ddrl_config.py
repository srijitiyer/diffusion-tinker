from __future__ import annotations

from dataclasses import dataclass

from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


@dataclass
class DDRLConfig(BaseDiffusionConfig):
    """Configuration for DDRL (Data-regularized Reinforcement Learning).

    DDRL replaces the unreliable reverse KL regularization with forward KL,
    which reduces to the standard diffusion denoising loss. Key parameters:

    - data_beta: weight of the diffusion loss (forward KL). Higher = more conservative.
    - kl_beta: weight of reverse KL. Set to 0.0 for DDRL (the key innovation).
    - condition_dropout: probability of dropping text condition during data loss (enables CFG).
    """

    # DDRL-specific
    data_beta: float = 0.01
    kl_beta: float = 0.0
    condition_dropout: float = 0.2

    # Advantage computation
    beta_temp: float = 1.0  # temperature for advantage normalization
    use_monotonic_transform: bool = True  # lambda(x) = -exp(-x) from Theorem 3.1

    # Timestep sampling for data loss
    timestep_sampling: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0

    # Data for forward KL regularization (denoising loss)
    # If None, falls back to using trajectory endpoints (less effective)
    train_dataset: str | None = None  # HF dataset name or local image folder path
    dataset_split: str = "train"
    image_column: str = "image"
