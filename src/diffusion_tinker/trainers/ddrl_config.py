from __future__ import annotations

from dataclasses import dataclass

from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


@dataclass
class DDRLConfig(BaseDiffusionConfig):
    """Configuration for DDRL (Data-regularized Reinforcement Learning).

    DDRL replaces the unreliable reverse KL regularization with forward KL,
    which reduces to the standard diffusion denoising loss. Key parameters:

    - data_beta: weight of the diffusion loss (forward KL). Higher = more conservative.
    - train_dataset: real-image anchor for the forward-KL data term.
    """

    # DDRL-specific
    data_beta: float = 0.01

    # Advantage computation
    beta_temp: float = 1.0  # temperature for advantage normalization
    # lambda(x) = -exp(-x) from Theorem 3.1: a risk-averse reweighting that
    # penalizes low advantages far more than it rewards high ones. Its output is
    # always negative, so it is re-centered per prompt group before use; without
    # that the PPO term discourages every sampled action and the policy collapses.
    use_monotonic_transform: bool = False

    # Timestep sampling for data loss
    timestep_sampling: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0

    # Data for forward KL regularization (denoising loss)
    # Required when data_beta > 0. Without a real dataset, data_beta is auto-disabled.
    train_dataset: str | None = None  # HF dataset name or local image folder path
    dataset_split: str = "train"
    image_column: str = "image"
    # DDRL_THEORY.md step 3b samples (x_0, c) from data and drops c with
    # probability 0.2. Without a caption column the data term can only ever be
    # unconditional, which bounds KL(p_data(x) || p_theta(x|"")) - the
    # unconditional branch - while the RL term moves p_theta(x|c) for real
    # prompts. On FLUX.1-dev, which has no unconditional branch at all, those are
    # disjoint conditioning states and the anchor cannot oppose conditional drift.
    # None keeps the old unconditional behaviour for datasets that have no
    # captions, but now as a stated choice rather than the only option.
    caption_column: str | None = None
    # Probability of dropping the caption, per DDRL_THEORY.md step 3b. Ignored
    # when caption_column is None (everything is already unconditional).
    condition_dropout: float = 0.2
