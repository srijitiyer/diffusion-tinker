from __future__ import annotations

from dataclasses import dataclass

from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


@dataclass
class SFTConfig(BaseDiffusionConfig):
    """Configuration for diffusion SFT (Supervised Fine-Tuning).

    Standard diffusion denoising loss on a dataset of images. No RL, no rewards.
    This is the simplest training method and the baseline for all RL methods.
    """

    # Dataset
    train_dataset: str | None = None  # HF dataset name or local image folder
    dataset_split: str = "train"
    image_column: str = "image"
    caption_column: str = "text"

    # Training
    train_batch_size: int = 4
    max_train_steps: int | None = None  # If set, overrides num_epochs
    dataloader_num_workers: int = 4

    # Timestep sampling
    timestep_sampling: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0

    # Condition dropout for CFG
    condition_dropout: float = 0.1

    # Examples held back from training and used to report a validation loss.
    # SFT optimizes a denoising loss rather than a reward, so "did it work" is
    # whether the loss falls on data the model is not fitting - the training loss
    # falls either way. This existed only as a run-script callback, which meant
    # anyone using SFTTrainer directly got no held-out signal at all. 0 disables.
    holdout_size: int = 0
    # Steps between validation passes. The held-out batch is encoded once and its
    # noise and timesteps are fixed, so a pass is one forward and cheap.
    eval_every_steps: int = 100
