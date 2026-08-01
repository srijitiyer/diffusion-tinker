from __future__ import annotations

from dataclasses import dataclass

from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


@dataclass
class DiffusionDPOConfig(BaseDiffusionConfig):
    """Configuration for DiffusionDPO (Direct Preference Optimization for Diffusion).

    DiffusionDPO is an offline method using preference pairs, not online RL.
    It adapts the DPO objective to diffusion models by comparing denoising errors
    between winner and loser images.
    """

    # DPO-specific
    beta: float = 5000.0  # DPO regularization (much larger than LLM DPO due to T multiplier)

    # Dataset
    dataset_name: str | None = None
    dataset_split: str = "train"
    image_column_winner: str = "jpg_0"
    image_column_loser: str = "jpg_1"
    caption_column: str = "caption"
    label_column: str | None = "label_0"

    # Training
    train_batch_size: int = 4
    max_train_steps: int = 2000
    dataloader_num_workers: int = 4

    # Held-out preference split. This trainer reported only its training loss,
    # which is the same failure mode the audit found in DRaFT: the number goes
    # down and says nothing about generalization. Set eval_split to a split the
    # training run does not touch and it reports preference accuracy - the
    # fraction of held-out pairs where the implicit reward prefers the winner,
    # which is 0.5 for an untrained adapter and is the metric the DPO paper
    # reports. None disables it.
    eval_split: str | None = None
    num_eval_pairs: int = 64
    # Denoising timesteps averaged per pair. The implicit reward is a Monte-Carlo
    # estimate over (t, noise), so a single draw per pair is far too noisy to
    # rank a pair; averaging a fixed set of draws makes the metric comparable
    # across evals.
    num_eval_timesteps: int = 8
