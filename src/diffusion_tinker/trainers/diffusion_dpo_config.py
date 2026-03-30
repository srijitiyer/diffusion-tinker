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
