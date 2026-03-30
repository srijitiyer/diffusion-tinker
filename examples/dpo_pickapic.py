"""DiffusionDPO on Pick-a-Pic preference dataset.

Requires: pip install diffusion-tinker[data]
"""

from diffusion_tinker import DiffusionDPOConfig, DiffusionDPOTrainer

config = DiffusionDPOConfig(
    dataset_name="yuvalkirstain/pickapic_v2",
    dataset_split="train",
    image_column_winner="jpg_0",
    image_column_loser="jpg_1",
    caption_column="caption",
    label_column="label_0",
    beta=5000,
    train_batch_size=2,
    resolution=512,
    learning_rate=1e-5,
    lora_rank=16,
    max_train_steps=2000,
    output_dir="./dpo_pickapic_output",
)

trainer = DiffusionDPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    config=config,
)

trainer.train()
