"""SFT on Naruto dataset - fine-tune SD3.5 to generate anime-style images.

Requires: pip install diffusion-tinker[data]
"""

from diffusion_tinker import SFTConfig, SFTTrainer

config = SFTConfig(
    train_dataset="lambdalabs/naruto-blip-captions",
    image_column="image",
    caption_column="text",
    train_batch_size=2,
    resolution=512,
    learning_rate=1e-4,
    lora_rank=16,
    lora_alpha=32,
    max_train_steps=500,
    log_every=10,
    save_every=100,
    output_dir="./sft_naruto_output",
)

trainer = SFTTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    config=config,
)

trainer.train()
