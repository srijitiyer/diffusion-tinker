"""DRaFT-1 with aesthetic reward - direct reward backprop through last denoising step."""

from diffusion_tinker import DRaFTConfig, DRaFTTrainer

prompts = [
    "a photograph of a mountain landscape at golden hour",
    "a portrait of a cat sitting on a windowsill",
    "an oil painting of a city street in the rain",
    "a macro photograph of a flower with morning dew",
    "a photograph of ocean waves crashing on rocks",
    "a painting of a Japanese garden in spring",
]

config = DRaFTConfig(
    truncation_steps=1,  # DRaFT-1 (last step only, fast)
    num_inference_steps=20,
    gradient_accumulation_steps=4,
    resolution=512,
    learning_rate=1e-5,  # lower LR for direct backprop
    lora_rank=16,
    num_epochs=30,
    save_every=10,
    output_dir="./draft_aesthetic_output",
)

trainer = DRaFTTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="aesthetic",
    train_prompts=prompts,
    config=config,
)

trainer.train()
