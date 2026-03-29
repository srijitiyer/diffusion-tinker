"""DDRL + SD3.5-Medium + Aesthetic Reward - Demo Script

Train a diffusion model to generate more aesthetically pleasing images
using DDRL (Data-regularized Reinforcement Learning).

Requirements:
    pip install -e ".[rewards]"
    GPU with >= 24GB VRAM (A100, RTX 4090)
"""

from diffusion_tinker import DDRLConfig, DDRLTrainer

prompts = [
    "a photograph of a mountain landscape at golden hour",
    "a portrait of a cat sitting on a windowsill",
    "an oil painting of a city street in the rain",
    "a macro photograph of a flower with morning dew",
    "a watercolor painting of a sailboat on calm water",
    "a photograph of a cozy library with warm lighting",
    "an illustration of a forest path in autumn",
    "a photograph of ocean waves crashing on rocks",
    "a digital art piece of a futuristic cityscape",
    "a photograph of a starry night sky over a desert",
    "a painting of a mountain village at sunrise",
    "a photograph of a field of sunflowers",
    "an artistic photograph of light through stained glass",
    "a photograph of a misty forest in the morning",
    "a painting of a Japanese garden in spring",
    "a photograph of northern lights over a frozen lake",
]

config = DDRLConfig(
    data_beta=0.01,
    num_samples_per_prompt=4,
    num_inference_steps=10,
    num_eval_inference_steps=28,
    guidance_scale=7.0,
    noise_level=0.7,
    resolution=512,
    learning_rate=1e-4,
    lora_rank=32,
    lora_alpha=64,
    clip_range=1e-4,
    num_epochs=50,
    save_every=10,
    eval_every=5,
    output_dir="./ddrl_aesthetic_output",
    gradient_checkpointing=True,
    mixed_precision="bf16",
)

trainer = DDRLTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="aesthetic",
    train_prompts=prompts,
    config=config,
)

trainer.train()
