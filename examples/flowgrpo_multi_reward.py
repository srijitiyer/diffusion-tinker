"""FlowGRPO + SD3.5-Medium + Multi-Reward

Combines aesthetic and CLIP score rewards with weighted aggregation.
FlowGRPO needs no dataset anchor, so it's easy to get started with.

Requirements:
    pip install diffusion-tinker
    GPU with >= 24GB VRAM
    HF_TOKEN env var set
"""

from diffusion_tinker import FlowGRPOConfig, FlowGRPOTrainer

prompts = [
    "a photograph of a mountain landscape at golden hour",
    "a portrait of a cat sitting on a windowsill",
    "an oil painting of a city street in the rain",
    "a macro photograph of a flower with morning dew",
    "a photograph of ocean waves crashing on rocks",
    "a painting of a Japanese garden in spring",
]

config = FlowGRPOConfig(
    num_epochs=50,
    eval_every=10,
    early_stop_patience=3,
    output_dir="./flowgrpo_multi_reward_output",
)

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs=["aesthetic", "clip_score"],
    reward_weights=[0.6, 0.4],
    reward_mode="advantage_level",
    train_prompts=prompts,
    config=config,
)

trainer.train()
