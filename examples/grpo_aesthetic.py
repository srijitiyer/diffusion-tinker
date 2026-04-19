"""FlowGRPO + SD3.5-Medium + Aesthetic Reward

The simplest example: improve image aesthetics with no dataset required.
FlowGRPO with aesthetic reward is a good first test to verify your setup.

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
    "a watercolor painting of a sailboat on calm water",
    "a photograph of a cozy library with warm lighting",
    "an illustration of a forest path in autumn",
    "a photograph of ocean waves crashing on rocks",
]

config = FlowGRPOConfig(
    num_epochs=30,
    eval_every=5,
    early_stop_patience=3,
    output_dir="./grpo_aesthetic_output",
)

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="aesthetic",
    train_prompts=prompts,
    config=config,
)

trainer.train()
