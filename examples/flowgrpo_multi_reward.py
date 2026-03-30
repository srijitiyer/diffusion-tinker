"""FlowGRPO with multi-reward training on SD3.5-Medium.

Combines aesthetic and CLIP score rewards with weighted aggregation.
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
    num_samples_per_prompt=4,
    num_inference_steps=10,
    guidance_scale=7.0,
    noise_level=0.7,
    resolution=512,
    learning_rate=1e-4,
    clip_range=0.2,
    kl_beta=0.01,  # light KL regularization
    num_epochs=50,
    save_every=10,
    eval_every=10,
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
