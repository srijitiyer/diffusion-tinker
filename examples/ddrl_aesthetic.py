"""DDRL + SD3.5-Medium + Aesthetic Reward

Train SD3.5-Medium to generate more aesthetically pleasing images using DDRL.

DDRL requires a real image dataset for its forward KL data-regularization term.
Without it, the model drifts from the base policy and eventually collapses.
Here we use yuvalkirstain/pickapic_v2 as the anchor dataset.

Requirements:
    pip install diffusion-tinker[data]
    GPU with >= 24GB VRAM (A5000, A6000, A100)
    HF_TOKEN env var set (for gated SD3.5 model)
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
]

config = DDRLConfig(
    # DDRL-specific: forward KL regularization anchored to real images
    data_beta=0.01,
    train_dataset="yuvalkirstain/pickapic_v2",
    use_monotonic_transform=True,
    condition_dropout=0.2,
    # Training
    num_epochs=50,
    save_every=10,
    eval_every=5,
    early_stop_patience=3,
    output_dir="./ddrl_aesthetic_output",
)

trainer = DDRLTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="aesthetic",
    train_prompts=prompts,
    config=config,
)

trainer.train()
