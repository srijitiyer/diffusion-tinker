"""FlowGRPO + SD3.5-Medium + OCR Reward

Train SD3.5-Medium to generate images with readable text using OCR reward.
This is the validated reproduction of the DDRL paper's OCR task, achieving
0.950 eval OCR accuracy (paper reports 0.823 for DDRL, 0.845 for FlowGRPO).

Requirements:
    pip install diffusion-tinker[ocr]
    GPU with >= 24GB VRAM
    HF_TOKEN env var set
"""

from diffusion_tinker import FlowGRPOConfig, FlowGRPOTrainer

prompts = [
    'A sign that says "HELLO"',
    'A poster that reads "OPEN"',
    'A neon sign that says "CAFE"',
    'A storefront sign that says "PIZZA"',
    'A chalkboard that says "MENU"',
    'A banner that says "SALE"',
    'A wooden sign that reads "WELCOME"',
    'A digital display that says "EXIT"',
]

config = FlowGRPOConfig(
    num_samples_per_prompt=2,
    num_epochs=40,
    eval_every=5,
    save_every=10,
    early_stop_patience=3,
    output_dir="./grpo_ocr_output",
)

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="ocr",
    train_prompts=prompts,
    config=config,
)

trainer.train()
