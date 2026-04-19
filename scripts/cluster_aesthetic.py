"""Aesthetic reward validation - FlowGRPO on SD3.5-Medium.

Validates that the library works with aesthetic reward (second validated reward
after OCR). Uses FlowGRPO (no dataset required).
"""
import os
import time
import torch

RESULTS_DIR = os.environ.get('OUTPUT_DIR', '/atlas/u/srijit/diffusion-tinker/output/aesthetic_run')
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_DIR = os.path.join(RESULTS_DIR, 'grpo_aesthetic')

PROMPTS = [
    "a photograph of a mountain landscape at golden hour",
    "a portrait of a cat sitting on a windowsill",
    "an oil painting of a city street in the rain",
    "a macro photograph of a flower with morning dew",
]


def main():
    start = time.time()
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')

    from diffusion_tinker import FlowGRPOConfig, FlowGRPOTrainer

    config = FlowGRPOConfig(
        num_samples_per_prompt=2,
        num_epochs=40,
        save_every=10,
        eval_every=5,
        early_stop_patience=3,
        log_every=1,
        output_dir=RUN_DIR,
    )

    trainer = FlowGRPOTrainer(
        model='stabilityai/stable-diffusion-3.5-medium',
        reward_funcs='aesthetic',
        train_prompts=PROMPTS,
        config=config,
    )

    print(f'Peak VRAM after load: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')
    trainer.train()

    print(f'Total time: {(time.time() - start) / 60:.1f} min')
    print(f'Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
