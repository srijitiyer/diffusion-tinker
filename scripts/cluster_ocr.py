"""OCR reproduction run - DDRL on SD3.5-Medium with OCR reward.

Config informed by empirical sweep on SD3.5-M / A5000 (see scripts/diagnose_ocr.py):
  - 28 inference steps + noise_level=0.1 is the highest-noise config where OCR
    can still read the base model outputs (baseline ~0.74). At 10 steps, any
    non-zero noise injection completely destroys text rendering.
  - data_beta=0 because we have no train_dataset here; the old fallback path
    used the model's own noisy outputs as "clean data" and reinforced bad policy.
"""
import os
import time
import torch

RESULTS_DIR = os.environ.get('OUTPUT_DIR', '/atlas/u/srijit/diffusion-tinker/output/ocr_run3')
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_DIR = os.path.join(RESULTS_DIR, 'ddrl_ocr_run')

PROMPTS = [
    'A sign that says "HELLO"',
    'A poster that reads "OPEN"',
    'A neon sign that says "CAFE"',
    'A storefront sign that says "PIZZA"',
]


def main():
    start = time.time()
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')

    from diffusion_tinker import DDRLConfig, DDRLTrainer

    config = DDRLConfig(
        data_beta=0.0,
        num_samples_per_prompt=2,
        num_inference_steps=28,
        num_eval_inference_steps=28,
        guidance_scale=7.0,
        noise_level=0.1,
        resolution=512,
        learning_rate=1e-4,
        lora_rank=32,
        lora_alpha=64,
        clip_range=1e-4,
        num_epochs=60,
        save_every=20,
        eval_every=10,
        log_every=1,
        output_dir=RUN_DIR,
        gradient_checkpointing=True,
        mixed_precision='bf16',
    )

    trainer = DDRLTrainer(
        model='stabilityai/stable-diffusion-3.5-medium',
        reward_funcs='ocr',
        train_prompts=PROMPTS,
        config=config,
    )

    print(f'Peak VRAM after load: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')
    trainer.train()

    print(f'Total time: {(time.time() - start) / 60:.1f} min')
    print(f'Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
