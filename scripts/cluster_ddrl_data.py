"""DDRL with real data anchor - validates the forward KL regularization.

Uses lambdalabs/naruto-blip-captions as the data anchor to prevent
the epoch 30+ collapse seen without data_beta.
"""
import os
import time
import torch

RESULTS_DIR = os.environ.get('OUTPUT_DIR', '/atlas/u/srijit/diffusion-tinker/output/ddrl_data_run')
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_DIR = os.path.join(RESULTS_DIR, 'ddrl_with_data')

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
        data_beta=0.01,
        train_dataset='lambdalabs/naruto-blip-captions',
        image_column='image',
        use_monotonic_transform=True,
        num_samples_per_prompt=2,
        num_epochs=60,
        save_every=20,
        eval_every=5,
        early_stop_patience=4,
        log_every=1,
        output_dir=RUN_DIR,
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
