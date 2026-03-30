"""RunPod GPU validation script for diffusion-tinker.

Runs DDRL training on SD3.5-Medium with aesthetic reward.
Generates before/after comparison grid.

Usage: python scripts/runpod_validate.py
Requires: A100 80GB, HF_TOKEN env var set, SD3.5 license accepted.
"""

import json
import os
import time

import torch
from PIL import Image, ImageDraw, ImageFont

RESULTS_DIR = "/workspace/validation_results"
RUN_DIR = os.path.join(RESULTS_DIR, "ddrl_run")

PROMPTS = [
    "a photograph of a mountain landscape at golden hour",
    "a portrait of a cat sitting on a windowsill",
    "an oil painting of a city street in the rain",
    "a macro photograph of a flower with morning dew",
    "a photograph of ocean waves crashing on rocks",
    "a painting of a Japanese garden in spring",
]


def check_gpu():
    assert torch.cuda.is_available(), "No GPU found"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {name} ({vram:.0f} GB)")
    assert vram >= 40, f"Need >= 40GB VRAM, got {vram:.0f} GB"
    return name, vram


def check_hf_access():
    from huggingface_hub import model_info

    try:
        model_info("stabilityai/stable-diffusion-3.5-medium")
        print("HuggingFace access: OK")
    except Exception as e:
        print(f"ERROR: Cannot access SD3.5-Medium: {e}")
        print("1. Accept license at: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium")
        print("2. Set HF_TOKEN env var")
        raise SystemExit(1)


def generate_baseline(trainer):
    """Generate baseline images from all prompts before training."""
    print("\n=== Generating baseline images ===")
    from diffusion_tinker.models.sd3_patch import sd3_sample_with_logprob
    from diffusion_tinker.rewards.protocol import RewardContext

    baseline_dir = os.path.join(RESULTS_DIR, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    trainer.transformer.eval()
    output = sd3_sample_with_logprob(
        pipeline=trainer.pipeline,
        prompts=PROMPTS,
        num_inference_steps=28,
        guidance_scale=7.0,
        noise_level=0.0,
        height=512,
        width=512,
    )

    ctx = RewardContext(images=output.images, prompts=PROMPTS, device=trainer.device)
    rewards = trainer.reward_fn(ctx).scores

    for i, (img, prompt, reward) in enumerate(zip(output.images, PROMPTS, rewards)):
        img.save(os.path.join(baseline_dir, f"sample_{i}.png"))
        print(f"  [{i}] reward={reward:.3f} | {prompt[:50]}")

    mean_reward = rewards.mean().item()
    print(f"  Baseline mean reward: {mean_reward:.3f}")
    return output.images, rewards.tolist(), mean_reward


def generate_final(trainer):
    """Generate final images from all prompts after training."""
    print("\n=== Generating final images ===")
    from diffusion_tinker.models.sd3_patch import sd3_sample_with_logprob
    from diffusion_tinker.rewards.protocol import RewardContext

    final_dir = os.path.join(RESULTS_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)

    trainer.transformer.eval()
    output = sd3_sample_with_logprob(
        pipeline=trainer.pipeline,
        prompts=PROMPTS,
        num_inference_steps=28,
        guidance_scale=7.0,
        noise_level=0.0,
        height=512,
        width=512,
    )

    ctx = RewardContext(images=output.images, prompts=PROMPTS, device=trainer.device)
    rewards = trainer.reward_fn(ctx).scores

    for i, (img, prompt, reward) in enumerate(zip(output.images, PROMPTS, rewards)):
        img.save(os.path.join(final_dir, f"sample_{i}.png"))
        print(f"  [{i}] reward={reward:.3f} | {prompt[:50]}")

    mean_reward = rewards.mean().item()
    print(f"  Final mean reward: {mean_reward:.3f}")
    return output.images, rewards.tolist(), mean_reward


def make_comparison_grid(baseline_images, final_images, baseline_rewards, final_rewards):
    """Create a side-by-side comparison grid."""
    print("\n=== Generating comparison grid ===")
    n = len(PROMPTS)
    img_size = 512
    padding = 10
    label_h = 40
    col_label_h = 30

    width = padding + (img_size + padding) * 2
    height = col_label_h + (label_h + img_size + padding) * n + padding

    grid = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    # Column headers
    draw.text((padding + img_size // 2 - 30, 5), "Before", fill="white", font=font)
    draw.text((padding + img_size + padding + img_size // 2 - 20, 5), "After", fill="white", font=font)

    for i in range(n):
        y_offset = col_label_h + (label_h + img_size + padding) * i

        # Prompt label
        label = f"{PROMPTS[i][:60]}  (reward: {baseline_rewards[i]:.2f} -> {final_rewards[i]:.2f})"
        draw.text((padding, y_offset + 5), label, fill=(200, 200, 200), font=small_font)

        # Before image
        y_img = y_offset + label_h
        grid.paste(baseline_images[i].resize((img_size, img_size)), (padding, y_img))

        # After image
        grid.paste(final_images[i].resize((img_size, img_size)), (padding + img_size + padding, y_img))

    grid_path = os.path.join(RESULTS_DIR, "comparison_grid.png")
    grid.save(grid_path)
    print(f"  Saved to {grid_path}")
    return grid_path


def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Pre-flight checks
    gpu_name, vram = check_gpu()
    check_hf_access()

    print(f"\nVRAM before loading: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Create trainer
    print("\n=== Loading model and creating trainer ===")
    load_start = time.time()

    from diffusion_tinker import DDRLConfig, DDRLTrainer

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
        num_epochs=30,
        save_every=10,
        eval_every=10,
        output_dir=RUN_DIR,
        gradient_checkpointing=True,
        mixed_precision="bf16",
    )

    trainer = DDRLTrainer(
        model="stabilityai/stable-diffusion-3.5-medium",
        reward_funcs="aesthetic",
        train_prompts=PROMPTS,
        config=config,
    )

    load_time = time.time() - load_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Model loaded in {load_time:.0f}s | Peak VRAM: {peak_vram:.1f} GB")

    # Baseline
    baseline_images, baseline_rewards, baseline_mean = generate_baseline(trainer)
    print(f"Peak VRAM after baseline: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # Train
    print("\n=== Starting DDRL training (30 epochs) ===")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    print(f"Training completed in {train_time / 60:.1f} minutes")

    # Final eval
    final_images, final_rewards, final_mean = generate_final(trainer)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # Comparison grid
    make_comparison_grid(baseline_images, final_images, baseline_rewards, final_rewards)

    # Summary
    total_time = time.time() - start_time
    summary = {
        "gpu": gpu_name,
        "vram_gb": vram,
        "peak_vram_gb": peak_vram,
        "model": "stabilityai/stable-diffusion-3.5-medium",
        "algorithm": "DDRL",
        "num_epochs": 30,
        "num_prompts": len(PROMPTS),
        "num_samples_per_prompt": 4,
        "baseline_mean_reward": baseline_mean,
        "final_mean_reward": final_mean,
        "reward_delta": final_mean - baseline_mean,
        "baseline_rewards": baseline_rewards,
        "final_rewards": final_rewards,
        "load_time_s": load_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  Baseline mean reward: {baseline_mean:.3f}")
    print(f"  Final mean reward:    {final_mean:.3f}")
    print(f"  Delta:                {final_mean - baseline_mean:+.3f}")
    print(f"  Peak VRAM:            {peak_vram:.1f} GB")
    print(f"  Total time:           {total_time / 60:.1f} minutes")
    print(f"  Results:              {RESULTS_DIR}/")
    print(f"  Comparison grid:      {RESULTS_DIR}/comparison_grid.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
