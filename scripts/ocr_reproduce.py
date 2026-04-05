"""OCR reproduction script for DDRL on SD3.5-Medium.

Reproduces the OCR accuracy task from the DDRL paper (arXiv:2512.04332).
Trains SD3.5-Medium to generate images containing specified text using
the OCR edit distance reward.

Usage: python scripts/ocr_reproduce.py
Requires: A100 80GB, HF_TOKEN env var, pip install .[ocr]
"""

import json
import os
import time

import torch

RESULTS_DIR = "/workspace/ocr_results"
RUN_DIR = os.path.join(RESULTS_DIR, "ddrl_ocr_run")

# OCR prompts - text in quotes is the target for the OCR reward
PROMPTS = [
    'A sign that says "HELLO"',
    'A poster that reads "OPEN"',
    'A banner that says "SALE"',
    'A neon sign that says "CAFE"',
    'A chalkboard that says "MENU"',
    'A storefront sign that says "PIZZA"',
    'A wooden sign that reads "WELCOME"',
    'A digital display that says "EXIT"',
    'A street sign that says "STOP"',
    'A label that reads "FRAGILE"',
    'A door sign that says "PUSH"',
    'A marquee that says "CINEMA"',
    'A road sign that reads "SLOW"',
    'A warning sign that says "DANGER"',
    'A shop window that says "CLOSED"',
    'A billboard that reads "DREAM"',
    'A wall with graffiti that says "LOVE"',
    'A license plate that reads "COOL"',
    'A bumper sticker that says "PEACE"',
    'A book cover that says "NOVEL"',
]


def check_gpu():
    assert torch.cuda.is_available(), "No GPU found"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name} ({vram:.0f} GB)")
    return name, vram


def check_hf_access():
    from huggingface_hub import model_info

    try:
        model_info("stabilityai/stable-diffusion-3.5-medium")
        print("HuggingFace access: OK")
    except Exception as e:
        print(f"Cannot access SD3.5: {e}")
        raise SystemExit(1)


def check_ocr():
    try:
        from paddleocr import PaddleOCR

        try:
            PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
        except (TypeError, ValueError):
            PaddleOCR(lang="en", use_gpu=False)
        print("PaddleOCR: OK")
        return True
    except ImportError:
        print("PaddleOCR not installed. Installing...")
        os.system("pip install paddleocr paddlepaddle python-Levenshtein --quiet")
        return True


def generate_eval_images(trainer, prompts, output_dir, label=""):
    """Generate images and compute OCR scores for a set of prompts."""
    from diffusion_tinker.models.sd3_patch import sd3_sample_with_logprob
    from diffusion_tinker.rewards.protocol import RewardContext
    from diffusion_tinker.rewards.resolve import resolve_reward

    os.makedirs(output_dir, exist_ok=True)
    trainer.transformer.eval()

    ocr_reward = resolve_reward("ocr", device="cpu")

    output = sd3_sample_with_logprob(
        pipeline=trainer.pipeline,
        prompts=prompts,
        num_inference_steps=28,
        guidance_scale=7.0,
        noise_level=0.0,
        height=512,
        width=512,
    )

    ctx = RewardContext(images=output.images, prompts=prompts)
    scores = ocr_reward(ctx).scores

    for i, (img, prompt, score) in enumerate(zip(output.images, prompts, scores)):
        img.save(os.path.join(output_dir, f"sample_{i}.png"))

    mean_score = scores.mean().item()
    print(f"  {label} OCR accuracy: {mean_score:.3f}")

    per_prompt = []
    for i, (prompt, score) in enumerate(zip(prompts, scores)):
        per_prompt.append({"prompt": prompt, "ocr_score": score.item()})
        print(f"    [{i}] {score.item():.3f} | {prompt[:50]}")

    return output.images, scores.tolist(), mean_score, per_prompt


def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gpu_name, vram = check_gpu()
    check_hf_access()
    check_ocr()

    print("\n=== Loading model ===")
    load_start = time.time()

    from diffusion_tinker import DDRLConfig, DDRLTrainer

    config = DDRLConfig(
        # DDRL params from paper
        data_beta=0.01,
        clip_range=1e-4,
        use_monotonic_transform=True,
        condition_dropout=0.2,
        # Training
        num_samples_per_prompt=4,
        num_inference_steps=10,
        num_eval_inference_steps=28,
        guidance_scale=7.0,
        noise_level=0.7,
        resolution=512,
        learning_rate=1e-4,
        lora_rank=32,
        lora_alpha=64,
        # Schedule
        num_epochs=150,
        save_every=30,
        eval_every=25,
        log_every=1,
        output_dir=RUN_DIR,
        gradient_checkpointing=True,
        mixed_precision="bf16",
    )

    trainer = DDRLTrainer(
        model="stabilityai/stable-diffusion-3.5-medium",
        reward_funcs="ocr",
        train_prompts=PROMPTS,
        config=config,
    )

    load_time = time.time() - load_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Model loaded in {load_time:.0f}s | Peak VRAM: {peak_vram:.1f} GB")

    # Baseline
    print("\n=== Baseline evaluation ===")
    eval_prompts = PROMPTS[:10]
    baseline_images, baseline_scores, baseline_mean, baseline_details = generate_eval_images(
        trainer, eval_prompts, os.path.join(RESULTS_DIR, "baseline"), "Baseline"
    )

    # Train
    print(f"\n=== Starting DDRL OCR training ({config.num_epochs} epochs) ===")
    train_start = time.time()

    trainer.train()

    train_time = time.time() - train_start
    print(f"Training completed in {train_time / 60:.1f} minutes")

    # Final eval
    print("\n=== Final evaluation ===")
    final_images, final_scores, final_mean, final_details = generate_eval_images(
        trainer, eval_prompts, os.path.join(RESULTS_DIR, "final"), "Final"
    )

    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # Summary
    total_time = time.time() - start_time
    summary = {
        "gpu": gpu_name,
        "peak_vram_gb": peak_vram,
        "model": "stabilityai/stable-diffusion-3.5-medium",
        "algorithm": "DDRL",
        "reward": "OCR (PaddleOCR edit distance)",
        "num_epochs": config.num_epochs,
        "num_prompts": len(PROMPTS),
        "num_eval_prompts": len(eval_prompts),
        "baseline_ocr_accuracy": baseline_mean,
        "final_ocr_accuracy": final_mean,
        "ocr_delta": final_mean - baseline_mean,
        "baseline_details": baseline_details,
        "final_details": final_details,
        "load_time_s": load_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("OCR REPRODUCTION COMPLETE")
    print("=" * 60)
    print(f"  Baseline OCR accuracy: {baseline_mean:.3f}")
    print(f"  Final OCR accuracy:    {final_mean:.3f}")
    print(f"  Delta:                 {final_mean - baseline_mean:+.3f}")
    print(f"  Peak VRAM:             {peak_vram:.1f} GB")
    print(f"  Training time:         {train_time / 60:.1f} minutes")
    print(f"  Total time:            {total_time / 60:.1f} minutes")
    print(f"  Results:               {RESULTS_DIR}/")
    print("=" * 60)

    # DDRL paper reference numbers (SD3.5-Medium, 512x512):
    # No RL:    OCR Score = 0.566
    # FlowGRPO: OCR Score = 0.845
    # DDRL:     OCR Score = 0.823
    print("\nReference (DDRL paper):")
    print("  No RL:    0.566")
    print("  FlowGRPO: 0.845")
    print("  DDRL:     0.823")


if __name__ == "__main__":
    main()
