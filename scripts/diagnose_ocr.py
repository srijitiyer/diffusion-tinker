"""Diagnose why OCR reward is stuck at 0 during training.

Samples images at both training settings (10 steps SDE noise=0.7) and eval
settings (28 steps ODE noise=0.0) on a frozen base model, then runs OCR on
both. If training-time samples all score 0 but eval samples score > 0, the
sampling settings are incompatible with the reward signal and need to change.
"""

import json
import os

import torch

PROMPTS = [
    'A sign that says "HELLO"',
    'A poster that reads "OPEN"',
    'A banner that says "SALE"',
    'A neon sign that says "CAFE"',
    'A chalkboard that says "MENU"',
    'A storefront sign that says "PIZZA"',
]


def main():
    out_dir = os.environ.get("OUTPUT_DIR", "/workspace/ocr_diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    from diffusers import StableDiffusion3Pipeline

    from diffusion_tinker.models.sd3_patch import sd3_sample_with_logprob
    from diffusion_tinker.rewards.protocol import RewardContext
    from diffusion_tinker.rewards.resolve import resolve_reward

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    ocr = resolve_reward("ocr", device="cpu")

    configs = [
        ("train_settings", 10, 0.7),
        ("eval_settings", 28, 0.0),
        ("train_steps_no_noise", 10, 0.0),
        ("eval_steps_with_noise", 28, 0.7),
    ]

    results = {}
    for label, steps, noise in configs:
        print(f"\n=== {label}: steps={steps}, noise={noise} ===")
        output = sd3_sample_with_logprob(
            pipeline=pipeline,
            prompts=PROMPTS,
            num_inference_steps=steps,
            guidance_scale=7.0,
            noise_level=noise,
            height=512,
            width=512,
        )

        ctx = RewardContext(images=output.images, prompts=PROMPTS)
        scores = ocr(ctx).scores

        sub_dir = os.path.join(out_dir, label)
        os.makedirs(sub_dir, exist_ok=True)
        for i, (img, prompt, score) in enumerate(zip(output.images, PROMPTS, scores)):
            img.save(os.path.join(sub_dir, f"{i}_score{score.item():.2f}.png"))
            print(f"  [{i}] {score.item():.3f} | {prompt}")

        mean = scores.mean().item()
        print(f"  Mean OCR: {mean:.3f}")
        results[label] = {"mean": mean, "per_prompt": scores.tolist()}

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    for label, r in results.items():
        print(f"  {label}: {r['mean']:.3f}")
    print("\nInterpretation:")
    print("  - If train_settings << eval_settings: training sampling is too degraded for OCR")
    print("    -> lower noise_level or increase num_inference_steps during training")
    print("  - If both are 0: base model can't render text at all (reward ceiling problem)")
    print("    -> need a better starting checkpoint or different prompts")


if __name__ == "__main__":
    main()
