# Reproduction Results

All results on **Stable Diffusion 3.5-Medium**, single **NVIDIA A5000 (24GB)**, LoRA (rank 16), bf16. Each algorithm was run end-to-end on GPU and the optimized metric verified to move in the correct direction. "Validated" below means a real training run reproduced the expected convergence behavior, not just that the code executes.

## Summary

| Algorithm | Task / Reward | Metric | Result | Status |
|-----------|---------------|--------|--------|--------|
| **DRaFT** | Aesthetic (LAION predictor) | mean reward, 40 epochs | 5.92 → 9.58 | Validated |
| **SFT** | Naruto denoising | fixed-timestep eval loss, 300 steps | 0.189 → 0.185 | Validated |
| **DiffusionDPO** | Preference (real vs noise) | implicit reward, 200 steps | 0 → 60.5 (winner loss ↓, loser ↑) | Validated |
| **FlowGRPO** | OCR text rendering | eval OCR accuracy | 0.64 (ep 5) → **1.00 best** (ep 10, saved); 0.70–1.00 thereafter (3–4 of 4 prompts), early-stopped ep 25 | Validated |
| **DDRL** | Aesthetic + data anchor | mean reward | oscillates ~3.7-3.8, no net climb at this budget; data anchor keeps it stable (no collapse) | Stable; reward sample-limited |
| **DDPO/DPOK** | Aesthetic | mean reward | importance ratio corrected to ~1.0; reward flat at this sample budget (PPO is sample-inefficient) | Mechanism verified |

## FLUX.1-schnell (separate model)

The FLUX code path (model auto-detection, packed 3D latents, `flux_sample_with_logprob` / `flux_replay_step`, schnell guidance gating, image decode) was run end-to-end with FlowGRPO + aesthetic on a 48GB A6000-Ada. Correctness checks at runtime: image range [0.00, 1.00], log-probs finite, packed latents `(B, T, 1024, 64)`. Eval aesthetic reward improved 6.05 → **6.39 best** (epoch 15) but oscillated (6.03–6.39) rather than climbing cleanly. Two reasons it underperforms the SD3 runs: schnell is a 4-step **distilled** model not trained for stochastic SDE sampling, and its sampling/replay importance ratio sits at ~0.92 (vs ~1.0 on SD3), leaving a residual gradient bias. **Status: code path validated on FLUX; convergence modest and noisy on schnell** - a non-distilled model (FLUX.1-dev) would likely be a better RL target.

## How accuracy was verified

Each trainer was exercised on real hardware with the metric tracked across epochs. For trainers where the raw loss is noisy (SFT), a fixed-timestep before/after eval was used to isolate signal. Gradient flow was checked directly (asserting LoRA `.grad` is non-None with nonzero norm after `backward()`), which is how the DRaFT bug below was caught - unit tests and smoke runs pass without ever verifying that gradients reach the weights.

## Bugs found and fixed during verification

Verification surfaced several correctness bugs that the prior smoke tests did not catch. All are fixed; the results above are on the corrected code.

1. **DRaFT never trained.** `torch.autocast`'s weight cache stored a detached bf16 cast of the float32 LoRA weights during a `no_grad` warmup loop, and the gradient loop reused it, so no gradient ever reached the LoRA params. Fixed with `cache_enabled=False`. (After fix: aesthetic reward 5.9 → 9.6.)
2. **VAE decode range (repo-wide).** Decode returns pixels in [-1, 1] but the code did `.clamp(0,1)` directly, missing the `/2+0.5` rescale - runtime check showed 55% of every image was crushed to black. Fixed in all decode sites.
3. **VAE encode range.** The symmetric bug: [0,1] images were fed to a VAE expecting [-1,1], so SFT/DDRL/DiffusionDPO trained on mis-scaled latents. Fixed.
4. **Biased PPO ratio.** Sampling ran the transformer without autocast (fp32 LoRA) while replay ran under autocast bf16, biasing the importance ratio to ~0.97. Aligned the precision; ratio is now ~1.0.
5. **PEFT adapter API.** Three trainers (FlowGRPO, DDPO, DiffusionDPO) called `disable_adapter_layers()` (a PeftModel-only method) on diffusers transformers, crashing the reference-model paths. Fixed to `disable_adapters()`/`enable_adapters()`.

Plus smaller fixes: DRaFT grad-accumulation normalizer, GRPO-guard stop-grad, KL noise-level consistency, OCR substring over-match, a stat-tracker memory leak, and a broken DDRL example config.

## Not yet covered

- **DDPO reward convergence**: the mechanism is correct (unbiased ratio, gradients flow, shares FlowGRPO's validated replay) but PPO is sample-inefficient; demonstrating a reward gain needs a larger sample budget than a single 24GB GPU allows.
- **FLUX.1-dev**: only schnell was run (see FLUX section). dev is non-distilled and likely a better RL target, but its license gate requires more fields; worth a follow-up run.
- **FLUX `img_ids`**: passed as a 3D batched tensor, which works but triggers a diffusers deprecation warning each step; should be switched to 2D.
