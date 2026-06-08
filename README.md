# diffusion-tinker

RL-based post-training for diffusion models. TRL-style API, built on HuggingFace diffusers.

## Quickstart

```bash
pip install git+https://github.com/srijitiyer/diffusion-tinker.git
```

```python
from diffusion_tinker import FlowGRPOTrainer, FlowGRPOConfig

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="aesthetic",
    train_prompts=[
        "a photograph of a mountain at golden hour",
        "a portrait of a cat on a windowsill",
        "an oil painting of a city street in the rain",
        "a macro photograph of a flower with morning dew",
    ],
    config=FlowGRPOConfig(num_epochs=30, early_stop_patience=3),
)
trainer.train()
```

OCR reward (train the model to render readable text):

```python
from diffusion_tinker import FlowGRPOTrainer, FlowGRPOConfig

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="ocr",
    train_prompts=[
        'A sign that says "HELLO"',
        'A poster that reads "OPEN"',
        'A neon sign that says "CAFE"',
        'A storefront sign that says "PIZZA"',
    ],
    config=FlowGRPOConfig(num_samples_per_prompt=2, num_epochs=40),
)
trainer.train()
# Reaches up to 1.00 best eval OCR accuracy on SD3.5-Medium (paper: 0.823)
```

## Supported Algorithms

| Algorithm | Trainer | Paper | Status |
|-----------|---------|-------|--------|
| **FlowGRPO** | `FlowGRPOTrainer` | [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) | Validated |
| **DDRL** | `DDRLTrainer` | [arXiv:2512.04332](https://arxiv.org/abs/2512.04332) | Validated |
| **DiffusionDPO** | `DiffusionDPOTrainer` | [arXiv:2311.12908](https://arxiv.org/abs/2311.12908) | Validated |
| **DRaFT** | `DRaFTTrainer` | [arXiv:2309.17400](https://arxiv.org/abs/2309.17400) | Validated |
| **DDPO/DPOK** | `DDPOTrainer` | [arXiv:2305.13301](https://arxiv.org/abs/2305.13301) | Smoke-tested |
| **SFT** | `SFTTrainer` | Standard denoising loss | Validated |

*Validated* = training demonstrably converges on SD3.5 (the optimized metric moves the right way on a real run). *Smoke-tested* = runs end-to-end with gradients flowing, but reward convergence not yet demonstrated. DDPO is wired correctly (shares FlowGRPO's validated replay) but is sample-inefficient; showing a reward gain needs a larger sample budget than a single 24GB GPU allows.

## Supported Models

| Model | Architecture | Supported Trainers |
|-------|-------------|-------------------|
| **SD3 / SD3.5** | MMDiT, flow matching | All 6 trainers |
| **FLUX.1** | Hybrid transformer, flow matching | FlowGRPO, DDRL, DDPO |

## Built-in Rewards

| Reward | Usage | Install |
|--------|-------|---------|
| **Aesthetic** | `"aesthetic"` | Included |
| **CLIP Score** | `"clip_score"` | Included |
| **OCR** | `"ocr"` | `pip install .[ocr]` |
| **HPS v2** | `"hps_v2"` | `pip install .[hps]` |
| **Custom** | `reward_funcs=my_fn` | - |
| **Multi-reward** | `["aesthetic", "clip_score"]` | - |

Custom reward functions receive a `RewardContext` and can return any of:
- `RewardOutput(scores=tensor)` - full control
- `torch.Tensor` - scores directly
- `list[float]` - converted to tensor automatically

```python
def my_reward(ctx):
    # ctx.images: list of PIL images, ctx.prompts: list of strings
    return [1.0 if "cat" in p else 0.0 for p in ctx.prompts]

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs=my_reward,
    ...
)
```

`RewardContext` exposes everything a custom reward might need:

| Field | Type | Description |
|-------|------|-------------|
| `images` | `list[PIL.Image]` | Generated images (always populated) |
| `prompts` | `list[str]` | Corresponding prompts |
| `device` | `torch.device` | Device for model computations |
| `latents` | `Tensor \| None` | Final denoised latents from the trajectory |
| `epoch` | `int \| None` | Current training epoch |
| `metadata` | `dict` | User-defined data from `reward_kwargs` |

Pass arbitrary data to your reward function via `reward_kwargs`:

```python
def my_reward(ctx):
    threshold = ctx.metadata.get("threshold", 0.5)
    target_style = ctx.metadata["target_style"]
    # use ctx.latents for latent-space rewards, ctx.epoch for curriculum, etc.
    ...

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs=my_reward,
    reward_kwargs={"threshold": 0.8, "target_style": "watercolor"},
    ...
)
```

Multi-reward supports two aggregation modes via `reward_mode`:
- `"weighted_sum"` (default) - weighted average of raw scores
- `"advantage_level"` - normalize each reward to zero mean/unit variance before weighting (useful when reward scales differ)

## Callbacks

Hook into the training loop with `TrainerCallback`:

```python
from diffusion_tinker import TrainerCallback

class WandbCallback(TrainerCallback):
    def on_epoch_end(self, trainer, epoch, metrics):
        wandb.log(metrics, step=epoch)

    def on_evaluate(self, trainer, epoch, mean_reward):
        wandb.log({"eval_reward": mean_reward}, step=epoch)

    def on_save(self, trainer, epoch, path):
        wandb.save(f"{path}/*")

trainer = FlowGRPOTrainer(
    ...,
    callbacks=[WandbCallback()],
)
```

Available hooks: `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`, `on_sample_end`, `on_evaluate`, `on_save`.

## Key Configuration

All trainers inherit from `BaseDiffusionConfig`. Important defaults:

```python
FlowGRPOConfig(
    # Sampling (tuned for SD3.5-Medium on A5000/A6000)
    num_inference_steps=28,       # denoising steps during training
    noise_level=0.1,              # SDE noise injection (higher = more exploration, lower = readable images)
    num_samples_per_prompt=4,     # samples per prompt for advantage estimation
    guidance_scale=7.0,           # CFG scale

    # RL
    clip_range=0.2,               # PPO clip range

    # Training
    learning_rate=1e-4,
    lora_rank=32,
    num_epochs=50,
    save_best=True,               # auto-save checkpoint when eval improves
    early_stop_patience=0,        # 0=disabled, N=stop after N evals without improvement

    # Memory
    gradient_checkpointing=True,
    mixed_precision="bf16",
    offload_text_encoders=False,  # move text encoders to CPU during the grad step
                                  # (frees ~10GB for larger sample batches on 24GB)
)
```

DDRL adds `data_beta` (forward KL weight) and `train_dataset` (required for data regularization):

```python
DDRLConfig(
    data_beta=0.01,
    train_dataset="yuvalkirstain/pickapic_v2",  # or local image folder
    use_monotonic_transform=False,              # Theorem 3.1; enable only with strong data anchor
)
```

## Examples

See `examples/`:

- `grpo_aesthetic.py` - FlowGRPO + aesthetic reward (simplest, good first test)
- `grpo_ocr.py` - FlowGRPO + OCR reward (validated, up to 1.00 best eval accuracy)
- `flowgrpo_multi_reward.py` - FlowGRPO + aesthetic + CLIP multi-reward
- `custom_reward.py` - Custom reward function with `reward_kwargs` and training callbacks
- `ddrl_aesthetic.py` - DDRL with data-regularized training (requires dataset)
- `dpo_pickapic.py` - DiffusionDPO on preference dataset
- `draft_aesthetic.py` - DRaFT with direct reward backprop
- `sft_naruto.py` - Supervised fine-tuning

## Installation

```bash
# Core (all trainers + aesthetic + CLIP rewards)
pip install git+https://github.com/srijitiyer/diffusion-tinker.git

# With OCR reward
pip install "diffusion-tinker[ocr] @ git+https://github.com/srijitiyer/diffusion-tinker.git"

# With dataset support (for DDRL data loss, SFT, DPO)
pip install "diffusion-tinker[data] @ git+https://github.com/srijitiyer/diffusion-tinker.git"
```

**Note on EasyOCR:** Install with `pip install easyocr --no-deps` to avoid it downgrading PyTorch, then install its dependencies separately: `pip install pyclipper shapely python-bidi scikit-image opencv-python-headless`.

## Requirements

- Python >= 3.10
- PyTorch >= 2.5 (2.4 has incompatibilities with diffusers)
- GPU with >= 24GB VRAM (A5000, A6000, A100)
- HuggingFace token with access to gated models (SD3.5)

## License

Apache 2.0
