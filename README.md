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
# Achieves 0.950 eval OCR accuracy on SD3.5-Medium (paper: 0.823)
```

## Supported Algorithms

| Algorithm | Trainer | Paper | Status |
|-----------|---------|-------|--------|
| **FlowGRPO** | `FlowGRPOTrainer` | [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) | Validated |
| **DDRL** | `DDRLTrainer` | [arXiv:2512.04332](https://arxiv.org/abs/2512.04332) | Validated |
| **DiffusionDPO** | `DiffusionDPOTrainer` | [arXiv:2311.12908](https://arxiv.org/abs/2311.12908) | Implemented |
| **DRaFT** | `DRaFTTrainer` | [arXiv:2309.17400](https://arxiv.org/abs/2309.17400) | Implemented |
| **DDPO/DPOK** | `DDPOTrainer` | [arXiv:2305.13301](https://arxiv.org/abs/2305.13301) | Implemented |
| **SFT** | `SFTTrainer` | Standard denoising loss | Implemented |

## Supported Models

| Model | Architecture |
|-------|-------------|
| **SD3 / SD3.5** | MMDiT, flow matching |
| **FLUX.1** | Hybrid transformer, flow matching |

## Built-in Rewards

| Reward | Usage | Install |
|--------|-------|---------|
| **Aesthetic** | `"aesthetic"` | Included |
| **CLIP Score** | `"clip_score"` | Included |
| **OCR** | `"ocr"` | `pip install .[ocr]` |
| **HPS v2** | `"hps_v2"` | `pip install .[hps]` |
| **Custom** | `reward_funcs=my_fn` | - |
| **Multi-reward** | `["aesthetic", "clip_score"]` | - |

Custom reward functions take a `RewardContext` (with `.images` and `.prompts`) and return a `RewardOutput` (with `.scores` tensor).

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
)
```

DDRL adds `data_beta` (forward KL weight) and `train_dataset` (required for data regularization):

```python
DDRLConfig(
    data_beta=0.01,
    train_dataset="yuvalkirstain/pickapic_v2",  # or local image folder
    use_monotonic_transform=True,               # Theorem 3.1 from the paper
)
```

## Examples

See `examples/`:

- `grpo_aesthetic.py` - FlowGRPO + aesthetic reward (simplest, good first test)
- `grpo_ocr.py` - FlowGRPO + OCR reward (validated, 0.950 eval accuracy)
- `flowgrpo_multi_reward.py` - FlowGRPO + aesthetic + CLIP multi-reward
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
