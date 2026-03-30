# diffusion-tinker

RL-based post-training for diffusion models. TRL-style API, built on HuggingFace diffusers.

## Quickstart

```bash
pip install git+https://github.com/srijitiyer/diffusion-tinker.git
```

```python
from diffusion_tinker import DDRLTrainer, DDRLConfig

trainer = DDRLTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="aesthetic",
    train_prompts=["a photograph of a mountain at golden hour", ...],
    config=DDRLConfig(data_beta=0.01),
)
trainer.train()
```

Multi-reward training:

```python
from diffusion_tinker import FlowGRPOTrainer, FlowGRPOConfig

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs=["aesthetic", "clip_score"],
    reward_weights=[0.6, 0.4],
    reward_mode="advantage_level",
    train_prompts=prompts,
    config=FlowGRPOConfig(kl_beta=0.01),
)
trainer.train()
```

## Supported Algorithms

| Algorithm | Trainer | Paper |
|-----------|---------|-------|
| **DDRL** | `DDRLTrainer` | [arXiv:2512.04332](https://arxiv.org/abs/2512.04332) |
| **FlowGRPO** | `FlowGRPOTrainer` | [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) |
| **DiffusionDPO** | `DiffusionDPOTrainer` | [arXiv:2311.12908](https://arxiv.org/abs/2311.12908) |
| **DRaFT** | `DRaFTTrainer` | [arXiv:2309.17400](https://arxiv.org/abs/2309.17400) |
| **DDPO/DPOK** | `DDPOTrainer` | [arXiv:2305.13301](https://arxiv.org/abs/2305.13301) |
| **SFT** | `SFTTrainer` | Standard denoising loss |

## Supported Models

| Model | Architecture |
|-------|-------------|
| **SD3 / SD3.5** | MMDiT, flow matching |
| **FLUX.1** | Hybrid transformer, flow matching |

## Built-in Rewards

| Reward | Usage | Optional Dependency |
|--------|-------|-------------------|
| **Aesthetic** | `"aesthetic"` | None (uses transformers) |
| **CLIP Score** | `"clip_score"` | None (uses transformers) |
| **HPS v2** | `"hps_v2"` | `pip install .[hps]` |
| **OCR** | `"ocr"` | `pip install .[ocr]` |
| **Custom** | `reward_funcs=my_fn` | None |
| **Multi-reward** | `["aesthetic", "clip_score"]` | None |

## Examples

See the `examples/` directory:

- `ddrl_aesthetic.py` - DDRL with aesthetic reward
- `flowgrpo_multi_reward.py` - FlowGRPO with combined aesthetic + CLIP score
- `dpo_pickapic.py` - DiffusionDPO on Pick-a-Pic preference dataset
- `draft_aesthetic.py` - DRaFT-1 with direct reward backprop
- `sft_naruto.py` - SFT on naruto image dataset

## Installation

Core (all RL trainers + aesthetic + CLIP score):
```bash
pip install git+https://github.com/srijitiyer/diffusion-tinker.git
```

With HPS v2 reward:
```bash
pip install "diffusion-tinker[hps] @ git+https://github.com/srijitiyer/diffusion-tinker.git"
```

With OCR reward:
```bash
pip install "diffusion-tinker[ocr] @ git+https://github.com/srijitiyer/diffusion-tinker.git"
```

With dataset support (for SFT and DiffusionDPO):
```bash
pip install "diffusion-tinker[data] @ git+https://github.com/srijitiyer/diffusion-tinker.git"
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- GPU with >= 24GB VRAM (A100 80GB recommended)

## Development

```bash
git clone https://github.com/srijitiyer/diffusion-tinker.git
cd diffusion-tinker
pip install -e ".[dev]"
ruff check src/ tests/
python tests/test_full_pipeline.py
```

## License

Apache 2.0
