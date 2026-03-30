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

## Supported Algorithms

| Algorithm | Paper | Status |
|-----------|-------|--------|
| **DDRL** | [arXiv:2512.04332](https://arxiv.org/abs/2512.04332) | Implemented |
| **FlowGRPO** | [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) | Implemented |
| **DiffusionDPO** | [arXiv:2311.12908](https://arxiv.org/abs/2311.12908) | Implemented |
| **DRaFT** | [arXiv:2309.17400](https://arxiv.org/abs/2309.17400) | Implemented |
| **DDPO/DPOK** | [arXiv:2305.13301](https://arxiv.org/abs/2305.13301) | Implemented |
| **SFT** | Standard denoising loss | Implemented |

## Supported Models

| Model | Architecture | Status |
|-------|-------------|--------|
| **SD3 / SD3.5** | MMDiT, flow matching | Implemented |
| **FLUX.1** | Hybrid transformer, flow matching | Implemented |
| SDXL | UNet, epsilon prediction | Planned |
| SD 1.5 / 2.x | UNet, epsilon/v-prediction | Planned |

## Built-in Rewards

| Reward | Type | Usage |
|--------|------|-------|
| **Aesthetic** | CLIP + MLP | `reward_funcs="aesthetic"` |
| **CLIP Score** | CLIP cosine similarity | `reward_funcs="clip_score"` |
| **HPS v2** | OpenCLIP ViT-H-14 | `reward_funcs="hps_v2"` |
| **OCR** | PaddleOCR edit distance | `reward_funcs="ocr"` |
| Custom function | Any callable | `reward_funcs=my_fn` |

## Design

Each algorithm is a **Trainer + Config** pair following TRL's pattern:

- `DDRLTrainer` + `DDRLConfig` with forward KL data regularization
- `FlowGRPOTrainer` + `FlowGRPOConfig` with optional KL and GRPO-Guard
- `DiffusionDPOTrainer` + `DiffusionDPOConfig` for offline preference learning
- `DDPOTrainer` + `DDPOConfig` with multi-epoch PPO and optional DPOK KL
- `DRaFTTrainer` + `DRaFTConfig` for direct reward backprop (DRaFT-1/K)
- `SFTTrainer` + `SFTConfig` for standard supervised fine-tuning
- Multi-reward: `reward_funcs=["aesthetic", "clip_score"]` with `reward_weights`
- Model loading via string ID with auto-detected architecture and LoRA targets
- SD3/SD3.5 and FLUX.1 pipeline support with SDE sampling and log-prob collection
- Reward functions via string lookup or custom callables
- Built on `diffusers` + `peft` with no custom infrastructure

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
