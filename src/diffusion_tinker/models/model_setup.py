"""Shared model setup for all trainers."""

from __future__ import annotations

import torch
from peft import LoraConfig


def setup_sd3_model(pipeline_or_id, config, device):
    """Load SD3 pipeline, apply LoRA, freeze non-trainable params.

    Returns (pipeline, transformer, vae, model_config).
    """
    from diffusers import StableDiffusion3Pipeline
    from diffusion_tinker.models.sd3_patch import SD3ModelConfig

    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

    if isinstance(pipeline_or_id, str):
        print(f"Loading {pipeline_or_id}...")
        pipeline = StableDiffusion3Pipeline.from_pretrained(pipeline_or_id, torch_dtype=dtype)
    else:
        pipeline = pipeline_or_id

    pipeline.to(device)

    transformer = pipeline.transformer
    vae = pipeline.vae
    model_config = SD3ModelConfig()

    vae.eval()
    vae.requires_grad_(False)
    for enc in [pipeline.text_encoder, pipeline.text_encoder_2, getattr(pipeline, 'text_encoder_3', None)]:
        if enc is not None:
            enc.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=model_config.lora_target_modules,
    )
    transformer.add_adapter(lora_config)

    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    for p in transformer.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    print(f"LoRA applied: rank={config.lora_rank}, alpha={config.lora_alpha}")

    return pipeline, transformer, vae, model_config
