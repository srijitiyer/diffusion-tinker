"""FLUX pipeline modification for SDE sampling with log-probability collection.

Patches FluxPipeline to return denoising trajectories and per-step log-probabilities.
Key difference from SD3: FLUX uses 2x2 patch packing of latents, different text
encoding (CLIP + T5 only, no CLIP-G), and 3D rotary position embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from diffusion_tinker.core.noise_strategy import sde_step_with_logprob


@dataclass
class FluxModelConfig:
    """Static configuration for FLUX.1 architecture."""

    vae_channels: int = 16  # 4 channels * 4 (2x2 packing)
    prediction_type: str = "flow_velocity"
    noise_type: str = "rectified_flow"
    lora_target_modules: list[str] | None = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Dual-stream block attention
            self.lora_target_modules = [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
            ]


@dataclass
class FluxSamplingOutput:
    """Output from FLUX pipeline sampling with log-probs."""

    images: list[Image.Image]
    latents_trajectory: torch.Tensor  # (B, T, seq_len, inner_dim) - packed latents
    next_latents_trajectory: torch.Tensor
    log_probs: torch.Tensor  # (B, T)
    timesteps: torch.Tensor  # (T+1,)
    prompt_embeds: torch.Tensor  # (B, seq_len, 4096)
    pooled_embeds: torch.Tensor  # (B, 768)
    img_ids: torch.Tensor
    txt_ids: torch.Tensor


def _pack_latents(latents, height, width):
    """Pack (B, C, H, W) latents to (B, num_patches, C*4) for FLUX transformer."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def _unpack_latents(latents, height, width, channels=16):
    """Unpack (B, num_patches, C*4) back to (B, C, H, W)."""
    b, n, d = latents.shape
    c = channels
    h = height // 8  # VAE compression
    w = width // 8
    latents = latents.view(b, h // 2, w // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(b, c, h, w)
    return latents


def _prepare_img_ids(batch_size, height, width, device, dtype):
    """Create image position IDs for FLUX's 3D rotary embeddings."""
    h = height // 16  # VAE (8x) * packing (2x) = 16x
    w = width // 16
    img_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    img_ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
    img_ids = img_ids.reshape(h * w, 3)
    img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return img_ids


@torch.no_grad()
def flux_sample_with_logprob(
    pipeline,
    prompts: list[str],
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    noise_level: float = 0.7,
    height: int = 1024,
    width: int = 1024,
    generator: torch.Generator | None = None,
) -> FluxSamplingOutput:
    """Run FLUX pipeline with SDE sampling, collecting trajectories and log-probs."""
    device = pipeline.transformer.device
    dtype = pipeline.transformer.dtype
    batch_size = len(prompts)

    # 1. Encode text (CLIP pooled + T5 sequence)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        device=device,
        max_sequence_length=512,
    )

    # 2. Prepare initial noise latents (in spatial format, then pack)
    latent_channels = 16
    latent_h = height // 8
    latent_w = width // 8
    latents_spatial = torch.randn(
        (batch_size, latent_channels, latent_h, latent_w),
        dtype=dtype,
        device=device,
        generator=generator,
    )

    # Pack to sequence format for transformer
    latents = _pack_latents(latents_spatial, height, width)

    # 3. Prepare position IDs
    img_ids = _prepare_img_ids(batch_size, height, width, device, dtype)
    txt_ids = (
        text_ids.to(device=device, dtype=dtype)
        if text_ids is not None
        else torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
    )

    # 4. Set up scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    sigmas = pipeline.scheduler.sigmas

    # Scale initial latents
    latents = latents * sigmas[0]

    # 5. Denoising loop
    all_latents = []
    all_next_latents = []
    all_log_probs = []

    # Guidance embed (for FLUX guidance-distilled models)
    guidance = torch.full((batch_size,), guidance_scale, device=device, dtype=dtype)

    for i, sigma in enumerate(sigmas[:-1]):
        sigma_next = sigmas[i + 1]
        all_latents.append(latents.detach().cpu())

        timestep = sigma.expand(batch_size)

        # Transformer forward
        noise_pred = pipeline.transformer(
            hidden_states=latents,
            timestep=timestep / 1000.0,  # FLUX uses raw sigma, not sigma*1000
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]

        # SDE step (operates on packed sequence format)
        step_noise_level = noise_level if i < len(sigmas) - 2 else 0.0
        sigma_batch = sigma.expand(batch_size)
        sigma_next_batch = sigma_next.expand(batch_size)

        latents, log_prob, _ = sde_step_with_logprob(
            model_output=noise_pred,
            sigma=sigma_batch,
            sigma_next=sigma_next_batch,
            sample=latents,
            noise_level=step_noise_level,
            generator=generator,
        )
        latents = latents.to(dtype)

        all_next_latents.append(latents.detach().cpu())
        all_log_probs.append(log_prob.detach().cpu())

    # 6. Unpack and VAE decode
    latents_spatial = _unpack_latents(latents, height, width, channels=latent_channels)
    latents_spatial = latents_spatial / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    images_tensor = pipeline.vae.decode(latents_spatial, return_dict=False)[0].clamp(0, 1)

    # Convert to PIL
    images_np = (images_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    images_np = images_np.transpose(0, 2, 3, 1)
    images = [Image.fromarray(img) for img in images_np]

    return FluxSamplingOutput(
        images=images,
        latents_trajectory=torch.stack(all_latents, dim=1),
        next_latents_trajectory=torch.stack(all_next_latents, dim=1),
        log_probs=torch.stack(all_log_probs, dim=1),
        timesteps=sigmas.cpu(),
        prompt_embeds=prompt_embeds.detach().cpu(),
        pooled_embeds=pooled_prompt_embeds.detach().cpu(),
        img_ids=img_ids.detach().cpu(),
        txt_ids=txt_ids.detach().cpu(),
    )


def flux_replay_step(
    transformer,
    latent_t: torch.Tensor,
    next_latent_t: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance_scale: float = 3.5,
    noise_level: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay a single FLUX denoising step through the current model."""
    batch_size = latent_t.shape[0]
    device = latent_t.device
    dtype = latent_t.dtype

    guidance = torch.full((batch_size,), guidance_scale, device=device, dtype=dtype)
    timestep = sigma.expand(batch_size)

    noise_pred = transformer(
        hidden_states=latent_t,
        timestep=timestep / 1000.0,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_embeds,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        return_dict=False,
    )[0]

    sigma_batch = sigma.expand(batch_size)
    sigma_next_batch = sigma_next.expand(batch_size)

    _, log_prob, prev_sample_mean = sde_step_with_logprob(
        model_output=noise_pred,
        sigma=sigma_batch,
        sigma_next=sigma_next_batch,
        sample=latent_t,
        noise_level=noise_level,
        prev_sample=next_latent_t,
    )

    return log_prob, prev_sample_mean
