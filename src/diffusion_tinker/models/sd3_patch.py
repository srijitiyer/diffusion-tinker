"""SD3 pipeline with SDE sampling and log-probability collection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from diffusion_tinker.core.noise_strategy import sde_step_with_logprob


@dataclass
class SD3ModelConfig:
    """Static configuration for SD3/SD3.5 architecture."""

    vae_channels: int = 16
    prediction_type: str = "flow_velocity"
    noise_type: str = "rectified_flow"
    lora_target_modules: list[str] | None = None

    def __post_init__(self):
        if self.lora_target_modules is None:
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
class SD3SamplingOutput:
    """Output from SD3 pipeline sampling with log-probs."""

    images: list[Image.Image]
    latents_trajectory: torch.Tensor  # (B, T, C, H, W) - latent before each step
    next_latents_trajectory: torch.Tensor  # (B, T, C, H, W) - latent after each step
    log_probs: torch.Tensor  # (B, T)
    timesteps: torch.Tensor  # (T+1,) - full sigma schedule (includes terminal sigma=0)
    prompt_embeds: torch.Tensor  # (B, seq_len, dim)
    pooled_embeds: torch.Tensor  # (B, pooled_dim)
    negative_prompt_embeds: torch.Tensor | None = None
    negative_pooled_embeds: torch.Tensor | None = None


@torch.no_grad()
def sd3_sample_with_logprob(
    pipeline: StableDiffusion3Pipeline,
    prompts: list[str],
    num_inference_steps: int = 10,
    guidance_scale: float = 7.0,
    noise_level: float = 0.7,
    height: int = 512,
    width: int = 512,
    generator: torch.Generator | None = None,
) -> SD3SamplingOutput:
    """Run SD3 pipeline with SDE sampling, collecting trajectories and log-probs."""
    device = pipeline.transformer.device
    dtype = pipeline.transformer.dtype
    batch_size = len(prompts)
    do_cfg = guidance_scale > 1.0

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=[""] * batch_size if do_cfg else None,
        do_classifier_free_guidance=do_cfg,
        device=device,
    )

    latent_channels = pipeline.transformer.config.in_channels
    latents = torch.randn(
        (batch_size, latent_channels, height // 8, width // 8),
        dtype=dtype,
        device=device,
        generator=generator,
    )

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    sigmas = pipeline.scheduler.sigmas

    latents = latents * sigmas[0]

    all_latents = []
    all_next_latents = []
    all_log_probs = []

    for i, sigma in enumerate(sigmas[:-1]):
        sigma_next = sigmas[i + 1]

        all_latents.append(latents.detach().cpu())

        latent_model_input = torch.cat([latents] * 2, dim=0) if do_cfg else latents
        sigma_input = sigma.expand(latent_model_input.shape[0])
        if do_cfg:
            model_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            model_pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            model_prompt_embeds = prompt_embeds
            model_pooled_embeds = pooled_prompt_embeds

        noise_pred = pipeline.transformer(
            hidden_states=latent_model_input,
            timestep=sigma_input * 1000.0,
            encoder_hidden_states=model_prompt_embeds,
            pooled_projections=model_pooled_embeds,
            return_dict=False,
        )[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

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

    decoded = _decode_latents(pipeline, latents)
    images = _tensor_to_pil(decoded)
    latents_traj = torch.stack(all_latents, dim=1)
    next_latents_traj = torch.stack(all_next_latents, dim=1)
    log_probs_traj = torch.stack(all_log_probs, dim=1)

    return SD3SamplingOutput(
        images=images,
        latents_trajectory=latents_traj,
        next_latents_trajectory=next_latents_traj,
        log_probs=log_probs_traj,
        timesteps=sigmas.cpu(),
        prompt_embeds=prompt_embeds.detach().cpu(),
        pooled_embeds=pooled_prompt_embeds.detach().cpu(),
        negative_prompt_embeds=negative_prompt_embeds.detach().cpu() if do_cfg else None,
        negative_pooled_embeds=negative_pooled_prompt_embeds.detach().cpu() if do_cfg else None,
    )


def sd3_replay_step(
    transformer: torch.nn.Module,
    latent_t: torch.Tensor,
    next_latent_t: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    guidance_scale: float = 7.0,
    noise_level: float = 0.7,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_pooled_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay a single denoising step to get updated log-probs for importance sampling."""
    batch_size = latent_t.shape[0]
    do_cfg = guidance_scale > 1.0

    if do_cfg:
        if negative_prompt_embeds is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        if negative_pooled_embeds is None:
            negative_pooled_embeds = torch.zeros_like(pooled_embeds)

        latent_input = torch.cat([latent_t, latent_t], dim=0)
        model_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        model_pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds], dim=0)
        sigma_input = sigma.expand(batch_size * 2)
    else:
        latent_input = latent_t
        model_prompt_embeds = prompt_embeds
        model_pooled_embeds = pooled_embeds
        sigma_input = sigma.expand(batch_size)

    noise_pred = transformer(
        hidden_states=latent_input,
        timestep=sigma_input * 1000.0,
        encoder_hidden_states=model_prompt_embeds,
        pooled_projections=model_pooled_embeds,
        return_dict=False,
    )[0]

    if do_cfg:
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    sigma_batch = sigma.expand(batch_size)
    sigma_next_batch = sigma_next.expand(batch_size)

    _, log_prob, prev_sample_mean = sde_step_with_logprob(
        model_output=noise_pred,
        sigma=sigma_batch,
        sigma_next=sigma_next_batch,
        sample=latent_t,
        noise_level=noise_level,
        prev_sample=next_latent_t,  # replay stored transition
    )

    return log_prob, prev_sample_mean


def _decode_latents(pipeline: StableDiffusion3Pipeline, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to pixel space with SD3 normalization."""
    latents = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    return images.clamp(0, 1)


def _tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """Convert (B, 3, H, W) float tensor in [0,1] to list of PIL Images."""
    images_np = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    images_np = images_np.transpose(0, 2, 3, 1)
    return [Image.fromarray(img) for img in images_np]
