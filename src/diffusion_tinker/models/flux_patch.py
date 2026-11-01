"""FLUX pipeline with SDE sampling and log-probability collection."""

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
    step_noise_levels: torch.Tensor  # (T,) noise level each step was sampled at
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
    sde_window: tuple[int, int] | None = None,
    init_latents: torch.Tensor | None = None,
    height: int = 1024,
    width: int = 1024,
    generator: torch.Generator | None = None,
) -> FluxSamplingOutput:
    """Run FLUX pipeline with SDE sampling, collecting trajectories and log-probs."""
    device = pipeline.transformer.device
    dtype = pipeline.transformer.dtype
    batch_size = len(prompts)

    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        device=device,
        max_sequence_length=512,
    )

    latent_channels = 16
    latent_h = height // 8
    latent_w = width // 8
    if init_latents is not None:
        # see sd3_patch: eval supplies its own noise so chunking cannot move it
        latents_spatial = init_latents.to(device=device, dtype=dtype)
    else:
        latents_spatial = torch.randn(
            (batch_size, latent_channels, latent_h, latent_w),
            dtype=dtype,
            device=device,
            generator=generator,
        )

    latents = _pack_latents(latents_spatial, height, width)

    img_ids = _prepare_img_ids(batch_size, height, width, device, dtype)
    txt_ids = (
        text_ids.to(device=device, dtype=dtype)
        if text_ids is not None
        else torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
    )

    # FLUX.1-dev's scheduler is configured with use_dynamic_shifting=True, which
    # shifts the timestep schedule by a resolution-dependent mu and refuses to
    # build a schedule without it. FLUX.1-schnell leaves it off, so passing no mu
    # worked there and hid this on the only FLUX model we had run.
    sched_kwargs = {}
    if getattr(pipeline.scheduler.config, "use_dynamic_shifting", False):
        sc = pipeline.scheduler.config
        image_seq_len = latents.shape[1]
        try:
            from diffusers.pipelines.flux.pipeline_flux import calculate_shift
            mu = calculate_shift(
                image_seq_len,
                sc.base_image_seq_len,
                sc.max_image_seq_len,
                sc.base_shift,
                sc.max_shift,
            )
        except ImportError:
            slope = (sc.max_shift - sc.base_shift) / (sc.max_image_seq_len - sc.base_image_seq_len)
            mu = image_seq_len * slope + (sc.base_shift - slope * sc.base_image_seq_len)
        sched_kwargs["mu"] = mu

    # FLUX needs its sigma grid passed explicitly. FluxPipeline.__call__ always
    # sets sigmas = linspace(1.0, 1/N, N) before calling the scheduler;
    # StableDiffusion3Pipeline leaves sigmas=None and lets the scheduler build its
    # own. This call was copied from the SD3 path and omitted the argument, so the
    # scheduler fell back to linspace over _sigma_to_t, which bottoms out at
    # 1/num_train_timesteps = 0.001 rather than 1/N.
    #
    # Measured at 512px, N=28: the schedule ended 0.1321, 0.0690, 0.0019 against
    # the correct 0.1839, 0.1262, 0.0650. The final Euler step covered |dt|=0.0019,
    # 0.19% of the trajectory - one wasted step in 28 - and the step before it had
    # to absorb the whole low-noise refinement range in a single jump. Every FLUX
    # rollout and every FLUX eval image in this repo before this fix was produced
    # on that grid, so FLUX's baseline was depressed against stock FLUX and
    # FLUX-vs-SD3 comparisons were not like-for-like.
    import numpy as _np
    sched_kwargs["sigmas"] = _np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device, **sched_kwargs)
    sigmas = pipeline.scheduler.sigmas

    latents = latents * sigmas[0]

    all_latents = []
    all_next_latents = []
    all_log_probs = []
    all_step_noise = []

    # FLUX.1-dev embeds a guidance scalar (guidance_embeds=True); FLUX.1-schnell
    # is not guidance-distilled and rejects a guidance tensor, so pass None there.
    use_guidance = getattr(pipeline.transformer.config, "guidance_embeds", True)
    guidance = torch.full((batch_size,), guidance_scale, device=device, dtype=dtype) if use_guidance else None
    # Match replay precision (float32 LoRA + bf16 base) so the importance ratio
    # is unbiased - see the equivalent note in sd3_patch.
    autocast_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16

    # See sd3_patch: FlowGRPO runs FLUX with a short stochastic window
    # (sde_window_size=3 inside (0, num_steps//2) on a 6-step model) and
    # deterministically elsewhere, training only on the window. This is the model
    # it actually matters for - FLUX.1-dev is guidance-distilled and few-step, and
    # injecting exploration noise at every step is not how they drive it.
    if sde_window is None:
        w_start, w_end = 0, len(sigmas) - 1
    else:
        w_start, w_end = sde_window

    for i, sigma in enumerate(sigmas[:-1]):
        sigma_next = sigmas[i + 1]
        in_window = w_start <= i < w_end
        if in_window:
            all_latents.append(latents.detach().cpu())

        timestep = sigma.expand(batch_size)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, cache_enabled=False):
            noise_pred = pipeline.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

        step_noise_level = noise_level if i < len(sigmas) - 2 else 0.0
        if not in_window:
            step_noise_level = 0.0
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

        # Store the float32 sampled latent as the replay target (see sd3_patch):
        # rounding to bf16 before storing shifts the target and, with the SDE
        # std near 0 at low sigma, biases the importance ratio. Round only for
        # the next transformer input.
        if in_window:
            all_next_latents.append(latents.detach().cpu())
            all_log_probs.append(log_prob.detach().cpu())
            # Record what this step was actually sampled at. The replay indexes
            # the window locally and cannot re-derive this: on the last stored
            # step its local index looks terminal even when the global one is
            # not, and replaying at noise 0 makes the Gaussian degenerate, the
            # ratio underflow to exactly 0, and the gradient vanish silently.
            all_step_noise.append(float(step_noise_level))

        latents = latents.to(dtype)

    latents_spatial = _unpack_latents(latents, height, width, channels=latent_channels)
    latents_spatial = latents_spatial / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    # VAE decode is [-1, 1]; rescale to [0, 1] before clamping (see sd3_patch).
    # Decode in small chunks so a large sample group doesn't OOM on the VAE
    # upsample (the biggest single allocation in the rollout).
    chunk_size = 4
    chunks = []
    for i in range(0, latents_spatial.shape[0], chunk_size):
        dec = pipeline.vae.decode(latents_spatial[i : i + chunk_size], return_dict=False)[0]
        chunks.append((dec / 2 + 0.5).clamp(0, 1))
    images_tensor = torch.cat(chunks, dim=0)

    images_np = (images_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    images_np = images_np.transpose(0, 2, 3, 1)
    images = [Image.fromarray(img) for img in images_np]

    return FluxSamplingOutput(
        images=images,
        latents_trajectory=torch.stack(all_latents, dim=1),
        next_latents_trajectory=torch.stack(all_next_latents, dim=1),
        log_probs=torch.stack(all_log_probs, dim=1),
        step_noise_levels=torch.tensor(all_step_noise),
        timesteps=sigmas[w_start:w_end + 1].cpu(),
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

    use_guidance = getattr(transformer.config, "guidance_embeds", True)
    guidance = torch.full((batch_size,), guidance_scale, device=device, dtype=dtype) if use_guidance else None
    timestep = sigma.expand(batch_size)

    noise_pred = transformer(
        hidden_states=latent_t,
        timestep=timestep,
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
