"""SDE step with log-probability for flow matching models."""

from __future__ import annotations

import math

import torch


def _left_broadcast(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Broadcast (B,) tensor to (B, 1, 1, ...) matching target ndim."""
    return t.reshape(t.shape[0], *([1] * (len(shape) - 1)))


def sde_diffusion_std(
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    noise_level: float,
) -> torch.Tensor:
    """Diffusion coefficient of the reverse SDE, bounded so the Euler step is stable.

    sqrt(sigma/(1-sigma)) diverges as sigma -> 1, and the SD3/FLUX schedules start
    at sigma exactly 1.0. The score term stays finite there, so the blow-up is
    purely this coefficient meeting a finite Euler step: unbounded, the first step
    multiplies the latent by about -56 at noise_level=0.7 (and by -0.16 at 0.1,
    which also makes noise levels incomparable). Capping so
    |correction_coeff * dt| <= 1/2 keeps the coefficient on `sample` in (0, 1] and
    leaves the SDE untouched wherever it is well conditioned - on the real
    schedules the cap binds only on the first step.

    This is the single source of truth: the KL terms in FlowGRPO and DDPO must use
    the same variance as the transition the policy actually sampled from, so they
    call this rather than re-deriving it.
    """
    sigma_clamped = sigma.clamp(max=0.9999)
    std_dev_t = torch.sqrt(sigma_clamped / (1.0 - sigma_clamped)) * noise_level

    dt = sigma_next - sigma
    max_std_sq = sigma.clamp(min=1e-8) / (-dt).clamp(min=1e-12)
    return torch.minimum(std_dev_t, torch.sqrt(max_std_sq))


def sde_step_with_logprob(
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    sample: torch.Tensor,
    noise_level: float = 0.7,
    prev_sample: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_output = model_output.float()
    sample = sample.float()

    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor([sigma], device=sample.device, dtype=torch.float32)
    if not isinstance(sigma_next, torch.Tensor):
        sigma_next = torch.tensor([sigma_next], device=sample.device, dtype=torch.float32)

    sigma = sigma.float()
    sigma_next = sigma_next.float()

    sigma_bc = _left_broadcast(sigma, sample.shape)
    sigma_next_bc = _left_broadcast(sigma_next, sample.shape)

    dt = sigma_next_bc - sigma_bc

    std_dev_t = sde_diffusion_std(sigma_bc, sigma_next_bc, noise_level)

    correction_coeff = std_dev_t**2 / (2.0 * sigma_bc.clamp(min=1e-8))
    prev_sample_mean = sample * (1.0 + correction_coeff * dt) + model_output * dt * (
        1.0 + correction_coeff * (1.0 - sigma_bc)
    )

    noise_std = std_dev_t * torch.sqrt((-dt).clamp(min=1e-12))

    if prev_sample is None:
        noise = torch.randn(sample.shape, dtype=sample.dtype, device=sample.device, generator=generator)
        prev_sample = prev_sample_mean + noise_std * noise
    else:
        prev_sample = prev_sample.float()

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * noise_std**2 + 1e-12)
        - torch.log(noise_std + 1e-12)
        - 0.5 * math.log(2.0 * math.pi)
    )

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean


def compute_flow_matching_loss(
    transformer: torch.nn.Module,
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    condition_dropout_mask: torch.Tensor | None = None,
    model_type: str = "sd3",
    guidance_scale: float = 3.5,
    uncond_embeds: torch.Tensor | None = None,
    uncond_pooled: torch.Tensor | None = None,
) -> torch.Tensor:
    sigmas_bc = _left_broadcast(sigmas, latents.shape)

    noisy_latents = (1.0 - sigmas_bc) * latents + sigmas_bc * noise

    if condition_dropout_mask is not None:
        # Dropping a caption means replacing it with the empty string and encoding
        # that, not multiplying its encoding by zero. The two are different inputs
        # and only the first is one the model has seen. SD3 builds its
        # unconditional branch exactly this way (`negative_prompt=[""]` in
        # sd3_patch) and evaluates it on every CFG step. FLUX.1-dev is
        # guidance-distilled and has no unconditional branch at all, so a zeroed
        # embedding handed to it alongside a live guidance scalar is a state it
        # has never seen in training or sampling - which is why this mattered
        # enough to change.
        if uncond_embeds is None or uncond_pooled is None:
            raise ValueError(
                "condition_dropout_mask requires uncond_embeds and uncond_pooled: "
                "dropped captions are replaced with an encoded empty prompt, not "
                "zeroed. Encode '' once and pass it here."
            )
        seq_mask = _left_broadcast(condition_dropout_mask.float(), prompt_embeds.shape)
        u_seq = uncond_embeds.to(prompt_embeds.dtype)
        if u_seq.shape[0] == 1:
            u_seq = u_seq.expand_as(prompt_embeds)
        prompt_embeds = prompt_embeds * (1.0 - seq_mask) + u_seq * seq_mask

        pool_mask = _left_broadcast(condition_dropout_mask.float(), pooled_embeds.shape)
        u_pool = uncond_pooled.to(pooled_embeds.dtype)
        if u_pool.shape[0] == 1:
            u_pool = u_pool.expand_as(pooled_embeds)
        pooled_embeds = pooled_embeds * (1.0 - pool_mask) + u_pool * pool_mask

    if model_type == "flux":
        # FLUX takes packed (B, tokens, 64) latents plus rotary position ids, and
        # reads the timestep as sigma directly rather than sigma * 1000 like SD3.
        # Packing is elementwise, so it commutes with the noising above.
        from diffusion_tinker.models.flux_patch import _pack_latents, _prepare_img_ids

        batch_size = noisy_latents.shape[0]
        height = noisy_latents.shape[2] * 8
        width = noisy_latents.shape[3] * 8
        device, dtype = noisy_latents.device, noisy_latents.dtype

        hidden = _pack_latents(noisy_latents, height, width)
        target = _pack_latents(noise - latents, height, width)
        img_ids = _prepare_img_ids(batch_size, height, width, device, dtype)
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

        use_guidance = getattr(transformer.config, "guidance_embeds", True)
        guidance = (
            torch.full((batch_size,), guidance_scale, device=device, dtype=dtype)
            if use_guidance
            else None
        )

        v_pred = transformer(
            hidden_states=hidden,
            timestep=sigmas,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
        return torch.nn.functional.mse_loss(v_pred, target)

    timestep = sigmas * 1000.0
    v_pred = transformer(
        hidden_states=noisy_latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_embeds,
        return_dict=False,
    )[0]

    v_target = noise - latents

    return torch.nn.functional.mse_loss(v_pred, v_target)
