"""SDE step with log-probability for flow matching models."""

from __future__ import annotations

import math

import torch


def _left_broadcast(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Broadcast (B,) tensor to (B, 1, 1, ...) matching target ndim."""
    return t.reshape(t.shape[0], *([1] * (len(shape) - 1)))


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

    sigma_clamped = sigma_bc.clamp(max=0.9999)
    std_dev_t = torch.sqrt(sigma_clamped / (1.0 - sigma_clamped)) * noise_level

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
) -> torch.Tensor:
    sigmas_bc = _left_broadcast(sigmas, latents.shape)

    noisy_latents = (1.0 - sigmas_bc) * latents + sigmas_bc * noise

    if condition_dropout_mask is not None:
        seq_mask = _left_broadcast(condition_dropout_mask.float(), prompt_embeds.shape)
        prompt_embeds = prompt_embeds * (1.0 - seq_mask)
        pool_mask = _left_broadcast(condition_dropout_mask.float(), pooled_embeds.shape)
        pooled_embeds = pooled_embeds * (1.0 - pool_mask)

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
