"""SDE step with log-probability computation for flow matching models.

This is the mathematical core of diffusion RL. It converts the deterministic
flow matching ODE into a stochastic differential equation (SDE) and computes
the Gaussian log-probability of each denoising transition.

Math reference: LOG_PROB_DERIVATIONS.md Section 2
Code reference: FlowGRPO's sd3_sde_with_logprob.py
"""

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
    """Perform one SDE denoising step and compute the Gaussian log-probability.

    Converts the flow matching ODE step into an SDE by injecting calibrated noise.
    The transition x_{t-1} | x_t is Gaussian: N(prev_sample_mean, noise_std^2 * I).

    Args:
        model_output: velocity prediction v_theta(x_t, t, c), shape (B, C, H, W)
        sigma: current noise level sigma_t, shape (B,) or scalar
        sigma_next: next noise level sigma_{t-1}, shape (B,) or scalar
        sample: current latent x_t, shape (B, C, H, W)
        noise_level: SDE noise injection strength (FlowGRPO default: 0.7)
        prev_sample: if provided, use this as x_{t-1} (for training replay).
                     If None, sample x_{t-1} from the SDE (for rollout).
        generator: random number generator for reproducibility

    Returns:
        prev_sample: the next latent x_{t-1}, shape (B, C, H, W)
        log_prob: Gaussian log-probability of the transition, shape (B,)
                  Mean-reduced over spatial dims (NOT summed).
        prev_sample_mean: deterministic mean of the transition, shape (B, C, H, W)
    """
    # Force float32 for numerical stability - bf16 overflows on these computations
    model_output = model_output.float()
    sample = sample.float()

    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor([sigma], device=sample.device, dtype=torch.float32)
    if not isinstance(sigma_next, torch.Tensor):
        sigma_next = torch.tensor([sigma_next], device=sample.device, dtype=torch.float32)

    sigma = sigma.float()
    sigma_next = sigma_next.float()

    # Broadcast sigma to match spatial dims: (B,) -> (B, 1, 1, 1)
    sigma_bc = _left_broadcast(sigma, sample.shape)
    sigma_next_bc = _left_broadcast(sigma_next, sample.shape)

    dt = sigma_next_bc - sigma_bc  # negative (going from noise toward clean)

    # SDE noise standard deviation
    # std_dev_t = sqrt(sigma / (1 - sigma)) * noise_level
    # Clamp sigma away from 1.0 to avoid division by zero
    sigma_clamped = sigma_bc.clamp(max=0.9999)
    std_dev_t = torch.sqrt(sigma_clamped / (1.0 - sigma_clamped)) * noise_level

    # Compute the SDE drift (mean of the transition)
    # From FlowGRPO Eq. 8: the SDE that preserves the same marginals as the ODE
    # prev_sample_mean = x_t + v_theta * dt + correction_term
    #
    # The correction term adjusts the ODE drift for the noise injection:
    #   correction = (std_dev_t^2 / (2*sigma)) * (x_t + (1-sigma)*v_theta) * dt
    #
    # Combined:
    #   mean = x_t * (1 + std_dev_t^2/(2*sigma) * dt) + v_theta * dt * (1 + std_dev_t^2*(1-sigma)/(2*sigma))
    correction_coeff = std_dev_t**2 / (2.0 * sigma_bc.clamp(min=1e-8))
    prev_sample_mean = sample * (1.0 + correction_coeff * dt) + model_output * dt * (
        1.0 + correction_coeff * (1.0 - sigma_bc)
    )

    # Per-step noise standard deviation
    noise_std = std_dev_t * torch.sqrt((-dt).clamp(min=1e-12))

    # Sample or replay the transition
    if prev_sample is None:
        noise = torch.randn(sample.shape, dtype=sample.dtype, device=sample.device, generator=generator)
        prev_sample = prev_sample_mean + noise_std * noise
    else:
        prev_sample = prev_sample.float()

    # Gaussian log-probability: log N(prev_sample | mean, noise_std^2 * I)
    # log_prob = -(x - mu)^2 / (2*sigma^2) - log(sigma) - 0.5*log(2*pi)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * noise_std**2 + 1e-12)
        - torch.log(noise_std + 1e-12)
        - 0.5 * math.log(2.0 * math.pi)
    )

    # Mean-reduce over spatial dims (C, H, W) -> (B,)
    # Using MEAN not SUM - clip_range is calibrated for mean-reduced log-probs
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
    """Compute the standard flow matching denoising loss (DDRL's data regularization term).

    This IS the forward KL divergence D_KL(p_ref || p_theta), which reduces to the
    standard diffusion training loss. See DDRL_THEORY.md Section 4.

    Args:
        transformer: the denoising model
        latents: clean latents x_0, shape (B, C, H, W)
        noise: sampled noise epsilon ~ N(0,I), shape (B, C, H, W)
        sigmas: noise levels in [0, 1], shape (B,)
        prompt_embeds: text conditioning, shape (B, seq_len, dim)
        pooled_embeds: pooled text conditioning, shape (B, pooled_dim)
        condition_dropout_mask: bool tensor (B,) - True = drop condition

    Returns:
        loss: scalar MSE loss between predicted and target velocity
    """
    sigmas_bc = _left_broadcast(sigmas, latents.shape)

    # Rectified flow forward process: x_t = (1-t)*x_0 + t*epsilon
    noisy_latents = (1.0 - sigmas_bc) * latents + sigmas_bc * noise

    # Apply condition dropout for CFG (zero both sequence and pooled embeddings)
    if condition_dropout_mask is not None:
        seq_mask = _left_broadcast(condition_dropout_mask.float(), prompt_embeds.shape)
        prompt_embeds = prompt_embeds * (1.0 - seq_mask)
        pool_mask = _left_broadcast(condition_dropout_mask.float(), pooled_embeds.shape)
        pooled_embeds = pooled_embeds * (1.0 - pool_mask)

    # Model predicts velocity
    timestep = sigmas * 1000.0  # scheduler convention
    v_pred = transformer(
        hidden_states=noisy_latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_embeds,
        return_dict=False,
    )[0]

    # Target velocity: v = epsilon - x_0
    v_target = noise - latents

    return torch.nn.functional.mse_loss(v_pred, v_target)
