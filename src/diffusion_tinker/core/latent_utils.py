from __future__ import annotations

import torch
from diffusers import AutoencoderKL


def encode_to_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    """Encode pixel images to latent space with proper normalization.

    Args:
        vae: the VAE model
        images: (B, 3, H, W) in [0, 1] range

    Returns:
        latents: (B, C, H//8, W//8) normalized for training
    """
    with torch.no_grad():
        latent_dist = vae.encode(images).latent_dist
        latents = latent_dist.sample()

    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def decode_from_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to pixel images.

    Args:
        vae: the VAE model
        latents: (B, C, H//8, W//8) in normalized space

    Returns:
        images: (B, 3, H, W) in [0, 1] range
    """
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor

    with torch.no_grad():
        images = vae.decode(latents, return_dict=False)[0]

    images = images.clamp(0, 1)
    return images


def prepare_noise_latents(
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create initial noise latents x_T ~ N(0, I)."""
    shape = (batch_size, num_channels, height // 8, width // 8)
    return torch.randn(shape, dtype=dtype, device=device, generator=generator)
