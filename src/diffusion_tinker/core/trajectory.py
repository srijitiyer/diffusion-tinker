from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image


@dataclass
class TrajectoryBatch:
    """Stores a batch of denoising trajectories with log-probabilities for RL training.

    Shapes (for SD3.5 at 512x512 with T=10 steps, B=batch_size):
        latents:        (B, T, 16, 64, 64)  - latent x_t before each step
        next_latents:   (B, T, 16, 64, 64)  - latent x_{t-1} after each step
        log_probs:      (B, T)              - old policy log-probs (mean-reduced over spatial)
        timesteps:      (T+1,)              - full sigma schedule (T steps + terminal 0)
        prompt_embeds:  (B, seq_len, dim)   - text encoder hidden states
        pooled_embeds:  (B, pooled_dim)     - pooled text embeddings
        rewards:        (B,)                - reward scores
        advantages:     (B,) or None        - computed advantages
    """

    latents: torch.Tensor
    next_latents: torch.Tensor
    log_probs: torch.Tensor
    timesteps: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_embeds: torch.Tensor
    prompts: list[str]
    negative_prompt_embeds: torch.Tensor | None = None
    negative_pooled_embeds: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    images: list[Image.Image] = field(default_factory=list)

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx) -> TrajectoryBatch:
        return TrajectoryBatch(
            latents=self.latents[idx],
            next_latents=self.next_latents[idx],
            log_probs=self.log_probs[idx],
            timesteps=self.timesteps,
            prompt_embeds=self.prompt_embeds[idx],
            pooled_embeds=self.pooled_embeds[idx],
            prompts=[self.prompts[i] for i in (range(len(self.prompts))[idx] if isinstance(idx, slice) else [idx])],
            negative_prompt_embeds=self.negative_prompt_embeds[idx] if self.negative_prompt_embeds is not None else None,
            negative_pooled_embeds=self.negative_pooled_embeds[idx] if self.negative_pooled_embeds is not None else None,
            rewards=self.rewards[idx] if self.rewards is not None else None,
            advantages=self.advantages[idx] if self.advantages is not None else None,
            images=[self.images[i] for i in (range(len(self.images))[idx] if isinstance(idx, slice) else [idx])]
            if self.images
            else [],
        )

    def to(self, device: torch.device | str) -> TrajectoryBatch:
        self.latents = self.latents.to(device)
        self.next_latents = self.next_latents.to(device)
        self.log_probs = self.log_probs.to(device)
        self.timesteps = self.timesteps.to(device)
        self.prompt_embeds = self.prompt_embeds.to(device)
        self.pooled_embeds = self.pooled_embeds.to(device)
        if self.negative_prompt_embeds is not None:
            self.negative_prompt_embeds = self.negative_prompt_embeds.to(device)
        if self.negative_pooled_embeds is not None:
            self.negative_pooled_embeds = self.negative_pooled_embeds.to(device)
        if self.rewards is not None:
            self.rewards = self.rewards.to(device)
        if self.advantages is not None:
            self.advantages = self.advantages.to(device)
        return self
