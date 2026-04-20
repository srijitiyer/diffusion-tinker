"""DDPO Trainer (arXiv:2305.13301)."""

from __future__ import annotations

import random

import torch

from diffusion_tinker.core.trajectory import TrajectoryBatch
from diffusion_tinker.models.sd3_patch import sd3_replay_step
from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer
from diffusion_tinker.trainers.ddpo_config import DDPOConfig


class DDPOTrainer(BaseDiffusionTrainer):
    """Trainer implementing DDPO/DPOK."""

    config: DDPOConfig

    def _training_step(self, trajectory: TrajectoryBatch) -> dict[str, float]:
        """DDPO training step with optional multi-epoch PPO."""
        device = self.device
        config = self.config

        trajectory = trajectory.to(device)
        num_steps = trajectory.log_probs.shape[1]

        total_rl_loss = 0.0
        total_kl_loss = 0.0
        total_ratio = 0.0
        total_computed = 0

        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        self.optimizer.zero_grad()

        for ppo_epoch in range(config.ppo_epochs):
            timestep_indices = list(range(num_steps))
            random.shuffle(timestep_indices)

            for j in timestep_indices:
                sigma = trajectory.timesteps[j]
                sigma_next = trajectory.timesteps[j + 1]

                if sigma_next.item() < 1e-6:
                    continue

                latent_t = trajectory.latents[:, j]
                next_latent_t = trajectory.next_latents[:, j]

                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    log_prob_new, prev_sample_mean = sd3_replay_step(
                        transformer=self.transformer,
                        latent_t=latent_t,
                        next_latent_t=next_latent_t,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        prompt_embeds=trajectory.prompt_embeds,
                        pooled_embeds=trajectory.pooled_embeds,
                        guidance_scale=config.guidance_scale,
                        noise_level=config.noise_level,
                    )

                log_prob_old = trajectory.log_probs[:, j]
                ratio = torch.exp(log_prob_new.float() - log_prob_old.float())

                advantages = trajectory.advantages
                unclipped = -advantages * ratio
                clipped = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                rl_loss = torch.mean(torch.maximum(unclipped, clipped))

                # Optional per-step KL (DPOK)
                kl_loss = torch.tensor(0.0, device=device)
                if config.kl_beta > 0:
                    with torch.no_grad():
                        self.transformer.disable_adapter_layers()
                        try:
                            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                                _, prev_sample_mean_ref = sd3_replay_step(
                                    transformer=self.transformer,
                                    latent_t=latent_t,
                                    next_latent_t=next_latent_t,
                                    sigma=sigma,
                                    sigma_next=sigma_next,
                                    prompt_embeds=trajectory.prompt_embeds,
                                    pooled_embeds=trajectory.pooled_embeds,
                                    guidance_scale=config.guidance_scale,
                                    noise_level=config.noise_level,
                                )
                        finally:
                            self.transformer.enable_adapter_layers()

                    sigma_val = sigma.float().clamp(max=0.9999)
                    dt = (sigma_next - sigma).float()
                    std_dev_t = torch.sqrt(sigma_val / (1.0 - sigma_val)) * config.noise_level
                    noise_std = std_dev_t * torch.sqrt((-dt).clamp(min=1e-12))

                    diff = (prev_sample_mean.float() - prev_sample_mean_ref.float()).pow(2)
                    kl_per_sample = diff.mean(dim=tuple(range(1, diff.ndim))) / (2.0 * noise_std.pow(2) + 1e-12)
                    kl_loss = kl_per_sample.mean()

                loss = rl_loss + config.kl_beta * kl_loss
                loss = loss / (len(timestep_indices) * config.ppo_epochs)
                if loss.requires_grad:
                    loss.backward()

                total_rl_loss += rl_loss.item()
                total_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                total_ratio += ratio.mean().item()
                total_computed += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.transformer.parameters() if p.requires_grad],
            config.max_grad_norm,
        )
        self.optimizer.step()

        n = max(total_computed, 1)
        return {
            "rl_loss": total_rl_loss / n,
            "kl_loss": total_kl_loss / n,
            "mean_ratio": total_ratio / n,
            "mean_reward": trajectory.rewards.mean().item(),
            "mean_advantage": trajectory.advantages.mean().item(),
        }
