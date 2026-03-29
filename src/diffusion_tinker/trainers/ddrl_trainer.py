"""DDRL Trainer - Data-regularized Reinforcement Learning for Diffusion Models.

Implements the DDRL algorithm from arXiv:2512.04332 (Haotian Ye et al.).

Key innovation: replaces reverse KL (which breaks under distribution shift) with
forward KL, which reduces to the standard diffusion denoising loss. Combined loss:

    L = L_RL + data_beta * L_data

where L_RL is the clipped policy gradient and L_data is the standard denoising loss.

Reference: DDRL_THEORY.md, RL_LOSS_FUNCTIONS.md Section 4
"""

from __future__ import annotations

import random

import torch

from diffusion_tinker.core.noise_strategy import compute_flow_matching_loss
from diffusion_tinker.core.trajectory import TrajectoryBatch
from diffusion_tinker.models.sd3_patch import sd3_replay_step
from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer
from diffusion_tinker.trainers.ddrl_config import DDRLConfig


class DDRLTrainer(BaseDiffusionTrainer):
    """Trainer implementing DDRL (Data-regularized RL for Diffusion Models).

    Usage:
        from diffusion_tinker import DDRLTrainer, DDRLConfig

        trainer = DDRLTrainer(
            model="stabilityai/stable-diffusion-3.5-medium",
            reward_funcs="aesthetic",
            train_prompts=["a photo of a cat", "a sunset over mountains"],
            config=DDRLConfig(data_beta=0.01),
        )
        trainer.train()
    """

    config: DDRLConfig

    def _compute_advantages(self, trajectory: TrajectoryBatch) -> TrajectoryBatch:
        """DDRL advantage computation with monotonic transform.

        Standard normalization: A = (r - mean) / (beta_temp * std + eps)
        Monotonic transform: A = -exp(-A)

        The -exp(-x) transform (Theorem 3.1) ensures the optimal policy takes the
        Boltzmann form p* ~ p_data * exp(r/beta). It compresses large positive
        advantages and amplifies negatives, providing natural gradient stability.
        """
        # Per-prompt normalization
        advantages = self.stat_tracker.update(trajectory.prompts, trajectory.rewards)

        # Temperature scaling
        if self.config.beta_temp != 1.0:
            advantages = advantages / self.config.beta_temp

        # Monotonic transform: lambda(x) = -exp(-x)
        if self.config.use_monotonic_transform:
            advantages = -torch.exp(-advantages)

        # Clip
        advantages = torch.clamp(advantages, -self.config.adv_clip_max, self.config.adv_clip_max)
        trajectory.advantages = advantages
        return trajectory

    def _training_step(self, trajectory: TrajectoryBatch) -> dict[str, float]:
        """DDRL training step: L = L_RL + data_beta * L_data.

        Iterates over denoising timesteps, accumulating gradients. Each timestep:
        1. Replay the stored transition through the current model to get new log-probs
        2. Compute importance ratio and clipped PPO loss (RL loss)
        3. Compute standard denoising loss on the same prompts (data loss)
        4. Combine: loss = rl_loss + data_beta * data_loss
        """
        device = self.device
        config = self.config
        batch_size = len(trajectory)
        num_timesteps = trajectory.log_probs.shape[1]

        # Move trajectory data to device
        trajectory = trajectory.to(device)

        total_rl_loss = 0.0
        total_data_loss = 0.0
        total_ratio = 0.0
        num_steps = 0

        # Randomize timestep order (per TRAINING_LOOP_INTERNALS.md Section 3.1)
        timestep_indices = list(range(num_timesteps))
        random.shuffle(timestep_indices)

        self.optimizer.zero_grad()

        for j in timestep_indices:
            sigma = trajectory.timesteps[j].to(device)
            sigma_next = (
                trajectory.timesteps[j + 1].to(device)
                if j + 1 < len(trajectory.timesteps)
                else torch.tensor(0.0, device=device)
            )

            # Skip last step (noise_level=0, no gradient signal)
            if sigma_next.item() < 1e-6:
                continue

            latent_t = trajectory.latents[:, j].to(device)
            next_latent_t = trajectory.next_latents[:, j].to(device)

            # === RL LOSS ===
            # Replay through current model to get updated log-probs
            log_prob_new, prev_sample_mean = sd3_replay_step(
                transformer=self.transformer,
                latent_t=latent_t,
                next_latent_t=next_latent_t,
                sigma=sigma,
                sigma_next=sigma_next,
                prompt_embeds=trajectory.prompt_embeds.to(device),
                pooled_embeds=trajectory.pooled_embeds.to(device),
                guidance_scale=config.guidance_scale,
                noise_level=config.noise_level,
            )

            log_prob_old = trajectory.log_probs[:, j].to(device)

            # Importance ratio
            ratio = torch.exp(log_prob_new - log_prob_old)

            # PPO clipped surrogate loss
            advantages = trajectory.advantages
            unclipped = -advantages * ratio
            clipped = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
            rl_loss = torch.mean(torch.maximum(unclipped, clipped))

            # === DATA LOSS (forward KL = standard denoising loss) ===
            data_loss = torch.tensor(0.0, device=device)
            if config.data_beta > 0:
                # Sample random timestep for denoising loss (logit-normal for SD3)
                u = torch.normal(
                    mean=config.logit_mean,
                    std=config.logit_std,
                    size=(batch_size,),
                    device=device,
                )
                t = torch.sigmoid(u)

                # Use the trajectory's latents as "clean" data for the denoising loss.
                # Ideally we'd use a separate dataset, but for MVP we approximate by
                # denoising from the current trajectory's initial/final latents.
                # The next_latents at the last step are closest to clean images.
                clean_latents = trajectory.next_latents[:, -1].to(device)
                noise = torch.randn_like(clean_latents)

                # Condition dropout mask
                dropout_mask = torch.rand(batch_size, device=device) < config.condition_dropout

                data_loss = compute_flow_matching_loss(
                    transformer=self.transformer,
                    latents=clean_latents,
                    noise=noise,
                    sigmas=t,
                    prompt_embeds=trajectory.prompt_embeds.to(device),
                    pooled_embeds=trajectory.pooled_embeds.to(device),
                    condition_dropout_mask=dropout_mask,
                )

            # === COMBINED LOSS ===
            loss = rl_loss + config.data_beta * data_loss

            # Scale by number of timesteps (gradient accumulation across steps)
            loss = loss / len(timestep_indices)
            loss.backward()

            total_rl_loss += rl_loss.item()
            total_data_loss += data_loss.item() if isinstance(data_loss, torch.Tensor) else data_loss
            total_ratio += ratio.mean().item()
            num_steps += 1

        # Optimizer step (after all timesteps accumulated)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.transformer.parameters() if p.requires_grad],
            config.max_grad_norm,
        )
        self.optimizer.step()

        n = max(num_steps, 1)
        return {
            "rl_loss": total_rl_loss / n,
            "data_loss": total_data_loss / n,
            "total_loss": (total_rl_loss + config.data_beta * total_data_loss) / n,
            "mean_ratio": total_ratio / n,
            "mean_reward": trajectory.rewards.mean().item(),
            "mean_advantage": trajectory.advantages.mean().item(),
        }
