"""DDPO Trainer (arXiv:2305.13301)."""

from __future__ import annotations

import random

import torch

from diffusion_tinker.core.noise_strategy import sde_diffusion_std
from diffusion_tinker.core.stat_tracking import BufferedPerPromptStatTracker
from diffusion_tinker.core.trajectory import TrajectoryBatch
from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer
from diffusion_tinker.trainers.ddpo_config import DDPOConfig


class DDPOTrainer(BaseDiffusionTrainer):
    """Trainer implementing DDPO/DPOK."""

    config: DDPOConfig

    def _compute_advantages(self, trajectory: TrajectoryBatch) -> TrajectoryBatch:
        """DDPO normalizes against a per-prompt reward history, not the batch."""
        if not hasattr(self, "_ddpo_tracker"):
            self._ddpo_tracker = BufferedPerPromptStatTracker(
                buffer_size=self.config.stat_buffer_size,
                min_count=self.config.stat_min_count,
            )
        advantages = self._ddpo_tracker.update(trajectory.prompts, trajectory.rewards)
        advantages = torch.clamp(advantages, -self.config.adv_clip_max, self.config.adv_clip_max)
        trajectory.advantages = advantages
        return trajectory

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

        # Mini-batch the replay over trajectories and step once per mini-batch
        # (true minibatch SGD), so large group sizes fit in memory and the reward
        # gets many updates per epoch instead of one. (See FlowGRPO trainer.)
        batch_size = len(trajectory)
        mb_size = config.train_batch_size or batch_size

        # See flowgrpo_trainer: when a window is active log_probs already holds
        # only the window's steps, so applying the fraction again takes a second
        # bite out of an already-small set. Upstream applies one or the other.
        if getattr(config, "sde_window_size", 0) > 0:
            num_train = num_steps
        else:
            num_train = max(1, int(num_steps * config.timestep_fraction))

        # Skip the update when the advantages carry no signal. FlowGRPO does this
        # by continuing past the timestep and DDRL by gating its anchor; DDPO had
        # neither, so its DPOK KL term would keep pulling the policy toward the
        # reference with no learning signal opposing it. This is reasoning about
        # the code path, not a diagnosis of an observed failure - the FLUX collapse
        # that motivated the DDRL gate turned out to happen with the RL term live
        # throughout, so no run here has been shown to reach this condition. It is
        # still wrong to apply a regulariser to a batch that cannot teach anything.
        # Less destructive than DDRL's case regardless, since a KL to the reference
        # pulls back toward the starting point rather than toward a different
        # distribution.
        # Magnitude, not spread. DDPO normalises against a per-prompt reward
        # history rather than within the batch, so it can legitimately produce
        # constant nonzero advantages - "all of these beat this prompt's average"
        # is a real signal and must not be skipped. A spread test would discard
        # it. For the group-normalised trainers spread is the right test, because
        # zero spread there implies all-zero.
        # Computed once here rather than inside the PPO loop: it does not depend
        # on the pass, and the metrics below read it, which would be a NameError
        # if ppo_epochs were 0.
        has_signal = (
            trajectory.advantages is not None
            and trajectory.advantages.abs().max().item() > 1e-6
        )

        for ppo_epoch in range(config.ppo_epochs):
            timestep_indices = list(range(num_train))
            random.shuffle(timestep_indices)

            # Reshuffle the mini-batches each PPO pass (see FlowGRPO trainer).
            perm = torch.randperm(batch_size, device=device)
            minibatches = [perm[s:s + mb_size] for s in range(0, batch_size, mb_size)]

            for sl in minibatches:
                self.optimizer.zero_grad()
                mb = trajectory[sl]

                if not has_signal:
                    continue

                for j in timestep_indices:
                    sigma = mb.timesteps[j]
                    sigma_next = mb.timesteps[j + 1]

                    if sigma_next.item() < 1e-6:
                        continue

                    latent_t = mb.latents[:, j]
                    next_latent_t = mb.next_latents[:, j]

                    step_noise_level = config.noise_level if j < num_steps - 1 else 0.0
                    # cache_enabled=False to match the sampling forward (see FlowGRPO)
                    # - otherwise the replay noise_pred drifts from sampling and the
                    # SDE log-prob biases the importance ratio below 1.0.
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, cache_enabled=False):
                        log_prob_new, prev_sample_mean = self._replay_step(
                            transformer=self.transformer,
                            latent_t=latent_t,
                            next_latent_t=next_latent_t,
                            sigma=sigma,
                            sigma_next=sigma_next,
                            prompt_embeds=mb.prompt_embeds,
                            pooled_embeds=mb.pooled_embeds,
                            guidance_scale=config.guidance_scale,
                            noise_level=step_noise_level,
                            negative_prompt_embeds=mb.negative_prompt_embeds,
                            negative_pooled_embeds=mb.negative_pooled_embeds,
                            img_ids=mb.img_ids,
                            txt_ids=mb.txt_ids,
                        )

                    log_prob_old = mb.log_probs[:, j]
                    ratio = torch.exp(log_prob_new.float() - log_prob_old.float())

                    advantages = mb.advantages
                    unclipped = -advantages * ratio
                    clipped = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                    rl_loss = torch.mean(torch.maximum(unclipped, clipped))

                    # Optional per-step KL (DPOK)
                    kl_loss = torch.tensor(0.0, device=device)
                    if config.kl_beta > 0:
                        with torch.no_grad():
                            self.transformer.disable_adapters()
                            try:
                                # cache_enabled=False, matching the policy forward above
                                # and FlowGRPO's identical block. Without it the two
                                # means come out under different cast regimes, so the
                                # DPOK KL has a nonzero floor that is numerical noise
                                # rather than policy drift - and it is divided by
                                # 2*sigma^2, which is small at low sigma, so a bf16
                                # difference is amplified. kl_beta defaults to 0, so no
                                # recorded result is affected.
                                with torch.autocast(device_type=device.type, dtype=autocast_dtype, cache_enabled=False):
                                    _, prev_sample_mean_ref = self._replay_step(
                                        transformer=self.transformer,
                                        latent_t=latent_t,
                                        next_latent_t=next_latent_t,
                                        sigma=sigma,
                                        sigma_next=sigma_next,
                                        prompt_embeds=mb.prompt_embeds,
                                        pooled_embeds=mb.pooled_embeds,
                                        guidance_scale=config.guidance_scale,
                                        noise_level=step_noise_level,
                                        negative_prompt_embeds=mb.negative_prompt_embeds,
                                        negative_pooled_embeds=mb.negative_pooled_embeds,
                                        img_ids=mb.img_ids,
                                        txt_ids=mb.txt_ids,
                                    )
                            finally:
                                self.transformer.enable_adapters()

                        dt = (sigma_next - sigma).float()
                        # Match the noise level the replay step used for the mean
                        # (step_noise_level, 0 on the final step) and the same
                        # bounded coefficient the sampler used, so the KL divides by
                        # the variance the policy actually sampled from.
                        std_dev_t = sde_diffusion_std(sigma.float(), sigma_next.float(), step_noise_level)
                        noise_std = std_dev_t * torch.sqrt((-dt).clamp(min=1e-12))

                        diff = (prev_sample_mean.float() - prev_sample_mean_ref.float()).pow(2)
                        kl_per_sample = diff.mean(dim=tuple(range(1, diff.ndim))) / (2.0 * noise_std.pow(2) + 1e-12)
                        kl_loss = kl_per_sample.mean()

                    loss = rl_loss + config.kl_beta * kl_loss
                    loss = loss / len(timestep_indices)
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
            # Magnitude, not spread - this mirrors DDPO's own guard, which
            # differs from the other trainers on purpose because a buffered
            # per-prompt baseline can legitimately produce constant nonzero
            # advantages.
            "advantage_absmax": trajectory.advantages.abs().max().item(),
            "has_signal": has_signal,
        }
