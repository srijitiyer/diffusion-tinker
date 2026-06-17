"""FlowGRPO Trainer (arXiv:2505.05470)."""

from __future__ import annotations

import random

import torch

from diffusion_tinker.core.trajectory import TrajectoryBatch
from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer
from diffusion_tinker.trainers.flowgrpo_config import FlowGRPOConfig


class FlowGRPOTrainer(BaseDiffusionTrainer):

    config: FlowGRPOConfig

    def _training_step(self, trajectory: TrajectoryBatch) -> dict[str, float]:
        device = self.device
        config = self.config

        trajectory = trajectory.to(device)
        has_signal = trajectory.advantages is not None and trajectory.advantages.std().item() > 1e-6
        num_steps = trajectory.log_probs.shape[1]

        # Train on the first timestep_fraction of steps (drop the low-sigma tail
        # where the log-prob / ratio is ill-conditioned, like FlowGRPO).
        num_train = max(1, int(num_steps * config.timestep_fraction))
        timestep_indices = list(range(num_train))
        if config.num_train_timesteps is not None and config.num_train_timesteps < len(timestep_indices):
            timestep_indices = sorted(random.sample(timestep_indices, config.num_train_timesteps))
        random.shuffle(timestep_indices)

        total_rl_loss = 0.0
        total_kl_loss = 0.0
        total_ratio = 0.0
        num_computed_steps = 0

        self.optimizer.zero_grad()
        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        for j in timestep_indices:
            sigma = trajectory.timesteps[j]
            sigma_next = trajectory.timesteps[j + 1]

            if sigma_next.item() < 1e-6:
                continue

            latent_t = trajectory.latents[:, j]
            next_latent_t = trajectory.next_latents[:, j]

            if not has_signal:
                continue

            step_noise_level = config.noise_level if j < num_steps - 1 else 0.0
            # cache_enabled=False to match the sampling forward exactly. With
            # the default cache on, the replay forward's noise_pred differs
            # slightly from sampling's, and the SDE log-prob amplifies that into
            # an importance ratio biased below 1.0 (badly at low sigma).
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, cache_enabled=False):
                log_prob_new, prev_sample_mean = self._replay_step(
                    transformer=self.transformer,
                    latent_t=latent_t,
                    next_latent_t=next_latent_t,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    prompt_embeds=trajectory.prompt_embeds,
                    pooled_embeds=trajectory.pooled_embeds,
                    guidance_scale=config.guidance_scale,
                    noise_level=step_noise_level,
                    negative_prompt_embeds=trajectory.negative_prompt_embeds,
                    negative_pooled_embeds=trajectory.negative_pooled_embeds,
                    img_ids=trajectory.img_ids,
                    txt_ids=trajectory.txt_ids,
                )

            log_prob_old = trajectory.log_probs[:, j]
            ratio = torch.exp(log_prob_new.float() - log_prob_old.float())

            if config.use_grpo_guard:
                # The guard renormalizes the batch-mean ratio to 1; the
                # normalizer must be a stop-grad scalar, otherwise gradients
                # leak through ratio.mean() and change the PPO objective.
                ratio = ratio / (ratio.mean().detach() + 1e-8)

            advantages = trajectory.advantages
            unclipped = -advantages * ratio
            clipped = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
            rl_loss = torch.mean(torch.maximum(unclipped, clipped))

            kl_loss = torch.tensor(0.0, device=device)
            if config.kl_beta > 0:
                # reference model = base model with LoRA disabled
                with torch.no_grad():
                    self.transformer.disable_adapters()
                    try:
                        with torch.autocast(device_type=device.type, dtype=autocast_dtype, cache_enabled=False):
                            _, prev_sample_mean_ref = self._replay_step(
                                transformer=self.transformer,
                                latent_t=latent_t,
                                next_latent_t=next_latent_t,
                                sigma=sigma,
                                sigma_next=sigma_next,
                                prompt_embeds=trajectory.prompt_embeds,
                                pooled_embeds=trajectory.pooled_embeds,
                                guidance_scale=config.guidance_scale,
                                noise_level=step_noise_level,
                                negative_prompt_embeds=trajectory.negative_prompt_embeds,
                                negative_pooled_embeds=trajectory.negative_pooled_embeds,
                                img_ids=trajectory.img_ids,
                                txt_ids=trajectory.txt_ids,
                            )
                    finally:
                        self.transformer.enable_adapters()

                sigma_val = sigma.float().clamp(max=0.9999)
                dt = (sigma_next - sigma).float()
                # Use the same noise level the replay step used for the mean
                # (step_noise_level, which is 0 on the final step) so the KL
                # variance is consistent with the policy/reference means.
                std_dev_t = torch.sqrt(sigma_val / (1.0 - sigma_val)) * step_noise_level
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
            num_computed_steps += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.transformer.parameters() if p.requires_grad],
            config.max_grad_norm,
        )
        self.optimizer.step()

        n = max(num_computed_steps, 1)
        return {
            "rl_loss": total_rl_loss / n,
            "kl_loss": total_kl_loss / n,
            "mean_ratio": total_ratio / n,
            "mean_reward": trajectory.rewards.mean().item(),
            "mean_advantage": trajectory.advantages.mean().item(),
        }
