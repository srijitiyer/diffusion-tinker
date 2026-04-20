"""DDRL Trainer (arXiv:2512.04332)."""

from __future__ import annotations

import os
import random
import warnings

import torch
import torchvision.transforms as T

from diffusion_tinker.core.latent_utils import encode_to_latents
from diffusion_tinker.core.noise_strategy import compute_flow_matching_loss
from diffusion_tinker.core.trajectory import TrajectoryBatch
from diffusion_tinker.models.sd3_patch import sd3_replay_step
from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer
from diffusion_tinker.trainers.ddrl_config import DDRLConfig


class DDRLTrainer(BaseDiffusionTrainer):

    config: DDRLConfig

    def __init__(self, model, reward_funcs, config, train_prompts=None, reward_weights=None, reward_mode="weighted_sum"):
        super().__init__(
            model=model, reward_funcs=reward_funcs, config=config,
            train_prompts=train_prompts, reward_weights=reward_weights, reward_mode=reward_mode,
        )
        self._data_latents = None
        self._setup_data()

    def _setup_data(self):
        if self.config.train_dataset is None:
            if self.config.data_beta > 0:
                warnings.warn(
                    "data_beta > 0 but no train_dataset provided. "
                    "Disabling data loss term - using trajectory endpoints creates a "
                    "self-distillation feedback loop that reinforces the current (bad) policy. "
                    "For proper DDRL, set train_dataset to a HF dataset or image folder.",
                    stacklevel=2,
                )
                self.config.data_beta = 0.0
            return

        print(f"Loading data for DDRL forward KL: {self.config.train_dataset}...")
        from datasets import load_dataset

        # Load dataset
        if os.path.isdir(self.config.train_dataset):
            ds = load_dataset("imagefolder", data_dir=self.config.train_dataset, split=self.config.dataset_split)
        else:
            ds = load_dataset(self.config.train_dataset, split=self.config.dataset_split)

        if self.config.image_column not in ds.column_names:
            raise ValueError(
                f"Dataset has no '{self.config.image_column}' column. "
                f"Available: {ds.column_names}. Set image_column in config."
            )

        # Pre-encode images to latents
        transform = T.Compose(
            [
                T.Resize(self.config.resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.config.resolution),
                T.ToTensor(),
            ]
        )

        all_latents = []
        batch_size = 8
        max_images = min(len(ds), 5000)  # Cap at 5K images to limit memory

        print(f"  Encoding {max_images} images to latents...")
        with torch.no_grad():
            for start in range(0, max_images, batch_size):
                end = min(start + batch_size, max_images)
                images = []
                for i in range(start, end):
                    img = ds[i][self.config.image_column]
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    images.append(transform(img))

                batch = torch.stack(images).to(self.device, dtype=self.vae.dtype)
                latents = encode_to_latents(self.vae, batch)
                all_latents.append(latents.cpu().half())  # Store as fp16 to save memory

        self._data_latents = torch.cat(all_latents, dim=0)
        print(f"  Cached {len(self._data_latents)} latents ({self._data_latents.nbytes / 1e6:.0f} MB)")

    def _compute_advantages(self, trajectory: TrajectoryBatch) -> TrajectoryBatch:
        raw_advantages = self.stat_tracker.update(trajectory.prompts, trajectory.rewards)

        # raw advantages needed for the zero-signal filter in base_trainer
        trajectory._raw_advantages = raw_advantages

        advantages = raw_advantages
        if self.config.beta_temp != 1.0:
            advantages = advantages / self.config.beta_temp

        if self.config.use_monotonic_transform:
            advantages = -torch.exp(-advantages)

        advantages = torch.clamp(advantages, -self.config.adv_clip_max, self.config.adv_clip_max)
        trajectory.advantages = advantages
        return trajectory

    def _training_step(self, trajectory: TrajectoryBatch) -> dict[str, float]:
        device = self.device
        config = self.config
        batch_size = len(trajectory)

        trajectory = trajectory.to(device)

        has_signal = trajectory.advantages is not None and trajectory.advantages.std().item() > 1e-6

        num_steps = trajectory.log_probs.shape[1]

        total_rl_loss = 0.0
        total_data_loss = 0.0
        total_ratio = 0.0
        num_computed_steps = 0

        timestep_indices = list(range(num_steps))
        random.shuffle(timestep_indices)

        self.optimizer.zero_grad()

        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        for j in timestep_indices:
            sigma = trajectory.timesteps[j]
            sigma_next = trajectory.timesteps[j + 1]

            if sigma_next.item() < 1e-6:
                continue

            latent_t = trajectory.latents[:, j]
            next_latent_t = trajectory.next_latents[:, j]

            if has_signal:
                step_noise_level = config.noise_level if j < num_steps - 1 else 0.0
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
                        noise_level=step_noise_level,
                        negative_prompt_embeds=trajectory.negative_prompt_embeds,
                        negative_pooled_embeds=trajectory.negative_pooled_embeds,
                    )

                log_prob_old = trajectory.log_probs[:, j]
                ratio = torch.exp(log_prob_new.float() - log_prob_old.float())

                advantages = trajectory.advantages
                unclipped = -advantages * ratio
                clipped = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                rl_loss = torch.mean(torch.maximum(unclipped, clipped))
            else:
                rl_loss = torch.tensor(0.0, device=device)
                ratio = torch.tensor(1.0, device=device)

            data_loss = torch.tensor(0.0, device=device)
            if config.data_beta > 0:
                u = torch.normal(mean=config.logit_mean, std=config.logit_std, size=(batch_size,), device=device)
                t = torch.sigmoid(u)

                assert self._data_latents is not None, "data_beta > 0 requires train_dataset"
                data_idx = torch.randint(0, len(self._data_latents), (batch_size,))
                clean_latents = self._data_latents[data_idx].to(device, dtype=autocast_dtype)

                noise = torch.randn_like(clean_latents)
                # full condition dropout: clean latents are from the dataset, not related
                # to the RL prompts, so conditioning on RL prompts is semantically wrong
                dropout_mask = torch.ones(batch_size, device=device, dtype=torch.bool)

                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    data_loss = compute_flow_matching_loss(
                        transformer=self.transformer,
                        latents=clean_latents,
                        noise=noise,
                        sigmas=t,
                        prompt_embeds=trajectory.prompt_embeds,
                        pooled_embeds=trajectory.pooled_embeds,
                        condition_dropout_mask=dropout_mask,
                    )

            loss = rl_loss + config.data_beta * data_loss
            loss = loss / len(timestep_indices)
            if loss.requires_grad:
                loss.backward()

            total_rl_loss += rl_loss.item()
            total_data_loss += data_loss.item() if isinstance(data_loss, torch.Tensor) else data_loss
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
            "data_loss": total_data_loss / n,
            "total_loss": (total_rl_loss + config.data_beta * total_data_loss) / n,
            "mean_ratio": total_ratio / n,
            "mean_reward": trajectory.rewards.mean().item(),
            "mean_advantage": trajectory.advantages.mean().item(),
        }
