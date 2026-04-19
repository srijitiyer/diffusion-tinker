"""Base trainer for diffusion RL methods.

Orchestrates the lifecycle: model loading, LoRA setup, sampling, reward
computation, advantage estimation, and the training loop. Algorithm-specific
trainers override `_training_step()`.

Reference: TRAINING_LOOP_INTERNALS.md
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from peft import LoraConfig

from diffusion_tinker.core.stat_tracking import PerPromptStatTracker
from diffusion_tinker.core.trajectory import TrajectoryBatch
from diffusion_tinker.models.sd3_patch import SD3ModelConfig, sd3_sample_with_logprob
from diffusion_tinker.rewards.protocol import RewardContext, RewardFunc
from diffusion_tinker.rewards.resolve import resolve_reward
from diffusion_tinker.trainers.base_config import BaseDiffusionConfig


class BaseDiffusionTrainer(ABC):
    """Base class for all diffusion RL trainers.

    Subclasses must implement `_training_step()`.
    """

    def __init__(
        self,
        model: str,
        reward_funcs: RewardFunc,
        config: BaseDiffusionConfig,
        train_prompts: list[str] | None = None,
        reward_weights: list[float] | None = None,
        reward_mode: str = "weighted_sum",
    ):
        self.config = config
        self.train_prompts = train_prompts or []
        self.global_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seed
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        # Load model
        self._setup_model(model)

        # Set up reward (supports multi-reward: reward_funcs=["aesthetic", "clip_score"])
        reward_device = self.device
        self.reward_fn = resolve_reward(
            reward_funcs, device=str(reward_device), reward_weights=reward_weights, reward_mode=reward_mode
        )

        # Stat tracking for per-prompt advantage normalization
        self.stat_tracker = PerPromptStatTracker()

        # Optimizer - only LoRA params have requires_grad
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self._best_eval_reward = -float("inf")
        self._evals_without_improvement = 0

        print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
        print(f"Total params: {sum(p.numel() for p in self.transformer.parameters()):,}")

    def _setup_model(self, model_id: str):
        """Load pipeline, apply LoRA, freeze non-trainable components."""
        from diffusers import StableDiffusion3Pipeline

        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16

        print(f"Loading {model_id}...")
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipeline.to(self.device)

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler
        self.model_config = SD3ModelConfig()

        # Freeze VAE and text encoders
        self.vae.eval()
        self.vae.requires_grad_(False)
        if self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.requires_grad_(False)
        if self.pipeline.text_encoder_2 is not None:
            self.pipeline.text_encoder_2.requires_grad_(False)
        if self.pipeline.text_encoder_3 is not None:
            self.pipeline.text_encoder_3.requires_grad_(False)

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.model_config.lora_target_modules,
        )
        self.transformer.add_adapter(lora_config)
        print(f"LoRA applied: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")

        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        # Cast trainable params to float32 for stable training
        for p in self.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    @torch.no_grad()
    def _sample_trajectories(self, prompts: list[str]) -> TrajectoryBatch:
        """Generate images from current policy and collect trajectories + log-probs.

        Processes prompts in mini-batches to avoid OOM. Each prompt generates
        num_samples_per_prompt images, so batch size = num_samples_per_prompt
        per prompt group.
        """
        self.transformer.eval()

        all_outputs = []
        expanded_prompts: list[str] = []

        # Process one prompt at a time (each generates num_samples_per_prompt images)
        for p in prompts:
            batch = [p] * self.config.num_samples_per_prompt
            expanded_prompts.extend(batch)

            output = sd3_sample_with_logprob(
                pipeline=self.pipeline,
                prompts=batch,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                noise_level=self.config.noise_level,
                height=self.config.resolution,
                width=self.config.resolution,
            )
            all_outputs.append(output)

        # Concatenate all mini-batch outputs
        has_neg = all_outputs[0].negative_prompt_embeds is not None
        trajectory = TrajectoryBatch(
            latents=torch.cat([o.latents_trajectory for o in all_outputs], dim=0),
            next_latents=torch.cat([o.next_latents_trajectory for o in all_outputs], dim=0),
            log_probs=torch.cat([o.log_probs for o in all_outputs], dim=0),
            timesteps=all_outputs[0].timesteps,  # Same schedule for all batches
            prompt_embeds=torch.cat([o.prompt_embeds for o in all_outputs], dim=0),
            pooled_embeds=torch.cat([o.pooled_embeds for o in all_outputs], dim=0),
            negative_prompt_embeds=torch.cat([o.negative_prompt_embeds for o in all_outputs], dim=0) if has_neg else None,
            negative_pooled_embeds=torch.cat([o.negative_pooled_embeds for o in all_outputs], dim=0) if has_neg else None,
            prompts=expanded_prompts,
            rewards=None,
            images=[img for o in all_outputs for img in o.images],
        )

        # Compute rewards on all images at once (reward models are lightweight)
        ctx = RewardContext(images=trajectory.images, prompts=expanded_prompts, device=self.device)
        reward_output = self.reward_fn(ctx)
        trajectory.rewards = reward_output.scores

        return trajectory

    def _compute_advantages(self, trajectory: TrajectoryBatch) -> TrajectoryBatch:
        """Compute per-prompt normalized advantages."""
        advantages = self.stat_tracker.update(trajectory.prompts, trajectory.rewards)
        advantages = torch.clamp(advantages, -self.config.adv_clip_max, self.config.adv_clip_max)
        trajectory.advantages = advantages
        return trajectory

    @abstractmethod
    def _training_step(self, trajectory: TrajectoryBatch) -> dict[str, float]:
        """Algorithm-specific training step. Must be implemented by subclass."""
        raise NotImplementedError

    def train(self):
        """Main training loop."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        if not self.train_prompts:
            raise ValueError("No training prompts provided. Pass train_prompts to the trainer.")

        print(f"Starting training: {self.config.num_epochs} epochs, {len(self.train_prompts)} prompts")

        for epoch in range(self.config.num_epochs):
            # Shuffle prompts each epoch
            epoch_prompts = self.train_prompts.copy()
            random.shuffle(epoch_prompts)

            # Process all prompts each epoch. Each prompt generates num_samples_per_prompt images,
            # so total images = len(prompts) * num_samples_per_prompt.
            batch_prompts = epoch_prompts

            # 1. Sample trajectories (no_grad)
            trajectory = self._sample_trajectories(batch_prompts)

            # 2. Compute advantages
            trajectory = self._compute_advantages(trajectory)

            # 3. Filter zero-advantage samples. Use raw (pre-transform) advantages
            # when available - DDRL's monotonic transform maps 0 to -1, so every
            # post-transform advantage is nonzero even in degenerate cases.
            adv_for_filter = getattr(trajectory, "_raw_advantages", trajectory.advantages)
            nonzero_mask = adv_for_filter.abs() > 1e-8
            if nonzero_mask.sum() < 2:
                print(f"Epoch {epoch}: all advantages are zero, skipping")
                continue

            # 4. Training step (algorithm-specific)
            self.transformer.train()
            metrics = self._training_step(trajectory)
            self.global_step += 1

            # 5. Log
            if epoch % self.config.log_every == 0:
                mean_reward = trajectory.rewards.mean().item()
                log_str = f"Epoch {epoch} | reward={mean_reward:.3f}"
                for k, v in metrics.items():
                    log_str += f" | {k}={v:.4f}"

                unique_prompts = list(dict.fromkeys(trajectory.prompts))
                per_prompt_rewards = []
                for p in unique_prompts:
                    idxs = [i for i, tp in enumerate(trajectory.prompts) if tp == p]
                    pr = trajectory.rewards[idxs].mean().item()
                    per_prompt_rewards.append(f"{p[:15]}={pr:.2f}")
                log_str += f" | per_prompt=[{', '.join(per_prompt_rewards)}]"
                print(log_str)

            # 6. Save checkpoint
            if epoch > 0 and epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch)

            # 7. Eval
            if epoch > 0 and epoch % self.config.eval_every == 0:
                self._evaluate(epoch)

                if (
                    self.config.early_stop_patience > 0
                    and self._evals_without_improvement >= self.config.early_stop_patience
                ):
                    print(
                        f"Early stopping: no eval improvement for {self.config.early_stop_patience} evals "
                        f"(best={self._best_eval_reward:.3f})"
                    )
                    break

        # Final save
        self._save_checkpoint(self.config.num_epochs)
        print("Training complete.")

    def _save_checkpoint(self, epoch: int):
        """Save LoRA adapter weights."""
        save_path = Path(self.config.output_dir) / f"checkpoint-{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(str(save_path))
        print(f"Saved checkpoint to {save_path}")

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> float:
        """Generate eval images and compute reward stats. Returns mean reward."""
        self.transformer.eval()
        eval_prompts = self.train_prompts[:4]

        output = sd3_sample_with_logprob(
            pipeline=self.pipeline,
            prompts=eval_prompts,
            num_inference_steps=self.config.num_eval_inference_steps,
            guidance_scale=self.config.guidance_scale,
            noise_level=0.0,
            height=self.config.resolution,
            width=self.config.resolution,
        )

        ctx = RewardContext(images=output.images, prompts=eval_prompts, device=self.device)
        reward_output = self.reward_fn(ctx)
        scores = reward_output.scores
        mean_reward = scores.mean().item()

        per_prompt = " | ".join(f"{p[:20]}={s:.2f}" for p, s in zip(eval_prompts, scores.tolist()))
        print(f"Eval (epoch {epoch}): mean_reward={mean_reward:.3f} [{per_prompt}]")

        eval_dir = Path(self.config.output_dir) / f"eval-{epoch}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(output.images):
            img.save(eval_dir / f"sample_{i}.png")

        if self.config.save_best and mean_reward > self._best_eval_reward:
            self._best_eval_reward = mean_reward
            self._evals_without_improvement = 0
            best_path = Path(self.config.output_dir) / "checkpoint-best"
            best_path.mkdir(parents=True, exist_ok=True)
            self.transformer.save_pretrained(str(best_path))
            print(f"New best eval reward: {mean_reward:.3f} (saved to {best_path})")
        else:
            self._evals_without_improvement += 1

        return mean_reward
