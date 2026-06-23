"""Base trainer for diffusion RL methods."""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from peft import LoraConfig

from diffusion_tinker.core.callbacks import TrainerCallback
from diffusion_tinker.core.stat_tracking import PerPromptStatTracker
from diffusion_tinker.core.trajectory import TrajectoryBatch
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
        reward_kwargs: dict | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ):
        self.config = config
        self.train_prompts = train_prompts or []
        self.global_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_kwargs = reward_kwargs or {}
        self.callbacks = callbacks or []

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self._setup_model(model)

        self.reward_fn = resolve_reward(
            reward_funcs, device=str(self.device), reward_weights=reward_weights, reward_mode=reward_mode
        )

        self.stat_tracker = PerPromptStatTracker(global_std=config.advantage_global_std)

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
        if self.config.model_type == "auto":
            self._model_type = "flux" if "flux" in model_id.lower() else "sd3"
        else:
            self._model_type = self.config.model_type

        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16

        print(f"Loading {model_id} (type={self._model_type})...")
        if self._model_type == "flux":
            from diffusers import FluxPipeline
            from diffusion_tinker.models.flux_patch import FluxModelConfig
            self.pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
            self.model_config = FluxModelConfig()
        else:
            from diffusers import StableDiffusion3Pipeline
            from diffusion_tinker.models.sd3_patch import SD3ModelConfig
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
            self.model_config = SD3ModelConfig()
        self.pipeline.to(self.device)

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler

        self.vae.eval()
        self.vae.requires_grad_(False)
        if self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.requires_grad_(False)
        if self.pipeline.text_encoder_2 is not None:
            self.pipeline.text_encoder_2.requires_grad_(False)
        if getattr(self.pipeline, "text_encoder_3", None) is not None:
            self.pipeline.text_encoder_3.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.model_config.lora_target_modules,
        )
        self.transformer.add_adapter(lora_config)
        print(f"LoRA applied: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")

        if self.config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        for p in self.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    @torch.no_grad()
    def _sample_trajectories(self, prompts: list[str]) -> TrajectoryBatch:
        self.transformer.eval()
        if self.config.offload_text_encoders:
            self._set_text_encoders_device(self.device)

        all_outputs = []
        expanded_prompts: list[str] = []

        for p in prompts:
            batch = [p] * self.config.num_samples_per_prompt
            expanded_prompts.extend(batch)

            if self._model_type == "flux":
                from diffusion_tinker.models.flux_patch import flux_sample_with_logprob
                output = flux_sample_with_logprob(
                    pipeline=self.pipeline,
                    prompts=batch,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    noise_level=self.config.noise_level,
                    height=self.config.resolution,
                    width=self.config.resolution,
                )
            else:
                from diffusion_tinker.models.sd3_patch import sd3_sample_with_logprob
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

        has_neg = all_outputs[0].negative_prompt_embeds is not None if hasattr(all_outputs[0], "negative_prompt_embeds") else False
        has_img_ids = hasattr(all_outputs[0], "img_ids")
        trajectory = TrajectoryBatch(
            latents=torch.cat([o.latents_trajectory for o in all_outputs], dim=0),
            next_latents=torch.cat([o.next_latents_trajectory for o in all_outputs], dim=0),
            log_probs=torch.cat([o.log_probs for o in all_outputs], dim=0),
            timesteps=all_outputs[0].timesteps,
            prompt_embeds=torch.cat([o.prompt_embeds for o in all_outputs], dim=0),
            pooled_embeds=torch.cat([o.pooled_embeds for o in all_outputs], dim=0),
            negative_prompt_embeds=torch.cat([o.negative_prompt_embeds for o in all_outputs], dim=0) if has_neg else None,
            negative_pooled_embeds=torch.cat([o.negative_pooled_embeds for o in all_outputs], dim=0) if has_neg else None,
            img_ids=torch.cat([o.img_ids for o in all_outputs], dim=0) if has_img_ids else None,
            txt_ids=all_outputs[0].txt_ids if has_img_ids else None,
            prompts=expanded_prompts,
            rewards=None,
            images=[img for o in all_outputs for img in o.images],
        )

        final_latents = trajectory.next_latents[:, -1] if trajectory.next_latents is not None else None
        metadata = dict(self.reward_kwargs)
        ctx = RewardContext(
            images=trajectory.images,
            prompts=expanded_prompts,
            device=self.device,
            metadata=metadata,
            latents=final_latents,
            epoch=self.global_step,
        )
        reward_output = self.reward_fn(ctx)
        trajectory.rewards = torch.nan_to_num(reward_output.scores, nan=0.0)

        if self.config.offload_text_encoders:
            self._set_text_encoders_device("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return trajectory

    def _replay_step(self, transformer, latent_t, next_latent_t, sigma, sigma_next,
                      prompt_embeds, pooled_embeds, guidance_scale, noise_level,
                      negative_prompt_embeds=None, negative_pooled_embeds=None,
                      img_ids=None, txt_ids=None):
        if self._model_type == "flux":
            from diffusion_tinker.models.flux_patch import flux_replay_step
            return flux_replay_step(
                transformer=transformer, latent_t=latent_t, next_latent_t=next_latent_t,
                sigma=sigma, sigma_next=sigma_next, prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds, img_ids=img_ids, txt_ids=txt_ids,
                guidance_scale=guidance_scale, noise_level=noise_level,
            )
        else:
            from diffusion_tinker.models.sd3_patch import sd3_replay_step
            return sd3_replay_step(
                transformer=transformer, latent_t=latent_t, next_latent_t=next_latent_t,
                sigma=sigma, sigma_next=sigma_next, prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds, guidance_scale=guidance_scale,
                noise_level=noise_level, negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_embeds=negative_pooled_embeds,
            )

    def _set_text_encoders_device(self, device):
        """Move the (frozen) text encoders to a device. Used to offload them to
        CPU during the gradient step when config.offload_text_encoders is set."""
        for name in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            enc = getattr(self.pipeline, name, None)
            if enc is not None:
                enc.to(device)

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

    def _fire_callbacks(self, method: str, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(self, **kwargs)

    def train(self):
        """Main training loop."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        if not self.train_prompts:
            raise ValueError("No training prompts provided. Pass train_prompts to the trainer.")

        print(f"Starting training: {self.config.num_epochs} epochs, {len(self.train_prompts)} prompts")
        self._fire_callbacks("on_train_begin")

        for epoch in range(self.config.num_epochs):
            self._fire_callbacks("on_epoch_begin", epoch=epoch)

            epoch_prompts = self.train_prompts.copy()
            random.shuffle(epoch_prompts)
            batch_prompts = epoch_prompts

            trajectory = self._sample_trajectories(batch_prompts)
            self._fire_callbacks("on_sample_end", epoch=epoch, trajectory=trajectory)

            trajectory = self._compute_advantages(trajectory)

            # use raw advantages for filter (DDRL's -exp(-x) maps 0 to -1)
            adv_for_filter = getattr(trajectory, "_raw_advantages", trajectory.advantages)
            nonzero_mask = adv_for_filter.abs() > 1e-8
            if nonzero_mask.sum() < 2:
                print(f"Epoch {epoch}: all advantages are zero, skipping")
                continue

            self.transformer.train()
            metrics = self._training_step(trajectory)
            self.global_step += 1

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

            self._fire_callbacks("on_epoch_end", epoch=epoch, metrics=metrics)

            if epoch > 0 and epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch)

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

            # Free this epoch's trajectory before the next rollout allocates a
            # new one. Otherwise the old trajectory (latents for every sample)
            # is still alive during the next epoch's sampling, doubling peak
            # memory - which OOMs a 24GB card on the second epoch.
            del trajectory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._save_checkpoint(self.config.num_epochs)
        self._fire_callbacks("on_train_end")
        print("Training complete.")

    def _save_checkpoint(self, epoch: int):
        save_path = Path(self.config.output_dir) / f"checkpoint-{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(str(save_path))
        print(f"Saved checkpoint to {save_path}")
        self._fire_callbacks("on_save", epoch=epoch, path=str(save_path))

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> float:
        """Generate eval images and compute reward stats. Returns mean reward."""
        self.transformer.eval()
        if self.config.offload_text_encoders:
            self._set_text_encoders_device(self.device)
        eval_prompts = self.train_prompts[:4]

        if self._model_type == "flux":
            from diffusion_tinker.models.flux_patch import flux_sample_with_logprob
            output = flux_sample_with_logprob(
                pipeline=self.pipeline,
                prompts=eval_prompts,
                num_inference_steps=self.config.num_eval_inference_steps,
                guidance_scale=self.config.guidance_scale,
                noise_level=0.0,
                height=self.config.resolution,
                width=self.config.resolution,
            )
        else:
            from diffusion_tinker.models.sd3_patch import sd3_sample_with_logprob
            output = sd3_sample_with_logprob(
                pipeline=self.pipeline,
                prompts=eval_prompts,
                num_inference_steps=self.config.num_eval_inference_steps,
                guidance_scale=self.config.guidance_scale,
                noise_level=0.0,
                height=self.config.resolution,
                width=self.config.resolution,
            )

        ctx = RewardContext(
            images=output.images, prompts=eval_prompts, device=self.device,
            metadata=dict(self.reward_kwargs),
        )
        reward_output = self.reward_fn(ctx)
        scores = reward_output.scores
        mean_reward = scores.mean().item()
        self._fire_callbacks("on_evaluate", epoch=epoch, mean_reward=mean_reward)

        per_prompt = " | ".join(f"{p[:20]}={s:.2f}" for p, s in zip(eval_prompts, scores.tolist()))
        print(f"Eval (epoch {epoch}): mean_reward={mean_reward:.3f} [{per_prompt}]")

        eval_dir = Path(self.config.output_dir) / f"eval-{epoch}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(output.images):
            img.save(eval_dir / f"sample_{i}.png")

        if mean_reward > self._best_eval_reward:
            self._best_eval_reward = mean_reward
            self._evals_without_improvement = 0
            if self.config.save_best:
                best_path = Path(self.config.output_dir) / "checkpoint-best"
                best_path.mkdir(parents=True, exist_ok=True)
                self.transformer.save_pretrained(str(best_path))
                print(f"New best eval reward: {mean_reward:.3f} (saved to {best_path})")
        else:
            self._evals_without_improvement += 1

        return mean_reward
