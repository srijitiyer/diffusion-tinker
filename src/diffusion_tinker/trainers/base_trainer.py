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

        reward_device = getattr(config, "reward_device", None) or str(self.device)
        self.reward_fn = resolve_reward(
            reward_funcs, device=reward_device, reward_weights=reward_weights, reward_mode=reward_mode
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

    def _save_adapter(self, path):
        """Write just the LoRA adapter, not the whole transformer.

        The model is a base transformer with an adapter injected via
        `add_adapter`, so `save_pretrained` serialises every base weight. On SD3.5
        that is ~5GB per save; on FLUX it is ~24GB, and against this cluster's
        storage a single "new best" write took longer than the training it was
        checkpointing. Only the adapter is trainable, so only the adapter is worth
        writing - about 26MB.
        """
        from peft.utils import get_peft_model_state_dict

        path.mkdir(parents=True, exist_ok=True)
        try:
            state = get_peft_model_state_dict(self.transformer)
            torch.save(state, path / "adapter_model.bin")
        except Exception:
            # if the adapter cannot be isolated, fall back rather than lose the
            # checkpoint entirely
            self.transformer.save_pretrained(str(path))

    def _fire_callbacks(self, method: str, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(self, **kwargs)

    def train(self):
        """Main training loop."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        if not self.train_prompts:
            raise ValueError("No training prompts provided. Pass train_prompts to the trainer.")

        start_epoch = self._load_state()
        print(f"Starting training: {self.config.num_epochs} epochs, {len(self.train_prompts)} prompts")
        self._fire_callbacks("on_train_begin")

        for epoch in range(start_epoch, self.config.num_epochs):
            self._fire_callbacks("on_epoch_begin", epoch=epoch)

            # Evaluate before the epoch's update, not after, so "Eval (epoch 0)"
            # is the untrained model. Evaluating after meant the first point was
            # already one update in, which is not a baseline and made two runs
            # that differ only in beta disagree at epoch 0.
            if epoch % self.config.eval_every == 0:
                self._evaluate(epoch)
                if (
                    self.config.early_stop_patience
                    and self._evals_without_improvement >= self.config.early_stop_patience
                ):
                    print(
                        f"Early stopping: no eval improvement for {self.config.early_stop_patience} evals "
                        f"(best={self._best_eval_reward:.3f})"
                    )
                    break

            epoch_prompts = self.train_prompts.copy()
            random.shuffle(epoch_prompts)
            ppe = self.config.prompts_per_epoch
            if ppe and ppe < len(epoch_prompts):
                # fresh random subset each epoch; over many epochs the whole pool
                # is covered, giving diversity without a huge per-epoch rollout
                batch_prompts = epoch_prompts[:ppe]
            else:
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

            # Free this epoch's trajectory before the next rollout allocates a
            # new one. Otherwise the old trajectory (latents for every sample)
            # is still alive during the next epoch's sampling, doubling peak
            # memory - which OOMs a 24GB card on the second epoch.
            del trajectory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.config.resumable:
                self._save_state(epoch + 1)

        self._save_checkpoint(self.config.num_epochs)
        # The run finished, so the state is spent. Left behind, the next start on
        # this tag would resume at num_epochs, fall straight out of the loop, and
        # report success having trained nothing.
        state_path = self._state_path()
        if state_path.exists():
            state_path.unlink()
        self._fire_callbacks("on_train_end")
        print("Training complete.")

    def _state_path(self) -> Path:
        return Path(self.config.output_dir) / "resume_state.pt"

    def _state_fingerprint(self) -> str:
        """Identity of the experiment this state belongs to.

        Resume keys off output_dir, and output_dir is keyed off a run tag. Reusing
        a tag with different settings would otherwise load one experiment's
        adapter into another's run and report the result as the new arm - silent,
        and fatal to a comparison. Only fields that change what is being measured
        are included; things like eval_every or output paths are not.
        """
        c = self.config
        parts = [
            type(c).__name__,
            f"lr={c.learning_rate}", f"rank={c.lora_rank}", f"seed={c.seed}",
            f"group={c.num_samples_per_prompt}", f"steps={c.num_inference_steps}",
            f"noise={c.noise_level}", f"res={c.resolution}", f"clip={c.clip_range}",
            f"tbs={c.train_batch_size}", f"ppe={c.prompts_per_epoch}",
            f"guid={c.guidance_scale}",
            f"beta={getattr(c, 'data_beta', None)}/{getattr(c, 'kl_beta', None)}",
            f"mono={getattr(c, 'use_monotonic_transform', None)}",
        ]
        return "|".join(parts)

    def _save_state(self, next_epoch: int):
        """Everything needed to continue as if the crash had not happened.

        Written to a temp file and renamed, because a crash during the write
        itself would otherwise leave a truncated state that poisons the restart.
        """
        from peft.utils import get_peft_model_state_dict

        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "adapter": get_peft_model_state_dict(self.transformer),
                "optimizer": self.optimizer.state_dict(),
                "next_epoch": next_epoch,
                "global_step": self.global_step,
                "best_eval_reward": self._best_eval_reward,
                "evals_without_improvement": self._evals_without_improvement,
                "stat_tracker": getattr(self.stat_tracker, "stats", None),
                "fingerprint": self._state_fingerprint(),
            },
            tmp,
        )
        os.replace(tmp, path)

    def _load_state(self) -> int:
        """Restore a saved state and return the epoch to start from."""
        from peft.utils import set_peft_model_state_dict

        path = self._state_path()
        if not self.config.resumable or not path.exists():
            return 0
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            print(f"Resume state at {path} is unreadable ({exc}); starting fresh")
            return 0

        saved_fp = state.get("fingerprint")
        now_fp = self._state_fingerprint()
        if saved_fp is not None and saved_fp != now_fp:
            print(
                f"Refusing to resume: {path} was written by a different configuration.\n"
                f"  saved: {saved_fp}\n  now:   {now_fp}\n"
                "Starting fresh. Delete the state file if the reuse is intentional.",
                flush=True,
            )
            return 0

        set_peft_model_state_dict(self.transformer, state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        self._best_eval_reward = state.get("best_eval_reward", -float("inf"))
        self._evals_without_improvement = state.get("evals_without_improvement", 0)
        tracked = state.get("stat_tracker")
        if tracked is not None and hasattr(self.stat_tracker, "stats"):
            self.stat_tracker.stats = tracked
        start = int(state.get("next_epoch", 0))
        print(f"Resuming from {path} at epoch {start} (best so far {self._best_eval_reward:.3f})", flush=True)
        return start

    def _save_checkpoint(self, epoch: int):
        save_path = Path(self.config.output_dir) / f"checkpoint-{epoch}"
        self._save_adapter(save_path)
        print(f"Saved checkpoint to {save_path}")
        self._fire_callbacks("on_save", epoch=epoch, path=str(save_path))

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> float:
        """Generate eval images and compute reward stats. Returns mean reward."""
        self.transformer.eval()
        if self.config.offload_text_encoders:
            self._set_text_encoders_device(self.device)
        base_prompts = self.config.eval_prompts or self.train_prompts[: self.config.num_eval_prompts]
        # Repeat each prompt so the mean is over more than one image per prompt.
        eval_prompts = [p for p in base_prompts for _ in range(self.config.num_eval_samples_per_prompt)]

        # Fresh generator each eval, seeded identically, so epoch N and epoch M
        # start from the same latents and the curve reflects the policy only.
        eval_gen = torch.Generator(device=self.device).manual_seed(self.config.eval_seed)

        if self._model_type == "flux":
            from diffusion_tinker.models.flux_patch import flux_sample_with_logprob as sample_fn
        else:
            from diffusion_tinker.models.sd3_patch import sd3_sample_with_logprob as sample_fn

        # A held-out eval set is far bigger than a rollout batch, so sample it in
        # chunks rather than one call that would OOM on the VAE decode.
        chunk = self.config.train_batch_size or len(eval_prompts)
        eval_images = []
        try:
            for i in range(0, len(eval_prompts), chunk):
                out = sample_fn(
                    pipeline=self.pipeline,
                    prompts=eval_prompts[i : i + chunk],
                    num_inference_steps=self.config.num_eval_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    noise_level=0.0,
                    height=self.config.resolution,
                    width=self.config.resolution,
                    generator=eval_gen,
                )
                eval_images.extend(out.images)
        finally:
            # Put the text encoders back on CPU. Eval pulls them onto the GPU to
            # encode its prompts and used to leave them there, so T5-XXL's ~9GB
            # stayed resident for the rest of training. DRaFT is the case that
            # breaks first: it offloads them on purpose after caching embeddings,
            # precisely to free room for the reward backward, and a single eval
            # silently undid that and OOMed the next step.
            if self.config.offload_text_encoders:
                self._set_text_encoders_device("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        ctx = RewardContext(
            images=eval_images, prompts=eval_prompts, device=self.device,
            metadata=dict(self.reward_kwargs),
        )
        reward_output = self.reward_fn(ctx)
        scores = reward_output.scores
        mean_reward = scores.mean().item()
        self._fire_callbacks("on_evaluate", epoch=epoch, mean_reward=mean_reward)

        sem = scores.std().item() / max(len(scores) ** 0.5, 1.0)
        print(f"Eval (epoch {epoch}): mean_reward={mean_reward:.3f} +/- {sem:.3f} (n={len(scores)})")

        eval_dir = Path(self.config.output_dir) / f"eval-{epoch}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(eval_images[:8]):
            img.save(eval_dir / f"sample_{i}.png")

        if mean_reward > self._best_eval_reward:
            self._best_eval_reward = mean_reward
            self._evals_without_improvement = 0
            if self.config.save_best:
                best_path = Path(self.config.output_dir) / "checkpoint-best"
                self._save_adapter(best_path)
                print(f"New best eval reward: {mean_reward:.3f} (saved to {best_path})")
        else:
            self._evals_without_improvement += 1

        return mean_reward
