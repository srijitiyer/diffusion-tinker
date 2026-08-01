"""DRaFT Trainer (arXiv:2309.17400)."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import torch
from PIL import Image as PILImage

from diffusion_tinker.core.callbacks import TrainerCallback
from diffusion_tinker.rewards.protocol import RewardContext, RewardFunc
from diffusion_tinker.rewards.resolve import resolve_reward
from diffusion_tinker.trainers.draft_config import DRaFTConfig


class DRaFTTrainer:
    """Trainer implementing DRaFT (Direct Reward Fine-Tuning)."""

    def __init__(
        self,
        model: str,
        reward_funcs: RewardFunc,
        config: DRaFTConfig,
        train_prompts: list[str] | None = None,
        reward_weights: list[float] | None = None,
        reward_mode: str = "weighted_sum",
        reward_kwargs: dict | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ):
        self.config = config
        self.train_prompts = train_prompts or []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.reward_kwargs = reward_kwargs or {}
        self.callbacks = callbacks or []

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self._setup_model(model)
        self.reward_fn = resolve_reward(
            reward_funcs, device=str(self.device), reward_weights=reward_weights, reward_mode=reward_mode
        )

        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

        self._embed_cache = {}
        self._precompute_prompt_embeds()

    def _setup_model(self, model_id: str):
        from diffusion_tinker.models.model_setup import setup_sd3_model
        self.pipeline, self.transformer, self.vae, self.model_config = setup_sd3_model(
            model_id, self.config, self.device
        )
        # DRaFT backprops through the denoising chain; gradient checkpointing
        # recomputes the forward and the SD3 joint-attention blocks don't replay
        # consistently, so it must stay off regardless of the config default.
        self.transformer.disable_gradient_checkpointing()

    def _precompute_prompt_embeds(self):
        """Encode all training prompts once, then offload the text encoders.

        SD3's text encoders (T5-XXL especially) are ~10GB resident. DRaFT trains
        on a fixed prompt set and re-samples noise each step, so the embeddings
        never change. Caching them and moving the encoders to CPU frees the VRAM
        needed for the reward backward pass, which is what lets DRaFT fit on a
        24GB consumer card.
        """
        if not self.train_prompts:
            return
        uniq = list(dict.fromkeys(self.train_prompts))
        # Encode in chunks. T5-XXL's attention is quadratic in sequence length and
        # linear in batch, so encoding the whole prompt set at once is the peak
        # allocation of the entire run, and it happens while the transformer is
        # already resident. On a 16GB card that alone OOMs.
        chunk = 4
        with torch.no_grad():
            for start in range(0, len(uniq), chunk):
                batch = uniq[start : start + chunk]
                prompt_embeds, _, pooled_embeds, _ = self.pipeline.encode_prompt(
                    prompt=batch,
                    prompt_2=None,
                    prompt_3=None,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    device=self.device,
                )
                for i, p in enumerate(batch):
                    self._embed_cache[p] = (
                        prompt_embeds[i].detach().cpu(),
                        pooled_embeds[i].detach().cpu(),
                    )
                del prompt_embeds, pooled_embeds

        for name in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            enc = getattr(self.pipeline, name, None)
            if enc is not None:
                enc.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _denoise_with_grad(self, prompts: list[str]) -> tuple[torch.Tensor, list]:
        """Run denoising with gradients through the last K steps."""
        device = self.device
        dtype = self.pipeline.transformer.dtype
        config = self.config
        batch_size = len(prompts)
        K = config.truncation_steps

        missing = [p for p in prompts if p not in self._embed_cache]
        if missing:
            with torch.no_grad():
                pe, _, ppe, _ = self.pipeline.encode_prompt(
                    prompt=missing,
                    prompt_2=None,
                    prompt_3=None,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    device=device,
                )
            for i, p in enumerate(missing):
                self._embed_cache[p] = (pe[i].detach().cpu(), ppe[i].detach().cpu())

        prompt_embeds = torch.stack([self._embed_cache[p][0] for p in prompts]).to(device, dtype)
        pooled_prompt_embeds = torch.stack([self._embed_cache[p][1] for p in prompts]).to(device, dtype)

        latent_channels = self.pipeline.transformer.config.in_channels
        latents = torch.randn(
            (batch_size, latent_channels, config.resolution // 8, config.resolution // 8),
            dtype=dtype,
            device=device,
        )

        self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=device)
        sigmas = self.pipeline.scheduler.sigmas
        latents = latents * sigmas[0]

        num_steps = len(sigmas) - 1
        grad_start = max(0, num_steps - K)

        self.transformer.eval()
        with torch.no_grad():
            for i in range(grad_start):
                sigma = sigmas[i].expand(batch_size)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=sigma * 1000.0,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                dt = sigmas[i + 1] - sigmas[i]
                latents = latents + noise_pred * dt

        self.transformer.train()
        latents = latents.detach().requires_grad_(True)

        for i in range(grad_start, num_steps):
            sigma = sigmas[i].expand(batch_size)
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=sigma * 1000.0,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            dt = sigmas[i + 1] - sigmas[i]
            latents = latents + noise_pred * dt

        latents_decoded = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        # VAE decode is [-1, 1]; rescale to [0, 1] (the range the reward expects)
        # before clamping, else ~55% of pixels (everything <0) crush to black.
        images_tensor = (self.vae.decode(latents_decoded, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)

        with torch.no_grad():
            images_np = (images_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images_np = images_np.transpose(0, 2, 3, 1)
            pil_images = [PILImage.fromarray(img) for img in images_np]

        return images_tensor, pil_images

    def _training_step(self, prompts: list[str], num_accum_steps: int = 1) -> dict[str, float]:
        config = self.config
        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        # cache_enabled=False is required: _denoise_with_grad runs a no_grad
        # warmup loop before the gradient-tracked steps, both inside this
        # autocast. With caching on, autocast stores the detached bf16 cast of
        # each float32 LoRA weight during the warmup and reuses it in the grad
        # loop, so gradients never reach the LoRA params and the model silently
        # never trains. Disabling the cache forces a grad-tracked re-cast.
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, cache_enabled=False):
            images_tensor, pil_images = self._denoise_with_grad(prompts)

        rewards = self._differentiable_reward(images_tensor, prompts)

        loss = -rewards.mean() / num_accum_steps
        if loss.requires_grad:
            loss.backward()

        return {
            "loss": loss.item() * num_accum_steps,
            "mean_reward": rewards.detach().mean().item(),
        }

    def _differentiable_reward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Compute reward on image tensor with gradient flow preserved."""
        scores = self.reward_fn.differentiable_forward(images, prompts)
        if scores is not None:
            return scores

        import warnings

        warnings.warn(
            "Reward does not support differentiable_forward(). "
            "Gradients will NOT flow through the reward. "
            "Use aesthetic or clip_score for DRaFT, or implement "
            "differentiable_forward() on your custom reward.",
            stacklevel=2,
        )
        ctx = RewardContext(
            images=[
                PILImage.fromarray(
                    (images[i].detach().cpu() * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                )
                for i in range(images.shape[0])
            ],
            prompts=prompts,
            device=self.device,
        )
        return self.reward_fn._compute(ctx).scores.to(self.device)

    def _evaluate(self, epoch: int) -> float:
        """Held-out eval matching the other trainers.

        DRaFTTrainer does not inherit BaseDiffusionTrainer, so it had no eval at
        all and reported only the training reward, which is the quantity it
        directly optimizes on the prompts it optimizes. Same held-out set, same
        fixed generator and same chunking as the base implementation, so the
        curves are comparable.
        """
        from diffusion_tinker.models.sd3_patch import sd3_sample_with_logprob

        config = self.config
        base_prompts = config.eval_prompts or self.train_prompts[: config.num_eval_prompts]
        eval_prompts = [p for p in base_prompts for _ in range(config.num_eval_samples_per_prompt)]

        self.transformer.eval()
        # sampling encodes the prompts, so the encoders have to come back for it
        for name in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            enc = getattr(self.pipeline, name, None)
            if enc is not None:
                enc.to(self.device)

        gen = torch.Generator(device=self.device).manual_seed(config.eval_seed)
        chunk = max(1, config.gradient_accumulation_steps)
        images = []
        try:
            with torch.no_grad():
                for i in range(0, len(eval_prompts), chunk):
                    out = sd3_sample_with_logprob(
                        pipeline=self.pipeline,
                        prompts=eval_prompts[i : i + chunk],
                        num_inference_steps=config.num_eval_inference_steps,
                        guidance_scale=config.guidance_scale,
                        noise_level=0.0,
                        height=config.resolution,
                        width=config.resolution,
                        generator=gen,
                    )
                    images.extend(out.images)
        finally:
            # back to CPU: DRaFT offloads them on purpose to leave room for the
            # reward backward, and leaving them resident OOMs the next step
            for name in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
                enc = getattr(self.pipeline, name, None)
                if enc is not None:
                    enc.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ctx = RewardContext(images=images, prompts=eval_prompts, device=self.device,
                            metadata=dict(self.reward_kwargs))
        scores = self.reward_fn(ctx).scores
        mean_reward = scores.mean().item()
        sem = scores.std().item() / max(len(scores) ** 0.5, 1.0)
        print(f"Eval (epoch {epoch}): mean_reward={mean_reward:.3f} +/- {sem:.3f} (n={len(scores)})", flush=True)

        self._fire_callbacks("on_evaluate", epoch=epoch, mean_reward=mean_reward)
        self.transformer.train()
        return mean_reward

    def _save_adapter(self, path: Path):
        """Write just the LoRA adapter.

        DRaFTTrainer does not inherit BaseDiffusionTrainer, so it did not pick up
        the adapter-only checkpointing fix and kept calling save_pretrained on the
        whole transformer - ~5GB of frozen base weights per checkpoint, against
        the ~26MB that is actually trainable.
        """
        from peft.utils import get_peft_model_state_dict

        path.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(get_peft_model_state_dict(self.transformer), path / "adapter_model.bin")
        except Exception:
            self.transformer.save_pretrained(str(path))

    def _fire_callbacks(self, method: str, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(self, **kwargs)

    def train(self):
        """Main training loop."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        config = self.config

        if not self.train_prompts:
            raise ValueError("No training prompts provided.")

        print(f"Starting DRaFT-{config.truncation_steps} training: {config.num_epochs} epochs")
        self._fire_callbacks("on_train_begin")

        for epoch in range(config.num_epochs):
            self._fire_callbacks("on_epoch_begin", epoch=epoch)

            # DRaFT has its own loop, so it never called _evaluate and reported
            # only the training reward. That is the reward it is directly
            # optimizing on the prompts it is optimizing, which says nothing about
            # generalization and is not comparable to the other trainers' curves.
            # Evaluate here, before the update, on the same held-out set at the
            # same fixed seed as everything else.
            if epoch % config.eval_every == 0:
                self._evaluate(epoch)

            epoch_prompts = self.train_prompts.copy()
            random.shuffle(epoch_prompts)

            self.optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_reward = 0.0
            num_steps = 0

            # ceil, not floor: range(0, N, step) runs ceil(N/step) iterations,
            # so the loss normalizer must match the iteration count or the
            # effective LR scales with the prompt count.
            total_accum_steps = max(1, math.ceil(len(epoch_prompts) / config.gradient_accumulation_steps))
            for i in range(0, len(epoch_prompts), config.gradient_accumulation_steps):
                batch = epoch_prompts[i : i + config.gradient_accumulation_steps]
                if not batch:
                    continue

                metrics = self._training_step(batch, num_accum_steps=total_accum_steps)
                epoch_loss += metrics["loss"]
                epoch_reward += metrics["mean_reward"]
                num_steps += 1

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.transformer.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            n = max(num_steps, 1)
            epoch_metrics = {"loss": epoch_loss / n, "mean_reward": epoch_reward / n}

            if epoch % config.log_every == 0:
                print(f"Epoch {epoch} | loss={epoch_metrics['loss']:.4f} | reward={epoch_metrics['mean_reward']:.3f}")

            self._fire_callbacks("on_epoch_end", epoch=epoch, metrics=epoch_metrics)

            if epoch > 0 and epoch % config.save_every == 0:
                save_path = Path(config.output_dir) / f"checkpoint-{epoch}"
                self._save_adapter(save_path)
                self._fire_callbacks("on_save", epoch=epoch, path=str(save_path))

        save_path = Path(config.output_dir) / f"checkpoint-{config.num_epochs}"
        self._save_adapter(save_path)
        self._fire_callbacks("on_save", epoch=config.num_epochs, path=str(save_path))
        self._fire_callbacks("on_train_end")
        print("DRaFT training complete.")
