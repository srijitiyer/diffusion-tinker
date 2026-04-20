"""DRaFT Trainer (arXiv:2309.17400)."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from PIL import Image as PILImage

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
    ):
        self.config = config
        self.train_prompts = train_prompts or []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self._setup_model(model)
        self.reward_fn = resolve_reward(reward_funcs, device=str(self.device))

        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    def _setup_model(self, model_id: str):
        from diffusers import StableDiffusion3Pipeline
        from peft import LoraConfig

        from diffusion_tinker.models.sd3_patch import SD3ModelConfig

        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16

        print(f"Loading {model_id}...")
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipeline.to(self.device)

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.model_config = SD3ModelConfig()

        self.vae.eval()
        self.vae.requires_grad_(False)
        for enc in [self.pipeline.text_encoder, self.pipeline.text_encoder_2, self.pipeline.text_encoder_3]:
            if enc is not None:
                enc.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.model_config.lora_target_modules,
        )
        self.transformer.add_adapter(lora_config)

        if self.config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        for p in self.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        print(f"LoRA applied: rank={self.config.lora_rank}")

    def _denoise_with_grad(self, prompts: list[str]) -> tuple[torch.Tensor, list]:
        """Run denoising with gradients through the last K steps."""
        device = self.device
        dtype = self.pipeline.transformer.dtype
        config = self.config
        batch_size = len(prompts)
        K = config.truncation_steps

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            self.pipeline.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                device=device,
            )
        )

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
        images_tensor = self.vae.decode(latents_decoded, return_dict=False)[0].clamp(0, 1)

        with torch.no_grad():
            images_np = (images_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images_np = images_np.transpose(0, 2, 3, 1)
            pil_images = [PILImage.fromarray(img) for img in images_np]

        return images_tensor, pil_images

    def _training_step(self, prompts: list[str]) -> dict[str, float]:
        config = self.config
        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            images_tensor, pil_images = self._denoise_with_grad(prompts)

        rewards = self._differentiable_reward(images_tensor, prompts)

        loss = -rewards.mean()
        loss.backward()

        return {
            "loss": loss.item(),
            "mean_reward": rewards.detach().mean().item(),
        }

    def _differentiable_reward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Compute reward on image tensor with gradient flow preserved."""
        from torchvision.transforms.functional import normalize, resize

        self.reward_fn._ensure_loaded()

        images_resized = resize(images, [224, 224], antialias=True)
        images_normalized = normalize(
            images_resized,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        if hasattr(self.reward_fn, "_clip"):
            clip = self.reward_fn._clip
            vision_out = clip.vision_model(pixel_values=images_normalized.to(clip.dtype))
            embed = clip.visual_projection(vision_out.pooler_output)
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

            if hasattr(self.reward_fn, "_mlp"):
                scores = self.reward_fn._mlp(embed).squeeze(-1)
            else:
                tokenizer = self.reward_fn._processor.tokenizer
                text_inputs = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                text_out = clip.text_model(**text_inputs)
                text_features = clip.text_projection(text_out.pooler_output)
                text_features = text_features / torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)
                scores = (embed * text_features.detach()).sum(dim=-1) * 100.0
        else:
            import warnings

            warnings.warn(
                "DRaFT reward does not have a CLIP model. "
                "Gradients will NOT flow through the reward. Use aesthetic or clip_score.",
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
            scores = self.reward_fn._compute(ctx).scores.to(self.device)

        return scores

    def train(self):
        """Main training loop."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        config = self.config

        if not self.train_prompts:
            raise ValueError("No training prompts provided.")

        print(f"Starting DRaFT-{config.truncation_steps} training: {config.num_epochs} epochs")

        for epoch in range(config.num_epochs):
            epoch_prompts = self.train_prompts.copy()
            random.shuffle(epoch_prompts)

            self.optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_reward = 0.0
            num_steps = 0

            for i in range(0, len(epoch_prompts), config.gradient_accumulation_steps):
                batch = epoch_prompts[i : i + config.gradient_accumulation_steps]
                if not batch:
                    continue

                metrics = self._training_step(batch)
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

            if epoch % config.log_every == 0:
                n = max(num_steps, 1)
                print(f"Epoch {epoch} | loss={epoch_loss / n:.4f} | reward={epoch_reward / n:.3f}")

            if epoch > 0 and epoch % config.save_every == 0:
                save_path = Path(config.output_dir) / f"checkpoint-{epoch}"
                save_path.mkdir(parents=True, exist_ok=True)
                self.transformer.save_pretrained(str(save_path))

        save_path = Path(config.output_dir) / f"checkpoint-{config.num_epochs}"
        save_path.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(str(save_path))
        print("DRaFT training complete.")
