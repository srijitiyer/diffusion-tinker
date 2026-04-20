"""DiffusionDPO Trainer (arXiv:2311.12908)."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig
from torch.utils.data import DataLoader

from diffusion_tinker.core.latent_utils import encode_to_latents
from diffusion_tinker.core.preference_dataset import PreferenceDataset, preference_collate_fn
from diffusion_tinker.models.sd3_patch import SD3ModelConfig
from diffusion_tinker.trainers.diffusion_dpo_config import DiffusionDPOConfig


class DiffusionDPOTrainer:
    """Trainer implementing DiffusionDPO for diffusion models."""

    def __init__(self, model: str, config: DiffusionDPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self._setup_model(model)
        self._setup_data()

        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

        print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    def _setup_model(self, model_id: str):
        from diffusers import StableDiffusion3Pipeline

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

        print(f"LoRA applied: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")

    def _setup_data(self):
        if self.config.dataset_name is None:
            raise ValueError("DiffusionDPO requires dataset_name in config.")

        from datasets import load_dataset

        print(f"Loading preference dataset: {self.config.dataset_name}...")
        hf_ds = load_dataset(self.config.dataset_name, split=self.config.dataset_split)

        pref_ds = PreferenceDataset(
            hf_dataset=hf_ds,
            winner_col=self.config.image_column_winner,
            loser_col=self.config.image_column_loser,
            caption_col=self.config.caption_column,
            label_col=self.config.label_column,
            resolution=self.config.resolution,
        )

        self.dataloader = DataLoader(
            pref_ds,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=preference_collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Dataset loaded: {len(pref_ds)} preference pairs")

    def _encode_prompts(self, prompts: list[str]):
        do_cfg = False
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            do_classifier_free_guidance=do_cfg,
            device=self.device,
        )
        return prompt_embeds, pooled_prompt_embeds

    def _training_step(self, batch: dict) -> dict[str, float]:
        """DiffusionDPO training step on a batch of preference pairs.

        Loss: -log sigmoid(-beta * T * (loss_w - loss_w_ref - loss_l + loss_l_ref))
        where loss = ||v_target - v_pred||^2 (denoising error).
        """
        device = self.device
        config = self.config
        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        B = batch["winner"].shape[0]

        with torch.no_grad():
            winner_latents = encode_to_latents(self.vae, batch["winner"].to(device, dtype=self.vae.dtype))
            loser_latents = encode_to_latents(self.vae, batch["loser"].to(device, dtype=self.vae.dtype))

        with torch.no_grad():
            prompt_embeds, pooled_embeds = self._encode_prompts(batch["prompts"])

        t = torch.rand(B, device=device)
        t_bc = t.view(B, 1, 1, 1)

        noise_w = torch.randn_like(winner_latents)
        noise_l = torch.randn_like(loser_latents)
        noisy_w = (1.0 - t_bc) * winner_latents + t_bc * noise_w
        noisy_l = (1.0 - t_bc) * loser_latents + t_bc * noise_l

        v_target_w = noise_w - winner_latents
        v_target_l = noise_l - loser_latents

        timestep = t * 1000.0

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            v_pred_w = self.transformer(
                hidden_states=noisy_w,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
            v_pred_l = self.transformer(
                hidden_states=noisy_l,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

        # reference model = base model with LoRA disabled
        with torch.no_grad():
            self.transformer.disable_adapter_layers()
            try:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    v_ref_w = self.transformer(
                        hidden_states=noisy_w,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0]
                    v_ref_l = self.transformer(
                        hidden_states=noisy_l,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0]
            finally:
                self.transformer.enable_adapter_layers()

        loss_w_model = (v_target_w.float() - v_pred_w.float()).pow(2).mean(dim=[1, 2, 3])
        loss_w_ref = (v_target_w.float() - v_ref_w.float()).pow(2).mean(dim=[1, 2, 3])
        loss_l_model = (v_target_l.float() - v_pred_l.float()).pow(2).mean(dim=[1, 2, 3])
        loss_l_ref = (v_target_l.float() - v_ref_l.float()).pow(2).mean(dim=[1, 2, 3])

        inside_sigmoid = -config.beta * ((loss_w_model - loss_w_ref) - (loss_l_model - loss_l_ref))
        loss = -F.logsigmoid(inside_sigmoid).mean()

        return {
            "dpo_loss": loss.item(),
            "winner_loss_delta": (loss_w_model - loss_w_ref).mean().item(),
            "loser_loss_delta": (loss_l_model - loss_l_ref).mean().item(),
            "implicit_reward": inside_sigmoid.mean().item(),
        }, loss

    def train(self):
        """Step-based training loop over preference pairs."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"Starting DiffusionDPO training: {self.config.max_train_steps} steps")

        self.transformer.train()
        data_iter = iter(self.dataloader)
        global_step = 0

        while global_step < self.config.max_train_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            self.optimizer.zero_grad()
            metrics, loss = self._training_step(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.transformer.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            global_step += 1

            if global_step % 50 == 0:
                log_str = f"Step {global_step}/{self.config.max_train_steps}"
                for k, v in metrics.items():
                    log_str += f" | {k}={v:.4f}"
                print(log_str)

            if global_step % 500 == 0:
                self._save_checkpoint(global_step)

        self._save_checkpoint(global_step)
        print("DiffusionDPO training complete.")

    def _save_checkpoint(self, step: int):
        save_path = Path(self.config.output_dir) / f"checkpoint-{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(str(save_path))
        print(f"Saved checkpoint to {save_path}")
