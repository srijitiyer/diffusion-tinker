"""SFT Trainer."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torchvision.transforms as T
from peft import LoraConfig
from torch.utils.data import DataLoader, Dataset

from diffusion_tinker.core.latent_utils import encode_to_latents
from diffusion_tinker.core.noise_strategy import compute_flow_matching_loss
from diffusion_tinker.models.sd3_patch import SD3ModelConfig
from diffusion_tinker.trainers.sft_config import SFTConfig


class _SFTDataset(Dataset):
    """Simple image-text dataset for SFT."""

    def __init__(self, hf_dataset, image_col, caption_col, resolution):
        self.dataset = hf_dataset
        self.image_col = image_col
        self.caption_col = caption_col
        self.transform = T.Compose(
            [
                T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(resolution),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        img = row[self.image_col]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return {"image": self.transform(img), "caption": row[self.caption_col]}


def _sft_collate(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }


class SFTTrainer:
    """Supervised fine-tuning trainer for diffusion models."""

    def __init__(self, model: str, config: SFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0

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

        print(f"LoRA applied: rank={self.config.lora_rank}")

    def _setup_data(self):
        if self.config.train_dataset is None:
            raise ValueError("SFT requires train_dataset in config.")

        from datasets import load_dataset

        print(f"Loading dataset: {self.config.train_dataset}...")
        if os.path.isdir(self.config.train_dataset):
            hf_ds = load_dataset("imagefolder", data_dir=self.config.train_dataset, split=self.config.dataset_split)
        else:
            hf_ds = load_dataset(self.config.train_dataset, split=self.config.dataset_split)

        for col in [self.config.image_column, self.config.caption_column]:
            if col not in hf_ds.column_names:
                raise ValueError(
                    f"Dataset has no '{col}' column. Available: {hf_ds.column_names}. "
                    f"Set image_column/caption_column in config."
                )

        ds = _SFTDataset(hf_ds, self.config.image_column, self.config.caption_column, self.config.resolution)
        self.dataloader = DataLoader(
            ds,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=_sft_collate,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Dataset: {len(ds)} images")

    def _training_step(self, batch: dict) -> dict[str, float]:
        config = self.config
        device = self.device
        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        B = batch["images"].shape[0]

        with torch.no_grad():
            latents = encode_to_latents(self.vae, batch["images"].to(device, dtype=self.vae.dtype))

        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=batch["captions"],
                prompt_2=None,
                prompt_3=None,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                device=device,
            )

        u = torch.normal(mean=config.logit_mean, std=config.logit_std, size=(B,), device=device)
        t = torch.sigmoid(u)

        noise = torch.randn_like(latents)

        dropout_mask = torch.rand(B, device=device) < config.condition_dropout

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            loss = compute_flow_matching_loss(
                transformer=self.transformer,
                latents=latents,
                noise=noise,
                sigmas=t,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_prompt_embeds,
                condition_dropout_mask=dropout_mask,
            )

        return {"loss": loss.item()}, loss

    def train(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        config = self.config

        max_steps = config.max_train_steps
        if max_steps is None:
            max_steps = config.num_epochs * len(self.dataloader)

        print(f"Starting SFT: {max_steps} steps")

        self.transformer.train()
        data_iter = iter(self.dataloader)

        while self.global_step < max_steps:
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
                config.max_grad_norm,
            )
            self.optimizer.step()
            self.global_step += 1

            if self.global_step % config.log_every == 0:
                print(f"Step {self.global_step}/{max_steps} | loss={metrics['loss']:.4f}")

            if self.global_step % config.save_every == 0:
                self._save_checkpoint(self.global_step)

        self._save_checkpoint(self.global_step)
        print("SFT training complete.")

    def _save_checkpoint(self, step: int):
        save_path = Path(self.config.output_dir) / f"checkpoint-{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(str(save_path))
        print(f"Saved checkpoint to {save_path}")
