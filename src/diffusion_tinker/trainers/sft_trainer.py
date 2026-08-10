"""SFT Trainer."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from diffusion_tinker.core.callbacks import TrainerCallback
from diffusion_tinker.core.latent_utils import encode_to_latents
from diffusion_tinker.core.noise_strategy import compute_flow_matching_loss
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

    def __init__(self, model: str, config: SFTConfig, callbacks: list[TrainerCallback] | None = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.callbacks = callbacks or []

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self._setup_model(model)
        self._setup_data()

        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    def _setup_model(self, model_id: str):
        from diffusion_tinker.models.model_setup import setup_sd3_model
        # SFTTrainer does not inherit BaseDiffusionTrainer, so it has no
        # _model_type of its own; the flow-matching loss needs one to pick the
        # SD3 or FLUX transformer signature. Only the SD3 path is wired up here,
        # so say so rather than silently mislabelling a FLUX model as sd3.
        if self.config.model_type == "auto":
            self._model_type = "flux" if "flux" in model_id.lower() else "sd3"
        else:
            self._model_type = self.config.model_type
        if self._model_type == "flux":
            raise NotImplementedError(
                "SFT is wired for SD3 only: _setup_model calls setup_sd3_model. "
                "compute_flow_matching_loss has a FLUX branch, but the pipeline and "
                "latent packing here do not, so a FLUX model would train on the "
                "wrong tensor layout rather than fail."
            )
        self.pipeline, self.transformer, self.vae, self.model_config = setup_sd3_model(
            model_id, self.config, self.device
        )

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

        self.holdout_rows = None
        if self.config.holdout_size > 0:
            n = self.config.holdout_size
            if n >= len(hf_ds):
                raise ValueError(f"holdout_size {n} leaves no training data ({len(hf_ds)} examples).")
            # take the tail, so the held-out examples are the same ones on every
            # run of the same dataset regardless of the shuffle seed
            self.holdout_rows = hf_ds.select(range(len(hf_ds) - n, len(hf_ds)))
            hf_ds = hf_ds.select(range(len(hf_ds) - n))
            print(f"Holding out {n} examples; training on {len(hf_ds)}")

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

        # Caption dropout substitutes an encoded empty prompt, not a zero tensor.
        # Encoded once and reused: the text encoders are already resident here, and
        # the result does not depend on the batch.
        if getattr(self, "_uncond_embeds", None) is None:
            with torch.no_grad():
                u_embeds, _, u_pooled, _ = self.pipeline.encode_prompt(
                    prompt=[""], prompt_2=None, prompt_3=None,
                    negative_prompt=None, do_classifier_free_guidance=False,
                    device=device,
                )
            self._uncond_embeds = u_embeds.detach()
            self._uncond_pooled = u_pooled.detach()

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            loss = compute_flow_matching_loss(
                transformer=self.transformer,
                latents=latents,
                noise=noise,
                sigmas=t,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_prompt_embeds,
                condition_dropout_mask=dropout_mask,
                uncond_embeds=self._uncond_embeds.to(device),
                uncond_pooled=self._uncond_pooled.to(device),
                model_type=self._model_type,
                guidance_scale=config.guidance_scale,
            )

        return {"loss": loss.item()}, loss

    def _prepare_holdout(self):
        """Encode the held-out batch once, with fixed noise and timesteps.

        The flow-matching loss is an expectation over (t, noise), so re-drawing
        them each eval would move the number by more than training does. Pinning
        both means a change in the reported loss is a change in the model.
        """
        cfg = self.config
        ds = _SFTDataset(self.holdout_rows, cfg.image_column, cfg.caption_column, cfg.resolution)
        batch = _sft_collate([ds[i] for i in range(len(ds))])

        with torch.no_grad():
            latents = encode_to_latents(self.vae, batch["images"].to(self.device, dtype=self.vae.dtype))
            prompt_embeds, _, pooled, _ = self.pipeline.encode_prompt(
                prompt=batch["captions"], prompt_2=None, prompt_3=None,
                negative_prompt=None, do_classifier_free_guidance=False, device=self.device,
            )

        n = latents.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(cfg.eval_seed)
        # an even spread over the schedule rather than a logit-normal draw, so the
        # held-out loss is not dominated by wherever the sample happened to land
        sigmas = torch.linspace(0.05, 0.95, n).to(self.device)
        noise = torch.randn(latents.shape, generator=gen).to(self.device, dtype=latents.dtype)

        self._holdout = {
            "latents": latents, "noise": noise, "sigmas": sigmas,
            "prompt_embeds": prompt_embeds, "pooled": pooled,
        }

    @torch.no_grad()
    def _evaluate(self, step: int) -> float | None:
        if self.holdout_rows is None:
            return None
        if getattr(self, "_holdout", None) is None:
            self._prepare_holdout()

        b = self._holdout
        autocast_dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        was_training = self.transformer.training
        self.transformer.eval()
        try:
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, cache_enabled=False):
                loss = compute_flow_matching_loss(
                    transformer=self.transformer,
                    latents=b["latents"], noise=b["noise"], sigmas=b["sigmas"],
                    prompt_embeds=b["prompt_embeds"], pooled_embeds=b["pooled"],
                    model_type=self._model_type,
                    guidance_scale=self.config.guidance_scale,
                )
        finally:
            if was_training:
                self.transformer.train()

        value = loss.item()
        print(f"Eval (step {step}): heldout_loss={value:.5f}", flush=True)
        self._fire_callbacks("on_evaluate", epoch=step, mean_reward=value)
        return value

    def _fire_callbacks(self, method: str, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(self, **kwargs)

    def train(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        config = self.config

        max_steps = config.max_train_steps
        if max_steps is None:
            max_steps = config.num_epochs * len(self.dataloader)

        print(f"Starting SFT: {max_steps} steps")
        self._fire_callbacks("on_train_begin")

        self.transformer.train()
        data_iter = iter(self.dataloader)

        # baseline before any update, so the curve starts from the untrained model
        self._evaluate(0)

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
                self._fire_callbacks("on_epoch_end", epoch=self.global_step, metrics=metrics)

            if self.global_step % config.eval_every_steps == 0:
                self._evaluate(self.global_step)

            if self.global_step % config.save_every == 0:
                self._save_checkpoint(self.global_step)

        self._save_checkpoint(self.global_step)
        self._fire_callbacks("on_train_end")
        print("SFT training complete.")

    def _save_checkpoint(self, step: int):
        # only the LoRA adapter is trainable, so save_pretrained would write ~5GB
        # of frozen base weights per checkpoint; same fix as the other trainers
        from peft.utils import get_peft_model_state_dict

        save_path = Path(self.config.output_dir) / f"checkpoint-{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(get_peft_model_state_dict(self.transformer), save_path / "adapter_model.bin")
        except Exception:
            self.transformer.save_pretrained(str(save_path))
        print(f"Saved checkpoint to {save_path}")
        self._fire_callbacks("on_save", epoch=step, path=str(save_path))
