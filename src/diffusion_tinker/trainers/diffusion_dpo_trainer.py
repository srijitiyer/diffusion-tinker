"""DiffusionDPO Trainer (arXiv:2311.12908)."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_tinker.core.callbacks import TrainerCallback
from diffusion_tinker.core.latent_utils import encode_to_latents
from diffusion_tinker.core.preference_dataset import PreferenceDataset, preference_collate_fn
from diffusion_tinker.trainers.diffusion_dpo_config import DiffusionDPOConfig


class DiffusionDPOTrainer:
    """Trainer implementing DiffusionDPO for diffusion models."""

    def __init__(self, model: str, config: DiffusionDPOConfig, callbacks: list[TrainerCallback] | None = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.pipeline, self.transformer, self.vae, self.model_config = setup_sd3_model(
            model_id, self.config, self.device
        )

    def _setup_data(self):
        if self.config.dataset_name is None:
            raise ValueError("DiffusionDPO requires dataset_name in config.")

        from datasets import load_dataset

        print(f"Loading preference dataset: {self.config.dataset_name}...")
        hf_ds = load_dataset(self.config.dataset_name, split=self.config.dataset_split)

        for col in [self.config.image_column_winner, self.config.image_column_loser, self.config.caption_column]:
            if col not in hf_ds.column_names:
                raise ValueError(
                    f"Dataset has no '{col}' column. Available: {hf_ds.column_names}."
                )

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

        self.eval_loader = None
        if self.config.eval_split:
            eval_hf = load_dataset(self.config.dataset_name, split=self.config.eval_split)
            eval_ds = PreferenceDataset(
                hf_dataset=eval_hf,
                winner_col=self.config.image_column_winner,
                loser_col=self.config.image_column_loser,
                caption_col=self.config.caption_column,
                label_col=self.config.label_column,
                resolution=self.config.resolution,
            )
            # not shuffled: every eval must see the same pairs in the same order
            # Eval is 4 forwards per pair per timestep, so it batches wider than
            # training even when training runs at batch 1 to fit memory. There is
            # no backward here, so the activations do not have to be kept.
            self.eval_loader = DataLoader(
                eval_ds,
                batch_size=max(4, self.config.train_batch_size),
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=preference_collate_fn,
            )
            print(f"Held-out split '{self.config.eval_split}': {len(eval_ds)} pairs")

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

        Loss: -log sigmoid(-0.5 * beta * (model_diff - ref_diff)), where
        model_diff = loss_w - loss_l under the policy, ref_diff is the same under
        the frozen reference, and loss = ||v_target - v_pred||^2. The 0.5 and the
        absence of an explicit T factor both match the reference implementation:
        beta is expected to already absorb T, which is why its default is 5000
        rather than an LLM-DPO-scale value.
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

        # One noise draw, shared by both branches, the way the reference
        # implementation does it (it chunks the concatenated batch and repeats
        # the winner's noise). The loss compares (loss_w_model - loss_w_ref)
        # against (loss_l_model - loss_l_ref), and sharing the draw lets the two
        # Monte-Carlo errors partly cancel. On a fixed pair with a perturbed
        # copy of the model as the policy, the sigmoid argument came out at
        # std 2.79 shared versus 3.65 independent, and the fraction of samples
        # pushed past |arg|>8 - where logsigmoid is flat and the gradient is
        # gone - went from 0.2% to 2.5%. beta is ~5000, so that tail is the part
        # that matters. The timestep was already shared.
        noise = torch.randn_like(winner_latents)
        noisy_w = (1.0 - t_bc) * winner_latents + t_bc * noise
        noisy_l = (1.0 - t_bc) * loser_latents + t_bc * noise

        v_target_w = noise - winner_latents
        v_target_l = noise - loser_latents

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
            self.transformer.disable_adapters()
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
                self.transformer.enable_adapters()

        loss_w_model = (v_target_w.float() - v_pred_w.float()).pow(2).mean(dim=[1, 2, 3])
        loss_w_ref = (v_target_w.float() - v_ref_w.float()).pow(2).mean(dim=[1, 2, 3])
        loss_l_model = (v_target_l.float() - v_pred_l.float()).pow(2).mean(dim=[1, 2, 3])
        loss_l_ref = (v_target_l.float() - v_ref_l.float()).pow(2).mean(dim=[1, 2, 3])

        inside_sigmoid = -0.5 * config.beta * ((loss_w_model - loss_w_ref) - (loss_l_model - loss_l_ref))
        loss = -F.logsigmoid(inside_sigmoid).mean()

        return {
            "dpo_loss": loss.item(),
            "winner_loss_delta": (loss_w_model - loss_w_ref).mean().item(),
            "loser_loss_delta": (loss_l_model - loss_l_ref).mean().item(),
            "implicit_reward": inside_sigmoid.mean().item(),
        }, loss

    @torch.no_grad()
    def _evaluate(self, step: int) -> float | None:
        """Preference accuracy on a held-out split.

        For each pair, score the implicit reward -(loss_model - loss_ref) on the
        winner and on the loser and check which is higher. That is the same
        quantity the training loss pushes on, measured on pairs the run never
        saw, so it is the DPO analogue of the held-out reward every other trainer
        reports. Chance is 0.5.

        Ties count as half, which matters more than it sounds: at step 0 the LoRA
        B matrix is zero, so the policy and the reference are the same network and
        every margin is exactly 0.0. Counting only margin>0 would report 0.000 at
        step 0 and make the curve appear to start from a floor it never had.

        The mean margin is reported alongside the accuracy. It uses the size of
        the gap rather than only its sign, so on a held-out set this small it is
        the number with any power in it - accuracy over ~100 pairs has a standard
        error near 0.05 all by itself.

        Everything that makes the number comparable across evals is pinned: the
        same pairs, the same timesteps, the same noise, every time. The eval
        draws from its own generator so training does not shift the stream.
        """
        if self.eval_loader is None:
            return None

        config = self.config
        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        was_training = self.transformer.training
        self.transformer.eval()

        correct = 0.0
        total = 0
        margin_sum = 0.0
        gen = torch.Generator(device="cpu").manual_seed(config.eval_seed)
        # a fixed spread of noise levels rather than random ones, so no eval is
        # luckier than another in where it sampled the schedule
        ts = torch.linspace(0.1, 0.9, config.num_eval_timesteps)

        try:
            for batch in self.eval_loader:
                if total >= config.num_eval_pairs:
                    break
                B = batch["winner"].shape[0]
                w_lat = encode_to_latents(self.vae, batch["winner"].to(self.device, dtype=self.vae.dtype))
                l_lat = encode_to_latents(self.vae, batch["loser"].to(self.device, dtype=self.vae.dtype))
                prompt_embeds, pooled_embeds = self._encode_prompts(batch["prompts"])

                margin = torch.zeros(B, device=self.device)
                for t_val in ts:
                    t_bc = torch.full((B, 1, 1, 1), float(t_val), device=self.device)
                    noise = torch.randn(w_lat.shape, generator=gen).to(self.device, dtype=w_lat.dtype)
                    timestep = torch.full((B,), float(t_val) * 1000.0, device=self.device)

                    deltas = []
                    for lat in (w_lat, l_lat):
                        noisy = (1.0 - t_bc) * lat + t_bc * noise
                        target = noise - lat
                        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                            v_model = self.transformer(
                                hidden_states=noisy, timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_embeds, return_dict=False,
                            )[0]
                            self.transformer.disable_adapters()
                            try:
                                v_ref = self.transformer(
                                    hidden_states=noisy, timestep=timestep,
                                    encoder_hidden_states=prompt_embeds,
                                    pooled_projections=pooled_embeds, return_dict=False,
                                )[0]
                            finally:
                                self.transformer.enable_adapters()
                        l_model = (target.float() - v_model.float()).pow(2).mean(dim=[1, 2, 3])
                        l_ref = (target.float() - v_ref.float()).pow(2).mean(dim=[1, 2, 3])
                        # implicit reward is the negative change in denoising loss
                        deltas.append(-(l_model - l_ref))
                    margin += deltas[0] - deltas[1]

                correct += (margin > 0).sum().item() + 0.5 * (margin == 0).sum().item()
                margin_sum += margin.sum().item()
                total += B
        finally:
            if was_training:
                self.transformer.train()

        if total == 0:
            return None
        acc = correct / total
        mean_margin = margin_sum / total
        print(f"Eval (step {step}): preference accuracy={acc:.3f} "
              f"mean_margin={mean_margin:+.5f} on {total} held-out pairs", flush=True)
        self._fire_callbacks("on_evaluate", epoch=step, mean_reward=acc)
        return acc

    def _fire_callbacks(self, method: str, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(self, **kwargs)

    def train(self):
        """Step-based training loop over preference pairs."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"Starting DiffusionDPO training: {self.config.max_train_steps} steps")
        self._fire_callbacks("on_train_begin")

        self.transformer.train()
        data_iter = iter(self.dataloader)
        global_step = 0

        # Baseline first, before any update, so the curve starts at the untrained
        # adapter. Chance is 0.5 there; anything else means the eval is leaking.
        eval_interval = max(1, self.config.max_train_steps // 10)
        self._evaluate(0)

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
                self._fire_callbacks("on_epoch_end", epoch=global_step, metrics=metrics)

            if global_step % eval_interval == 0:
                self._evaluate(global_step)

            if global_step % 500 == 0:
                self._save_checkpoint(global_step)

        self._save_checkpoint(global_step)
        self._fire_callbacks("on_train_end")
        print("DiffusionDPO training complete.")

    def _save_checkpoint(self, step: int):
        # Same reason as BaseDiffusionTrainer._save_adapter: only the adapter is
        # trainable, so save_pretrained would write ~5GB of frozen base weights
        # every 500 steps. This trainer was missed when that was fixed elsewhere.
        from peft.utils import get_peft_model_state_dict

        save_path = Path(self.config.output_dir) / f"checkpoint-{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(get_peft_model_state_dict(self.transformer), save_path / "adapter_model.bin")
        except Exception:
            self.transformer.save_pretrained(str(save_path))
        print(f"Saved checkpoint to {save_path}")
        self._fire_callbacks("on_save", epoch=step, path=str(save_path))
