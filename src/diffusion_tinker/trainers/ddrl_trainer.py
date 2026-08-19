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
from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer
from diffusion_tinker.trainers.ddrl_config import DDRLConfig


class DDRLTrainer(BaseDiffusionTrainer):

    config: DDRLConfig

    def __init__(self, model, reward_funcs, config, train_prompts=None, reward_weights=None, reward_mode="weighted_sum", reward_kwargs=None, callbacks=None):
        super().__init__(
            model=model, reward_funcs=reward_funcs, config=config,
            train_prompts=train_prompts, reward_weights=reward_weights, reward_mode=reward_mode,
            reward_kwargs=reward_kwargs, callbacks=callbacks,
        )
        self._data_latents = None
        self._data_captions = None
        self._uncond_embeds = None
        self._uncond_pooled = None
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
                # Mutating the caller's dataclass would silently carry
                # data_beta=0 into any other trainer built from the same object,
                # and the resume fingerprint would then record beta=0.0 for a run
                # the user launched at 0.3. Copy first.
                import dataclasses
                self.config = dataclasses.replace(self.config, data_beta=0.0)
            return

        if self.config.data_beta == 0:
            # The data term is switched off, so encoding the anchor images would
            # cost a VAE pass and hold latents the loss never reads. Skipping it
            # also makes data_beta=0 exactly FlowGRPO, which is what the bottom
            # of a beta sweep is supposed to be.
            print("data_beta=0, skipping anchor dataset (no forward-KL term)")
            return

        print(f"Loading data for DDRL forward KL: {self.config.train_dataset}...")
        from datasets import load_dataset

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

        # Captions for the anchor images, if the dataset has them. DDRL_THEORY.md
        # step 3b conditions the data term on the anchor image's own caption and
        # drops it with probability 0.2; without captions the term is
        # unconditional for every sample, which is a different (weaker) bound.
        self._data_captions = None
        cap_col = getattr(self.config, "caption_column", None)
        if cap_col:
            if cap_col not in ds.column_names:
                raise ValueError(
                    f"caption_column={cap_col!r} not in dataset. Available: "
                    f"{ds.column_names}. Set caption_column=None for an "
                    "unconditional anchor."
                )
            self._data_captions = [ds[i][cap_col] for i in range(max_images)]
            print(f"  Cached {len(self._data_captions)} anchor captions")

        # Index anchors by the prompt they were generated from, so the data term
        # can use p_data(.|c) for the c actually being rolled out (Algorithm 1
        # line 2) instead of an unrelated draw from the pool.
        self._data_by_prompt = None
        if getattr(self.config, "anchor_matches_prompt", False):
            if self._data_captions is None:
                raise ValueError(
                    "anchor_matches_prompt=True needs caption_column set: the "
                    "anchor images have to be matchable to the prompt they were "
                    "generated from."
                )
            by_prompt: dict[str, list[int]] = {}
            for i, cap in enumerate(self._data_captions):
                by_prompt.setdefault(cap, []).append(i)
            self._data_by_prompt = by_prompt

            missing = [p for p in self.train_prompts if p not in by_prompt]
            if missing:
                # Silently falling back to a random anchor for the unmatched
                # prompts would make the run a blend of two different objectives
                # with no record of the mix.
                raise ValueError(
                    f"{len(missing)} of {len(self.train_prompts)} training prompts "
                    f"have no anchor image (e.g. {missing[0]!r}). Generate anchors "
                    "for every training prompt, or set anchor_matches_prompt=False."
                )
            print(f"  Anchors indexed by prompt: {len(by_prompt)} prompts, "
                  f"{min(len(v) for v in by_prompt.values())}-"
                  f"{max(len(v) for v in by_prompt.values())} images each")
        print(f"  Cached {len(self._data_latents)} latents ({self._data_latents.nbytes / 1e6:.0f} MB)")

        # Cache a real unconditional embedding for the anchor term.
        #
        # The anchor images have nothing to do with the RL prompts, so the data
        # term must not condition on them - that part was always right. It was
        # implemented by multiplying the embeddings by zero, which is a different
        # thing from "no caption" and is not what either model sees. SD3 builds
        # its unconditional branch by *encoding the empty string* and evaluates
        # that on every CFG step; FLUX.1-dev is guidance-distilled and has no
        # unconditional branch at all, so zeroed T5 and pooled-CLIP embeddings
        # passed alongside a live guidance scalar are a conditioning state it has
        # never seen in training or sampling.
        #
        # That asymmetry runs in the same direction as the measured one: the
        # anchor helps SD3.5 by 0.097 and costs FLUX 0.108, and it does so
        # regardless of which model drew the anchor images or whether they contain
        # text, all of which this explains and none of which the anchor images
        # themselves did. Encoding "" once here costs one text-encoder pass and
        # gives both models the input they were trained on.
        with torch.no_grad():
            if self.config.offload_text_encoders:
                self._set_text_encoders_device(self.device)
            if self._model_type == "flux":
                u_embeds, u_pooled, _ = self.pipeline.encode_prompt(
                    prompt=[""], prompt_2=None, device=self.device, max_sequence_length=512
                )
            else:
                u_embeds, _, u_pooled, _ = self.pipeline.encode_prompt(
                    prompt=[""], prompt_2=None, prompt_3=None,
                    do_classifier_free_guidance=False, device=self.device,
                )
            if self.config.offload_text_encoders:
                self._set_text_encoders_device("cpu")
        self._uncond_embeds = u_embeds.detach().cpu()
        self._uncond_pooled = u_pooled.detach().cpu()
        print(f"  Cached unconditional embedding {tuple(self._uncond_embeds.shape)} "
              f"for the anchor term")

    @staticmethod
    def _recenter_per_group(advantages: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Subtract each prompt group's mean, so the advantages are zero-mean."""
        out = advantages.clone()
        for prompt in dict.fromkeys(prompts):
            idx = [i for i, p in enumerate(prompts) if p == prompt]
            out[idx] = advantages[idx] - advantages[idx].mean()
        return out

    def _compute_advantages(self, trajectory: TrajectoryBatch) -> TrajectoryBatch:
        raw_advantages = self.stat_tracker.update(trajectory.prompts, trajectory.rewards)

        # raw advantages needed for the zero-signal filter in base_trainer
        trajectory._raw_advantages = raw_advantages

        advantages = raw_advantages
        if self.config.beta_temp != 1.0:
            advantages = advantages / self.config.beta_temp

        if self.config.use_monotonic_transform:
            advantages = -torch.exp(-advantages)
            # lambda(x) = -exp(-x) is increasing, so it preserves the ranking of
            # advantages, but its range is (-inf, 0): every transformed advantage
            # comes out negative. In the PPO term -A * ratio that makes every
            # sampled action a discouraged one, so the policy walks away from its
            # own distribution and collapses - and the unbounded side of the
            # exponential makes it violent. Re-centering per prompt group
            # restores the zero-mean advantage a policy-gradient estimator needs
            # while keeping the transform's reweighting of the ranking.
            advantages = self._recenter_per_group(advantages, trajectory.prompts)

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
        total_grad_norm = 0.0
        num_clips = 0
        num_computed_steps = 0

        num_train = max(1, int(num_steps * config.timestep_fraction))
        stride = max(1, getattr(config, "timestep_stride", 1))
        timestep_indices = list(range(0, num_steps if stride > 1 else num_train, stride))
        if stride > 1:
            timestep_indices = timestep_indices[:max(1, num_steps // stride)]
        random.shuffle(timestep_indices)

        autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        # Mini-batch the replay over trajectories and step once per mini-batch
        # (true minibatch SGD), so a large group size * many prompts fits in
        # memory and the reward gets many updates per epoch instead of one.
        # (See FlowGRPO trainer.)
        mb_size = config.train_batch_size or batch_size
        # Shuffle so each mini-batch mixes prompts (see FlowGRPO trainer).
        perm = torch.randperm(batch_size, device=device)
        minibatches = [perm[s:s + mb_size] for s in range(0, batch_size, mb_size)]

        for sl in minibatches:
            self.optimizer.zero_grad()
            mb = trajectory[sl]
            mb_batch_size = len(mb)

            for j in timestep_indices:
                sigma = mb.timesteps[j]
                sigma_next = mb.timesteps[j + 1]

                if sigma_next.item() < 1e-6:
                    continue

                latent_t = mb.latents[:, j]
                next_latent_t = mb.next_latents[:, j]

                if has_signal:
                    step_noise_level = config.noise_level if j < num_steps - 1 else 0.0
                    # cache_enabled=False to match the sampling forward (see FlowGRPO)
                    # so the importance ratio isn't biased by autocast cast drift.
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, cache_enabled=False):
                        log_prob_new, prev_sample_mean = self._replay_step(
                            transformer=self.transformer,
                            latent_t=latent_t,
                            next_latent_t=next_latent_t,
                            sigma=sigma,
                            sigma_next=sigma_next,
                            prompt_embeds=mb.prompt_embeds,
                            pooled_embeds=mb.pooled_embeds,
                            guidance_scale=config.guidance_scale,
                            noise_level=step_noise_level,
                            negative_prompt_embeds=mb.negative_prompt_embeds,
                            negative_pooled_embeds=mb.negative_pooled_embeds,
                            img_ids=mb.img_ids,
                            txt_ids=mb.txt_ids,
                        )

                    log_prob_old = mb.log_probs[:, j]
                    ratio = torch.exp(log_prob_new.float() - log_prob_old.float())

                    advantages = mb.advantages
                    unclipped = -advantages * ratio
                    clipped = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                    rl_loss = torch.mean(torch.maximum(unclipped, clipped))
                else:
                    rl_loss = torch.tensor(0.0, device=device)
                    ratio = torch.tensor(1.0, device=device)

                data_loss = torch.tensor(0.0, device=device)
                # Only anchor when the RL term is actually contributing. With no
                # within-group variance there is nothing to learn from, rl_loss is
                # zero, and the anchor would be the entire gradient - pulling the
                # policy toward the anchor images with nothing opposing it.
                #
                # This is defensive rather than demonstrated. It was written for a
                # FLUX result where held-out exact-match OCR fell 0.625 -> 0.29
                # with the anchor on and held with it off, and that result is real
                # - but the group scores in those runs show has_signal was true in
                # every one of 59 epochs, so this gate would have been open
                # throughout and is not what explains it. Keep the gate anyway:
                # regularising a batch that cannot teach anything is wrong on its
                # own terms. Do not cite it as the fix for the FLUX collapse.
                # See report/analyze_flux_anchor.py.
                if config.data_beta > 0 and has_signal:
                    # timestep_sampling was declared in DDRLConfig and read
                    # nowhere, so "uniform" silently trained logit-normal.
                    if config.timestep_sampling == "matched":
                        # Algorithm 1 line 5: the anchor is noised to the same
                        # timestep the RL term is optimizing this pass, not to an
                        # independent draw. `sigma` is already the sigma for step
                        # j; broadcast it over the minibatch.
                        t = sigma.detach().reshape(-1)[:1].expand(mb_batch_size).to(device)
                    elif config.timestep_sampling == "uniform":
                        t = torch.rand(mb_batch_size, device=device)
                    elif config.timestep_sampling == "logit_normal":
                        u = torch.normal(mean=config.logit_mean, std=config.logit_std,
                                         size=(mb_batch_size,), device=device)
                        t = torch.sigmoid(u)
                    else:
                        raise ValueError(
                            "timestep_sampling must be 'matched', 'logit_normal' or "
                            f"'uniform', got {config.timestep_sampling!r}"
                        )

                    assert self._data_latents is not None, "data_beta > 0 requires train_dataset"
                    assert self._uncond_embeds is not None, (
                        "anchor term needs the cached unconditional embedding; "
                        "_setup_data did not run"
                    )
                    if getattr(self, "_data_by_prompt", None) is not None:
                        # Algorithm 1 line 2: the anchor is a base-model sample of
                        # the prompt being rolled out, not an unrelated image.
                        data_idx = torch.tensor(
                            [
                                self._data_by_prompt[p][
                                    int(torch.randint(len(self._data_by_prompt[p]), (1,)))
                                ]
                                for p in mb.prompts
                            ]
                        )
                    else:
                        data_idx = torch.randint(0, len(self._data_latents), (mb_batch_size,))
                    clean_latents = self._data_latents[data_idx].to(device, dtype=autocast_dtype)

                    noise = torch.randn_like(clean_latents)
                    # The anchor images are unrelated to the RL prompts, so the data
                    # term is evaluated unconditionally. Use the cached embedding of
                    # the empty string rather than zeroing the RL prompt's embedding:
                    # zeros are not what "no caption" looks like to either model, and
                    # FLUX has no unconditional branch to fall back on. See the note
                    # in _setup_data.
                    uncond_embeds = self._uncond_embeds.to(device, dtype=autocast_dtype)
                    uncond_pooled = self._uncond_pooled.to(device, dtype=autocast_dtype)
                    uncond_embeds = uncond_embeds.expand(mb_batch_size, -1, -1)
                    uncond_pooled = uncond_pooled.expand(mb_batch_size, -1)

                    # With captions available, condition on the anchor image's own
                    # caption and drop it with config.condition_dropout, which is
                    # DDRL_THEORY.md step 3b. Without them everything stays
                    # unconditional, as before.
                    anchor_embeds, anchor_pooled = uncond_embeds, uncond_pooled
                    drop_mask = None
                    if getattr(self, "_data_by_prompt", None) is not None:
                        # c is the rollout prompt, already encoded on the batch, so
                        # the data term conditions on exactly the same embedding the
                        # RL term does - no re-encoding and no text encoders needed.
                        anchor_embeds = mb.prompt_embeds.to(device, dtype=autocast_dtype)
                        anchor_pooled = mb.pooled_embeds.to(device, dtype=autocast_dtype)
                        drop_mask = (
                            torch.rand(mb_batch_size, device=device)
                            < config.condition_dropout
                        )
                    elif self._data_captions is not None:
                        caps = [self._data_captions[int(i)] for i in data_idx]
                        with torch.no_grad():
                            if config.offload_text_encoders:
                                self._set_text_encoders_device(device)
                            if self._model_type == "flux":
                                a_e, a_p, _ = self.pipeline.encode_prompt(
                                    prompt=caps, prompt_2=None, device=device,
                                    max_sequence_length=512)
                            else:
                                a_e, _, a_p, _ = self.pipeline.encode_prompt(
                                    prompt=caps, prompt_2=None, prompt_3=None,
                                    do_classifier_free_guidance=False, device=device)
                            if config.offload_text_encoders:
                                self._set_text_encoders_device("cpu")
                        anchor_embeds = a_e.to(device, dtype=autocast_dtype)
                        anchor_pooled = a_p.to(device, dtype=autocast_dtype)
                        drop_mask = (
                            torch.rand(mb_batch_size, device=device)
                            < config.condition_dropout
                        )

                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        data_loss = compute_flow_matching_loss(
                            transformer=self.transformer,
                            latents=clean_latents,
                            noise=noise,
                            sigmas=t,
                            prompt_embeds=anchor_embeds,
                            pooled_embeds=anchor_pooled,
                            condition_dropout_mask=drop_mask,
                            uncond_embeds=uncond_embeds,
                            uncond_pooled=uncond_pooled,
                            model_type=self._model_type,
                            guidance_scale=config.guidance_scale,
                        )

                loss = rl_loss + config.data_beta * data_loss
                loss = loss / len(timestep_indices)
                if loss.requires_grad:
                    loss.backward()

                total_rl_loss += rl_loss.item()
                total_data_loss += data_loss.item() if isinstance(data_loss, torch.Tensor) else data_loss
                total_ratio += ratio.mean().item()
                num_computed_steps += 1

            # clip_grad_norm_ returns the norm BEFORE clipping, and that number is
            # diagnostic in a way nothing else logged here is. The two terms are
            # summed into one loss and backpropagated together, so clipping scales
            # the combined gradient - a large anchor gradient shrinks the RL
            # component along with it. FLUX's data_loss runs about 50% higher than
            # SD3.5's on the same anchor set, so at equal data_beta it consumes
            # more of the clip budget on FLUX. Whether that is actually happening
            # is answerable only by logging the norm, which nothing did.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.transformer.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            total_grad_norm += float(grad_norm)
            num_clips += 1
            self.optimizer.step()

        n = max(num_computed_steps, 1)
        c = max(num_clips, 1)
        return {
            "rl_loss": total_rl_loss / n,
            "data_loss": total_data_loss / n,
            "total_loss": (total_rl_loss + config.data_beta * total_data_loss) / n,
            "mean_ratio": total_ratio / n,
            "mean_reward": trajectory.rewards.mean().item(),
            "mean_advantage": trajectory.advantages.mean().item(),
            # The statistic `has_signal` actually gates on. Worth logging because
            # `rl_loss` was used as a stand-in for it and is a bad one: it tracks
            # the mean advantage, which group normalisation drives to zero
            # whether or not there is spread, so a live epoch and a quiet one
            # both print 0.0000. Reading quiet-epoch counts off that gave upper
            # bounds that were written down as measurements.
            # Pre-clip gradient norm. Above max_grad_norm means clipping is
            # active and both terms are being scaled down together.
            "grad_norm": total_grad_norm / c,
            "advantage_std": trajectory.advantages.std().item(),
            "has_signal": has_signal,
        }
