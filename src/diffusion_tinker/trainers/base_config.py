from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseDiffusionConfig:
    """Base configuration for all diffusion RL trainers."""

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64

    # Optimization
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4

    # Training schedule
    num_epochs: int = 100
    save_every: int = 20
    eval_every: int = 10
    log_every: int = 1

    # Mini-batch size for the gradient-tracked replay/training pass. The rollout
    # collects a full batch (num_prompts * num_samples_per_prompt trajectories),
    # but replaying all of them through the transformer at once OOMs once the
    # group size is large. When set, the training step splits the batch into
    # chunks of this many trajectories, so memory is bounded by train_batch_size
    # instead of the full batch. This is what lets you use many diverse prompts
    # at a large group size (FlowGRPO's actual setup) on a single GPU.
    # None = replay the whole batch at once.
    #
    # This is minibatch SGD, NOT gradient accumulation: the trainers zero_grad
    # and step once per chunk, so halving this doubles the number of optimizer
    # steps per epoch and halves the batch each one sees. It therefore changes
    # the optimization, and two runs that differ in train_batch_size are not a
    # controlled comparison. (This comment previously said "accumulates gradients
    # across them", which is what the memory-knob reading assumes and is wrong.)
    train_batch_size: int | None = None

    # Accumulate gradients across the train_batch_size chunks and step ONCE per
    # epoch, instead of stepping per chunk. With it on, the optimizer sees the
    # full rollout batch and halving the chunk no longer changes the optimization.
    #
    # It does NOT make train_batch_size free of consequence, which an earlier
    # version of this comment claimed. bf16 matmuls select cuBLAS kernels by
    # shape, so a different chunk produces slightly different activations, and
    # OCR is near-binary per image - a nudge flips an image between 0 and 1.
    # Measured on one node, same model, same prompts, same pinned eval latents:
    # train_batch_size=8 scored 0.607 and 4 scored 0.565. Arms being compared
    # must therefore share train_batch_size as well as GPU model.
    #
    # Needed to reproduce DDRL. Algorithm 1 collects a rollout, computes every
    # L^n_t, and line 8 updates once "w.r.t. all L^n_t"; the paper's ~300
    # iterations are therefore ~300 optimizer steps at a batch of 48 groups x 24
    # samples. Under minibatch SGD the same run at a chunk of 4 takes 288 steps
    # per epoch, so 300 epochs is 86,400 steps at the paper's learning rate - a
    # different optimization, not a different batching.
    grad_accumulate: bool = False

    # Exponential moving average of the LoRA weights, evaluated instead of the
    # live ones. FlowGRPO's SD3.5 OCR config sets train.ema = True and the DDRL
    # paper follows FlowGRPO's settings, so results without it are not comparable
    # to the published table. Defaults match flow_grpo/ema.py.
    use_ema: bool = False
    ema_decay: float = 0.9
    ema_update_interval: int = 8

    # When train_prompts is a large pool, sample this many prompts per epoch
    # (a fresh random subset each epoch) instead of rolling out every prompt.
    # Lets you train on a big, diverse prompt set - which gives the reward room
    # to climb above the base model - without OOMing on num_prompts * group_size
    # trajectories per epoch. None = use all prompts every epoch.
    prompts_per_epoch: int | None = None

    # Device for the reward model. None means the training device. Some reward
    # models do not coexist happily with a large transformer on one card: FLUX
    # plus OWLv2 (the counting detector) segfaults in a linear forward at every
    # group size and at both 512px and 256px, while the same setup with the OCR
    # reward trains fine, so it is not a memory ceiling. "cpu" trades reward
    # throughput for getting the run to complete at all.
    reward_device: str | None = None

    # Held-out prompts for evaluation. None = fall back to the first
    # num_eval_prompts training prompts, which measures fit on the training set
    # rather than generalization, so pass a real held-out list for any result
    # you intend to report.
    eval_prompts: list[str] | None = None
    num_eval_prompts: int = 4
    # Images per eval prompt. The eval mean is over num_eval_prompts *
    # num_eval_samples_per_prompt images, so this is the main knob on how much
    # sampling noise is left in the eval curve.
    num_eval_samples_per_prompt: int = 1
    # Eval draws its noise from a dedicated generator seeded with this, so every
    # epoch evaluates on identical latents and the curve moves only when the
    # policy moves. Without it eval shares the global RNG stream, which training
    # advances by a different amount each epoch and which swamps the signal.
    eval_seed: int = 1234
    # Extra reward names scored on the eval images alongside the training reward,
    # reported but never optimized. Needed to measure a regularizer at all: DDRL
    # buys off-distribution behaviour by *spending* task reward, and its own
    # paper reports a lower OCR than the baselines it beats (0.823 against
    # FlowGRPO's 0.845) while holding ClipScore/PickScore/ImageReward at the base
    # model's values, which is where its human-preference win comes from. Track
    # only the optimized reward and the anchor can only ever look like a cost.
    # e.g. ["clip_score", "pickscore"]. None = just the training reward.
    eval_metrics: list[str] | None = None

    # Sampling / rollout
    num_samples_per_prompt: int = 4
    num_inference_steps: int = 28
    num_eval_inference_steps: int = 28
    guidance_scale: float = 7.0
    # Guidance for the eval sampler. None = use guidance_scale, which is what this
    # did before and is wrong whenever an algorithm's rollout guidance differs
    # from the settings its numbers are meant to be comparable at. DDRL drops CFG
    # for rollout ("DDRL does not adopt CFG") while FlowGRPO keeps it, so with one
    # shared knob the two arms are also evaluated differently and the reward gap
    # between them partly measures the eval, not the policy. Set this to the base
    # model's default (4.5 for SD3.5) across every arm being compared.
    eval_guidance_scale: float | None = None
    # SDE exploration noise. For FlowGRPO-style RL with a KL anchor, use ~0.7
    # (the FlowGRPO value). Beware: the per-step KL penalty scales as
    # 1/noise_level**2 (its variance is std_dev_t**2 ~ noise_level**2), so a
    # small noise_level with kl_beta>0 silently over-weights the anchor and
    # freezes the policy at the reference - stable but the reward never moves.
    # 0.7, which is FlowGRPO's value and the one every result in VALIDATION.md
    # was produced at. The old default of 0.1 is a leftover from before the
    # sigma=1 singularity was fixed: back then 0.1 was the only level that
    # appeared to work, because the uncapped diffusion coefficient multiplied
    # the latent by -0.16 at 0.1 against -55.8 at 0.7. With the cap in place
    # that reason is gone.
    #
    # It also couples to clip_range. Measured on an A5000 after the fix: the
    # importance ratio sits at 0.9993 at noise 0.7 and 0.9721 at noise 0.1, so
    # |ratio - 1| is 0.0007 against 0.028. A clip_range of 1e-3 is matched to
    # the first and would bind on every step at the second. Change one and
    # check the other; `mean_ratio` in the logged metrics tells you which
    # regime you are in.
    noise_level: float = 0.7
    # Windowed SDE: sample stochastically on a randomly placed contiguous window
    # of this many steps and deterministically elsewhere, training only on the
    # window. 0 = stochastic on every step, which is what SD3.5 uses.
    #
    # FlowGRPO turns this on for FLUX and not for SD3: geneval_flux_fast sets
    # sde_window_size=3 with sde_window_range=(0, num_steps//2) on a 6-step model,
    # alongside noise_level 0.8 and clip_range 1e-5. FLUX.1-dev is
    # guidance-distilled and few-step, and putting exploration noise on every step
    # of a 6-step trajectory is not how they drive it - which is what every FLUX
    # number in this repo was measured under.
    sde_window_size: int = 0
    # (lo, hi) bounds the window start; the window is [start, start+size) with
    # start drawn uniformly from [lo, hi-size]. None = (0, num_inference_steps//2),
    # their setting.
    sde_window_range: tuple[int, int] | None = None
    resolution: int = 512

    # RL
    # 1e-3, not the 0.2 that PPO implementations usually carry, because the
    # log-prob here is *mean*-reduced over the latent dimensions rather than
    # summed (`log_prob.mean(dim=...)` in noise_strategy). That keeps the
    # importance ratio within a whisker of 1 - measured 0.9977 to 0.9995 across
    # every validated run, so |ratio - 1| stays under 0.0025. A clip_range of 0.2
    # clamps to [0.8, 1.2] and therefore never binds: the trust region silently
    # does nothing and training runs as unclipped policy gradient. This default
    # was 0.2 and every result in VALIDATION.md was produced at 1e-3.
    #
    # If you switch the reduction to a sum, raise this accordingly - the right
    # value scales with how far the ratio actually moves, which `mean_ratio` in
    # the logged metrics reports.
    clip_range: float = 1e-3
    adv_clip_max: float = 5.0
    # Train only on the first timestep_fraction of denoising steps (the higher-
    # sigma ones). The last low-sigma steps have std_dev_t -> 0, where the SDE
    # log-prob is ill-conditioned and the importance ratio is unreliable;
    # FlowGRPO drops them with timestep_fraction=0.99. 1.0 = train on all steps.
    timestep_fraction: float = 0.99
    # Stride over the denoising steps, so T = {0, stride, 2*stride, ...}. DDRL's
    # published algorithm trains on T = {0,2,4,...,18} out of 20, i.e. every other
    # step across the whole trajectory. timestep_fraction alone cannot express
    # that: it takes a contiguous prefix, so 0.5 gives steps 0-9, which is the
    # high-sigma half rather than a spread. Leave at 1 for the prefix behaviour.
    timestep_stride: int = 1
    # Normalize advantages by the global batch std instead of per-prompt std.
    # Helps low-variance rewards (e.g. aesthetic) where a tiny per-prompt std
    # otherwise amplifies sampling noise. Matches FlowGRPO's global_std option.
    advantage_global_std: bool = False

    # Model
    model_type: str = "auto"  # "auto", "sd3", or "flux"

    # Memory
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    # Offload the (frozen) text encoders to CPU during the gradient-tracked
    # training step, keeping them on GPU only for sampling/eval. Frees ~10GB
    # (T5-XXL) so more samples/larger replay batches fit on a 24GB card, at the
    # cost of a per-epoch CPU<->GPU transfer. Off by default.
    offload_text_encoders: bool = False

    # Output
    output_dir: str = "./output"
    seed: int = 42
    save_best: bool = True
    # Write a resumable state (adapter, optimizer moments, epoch, advantage
    # tracker) after every epoch, and pick it up automatically on the next start
    # if one is found in output_dir. Off by default because it costs a ~80MB
    # write per epoch, but it is what makes a run survive a node that kills jobs
    # at random: without it a crash discards every epoch of training and the run
    # restarts from scratch, so no experiment can outlast the mean time between
    # crashes. The optimizer moments are saved too, deliberately - resuming with
    # a fresh AdamW leaves a transient in the curve exactly where a drift
    # measurement would read it.
    #
    # The RNG state (torch, CUDA, python) is saved and restored too, so a resumed
    # run draws the same randomness an uninterrupted one would. This was initially
    # left out on the reasoning that unequal randomness is a nuisance shared by
    # every arm - which only holds if arms resume equally often. Measured on this
    # cluster they do not: the data_beta=0.3 arms took 40 crashes to data_beta=0's
    # 31, so resume frequency would have varied alongside the quantity under
    # comparison and shown up as an effect of the anchor.
    resumable: bool = False

    # Stop after this many consecutive evals without a new best. 0 disables it.
    # Worth turning on: measured runs give back real performance past their peak.
    # SFT reaches a held-out minimum of 0.19742 around step 3098 and drifts back
    # to 0.20035; FLUX + DDRL degrades held-out on every task tried while its
    # training reward keeps rising (counting 0.667 -> 0.580, exact-match OCR
    # 0.625 -> 0.375). The training loss and the training reward both keep
    # improving throughout, so nothing about this is visible without a held-out
    # eval, and the best checkpoint is already being written on improvement.
    early_stop_patience: int = 0

    def __post_init__(self):
        """Reject configurations that would otherwise fail silently.

        Every check here corresponds to a way the library previously accepted a
        setting and then did something other than what it said. None of these
        raised before; they produced a run that looked normal and reported a
        number that meant something else.
        """
        if self.mixed_precision not in ("bf16", "fp16"):
            # Ten call sites read `bfloat16 if mixed_precision == "bf16" else
            # float16`, so "fp32", "no", or a typo silently selected fp16 - not an
            # error, and specifically not the fp32 a user asking for "fp32" wanted.
            raise ValueError(
                f"mixed_precision must be 'bf16' or 'fp16', got {self.mixed_precision!r}. "
                "Any other value silently selected fp16."
            )

        if self.model_type not in ("auto", "sd3", "flux"):
            # _setup_model branches on == "flux" and falls through to SD3, so
            # model_type="sdxl" loaded SD3.5 without complaint.
            raise ValueError(
                f"model_type must be 'auto', 'sd3' or 'flux', got {self.model_type!r}. "
                "Any other value silently loaded the SD3 pipeline."
            )

        if self.num_samples_per_prompt < 2:
            # A group of one has zero within-group spread, so every advantage is
            # (r - r)/(0 + eps) = 0, the zero-signal filter skips every epoch, and
            # the run prints "training complete" having never stepped the
            # optimizer.
            raise ValueError(
                f"num_samples_per_prompt must be at least 2, got "
                f"{self.num_samples_per_prompt}. A group of one has no within-group "
                "spread, so every advantage is exactly zero and no epoch ever trains."
            )

        if not 0.0 < self.timestep_fraction <= 1.0:
            raise ValueError(
                f"timestep_fraction must be in (0, 1], got {self.timestep_fraction}"
            )
