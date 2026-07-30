# Audit, July 2026

A full pass over the RL math, the trainers, and the eval path, checking the code
against the theory documents and the reference implementations rather than
against its own tests. Ten defects, listed worst first. Every claim below is
either a measurement on an A5000 or a check you can run locally.

The short version: the SDE sampler had a singularity at the first timestep that
silently destroyed rollouts, and the eval never used a fixed seed. Between them
those two explain nearly every confusing result this project has produced, and
they invalidate a number of earlier conclusions (§11).

---

## 1. The SDE step was singular at sigma = 1

`sde_step_with_logprob` uses `std_dev_t = sqrt(sigma / (1 - sigma)) * noise_level`
as the diffusion coefficient. The SD3.5 and FLUX schedules start at sigma
*exactly* 1.0 - verified directly:

```
set_timesteps(16/20/28) -> sigmas[0] == 1.000000
```

and SD3's shift maps 1.0 to 1.0, so no shift value avoids it. A
`clamp(max=0.9999)` kept the value finite but left
`correction_coeff = noise^2 / (2 (1 - sigma))` enormous. The resulting
coefficient on `sample` in the Euler step:

| noise_level | coefficient at step 0 | every other step |
|---|---|---|
| 0.1 | -0.160 | 0.73 - 1.00 |
| 0.7 | **-55.834** | 0.73 - 1.00 |
| 1.0 | -114.988 | 0.73 - 1.00 |

Step 0 was the sole pathology, and it both amplified and *sign-flipped* the
latent.

The algebra itself is correct. Our mean is
`x + dt [v + (std^2 / 2 sigma)(x + (1 - sigma) v)]`, which is the reverse SDE for
rectified flow with score `grad log p_t = -(x + (1 - sigma) v) / sigma`; it also
matches this repo's `NOISE_SCHEDULES_AND_LOSS.md:424-442` and FlowGRPO's
reference. The score is finite at sigma = 1 (`-(x + 0)/1 = -x`). Only the
diffusion coefficient diverges, so the singularity is inherited from the
FlowGRPO formulation rather than mistranscribed.

**Measured impact** (A5000, OCR reward, 20 steps, group 8, one prompt):

| variant | noise | final latent std | reward mean | reward std | ratio |
|---|---|---|---|---|---|
| uncapped | 0.7 | 10.259 | **0.000** | **0.000** | 1.0000 (degenerate) |
| uncapped | 0.1 | 1.371 | 0.479 | 0.259 | 0.9420 |
| capped | 0.7 | 1.052 | 0.417 | 0.454 | **0.9993** |
| capped | 0.1 | 1.076 | 0.375 | 0.518 | 0.9721 |

At noise 0.7 the rollout reward was *exactly* zero with *exactly* zero
within-group spread. Since the GRPO advantage is
`(r - group_mean) / (group_std + eps)`, zero spread means zero advantage, and the
zero-signal filter then skipped the epoch outright. Every OCR rollout ever run at
noise 0.7 was in this state.

Eval never revealed it: eval samples at `noise_level = 0`, where `std_dev_t` is 0
and the term vanishes entirely. That is why eval curves and training never lined
up - a noise-0 eval of the same prompts scored 0.569 while the rollout scored
0.005.

**Fix.** Cap the coefficient so the Euler step stays contractive:
`std_dev_t = min(std_dev_t, sqrt(sigma / -dt))`, which bounds
`|correction_coeff * dt| <= 1/2` and so keeps the coefficient on `sample` in
(0, 1]. On the real schedules this binds only at step 0. After the fix the step-0
coefficient is +0.500 at every noise level, and the cumulative product over the
schedule goes from -16.91 to +0.15.

Confirmed in live training, not just in isolation: `mean_ratio = 0.9977` at epoch
0 of a real FlowGRPO run.

## 2. Eval measured noise, not the policy

`_evaluate` did this:

```python
eval_prompts = self.train_prompts[:4]
```

Three separate problems. It was fixed at four prompts and not configurable; the
prompts came from the *training* set, so nothing was held out; and no `generator`
was passed to the sampler. `torch.manual_seed(config.seed)` runs once at
construction, so eval drew whatever global RNG state training happened to leave
behind - different latents every epoch.

That is the +/-0.5 epoch-to-epoch swing visible in every curve this project has
produced. The curves were not noisy measurements of a trend; they were noise.

**Fix.** Added `eval_prompts`, `num_eval_prompts`, `num_eval_samples_per_prompt`
and `eval_seed`; eval now builds a dedicated
`torch.Generator(device).manual_seed(eval_seed)` so every epoch evaluates
identical latents and the curve moves only when the policy moves. Eval sampling
is chunked, because a held-out set is far larger than a rollout batch and OOMs the
VAE decode in one call. Also reports the standard error and n.

## 3. The KL terms divided by a variance the policy never used

`sde_step_with_logprob` was not the only place the diffusion coefficient was
derived. FlowGRPO and DDPO each re-derived it inline for their KL penalty, and
neither was capped, so at sigma = 1 they divided by a variance about 1e4 too
large and the KL silently vanished on the first step. After fixing §1 they were
also inconsistent with the variance the policy actually sampled from, which is
the quantity a KL between transition distributions has to use.

**Fix.** Extracted `sde_diffusion_std()` as the single source of truth; all three
call sites use it.

## 4. The flow-matching loss only ever spoke SD3

`compute_flow_matching_loss` called the transformer with SD3's signature: no
`img_ids`, no `txt_ids`, no `guidance`, unpacked spatial latents, and
`timestep = sigma * 1000`. FLUX needs packed `(B, tokens, 64)` latents, rotary
position ids, a guidance scalar on dev, and reads the timestep as sigma directly.

That loss is DDRL's data-anchor term and SFT's entire objective. So **DDRL with
`data_beta > 0` and SFT could not run on FLUX at all** - not "untested",
unimplemented. This would have taken out 6 of the 16 cells of a FLUX beta grid.

**Fix.** Added a flux branch and threaded `model_type` through both call sites.
Verified shapes at 512px: `hidden (B, 1024, 64)`, `img_ids (B, 1024, 3)`.

## 5. FLUX.1-dev could not start

```
ValueError: `mu` must be passed when `use_dynamic_shifting` is set to be `True`
  flux_patch.py:126  pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
```

FLUX.1-dev configures its scheduler with `use_dynamic_shifting=True`, a
resolution-dependent timestep shift that requires `mu`. FLUX.1-schnell leaves it
off, and schnell was the only FLUX model ever run, so this was invisible. Any
claim that "the FLUX path works" was a claim about schnell - a 4-step distilled
model that is a poor target for SDE-based RL, and whose one aesthetic run
declined (6.06 -> 5.11 -> 5.81).

**Fix.** Compute `mu` from the token count via `calculate_shift`, with a fallback
to the closed form.

## 6. DDPO was FlowGRPO wearing a different config

`ddpo_config.py` claimed:

> On flow matching models (SD3, FLUX), DDPO reduces to FlowGRPO without the
> GRPO-specific enhancements (no group normalization, no guard).

`DDPOTrainer` never overrode `_compute_advantages`, so it inherited the base
within-batch group-normalized tracker. Our DDPO *was* FlowGRPO with
`clip_range=0.2`.

The claim is also backwards historically. `PerPromptStatTracker` comes from DDPO
(Black et al., 2023); the real one keeps a rolling per-prompt reward history with
a `min_count` fallback to batch statistics. The stateless within-batch variant is
the GRPO one.

The difference is not cosmetic. With one sample per prompt:

```
rewards      [0.98, 0.19, 0.39, 0.43]
grpo  adv    [0.0, 0.0, 0.0, 0.0]          <- no within-prompt group exists
ddpo  adv    [1.643, -1.058, -0.356, -0.229]
```

The all-zero case trips the zero-signal filter and the epoch is skipped entirely.

**Fix.** Added `BufferedPerPromptStatTracker` with the reference semantics
(population std, buffer, `min_count` fallback) and used it in DDPO. Confirmed
distinct in live training: DDPO's `rl_loss` is -0.0005 where FlowGRPO's and
DDRL's are +0.0005/+0.0004 on the same epoch.

**Any prior DDPO number is a FlowGRPO number.**

## 7. DDRL's monotonic transform discouraged every action

`lambda(x) = -exp(-x)` (Theorem 3.1) is a legitimate monotonic, risk-averse
reweighting, but its range is (-inf, 0). Applied to a group-normalized zero-mean
advantage, *every* transformed advantage comes out negative:

```
A            [-2.0, -1.0,  0.0,  1.0,  2.0]
-exp(-A)     [-7.389, -2.718, -1.0, -0.368, -0.135]   mean -2.322
```

In `-A * ratio` that makes every sampled action a discouraged one, so the policy
walks away from its own distribution instead of re-weighting within it, and the
unbounded negative side makes it violent.

Why the theorem does not have this problem: it maximizes J over `p_theta` as a
distribution, so `integral p_theta = 1` is present and carried by the multiplier
mu, which is what makes an everywhere-negative lambda harmless - the stationary
condition still yields the normalized
`p_theta = p_tilde_ref / (mu + exp(-(r - Z)/beta))`. The PPO surrogate enforces no
such constraint.

**Fix.** Re-center per prompt group. For any action-independent baseline b,
`E[(A - b) grad log pi] = E[A grad log pi]` because `E[grad log pi] = 0`, so this
leaves the policy gradient unbiased while restoring the zero-mean advantage the
surrogate needs, and lambda's monotonicity means the ranking is preserved exactly:

```
recentered   [-5.067, -0.396, 1.322, 1.954, 2.187]    mean 0.0, ranking identical
```

Also worth recording: the theorem applies lambda to `(r - Z)/beta` with
`Z = beta log E_p_ref[exp(r/beta)]` the log-partition function. The
implementation substitutes the group-standardized advantage, i.e. approximates Z
by the group mean and beta by `beta_temp * std`. That substitution is a practical
choice, not part of the theorem. `DDRL_THEORY.md` now records both points.

## 8. DiffusionDPO was running at twice the intended beta

The reference implementation uses
`inside_term = -0.5 * beta_dpo * (model_diff - ref_diff)`. Ours omitted the 0.5,
so the paper's published `beta = 5000` behaved like 10000. The docstring also
advertised a `T` factor the code never applied.

**Fix.** Restored the 0.5 and corrected the docstring; `beta` absorbs T, which is
what its large default reflects.

## 9. OCR prompts were at the wrong end of the scale

The prompt set was 3-letter words (`"CAT"`, `"DOG"`, `"SUN"`). Difficulty was
then calibrated against the base model properly - fixed seed, noise 0, 28 steps,
8 samples per prompt - rather than guessed:

| tier | base mean | within-group std |
|---|---|---|
| 3 char | 0.319 | 0.470 |
| 4-5 char | 0.585 | 0.457 |
| 6 char | 0.549 | 0.468 |
| 8 char | 0.422 | 0.436 |
| two words | 0.569 | 0.443 |

Two things fall out. The reward is length-normalized edit distance, so one wrong
character costs 1/3 on "CAT" but 1/6 on "BAKERY" - longer targets score *higher*,
not lower. And no tier is saturated: the old grid's report of OCR base = 1.00 is
not reproducible under a fixed-seed measurement, which is consistent with §2.

**Fix.** Prompt sets now have disjoint train/eval splits (64/22 for OCR, 22/12 for
counting) with disjoint target strings, so eval cannot leak train targets.

## 10. The OCR recognizer ran on CPU

`easyocr.Reader(..., gpu=False)` was hardcoded. A rollout scores
`num_prompts * group_size` images every epoch plus the held-out eval set. The
recognizer's weights are small next to the transformer, so there was no reason
for it.

---

## 11. Conclusions this invalidates

Do not reuse these:

- *"The importance ratio is ~0.92 because of bf16 rounding, gradient
  checkpointing, or CFG."* It was §1. Capped at noise 0.7 the ratio is 0.9993.
  CFG was separately ruled out by measurement (0.92 on vs 0.94 off) and is
  consistent by construction: sampling encodes `negative_prompt=[""]`, stores
  those embeddings, and replay reuses them, so both branches see identical
  conditioning.
- *"noise_level 0.1 gives a better reward signal than 0.7."* Artifact of §1: 0.1
  produced a survivable -0.16 step-0 coefficient while 0.7 produced a fatal
  -55.8.
- *"Detection rewards need noise 0.1, OCR needs 0.7, and neither works well."*
  The tradeoff was an artifact. Capped noise 0.7 gives both a clean ratio and the
  largest within-group spread.
- *"DDRL collapses without a sufficient data anchor; FlowGRPO's KL is more
  robust."* That was §7, the un-centered transform, not a property of the anchor.
- *"DDPO is genuinely weaker than GRPO because it lacks a group-relative
  advantage."* It had the identical estimator (§6).
- *"The FLUX path is validated."* Only schnell, and §4 and §5 mean FLUX.1-dev
  could not run and FLUX + DDRL/SFT was unimplemented.
- Any eval curve produced before this audit, including the "corrected" c2_* grid
  (§2).

## 12. Known deviations, deliberate

Recorded so they are not rediscovered as bugs:

- **Log-prob is mean-reduced, not summed.** The true Gaussian log-prob sums over
  D = 65,536 dimensions; `exp()` of that overflows. `LOG_PROB_DERIVATIONS.md:206`
  documents the choice and FlowGRPO does the same. Consequence: the "ratio" is a
  per-dimension geometric mean, so `clip_range` operates on a compressed
  quantity - which is why 1e-3 is a sane value here.
- **`torch.std` is Bessel-corrected**, FlowGRPO's numpy is not. A 4.5% advantage
  scale difference at group 12, largely absorbed by AdamW.
- **The count reward is graded**, `max(0, 1 - |c - n| / max(n, 1))`, where
  GenEval's is binary. Smoother for RL, but our counting numbers are not directly
  comparable to GenEval-reported figures.

## 13. Tests added

`test_sde_step_at_sigma_one` asserts the step-0 coefficient stays in (0, 1] at
noise 0.1/0.7/1.0. The pre-existing `test_sde_step` used sigma = 0.8 and passed
happily, which is how §1 survived an earlier multi-agent audit.

`test_trajectory_shuffled_indexing_alignment` checks that a shuffled mini-batch
keeps tensor fields aligned with the prompts list. A misalignment there pairs each
trajectory with another sample's advantage, which trains without error and learns
nothing.

## 14. Infrastructure

The sbatch preflight now runs under `timeout` and performs a real allocation and
matmul rather than only `torch.cuda.is_available()`, because a wedged GPU can
pass the latter and then block forever - which cost one 46-minute allocation that
used 3 seconds of CPU. At least one a6000-Ada on atlas29 is in this state, and it
is worth naming the wrong turn it caused: the symptom was a process crawling
inside `pipeline.to(device)` with RSS growing ~44 MB/min, which looked exactly
like `safetensors` mmap faulting pages off a contended NFS. Measuring settled it -
sequential reads of the same blobs run at **244 MB/s**, so storage was never
involved and the slow crawl was CUDA calls blocking on the bad GPU. The lesson is
that "slow" and "wedged" present identically through RSS, and one `dd` separates
them. Separately, importing `torchvision.ops` /
`torch._dynamo` off this NFS can take five minutes or more under load, so an
empty log early in a job is not evidence of a hang; `faulthandler.dump_traceback_later`
is the way to tell the two apart.
