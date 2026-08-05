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

## 10b. Defects found by asking which trainers had never produced a curve

Every earlier section came from reading code or chasing a symptom. This set came
from a different question: which trainers have ever produced a real held-out
curve? Two had not. DDPO had exactly one eval point, and DiffusionDPO had zero.
Reading DiffusionDPO closely for the first time found three more defects.

**Winner and loser got independent noise draws.** The reference chunks the
concatenated batch and repeats the winner's noise, so both branches see the same
noise and the two Monte-Carlo errors partly cancel. Measured on a fixed pair with
a perturbed copy of the model as the policy:

| | sigmoid argument std | fraction past \|arg\|>8 |
|---|---|---|
| shared noise (reference) | 2.79 | 0.2% |
| independent draws (ours) | 3.65 | 2.5% |

Past `|arg| > 8` logsigmoid is flat and the gradient is gone. With `beta = 5000`
that tail is the part that matters. A first attempt to measure this showed no
difference at all, because it averaged over the batch and all spatial dimensions
before comparing - the effect is per-sample, and per-sample is what enters the
loss.

**Two more trainers still serialized the whole transformer.** `SFTTrainer` and
`DRaFTTrainer` do not inherit `BaseDiffusionTrainer`, so neither picked up the
adapter-only checkpointing fix; both were writing ~5GB of frozen base weights per
checkpoint against the ~26MB that is trainable.

**Two trainers had no held-out metric in the library at all.** DiffusionDPO
reported only its own training loss. SFT's held-out loss existed, but only as a
`HeldOutLoss` callback defined in a cluster run script, so the measurement was
real for that one run and absent for anyone using `SFTTrainer` directly. Both now
report inside the trainer: DiffusionDPO gives preference accuracy (the fraction of
unseen pairs where the implicit reward ranks the winner higher, chance 0.5) plus
the mean margin, and SFT gives a held-out flow-matching loss. In both cases the
pairs, timesteps and noise are pinned, so a move in the number is a move in the
model.

The DPO metric's own first output caught a defect in itself: it reported
`accuracy = 0.000` at step 0. That is correct arithmetic - LoRA's B matrix is
zero-initialized, so the policy and the reference are the same network and every
margin is exactly 0.0, which `margin > 0` scores as a loss. Ties now count as
half, which puts step 0 at exactly 0.500 and doubles as an initialization check.

## 10c. The logger was deleting the results

`CSVLogger` opened both curve files with `"w"`. Every restart therefore truncated
the eval curve and began again at epoch 0. That was harmless while runs completed,
and became severe once the supervisor started resubmitting after segfaults: the
mechanism built to keep runs alive was erasing the measurements it existed to
protect, and no run on an unstable node could ever accumulate the ~8 points these
curves need. Files are now rotated to `<tag>_eval.runN.csv`, so a crash costs
training progress - there is no resume - but never data.

Relatedly, eval ran every 2 epochs, so 8 points needed roughly 4.3 uninterrupted
hours. Eval costs about a third of an epoch here, so moving to every epoch is
~17% more compute to halve the window a usable curve requires.

## 10d. The data anchor trains alone when the reward goes sparse

Found Aug 3 2026, and the only defect since the audit that would silently degrade
a user's run rather than merely mislead a measurement.

`ocr_exact` on FLUX collapsed from an untrained 0.625 to 0.29 across three seeds.
The training reward collapsed with it, 0.75 to 0.08, which rules out overfitting -
the policy was getting worse at the objective it optimises, on its own training
distribution. One log line gives the mechanism:

```
Epoch 5 | reward=0.083 | rl_loss=0.0000 | data_loss=0.4452
        | total_loss=0.0223 | mean_ratio=0.9999
```

`total_loss` matches `data_beta * data_loss` to the printed precision, which
looks like the RL term having gone to zero and the anchor being left as the
entire gradient.

**That reading is wrong, and the rest of this section is retracted.** It said:
`ocr_exact` is binary, so at group 6 all six samples in a group frequently score
the same; the group-normalised advantage then has no within-group variance,
`has_signal` is false, `rl_loss` is set to zero, and the anchor is applied
anyway, compounding as a worse policy makes groups more uniform. Every step of
that is checkable against the group scores the logs already carried, and it does
not survive the check (`report/analyze_flux_anchor.py`):

| | `data_beta=0` | `data_beta=0.05` |
|---|---|---|
| runs | 3 | 3 |
| mean train reward | 0.694 -> 0.694 | 0.694 -> 0.306 |
| epochs where *every* group was flat | 0 / 26 | 0 / 33 |
| `rl_loss` printed as `0.0000` | 92.3% | 92.9% |

Across all six runs, 91% of group means are interior - 0.17, 0.33, 0.5, 0.67,
0.83, which are groups of six splitting their scores - and the count of epochs in
which every group was flat is **zero out of fifty-nine**. `has_signal` thresholds
spread across the whole advantage tensor, so it is false only in such an epoch.
It was true in every logged epoch of both arms.

So the RL term was live throughout, the anchor was never the only thing training
the model, and the gate this audit added would have been open on every one of
these epochs. The single log line above was read as diagnostic when it is not:
`rl_loss` prints `0.0000` in ~92% of epochs in *both* arms, including the three
that came to no harm, and four decimals means `< 0.00005` rather than zero.
`rl_loss` also tracks the mean advantage rather than its spread, and group
normalisation drives that mean to zero regardless.

What stands is the measurement, and it is clean: both arms start at an identical
0.694 and only the anchored one falls, across three seeds each. **The anchor at
0.05 hurts FLUX on this task. Why is not established.**

The gate remains correct code - applying a regulariser to a batch with no
learning signal is wrong wherever it happens, and DDPO had no such guard at all -
but it is defensive rather than the fix for this result, and this document
previously claimed otherwise.

The existing zero-signal guard did not catch it. `BaseDiffusionTrainer` skips an
epoch when fewer than two advantages are nonzero; that is a different condition
from their having no spread, and a group can hold several nonzero but internally
uniform values.

**Fix.** The anchor term is now gated on `has_signal`. Confirmed by prediction
rather than after the fact: the mechanism implies the anchor should matter for
binary rewards and not for graded ones, which is why `data_beta` had made no
measurable difference on counting. Tested directly - `ocr_exact` at
`data_beta=0`, three seeds, everything else matched:

| window | anchor on | anchor off | |
|---|---|---|---|
| K=4 | 0.6112 | 0.6157 | 0.1 SE |
| K=5 | 0.5591 | 0.6284 | 2.0 SE |
| K=6 | 0.5250 | 0.6374 | 3.1 SE |

Identical at K=4 and diverging after, which is the shape the mechanism predicts:
the advantages only vanish once the reward has fallen far enough. All three
unanchored seeds held at baseline with SD 0.006; all three anchored seeds
collapsed. The decline does not bottom out: the longest anchored run reached
0.167 by epoch 15 against an untrained 0.625.

**Correction to the scope.** This was first written as "model-independent - any
sparse or binary reward at a small group size will hit it". That is wrong.
SD3.5 with the identical reward, group size and recipe reaches the condition far
less often, over every matched run on disk rather than the four sampled first:

| model | reward | group | epochs printing `rl_loss=0.0000` | of |
|---|---|---|---|---|
| FLUX.1-dev | `ocr_exact` | 6 | at most 53 | 59 |
| SD3.5-medium | `ocr_exact` | 6 | at most 12 | 135 |

These are upper bounds, not counts. Four decimals means `< 0.00005`: a quiet
epoch does force `rl_loss` to exactly zero, so quiet implies a printed `0.0000`,
but a live epoch with a small value prints it too. `rl_loss` also tracks the mean
advantage rather than the spread, and group normalisation centres advantages, so
it is not the quantity `has_signal` tests. An earlier version of this table read
"0/4, 0/4, 0/5, 0/5" against "12/16, 10/12" and concluded SD3.5 *never* reaches
the condition; on the full set it does, in 12 epochs. The asymmetry is large and
holds in direction - which is what the mechanism argument needs - but it is
evidence from a proxy, not a measurement.

And the outcomes invert with it: on SD3.5 the *anchored* runs rise (0.375 to
0.458-0.500) while an unanchored one falls, the reverse of FLUX.

So the defect is a real code path - applying a regulariser to a batch with no
learning signal is wrong wherever it happens - but reaching it often appears to
require a model whose within-group samples score identically. FLUX at group 6
does so far more than SD3.5, which is the asymmetry above. The fix is still
correct and still worth having; the claim about who will encounter it was too
broad, and how much more often remains bounded rather than measured.

This also points at what is actually different about FLUX: its rollouts at
noise 0.7 are far less diverse than SD3.5's, so a group of six lands on the same
reward. That is a property of the sampler and the model rather than of the
reward, and it suggests exploration noise as the lever - untested, because the
runs that would have tested it were blocked by node contention.

The tell in a log is `total_loss` printing as exactly `data_beta * data_loss`.

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

Three of the fixes above were held in place by nothing but a comment, so they now
have tests. `test_ddrl_monotonic_transform_recentered` asserts the transformed
advantages are zero-mean per group and that the transform preserves the ranking;
`test_buffered_tracker_differs_from_grpo` asserts DDPO's estimator responds to a
batch that beats the prompt's history where GRPO's is centered by construction;
`test_kl_variance_finite_at_sigma_one` asserts the KL divisor stays finite and
non-zero at sigma = 1. Each new assertion was checked against the pre-fix
behaviour to confirm it actually fails there - the uncentered group mean is
-1.415 and the uncapped std is `inf`, so none of the three is vacuous.

## 14. Infrastructure

The sbatch preflight now runs under `timeout` and performs a real allocation and
matmul rather than only `torch.cuda.is_available()`, because a wedged GPU can
pass the latter and then block forever - which cost one 46-minute allocation that
used 3 seconds of CPU. At least one a6000-Ada on atlas29 is in this state.

FLUX.1-dev also hit a second, independent problem on that node, and it took two
wrong guesses to pin down. The symptom was a process crawling inside
`pipeline.to(device)` with RSS growing ~44 MB/min. First guess: contended storage.
A `dd` on the login node returned **244 MB/s**, which seemed to rule that out, so
the second guess was that the wedged GPU was blocking the CUDA copies. Both were
wrong about the location. Reading the *same file* from a job **on atlas29** gives:

| where | sequential read of the same blob |
|---|---|
| login node (sc) | 244 MB/s |
| atlas29 (compute) | **4.7 MB/s** |

A 52x difference, and atlas24 loads SD3.5's ~10 GB in about two minutes, so this
is specific to atlas29's path to `/atlas`, not to the filesystem generally. At
4.7 MB/s a 33 GB model needs roughly two hours of pure reading, and letting
`safetensors` mmap fault the pages in semi-randomly instead measured ~0.7 MB/s,
which is the ~13-hour version. Streaming the blobs once before loading (`WARM` in
the sbatch) converts the second case into the first.

**atlas29 also segfaults runs outright**, inside a transformer forward, at random
points, and this was recorded for a while as FLUX-specific. It is not: `hb005`, an
SD3.5-medium run, crashed the same way in `diffusers/models/normalization.py` with
node memory at 507 of 770 GB. Counting the crashes by node settles what to do
about it without needing a mechanism:

| node | segfaulted | completed cleanly |
|---|---|---|
| atlas29 (a6000-Ada) | 20 | 0 |
| atlas24 (a5000) | 1 | 2 |

An earlier version of this table read `atlas24: 0` and was used to argue the
crash was specific to atlas29. That was wrong: `h_ddpo` (job 16429553) later
segfaulted on atlas24 with the same signature, inside a linear forward. The rate
still differs by more than an order of magnitude, so keeping long runs off
atlas29 remains the right operational call, but **atlas24 is not safe, only
safer**, and the node is not a sufficient explanation.

Three causal explanations were proposed earlier and each was disproved by the next
run (a 48GB ceiling - 256px quarters the memory and crashes the same; OWLv2
co-residency - `reward_device=cpu` crashes the same; a config edge case - a matched
pair diverges). A fourth is not worth proposing without single-variable evidence.
The table is not a mechanism and is not meant as one; it is only the operational
conclusion, which is to keep long runs off atlas29.

**Resume, and the measured before/after.** Until Aug 2 2026 a segfault discarded
every epoch of training, so a run restarted from zero and no experiment could
outlast the mean time between crashes. That is why nothing here ever reached the
twelve epochs a drift measurement needs. `resumable=True` now saves the adapter,
the AdamW moments, the epoch, the advantage tracker's history and the RNG state
after every epoch (113 MB, 0.25s, no measurable slowdown).

| | crashes | curve resets |
|---|---|---|
| before (`b0`/`b3` arms) | 73 | ~70 rotations |
| after (`d0`/`d3` arms) | 5 | **0** |

Three defects surfaced only by testing it on hardware rather than trusting the
unit test. A resumed run re-evaluated the epoch it resumed into, because eval
runs at the top of an epoch and the state is written at the bottom - the curve
came back as 0,1,1 and every later row was offset, so curve writes are now
idempotent per epoch. A stale state could be resumed into a different
configuration, so it carries a fingerprint of the fields that change what is
measured and refuses on mismatch. And a completed run's state left in place made
the next start resume at `num_epochs`, exit immediately and report success having
trained nothing, so it is deleted on completion.

The RNG state was initially left out, on the argument that unequal randomness is
a nuisance shared by every arm. That holds only if arms resume equally often, and
the next section shows they do not.

**A segfault that looked reproducible and was not.** `d3s43` (data_beta=0.3,
seed 43) died six consecutive times in the backward pass of epoch 0, inside
gradient-checkpoint recompute, never surviving to its first state save. Six in a
row at the same line, with a fixed seed replaying the same rollout, read as
deterministic and was written up here as the first reproducible crash in the
project. On the seventh attempt it went straight through and is now training
normally.

So it was a streak, not a mechanism. Worth keeping as a calibration point: at a
background rate near 0.5 crashes per GPU-hour, six identical-looking failures is
unlikely enough to feel like a signal and still be luck, and "it failed the same
way every time" is not evidence of determinism unless the alternative was
actually ruled out. The claim was retracted within the hour; the drift analysis's
exclusion note went with it.

**The anchor costs about 24% throughput, which explains a pattern misattributed
twice.** Across three cohorts the data_beta=0 arms consistently accumulated epochs
faster, and this was attributed first to nothing (a single noisy pair of runs
suggested no pace difference) and then to the crash-rate signal below. Measured
over 81 GPU-hours in the powered cohort:

| arm | epochs | GPU-hours | epochs/GPU-hour |
|---|---|---|---|
| `data_beta`=0 | 101 | 47.6 | 2.12 |
| `data_beta`=0.3 | 55 | 33.8 | 1.62 |

Crash counts in the same cohort were 18 for data_beta=0 against 15 for
data_beta=0.3 - the opposite direction from the earlier signal, so crashes explain
neither the lag nor themselves. The cause is ordinary: `data_beta > 0` adds a
forward and backward pass on anchor latents at every timestep, so an epoch is
simply more work. Worth knowing as a planning number - turning the anchor on costs
roughly a quarter of throughput - and worth remembering as an instance of
reaching for an exotic explanation twice before measuring the obvious one.

**The configuration-dependent signal did not replicate.** The finding below was
recorded post-hoc and flagged as needing a declared test. That test was written
before the replication cohort's crash counts were read, with exposure in
GPU-seconds, a threshold of |z| > 2 and a 30-crash minimum
(`report/analyze_crashrate.py`). On the closed drift cohort it reads **z = 1.3,
not replicated**. The original z = 2.2 was inflated by having been found after
noticing the pattern, which is what the pre-registration existed to detect. The
standing conclusion stands unchanged: segfaults here are intermittent and no
configuration has been shown to cause them. What follows is kept for provenance,
not as a live claim.

**A configuration-dependent signal, found Aug 2 2026 (SUPERSEDED - see above).** The seeded sweep is the
first setup here with matched arms differing in exactly one setting, and the
segfaults are not distributed evenly across them:

| arm | GPU-hours | crashes | crashes/GPU-hour | runs reaching 12 epochs |
|---|---|---|---|---|
| `data_beta`=0 | 82.6 | 34 | 0.412 | 4 of 4 |
| `data_beta`=0.3 | 64.4 | 44 | 0.683 | 0 of 4 |

A 1.66x rate difference, z = 2.2 (p ~ 0.025) against equal rates given the
exposure, and the 4-vs-0 split on run length is p ~ 0.029 by Fisher. The arms are
matched in everything else and the seeds are crossed, so this is single-variable
in a way the three earlier hypotheses were not - none of them varied `data_beta`.

Two reasons not to promote it to a cause yet. It is post-hoc: the run-length gap
was noticed first and the rate computed afterwards, which is the forking-paths
setup, so the p-values are optimistic. And the obvious mechanism does not
implicate the anchor as such - `data_beta > 0` adds a forward and backward on
anchor latents at every timestep, so those runs execute more kernels and
allocations per GPU-hour, and the right exposure unit may be kernel launches
rather than wall-clock. Under that reading the anchor is not special, it is just
doing more work in the same time.

The test that would settle it is a pre-registered one: matched runs differing
only in `data_beta`, crashes counted per unit of GPU work rather than per hour.
Recorded here so the earlier "not configuration-dependent" line is read as what it
was - true of the variables tested then, not of this one.

Moving off it is not currently possible for these runs. The a5000 node is stable
but full, and the 16GB a4000s - 28 of them free - cannot hold an SD3.5 rollout at
group 12 and 512px. Two probes OOMed with **byte-identical** figures at
`train_batch_size` 3 and 1 (15.41 GiB in use, 512 MiB requested, 186.56 MiB free),
which places the failure before the replay minibatch loop and rules out the batch
size as the lever. Fitting a4000 would mean changing the group size or the
resolution, either of which changes what is being measured.

Two lessons worth keeping. "Slow" and "wedged" look identical through RSS growth,
and a real allocation under `timeout` separates them. And a throughput number is
only meaningful from the node that will do the reading - measuring on the login
node produced a confidently wrong conclusion twice. Separately, importing `torchvision.ops` /
`torch._dynamo` off this NFS can take five minutes or more under load, so an
empty log early in a job is not evidence of a hang; `faulthandler.dump_traceback_later`
is the way to tell the two apart.
