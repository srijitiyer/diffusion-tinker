# Validation status

Convergence runs on the Stanford cluster (SD3.5-Medium unless noted), not just
smoke tests. "Validated" = the intended objective actually moves in a real run.

> **Rewritten July 2026 after the audit in `AUDIT.md`.** Two defects invalidated
> most of the previous contents: the SDE sampler was singular at its first
> timestep, which drove rollout reward and within-group spread to *exactly* zero
> at noise 0.7, and eval used four training prompts with no fixed seed, so every
> curve here was measuring resampling noise. **Treat any pre-audit number as
> unverified**, including the grid figure. Rows below are either re-measured on
> fixed code or explicitly marked as pending.

## Trainers

| Trainer | Status | Evidence |
|---------|--------|----------|
| FlowGRPO | **validated** | held-out OCR 0.300±0.064 (untrained) → 0.494 → 0.702 → 0.640 → 0.847 → 0.757 (ep2..ep10), n=44. Climbs with oscillation, no collapse through epoch 10. ratio 0.9977-0.9995 |
| DDRL (β=0) | **validated** | held-out OCR → 0.569 at epoch 2, train reward 0.130→0.146→0.315, ratio 0.9986. Identical to FlowGRPO(β=0) by construction, see below |
| DDPO | **validated** | held-out OCR → 0.568 at epoch 2, train reward 0.130→0.399→0.530, ratio 0.9953. Now uses its own buffered per-prompt estimator |
| DDRL (anchor, transform off) | **validated: gradual climb, no collapse** | The canonical configuration, eleven evals each against the 0.300 baseline. beta=0.01 reaches **0.815**, beta=0.05 reaches **0.860**. Both oscillate on the way up and neither collapses through epoch 20. Anchor active throughout, ratio ~0.999. This is the gradual increase at or above FlowGRPO's level that Haotian described for a small beta |
| DDRL (anchor + monotonic transform) | **opt-in flag, off by default** | 0.300 → 0.482 → 0.893±0.036 → 0.892 → 0.873 → 0.851 (ep2..ep10), n=44, anchor active (`data_loss` ~0.268), ratio 0.9993. Climbs then holds past epoch 10, where the old collapse appeared. Kept as evidence the re-centering fix works, not as a headline result |
| DRaFT | **validated on a held-out set for the first time** | 5.440 → 5.718 across 13 evals. The pre-audit 5.9→9.6 was **train** reward: DRaFTTrainer does not inherit BaseDiffusionTrainer, so it never called _evaluate and only ever reported the reward it directly optimizes on the prompts it optimizes. It now has a matching held-out eval |
| SFT | **validated, and it overfits; eval moved into the library Aug 1** | held-out flow-matching loss on 16 unseen examples at fixed noise and timesteps: 0.21184 → **minimum 0.19742** (−6.8%) → 0.20035 by the end, so it turns over and gives back 1.5%. The pre-audit note only said the training loss converges, which a denoising objective does by construction; the turnover is invisible without a held-out set and is an argument for `early_stop_patience`, which the config supports and this run did not use. That held-out loss came from a callback defined in the cluster run script, so it was real for that run and absent for anyone calling `SFTTrainer` directly; `holdout_size`/`eval_every_steps` now do it inside the trainer, and `s_sft3` is queued to confirm the built-in path reproduces it |
| DiffusionDPO | **runs, separates one-sidedly; held-out metric added Aug 1** | the blocker was not HF gating: yuvalkirstain/pickapic_v1 and v2 both 404 now, so accepting terms would not have helped. `kashif/pickascore` is an ungated mirror with the same schema. Two real defects also had to be fixed first: the missing 0.5 in the loss (paper beta=5000 was behaving like 10000), and PreferenceDataset assuming PIL images where Pick-a-Pic stores raw jpeg bytes. 2000 steps, EXIT 0, peak 16.8 GB. Over 40 logged steps the separation is asymmetric: the loser's loss rises in 34/40 and the implicit reward is positive in 27/40, but the winner's loss falls in only 15/40, below chance. So it learns the preference mainly by suppressing the loser rather than by favouring the winner. That run had no held-out metric at all - only its own training loss, the same failure DRaFT had. It now reports preference accuracy on unseen pairs plus the mean margin, and a re-run with it is in flight. Two further defects were found in the same pass: it drew independent noise for winner and loser where the reference shares one draw, and it still checkpointed the whole transformer |

**FlowGRPO(β=0) ≡ DDRL(β=0).** Not approximately: the base advantage clamps the
group-normalized reward to ±5, and DDRL's reduces to exactly that when
`beta_temp=1` and the transform is off. At epoch 0 from the same seed both picked
the same prompt subset in the same order and produced identical per-prompt
rewards, mean reward (0.1302), mean advantage (0), and ratio (0.9977). Only the
printed loss differs in the 4th decimal, which is bf16 GEMM nondeterminism across
processes.

**On the monotonic transform: do not use it, and do not report it.** Haotian said
in person (July 27) that they do not use it, because of the negative exponential.
That matches what it does numerically: lambda(A) = -exp(-A) has range (-inf, 0),
so applied to a zero-mean advantage every value is negative and the PPO term
`-A * ratio` discourages every sampled action. His email framing requires it off
too, since "if the data anchor beta decreases to 0, it's exactly FlowGRPO without
KL divergence" is false with lambda applied.

So the canonical DDRL is **anchor on, transform off**, which is also the library
default (`use_monotonic_transform=False`). Any DDRL number we report should be
from that configuration.

The re-centering is kept as a bug fix, because the flag exists and un-centered it
collapses the policy, but the run below is **not a result and should not be
quoted as one**. In particular, disregard any framing of it as the best
configuration: it reached 0.893 against plain FlowGRPO's 0.702, which is
explicable (λ is risk-averse, penalizing low advantages far more than it rewards high ones, so
once it is centered (and therefore a valid policy-gradient weight) it concentrates
updates on avoiding bad samples, while the forward-KL data anchor keeps the policy
from drifting off the data manifold. Un-centered it did the opposite: it pushed
down every sampled action.

**DDPO is now a different algorithm from FlowGRPO.** It previously never overrode
`_compute_advantages`, so it inherited GRPO's within-batch tracker and *was*
FlowGRPO with `clip_range=0.2`. The distinction is visible in the logs: FlowGRPO
and DDRL report mean advantage exactly 0 (group normalization forces it), DDPO
reports 0.089 then 0.101 (it normalizes against a per-prompt history). **Any
pre-audit DDPO number is a FlowGRPO number.**

## Rewards

| Reward | Climbs? | Notes |
|--------|---------|-------|
| OCR | **yes, 0.300 → 0.702** | held-out, fixed seed, n=44; ~4.7σ at epoch 4 |
| aesthetic (FlowGRPO) | **yes, 5.651 → 6.063** | eleven evals, near-monotonic. The pre-audit note called this flat and blamed RL being weak on a smooth reward. That was the zero-signal rollouts, not the reward |
| counting (OWLv2) | **yes, 0.561 → 0.679** | at lr 1e-4 it dips to 0.522 around epoch 10 before recovering. At lr 3e-5 it climbs smoothly instead: 0.561 → 0.589 → 0.631 → 0.638 → 0.642, no dip. Run at noise 0.7 throughout |
| CLIP-score | **modest rise, oscillates** | 28.18 (untrained) → peak 28.94 → settles ~28.7. Net above baseline but the swing is comparable to the gain |
| PickScore | **no, flat through epoch 12** | seven evals, 22.85 to 23.05, all within one SEM (0.23) of the 23.03 baseline. Retired as conclusive |
| hps_v2 | **no, flat** | eleven evals, 0.293 to 0.302 against a 0.297 baseline. A drift of 0.004 over the whole run is not a climb |
| aesthetic (DRaFT) | **yes, 5.440 → 5.718 held-out** | 13 evals, near-monotonic, n=24 at 256px. Peak memory 17.7 GB after the text-encoder fix, having previously OOMed at 23.19 GB on the same card. Note 256px, so not directly comparable to the 512px FlowGRPO aesthetic run |

**The headroom thesis survives in its sharper form, and both my retraction and the original wording were partly wrong.** The
original claim was that RL raises a reward when the task has headroom and stays
flat when the base model is already near the reward's ceiling. I retracted the
"flat" half during the audit, on the grounds that those runs had zero-signal
rollouts so flatness proved nothing. That reasoning was right, but the conclusion
was not: re-run on fixed code with rollouts that demonstrably carry signal
(ratio ~0.999, non-zero within-group spread), PickScore is still flat across
seven evals and HPS across four, while OCR and CLIP-score climb clearly. So the
ceiling explanation now stands on evidence rather than on an artifact.

Two corrections to my own earlier readings here. Counting does not decay below
baseline; it dips around epoch 10 at lr 1e-4 and then recovers to 0.679, and at
lr 3e-5 it climbs smoothly with no dip at all, so the dip was an lr artifact
rather than a property of the reward. And aesthetic under FlowGRPO climbs
(5.651 → 6.063 over eleven evals), which contradicts the long-standing claim that
FlowGRPO is too weak for a smooth reward and only DRaFT works on it. What the
sharper thesis says is that a reward moves when the base model has headroom on
it, and that the two preference models (PickScore, HPS) are the ones genuinely at
their ceiling, not smooth rewards as a class.

## Models

| Model | Status |
|-------|--------|
| SD3.5-Medium | **validated** on fixed code (table above) |
| FLUX.1-dev + DDRL anchor | **mechanically correct; held-out degrades on saturated tasks** | Mechanism confirmed: `data_loss` 0.42-0.46 from the FLUX branch of `compute_flow_matching_loss`, ratio 0.9998-0.9999, mean advantage exactly 0, zero skips. Held-out behaviour on the saturated tasks is oscillation rather than a climb: counting runs 0.667, 0.663, 0.667, 0.651, 0.618, 0.580, 0.681 - it dips well below baseline and then recovers above it, so an earlier reading of this as monotonic overfitting was premature and is withdrawn. Exact-match OCR dropped 0.625 → 0.375 but has only two points. Edit-distance OCR is flat at 0.792 → 0.787 while its train reward goes 0.167 → 0.800. What is solid is that none of these tasks *climbs* on FLUX, and that the train/held-out gap is only visible because eval is held out at a fixed seed - the pre-audit eval used four *training* prompts with a drifting seed. FLUX starts near its ceiling on each task (counting 0.667 vs SD3.5's 0.561 untrained, OCR 0.792 vs 0.300), so there is little to gain and room to lose. Segfaults on atlas29 are intermittent and not configuration-dependent |
| FLUX.1-dev + high-count task | **has headroom; rollout coherence matters more than noise** | FLUX scores 0.667 untrained on 2-5 objects but only **0.507** on 6-9, so counting above five is where it is genuinely weak; every earlier FLUX task was saturated (OCR 0.792, exact-match OCR 0.625, counting 0.667). A second finding came out of running it: at 20 rollout steps the train reward sat at **0.083** against a 0.507 eval, because the rollout images were too incoherent for OWLv2 to count 6-9 objects. Matching the rollout to the eval schedule (28 steps) raised it to **0.381**, a 4.6x gain in usable signal with nothing else changed. The old note that detection rewards need low noise had the wrong lever - it is **step count**, and with the sigma=1 fix in place noise 0.7 is fine. Curve incomplete: atlas29 segfaults killed the run four times at different points |
| FLUX.1-dev | **validated end to end.** Four epochs on one 48GB a6000-Ada, `EXIT 0`: reward 0.312 → 0.806 → 0.750, ratio 0.9992 → 0.9998, 26.1M LoRA params, **peak 36.5 GB** so ~11.5 GB headroom at 512px group 6. The ratio matters twice over: FLUX has a different schedule (dynamic shifting, `mu`), so ~0.999 shows the sigma=1 fix is not SD3-specific. Getting there needed two workarounds, neither in the code: a wedged a6000-Ada on atlas29 (passes `torch.cuda.is_available()`, then hangs on the first allocation; the preflight now catches it in 2 min and a driver retries), and that node reading `/atlas` at **4.7 MB/s** against 244 MB/s from the login node. Streaming the 32GB once took `real 133m20s`, after which the model loaded from page cache in about a minute, versus the ~13h the mmap page-fault path was on track for |
| FLUX.1-schnell | runs, but a 4-step distilled model is a poor SDE-RL target; its one aesthetic run declined (6.06→5.11→5.81) |

FLUX also could not run DDRL with an anchor or SFT at all, because
`compute_flow_matching_loss` spoke only SD3 (no `img_ids`/`txt_ids`/`guidance`,
unpacked latents, `timestep = sigma*1000`). Fixed and shape-verified.

## Anchor strength (beta)

Two questions, answered separately, and the order they are written in below is the
order they were asked rather than the order they matter:

1. **Does beta change held-out reward?** No, over six epochs, bounded under ~0.03.
   Settled by a four-seeds-per-arm design after a first attempt was shown
   incapable of detecting anything. Sections: *Result: beta does not separate*,
   *Seeded design: final*.
2. **Does beta change degradation over a longer horizon?** A 2.2 SE signal that
   runs *against* the anchor, with four named weaknesses and a replication in
   flight. Section: *Drift result*.

The first is finished. The second is not, and is written up so that a reader can
see exactly what would overturn it.

Haotian asked for beta 0, 0.01 and upward. An earlier grid (`t_grid2.py`, the
`c2_*` curves) did sweep both `kl_beta` and `data_beta`, so the sweep existed -
but it passed no `eval_prompts` and no `eval_seed`, so it scored the first four
*training* prompts, and it ran on counting 2-4 objects where the base model is
already at 0.792 and every arm saturates at 1.000. It is retracted for those two
reasons rather than for the sweep being absent.

The replacement runs on counting 6-9 objects, the one task with measured headroom,
with everything matched except the regularizer weight: `hb0` (data_beta 0, which
is FlowGRPO), `hb001` (0.01), `hb005` (0.05), `hb03` (0.3), and `hk001` (FlowGRPO
reverse-KL beta 0.01). Held-out prompts, fixed eval seed, 12 prompts x 2 samples.

Two results are already in and do not need the curves.

**Eval determinism, tested the hard way.** All four DDRL arms reported an epoch-0
held-out reward of 0.541 +/- 0.071 (n=24), identical to three decimals across four
independent processes - same seed, same held-out set, four separate jobs, rather
than the claim being asserted from the code. `hb0` and `hb03` also report an
identical epoch-0 train reward (0.532) with identical per-prompt rewards, so the
arms are genuinely matched and can only diverge through the regularizer.

**Both regularizer terms demonstrably fire**, which is what makes a beta sweep
mean anything:

| arm | term | logged |
|---|---|---|
| `hk001` FlowGRPO, `kl_beta=0.01` | reverse KL to the frozen reference | `kl_loss` 0.0009, 0.0013 |
| `hb0` DDRL, `data_beta=0` | forward-KL data term | `data_loss` 0.0000 (off) |
| `hb03` DDRL, `data_beta=0.3` | forward-KL data term | `data_loss` 0.2775 |

`hb03`'s `total_loss` of 0.0835 reconciles exactly with
`rl_loss + beta * data_loss = 0.0003 + 0.3 * 0.2775`, so the weight is applied
rather than only reported.

**Both new held-out metrics validated on hardware.** DiffusionDPO reports at step
0, before any update:

```
Eval (step 0): preference accuracy=0.500 mean_margin=+0.00000 on 100 held-out pairs
```

`mean_margin` being *exactly* zero is a stronger check than the accuracy. LoRA's
B matrix is zero-initialized, so the policy and the adapter-disabled reference
must be the same network and every pair must score an exact tie. Any nonzero
value there would mean the reference path is leaking - the adapter failing to
disable, or the two branches seeing different noise. It is the initialization
test the trainer never had.

SFT's built-in held-out loss (16 examples held back, fixed noise and timesteps):

| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|
| held-out loss | 0.26583 | 0.25435 | 0.25398 | 0.25277 | 0.25291 | 0.25315 | 0.25219 |

Falls then flattens, which is the shape the run-script callback showed
(0.21184 -> 0.19742). The absolute values differ because the holdout is now the
dataset tail with an even spread of sigmas rather than whatever the callback
sampled, so the two are not comparable in level - only in behaviour.

**DiffusionDPO gains little or nothing on held-out preference**, now that it can
be measured. Accuracy over 11 evals: 0.500, 0.550, 0.580, 0.610, 0.530, 0.590,
0.580, 0.520, 0.520, 0.510, 0.540 - a peak of 0.610 at step 600, then a drift back
to 0.540, mean 0.553 against chance 0.500.

The tempting summary is "mean 0.553 with a standard error of 0.016 over ten
points, so clearly above chance". That is wrong. The 100 held-out pairs and the
noise draws are *fixed across evals* - deliberately, so the curve moves only when
the model does. Repeated evals therefore do not resample the pairs, and the
binomial error from having only 100 of them does not average down. The right
uncertainty is 0.05, so the peak is ~2 SE above chance and the mean ~1 SE.
Suggestive, not established.

That is consistent with what the training logs already showed: the loser's loss
rises far more reliably than the winner's falls, so the objective separates the
pair mainly by pushing one side down.

**Baselines are only comparable within a GPU model.** `h_ddpo` on an a5000 starts
at 0.499 +/- 0.073 where the atlas29 a6000-Ada arms start at 0.541, same seed and
same held-out prompts. That is within one SEM, but the four same-node arms agreed
to three decimals while this differs in the second, so the variable is bf16 GEMM
kernels differing by architecture. Arms must therefore stay on one node type.

### The anchor helps when the RL term is live, and only then

The beta work found no effect of `data_beta` on counting, at two horizons and up
to eight seeds per arm. That turns out to be a statement about the reward, not
about the anchor. On a **sparse** reward the anchor does something measurable,
and its sign depends on whether the policy-gradient term is contributing at all.

SD3.5 with `ocr_exact` (untrained 0.375, group 6, two seeds per arm):

| | K=5 mean | latest points |
|---|---|---|
| anchor 0.05 | 0.4582 | 0.417, 0.500 |
| anchor 0 | 0.3491 | 0.292, 0.417 |

**Extended to five seeds per arm**, because at two the SE was set by between-seed
spread and swung 1.8-2.6 with the analysis window - the same underpowered shape
that produced this project's withdrawn findings:

| window | anchored | unanchored | difference | |
|---|---|---|---|---|
| K=4 | 0.4722 (n=5) | 0.3835 (n=5) | +0.0887 | 3.9 SE |
| K=5 | 0.4791 (n=4) | 0.3834 (n=5) | +0.0957 | 3.2 SE |
| K=6 | 0.4861 (n=3) | 0.3771 (n=4) | +0.1089 | 2.4 SE |

Per-seed at K=4 - anchored 0.444, 0.445, 0.472, 0.472, 0.528 against unanchored
0.334, 0.361, 0.389, 0.403, 0.431. The lowest anchored seed beats four of five
unanchored, a rank separation of 24 of 25 pairs, which does not rest on the SE
estimate at all. The falling SE at larger K reflects fewer runs qualifying, not a
weakening effect; the difference itself grows.

Holding as the curves extend, and stable in all three respects at nine points per
seed with five seeds per arm:

| window | anchored | unanchored | difference | | rank |
|---|---|---|---|---|---|
| K=5 | 0.4667 | 0.3834 | +0.083 | 2.8 SE | 23/25 |
| K=7 | 0.4653 | 0.3737 | +0.092 | 2.6 SE | 23/25 |
| K=9 | 0.4750 | 0.3730 | +0.102 | 2.9 SE | 23/25 |
| K=12 | 0.4795 | 0.3826 | +0.097 | 3.0 SE | 23/25 |

The effect size grows slightly with the window rather than swinging, the SE sits
at 2.6-2.9 throughout, and the rank separation is identical at all three cuts.
That combination is what none of this project's withdrawn findings had: the drift
result moved 4.2 SE to 0.4 SE between adjacent windows, and the FLUX
learning-rate result held its SE while being confounded. Per-seed at K=12, anchored 0.428, 0.443, 0.500, 0.504, 0.523 against unanchored
0.299, 0.371, 0.383, 0.398, 0.462. The worst anchored seed (0.428) sits above
four of the five unanchored, and well above the worst (0.299).

**What this does not say.** The anchor is not necessary for the task to work.
`sb44`, unanchored, reaches 0.500 and 0.583 late in its run, and one anchored
seed (0.410) is now beaten by one unanchored seed (0.417). Removing the anchor
shifts the average down and widens the downside - the worst unanchored seed hits
0.125, well below anything anchored - rather than preventing improvement outright.
`sa43` reached 0.625 against a 0.375 baseline, the largest climb on this task, but
it is a best case and not the typical one. Two seeds per arm makes that SE soft, but
the direction is consistent and it is the first positive evidence in this project
that `data_beta` does anything.

FLUX with the identical reward and group size ran the other way: anchored
collapsed to 0.29, unanchored held 0.625. The distinguishing variable is whether
the RL term is live. SD3.5 never lost it (0 of 18 epochs); FLUX lost it in 22 of
28. So:

- **RL term live** - the anchor regularises toward the data manifold and helps,
  which is what it is for.
- **RL term quiet** - the anchor is the only gradient and walks the policy away
  from the task.

That is exactly the behaviour the fix produces by gating the anchor on
`has_signal`: keep it where it earns its place, drop it where it would be acting
alone. The gate was written from the FLUX failure before this SD3.5 result
existed, so this is confirmation rather than justification.

Why counting showed nothing: it is graded, its advantages never vanish, so the
anchor is always in the "live" regime - and there its measured effect is under
0.02, which is the beta null. Both results are consistent; the sweep simply
picked the reward where the anchor has least to do.

### Drift, adequately powered: no effect at threshold, direction consistent

Eight seeds per arm, fixed window K=12, read once at the declared point with all
sixteen runs qualifying. Minimum detectable difference 0.0232.

| statistic | data_beta 0 | data_beta 0.3 | difference | |
|---|---|---|---|---|
| drift (higher = less decline) | +0.0097 | -0.0115 | +0.0212, SE 0.0113 | 1.9 SE |
| giveback (higher = surrendered more) | +0.0240 | +0.0425 | -0.0186, SE 0.0146 | 1.3 SE |

**Neither crosses 2 SE, so by the pre-registered criterion this is a null.** The
declared interpretation for that case was "no effect larger than 0.023 at this
horizon", and the observed difference of 0.0212 sits just under the 0.0232 the
design can resolve.

What is worth stating without claiming: **both statistics agree in direction, and
both say the anchored arm is worse** - it declines more and surrenders more of its
peak. That is the same direction as the original 2.2 SE result, now measured at
twice the seeds. Across four cohorts the direction has been: anchor worse (2.2 SE,
n=4/3), nothing (0.3 SE, n=4/4), anchor worse (1.9 SE, n=8/8). The 4.2 SE reversal
was a window artifact and never existed.

So the honest summary is not "the anchor does nothing" but **"if the anchor has an
effect on late-training degradation, it is at or below 0.02 and it makes things
slightly worse rather than better."** Distinguishing that from zero needs roughly
four times these seeds - about 30 per arm - which at 1.62 epochs/GPU-hour on this
node is not a reasonable ask. Recording the bound is more useful than continuing.

### Superseded readings: the original result and its apparent reversal

Read once at the declared point on an independent cohort (seeds 46-49, four per
arm, `report/driftrep_csv/`, common window K=14, `r3s48` short at 11 epochs):

| statistic | data_beta 0 | data_beta 0.3 | difference | | vs original |
|---|---|---|---|---|---|
| drift | +0.0080 | +0.0201 | -0.0121, SE 0.0231 | 0.5 SE | **sign flipped** |
| giveback | +0.0660 | +0.0342 | +0.0317, SE 0.0075 | **4.2 SE** | sign flipped |

The pre-registered criterion was explicit: *an opposite sign, at any strength,
means the first result does not stand*. Drift's sign flipped, so **the 2.2 SE
finding that the anchor makes degradation worse is withdrawn.**

**Definitive reading, fixed window K=12.** The window is now the declared 12
rather than the common run length, which is what the registration said and the
code did not do. Both cohorts, same window, same statistics:

| cohort | seeds | drift | giveback |
|---|---|---|---|
| original | 4 / 3 | **2.2 SE**, anchor worse | 1.0 SE |
| replication | 4 / 4 | 0.3 SE | 0.4 SE |

**The original does not replicate.** An independent cohort at equal or better
power finds nothing on either statistic, and the 4.2 SE that briefly appeared to
reverse it was the K=14 artifact and does not exist under the fixed window. No
claim about the anchor survives.

**And the replacement did not survive one more run.** The read above was taken at
seven qualifying runs under the stall clause of the declared read point. The
eighth run qualified minutes later, which shortened the common window from K=14
to K=12, and the picture changed completely:

| read | drift | giveback |
|---|---|---|
| 7 runs, K=14 (declared point) | 0.5 SE | **4.2 SE** |
| 8 runs, K=12 (secondary) | 0.3 SE | **0.4 SE** |

A 4.2 SE result became 0.4 SE by including one additional run. It was an artifact
of the window, not a property of the anchor.

**This is a mistake I had already diagnosed and repeated anyway.** After the level
experiment I wrote that deriving K from the common run length makes the result a
moving target and that "a future design should fix the epoch window in advance
instead of deriving it from whatever the runs deliver." I then pre-registered the
drift analysis with common-length truncation regardless. Having identified the
flaw in writing did not stop me building on it, and the cost was reporting a
4.2 SE finding that one more run erased.

So nothing below should be read as a claim about the anchor. Kept for provenance:

What appeared to replace it, before the eighth run. `giveback` - the
statistic built specifically for the peak-then-surrender shape, and the one that
missed threshold originally - now separates at **4.2 SE**, and the *unanchored*
arm gives back more of its peak (0.0660 against 0.0342). On this cohort the
anchor reduces late-training give-back by about 0.032, which is what a data
anchor is supposed to do.

**A bug in my own reporting, disclosed.** The verdict line printed "the ANCHORED
arm is worse" for this result. The two statistics have opposite polarity - higher
`drift` means less decline and is better, higher `giveback` means more was
surrendered and is worse - and the direction test assumed a single polarity. It
was right for `drift` (which is why the original read correctly) and inverted for
`giveback`. Had I not checked the arithmetic by hand against the arm means, I
would have reported a 4.2 SE result backwards, and it would have looked like
confirmation of a finding it actually overturns. Fixed; both cohorts re-verified.

**The resume confound is ruled out.** Declared alongside: resume rates in the
replication are 0.224/epoch (data_beta=0) against 0.198 (data_beta=0.3), a
difference of 0.5 SE. Per the four outcomes fixed in advance, drift failing to
replicate while resume rates do not differ is the case where the original 2.2 SE
was noise - which two registered statistics at an 0.089 effective bar always made
plausible.

**Where this leaves the anchor: nowhere, and that is the result.** Three readings
were taken. The original cohort said the anchor hurts at 2.2 SE. The replication
at seven runs said it helps at 4.2 SE. The same replication at eight runs said
nothing, twice over. Every apparent effect dissolved under either an independent
cohort or a single extra run.

The honest conclusion is not about beta at all. It is that this measurement -
four seeds per arm, a window derived from run length, curves at ~0.07 SEM on a
node that kills jobs hourly - cannot resolve differences of the size being looked
for, and will manufacture 2-to-4 SE results in either direction if read at a
moment of its choosing. The six-epoch null stands as the only firm result in the
beta work, and it stands partly because it was read once at a fixed design point
and was robust across four re-reads.

What a real answer would need: the epoch window fixed in advance rather than
derived, many more seeds than four, and a cluster that lets a run finish.

**That run is now in flight.** Both cohorts had a minimum detectable difference of
0.033-0.038 at 3-4 seeds, against a measured between-seed SD of 0.0232 - so the
first cohort's reported 0.0334 sat exactly on its own detection limit. Neither was
capable of the question. `report/analyze_drift_powered.py` is pre-registered for
eight seeds per arm (tags `p0s50`-`p0s57`, `p3s50`-`p3s57`), giving a minimum
detectable effect of 0.0232 with the window fixed at 12 by import rather than
derived. Until it reads, the honest status of the drift question is not "no
effect" but **"not yet measured at adequate power"**, and the two nulls above
should be read as that rather than as evidence of absence.

### Superseded: the original drift result (over 12 epochs the anchored arm degrades more)

`report/analyze_drift.py`, pre-registered before the runs existed, read once at
the declared point (all runs that could qualify: 4 seeds at data_beta=0, 3 at
data_beta=0.3, `d3s43` short on length). Curves in `report/drift_csv/`, grown
entirely under RNG-restoring resume. Common window K=12 epochs.

| statistic | data_beta 0 | data_beta 0.3 | difference | |
|---|---|---|---|---|
| drift (late third minus early third) | +0.0007 | -0.0326 | +0.0334, SE 0.0154 | **2.2 SE** |
| giveback (peak minus final) | +0.0676 | +0.0914 | -0.0238, SE 0.0229 | 1.0 SE |

**Both point the same way and it is the opposite of the anchor's purpose.** The
unanchored arm is flat late in training; the anchored arm declines and gives back
more of its peak. The forward-KL data term exists to hold the policy near the
data manifold over long training, and over these twelve epochs it is associated
with more degradation, not less.

**An asymmetry in resume rate, with the mechanism I proposed for it withdrawn.**
In the same cohort the anchored arm resumed more often per epoch:

| arm | resumes/epoch | n |
|---|---|---|
| data_beta 0 | 0.209 (SD 0.071) | 4 |
| data_beta 0.3 | 0.311 (SD 0.020) | 3 |

Difference +0.103 at 2.7 SE. I recorded this as a confound that might explain the
drift result entirely, on the reasoning that a crash mid-epoch makes the resume
repeat that epoch's training so the model sees part of it twice. **That reasoning
was wrong and the claim is withdrawn.** The state is written after an epoch's
training with `next_epoch = epoch + 1`, so a crash during epoch N discards N's
partial updates - they were only ever in memory - and the resume restores the end
of N-1 and redoes N from its correct starting point with the RNG restored.
Nothing is trained twice. I asserted a mechanism without reading the code path
that implements it, in a file I had written that same day.

What survives is a real but unexplained asymmetry: the anchored arm crashes more,
and no route from crash count to held-out drift has been established. Resume is
transparent by construction here, up to bf16 nondeterminism which both arms carry
equally. The declared test in `report/analyze_resume_confound.py` still runs,
because an unexplained asymmetry that tracks the outcome would be worth knowing
about even without a mechanism - but it is no longer a candidate explanation, it
is a check for one.

**Disclosure: I saw a partial replication value before the declared read point.**
Running `analyze_drift_rep.py` to check that a new fetch script worked printed
per-arm drift means at n=3 for data_beta=0 and n=1 for data_beta=0.3 - the
analysis reported as soon as four runs qualified in total, without requiring two
per arm. The values were +0.0222 and +0.0414, which at n=1 in one arm means
nothing statistically but points *opposite* to the original result. I am
recording it because I cannot un-see it and because a reader weighing the final
replication is entitled to know I had a partial look first. The analysis now
stays silent until both arms have at least two qualifying runs, and the declared
read point is unchanged.

Four reasons not to treat this as established:Four reasons not to treat this as established:

- **Two statistics were pre-registered**, so under the null the chance of one
  exceeding 2 SE is 0.089, not 0.045. A single 2.2 SE result across two tests is
  roughly a 2-sigma-equivalent finding at best.
- **n=3 in the anchored arm**, and a three-run SD is not a reliable SD. The
  earlier lesson here was a two-run SD reading 0.0276 against 0.0229 at four.
- **The weaker statistic is the one designed for this shape.** `giveback` was
  added specifically because `drift` is blind to peak-then-collapse, and it is
  the one that does not reach threshold.
- **One read, one task, one horizon.** Counting 6-9 objects at twelve epochs on
  SD3.5-medium.

What it does establish is that the earlier null is not the whole story: the level
comparison found nothing over six epochs, and something is visible at twelve. The
direction is worth taking seriously enough to replicate, which is the honest
status - a signal that survives its own pre-registration and fails to be
comfortable.

**Disclosure.** The verdict string in the analysis read "the anchor acts where it
was meant to" for a difference of either sign. The first real result came out the
opposite way, so the message would have described the finding backwards. It was
corrected after seeing the result; no threshold, statistic or inclusion rule
changed.

### Result: the first design could not have separated anything

`report/beta_sweep.png` (`report/make_beta_sweep.py`, curves in
`report/beta_csv/`). Longest surviving run per arm, since atlas29 killed and
restarted several of them:

| arm | n | epoch 0 | mean of the rest | best | last 5 |
|---|---|---|---|---|---|
| `hb0` data_beta 0 (= FlowGRPO) | 15 | 0.541 | 0.546 | 0.562 | 0.551 |
| `hb0` rerun, same seed | 9 | 0.541 | 0.575 | 0.621 | 0.586 |
| `hb001` 0.01 | 6 | 0.541 | 0.564 | 0.596 | 0.564 |
| `hb005` 0.05 | 10 | 0.541 | 0.546 | 0.565 | 0.544 |
| `hb03` 0.3 | 13 | 0.541 | 0.560 | 0.594 | 0.567 |
| `hk001` FlowGRPO reverse-KL 0.01 | 8 | 0.541 | 0.562 | 0.583 | 0.566 |

**A rerun of one arm settles it.** atlas29 killed `hb0` partway and the supervisor
resubmitted it, which produced two complete runs of the *identical* configuration
- same seed, same held-out set, same everything:

| `hb0` (data_beta 0) | n | peak | last 5 | mean after ep 0 |
|---|---|---|---|---|
| first run | 9 | 0.621 | 0.586 | 0.575 |
| rerun | 14 | 0.562 | 0.550 | 0.546 |

The two differ in mean by **0.029**. The spread across the four different beta
arms is **also 0.029**. Run-to-run variation at a fixed configuration is the same
size as the variation the sweep was built to detect, so the sweep detects nothing.

The consequence is worth stating bluntly: depending on which of the two runs you
happen to have, `hb0` is either the best arm in the sweep (0.575, top of the
table) or tied for the worst (0.546). Any ranking of these arms is a statement
about which run finished, not about beta.

Runs at the same seed diverge for the reason already recorded for
FlowGRPO-vs-DDRL past epoch 0: bf16 GEMM nondeterminism across processes, which
RL's feedback loop amplifies. This is the first time it has been measured at the
scale of a whole run rather than a single epoch.

A second replicate pair turned up independently in the seeded design, where
`b0s42` crashed and was resubmitted and both runs reached 8 points:

| `b0s42` (data_beta 0, seed 42) | n | mean(ep 1..7) | peak |
|---|---|---|---|
| run A | 8 | 0.5667 | 0.603 |
| run B | 8 | 0.5472 | 0.599 |

Spread 0.020, against 0.029 for the `hb0` pair. Two independent measurements of
run-to-run divergence at a fixed configuration, both in the 0.02-0.03 range, and
both the size of the between-arm spread the first sweep was trying to interpret.
The estimate the seeded design was sized on is holding up.

**Every point of every arm lies inside +/-1 SEM of the untrained baseline.** The
spread between arm means is 0.029 (0.546 to 0.575, excluding `hk001`), while the
standard error on a single point is 0.07 and on an arm mean is 0.023-0.029. The
between-arm differences are the same size as the noise on each arm, so no arm can
be ranked against another and none is distinguishable from doing nothing.

The figure is drawn with the SEM band shaded for that reason: the arms only span
0.51-0.62, so autoscaling to the data makes noise look like structure. Extending
the longest arm to 13 points did not change any of this - its last five average
0.567, still inside the band.

**The sweep was redesigned rather than extended.** Five betas at one seed each
cannot separate a 0.03 effect when two runs of one config differ by 0.029; more
epochs of the same design only add draws from the same band. Replicate seeds are
the only lever - the eval latents are pinned by `eval_seed` and identical across
runs, so the divergence is the policy and not the measurement, and raising the
eval sample count would not touch it. The second design is two betas at the ends
of the range with four seeds each (`b0s42`..`b0s45`, `b3s42`..`b3s45`), which puts
the standard error on the arm difference at about 0.015 and a 0.03 effect at
roughly 2 SE. All five arms that have started report an epoch-0 held-out reward of
0.541 across four different training seeds, which is the check that `SEED` touches
only training: eval is driven by `eval_seed` and must not move with it. The middle betas are dropped: if 0 and 0.3 do not separate, 0.01 and
0.05 will not.

#### Seeded design: final, all four seeds per arm

| arm | mean | between-seed SD | n |
|---|---|---|---|
| data_beta 0 | 0.5465 | 0.0180 | 4 |
| data_beta 0.3 | 0.5607 | 0.0229 | 4 |

**Difference -0.0143, SE 0.0146, 1.0 SE. No effect detected.** The 95% interval on
the difference is roughly [-0.044, +0.015], so it covers zero comfortably. The two
arms' seed values interleave: beta=0 spans 0.5277-0.5688 and beta=0.3 spans
0.5362-0.5897.

The last run is the point. At seven of eight runs the difference was 0.0225 and
projecting to about 1.7 SE; `b3s42` landed at 0.5362, the lowest beta=0.3 seed, and
pulled it back to 0.0143 - the same value it had at three-and-two seeds. Every
intermediate reading of this experiment (0.9 SE, then 1.1, then 1.5, then 1.0)
would have supported a different story if quoted alone, which is why the analysis
was pre-registered and why the verdict is the one the script prints rather than
the one the trend suggested.

**Robustness of the level result.** It has now been read four times from
different snapshots of the rotated files, since the longest-run rule selects
whichever run happens to be longest at read time:

| read | difference | in SE |
|---|---|---|
| at design completion (reported above) | -0.0143 | 1.0 |
| an hour later, longer curves | -0.0088 | 0.7 |
| from the archived copies | -0.0101 | 1.1 |

The point estimate wanders by about 0.005 but every reading returns the same
verdict at 0.7-1.1 SE. The conclusion is robust to the read-time ambiguity even
though the number is not, which is the more useful statement.

The archive replay also had one curve rejected by the epoch-contiguity check
(`b3s43`, 8 rows spanning epochs 0-6), a duplicate epoch left by the resume bug
before it was fixed. The guard caught real corruption in production data rather
than only in its own test.

**A gap in the pre-registration, recorded rather than patched over.** The script
fixed the statistic but not the moment to read it. Both the common length K and
which run is longest per tag keep moving as runs grow, so re-running it later
gives a different number: at design completion it read -0.0143 / 1.0 SE, and an
hour later -0.0088 / 0.7 SE on longer curves. The verdict is the same either way,
and the value above is the one taken at the pre-registered stopping point (four
seeds per arm), which is the honest place to read it. A future design should fix
the epoch window in advance instead of deriving it from whatever the runs
deliver.

**Conclusion.** Anchor strength between 0 and 0.3 has no detectable effect on
held-out counting accuracy over the first six epochs on this task. Stated as a
bound: any true effect is smaller than about 0.03, which is the resolution four
seeds per arm buys at a between-seed SD of 0.02. Going further needs either many
more seeds or a task where the anchor has more to do - its role is preventing
drift over long training, and six epochs is early for drift.

**On reading this experiment while it ran.** The intermediate values are kept in
git history rather than here. Two mistakes are worth naming: I projected a final
verdict from a two-run-per-arm difference and had to withdraw it, and the between-
seed SD estimated from two runs (0.0276) was badly off the four-run value (0.0229
and 0.0180). A two-run standard deviation is not a standard deviation.

What this does **not** say. It does not say the anchor is useless - its job is to
stop drift over long training, and ten epochs is early for drift to appear. It
does not say the arms are equal, only that this experiment cannot tell them
apart. Resolving a 0.03 difference at this SEM needs roughly 20x the eval samples
per point, or far longer runs, or both. And `hk001` has no verdict at all: it
segfaulted four times and never held more than three points.

Curves as of 11:30 Aug 1, all against the shared 0.541 baseline (per-point SEM
~0.07, so these are recorded, not yet interpreted):

| arm | held-out, epoch 0.. |
|---|---|
| `hb0` (data_beta 0) | 0.541 0.533 0.530 0.607 0.581 0.621 0.602 |
| `hb001` (0.01) | 0.541 0.531 0.544 0.596 0.572 0.577 |
| `hb005` (0.05) | 0.541 0.530 0.555 (restarted twice after segfaults) |
| `hb03` (0.3) | 0.541 0.525 0.544 0.558 0.594 0.553 0.568 |
| `hk001` (FlowGRPO reverse-KL 0.01) | restarted three times, no points retained |

All three arms with enough length sit above baseline from epoch 3 on. No arm is
being called ahead of another: at eight points and this SEM the gaps between them
are not resolvable, and the arms have unequal lengths because atlas29 killed two
of them repeatedly.

**FLUX counting, revised.** The standing description was train reward rising while
held-out falls. This run does not show that: held-out is 0.667, 0.660, 0.674,
0.658 - flat on its baseline - and train reward is 0.689, 0.800, 0.637, 0.461,
which is falling, not rising. With `prompts_per_epoch=2` the train number is
dominated by which two prompts were drawn, so it is not evidence of much in
either direction. What survives is the weaker claim already recorded: FLUX does
not climb on tasks where it starts near its ceiling.

## Known gaps

Rewritten Aug 2 2026. The previous version had gone stale in a way worth naming:
it called FLUX.1-dev feasibility "the main open question" while the Models table
two sections above records it validated end to end, and it said several reward
rows still needed re-measuring when all seven carry measurements on fixed code.
Both statements were true when written and false by the time anyone would read
them.

What is actually unresolved:

- **RESOLVED, negatively: FLUX does not improve even on the task where it has
  headroom.** Counting 6-9 objects, untrained 0.507, three seeds, DDRL with anchor
  (data_beta=0.05, 512px, group 6, 28 steps), 7-9 epochs each:

  | seed | mean(ep 1..6) | final | peak |
  |---|---|---|---|
  | 42 | 0.5020 | 0.497 | 0.531 |
  | 43 | 0.5113 | 0.505 | 0.538 |
  | 44 | 0.4698 | 0.481 | 0.507 |

  Across seeds 0.4944 (SD 0.0218, SE 0.0126) against the untrained 0.507: a
  difference of -0.013, about 1 SE. Flat, not rising, and no seed's mean exceeds
  its own baseline. The best single point in any run is 0.538, within one
  per-point SEM (~0.07) of the start.

  **Updated at 10-12 points per run.** The effect is stable in size and the
  precision has improved:

  | window K | across-seed mean | vs 0.507 | |
  |---|---|---|---|
  | 6 | 0.4940 | -0.0130 | 0.9 SE |
  | 8 | 0.4932 | -0.0138 | 1.3 SE |
  | 10 | 0.4900 | -0.0170 | 2.0 SE |
  | 12 | 0.4865 | -0.0205 | 2.4 SE |

  The difference sits between -0.013 and -0.021 while the standard error falls
  from 0.0137 to 0.0086. Both move, but not equally: the SE roughly halves while
  the difference grows by about half, so most of the rise in the SE-count is
  precision rather than a moving quantity. That is a weaker claim than I made at
  10 points, where I said the difference was stable and only the SE improved; the
  K=12 value of -0.021 is outside the range I quoted then, so the honest reading
  is that the deficit is somewhere around 0.015-0.020 and may be widening slowly
  as training continues.
  That is the opposite of the drift result, where the difference itself swung
  between windows - and it is the distinction between a measurement getting
  sharper and one being read at a lucky moment.

  The within-run trend is a separate question and the answer there is no: late
  third minus early third gives +0.007, -0.053 and +0.022 across the three seeds,
  0.4 SE. So FLUX **settles below its untrained score and stays there** rather
  than degrading progressively. Final points are 0.488, 0.459, 0.442 against a
  0.507 start.

  This run has no declared read point - it is exploratory, not a pre-registered
  test - so the SE-count will keep drifting as curves lengthen. The stable
  quantity is the effect size of about -0.015, and that is what should be quoted.

  The earlier reading at 7-9 points, kept for provenance:

  | window K | across-seed mean | vs 0.507 | |
  |---|---|---|---|
  | 5 | 0.4935 | -0.0135 | 0.8 SE |
  | 6 | 0.4940 | -0.0130 | 0.9 SE |
  | 7 | 0.4944 | -0.0126 | 1.0 SE |
  | 8 | 0.4932 | -0.0138 | 1.3 SE |

  Flat at every cut. Compare the drift result, where the verdict moved from
  4.2 SE to 0.4 SE between K=14 and K=12 - a difference that size across windows
  is the signature of reading noise, and its absence here is why this null can be
  stated plainly.

  **A second reward collapses far harder, and the mechanism is visible in the
  logs.** `ocr_exact` on FLUX (untrained 0.625, three seeds, anchor 0.05,
  everything else matched) falls to 0.333, 0.375, 0.333 - roughly -0.28 where the
  per-point SEM is 0.10. Counting moved only -0.013 under the same anchor, so
  this is an order of magnitude larger.

  The training reward collapses too (0.75 -> 0.08 on one seed), which rules out
  overfitting: the policy is getting worse at the objective it optimises, on its
  own training distribution. The log shows why:

  ```
  Epoch 5 | reward=0.083 | rl_loss=0.0000 | data_loss=0.4452
          | total_loss=0.0223 | mean_ratio=0.9999
  ```

  `total_loss` is exactly `0.05 * data_loss`. The RL term has gone to zero and
  the anchor is the only gradient still training the model. `ocr_exact` is binary,
  so at group size 6 all six samples in a group frequently score the same, the
  group-normalised advantage vanishes, and `rl_loss` goes with it - while the
  anchor keeps pulling toward generic anchor images regardless. The worse it gets,
  the more uniform the groups become, so it compounds.

  **Confirmed.** `ocr_exact` at `data_beta=0`, three seeds, everything else
  matched:

  | window K | anchor on | anchor off | difference | |
  |---|---|---|---|---|
  | 4 | 0.6112 | 0.6157 | +0.0044 | 0.1 SE |
  | 5 | 0.5591 | 0.6284 | +0.0693 | 2.0 SE |
  | 6 | 0.5250 | 0.6374 | +0.1124 | 3.1 SE |

  Identical at K=4 and diverging steadily after, which is the shape the mechanism
  predicts: the advantages only vanish once the reward has fallen far enough for
  groups to become uniform, so the anchor cannot dominate until then. All three
  unanchored seeds hold at the 0.625 baseline with SD 0.006; all three anchored
  seeds collapse, and the collapse does not plateau - `fo42` reached 0.167 by
  epoch 15, a 73% relative fall from its untrained 0.625.

  **And with the anchor gated, FLUX stops degrading.** At K=7 the arms read 0.4965
  anchored against 0.6389 unanchored, and the unanchored latest points are 0.625,
  0.708, 0.625 - mean 0.653 against a 0.625 baseline. Not a climb (0.5 SE, and
  `ocr_exact` quantises to k/24 so the per-point SEM is ~0.10), but it is the
  first configuration in the entire FLUX investigation that does not go down.

  That is worth stating carefully against the section above. The earlier finding -
  that FLUX degrades in proportion to learning rate on *counting* - stands: those
  runs used a graded reward whose advantages never vanish, so the anchor was not
  the cause there and the fix does not address them. What the fix removes is a
  separate and much larger failure that only appears with sparse rewards. Two
  different problems, one now solved.

  **This is a library defect, not a FLUX quirk.** With a sparse reward and a small
  group, `has_signal` goes false, `rl_loss` is set to zero - and the trainer keeps
  applying `data_beta * data_loss` anyway. The zero-signal guard in
  `base_trainer` only skips an epoch when fewer than two advantages are nonzero,
  which does not catch the case where the advantages exist but carry no
  within-group variance. Any sparse or binary reward at a small group size will
  hit this, on any model. It is visible in one log line: `total_loss` equal to
  `data_beta * data_loss` with `rl_loss` at zero.

  That also explains why turning the anchor off changed nothing on counting:
  counting is graded, its advantages do not vanish, and the RL term keeps working.
  `fn42`/`fn43`/`fn44` test the prediction directly - `ocr_exact` at
  `data_beta=0`. If the collapse stops, this is a real design flaw: a sparse
  reward plus a small group plus any anchor means the anchor silently takes over.

  **CORRECTION TO THE CORRECTION (Aug 4).** The contamination claim below was
  itself over-read and is downgraded. It counted epochs where `rl_loss` printed
  as `0.0000` and treated those as epochs where the RL term had switched off.
  Four decimal places means `< 0.00005`, not zero - and SD3.5's own `rl_loss` runs
  0.0001-0.0003, barely above the print threshold. So "the RL term was off in 70%
  of FLUX epochs" conflated *prints as zero* with *switched off*, and the
  frequency table does not support the stronger reading.

  What is still solid is the OCR case, where the arithmetic distinguishes:
  `total_loss` 0.0223 against `data_loss` 0.4452 gives `0.05 * 0.4452 = 0.022260`,
  which prints as 0.0223, whereas a `rl_loss` of 0.0001 would print 0.0224. There
  the RL term really was negligible and the anchor really was the whole gradient.
  That case also had a collapsed reward (0.083), which is the condition under
  which uniform groups are expected.

  So: the anchor defect is real and the fix stands, confirmed by a prediction
  that held on three seeds. The claim that it contaminated the *counting* runs is
  not established, and the learning-rate withdrawal below rests on it. Both are
  left recorded rather than deleted, because the sequence - finding, correction,
  correction of the correction - is the useful part.

  **[DOWNGRADED] The FLUX results below are contaminated by the anchor defect.** Counting
  the epochs where `rl_loss` printed as exactly zero:

  | run | epochs with rl_loss = 0 |
  |---|---|
  | FLUX counting, lr 3e-5 | 11/15, 8/12, 12/16 |
  | FLUX counting, lr 2e-4 | 5/9, 3/7, 6/8 |
  | SD3.5 counting (`hb0`, `hb03`) | **0/25, 0/20** |

  On FLUX the RL term was off in roughly 70% of epochs; on SD3.5 it never switched
  off at all. Those FLUX runs were therefore trained mostly by the anchor, which
  pulls toward the anchor images - so "FLUX does not climb" was substantially a
  measurement of the defect rather than of FLUX.

  The cause is group size. FLUX ran at group 6 because of memory; SD3.5 ran at 12.
  The counting reward is graded but heavily zero-inflated, since the detector
  frequently finds nothing, so with only six samples a group often scores
  uniformly and the advantage vanishes.

  What this withdraws:
  - **The learning-rate dose-response.** Both arms were heavily affected (43-75%
    of epochs), so the comparison was between two differently-contaminated runs,
    not between learning rates. The window-stability that made it look solid does
    not rescue it - a confound can be perfectly stable.
  - **"The gradient is harmful, not absent."** The gradient was largely absent.
  - **The saturation and rollout-size eliminations** are unaffected in their own
    terms but were interpreting a contaminated baseline.

  What survives: the anchor defect itself, which is confirmed by a prediction that
  held, and the fix. And the observation that FLUX at group 6 cannot reliably
  produce within-group variance on these rewards, which is a real constraint worth
  knowing independently.

  **The group-12 test is not completable on this hardware, and that is itself a
  finding.** FLUX at group 12 with the anchor fix produced 4, 3 and 2 eval points
  in five hours before being stopped - one epoch still unfinished after 95
  minutes. Doubling the group doubles the rollout, and against an unchanged crash
  rate each failure costs twice as much work. Second configuration to hit this
  wall; the `prompts_per_epoch=8` test managed 10 points across 33 crashes in
  twelve hours.

  Partial curves before stopping: 0.521, 0.491, 0.537 / 0.521, 0.542 / 0.504
  against a 0.507 baseline - at or slightly above, where the group-6 runs were
  already below by this stage. Suggestive only; two to four points per seed is not
  a result and is not recorded as one.

  A fourth attempt on a single free card got further and still produced nothing:
  the 32GB of model blobs streamed in four hours (against a ~113 minute estimate
  at this node's nominal 4.7 MB/s), `PREWARM_DONE` was reached, and the job then
  spent another hour stuck in Python imports - the prewarm covers model blobs but
  not the venv, so torch's 1.6GB and nvidia's 2.7GB demand-page off the same
  filesystem. Five hours, zero training epochs, stopped.

  A third attempt, raising exploration noise to 1.0 at group 6 - the one lever
  that costs neither memory nor time and directly controls within-group sample
  diversity - could not be evaluated either, for a reason unrelated to the
  configuration. It ran 342 transformer forward calls in two hours where the
  identical group-6 setup had needed 3,320 to reach its first eval, roughly a 10x
  slowdown, with no NaNs and the node reporting MIXED+PLANNED rather than
  draining. atlas29 was simply full: 7 of 7 a6000-Adas across three users. At that
  rate epoch 0 alone would have taken 19 hours, so the runs were stopped. Whether
  noise level fixes the within-group variance problem on FLUX is therefore still
  untested.

  The constraint is worth stating for anyone planning FLUX RL on a 48GB card:
  group 6 fits comfortably but produces frequent uniform groups on these rewards,
  group 12 fits but makes epochs slow enough that an unstable node cannot finish
  them. That is a real corner rather than a tuning preference.

  **[WITHDRAWN] The likely answer: the gradient is harmful, not absent.** Learning rate was
  the last cheap variable. Three seeds at `lr 2e-4` against the `fh` cohort's
  `3e-5`, everything else matched, held-out mean over epochs 1-4:

  | learning rate | held-out mean | vs untrained | SE |
  |---|---|---|---|
  | 0 (untrained) | 0.5070 | - | - |
  | 3e-5 | 0.4935 | -0.0135 | 0.0169 |
  | 2e-4 | 0.4473 | -0.0597 | 0.0049 |

  Monotonic: more optimisation, further below baseline, with the high-rate arm
  tight across seeds. That is a dose-response rather than a tuning miss, and it
  changes the question. FLUX is not failing to learn - it is learning something
  that costs held-out counting accuracy, in proportion to how hard the gradient
  is followed. "The adapter barely moves at a rate tuned for a 2.5B model" is
  ruled out; the policy responds to the learning rate strongly and in the wrong
  direction.

  So the remaining suspect is the reward signal itself: whether OWLv2's gradient
  on FLUX's output distribution points somewhere useful. On SD3.5 the same reward
  and recipe climb (0.541 to ~0.60 across the beta arms). The natural next test
  is a different reward on FLUX with everything else fixed - if OCR or aesthetic
  climbs where counting degrades, the detector is implicated; if every reward
  degrades, it is the model or the sampler.

  **This is the only comparison in the project that survives a window sweep.**

  | window K | lr 3e-5 | lr 2e-4 | difference | |
  |---|---|---|---|---|
  | 5 | 0.4935 | 0.4473 | +0.0462 | 2.6 SE |
  | 6 | 0.4940 | 0.4406 | +0.0534 | 3.2 SE |
  | 7 | 0.4944 | 0.4397 | +0.0547 | 3.3 SE |

  Same sign, same magnitude, above 2 SE at every cut, and growing slightly as
  precision improves rather than swinging. Every other 2-SE-plus result here
  failed exactly this check - the drift finding went from 4.2 SE to 0.4 SE
  between adjacent windows and was withdrawn.

  Caveat kept: three seeds at two rates plus the untrained point, and a
  dose-response on two doses is weaker than on three. A third rate (say 1e-5)
  would test whether the relationship is monotonic across the range or whether
  3e-5 is already past a turning point.

  **The data anchor is not the explanation either.** Every FLUX run on this task
  had `data_beta=0.05`; the anchor had never been varied on FLUX, only on SD3.5.
  Three seeds at `data_beta=0` (`fz42`/`fz43`/`fz44`), everything else matched to
  the `fh` cohort, common six-epoch window:

  | | per-seed | mean | vs 0.507 |
  |---|---|---|---|
  | anchor on (0.05) | 0.5022, 0.5126, 0.4672 | 0.4940 | -0.0130 |
  | anchor off (0) | 0.4974, 0.4842, 0.4802 | 0.4873 | -0.0197 |

  Difference +0.0067 at 0.5 SE. Both settle below the untrained score and the
  anchor does not separate them - if anything the unanchored runs are marginally
  lower, and they are tighter across seeds (SD 0.009 against 0.024), so this is
  not a power problem concealing an effect.

  **Where the FLUX question stands.** Four things checked, none of them the cause:
  saturation (0.49 of headroom on this task), rollout size (4x the batch, same
  result), the data anchor (off, same result), and rollout quality (train reward
  0.44-0.58 brackets the 0.49 held-out score, so the samples carry signal). What
  remains untested is learning rate, guidance scale, group size, and whether
  OWLv2's gradient signal behaves differently on FLUX's output distribution than
  on SD3.5's. The honest status is that RL under this recipe moves FLUX slightly
  below its starting point on a task where it has room, and four plausible causes
  have been ruled out rather than one confirmed.

  **Follow-up: rollout size is not the explanation, and the test hit a wall.**
  `fw42`/`fw43` repeated the run at `prompts_per_epoch=8` (48 trajectories per
  epoch, matching SD3.5) with nothing else changed. Over the same six-epoch
  window:

  | rollout | per-seed means | mean | vs 0.507 |
  |---|---|---|---|
  | 12 traj/epoch | 0.5020, 0.5113, 0.4698 | 0.4944 | -0.013 |
  | 48 traj/epoch | 0.4860 | 0.4860 | -0.021 |

  The larger batch did what it should proximally - train reward steadied at
  0.49-0.50 against the 0.14-0.86 swings of the small-batch runs - and the
  held-out curve declined the same way regardless, `fw42` running
  0.525, 0.488, 0.482, 0.477, 0.476, 0.468. So the hypothesis is not supported.

  Stated with its limit: this rests on **one** completed larger-batch run. `fw43`
  reached three points across seventeen crashes and never recovered. Quadrupling
  the rollout made each epoch ~45 minutes instead of ~7, and since a crash costs
  the epoch in progress, the same crash rate consumed four times more work per
  failure - 33 crashes produced 10 points between the two runs over twelve hours.
  The configuration that would test this properly is the one this node is least
  able to finish, so the test was stopped rather than continued. One run pointing
  clearly the same way as three smaller ones is suggestive, not conclusive, and
  the honest status of *why* FLUX does not climb remains open.

  **Superseded note (follow-up running):** The rollouts are not the problem - train reward means
  0.44-0.58 bracket the 0.49 held-out score, so they carry signal, unlike the
  earlier 20-step attempt where train sat at 0.083 against a 0.507 eval because
  the images were too incoherent to count. What differs from the SD3.5 runs is
  batch: FLUX used `prompts_per_epoch=2` x group 6 = 12 trajectories per epoch
  against SD3.5's 4 x 12 = 48, so two prompts decided each update and train reward
  swung 0.14-0.86 epoch to epoch. `fw42`/`fw43` repeat the run with
  `prompts_per_epoch=8` (48 trajectories, matching SD3.5) and nothing else
  changed. If it climbs, the limit was rollout size; if it does not, that
  hypothesis is out and the question stays open.

  This closes the question the section previously called the most useful thing
  left to do, and the answer is that the headroom explanation was incomplete.
  FLUX being near its ceiling explained the flat curves on OCR and 2-5 object
  counting; here there is 0.49 of room above the untrained score and it still does
  not climb. Whatever limits FLUX under this recipe is not only saturation. Three
  seeds at 7-9 epochs is enough to say it does not rise, not enough to say why.

- **Previously recorded here as open: no FLUX task has shown a held-out
  improvement.** FLUX.1-dev trains correctly
  (peak 36.5 GB, ratio 0.9998, exit 0) but starts near its ceiling on every task
  tried - OCR 0.792, counting 0.667, exact-match OCR 0.625 - so there is nothing
  to demonstrate on. Counting 6-9 objects (untrained 0.507) is the one candidate
  with headroom and its runs were repeatedly killed before producing a curve
  longer than seven points. **Now running** (`fh42`/`fh43`/`fh44`, three seeds,
  DDRL with anchor at data_beta=0.05, 512px group 6, 28 steps): the earlier
  attempts died at one epoch because a crash discarded the run, which is the
  thing `resumable=True` fixes, so this is the first attempt with a mechanism to
  survive the node. Three seeds deliberately, not sixteen - the question is
  whether it climbs at all, and scaling a measurement before knowing there is
  something to measure is the mistake the beta work already made.
- **DiffusionDPO's held-out gain is not established.** Preference accuracy peaks
  at 0.610 and settles at 0.540 against chance 0.500, which is 1-2 SE on 100 fixed
  pairs. The trainer is correct and the metric is sound; the result is weak.
- **The anchor's effect on drift is bounded, not measured.** At or below 0.02 and
  slightly negative; separating it from zero needs ~30 seeds per arm.
- **atlas29's segfaults have no explanation.** Four hypotheses proposed, all
  disproved or failed replication. The working response is to resubmit.

Deliberate deviations, not bugs: the log-prob is mean-reduced rather than summed
(which is what makes `clip_range=1e-3` sane), `torch.std` is Bessel-corrected
where the reference numpy is not (~4.5% advantage scale at group 12), and the
counting reward is graded where GenEval's is binary.

**Cost note for planning:** `data_beta > 0` runs at 1.62 epochs/GPU-hour against
2.12 with the anchor off, measured over 81 GPU-hours - about 24% of throughput for
the extra forward and backward per timestep.
