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

## Beta sweep (in flight)

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

One result is already in and does not need the curves: **all four DDRL arms
reported an epoch-0 held-out reward of 0.541 +/- 0.071 (n=24), identical to three
decimals across four independent processes.** That is the eval determinism claim
tested the hard way - same seed, same held-out set, four separate jobs - rather
than asserted from the code.

Curves are not yet long enough to compare. On these the SEM is ~0.06-0.08, so
fewer than about eight points says nothing, and nothing below that bar will be
reported here.

## Known gaps

- **FLUX.1-dev feasibility** (does 12B + LoRA fit and train on one 48GB card) is
  the main open question. Only atlas29 has 48GB cards; at least one is wedged, and
  the node reads /atlas at 4.7 MB/s, so each attempt costs ~2h of model loading.
- **DiffusionDPO** is running on an ungated mirror; no account action is needed after all.
- Several reward rows above need re-measuring on fixed code; they are marked.
- Deliberate deviations, not bugs: the log-prob is mean-reduced rather than
  summed (which is what makes `clip_range=1e-3` sane), `torch.std` is
  Bessel-corrected where the reference numpy is not (~4.5% advantage scale at
  group 12), and the counting reward is graded where GenEval's is binary.
