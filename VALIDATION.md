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
| DRaFT | **validated (pre-audit)** | aesthetic 5.9→9.6. Has no importance ratio, so the SDE defect does not apply; the number stands but predates the eval fix |
| SFT | **validated (pre-audit)** | flow-matching loss converges. Now also implemented for FLUX (was SD3-only) |
| DiffusionDPO | **pending** | re-run blocked by Pick-a-Pic HF gating. Also had a missing 0.5 factor, so any prior run was at twice the intended beta |

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
| aesthetic (DRaFT) | **pending** | OOMs a 24GB A5000 even at truncation 1, group 1 (23.19 GB in use): DRaFT holds activations for every backpropagated step. Requeued on the 48GB node |

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
| FLUX.1-dev | **validated end to end.** Four epochs on one 48GB a6000-Ada, `EXIT 0`: reward 0.312 → 0.806 → 0.750, ratio 0.9992 → 0.9998, 26.1M LoRA params, **peak 36.5 GB** so ~11.5 GB headroom at 512px group 6. The ratio matters twice over: FLUX has a different schedule (dynamic shifting, `mu`), so ~0.999 shows the sigma=1 fix is not SD3-specific. Getting there needed two workarounds, neither in the code: a wedged a6000-Ada on atlas29 (passes `torch.cuda.is_available()`, then hangs on the first allocation; the preflight now catches it in 2 min and a driver retries), and that node reading `/atlas` at **4.7 MB/s** against 244 MB/s from the login node. Streaming the 32GB once took `real 133m20s`, after which the model loaded from page cache in about a minute, versus the ~13h the mmap page-fault path was on track for |
| FLUX.1-schnell | runs, but a 4-step distilled model is a poor SDE-RL target; its one aesthetic run declined (6.06→5.11→5.81) |

FLUX also could not run DDRL with an anchor or SFT at all, because
`compute_flow_matching_loss` spoke only SD3 (no `img_ids`/`txt_ids`/`guidance`,
unpacked latents, `timestep = sigma*1000`). Fixed and shape-verified.

## Known gaps

- **FLUX.1-dev feasibility** (does 12B + LoRA fit and train on one 48GB card) is
  the main open question. Only atlas29 has 48GB cards; at least one is wedged, and
  the node reads /atlas at 4.7 MB/s, so each attempt costs ~2h of model loading.
- **DiffusionDPO re-run** needs Pick-a-Pic terms accepted on HF.
- Several reward rows above need re-measuring on fixed code; they are marked.
- Deliberate deviations, not bugs: the log-prob is mean-reduced rather than
  summed (which is what makes `clip_range=1e-3` sane), `torch.std` is
  Bessel-corrected where the reference numpy is not (~4.5% advantage scale at
  group 12), and the counting reward is graded where GenEval's is binary.
