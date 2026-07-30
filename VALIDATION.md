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
| FlowGRPO | **validated** | held-out OCR 0.300 (untrained baseline) → 0.494 at epoch 2, ratio 0.9977; longer curve in flight |
| DDRL (β=0) | **validated** | held-out OCR → 0.569 at epoch 2, train reward 0.130→0.146→0.315, ratio 0.9986. Identical to FlowGRPO(β=0) by construction, see below |
| DDPO | **validated** | held-out OCR → 0.568 at epoch 2, train reward 0.130→0.399→0.530, ratio 0.9953. Now uses its own buffered per-prompt estimator |
| DDRL (anchor + monotonic transform) | **climbing, non-collapse not yet proven** | held-out OCR → 0.482 at epoch 2 vs 0.300 baseline, anchor term active (`data_loss=0.268`). Previously this configuration collapsed; it is now above baseline. But the old collapse appeared around epoch 10, so this needs ~10 epochs before "does not collapse" is a claim rather than a hope |
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

**DDPO is now a different algorithm from FlowGRPO.** It previously never overrode
`_compute_advantages`, so it inherited GRPO's within-batch tracker and *was*
FlowGRPO with `clip_range=0.2`. The distinction is visible in the logs: FlowGRPO
and DDRL report mean advantage exactly 0 (group normalization forces it), DDPO
reports 0.089 then 0.101 (it normalizes against a per-prompt history). **Any
pre-audit DDPO number is a FlowGRPO number.**

## Rewards

| Reward | Climbs? | Notes |
|--------|---------|-------|
| OCR | **yes, 0.300 → 0.494-0.569 by epoch 2** | held-out, fixed seed, n=44, SEM ~0.065 |
| counting (OWLv2) | **pending re-measure** | climbed pre-audit, but at noise 0.1; the noise tradeoff it implied was an artifact |
| clip_score | pending re-measure | 27→29 pre-audit |
| aesthetic (DRaFT) | **yes 5.9→9.6** | no importance ratio, unaffected by the SDE defect |
| aesthetic (FlowGRPO) | pending re-measure | previously flat; every FlowGRPO rollout at noise 0.7 had zero signal |
| PickScore | pending re-measure | previously flat and attributed to ceiling; that conclusion is no longer supported |
| hps_v2 | pending re-measure | same |

The old headline claim was "RL increases a reward when the task has headroom, and
stays flat when the base model is near the ceiling." The headroom half still
holds and OCR demonstrates it. The *flat* half is no longer supported: those runs
were at noise 0.7, where rollouts carried no signal at all, so flatness is not
evidence about the reward's ceiling. Prompt difficulty is now calibrated against
the base model rather than assumed, and no OCR tier is saturated (base 0.32-0.59
depending on target length; the reward is length-normalized, so longer targets
score higher).

## Models

| Model | Status |
|-------|--------|
| SD3.5-Medium | **validated** on fixed code (table above) |
| FLUX.1-dev | **code path fixed, feasibility pending.** Previously could not start at all: its scheduler sets `use_dynamic_shifting=True` and requires `mu`. The remaining blocker is a wedged a6000-Ada on atlas29 (passes `torch.cuda.is_available()`, then hangs on the first allocation), not storage: sequential reads of the model blobs measure 244 MB/s |
| FLUX.1-schnell | runs, but a 4-step distilled model is a poor SDE-RL target; its one aesthetic run declined (6.06→5.11→5.81) |

FLUX also could not run DDRL with an anchor or SFT at all, because
`compute_flow_matching_loss` spoke only SD3 (no `img_ids`/`txt_ids`/`guidance`,
unpacked latents, `timestep = sigma*1000`). Fixed and shape-verified.

## Known gaps

- **FLUX.1-dev feasibility** (does 12B + LoRA fit and train on one 48GB card) is
  the main open question. Only atlas29 has 48GB cards, and at least one of them is
  wedged, so getting a healthy allocation takes retries.
- **DiffusionDPO re-run** needs Pick-a-Pic terms accepted on HF.
- Several reward rows above need re-measuring on fixed code; they are marked.
- Deliberate deviations, not bugs: the log-prob is mean-reduced rather than
  summed (which is what makes `clip_range=1e-3` sane), `torch.std` is
  Bessel-corrected where the reference numpy is not (~4.5% advantage scale at
  group 12), and the counting reward is graded where GenEval's is binary.
