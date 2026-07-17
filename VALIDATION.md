# Validation status

Convergence runs on the Stanford cluster (SD3.5-Medium unless noted), not just
smoke tests. "Validated" = the intended objective actually moves in a real run.

## Trainers

| Trainer | Status | Evidence |
|---------|--------|----------|
| FlowGRPO | **validated** | OCR→1.0, counting 0.79→~1.0, clip_score 27→29; robust across anchor β |
| DRaFT | **validated** | aesthetic 5.9→9.6 (direct reward backprop) |
| DDRL | **validated (with anchor)** | climbs with a data anchor; β=0 collapses (see grid). Anchor is less robust than FlowGRPO's KL. |
| SFT | **validated** | Naruto fine-tune, healthy flow-matching loss, completes |
| DiffusionDPO | **validated (earlier)** | preference learning confirmed previously; current re-run blocked by Pick-a-Pic HF gating (needs terms acceptance) |
| DDPO | **tested → weak** | converges mechanically (ratio ~1.0, gradients flow) but does NOT reliably increase reward: count degraded 0.34→0.10, OCR net-flat. Genuinely weaker than FlowGRPO's GRPO. |

## Rewards (via FlowGRPO unless noted)

| Reward | Climbs? | Notes |
|--------|---------|-------|
| OCR | **yes → 1.0** | large headroom (SD3.5 poor at text) |
| counting (OWLv2) | **yes 0.79→~1.0** | needs coherent rollout images (≥20 steps, low noise) for the detector |
| clip_score | **yes 27→29** | modest, real |
| aesthetic (DRaFT) | **yes 5.9→9.6** | |
| aesthetic (FlowGRPO) | no (flat ~6.1) | RL signal too weak on a smooth reward SD3.5 is already decent at; DRaFT works |
| PickScore | no (flat ~22.5) | SD3.5 already near the PickScore ceiling - no headroom, not a bug |
| hps_v2 | no (flat ~0.28) | same story as PickScore (preference model, near ceiling) |

**The pattern:** RL increases a reward when the task has headroom (OCR, counting,
clip); it stays flat when the base model is already near the reward's ceiling
(PickScore, hps_v2). Confirmed across scales.

## Models

| Model | Status |
|-------|--------|
| SD3.5-Medium | **fully validated** (all of the above) |
| FLUX.1-schnell | runs on current code (FlowGRPO+aesthetic trains); noisy - schnell is 4-step distilled, a poor SDE-RL target. FLUX.1-dev (non-distilled) is the better target, not yet run. |

## Comprehensive grid

`report/GRID_RESULTS.md` + `report/grid_comparison.png`: {FlowGRPO, DDRL} ×
anchor-β {0, mid, high} × {counting, OCR}. FlowGRPO's KL is robust; DDRL collapses
without a sufficient data anchor (even β=0.3 only delays collapse on OCR).

## Known gaps / caveats

- **DDPO** is the one trainer that doesn't reliably improve rewards (weaker algorithm, not a bug).
- **DiffusionDPO re-run** needs Pick-a-Pic terms accepted on HF (gated dataset).
- **FLUX.1-dev** (proper non-distilled FLUX RL target) not yet run.
- Eval curves use few prompts and are noisy; read trends, not single points.
