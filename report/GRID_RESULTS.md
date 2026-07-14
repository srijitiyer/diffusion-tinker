# Comprehensive grid: algorithms x anchor strength x reward model

Matched budget for every cell: SD3.5-Medium, group size 12, 20 rollout steps,
noise 0.1, lr 1e-4, per-prompt-std advantages, up to 60 epochs (LoRA rank 16).
Two "headroom" rewards where the base model is far from perfect: object counting
(OWLv2 detection vs. a target count) and OCR (rendering a target word).

Anchor knob: FlowGRPO uses the KL weight `kl_beta`; DDRL uses the data-loss
weight `data_beta` (with the monotonic transform). "β=0 / mid / high" =
FlowGRPO {0, 1e-3, 4e-3}, DDRL {0, 0.1, 0.3}.

Eval = task accuracy on held-out prompts, clean 28-step sampling (4 eval prompts,
so individual points are noisy; read the trend, not single points).

| reward | algorithm | β=0 | β=mid | β=high |
|--------|-----------|-----|-------|--------|
| count  | FlowGRPO (KL)   | climb 0.79→0.88 | climb 0.79→1.00 | climb 0.79→0.94 |
| count  | DDRL (data)     | **collapse** | **collapse** | climbs to 0.94, then drifts down |
| OCR    | FlowGRPO (KL)   | →1.00 | climb 0.83→1.00 | climb 0.75→1.00 |
| OCR    | DDRL (data)     | **collapse →0** | **collapse →0** | climbs to 1.00 (ep6), then **collapses →0.25** |

(peak / final per cell, plus the auto-classification, are printed by
`python report/make_grid.py`.)

## Takeaways

1. **The reward increases when the task has headroom.** Both counting and OCR
   climb under FlowGRPO (counting 0.79→~0.9-1.0, OCR to a perfect 1.0). This is
   the increase PickScore couldn't show, because SD3.5 is already near its
   PickScore ceiling.

2. **FlowGRPO's KL anchor is robust.** It climbs or stays high at every anchor
   strength, on both rewards, over the full run. It does not collapse.

3. **DDRL's data anchor helps but is weaker than the KL.** With `data_beta` = 0
   or small (0.1) the policy overshoots and collapses toward 0. A stronger anchor
   (0.3) lets it climb first - to 0.94 on counting and a perfect 1.0 on OCR - but
   it does **not** fully prevent collapse over a longer run: on OCR it crashes
   back to ~0.25 after peaking, and on counting it drifts down from 0.94. So the
   ordering is clear: at these settings the KL anchor is more stable than the
   data anchor, and DDRL needs a larger `data_beta` (or a task-matched anchor
   set - here the anchor images were generic base-model samples) for lasting
   stability. Worth a follow-up sweep of higher `data_beta`.

Figure: `report/grid_comparison.png` (`report/figures/grid_comparison.pdf`).
Per-cell eval curves in `report/grid_csv/`; regenerate with
`python report/make_grid.py`.
