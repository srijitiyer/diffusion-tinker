# Comprehensive grid: algorithms x anchor strength x reward model

Matched budget for every cell: SD3.5-Medium, group size 12, 20 rollout steps,
noise 0.1, lr 1e-4, per-prompt-std advantages, 60 epochs (LoRA rank 16).
Two "headroom" rewards where the base model is far from perfect: object counting
(OWLv2 detection vs. a target count) and OCR (rendering a target word).

Anchor knob: FlowGRPO uses the KL weight `kl_beta`; DDRL uses the data-loss
weight `data_beta` (with the monotonic transform). "β=0 / mid / high" =
FlowGRPO {0, 1e-3, 4e-3}, DDRL {0, 0.1, 0.3}.

Eval = task accuracy on held-out prompts, clean 28-step sampling.

| reward | algorithm | β=0 | β=mid | β=high |
|--------|-----------|-----|-------|--------|
| count  | FlowGRPO (KL)   | climb 0.79→0.88 | climb 0.79→1.00 | climb 0.79→0.92 |
| count  | DDRL (data)     | **collapse** | **collapse** | climb 0.79→0.94 |
| OCR    | FlowGRPO (KL)   | →1.00 | climb 0.83→1.00 | climb 0.75→1.00 |
| OCR    | DDRL (data)     | **collapse →0** | **collapse →0** | climb 0.33→0.75 |

## Takeaways

1. **The reward increases when the task has headroom.** Both counting and OCR
   climb under FlowGRPO (counting 0.79→~0.9-1.0, OCR to a perfect 1.0). This is
   the increase PickScore couldn't show, because SD3.5 is already near its
   PickScore ceiling.

2. **FlowGRPO's KL anchor is robust.** It climbs or stays stable at every anchor
   strength, on both rewards. It does not collapse.

3. **DDRL needs a sufficient data anchor.** With `data_beta` = 0 or too small
   (0.1) the policy overshoots and collapses (reward crashes toward 0). A strong
   enough anchor (0.3) matches FlowGRPO's behavior without collapsing. This is
   the anchor effect the DDRL formulation is about, and it holds on both rewards.

Figure: `report/grid_comparison.png` (`report/figures/grid_comparison.pdf`).
Regenerate from the per-cell eval curves in `report/grid_csv/` with
`python report/make_grid.py`.
