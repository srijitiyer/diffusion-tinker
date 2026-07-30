"""Held-out OCR curves on audited code.

Replaces make_grid.py / make_grid2.py, whose input curves came from an eval that
used four training prompts with no fixed seed and so measured resampling noise.
These curves are 22 held-out prompts x 2 samples at a fixed eval seed (n=44),
against an untrained baseline measured on the same prompts and seed.
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = pathlib.Path(__file__).parent / "verified_csv"
BASELINE = 0.300
BASELINE_SEM = 0.064
EVAL_EVERY = 2

RUNS = [
    ("v_ddrl_mono", "DDRL + anchor + transform", "#1b6ca8"),
    ("v_fg_b0", "FlowGRPO (beta=0)", "#c8553d"),
    ("v_ddrl_b0", "DDRL (beta=0)", "#e0a458"),
    ("v_ddpo", "DDPO", "#5c7457"),
]


def load(tag):
    path = CSV_DIR / f"{tag}_eval.csv"
    if not path.exists():
        return [], []
    epochs, vals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            vals.append(float(row["eval"]))
    return epochs, vals


fig, ax = plt.subplots(figsize=(7.2, 4.2))

ax.axhspan(BASELINE - BASELINE_SEM, BASELINE + BASELINE_SEM, color="0.85", zorder=0)
ax.axhline(BASELINE, color="0.45", ls="--", lw=1.2, zorder=1)
ax.text(0.15, BASELINE + 0.018, f"untrained baseline {BASELINE:.3f}", fontsize=8.5, color="0.35")

retired = 0
for tag, label, color in RUNS:
    epochs, vals = load(tag)
    if not vals:
        continue
    if len(vals) == 1:
        # A single point is a run retired once its audit point was made. Draw it
        # as a marker so it does not read as a flat line, and nudge each one along
        # x so the retired runs (which landed within 0.001 of each other) stay
        # individually visible.
        retired += 1
        ax.plot([e + 0.28 * retired for e in epochs], vals, color=color,
                marker="D", ms=7, ls="none", label=f"{label} (retired)")
    else:
        ax.plot(epochs, vals, color=color, lw=1.9, marker="o", ms=4.5, label=label)

ax.set_xlabel("epoch")
ax.set_ylabel("held-out OCR reward")
ax.set_title("Held-out OCR on audited code (22 prompts x 2, fixed eval seed, n=44)", fontsize=10.5)
ax.set_ylim(0.20, 1.0)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
fig.tight_layout()

out_png = pathlib.Path(__file__).parent / "verified_curves.png"
out_pdf = pathlib.Path(__file__).parent / "figures" / "verified_curves.pdf"
out_pdf.parent.mkdir(exist_ok=True)
fig.savefig(out_png, dpi=150)
fig.savefig(out_pdf)
print(f"wrote {out_png} and {out_pdf}")
