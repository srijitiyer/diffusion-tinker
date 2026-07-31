"""Held-out curves on audited code.

Replaces make_grid.py / make_grid2.py, whose inputs came from an eval that used
four training prompts with no fixed seed and so measured resampling noise. Every
curve here is a held-out prompt set at a fixed eval seed, and every panel starts
from an untrained point measured the same way, so "it climbed" is checkable.

Left panel: OCR, where the algorithms are compared. Right panel: the reward
sweep, each normalised to its own untrained baseline so rewards on different
scales (0-1 accuracy, ~23 PickScore, ~5.7 aesthetic) can share an axis.
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = pathlib.Path(__file__).parent / "verified_csv"
OCR_BASELINE = 0.300
OCR_BASELINE_SEM = 0.064

# tag, label, colour
ALGOS = [
    ("v_ddrl_anchor_hi", "DDRL anchor $\\beta$=0.05", "#1b6ca8"),
    ("v_ddrl_anchor_lo", "DDRL anchor $\\beta$=0.01", "#5aa9d6"),
    ("v_fg_b0", "FlowGRPO $\\beta$=0", "#c8553d"),
]
# tag, label, colour, untrained baseline (measured at epoch 0 by the run itself)
REWARDS = [
    ("r_aes", "aesthetic (FlowGRPO)", "#1b6ca8", None),
    ("r_draft_ev2", "aesthetic (DRaFT)", "#3d9970", None),
    ("r_count_lo", "counting (lr 3e-5)", "#5c7457", None),
    ("r_clip", "CLIP-score", "#e0a458", None),
    ("r_pick", "PickScore", "#c8553d", None),
    ("r_hps", "HPS v2", "#8c6bb1", None),
]


def load(tag):
    path = CSV_DIR / f"{tag}_eval.csv"
    if not path.exists():
        return [], []
    ep, val = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            val.append(float(row["eval"]))
    return ep, val


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

# ---- OCR, algorithms ----
ax1.axhspan(OCR_BASELINE - OCR_BASELINE_SEM, OCR_BASELINE + OCR_BASELINE_SEM, color="0.86", zorder=0)
ax1.axhline(OCR_BASELINE, color="0.45", ls="--", lw=1.2, zorder=1)
ax1.text(0.4, OCR_BASELINE + 0.022, f"untrained {OCR_BASELINE:.3f}", fontsize=8.5, color="0.35")

for tag, label, colour in ALGOS:
    ep, val = load(tag)
    if val:
        ax1.plot(ep, val, color=colour, lw=1.9, marker="o", ms=4, label=label)

ax1.set_xlabel("epoch")
ax1.set_ylabel("held-out OCR reward")
ax1.set_title("OCR: canonical DDRL vs FlowGRPO", fontsize=10.5)
ax1.set_ylim(0.20, 1.0)
ax1.grid(alpha=0.25, lw=0.6)
ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

# ---- reward sweep, each relative to its own untrained point ----
ax2.axhline(0.0, color="0.45", ls="--", lw=1.2, zorder=1)
ax2.text(0.98, 0.03, "each reward's own untrained value", fontsize=8.5, color="0.35",
         ha="right", va="bottom", transform=ax2.get_yaxis_transform())

for tag, label, colour, _ in REWARDS:
    ep, val = load(tag)
    if not val:
        continue
    base = val[0]
    rel = [(v - base) / abs(base) for v in val]
    ax2.plot(ep, rel, color=colour, lw=1.8, marker="o", ms=3.5, label=label)

ax2.set_xlabel("epoch")
ax2.set_ylabel("change from untrained (fraction)")
ax2.set_title("Rewards: which ones have headroom", fontsize=10.5)
ax2.grid(alpha=0.25, lw=0.6)
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.92, ncol=2)

fig.tight_layout()
out_png = pathlib.Path(__file__).parent / "verified_curves.png"
out_pdf = pathlib.Path(__file__).parent / "figures" / "verified_curves.pdf"
out_pdf.parent.mkdir(exist_ok=True)
fig.savefig(out_png, dpi=150)
fig.savefig(out_pdf)
print(f"wrote {out_png.name} and figures/{out_pdf.name}")
