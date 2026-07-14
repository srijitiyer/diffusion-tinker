"""Comprehensive grid: algorithms x anchor-beta x 2 reward models.

FlowGRPO (KL anchor) vs DDRL (data anchor), each at anchor beta in {0, mid, high},
on two headroom rewards (object counting, OCR text). Matched budget: SD3.5,
group 12, 20 rollout steps, noise 0.1, lr 1e-4, per-prompt-std advantages.

Shows: FlowGRPO's KL is robust (climbs/stable at every beta). DDRL collapses
without a sufficient data anchor (beta 0 and small beta crash to ~0); a strong
enough anchor (beta 0.3) stays stable and climbs. Holds on both rewards.
"""
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(tag):
    xs, ys = [], []
    try:
        with open(f"report/grid_csv/{tag}_eval.csv") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) == 2:
                    xs.append(int(row[0]))
                    ys.append(float(row[1]))
    except FileNotFoundError:
        pass
    return xs, ys


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
specs = [
    ("fg_b0", "FlowGRPO (KL β=0)", "#9ecae1", "-"),
    ("fg_b1", "FlowGRPO (KL β=mid)", "#4292c6", "-"),
    ("fg_b2", "FlowGRPO (KL β=high)", "#08519c", "-"),
    ("ddrl_b0", "DDRL (anchor β=0)", "#d62728", "--"),
    ("ddrl_b1", "DDRL (anchor β=0.1)", "#ff7f0e", "--"),
    ("ddrl_b2", "DDRL (anchor β=0.3)", "#2ca02c", "-"),
]
for ax, reward, title in [(axes[0], "count", "Object counting"), (axes[1], "ocr", "OCR text")]:
    for pre, label, color, ls in specs:
        xs, ys = load(f"{pre}_{reward}")
        if xs:
            ax.plot(xs, ys, ls, marker="o", ms=3, color=color, label=label, lw=1.8)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.03, 1.05)
axes[0].set_ylabel("eval reward (task accuracy)")
axes[1].legend(loc="center right", fontsize=8, framealpha=0.9)
fig.suptitle("FlowGRPO vs DDRL across anchor strength on two rewards\n"
             "FlowGRPO's KL is robust (climbs/stable); DDRL collapses without a strong anchor - "
             "even β=0.3 only delays it",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("report/grid_comparison.png", dpi=140)
fig.savefig("report/figures/grid_comparison.pdf")
print("wrote report/grid_comparison.png and report/figures/grid_comparison.pdf")

# summary table
print("\n=== SUMMARY (base / peak / final / verdict) ===")
for reward in ["count", "ocr"]:
    for pre, label, _, _ in specs:
        xs, ys = load(f"{pre}_{reward}")
        if not ys:
            print(f"{reward:5} {label:22} no data")
            continue
        base, peak, final = ys[0], max(ys), ys[-1]
        # headroom rewards: working runs reach ~0.9-1.0. A run ending well below
        # that has degraded. Distinguish a run that climbed first then crashed
        # (peak high, final low) from one that never got off the ground.
        if final >= 0.85:
            v = "climb/stable-high"
        elif final < 0.5 and peak >= 0.7:
            v = "climb-then-COLLAPSE"
        elif final < 0.5:
            v = "COLLAPSE"
        else:
            v = "partial (climbs, degrades)"
        print(f"{reward:5} {label:22} base={base:.2f} peak={peak:.2f} final={final:.2f}  {v}")
