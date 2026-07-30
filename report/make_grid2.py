"""Corrected grid: FlowGRPO vs DDRL x anchor-beta x 2 rewards.

Reruns with use_monotonic_transform=False (the default) and per-reward noise
(OCR 0.7, counting 0.1). The earlier grid had the transform on, which forced
every group-normalized advantage negative and manufactured a DDRL "collapse."
With it off: DDRL(beta=0) tracks FlowGRPO(beta=0) (they are the same update),
and DDRL with a data anchor is stable and climbs. Nothing collapses.
"""
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(tag):
    xs, ys = [], []
    try:
        with open(f"report/grid_csv2/c2_{tag}_eval.csv") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) == 2:
                    xs.append(int(row[0]))
                    ys.append(float(row[1]))
    except FileNotFoundError:
        pass
    return xs, ys


def ema(ys, a=0.4):
    out, s = [], None
    for y in ys:
        s = y if s is None else a * y + (1 - a) * s
        out.append(s)
    return out


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
specs = [
    ("fg_b0", "FlowGRPO (KL β=0)", "#9ecae1", "-"),
    ("fg_b1", "FlowGRPO (KL β=1e-3)", "#4292c6", "-"),
    ("fg_b2", "FlowGRPO (KL β=4e-3)", "#08519c", "-"),
    ("ddrl_b0", "DDRL (data β=0)", "#d62728", "--"),
    ("ddrl_b1", "DDRL (data β=0.05)", "#ff7f0e", "--"),
    ("ddrl_b2", "DDRL (data β=0.2)", "#2ca02c", "--"),
]
for ax, reward, title in [(axes[0], "count", "Object counting"), (axes[1], "ocr", "OCR text")]:
    for pre, label, color, ls in specs:
        xs, ys = load(f"{pre}_{reward}")
        if xs:
            ax.plot(xs, ys, "o", ms=2.5, color=color, alpha=0.25)
            ax.plot(xs, ema(ys), ls, color=color, label=label, lw=1.9)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.35, 1.05)
axes[0].set_ylabel("eval reward (task accuracy)")
axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.suptitle("FlowGRPO vs DDRL across anchor strength on two rewards (corrected reruns)\n"
             "DDRL(β=0) tracks FlowGRPO(β=0) - same update; a data anchor stays stable. "
             "Nothing collapses.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("report/grid_comparison.png", dpi=140)
fig.savefig("report/figures/grid_comparison.pdf")
print("wrote report/grid_comparison.png and report/figures/grid_comparison.pdf")

print("\n=== SUMMARY (base / peak / final) ===")
for reward in ["count", "ocr"]:
    for pre, label, _, _ in specs:
        xs, ys = load(f"{pre}_{reward}")
        if not ys:
            print(f"{reward:5} {label:22} no data")
            continue
        print(f"{reward:5} {label:22} base={ys[0]:.2f} peak={max(ys):.2f} final={ys[-1]:.2f} n={len(ys)}")
