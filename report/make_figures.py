import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = os.path.join(os.path.dirname(__file__), "csv")
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10.5,
})
NAVY = "#1f3b73"

def series(name, xk, yk):
    p = os.path.join(CSV, name)
    if not os.path.exists(p):
        return [], []
    xs, ys = [], []
    for r in csv.DictReader(open(p)):
        v = r.get(yk, "")
        if v in ("", None):
            continue
        try:
            ys.append(float(v)); xs.append(float(r[xk]))
        except ValueError:
            continue
    return xs, ys

fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))

def plot(a, name, xk, yk, title, xlab, ylab, baseline=None, bl_lab=None):
    xs, ys = series(name, xk, yk)
    if not xs:
        a.set_title(f"{title} (no data)"); return
    a.plot(xs, ys, marker="o", ms=2.5, lw=1.4, color=NAVY)
    if baseline is not None:
        a.axhline(baseline, ls="--", lw=1, color="#aa3333")
        a.text(xs[-1], baseline, " " + bl_lab, va="bottom", ha="right", fontsize=8, color="#aa3333")
    a.set_title(title); a.set_xlabel(xlab); a.set_ylabel(ylab)

plot(ax[0][0], "draft.csv", "epoch", "train", "DRaFT - aesthetic reward", "epoch", "reward")
plot(ax[0][1], "ocr_eval.csv", "epoch", "eval", "FlowGRPO - OCR eval accuracy", "epoch", "accuracy",
     baseline=0.845, bl_lab="FlowGRPO paper 0.845")
plot(ax[0][2], "ddrl.csv", "epoch", "train", "DDRL - aesthetic reward", "epoch", "reward")
plot(ax[1][0], "dpo.csv", "step", "implicit_reward", "DiffusionDPO - implicit reward", "step", "implicit reward")
plot(ax[1][1], "flux_eval.csv", "epoch", "eval", "FLUX.1-schnell - aesthetic eval", "epoch", "reward")
ax[1][2].axis("off")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "repro_curves.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "repro_curves.png"), dpi=150, bbox_inches="tight")
print("wrote", os.path.join(OUT, "repro_curves.pdf"))
