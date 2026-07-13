"""Counting reward (GenEval-style) climbs under FlowGRPO - the reward-increase
result that PickScore couldn't show, because counting has real headroom.

SD3.5 / OWLv2 object-counting reward, FlowGRPO, group 12, 20 rollout steps,
noise 0.1. eval = count-accuracy on clean 28-step images (fraction of the target
count achieved). Baseline ~0.79 -> climbs toward 1.0.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ep = {2: 0.792, 4: 0.604, 6: 0.729, 8: 0.750, 10: 0.854, 12: 0.813, 14: 0.750,
      16: 0.854, 18: 0.813, 20: 0.938, 22: 1.000, 24: 0.875, 26: 0.917, 28: 0.917,
      30: 0.833, 32: 0.521, 34: 0.688, 36: 0.854, 38: 0.938, 40: 0.938}

fig, ax = plt.subplots(figsize=(7, 4.3))
xs, ys = list(ep), list(ep.values())
ax.plot(xs, ys, "-o", ms=4, color="#2ca02c", label="count accuracy (eval)")
ax.axhline(ys[0], ls="--", lw=1, color="#888", label=f"baseline ≈ {ys[0]:.2f}")
# simple running trend to show the climb through the eval noise
import statistics
trend = [statistics.mean(ys[max(0, i - 2):i + 1]) for i in range(len(ys))]
ax.plot(xs, trend, "-", lw=2, color="#1f77b4", alpha=0.7, label="3-point moving avg")
ax.set_xlabel("epoch")
ax.set_ylabel("count accuracy")
ax.set_title("RL increases a reward with headroom: object counting\nSD3.5 / OWLv2 counting reward, FlowGRPO (group 12, 20 steps, noise 0.1)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.5, 1.03)
fig.tight_layout()
fig.savefig("report/count_climb.png", dpi=140)
fig.savefig("report/figures/count_climb.pdf")
print("wrote report/count_climb.png and report/figures/count_climb.pdf")
