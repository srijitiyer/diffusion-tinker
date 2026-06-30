"""DDRL anchor comparison figure: FlowGRPO vs DDRL(beta=0) vs DDRL(anchor),
fully matched budget (SD3.5, PickScore, group 12, 8 prompts, noise 0.7,
lr 1e-4, 60 epochs). All three runs differ only as labeled.

Reproduces Haotian's claim: FlowGRPO sits stable at X; DDRL with beta=0 (no
anchor) overshoots X then collapses; DDRL with a data anchor stays at X without
collapsing. The only difference between the collapse and the stable DDRL run is
the data anchor (same lr / budget).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# eval PickScore vs epoch, all at lr 1e-4, group 12, matched budget
fg = {2: 22.782, 4: 22.819, 6: 22.501, 8: 22.677, 10: 22.674, 12: 22.893, 14: 22.386,
      16: 22.468, 18: 22.710, 20: 22.239, 22: 22.544, 24: 22.633, 26: 22.782, 28: 22.667, 30: 23.165}
d0 = {2: 23.169, 4: 22.830, 6: 22.665, 8: 22.626, 10: 21.119, 12: 16.246, 14: 16.047,
      16: 16.822, 18: 16.166, 20: 16.433, 22: 17.145, 24: 17.877, 26: 17.358, 28: 18.854,
      30: 19.174, 32: 18.402, 34: 16.580, 36: 16.621}
db = {2: 22.591, 4: 22.679, 6: 22.635, 8: 22.822, 10: 22.530, 12: 22.447, 14: 22.498,
      16: 22.668, 18: 22.953, 20: 22.988, 22: 22.537, 24: 22.525, 26: 21.875, 28: 22.337, 30: 22.009}

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(list(fg), list(fg.values()), "-o", ms=3, color="#1f77b4", label="FlowGRPO + KL anchor  →  stable at X")
ax.plot(list(db), list(db.values()), "-o", ms=3, color="#2ca02c", label="DDRL + data anchor (β=0.3)  →  stable at X")
ax.plot(list(d0), list(d0.values()), "-o", ms=3, color="#d62728", label="DDRL, no anchor (β=0)  →  overshoots, collapses")
ax.set_xlabel("epoch")
ax.set_ylabel("eval PickScore")
ax.set_title("The data anchor prevents the collapse pure RL suffers\nSD3.5 / PickScore, matched budget (group 12, 8 prompts, lr 1e-4)")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(15.5, 23.6)
fig.tight_layout()
fig.savefig("report/ddrl_comparison.png", dpi=140)
fig.savefig("report/figures/ddrl_comparison.pdf")
print("wrote report/ddrl_comparison.png and report/figures/ddrl_comparison.pdf")
