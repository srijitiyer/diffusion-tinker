"""Anchor-strength sweep on the one task with headroom.

The earlier grid (report/make_grid2.py, `c2_*`) did sweep beta, but it is
retracted: it evaluated on the first four *training* prompts with no eval seed,
and it ran on counting 2-4 objects where SD3.5 already scores 0.792 and every
arm saturates at 1.000. A sweep whose arms all reach the ceiling measures the
ceiling.

This one holds everything fixed except the regularizer weight and runs on
counting 6-9 objects, where the untrained model scores 0.541 and there is room
to separate the arms. Held-out prompts, fixed eval seed, 12 eval prompts x 2
samples.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = os.path.join(os.path.dirname(__file__), "beta_csv")

# tag, label, colour
ARMS = [
    ("hb0", r"DDRL  data $\beta$=0  (= FlowGRPO)", "#999999"),
    ("hb001", r"DDRL  data $\beta$=0.01", "#9ecae1"),
    ("hb005", r"DDRL  data $\beta$=0.05", "#4292c6"),
    ("hb03", r"DDRL  data $\beta$=0.3", "#08519c"),
    ("hk001", r"FlowGRPO  reverse-KL $\beta$=0.01", "#d62728"),
]


def load(tag):
    xs, ys = [], []
    path = os.path.join(CSV_DIR, f"{tag}_eval.csv")
    if not os.path.exists(path):
        return xs, ys
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                xs.append(int(row[0]))
                ys.append(float(row[1]))
    return xs, ys


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    baseline = None
    summary = []
    for tag, label, color in ARMS:
        xs, ys = load(tag)
        if not xs:
            print(f"  {tag}: no data yet")
            continue
        if baseline is None:
            baseline = ys[0]
        ls = "--" if tag == "hk001" else "-"
        ax.plot(xs, ys, ls, color=color, label=label, lw=1.9, marker="o", ms=3)
        # last five points rather than the single best: with SEM ~0.07 per point
        # the max over fifteen points is mostly a draw from the noise.
        tail = ys[-5:]
        summary.append((tag, ys[0], max(ys), sum(tail) / len(tail), len(ys)))

    if baseline is not None:
        ax.axhline(baseline, color="black", lw=0.9, ls=":", alpha=0.7)
        ax.text(0.99, baseline + 0.006, f"untrained {baseline:.3f}", ha="right",
                fontsize=8, color="black", transform=ax.get_yaxis_transform())

    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out reward (counting accuracy, 6-9 objects)")
    ax.set_title("Anchor strength on a task with headroom\n"
                 "SD3.5-medium, 12 held-out prompts x 2 samples, fixed eval seed",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "beta_sweep.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    print(f"\n{'arm':<8} {'ep0':>7} {'best':>7} {'last5':>7} {'n':>4}")
    for tag, first, best, tail, n in summary:
        print(f"{tag:<8} {first:>7.3f} {best:>7.3f} {tail:>7.3f} {n:>4}")


if __name__ == "__main__":
    main()
