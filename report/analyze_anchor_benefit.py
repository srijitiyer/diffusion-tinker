"""The anchor's benefit on a sparse reward: SD3.5, ocr_exact, five seeds per arm.

This is the one clearly positive result for the data anchor in this project, and
until now it was the only headline number without committed data behind it. The
figures were recorded in VALIDATION.md and the report but could not be re-derived
by anyone reading them. The curves are now in `anchorbenefit_csv/` and this
recomputes the table from them.

Design: SD3.5-medium, `ocr_exact` at group 6, everything matched but
`data_beta`. `sa*` anchored, `sb*` unanchored, seeds 42-46.

Read at four windows rather than one, because a result that only exists at a
single K is the failure mode that produced this project's retracted 4.2 SE
giveback finding. Here the effect grows slightly with the window (0.083 to 0.102)
while the SE stays at 2.6-3.0 and the rank separation does not move at all, which
is the opposite pattern and the reason this one is trusted.

Rank separation is reported alongside the SE deliberately. It counts pairs where
an anchored seed beats an unanchored one, out of 25, and it does not depend on
the standard-error estimate - so it survives the small-n objection that sinks
several other results here.

Caveat that belongs with the number: the anchor images are SD3.5 samples, so this
is the in-distribution case. The same anchor set applied to FLUX is out of
distribution and harms it. This result is evidence that the anchor works when
anchored to the right thing, not that it works unconditionally.
"""
import csv
import os
import statistics
import sys

ARMS = {"anchored": ["sa42", "sa43", "sa44", "sa45", "sa46"],
        "unanchored": ["sb42", "sb43", "sb44", "sb45", "sb46"]}
WINDOWS = (5, 7, 9, 12)


def load(csv_dir, tag):
    """Longest rotation file for a tag, as {epoch: eval}."""
    best = {}
    for name in os.listdir(csv_dir):
        if not name.startswith(tag + "_eval"):
            continue
        rows = {}
        with open(os.path.join(csv_dir, name)) as f:
            for r in csv.DictReader(f):
                try:
                    rows[int(r["epoch"])] = float(r["eval"])
                except (KeyError, ValueError):
                    continue
        if len(rows) > len(best):
            best = rows
    return best


def main(csv_dir):
    curves = {t: load(csv_dir, t) for arm in ARMS.values() for t in arm}
    have = {t: c for t, c in curves.items() if c}
    print(f"loaded {len(have)} curves from {os.path.basename(csv_dir)}\n")
    print(f"  {'window':>7} {'anchored':>9} {'unanchored':>11} {'diff':>8} {'SE':>7} {'rank':>7}")

    for K in WINDOWS:
        per_arm = {}
        for arm, tags in ARMS.items():
            vals = []
            for t in tags:
                c = curves.get(t) or {}
                w = [c[e] for e in range(1, K + 1) if e in c]
                if len(w) == K:
                    vals.append(sum(w) / K)
            per_arm[arm] = vals
        a, u = per_arm["anchored"], per_arm["unanchored"]
        if len(a) < 2 or len(u) < 2:
            print(f"  {K:>7}   not enough runs reach this window ({len(a)} vs {len(u)})")
            continue
        ma, mu = statistics.mean(a), statistics.mean(u)
        sea = statistics.stdev(a) / len(a) ** 0.5
        seu = statistics.stdev(u) / len(u) ** 0.5
        se = (sea ** 2 + seu ** 2) ** 0.5
        wins = sum(1 for x in a for y in u if x > y)
        print(f"  {K:>7} {ma:>9.4f} {mu:>11.4f} {ma-mu:>+8.3f} {abs(ma-mu)/se:>6.1f}"
              f"  {wins:>3}/{len(a)*len(u)}")

    K = 12
    for arm, tags in ARMS.items():
        vals = []
        for t in tags:
            c = curves.get(t) or {}
            w = [c[e] for e in range(1, K + 1) if e in c]
            if len(w) == K:
                vals.append(round(sum(w) / K, 3))
        print(f"\n  per-seed at K={K}, {arm}: {sorted(vals)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(__file__), "anchorbenefit_csv"))
