"""Does the anchor collapse FLUX because the anchor images came from another model?

Three arms, FLUX on `ocr_exact` at group 6, everything matched but the anchor:

  fn42/43/44   data_beta=0            no anchor
  fo42/43/44   data_beta=0.05         anchored to 96 SD3.5-generated images
  fa42/43/44   data_beta=0.05         anchored to 96 FLUX-generated images

The third arm is the test. `gen_anchor_flux.py` regenerated the anchor set with
FLUX itself using the same eight prompts, the same twelve images per prompt, the
same steps and resolution as `gen_anchor.py`, so the only thing differing between
the two anchored arms is which model drew the pictures. What the images depict is
identical, and none of them contain text.

That separation is the point. The anchor-set explanation had two halves:

  1. distribution mismatch - FLUX pulled toward SD3.5's output manifold
  2. content mismatch - a text-rendering task pulled toward text-free images

Regenerating with FLUX removes (1) and leaves (2) exactly as it was. So if the
FLUX-anchored arm holds near baseline, (1) was the cause; if it collapses like
the SD3.5-anchored arm, (1) is not the cause and (2) is what survives.

A fourth arm then tests (2). `ft42/43/44` anchor on 96 FLUX-generated images in
which every picture is a sign or a written surface, matched to the `fa` set on
everything except what the images depict. If that arm holds near baseline, the
anchor is fine as long as it points at task-relevant data. If it collapses too,
both content explanations are gone and the anchor harms FLUX on this task
whatever the images contain.

Window handling: read at the longest window all nine runs reach, computed here
rather than fixed in advance, and also at a few shorter ones. This project has
twice produced a result that existed only at one K - a 4.2 SE giveback that
became 0.4 SE when one more run qualified, and a seeded difference that moved
from -0.0143 to -0.0088 an hour after the read. A result that only appears at one
window should not be believed, so every window is printed.

Rank separation is reported next to the SE because it does not depend on the SE
estimate, which is doing a lot of work at three seeds per arm.
"""
import csv
import os
import statistics
import sys

ARMS = {
    "unanchored (fn)": ["fn42", "fn43", "fn44"],
    "SD3.5-anchored (fo)": ["fo42", "fo43", "fo44"],
    "FLUX-anchored (fa)": ["fa42", "fa43", "fa44"],
    "text-anchored (ft)": ["ft42", "ft43", "ft44"],
}
BASELINE = 0.625


def load(csv_dir, tag):
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


def arm_mean(curves, tags, K):
    vals = []
    for t in tags:
        c = curves.get(t) or {}
        w = [c[e] for e in range(1, K + 1) if e in c]
        if len(w) == K:
            vals.append(sum(w) / K)
    return vals


def main(csv_dir):
    curves = {t: load(csv_dir, t) for tags in ARMS.values() for t in tags}
    for t, c in sorted(curves.items()):
        if c:
            last = max(c)
            print(f"  {t}: {len(c)} points, ep0={c.get(0, float('nan')):.3f}, "
                  f"ep{last}={c[last]:.3f}")
    print(f"\n  untrained baseline on this eval set and seed: {BASELINE}")

    reach = [max(c) for c in curves.values() if c]
    if not reach:
        print("  no curves")
        return
    Kmax = min(reach)
    windows = sorted({w for w in (4, 6, 8, 10, 12, Kmax) if 4 <= w <= Kmax})

    print(f"\n  common window across all nine runs: K={Kmax}\n")
    header = f"  {'K':>3}"
    for name in ARMS:
        header += f" {name:>21}"
    print(header + f" {'fa vs fn':>10} {'fa vs fo':>10}")

    for K in windows:
        cells, row = {}, f"  {K:>3}"
        for name, tags in ARMS.items():
            vals = arm_mean(curves, tags, K)
            cells[name] = vals
            row += f" {statistics.mean(vals):>14.4f} (n={len(vals)})" if vals else f" {'-':>21}"
        fa = cells["FLUX-anchored (fa)"]
        fn = cells["unanchored (fn)"]
        fo = cells["SD3.5-anchored (fo)"]

        def sep(a, b):
            if len(a) < 2 or len(b) < 2:
                return "-"
            se = (statistics.stdev(a) ** 2 / len(a) + statistics.stdev(b) ** 2 / len(b)) ** 0.5
            return f"{abs(statistics.mean(a) - statistics.mean(b)) / se:.1f} SE" if se else "-"

        print(row + f" {sep(fa, fn):>10} {sep(fa, fo):>10}")

    K = Kmax
    print(f"\n  per-seed at K={K}:")
    for name, tags in ARMS.items():
        vals = arm_mean(curves, tags, K)
        print(f"    {name:<21} {[round(v, 3) for v in sorted(vals)]}")

    fa = arm_mean(curves, ARMS["FLUX-anchored (fa)"], K)
    fn = arm_mean(curves, ARMS["unanchored (fn)"], K)
    fo = arm_mean(curves, ARMS["SD3.5-anchored (fo)"], K)
    if fa and fn:
        wins = sum(1 for x in fa for y in fn if x > y)
        print(f"\n  rank: FLUX-anchored beats unanchored in {wins}/{len(fa)*len(fn)} pairs")
    if fa and fo:
        wins = sum(1 for x in fa for y in fo if x > y)
        print(f"  rank: FLUX-anchored beats SD3.5-anchored in {wins}/{len(fa)*len(fo)} pairs")

    print("\n  Reading, fixed before these curves existed: the FLUX-anchored arm holding")
    print("  near baseline would mean distribution mismatch caused the collapse; it")
    print("  falling like the SD3.5-anchored arm means mismatch is not the cause and the")
    print("  text-free content of the anchor images is what remains to explain.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(__file__), "anchorsrc_csv"))
