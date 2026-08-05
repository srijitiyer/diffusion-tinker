"""Was the RL term quiet in the FLUX runs the anchor collapsed? It was not.

The behavioural result is solid and is not what this questions. FLUX with
`ocr_exact` at group 6, everything matched but the anchor weight: three runs at
`data_beta=0.05` fall from a mean training reward of 0.694 to 0.306, and three at
`data_beta=0` start at the identical 0.694 and end there. That is a clean effect
of the anchor.

The explanation attached to it is what fails. The recorded mechanism is that a
binary reward at group 6 leaves FLUX's six samples scoring identically, so the
group-normalised advantage has no spread, `has_signal` is false, `rl_loss` is
zeroed, and the anchor is left as the entire gradient.

Two checks, in the order they should have been done.

`rl_loss` printing `0.0000` was the evidence, and it fires in 92.3% of epochs in
the arm that held and 92.9% in the arm that collapsed. A 0.5% difference. On its
own that is ambiguous rather than damning: the mechanism predicts both arms go
quiet and says only the anchored one suffers for it, so an equal rate is
consistent with it. `rl_loss` also tracks the mean advantage rather than its
spread, and group normalisation drives that mean to zero either way, so a printed
zero was never evidence about spread at all.

The decisive check is the group scores themselves, which the logs carry. A group
contributes nothing only when its rollouts all score the same, which for a bounded
reward means a group mean at 0.00 or 1.00; `has_signal` looks at spread across the
whole advantage tensor, so the term goes quiet only when *every* group in an epoch
is flat. Across all six runs the group means sit overwhelmingly at interior values
- 0.17, 0.33, 0.5, 0.67, 0.83, which are groups of six splitting 1/5, 2/4, 3/3 -
and the count of epochs in which every group was flat is zero out of fifty-three.

So `has_signal` was true in every logged epoch of both arms. The RL term was live
throughout. The anchor was never the only gradient, and the condition the fix
gates on was never reached in the runs the fix was written for.

What survives: the anchor at 0.05 hurts FLUX on this task, cleanly and
reproducibly across three seeds. Why is not established. Applying a regulariser to
a batch with no learning signal is still wrong and the `has_signal` gate is still
correct code, but it should not be described as the explanation for this result or
as the fix for it - on these runs the gate is open and the anchor is applied
exactly as before.

Run against `epochs.txt.gz`.
"""
import gzip
import os
import re
import sys
from collections import Counter, defaultdict

PP = re.compile(r"per_prompt=\[([^\]]*)\]")
VAL = re.compile(r"=(-?\d+\.\d+)")


def main(path):
    runs = defaultdict(lambda: {"hdr": "", "mr": [], "rl": [], "groups": Counter(),
                                "eps": 0, "flat_eps": 0, "flat_groups": 0, "n_groups": 0})
    opener = gzip.open if path.endswith(".gz") else open
    for line in opener(path, "rt"):
        p = line.split("@@")
        if len(p) < 4:
            continue
        tag, hdr, body = p[1], p[2], p[3]
        if "ocr_exact" not in hdr or "FLUX" not in hdr:
            continue
        r = runs[tag]
        r["hdr"] = hdr
        mr = re.search(r"mean_reward=([0-9.]+)", body)
        rl = re.search(r"rl_loss=([0-9.]+)", body)
        if mr:
            r["mr"].append(float(mr.group(1)))
        if rl:
            r["rl"].append(float(rl.group(1)))
        m = PP.search(body)
        if not m:
            continue
        vals = [float(v) for v in VAL.findall(m.group(1))]
        if not vals:
            continue
        r["eps"] += 1
        nflat = sum(1 for v in vals if v <= 0.0001 or v >= 0.9999)
        r["flat_groups"] += nflat
        r["n_groups"] += len(vals)
        if nflat == len(vals):
            r["flat_eps"] += 1
        for v in vals:
            r["groups"][v] += 1

    arms = defaultdict(list)
    print("FLUX, ocr_exact, group 6, by anchor weight\n")
    print(f"  {'run':<7} {'dbeta':<6} {'eps':>4} {'reward':>14}  {'flat groups':>12}"
          f"  {'ALL-flat epochs':>16}  {'rl_loss==0':>11}")
    for tag, r in sorted(runs.items()):
        m = re.search(r"dbeta=([0-9.]+)", r["hdr"])
        if not m or not r["mr"] or not r["n_groups"]:
            continue
        db = m.group(1)
        mr, rl = r["mr"], r["rl"]
        z = sum(1 for x in rl if x == 0.0)
        print(f"  {tag:<7} {db:<6} {r['eps']:>4} {mr[0]:>6.3f}->{mr[-1]:<7.3f}"
              f"  {r['flat_groups']:>4}/{r['n_groups']:<7}"
              f"  {r['flat_eps']:>7}/{r['eps']:<8}  {z:>4}/{len(rl):<6}")
        arms[db].append(r)

    print()
    tot_flat_eps = tot_eps = 0
    for db, es in sorted(arms.items()):
        f = sum(e["mr"][0] for e in es) / len(es)
        l = sum(e["mr"][-1] for e in es) / len(es)
        z = sum(sum(1 for x in e["rl"] if x == 0.0) for e in es)
        n = sum(len(e["rl"]) for e in es)
        fe = sum(e["flat_eps"] for e in es)
        ne = sum(e["eps"] for e in es)
        tot_flat_eps += fe
        tot_eps += ne
        print(f"  data_beta={db}: {len(es)} runs, reward {f:.3f} -> {l:.3f} ({l-f:+.3f}), "
              f"rl_loss==0 {z}/{n}={z/n:.1%}, all-flat epochs {fe}/{ne}")

    print(f"\n  Epochs in which every group was flat, both arms pooled: {tot_flat_eps}/{tot_eps}.")
    print("  has_signal thresholds spread across the whole advantage tensor, so it is")
    print("  false only in such an epoch. It was true throughout. The RL term was live,")
    print("  the anchor was never the only gradient, and the condition the fix gates on")
    print("  was not reached in the runs the fix was written for.")

    merged = Counter()
    for r in runs.values():
        merged.update(r["groups"])
    if merged:
        tot = sum(merged.values())
        interior = sum(n for v, n in merged.items() if 0.0001 < v < 0.9999)
        print(f"\n  group-mean distribution over all runs ({tot} groups), "
              f"{interior/tot:.0%} interior:")
        print("   ", {v: n for v, n in sorted(merged.items())})
        print("  Interior means are groups of six splitting their scores, which is spread.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(__file__), "epochs.txt.gz"))
