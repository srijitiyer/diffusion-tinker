"""Is the quiet-RL condition a property of the model, or of the reward?

FAILED ATTEMPT, kept for the record. It answered no question it set out to
answer, and it is here because the way it failed is what turned up a real error
elsewhere. Two things went wrong and both are worth not repeating.

The premise was wrong. This was written suspecting that the "SD3.5 never reaches
the quiet condition, FLUX reaches it in 22 of 28 epochs" comparison had varied
model and reward together - FLUX on exact-match OCR against SD3.5 on graded
counting - so that a reward effect would be indistinguishable from a model one.
That suspicion was simply false: both sides of the comparison ran `ocr_exact` at
group 6, as VALIDATION.md says plainly. Reading the source before building the
challenge would have cost a minute.

The measure was too coarse. `per_prompt` logs a group's *mean* reward at two
decimals, so a group scoring [0.02, 0, 0, 0, 0, 0] prints 0.00 and reads as flat
when it is not. Endpoint-pinning is only exact for a binary reward at full
precision, and it is not available at two. Batch-level flatness came out as 0 of
1315 epochs across every run on disk, which is not a finding about diversity but
a statement that this proxy cannot resolve the question.

An earlier version of it was also outright buggy: it applied the endpoint test to
`aesthetic`, `clip_score` and `pickscore`, which score around 6, 29 and 23, so
every group cleared `>= 0.9999` and all three read as 100% flat. All three are
SD3.5 runs, so the artifact landed entirely on one arm of the comparison it was
supposed to adjudicate.

What it did establish, by forcing a look at the raw fields: `rl_loss` sits at
`-mean_advantage` to four decimals throughout these logs, so it tracks the mean
advantage and not the spread. Group normalisation centres advantages, which
makes that mean near zero whether or not there is signal. The quiet-epoch counts
in AUDIT.md and VALIDATION.md were read off `rl_loss` printing as `0.0000`, and
four decimals means `< 0.00005` rather than zero - so those counts are upper
bounds on quiet epochs, not measurements. Both documents now say so, and both
numbers were widened from the 46 epochs first sampled to all 194 on disk, where
"SD3.5 never" became "SD3.5 in at most 12 of 135".

Advantage spread is not logged anywhere. Deciding this properly needs that field
recorded during a run; it cannot be recovered afterwards from what is on disk.

The measure. Each epoch line logs `per_prompt`, the mean reward over that
prompt's group of rollouts. Advantages are computed within a group, so a group
whose rollouts all scored the same contributes nothing regardless of what they
scored. For a reward bounded on [0, 1], a group mean of exactly 0.00 or exactly
1.00 can only arise when every rollout in it scored identically. So:

    flat fraction = share of prompt-groups pinned at 0.00 or 1.00

For a binary reward this is exact: mean 0 iff all zero, mean 1 iff all one. For
a graded reward it is a lower bound, since a graded group can be flat at an
interior value too. The bound runs in the conservative direction for the
question being asked - graded rewards can only be flatter than measured, so if
graded rewards still come out much less flat than sparse ones, the gap is real.

What would distinguish the two explanations:

  - Flatness tracks the reward and not the model (SD3.5 flat on exact-match OCR,
    FLUX not flat on a graded reward) -> the 0-of-18 figure measured the reward,
    the confound is real, and the corrected claim needs correcting again.
  - Flatness tracks the model regardless of reward -> the correction stands as
    written and this file records a failed challenge to it.
  - Cells missing -> say so. The confound is only resolvable where both models
    have been run on both reward types.
"""
import os
import re
import sys
from collections import defaultdict

LINE = re.compile(r"@@(?P<tag>[^@]+)@@(?P<hdr>[^@]+)@@(?P<body>.*)")
PP = re.compile(r"per_prompt=\[(?P<items>[^\]]*)\]")
VAL = re.compile(r"=(-?\d+\.\d+)")


# The endpoint test only means anything for a reward bounded on [0, 1]. The
# first version of this file applied it to everything and reported aesthetic,
# clip_score and pickscore as 100% flat, which is not a finding: those score
# around 6, 29 and 23 respectively, so every group cleared the >= 0.9999 test.
# That artifact was inflating SD3.5's graded cell, since all three are SD3.5
# runs. They are excluded rather than rescaled - an endpoint is only meaningful
# where the reward actually has one.
BOUNDED = {"count", "count_hard", "ocr", "ocr_exact", "ocr_hard"}


def reward_kind(name):
    """Sparse rewards have a floor most rollouts sit on; graded ones do not.

    `ocr_exact`/`ocr_hard` are string equality, so a group is all-or-nothing.
    `count`/`count_hard` are fractions of objects counted correctly, which take
    interior values.
    """
    return "sparse" if "ocr" in name else "graded"


def main(path):
    cells = defaultdict(lambda: {"flat": 0, "total": 0, "quiet": 0, "epochs": 0, "runs": set()})
    for raw in open(path):
        m = LINE.match(raw.strip())
        if not m:
            continue
        hdr, body = m.group("hdr"), m.group("body")
        rw = re.search(r"reward=(\S+)", hdr)
        md = re.search(r"model=(\S+)", hdr)
        if not rw or not md:
            continue
        model = "FLUX" if "FLUX" in md.group(1) else "SD3.5"
        reward = rw.group(1)
        pp = PP.search(body)
        if not pp:
            continue
        vals = [float(v) for v in VAL.findall(pp.group("items"))]
        if not vals:
            continue
        if reward not in BOUNDED:
            continue
        key = (model, reward, reward_kind(reward))
        c = cells[key]
        c["runs"].add(m.group("tag"))
        nflat = sum(1 for v in vals if v <= 0.0001 or v >= 0.9999)
        c["total"] += len(vals)
        c["flat"] += nflat
        # The batch-level unit, which is what `has_signal` actually tests: it
        # looks at spread across the whole advantage tensor, so the term only
        # goes quiet when EVERY group in the epoch is flat. The "22 of 28
        # epochs" figure is this quantity, not the per-group share, and the two
        # are not interchangeable.
        c["epochs"] += 1
        if nflat == len(vals):
            c["quiet"] += 1

    print("Share of prompt-groups pinned at a reward endpoint (no within-group")
    print("spread, so no gradient). Exact for sparse rewards, a lower bound for")
    print("graded ones.\n")
    print(f"  {'model':<7} {'reward':<11} {'kind':<7} {'groups':>7} {'flat':>6} {'share':>7}   {'epochs':>6} {'quiet':>5} {'share':>7}  runs")
    by_kind = defaultdict(lambda: [0, 0, 0, 0])
    for (model, reward, kind), c in sorted(cells.items()):
        share = c["flat"] / c["total"]
        qshare = c["quiet"] / c["epochs"]
        print(f"  {model:<7} {reward:<11} {kind:<7} {c['total']:>7} {c['flat']:>6} {share:>6.1%}   "
              f"{c['epochs']:>6} {c['quiet']:>5} {qshare:>6.1%}  {len(c['runs'])}")
        agg = by_kind[(model, kind)]
        agg[0] += c["flat"]; agg[1] += c["total"]
        agg[2] += c["quiet"]; agg[3] += c["epochs"]

    print("\n  collapsed to model x reward-kind (batch-quiet is the one that")
    print("  corresponds to has_signal being false):")
    grid = {}
    for (model, kind), (f, t, q, e) in sorted(by_kind.items()):
        grid[(model, kind)] = q / e
        print(f"    {model:<7} {kind:<7} groups {f/t:>6.1%} ({f}/{t})   batch-quiet {q/e:>6.1%} ({q}/{e})")

    print()
    have = set(grid)
    need = {(m, k) for m in ("SD3.5", "FLUX") for k in ("sparse", "graded")}
    missing = need - have
    if missing:
        print("  cells with no data: " + ", ".join(f"{m}/{k}" for m, k in sorted(missing)))
        print("  the confound is not fully resolvable from existing logs.")

    # Within a model, does the reward kind move flatness? That is the comparison
    # the retracted claim never made.
    for model in ("SD3.5", "FLUX"):
        s, g = grid.get((model, "sparse")), grid.get((model, "graded"))
        if s is not None and g is not None:
            print(f"  {model}: sparse {s:.1%} vs graded {g:.1%} -> reward moves batch-quiet by {abs(s-g):.1%}")
    # And across models at matched reward kind.
    for kind in ("sparse", "graded"):
        a, b = grid.get(("SD3.5", kind)), grid.get(("FLUX", kind))
        if a is not None and b is not None:
            print(f"  {kind}: SD3.5 {a:.1%} vs FLUX {b:.1%} -> model moves batch-quiet by {abs(a-b):.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(__file__), "epochs.txt"))
