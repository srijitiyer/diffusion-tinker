"""Pre-registered drift analysis: does the anchor hold the policy up late?

The completed seeded comparison answered a narrow question - whether anchor
strength changes held-out reward over the first six epochs - and found nothing at
a resolution of about 0.03. Its own stated limitation is the interesting part:
the forward-KL anchor exists to keep the policy near the data manifold over long
training, and six epochs is early for drift to appear at all. So the null there
does not speak to what the anchor is for.

This is the follow-up, written before the long runs exist so the statistic cannot
be chosen once the shape of the curves is visible. It reuses the seeded runs; the
only thing needed is for them to get longer.

Fixed in advance:

  - The unit is a run, as before. Seeds are the replicates.
  - Each run contributes TWO numbers, both over epochs 1..K-1: DRIFT, the late
    third mean minus the early third mean, and GIVEBACK, the peak minus the
    final value. See `giveback` for why one is not enough - the second was added
    before any clean run had a point, after the first was shown blind to the
    peak-then-collapse shape.
    Negative means the run gave back what it gained. This is the quantity the
    anchor should act on, and it is the same shape as the give-back already
    measured on SFT (held-out minimum 0.19742, ending 0.20035) and on FLUX.
  - Only runs of at least DRIFT_MIN_K epochs qualify, because a peak has to have
    formed before a decline from it means anything. Twelve is chosen as double
    the six the first analysis could reach, and is fixed here rather than tuned
    to whatever the runs deliver.
  - Runs are truncated to the common length, as before, so a longer run cannot
    contribute extra late epochs the others never saw. Drift is measured over
    the same epoch window for every run.
  - Comparison is the difference of arm mean drift, standard error from
    between-seed spread, threshold 2 SE. Below that the answer is "no difference
    in drift detected at this power", which is again not "no difference".

If the arms differ here and not in the level comparison, that is the anchor doing
its job in the place it was designed to work rather than in the place that was
convenient to measure.

Read resolution, Aug 2 2026, written before any drift value was computed. The
declared point was "all runs that can qualify", which did not anticipate a run
that recovers but is far behind. `d3s43` is alive at 3 epochs against a threshold
of 12, roughly an hour away and liable to crash again. The primary read is
therefore 4 seeds at data_beta=0 against 3 at data_beta=0.3, with `d3s43`
excluded on length rather than on being broken - an ordinary shortfall, not the
configuration-linked exclusion the retracted note described. If it later
qualifies, that reading is reported as a secondary result alongside this one and
does not replace it.

Read point, declared before looking. The level comparison had no defined moment
to be read, so its number moved every time runs grew and I had to record that as
a gap. This one is read ONCE, when every run that can qualify has qualified -
that is four seeds at data_beta=0, and four at data_beta=0.3 if `d3s43` recovers
or three if it does not (see below - it was written off and then recovered). At the moment of writing this the counts were 4 and
2, with `d3s44` one epoch short; that state was not analysed. Re-reads after the
declared point are reported as such, not substituted for it.

Formerly-excluded run. `d3s43` failed six consecutive times in the epoch-0
backward pass and was written up as permanently broken, which is why the read
point below names three seeds for data_beta=0.3. It then went through on the
seventh attempt and is training normally, so the streak was luck rather than a
mechanism and there is no exclusion. If it reaches the threshold it is included
like any other run; if it does not, the arm reports three seeds and says so.

Provenance note. The runs feeding this are the `d0*`/`d3*` tags, started clean on
Aug 2 2026. The
earlier curves (archived under repro_csv/premixed) were grown across a change in
how resume handled the RNG: before the fix a resumed run drew different rollout
noise than an uninterrupted one, and because the data_beta=0.3 arms crash roughly
1.66x more often they absorbed more of those unrestored resumes. That is a
difference between the arms with nothing to do with the anchor, arriving exactly
where this analysis reads. Those curves remain valid for the level comparison,
which was read at its stopping point before any of this, and are not used here.
"""
import csv
import os
import statistics

# Its own directory, not the level experiment's. The fetch that refreshes curves
# wipes and rebuilds whatever directory it targets, so sharing one would delete
# the level experiment's committed record every time the drift runs are polled -
# which it did once before this was split.
CSV_DIR = os.path.join(os.path.dirname(__file__), "drift_csv")

# New tags for the clean run, not the b0/b3 ones the level comparison used. A
# rename rather than a filter: the analysis picks the longest run per tag across
# rotated files, so as long as the old curves share a tag prefix they can be
# selected by accident no matter what a docstring says. Different tags make the
# contamination structurally impossible instead of merely cleaned up.
ARMS = {
    "data_beta=0": ["d0s42", "d0s43", "d0s44", "d0s45"],
    "data_beta=0.3": ["d3s42", "d3s43", "d3s44", "d3s45"],
}
DRIFT_MIN_K = 12


def _read(path):
    """Values indexed by epoch, verified rather than assumed.

    Everything downstream treats position in this list as the epoch number. That
    held while every run started at epoch 0 and wrote one row per epoch, but
    resume makes it an assumption worth checking: a resumed job whose CSV was
    rotated away would start its file at epoch 8, and position-as-epoch would
    then silently compare one arm's epoch 8 against another's epoch 0. Returning
    nothing is the right failure here - a dropped run costs power, a misaligned
    one costs correctness.
    """
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        rows = [(int(float(row[0])), float(row[1])) for row in r if len(row) >= 2]
    if not rows:
        return []
    epochs = [e for e, _ in rows]
    if epochs != list(range(len(epochs))):
        print(f"  SKIPPING {os.path.basename(path)}: epochs are not 0..{len(epochs)-1} "
              f"(starts at {epochs[0]}, ends at {epochs[-1]}, {len(epochs)} rows)")
        return []
    return [v for _, v in rows]


def load(tag):
    """Longest run for this tag, across the logger's rotated files."""
    best = []
    for name in os.listdir(CSV_DIR) if os.path.isdir(CSV_DIR) else []:
        if name.startswith(f"{tag}_eval") and name.endswith(".csv"):
            v = _read(os.path.join(CSV_DIR, name))
            if len(v) > len(best):
                best = v
    return best


def drift(v, K):
    """Late-third mean minus early-third mean, over epochs 1..K-1."""
    body = v[1:K]
    third = max(1, len(body) // 3)
    return statistics.mean(body[-third:]) - statistics.mean(body[:third])


def giveback(v, K):
    """Peak minus final, over epochs 1..K-1. Positive means it gave gain back.

    Added Aug 2 2026, BEFORE any clean run had produced a point, after testing
    `drift` against synthetic curves of known shape. That test showed `drift`
    returns exactly 0.0000 for a curve that rises to a peak and then gives all of
    it back, because the late and early thirds both sit low - which is the exact
    shape SFT showed (held-out minimum 0.19742, ending 0.20035) and the exact
    thing the anchor is supposed to prevent. The registered statistic was blind
    to the failure mode it was written for.

    Peak-minus-final is upward biased on any noisy curve, since a maximum is a
    maximum of noise too. That bias is identical across arms, so it cancels in
    the between-arm difference this reports; it would not be safe to read a
    single arm's value as an absolute amount of give-back.

    Both statistics are reported. `drift` catches a steady decline that never
    peaked; `giveback` catches a rise that was surrendered. They fail in
    different directions and neither alone covers the question.
    """
    body = v[1:K]
    return max(body) - body[-1]


def main():
    runs = {}
    for arm, tags in ARMS.items():
        for t in tags:
            v = load(t)
            if len(v) >= DRIFT_MIN_K:
                runs[t] = (arm, v)
            else:
                print(f"  not yet: {t} has {len(v)} epochs, need {DRIFT_MIN_K}")

    if len(runs) < 4:
        print(f"\n{len(runs)} runs long enough; need at least 2 per arm before this says anything")
        return

    K = min(len(v) for _, v in runs.values())
    print(f"\ncommon length K={K}; {len(runs)} runs qualify\n")

    for label, fn in (("drift", drift), ("giveback", giveback)):
        per_arm = {}
        print(f"--- {label} ---")
        for tag, (arm, v) in sorted(runs.items()):
            d = fn(v, K)
            per_arm.setdefault(arm, []).append(d)
            print(f"  {tag:<7} {arm:<14} {label} = {d:+.4f}")

        summary = {}
        for arm, ds in per_arm.items():
            m = statistics.mean(ds)
            sd = statistics.stdev(ds) if len(ds) > 1 else float("nan")
            se = sd / len(ds) ** 0.5 if len(ds) > 1 else float("nan")
            summary[arm] = (m, se, len(ds))
            print(f"  {arm:<14} mean={m:+.4f}  SD={sd:.4f}  SE={se:.4f}  n={len(ds)}")

        if len(summary) == 2 and all(x[2] > 1 for x in summary.values()):
            (a, (ma, sea, _)), (b, (mb, seb, _)) = sorted(summary.items())
            diff = ma - mb
            se_diff = (sea ** 2 + seb ** 2) ** 0.5
            print(f"  {a} minus {b} = {diff:+.4f}, SE {se_diff:.4f} -> {abs(diff)/se_diff:.1f} SE")
            if abs(diff) > 2 * se_diff:
                print(f"  VERDICT ({label}): the arms differ - the anchor acts where it was meant to")
            else:
                print(f"  VERDICT ({label}): no difference detected at this power.")
        print()


if __name__ == "__main__":
    main()
