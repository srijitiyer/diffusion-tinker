"""Pre-registered analysis of the seeded anchor-strength comparison.

Written and committed before the runs finished, deliberately. The first sweep
produced a table that could be read several ways depending on which statistic you
reached for (best point, last five, mean over epochs) and which run happened to
survive, and picking among those after seeing the data is how a null result turns
into a positive one. So the choices are fixed here in advance:

  - The unit of analysis is a RUN, not an eval point. Points within a run are a
    time series of one policy and are not independent; runs at different seeds
    are the replicates. This is the whole reason the design has four seeds.
  - Each run contributes ONE number: the mean held-out reward over epochs 1..K,
    excluding epoch 0 (which is the untrained model and identical everywhere).
  - K is the common length across all runs, so a longer-lived run cannot
    contribute extra late epochs that the others never got to.
  - The comparison is a two-sample difference of arm means with the standard
    error taken from BETWEEN-seed spread. Nothing here uses the per-point SEM,
    because eval latents are pinned and identical across runs, so that error
    cancels in a run-to-run comparison.
  - The verdict threshold is |difference| > 2 SE. Anything else is "no effect
    detected at this power", which is not the same as "no effect".
  - A tag can have SEVERAL runs, because the node crashes jobs and the supervisor
    resubmits them, and the logger rotates rather than truncating. The rule is
    the LONGEST run for that tag, and it lives here rather than in whoever copies
    the files, which is where it was for the first pass. It is a non-random
    selection, so it is only sound because crash timing is independent of the
    reward values - which is what the near-equal crash counts across arms (7 vs
    6 across two different regularizer settings) support.
"""
import csv
import os
import statistics

CSV_DIR = os.path.join(os.path.dirname(__file__), "seeded_csv")

ARMS = {
    "data_beta=0": ["b0s42", "b0s43", "b0s44", "b0s45"],
    "data_beta=0.3": ["b3s42", "b3s43", "b3s44", "b3s45"],
}
MIN_POINTS = 6  # a run with fewer than this is dropped rather than padded


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
        if not name.startswith(f"{tag}_eval") or not name.endswith(".csv"):
            continue
        v = _read(os.path.join(CSV_DIR, name))
        if len(v) > len(best):
            best = v
    return best


def main():
    runs = {}
    for arm, tags in ARMS.items():
        for t in tags:
            v = load(t)
            if len(v) >= MIN_POINTS:
                runs[t] = (arm, v)
            else:
                print(f"  dropped {t}: {len(v)} points, need {MIN_POINTS}")

    if not runs:
        print("no runs long enough yet")
        return

    K = min(len(v) for _, v in runs.values())
    print(f"\ncommon length K={K} epochs; {len(runs)} runs qualify\n")

    per_arm = {}
    for tag, (arm, v) in sorted(runs.items()):
        score = statistics.mean(v[1:K])
        per_arm.setdefault(arm, []).append(score)
        print(f"  {tag:<7} {arm:<14} mean(ep 1..{K-1}) = {score:.4f}")

    print()
    summary = {}
    for arm, scores in per_arm.items():
        m = statistics.mean(scores)
        sd = statistics.stdev(scores) if len(scores) > 1 else float("nan")
        se = sd / len(scores) ** 0.5 if len(scores) > 1 else float("nan")
        summary[arm] = (m, se, len(scores))
        print(f"  {arm:<14} mean={m:.4f}  between-seed SD={sd:.4f}  SE={se:.4f}  n={len(scores)}")

    if len(summary) == 2 and all(s[2] > 1 for s in summary.values()):
        (a, (ma, sea, _)), (b, (mb, seb, _)) = sorted(summary.items())
        diff = ma - mb
        se_diff = (sea ** 2 + seb ** 2) ** 0.5
        print(f"\n  {a} minus {b} = {diff:+.4f}, SE {se_diff:.4f} -> {abs(diff)/se_diff:.1f} SE")
        if abs(diff) > 2 * se_diff:
            print("  VERDICT: difference exceeds 2 SE - anchor strength has a detectable effect")
        else:
            print("  VERDICT: no effect detected at this power.")
            print("  Not the same as no effect: the anchor's role is preventing drift over long")
            print(f"  training, and this compares only the first {K-1} epochs.")


if __name__ == "__main__":
    main()
