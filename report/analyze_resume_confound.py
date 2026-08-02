"""Pre-registered test of the resume-rate confound on the replication cohort.

The drift result (anchored arm degrades more, 2.2 SE) has a candidate explanation
that is not the anchor: in that cohort the anchored arm resumed 0.311 times per
epoch against 0.209, a difference of 2.7 SE. A crash mid-epoch loses that epoch's
training and the resume repeats it, so the model sees part of that epoch twice.
Restoring the RNG made the draws identical and does nothing about how often an
epoch is partially replayed, so the fix that was supposed to remove resume as a
variable did not remove this channel.

That observation is post-hoc - it was found by going looking after the drift
result appeared, which is the same route that produced the crash-rate signal that
then failed its own replication. So it gets a declared test rather than a
narrative.

Committed while the replication cohort was at 1-8 epochs of the 12 required, and
before any of its resume counts were read.

Fixed in advance:

  - Unit is a run. Statistic is resumes per epoch: lines matching "Resuming from"
    in that tag's logs, divided by the run's epoch count.
  - Compared between arms with between-seed SE, threshold 2 SE, same as
    everything else here.
  - Read at the same moment as the drift replication, from the same runs.

The four outcomes and what each means, stated now so the reading is not a choice:

  1. Drift replicates AND resume rates do not differ -> the confound is ruled
     out and the drift finding is materially strengthened.
  2. Drift replicates AND resume rates differ in the same direction -> unresolved.
     Two candidate causes both present; neither can be preferred on this data and
     the honest report says so.
  3. Drift does not replicate AND resume rates differ -> the confound is the
     better explanation of the original result.
  4. Neither differs -> the original 2.2 SE was noise, which two registered
     statistics at a 0.089 effective bar always made plausible.
"""
import os
import subprocess

ARMS = {"data_beta=0": ["r0s46", "r0s47", "r0s48", "r0s49"],
        "data_beta=0.3": ["r3s46", "r3s47", "r3s48", "r3s49"]}
MIN_EPOCHS = 12
HOST = ["ssh", "-o", "ConnectTimeout=45", "-J", "srijit@scdt.stanford.edu", "srijit@sc.stanford.edu"]


def sh(cmd):
    return subprocess.run(HOST + [cmd], capture_output=True, text=True).stdout.strip()


def main():
    import statistics
    base = "/atlas/u/srijit"
    per_arm = {}
    for arm, tags in ARMS.items():
        for t in tags:
            ep = int(sh(
                f"cd {base}/repro_csv && n=0; for f in {t}_eval.csv {t}_eval.run*.csv; do "
                f"[ -s \"$f\" ] || continue; c=$(tail -n +2 \"$f\"|wc -l); "
                f"[ \"$c\" -gt \"$n\" ] && n=$c; done; echo $n") or 0)
            if ep < MIN_EPOCHS:
                print(f"  not yet: {t} has {ep} epochs, need {MIN_EPOCHS}")
                continue
            r = int(sh(f"grep -h 'Resuming from' {base}/{t}_*.log 2>/dev/null | wc -l") or 0)
            rate = r / ep
            per_arm.setdefault(arm, []).append(rate)
            print(f"  {t:<6} {arm:<14} {ep:>3} epochs, {r:>2} resumes -> {rate:.3f}/epoch")

    if not all(len(v) > 1 for v in per_arm.values()) or len(per_arm) < 2:
        print("\nneed at least 2 qualifying runs per arm")
        return

    print()
    stats = {}
    for arm, rs in per_arm.items():
        m, sd = statistics.mean(rs), statistics.stdev(rs)
        stats[arm] = (m, sd / len(rs) ** 0.5, len(rs))
        print(f"  {arm:<14} mean={m:.3f}/epoch  SD={sd:.3f}  n={len(rs)}")

    (a, (ma, sea, _)), (b, (mb, seb, _)) = sorted(stats.items())
    diff = ma - mb
    se = (sea ** 2 + seb ** 2) ** 0.5
    print(f"\n  {a} minus {b} = {diff:+.3f}, SE {se:.3f} -> {abs(diff) / se:.1f} SE")
    print("  RESUME RATES DIFFER" if abs(diff) > 2 * se else "  resume rates do not differ at this power")
    print("  (read alongside analyze_drift_rep.py; see this file's header for what each pairing means)")


if __name__ == "__main__":
    main()
