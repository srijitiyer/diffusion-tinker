"""Pre-registered replication: does data_beta change the segfault rate?

The level experiment turned up a signal that the anchor arms crash more often -
0.412 vs 0.683 crashes per GPU-hour, z = 2.2 - but it was found post-hoc. The
run-length gap was noticed first and the rate computed afterwards, which is the
forking-paths setup, so that p-value is optimistic by an unknown amount and the
finding was recorded as needing replication rather than as a cause.

The drift runs (`d0*` at data_beta=0, `d3*` at data_beta=0.3) are a clean
replication: matched in everything else, seeds crossed, all started fresh on
Aug 2 2026 under identical code. This file is committed BEFORE their crash counts
were looked at, so the test is declared rather than fitted.

Fixed in advance:

  - Exposure is GPU-seconds from sacct, not wall-clock or run count. A crash is
    an event in time and the arms do not get equal time.
  - The statistic is the crash count split between arms, tested against the split
    their exposure implies. Under equal rates the count on one arm is binomial
    with p = that arm's share of exposure.
  - Threshold |z| > 2. Below that: not replicated.
  - Minimum 30 crashes total before reading, so the test is not run on a handful
    of events where a single crash swings it.

Interpretation is fixed too, because the obvious mechanism does not implicate the
anchor as such: data_beta > 0 adds a forward and backward on anchor latents at
every timestep, so those runs execute more kernels and allocations in the same
wall-clock. A confirmed rate difference per GPU-hour is therefore consistent with
"more GPU work per unit time" and does NOT establish anything specific about the
anchor. Distinguishing those needs exposure measured in kernel launches, which
this does not do.
"""
import os
import subprocess
import sys

ARMS = {"data_beta=0": ["d0s42", "d0s43", "d0s44", "d0s45"],
        "data_beta=0.3": ["d3s42", "d3s43", "d3s44", "d3s45"]}
MIN_CRASHES = 30
SINCE = "2026-08-02"
HOST = ["ssh", "-o", "ConnectTimeout=45", "-J", "srijit@scdt.stanford.edu", "srijit@sc.stanford.edu"]


def sh(cmd):
    return subprocess.run(HOST + [cmd], capture_output=True, text=True).stdout.strip()


def main():
    log = "/atlas/u/srijit/supervise.log"
    rows = {}
    for arm, tags in ARMS.items():
        secs = crashes = 0
        for t in tags:
            raw = sh(f"sacct -u srijit -S {SINCE} --name={t} -o ElapsedRaw -n -P | awk 'NF' | paste -sd+ | bc")
            secs += int(raw or 0)
            crashes += int(sh(f"grep -c '^\\[.*\\] {t}: segfault' {log}") or 0)
        rows[arm] = (secs / 3600.0, crashes)

    total = sum(c for _, c in rows.values())
    print(f"{'arm':<16} {'GPU-hours':>10} {'crashes':>8} {'per GPU-hour':>13}")
    for arm, (h, c) in rows.items():
        print(f"{arm:<16} {h:>10.1f} {c:>8} {c / h if h else float('nan'):>13.3f}")

    if total < MIN_CRASHES:
        print(f"\n{total} crashes so far; not reading until {MIN_CRASHES}")
        return

    (a, (ha, ca)), (b, (hb, _)) = sorted(rows.items())
    frac = ha / (ha + hb)
    exp = total * frac
    sd = (total * frac * (1 - frac)) ** 0.5
    z = (exp - ca) / sd if sd else float("nan")
    print(f"\nunder equal rates, expected {exp:.1f} of {total} on {a}; observed {ca}")
    print(f"z = {z:.1f}")
    if abs(z) > 2:
        print("  REPLICATED: the arms crash at different rates per GPU-hour.")
        print("  Consistent with the anchor's extra forward/backward per timestep doing")
        print("  more GPU work per unit time; does NOT single out the anchor itself.")
    else:
        print("  NOT REPLICATED at |z| > 2. The post-hoc finding does not survive.")


if __name__ == "__main__":
    sys.exit(main())
