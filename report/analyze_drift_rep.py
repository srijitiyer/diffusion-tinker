"""Pre-registered replication of the drift result.

The first drift read found the anchored arm degrading more over twelve epochs:
drift +0.0007 (data_beta=0, n=4) against -0.0326 (data_beta=0.3, n=3), a
difference of +0.0334 at 2.2 SE, with `giveback` agreeing in direction at 1.0 SE.
That is the only non-null result this project has produced, and it is exactly the
kind that deserves suspicion rather than a headline: two statistics were
registered so the effective bar is 0.089 not 0.045, one arm had three seeds, and
the statistic purpose-built for the shape is the one that missed.

This replicates it on four fresh seeds per arm (46-49) with everything else
identical. Committed before those runs produced a single point.

Fixed in advance, all inherited unchanged from the first analysis so the two are
comparable:

  - Same two statistics, `drift` and `giveback`, same definitions.
  - Same 12-epoch minimum per run, same common-window truncation.
  - Same unit (a run), same error (between-seed), same 2 SE threshold.
  - Same longest-run selection across rotated files, same epoch-contiguity check.

Declared read point: once, when all eight runs have qualified, or when six have
qualified and no further run has advanced for an hour - the first read stalled on
one arm waiting for a straggler and the ambiguity had to be resolved mid-flight.

What would count as replication: the same sign at 2 SE or better on `drift`. A
same-sign result below threshold is weak support and is reported as such. An
opposite sign, at any strength, means the first result does not stand.
"""
import os

# Reuse the first analysis wholesale rather than restating it, so the two cannot
# drift apart through independent edits.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_drift_base", os.path.join(os.path.dirname(__file__), "analyze_drift.py")
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

_base.CSV_DIR = os.path.join(os.path.dirname(__file__), "driftrep_csv")
_base.ARMS = {
    "data_beta=0": ["r0s46", "r0s47", "r0s48", "r0s49"],
    "data_beta=0.3": ["r3s46", "r3s47", "r3s48", "r3s49"],
}

if __name__ == "__main__":
    print("replication cohort, seeds 46-49\n")
    _base.main()
