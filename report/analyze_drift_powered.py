"""Adequately powered drift comparison: 8 seeds per arm.

Two cohorts have now failed to answer whether the data anchor changes late-training
degradation, and the reason is arithmetic rather than bad luck. Between-seed SD on
`drift` is about 0.0232, measured across four arms in two cohorts. At the 3-4 seeds
those cohorts ran, the smallest difference detectable at 2 SE is 0.033-0.038:

    seeds/arm     SE(difference)     min detectable at 2 SE
        3            0.0189                 0.0378
        4            0.0164                 0.0328
        8            0.0116                 0.0232
       12            0.0095                 0.0189

The difference the first cohort reported was 0.0334 - sitting exactly on its own
detection limit, which is where noise is most likely to look like signal. The
second cohort saw 0.0059 and, at a limit of 0.033, could not have distinguished
that from anything smaller than 0.033 either. Neither run was capable of the
question it was asked.

This one uses eight seeds per arm (50-57), giving a minimum detectable effect of
0.0232. That is a real but stated limit: an effect smaller than ~0.023 remains
out of reach and no result here should be read as excluding one.

Everything else is inherited unchanged from `analyze_drift.py` by import, so the
three readings stay comparable:

  - Same two statistics, same definitions, same opposite-polarity direction test.
  - Window FIXED at DRIFT_MIN_K = 12, not derived from run length. This is the
    defect that manufactured a 4.2 SE result in the second cohort and erased it
    when one more run qualified.
  - Same longest-run selection, same epoch-contiguity check, same 2 SE threshold.

Declared read point: once, when all sixteen runs have qualified, or when twelve
have qualified and no further run has crossed the threshold for an hour. The
second clause exists because a single straggler held up the last read and the
ambiguity had to be resolved mid-flight.

Declared interpretation, fixed now:
  - |difference| > 2 SE on either statistic, with the other agreeing in sign:
    a real effect of the anchor at this horizon.
  - |difference| > 2 SE on one, with the other disagreeing in sign: unresolved,
    reported as such, because the two statistics measure different failure shapes
    and disagreement means neither shape is cleanly present.
  - Neither above threshold: no effect larger than 0.023 at this horizon. That is
    the most likely outcome given two prior nulls and is not a disappointment.
"""
import os
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_drift_base", os.path.join(os.path.dirname(__file__), "analyze_drift.py")
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

_base.CSV_DIR = os.path.join(os.path.dirname(__file__), "driftpow_csv")
_base.ARMS = {
    "data_beta=0": [f"p0s{s}" for s in range(50, 58)],
    "data_beta=0.3": [f"p3s{s}" for s in range(50, 58)],
}

if __name__ == "__main__":
    print("powered cohort, 8 seeds per arm (50-57), fixed window K=12")
    print("minimum detectable difference at 2 SE: 0.0232\n")
    _base.main()
