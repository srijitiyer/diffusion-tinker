# A debugging watchdog destroyed 86% of every run in this project

`t_rewards.py` armed `faulthandler.dump_traceback_later(300, repeat=True)` so that
a job stalled on a slow torch import off NFS would leave a diagnosable stack
instead of silence. On Python 3.12 that call walks the frames of every *other*
thread from a watchdog thread, every five minutes, for the entire run. It is the
reason the runs kept dying.

## Scale

| model | logs on disk | ended in a segfault |
|---|---|---|
| SD3.5 (`sw*`, `cw*`) | 712 | 598 (84%) |
| FLUX (`fx*`) | 298 | 264 (89%) |

Every crashed log ends truncated part way through one of the watchdog's dumps.
Ticks before death in three sampled runs: 9, 24 and 31, against elapsed times of
45m, 2h00m and 2h35m. Exactly 300 seconds apart.

## The controlled test

Same node, same Python 3.12.3, same torch 2.5.1+cu124, same multi-threaded CPU
workload, armed at a 5 second period so a run reaches in a minute the tick count
a training job took two hours to reach.

| arm | watchdog | outcome |
|---|---|---|
| `wdon` | `dump_traceback_later(5, repeat=True)` | segfault at 77s, about 15 ticks |
| `wdoff` | nothing armed | 8000 forwards, 119s, still clean |

## Why four earlier diagnoses were wrong

The crash site moved between `pickscore.py`, `clip_score.py`, `ocr.py` and a
bare `nn.Linear.forward`, because it was never the workload. The dumper was
walking whatever happened to be executing. Each of these was tested and is not
the cause:

- **PickScore specifically.** Attributing stacks across all crashed logs gives
  `clip_score.py` 10 times and `ocr.py` twice, not PickScore alone. Dropping
  PickScore from the in-training eval did not stop the crashes.
- **OpenMP thread oversubscription.** `torch.get_num_threads()` reports 4 even
  with `OMP_NUM_THREADS` unset, because torch reads the cgroup affinity.
- **bf16 CPU kernels.** The rewards default to `torch.float32`.
- **Host memory.** `sacct` MaxRSS peaks near 20GB against a 48G request.
- **Three OpenMP runtimes in one process.** The venv ships paddle's `libiomp5`,
  torch's `libgomp` and scikit-learn's `libgomp`, but `/proc/self/maps` shows
  only torch's is ever mapped.

## What changed

`faulthandler.enable()`, which only fires on a real fatal signal, plus a single
non-repeating dump sized to cover the import stall the watchdog was written for,
cancelled before the training loop starts. Applied to `t_rewards.py`,
`t_diversity.py`, `t_flux_smoke.py` and `t_probe_server.py`.

## What it does and does not change

No published number moves. The supervisor resubmits a dead arm and training
resumes from its checkpoint, so results were never wrong, they were slow: each
crash cost up to 10 epochs of redone work and a requeue behind whoever held the
GPU next.

Two things made this hard to see for as long as it was. The batch wrapper ends
with `echo "=== EXIT $? ==="`, which succeeds, so SLURM records a segfaulted run
as `COMPLETED 0:0` and `sacct` looks healthy. And `ulimit -c` is 0 with
`core_pattern` piped to apport, so the "(core dumped)" in the logs went nowhere
and there was never a core to read.
