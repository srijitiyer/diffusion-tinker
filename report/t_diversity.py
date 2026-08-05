"""Does raising exploration noise give FLUX the within-group spread it lacks?

This is the mechanism behind the anchor defect's model asymmetry, and it has
never been tested. Five attempts tried to test it by training FLUX at noise 1.0
for twenty epochs and comparing held-out curves. All five died to node
contention, filesystem throughput or segfaults without producing a curve, and
even a successful one would have answered a different question - whether FLUX
improves - with the mechanism inferred from the outcome.

Within-group spread is a property of sampling, not of training. It needs no
optimizer, no anchor, no epochs: draw a group, score it, measure the spread.
That is what this does, at a cost of two rollout passes instead of twenty
epochs.

The statistic is the one `has_signal` actually gates on. Everything published on
this so far used `rl_loss` printing as `0.0000` as a stand-in, which is the wrong
field twice over - it tracks the mean advantage rather than the spread, and group
normalisation drives that mean to zero whether or not there is signal, and four
decimals only says `< 0.00005`. Here the advantages come from the production
`_compute_advantages` and the spread is read at full float precision.

Design. Paired on everything except noise:

  - The same prompts at both noise levels, in the same order.
  - `torch.manual_seed(BASE + rep)` immediately before each rollout, so a given
    rep starts from the same initial latents at both levels. Trajectories diverge
    after the first noised step, which is the effect being measured, but they do
    not diverge because they started somewhere else.
  - The production sampling path (`_sample_trajectories`), so what is measured is
    what training does rather than a sampler written for this file.

Reported per (noise, rep), with reps giving the error bar:

  - `zero_groups`: share of prompt-groups whose rollouts all scored identically.
    This is the quiet condition at full precision, and it is the number the
    logged `per_prompt` means could not resolve at two decimals.
  - `reward_std`: mean within-group reward standard deviation.
  - `adv_std`: std of the advantage tensor, which is exactly what `has_signal`
    thresholds at 1e-6.

Read it as: noise 1.0 raises spread and cuts `zero_groups` -> the mechanism
holds and exploration noise is the lever. Unchanged -> the mechanism is wrong,
and FLUX's failure to climb needs a fifth explanation rather than a fourth. Note
that this measures diversity only; whether more diversity makes FLUX improve is
the separate and far more expensive question.
"""
import faulthandler, os, re, sys, torch

faulthandler.dump_traceback_later(300, repeat=True)
print("startup: torch imported", flush=True)
sys.path.insert(0, "/atlas/u/srijit")
from prompts_sets import count_hard_prompts, ocr_hard_prompts

from diffusion_tinker import DDRLConfig, DDRLTrainer

REWARD = os.environ.get("REWARD", "count_hard")
SCORER = {"ocr_hard": "ocr", "count_hard": "count", "ocr_exact": "ocr_exact"}.get(REWARD, REWARD)
MODEL = os.environ.get("MODEL", "black-forest-labs/FLUX.1-dev")
GROUP = int(os.environ.get("GROUP", "6"))
PPE = int(os.environ.get("PPE", "4"))
REPS = int(os.environ.get("REPS", "3"))
STEPS = int(os.environ.get("STEPS", "28"))
GUID = float(os.environ.get("GUID", "3.5"))
RES = int(os.environ.get("RES", "512"))
BASE_SEED = int(os.environ.get("SEED", "42"))
# Colon-separated, not comma. sbatch --export itself takes a comma-separated
# list, so a comma here is eaten by the export parsing and the sweep silently
# truncates to its first level - the first run of this probe measured only noise
# 0.7 for exactly that reason and produced no comparison at all.
LEVELS = [float(x) for x in re.split(r"[:,]", os.environ.get("LEVELS", "0.7:1.0")) if x]

PROMPTS = {"count_hard": count_hard_prompts, "ocr_hard": ocr_hard_prompts,
           "ocr_exact": ocr_hard_prompts}[REWARD]
train_p = PROMPTS("train")

print(f"diversity probe: model={MODEL} reward={REWARD} group={GROUP} "
      f"prompts={PPE} reps={REPS} levels={LEVELS}", flush=True)

cfg = DDRLConfig(
    num_epochs=1,
    num_inference_steps=STEPS,
    num_eval_inference_steps=STEPS,
    num_samples_per_prompt=GROUP,
    train_batch_size=1,
    prompts_per_epoch=PPE,
    resolution=RES,
    lora_rank=16,
    learning_rate=3e-5,
    guidance_scale=GUID,
    noise_level=LEVELS[0],
    clip_range=1e-3,
    timestep_fraction=0.99,
    advantage_global_std=False,
    offload_text_encoders=True,
    # No anchor: this probe never steps the optimizer, and data_beta=0 skips
    # encoding the anchor images entirely, which is several minutes on this
    # filesystem for something that is never read.
    data_beta=0.0,
    seed=BASE_SEED,
    eval_every=999,
    save_every=999,
    eval_prompts=train_p[:2],
    num_eval_samples_per_prompt=1,
    eval_seed=1234,
    output_dir="/atlas/u/srijit/tune_out/probe",
)

trainer = DDRLTrainer(model=MODEL, reward_funcs=SCORER,
                      train_prompts=train_p, config=cfg)
print("MODEL_LOADED", flush=True)

rows = []
for noise in LEVELS:
    trainer.config.noise_level = noise
    for rep in range(REPS):
        prompts = train_p[rep * PPE:(rep + 1) * PPE]
        if len(prompts) < PPE:
            prompts = (train_p * 4)[rep * PPE:(rep + 1) * PPE]

        # Pair the starting latents across noise levels for this rep.
        torch.manual_seed(BASE_SEED + rep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(BASE_SEED + rep)

        traj = trainer._sample_trajectories(prompts)
        rewards = traj.rewards.detach().float().cpu().reshape(len(prompts), GROUP)
        traj = trainer._compute_advantages(traj)
        adv = traj.advantages.detach().float().cpu()

        per_group_std = rewards.std(dim=1, unbiased=True)
        zero_groups = (per_group_std <= 1e-9).float().mean().item()
        row = dict(noise=noise, rep=rep,
                   mean_reward=rewards.mean().item(),
                   reward_std=per_group_std.mean().item(),
                   zero_groups=zero_groups,
                   adv_std=adv.std().item(),
                   has_signal=bool(adv.std().item() > 1e-6))
        rows.append(row)
        print(f"PROBE noise={noise} rep={rep} mean_reward={row['mean_reward']:.4f} "
              f"within_group_std={row['reward_std']:.4f} zero_groups={zero_groups:.3f} "
              f"adv_std={row['adv_std']:.6f} has_signal={row['has_signal']} "
              f"per_group={[round(v, 4) for v in per_group_std.tolist()]}", flush=True)

print("\n=== summary ===", flush=True)
for noise in LEVELS:
    sel = [r for r in rows if r["noise"] == noise]
    n = len(sel)
    mu = lambda k: sum(r[k] for r in sel) / n
    sd = lambda k: (sum((r[k] - mu(k)) ** 2 for r in sel) / max(n - 1, 1)) ** 0.5
    print(f"noise={noise}: within_group_std {mu('reward_std'):.4f} (SD {sd('reward_std'):.4f}), "
          f"zero_groups {mu('zero_groups'):.3f}, adv_std {mu('adv_std'):.6f}, "
          f"quiet_reps {sum(1 for r in sel if not r['has_signal'])}/{n}", flush=True)

if len(LEVELS) == 2:
    a = [r for r in rows if r["noise"] == LEVELS[0]]
    b = [r for r in rows if r["noise"] == LEVELS[1]]
    d = sum(r["reward_std"] for r in b) / len(b) - sum(r["reward_std"] for r in a) / len(a)
    dz = sum(r["zero_groups"] for r in b) / len(b) - sum(r["zero_groups"] for r in a) / len(a)
    print(f"noise {LEVELS[1]} minus {LEVELS[0]}: within_group_std {d:+.4f}, "
          f"zero_groups {dz:+.3f}", flush=True)

print("PROBE_DONE", flush=True)
