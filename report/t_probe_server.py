"""Load FLUX once, then run probe configs from a queue until the wall clock ends.

The cost structure on atlas29 is lopsided in a way that shaped every experiment
here badly. Reading FLUX off NFS takes hours; a rollout takes minutes. The same
2 GB read that takes seconds on the login node takes most of an hour on that
node, so the bottleneck is not the storage and not the GPU. Submitting one job
per question means paying the load again for each, which is why five attempts at
one question produced nothing.

So: pay the load once and keep the process alive. Configs arrive in a queue file
that can be appended to while this is running, results are written as they finish,
and a new question costs a rollout rather than a job. Both scorers are built up
front so the reward can change between configs without reloading anything.

Queue is JSON lines at QUEUE, one object per config:

    {"id": "ocr-0.7", "reward": "ocr_exact", "noise": 0.7, "reps": 3, "group": 6, "ppe": 4}

`id` must be unique; an id already present in RESULTS is skipped, so re-queuing is
harmless and a restart resumes rather than repeats. Results are appended to
RESULTS as JSON lines with the same id. Unknown keys are ignored, missing ones
take the defaults below.

Each config is measured exactly as the standalone probe did: production
`_sample_trajectories` and `_compute_advantages`, seeded per rep so a given rep
starts from the same latents at every noise level, and the reported spread is the
statistic `has_signal` gates on rather than `rl_loss` standing in for it.
"""
import faulthandler, json, os, sys, time, traceback
import torch

faulthandler.dump_traceback_later(600, repeat=True)
print("startup: torch imported", flush=True)
sys.path.insert(0, "/atlas/u/srijit")
from prompts_sets import count_hard_prompts, ocr_hard_prompts

from diffusion_tinker import DDRLConfig, DDRLTrainer
from diffusion_tinker.rewards import resolve_reward

QUEUE = os.environ.get("QUEUE", "/atlas/u/srijit/probe_queue.jsonl")
RESULTS = os.environ.get("RESULTS", "/atlas/u/srijit/probe_results.jsonl")
MODEL = os.environ.get("MODEL", "black-forest-labs/FLUX.1-dev")
STEPS = int(os.environ.get("STEPS", "28"))
GUID = float(os.environ.get("GUID", "3.5"))
RES = int(os.environ.get("RES", "512"))
BASE_SEED = int(os.environ.get("SEED", "42"))
IDLE_EXIT = int(os.environ.get("IDLE_EXIT", "3600"))

PROMPT_SETS = {"count_hard": count_hard_prompts, "ocr_hard": ocr_hard_prompts,
               "ocr_exact": ocr_hard_prompts}
SCORER_OF = {"ocr_hard": "ocr", "count_hard": "count", "ocr_exact": "ocr_exact"}

print(f"probe server: model={MODEL} queue={QUEUE} results={RESULTS}", flush=True)

cfg = DDRLConfig(
    num_epochs=1, num_inference_steps=STEPS, num_eval_inference_steps=STEPS,
    num_samples_per_prompt=6, train_batch_size=1, prompts_per_epoch=4,
    resolution=RES, lora_rank=16, learning_rate=3e-5, guidance_scale=GUID,
    noise_level=0.7, clip_range=1e-3, timestep_fraction=0.99,
    advantage_global_std=False, offload_text_encoders=True,
    data_beta=0.0, seed=BASE_SEED, eval_every=999, save_every=999,
    eval_prompts=count_hard_prompts("train")[:2], num_eval_samples_per_prompt=1,
    eval_seed=1234, output_dir="/atlas/u/srijit/tune_out/probe_server",
)

trainer = DDRLTrainer(model=MODEL, reward_funcs="count",
                      train_prompts=count_hard_prompts("train"), config=cfg)
print("MODEL_LOADED", flush=True)

# Both scorers up front. Building one lazily inside the loop would put a
# multi-minute detector load in the middle of a config and charge it to that
# config's timing.
scorers = {}
for name in ("count", "ocr_exact"):
    scorers[name] = resolve_reward(name, device=trainer.device)
    print(f"scorer ready: {name}", flush=True)
print("SCORERS_READY", flush=True)


def done_ids():
    if not os.path.exists(RESULTS):
        return set()
    out = set()
    with open(RESULTS) as f:
        for line in f:
            try:
                out.add(json.loads(line)["id"])
            except Exception:
                continue
    return out


def pending():
    if not os.path.exists(QUEUE):
        return []
    have = done_ids()
    jobs, seen = [], set()
    with open(QUEUE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                j = json.loads(line)
            except Exception:
                print(f"skipping unparseable queue line: {line[:80]}", flush=True)
                continue
            i = j.get("id")
            if not i or i in have or i in seen:
                continue
            seen.add(i)
            jobs.append(j)
    return jobs


def run_one(j):
    reward = j.get("reward", "count_hard")
    noise = float(j.get("noise", 0.7))
    reps = int(j.get("reps", 3))
    group = int(j.get("group", 6))
    ppe = int(j.get("ppe", 4))

    trainer.reward_fn = scorers[SCORER_OF[reward]]
    trainer.config.noise_level = noise
    trainer.config.num_samples_per_prompt = group
    trainer.config.prompts_per_epoch = ppe
    train_p = PROMPT_SETS[reward]("train")

    rows = []
    for rep in range(reps):
        prompts = (train_p * 8)[rep * ppe:(rep + 1) * ppe]
        torch.manual_seed(BASE_SEED + rep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(BASE_SEED + rep)

        traj = trainer._sample_trajectories(prompts)
        rewards = traj.rewards.detach().float().cpu().reshape(len(prompts), group)
        traj = trainer._compute_advantages(traj)
        adv = traj.advantages.detach().float().cpu()

        pgs = rewards.std(dim=1, unbiased=True)
        rows.append(dict(
            rep=rep,
            mean_reward=round(rewards.mean().item(), 6),
            within_group_std=round(pgs.mean().item(), 6),
            zero_groups=round((pgs <= 1e-9).float().mean().item(), 6),
            adv_std=round(adv.std().item(), 8),
            has_signal=bool(adv.std().item() > 1e-6),
            per_group=[round(v, 6) for v in pgs.tolist()],
        ))
        print(f"PROBE id={j['id']} reward={reward} noise={noise} rep={rep} "
              f"within_group_std={rows[-1]['within_group_std']:.4f} "
              f"zero_groups={rows[-1]['zero_groups']:.3f} "
              f"adv_std={rows[-1]['adv_std']:.6f} "
              f"has_signal={rows[-1]['has_signal']}", flush=True)

    n = len(rows)
    agg = lambda k: sum(r[k] for r in rows) / n
    return dict(id=j["id"], reward=reward, noise=noise, group=group, ppe=ppe,
                reps=n, rows=rows,
                mean_within_group_std=round(agg("within_group_std"), 6),
                mean_zero_groups=round(agg("zero_groups"), 6),
                mean_adv_std=round(agg("adv_std"), 8),
                quiet_reps=sum(1 for r in rows if not r["has_signal"]))


last_work = time.time()
while True:
    jobs = pending()
    if not jobs:
        if time.time() - last_work > IDLE_EXIT:
            print("idle past IDLE_EXIT, exiting", flush=True)
            break
        time.sleep(60)
        continue

    for j in jobs:
        t0 = time.time()
        try:
            res = run_one(j)
            res["seconds"] = round(time.time() - t0, 1)
        except Exception as e:
            traceback.print_exc()
            res = dict(id=j.get("id", "?"), error=f"{type(e).__name__}: {e}",
                       seconds=round(time.time() - t0, 1))
        # Append and flush per config: this process is killed by segfault or
        # wall clock often enough that buffering results loses them.
        with open(RESULTS, "a") as f:
            f.write(json.dumps(res) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"RESULT {json.dumps(res)}", flush=True)
        last_work = time.time()

print("SERVER_DONE", flush=True)
