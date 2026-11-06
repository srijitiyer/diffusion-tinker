# FlowGRPO: FLUX.1-dev vs. Stable Diffusion 3 configuration regimes

**Primary-source basis.** The repository `yifan123/flow_grpo` was cloned in full and read directly (HEAD = commit `879042c`, "Merge pull request #241 from GongyeLiu/docs/add-verl-omni-support"). All config/code quotes below are verbatim from that clone. Paper quotes are from the arXiv HTML of v5 (`arXiv:2505.05470v5 [cs.CV] 27 Oct 2025`), which is the NeurIPS 2025 camera-ready version. Issue text is from the GitHub API via `gh`. Chinese-language text is quoted verbatim with my translation marked as such.

---

## Q1. Exact hyperparameters: FLUX config vs. SD3 configs, side by side

### Takeaway
Every number lives in one file, `config/grpo.py`, layered over `config/base.py`. The FLUX and SD3 recipes differ on: inference steps (6 vs 10), guidance (3.5 vs 4.5), noise level (0.9 or 0.8 vs 0.7), per-GPU batch (3 vs 9), LoRA rank (64/α128 vs 32/α64), precision (bf16 vs fp16), KL β (0 vs 0.04/0.01), and — for the "fast" variants — clip_range (1e-5 vs the 1e-4 default) plus a windowed SDE. Learning rate is *not* set in any SD3 or FLUX config, so both inherit `3e-4` from `base.py`.

### Cited Findings

**Defaults inherited by everything** — [config/base.py](https://github.com/yifan123/flow_grpo/blob/main/config/base.py):
- `config.seed = 42`; `config.num_epochs = 100000`; `config.resolution = 768`; `config.mixed_precision = "fp16"`; `config.use_lora = True`
- `sample.num_steps = 40`, `sample.eval_num_steps = 40`, `sample.guidance_scale = 4.5`, `sample.noise_level = 0.7`, `sample.global_std = True`, `sample.same_latent = False`, `sample.sde_window_size = 2`, `sample.sde_window_range = (0, 10)`
- `train.learning_rate = 3e-4`, `train.adam_beta1 = 0.9`, `train.adam_beta2 = 0.999`, `train.adam_weight_decay = 1e-4`, `train.adam_epsilon = 1e-8`, `train.max_grad_norm = 1.0`, `train.num_inner_epochs = 1`, `train.cfg = True`, `train.adv_clip_max = 5`, `train.clip_range = 1e-4`, `train.timestep_fraction = 1.0`, `train.beta = 0.0`, `train.ema = False`
- `grep -rn "learning_rate" config/grpo.py` returns only three hits (lines 55, 937, 992) — `general_ocr_wan2_1` and the two Bagel configs, all `1e-4`. **No SD3 or FLUX config overrides the learning rate**, so all of them train at `3e-4`.

**Side-by-side table** (all values verbatim from `config/grpo.py` at commit `879042c`; `—` = not set, so the `base.py` default applies; derived quantities marked *(derived)*):

| Parameter | `general_ocr_sd3` / `geneval_sd3` (L83/L125) | `general_ocr_sd3_1gpu` (L444) | `geneval_sd3_fast_nocfg` (L163) | `pickscore_flux` (L484) | `geneval_flux_fast` (L565) / `pickscore_flux_fast` (L609) |
|---|---|---|---|---|---|
| model | `stabilityai/stable-diffusion-3.5-medium` | same | same | `black-forest-labs/FLUX.1-dev` | `black-forest-labs/FLUX.1-dev` |
| `sample.num_steps` (train) | 10 | 10 | 10 | **6** | **6** |
| `sample.eval_num_steps` | 40 | 40 | 40 | **28** | **28** |
| `sample.guidance_scale` | 4.5 | 4.5 | **1** (`train.cfg = False`) | **3.5** | **3.5** (`eval_guidance_scale` 3.5) |
| `sample.noise_level` | — (**0.7**) | — (0.7) | **0.8** | **0.9** | **0.8** |
| `sample.sde_window_size` | — (unused by `train_sd3.py`) | — (unused) | **3** | — (unused by `train_flux.py`) | **3** |
| `sample.sde_window_range` | — | — | **(0, num_steps//2) = (0,5)** | — | **(0, num_steps//2) = (0,3)** |
| `sample.sde_type` | — (`'sde'`) | — (`'sde'`) | **`"cps"`** | — (`'sde'`) | **`"cps"`** |
| `train.clip_range` | — (**1e-4**) | — (1e-4) | **1e-5** | — (**1e-4**) | **1e-5** |
| `train.learning_rate` | — (**3e-4**) | — (3e-4) | — (3e-4) | — (**3e-4**) | — (3e-4) |
| `train.beta` (KL) | **0.04** | 0.04 | **0** | **0** | **0** |
| `sample.train_batch_size` (per GPU) | 9 | 8 | 9 | **3** | **3** |
| `sample.num_image_per_prompt` (group G) | 24 | 8 | 24 | 24 | 24 |
| `gpu_number` in config | 32 | 1 | 32 | 32 | 32 |
| `num_batches_per_epoch` *(derived)* | `int(48/(32*9/24))` = **4** | `int(8/(1*8/8))` = **8** | 4 | `int(48/(32*3/24))` = **12** | 12 |
| `train.gradient_accumulation_steps` *(derived)* | 2 | 4 | 2 | **6** | 6 |
| images per epoch *(derived)* | 4·9·32 = **1152** | 8·8·1 = **64** | 1152 | 12·3·32 = **1152** | 1152 |
| prompts (groups) per epoch *(derived)* | 1152/24 = **48** | 64/8 = **8** | 48 | 1152/24 = **48** | 48 |
| optimizer steps per epoch *(derived)* | **2** | 2 | 2 | **2** | 2 |
| `train.timestep_fraction` | 0.99 | 0.99 | — (1.0, and unused) | 0.99 | — (1.0, and unused) |
| steps receiving gradient *(derived)* | `int(10*0.99)` = **9 of 10** | 9 of 10 | **3 of 10** (window) | `int(6*0.99)` = **5 of 6** | **3 of 6** (window) |
| `sample.global_std` | True | True | True | True | True (geneval) / **False** (pickscore) |
| `train.ema` | True | True | True | True | True |
| `mixed_precision` | — (**fp16**) | — (fp16) | — (fp16) | **"bf16"** | **"bf16"** |
| LoRA | `r=32, lora_alpha=64` (`train_sd3.py` L429-433) | r=32/α64 | r=32/α64 (`train_sd3_fast.py` L435) | **`r=64, lora_alpha=128`** (`train_flux.py` L425-429) | **r=64/α128** (`train_flux_fast.py` L430-434) |
| LoRA target modules | 8 attention projections only | same | same | **12 modules incl. `ff.net.0.proj`, `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2`** | same 12 |
| `num_epochs` | — (**100000**) | 100000 | 100000 | 100000 | 100000 |
| `save_freq` / `eval_freq` | 60 / 60 | 60 / 60 | 60 / 60 | 30 / 30 | 30 / 30 |
| `resolution` | 512 | 512 | 512 | 512 | 512 |

- `num_epochs = 100000` carries the comment: `# A large num_epochs is intentionally set here. Training will be manually stopped once sufficient` — [config/grpo.py](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- An assertion in every config pins the gradient-accumulation relationship: `assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."` — [config/grpo.py](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- There is **no `geneval_flux` (non-fast) config** in the repo — the only FLUX GenEval recipe is `geneval_flux_fast`. The non-fast FLUX configs are `pickscore_flux`, `pickscore_flux_8gpu` and `counting_flux_kontext`. Issue #163 ("Flux-dev Geneval configuration") was opened precisely because of this, and the author answered: *"I will push a configuration for training Flux using flowgrpo-fast by tomorrow. You can refer to the log here: https://api.wandb.ai/links/ljie/72owwyku"* — [issue #163](https://github.com/yifan123/flow_grpo/issues/163)
- Author's stated rationale for FLUX's step counts, from the same issue: *"1. Because the default inference steps for Flux are 28. 2. This is just a hyperparameter I think works well—it's fast and doesn't affect evaluation performance. Of course, 7 or 8 steps are also acceptable."* — [issue #163](https://github.com/yifan123/flow_grpo/issues/163)
- `pickscore_sd3_fast` (L281) is an *older* fast mechanism, not a window: it sets `config.sample.train_num_steps = 2`, `config.sample.train_batch_size = 1`, `config.sample.mini_num_image_per_prompt = 9`, `clip_range = 1e-5`, `noise_level = 0.8`, and does **not** set `sde_window_size`/`sde_type`. Only the `*_nocfg` SD3 variant and the FLUX `*_fast` variants use the window + CPS. — [config/grpo.py](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- `general_ocr_wan2_1` (L29) is the only config with `clip_range = 1e-3`, together with `learning_rate = 1e-4`, `beta = 0.004`, `global_std = False`, `mixed_precision = "bf16"` — it is a **Wan2.1 video** config, not SD3. — [config/grpo.py](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)

### Inferences
- The two regimes keep *total* throughput identical (1152 images / 48 groups / 2 optimizer steps per epoch) and absorb FLUX's larger memory footprint purely by shrinking per-GPU batch 9→3 and raising `gradient_accumulation_steps` 2→6. So "FLUX gets a different hyperparameter regime" is not about data volume; it is about precision, noise injection, step placement and trust-region size.
- Group size G = 24 is held fixed across SD3 and FLUX, which matches the paper's ablation that G = 12 and G = 6 collapse (see Q5).

### Gaps
- The *number of distinct prompts in each dataset* (and hence epochs-to-one-pass) is not stated in the configs; only `dataset/{geneval,ocr,pickscore}` paths. I did not enumerate the dataset files.
- No config records a total training-step/epoch count actually used for the published numbers — `num_epochs = 100000` with manual stopping means the published run lengths are not recoverable from the repo.

---

## Q2. Why a windowed SDE for FLUX (`sde_window_size=3`) while SD3 trains SDE on every step

### Takeaway
The window is **not FLUX-specific in principle** — it is the "Flow-GRPO-Fast" mechanism, and an SD3 config also uses it (`geneval_sd3_fast_nocfg`). But the repo's FAQ gives a FLUX-specific *reason* why FLUX in particular needs it: FLUX cannot be trained in fp16, bf16 inflates the log-probability mismatch between rollout and update, that mismatch is worse at low-noise steps, and therefore "training only on high-noise steps yields better results in this case." A second, independent reason appears in the code comments and in issue #141: full-trajectory SDE leaves visible noise in the final image handed to the reward model.

### Cited Findings
- **The decisive FAQ sentence** (README, FAQ section): *"Please use **fp16** for training whenever possible, as it provides higher precision than bf16, resulting in smaller log-probability errors between data collection and training. For Flux and Wan, becauase fp16 inference cannot produce valid images or videos, you will have to use **bf16** for training. Note that log-probability errors tend to be smaller at high-noise steps and larger at low-noise steps. **Training only on high-noise steps yields better results in this case.** Thanks to [Jing Wang] for these observations."* (typo "becauase" is in the original; emphasis mine) — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)
- **What the window does** — [flow_grpo/diffusers_patch/flux_pipeline_with_logprob_fast.py](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/flux_pipeline_with_logprob_fast.py) L135-142:
```python
# 假设一共denoising10步，sde_window_size=2, sde_window_range=(0, 10), sde_window可能是(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)
if sde_window_size > 0:
    start = random.randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
    end = start + sde_window_size
    sde_window = (start, end)
else:
    # 最后一步接近图片，高斯分布很尖，概率很大，容易精度溢出，不参与训练
    sde_window = (0, len(timesteps)-1)
```
  My translation of the two Chinese comments: (1) *"Suppose there are 10 denoising steps in total, sde_window_size=2, sde_window_range=(0,10); then sde_window may be (0,2), (1,3), (2,4), ... (8,10)."* (2) *"The last step is close to the image, the Gaussian distribution is very sharp, the probability is very large, and precision overflow happens easily, so it does not take part in training."*
- Inside the window the noise level is the configured one; outside it the step is pure ODE (`cur_noise_level = 0`), and only in-window steps are recorded for training:
```python
if i < sde_window[0]:
    cur_noise_level = 0
elif i == sde_window[0]:
    cur_noise_level= noise_level
    all_latents.append(latents)
elif i > sde_window[0] and i < sde_window[1]:
    cur_noise_level = noise_level
else:
    cur_noise_level= 0
...
if i >= sde_window[0] and i < sde_window[1]:
    all_latents.append(latents)
    all_log_probs.append(log_prob)
    all_timesteps.append(t)
```
— [flux_pipeline_with_logprob_fast.py](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/flux_pipeline_with_logprob_fast.py) L152-199
- **The non-fast pipelines have no window at all.** `flux_pipeline_with_logprob.py` and `sd3_pipeline_with_logprob.py` pass the single `noise_level` to `sde_step_with_logprob` on every iteration of the denoising loop and append `log_prob` unconditionally (`grep` shows `noise_level=noise_level` and an unguarded `all_log_probs.append(log_prob)` in both). So `train_sd3.py`/`train_flux.py` = stochastic every step; `train_sd3_fast.py`/`train_flux_fast.py` = windowed. — [sd3_pipeline_with_logprob.py](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py), [flux_pipeline_with_logprob.py](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/flux_pipeline_with_logprob.py)
- **README's own framing of the mechanism** (section "Flow-GRPO-Fast"): *"We propose Flow-GRPO-Fast, an accelerated variant of Flow-GRPO that requires training on **only one or two denoising step** per trajectory. For each prompt, we first generate a deterministic trajectory using ODE sampling. At a randomly chosen intermediate step, we inject noise and switch to SDE sampling to generate a group. The rest of the process continues with ODE sampling. This confines stochasticity to one or two steps, allowing training to focus solely on that steps. This few-step training idea was primarily proposed by [Ziyang Yuan] during our discussions in early June."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)
- **README "Training Speed" section** lists the window as one of three speed levers, alongside no-CFG and CPS: *"Use the window mechanism from **Flow-GRPO-Fast** or **[MixGRPO](https://www.arxiv.org/abs/2507.21802)** — only train on partial steps."* and *"Adopt **[Coefficients-Preserving Sampling](https://arxiv.org/abs/2509.05952) (CPS)** — CPS provides a notable improvement on GenEval, and produces higher-quality samples. A typical setting is `noise_level = 0.8`, which works well without tuning for different models or step counts."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)
- **The problem it solved on FLUX, in a real thread** — issue #141 "FLUX-dev更高步数的训练" (FLUX-dev training with more steps). A user reports bright-spot / noise artefacts in images when training FLUX at 8 steps. Author's first reply: *"如果步数较多，请增大config.sample.noise_level"* (my translation: "If the number of steps is large, please increase `config.sample.noise_level`"). After that failed, commenter `Jayce-Ping` wrote: *"仓库实现的train_flux.py是在所有时间步上都采用sde加噪的，这确实可能会导致喂给reward model的最终图像带有噪声。你可以试试像FlowGRPO-Fast一样只在前一半的时间步内随机选取一步采用sde，甚至可以直接固定只在timestep_index=1处采用sde，这一方面增加了多样性来鼓励RL探索，一方面使得最终图像不会带有噪声，还可以加速训练。我之前试过采用10步训练（timestep_indices=0~9），在timestep_index=1处采用sde（noise_level=0.9），其余采用ode。最后没有出现噪点，而且图像质量还有提升。"* (my translation: *"`train_flux.py` as implemented in this repo applies SDE noise at **every** timestep, which can indeed cause the final image fed to the reward model to carry noise. You can try, like FlowGRPO-Fast, randomly picking a single step within the **first half** of the timesteps to use SDE — or even fixing SDE at `timestep_index=1` only. On one hand this adds diversity to encourage RL exploration; on the other it keeps the final image free of noise, and it also speeds up training. I previously tried 10-step training (`timestep_indices=0~9`) with SDE at `timestep_index=1` (`noise_level=0.9`) and ODE elsewhere. In the end no noise specks appeared, and image quality even improved."*) The repo author endorsed this: *"@sharkDDD @Jayce-Ping 的comments非常准确"* ("the comments by @sharkDDD and @Jayce-Ping are very accurate"), and the original reporter confirmed the fix: *"我尝试了像flow-grpo-fast一样只在高噪部分进行训练，现在reward是可以正常上升的"* (my translation: *"I tried training only on the high-noise portion, like flow-grpo-fast, and now the reward rises normally."*) — [issue #141](https://github.com/yifan123/flow_grpo/issues/141)
- Same thread, independent corroboration of the artefact mode from `garychan22`: *"I am working on a new generative model without any timestep distillation, and sde with noise level 0.8, num_steps 28 would sometime (10-20%) produce meaningless noisy images. Should I decrease the noise level or avoid sde in early timesteps?"* — [issue #141](https://github.com/yifan123/flow_grpo/issues/141)
- Evaluation is pure ODE (`noise_level = 0`) in `train_flux.py`, which the same thread flags as a train/eval mismatch; the README's model-porting guide uses the same trick as a correctness check: *"Set `noise_level = 0` in sde_demo.py to check whether the generated images look normal."* — [issue #141](https://github.com/yifan123/flow_grpo/issues/141), [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)
- Commit provenance: `25f201b`/`a7cde10` "Add Flow-GRPO-Fast support for FLUX" and `3d02fd9` "Refactor FlowGRPO-Fast for compatibility with FlowGRPO; add CPS sampling support and No-CFG training on SD3" are the commits that introduce `sde_window_size` to the FLUX and SD3 paths; `git log -S"sde_window_size"` returns exactly `25f201b`, `914a82a`, `3d02fd9`, `40e1e38`, `47301ad`, `4b13ec7`. The earliest is `4b13ec7` "support Qwen Image". — repo `git log`

### Inferences
- **Important mechanical detail the configs hide:** for FLUX, `sde_window_range = (0, num_steps//2) = (0, 3)` with `sde_window_size = 3` makes `random.randint(0, 3 - 3)` = `random.randint(0, 0)` ≡ **0**. The FLUX "random" window is therefore **degenerate: it is always exactly steps 0, 1, 2** — the three highest-noise steps of six. By contrast `geneval_sd3_fast_nocfg` (`num_steps=10`, range `(0,5)`, size 3) draws `start ∈ {0,1,2}`, giving windows (0,3), (1,4), (2,5). So the FLUX recipe is not "a randomly placed window" — it is a fixed high-noise prefix, while the SD3 fast recipe really does randomise. This follows directly from the code and the configs; I found no comment or issue acknowledging it.
- README says Fast trains "only one or two denoising step", but every shipped window config uses `sde_window_size = 3`, i.e. three steps receive gradient. This is a documentation/code discrepancy (flag for the report).
- The two stated motivations are different failure modes and both point the same way: bf16 log-prob error (worse at low noise) argues for *excluding late steps from the loss*; residual noise in the reward-model input argues for *running late steps as ODE*. The window does both at once, which is why it reads as a FLUX-shaped fix even though the mechanism is model-agnostic.

### Gaps
- No paper section discusses the window at all — "Flow-GRPO-Fast", CPS and the FLUX window are **post-paper repository additions** (README changelog: *"- Adding Flow-GRPO-Fast."*, *"- Adding support for **FLUX.1-dev**."*, *"Refactor FlowGRPO-Fast for compatibility with FlowGRPO, add CPS sampling and No-CFG training on SD3"*). There is no peer-reviewed text justifying `sde_window_size = 3` for FLUX.
- No commit message explains the specific choice of `3` or of `(0, num_steps//2)`.

---

## Q3. Why `clip_range = 1e-5` for FLUX-fast vs `1e-4` (SD3 default) / `1e-3` (Wan2.1), and how the ratio is computed

### Takeaway
There *is* a stated rationale, in both the README FAQ and an author answer: small clip_range is required by the Fast/windowed variant or "training may crash", and the author explicitly says SD3 is robust to clip_range while other models are not. Mechanically, the importance ratio is `exp(log_prob - log_prob_old)` where `log_prob` is **mean-reduced over all non-batch latent dimensions**, not summed — which is the only reason clip values as small as 1e-4/1e-5 bind at all. Under `sde_type="cps"` the "log prob" additionally drops the Gaussian normalisation, so clip_range is not comparable between the `sde` and `cps` branches.

### Cited Findings
- **README FAQ:** *"When using **Flow-GRPO-Fast**, set a relatively small `clip_range`, otherwise training may crash."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md). Commit `7fa84b6` ("fix flow-grpo-s1 clip range, change s1 to fast", liujie <jieliu@link.cuhk.edu.hk>, Thu Aug 14 2025) is where this line was renamed from the earlier "Flow-GRPO-S1" wording, and the same commit changes `config.sample.noise_level` from `5` to `0.8` alongside `config.train.clip_range = 1e-5`.
- **Author's explicit rationale**, issue #105 "What is the role of clip range in training": *"The `clip_range` comes from the trust region proposed in PPO, which ensures that each update step of the model is relatively small so that the reward can increase monotonically. **The SD3 model is quite robust, so `clip_range` is not very sensitive in that case. However, for other models, it makes a bigger difference.** This is also why `general_ocr_wan2_1` was changed to `1e-3` and `pickscore_sd3_fast` was changed to `1e-5`."* — [issue #105](https://github.com/yifan123/flow_grpo/issues/105)
- **Author's debugging advice when ratios misbehave**, issue #131 "关于GRPO的ratio": *"可以调小clip_range到1e-5看看"* (my translation: *"You could try reducing `clip_range` down to 1e-5 and see."*) — [issue #131](https://github.com/yifan123/flow_grpo/issues/131)
- **How the log-prob is reduced** — [flow_grpo/diffusers_patch/sd3_sde_with_logprob.py](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py), end of the function:
```python
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
```
  i.e. **mean-reduced, not summed**, over every latent dimension.
- **The two SDE branches** in the same file:
```python
if sde_type == 'sde':
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    ...
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

elif sde_type == 'cps':
    std_dev_t = sigma_prev  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
    pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
    noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
    prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
    ...
    # remove all constants
    log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
```
  Note `# remove all constants`: in the CPS branch the `1/(2σ²)` factor and the normalising terms are **dropped**, so the quantity exponentiated in the ratio is a bare negative squared error, mean-reduced. — [sd3_sde_with_logprob.py](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py)
- Note the `sde` branch's `std_dev_t = sqrt(sigma/(1-sigma)) * noise_level` matches the paper's definition exactly: *"We use σ_t = a·sqrt(t/(1−t)) in this paper, where a is a scalar hyper-parameter that controls the noise level"* — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)
- **The loss** — [scripts/train_flux_fast.py](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_flux_fast.py) L842-855:
```python
advantages = torch.clamp(
    sample["advantages"][:, j],
    ...
ratio = torch.exp(log_prob - sample["log_probs"][:, j])
print("ratio", ratio)
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(
    ratio,
    1.0 - config.train.clip_range,
    1.0 + config.train.clip_range,
```
  — one scalar ratio per sample per step, so `clip_range` acts on a per-sample, dimension-averaged ratio.
- **The repo's own on-policy sanity check confirms ratios sit at exactly 1 when nothing differs**, README: *"Set `config.sample.num_batches_per_epoch = 1` and `config.train.gradient_accumulation_steps = 1` to enforce a purely on-policy setup... Under this setting, the ratio should remain **exactly 1**. If it's not, please check whether the sampling and training code paths differ."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)
- **The ratio is known to be biased, by the same group.** README "Over-optimization (GRPO-Guard)" section: *"We first observe that the importance ratio exhibits an inherent bias: 1. Its mean is consistently **below 1** and becomes significantly pronounced at low-noise steps (e.g., step 8 in SD3.5-M). 2. The variance varies notably across different steps. Ideally, the importance ratio distribution should have a mean of 1 and stable variance. The clipping operation truncates overly confident positive or negative samples outside the region [1−ϵ,1+ϵ], ensuring stable gradient updates. However, the observed bias in the importance ratio disrupts this mechanism—gradients of positive samples are no longer properly constrained, **leading the policy model into over-optimization**. As a result, the proxy score continues to rise while the gold score declines, causing a severe degradation in image quality."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md), referencing [GRPO-Guard, arXiv:2510.22319](https://arxiv.org/abs/2510.22319)

### Inferences
- Because the log-prob is a **mean** over latent dimensions (for FLUX at 512², packed latents are on the order of 10⁴–10⁵ elements per sample), `log_prob - log_prob_old` is a *per-dimension average* discrepancy. A summed log-prob would put the ratio astronomically far from 1 and make any ε of order 1e-4 vacuous (always clipped). So clip ranges of 1e-4/1e-5 are only meaningful in this mean-reduced parameterisation — they correspond to roughly 1e-4/1e-5 *nats per latent dimension*. Any comparison of FlowGRPO's ε to a text-LLM PPO ε (typically 0.2, on a summed-over-one-token log-prob) is a category error.
- Since the CPS branch drops the `1/(2σ²)` and log-normaliser terms, the scale of `log_prob` differs from the true log-density by a step-dependent factor `2·std_dev_t²·|dt|`. At `noise_level = 0.8`, `std_dev_t ≈ 0.951·σ_prev`, so that factor is well below 1 for mid/low σ and above 1 near σ≈1. The upshot: the magnitude of `ratio − 1` under `cps` is *not* on the same scale as under `sde`, so `clip_range` must be retuned when switching branches. This is a plausible mechanical explanation for why both window+CPS configs (FLUX and SD3-fast-nocfg) use 1e-5 while the plain-SDE configs use 1e-4 — **this is my inference from the code, not a stated claim.**
- Note the confound in the headline question: `clip_range = 1e-5` is attached to the **Fast/CPS/windowed** recipe, and the only SD3 configs that use that recipe *also* use 1e-5. The 1e-3 value is a Wan2.1 *video* config, not SD3. So the real dichotomy is **plain SDE → 1e-4 (both SD3 and FLUX) vs window+CPS → 1e-5 (both SD3 and FLUX)**, plus the author's separate statement that SD3 is less sensitive to this knob than other models.

### Gaps
- No stated numerical reasoning (no derivation, no logged ratio histograms in the repo) for *why* 1e-5 rather than, say, 3e-5. The justification is empirical ("otherwise training may crash").
- `train.clip_range_lt` / `clip_range_gt` exist in `base.py` (both 1e-4) and appear in asymmetric-clip diagnostics in `train_flux_fast.py`, but no shipped FLUX/SD3 config overrides them; I did not trace whether the dual-sided variant is active in the loss for all scripts.

---

## Q4. Reported results: FLUX vs SD3

### Takeaway
The paper's headline results are **SD3.5-Medium only**. FLUX.1-dev appears in the paper exclusively in Appendix C.4, as a PickScore run evaluated on DrawBench — no GenEval, no OCR, no reward-curve table. The improvement is real but far smaller and narrower than SD3's (e.g. PickScore 22.62 → 23.97; DeQA actually *drops* 4.31 → 4.24). FLUX GenEval results exist only as W&B links in the README, with no numbers written down.

### Cited Findings
**SD3.5-M (paper, Table 2 — "Performance on Compositional Image Generation, Visual Text Rendering, and Human Preference benchmarks, evaluated by task performance on test prompts, and by image quality and preference scores on DrawBench prompts"):** — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)

| Model | GenEval | OCR Acc. | PickScore (task) | Aesthetic | DeQA | ImageReward | PickScore | UnifiedReward |
|---|---|---|---|---|---|---|---|---|
| SD3.5-M (base) | 0.63 | 0.59 | 21.72 | 5.39 | 4.07 | 0.87 | 22.34 | 3.33 |
| + Flow-GRPO, GenEval task (w/o KL) | **0.95** | — | — | 4.93 | 2.77 | 0.44 | 21.16 | 2.94 |
| + Flow-GRPO, GenEval task (w/ KL) | **0.95** | — | — | 5.25 | 4.01 | 1.03 | 22.37 | 3.51 |
| + Flow-GRPO, Text task (w/o KL) | — | **0.93** | — | 5.13 | 3.66 | 0.58 | 21.79 | 3.15 |
| + Flow-GRPO, Text task (w/ KL) | — | **0.92** | — | 5.32 | 4.06 | 0.95 | 22.44 | 3.42 |
| + Flow-GRPO, Preference task (w/o KL) | — | — | **23.41** | 6.15 | 4.16 | 1.24 | 23.56 | 3.57 |
| + Flow-GRPO, Preference task (w/ KL) | — | — | **23.31** | 5.92 | 4.22 | 1.28 | 23.53 | 3.66 |

- Abstract: *"For compositional generation, RL-tuned SD3.5-M generates nearly perfect object counts, spatial relations, and fine-grained attributes, increasing GenEval accuracy from 63% to 95%. In visual text rendering, accuracy improves from 59% to 92%, greatly enhancing text generation. Flow-GRPO also achieves substantial gains in human preference alignment. Notably, very little reward hacking occurred, meaning rewards did not increase at the cost of appreciable image quality or diversity degradation."* — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)
- GenEval sub-scores (paper Table 1): SD3.5-M `0.63 / 0.98 / 0.78 / 0.50 / 0.81 / 0.24 / 0.52` (Overall, Single Obj., Two Obj., Counting, Colors, Position, Attr. Binding) → SD3.5-M+Flow-GRPO `0.95 / 1.00 / 0.99 / 0.95 / 0.92 / 0.99 / 0.86`. For reference, the same table lists untuned **FLUX.1 Dev** at `0.66 / 0.98 / 0.81 / 0.74 / 0.79 / 0.22 / 0.45` — a baseline row, *not* an RL-trained FLUX row. — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)
- T2I-CompBench++ (paper Table 3): untuned FLUX.1 Dev `0.7407 / 0.5718 / 0.6922 / 0.2863 / 0.3866 / 0.6185 / 0.3127`; SD3.5-M `0.7994 / 0.5669 / 0.7338 / 0.2850 / 0.3739 / 0.5927 / 0.3146`; SD3.5-M+Flow-GRPO `0.8379 / 0.6130 / 0.7236 / 0.5447 / 0.4471 / 0.6752 / 0.3195`. Again FLUX is a baseline only. — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)
- **The entire FLUX result in the paper** is Appendix C.4: *"We run Flow-GRPO on FLUX.1-Dev [5] using PickScore as the reward signal. The reward curve rises steadily throughout training without noticeable reward hacking. Figure 11 shows the reward values over the training process, and Table 7 compares FLUX.1-Dev with FLUX.1-Dev + Flow-GRPO on DrawBench."* — Table 7 ("Comparison of FLUX.1-Dev and Flow-GRPO fine-tuned models"): — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)

| Model | Aesthetic | DeQA | ImageReward | PickScore | UnifiedReward |
|---|---|---|---|---|---|
| FLUX.1-Dev | 5.71 | 4.31 | 0.85 | 22.62 | 3.65 |
| FLUX.1-Dev + Flow-GRPO | **6.02** | **4.24** (↓) | **1.32** | **23.97** | **3.81** |

- The paper's own section index confirms FLUX is appendix-only: *"Section C offers some additional experimental results, including 1) the comparison with other alignment methods, 2) ablation of denoising reduction on OCR accuracy and pickscore, 3) ablation of initial noise, **4) additional results on FLUX.1-Dev**, 5) the learning curves of Flow-GRPO on three tasks, 6) additional qualitative results, and 7) evolution of evaluation images during training."* — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)
- **FLUX GenEval/PickScore results in the repo are W&B links only, no tabulated numbers:** *"For Flow-GRPO-Fast, please use `scripts/multi_node/flux_fast.sh`. See the W&B logs for [Geneval](https://api.wandb.ai/links/ljie/qz47q208) (with `geneval_flux_fast` in the config) and [PickScore](https://api.wandb.ai/links/ljie/ncdwa0wo) (with `pickscore_flux_fast` in the config)."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md). An earlier FLUX log is linked from [issue #163](https://github.com/yifan123/flow_grpo/issues/163): `https://api.wandb.ai/links/ljie/72owwyku`.
- Released checkpoints exist only for SD3.5-M — `🤗GenEval` (`jieliu/SD3.5M-FlowGRPO-GenEval`), `🤗Text` (`jieliu/SD3.5M-FlowGRPO-Text`), `🤗PickScore` (`jieliu/SD3.5M-FlowGRPO-PickScore`). **No FLUX checkpoint is released.** — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)

### Inferences
- FLUX does **not** improve as much as SD3 does in anything that was published: the SD3 story is a +32-point GenEval jump and +33-point OCR jump on task metrics; the FLUX story is a +1.35 PickScore / +0.47 ImageReward / +0.31 Aesthetic move on DrawBench quality proxies with a small DeQA regression, and no task-benchmark number at all.
- The asymmetry is partly structural: FLUX was only ever run with PickScore in the paper (a preference reward evaluated by preference metrics, i.e. partly in-distribution with the reward), never on GenEval or OCR, so there is no FLUX measurement of the headline capability gains.

### Gaps
- No FLUX GenEval or OCR numbers exist in any primary text source. The W&B links would have them, but I did not fetch W&B (requires rendering a JS dashboard) — flagging as retrievable-but-unretrieved.
- No FLUX.1-Kontext numbers; the README explicitly says the editing pipeline is unvalidated: *"This implementation offers a runnable pipeline, but the training set contains only 800 samples. Making Flow-GRPO truly effective for editing tasks still requires further exploration by the community."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)

---

## Q5. Seeds, error bars, run-to-run variance

### Takeaway
**None.** There is no seed sweep, no error bar, no standard deviation across runs, and no statement of how many runs any number represents, in either the paper or the repo. `base.py` hard-codes a single `config.seed = 42`. The only "variance" language in the paper is about advantage-estimation variance within a batch (group size) and about output diversity, not run-to-run variance.

### Cited Findings
- `config/base.py`: `# random seed for reproducibility.` / `config.seed = 42` — a single seed, never overridden in any SD3 or FLUX config. — [config/base.py](https://github.com/yifan123/flow_grpo/blob/main/config/base.py)
- A full-text search of the paper for `seed`, `error bar`, `std`, `variance`, `three runs`, `averaged over` returns only these relevant hits:
  - *"RL depends on exploration. As shown in Section 5.3, reduced randomness greatly lowers training efficiency. **Deterministic sampling, with no randomness beyond the initial seed, is especially problematic.**"*
  - *"In the Human Preference Alignment task, removing KL does not affect image quality, likely due to overlap between PickScore and evaluation metrics, but causes a collapse in visual diversity. Outputs converge to a single style, **with different seeds producing nearly identical results.** KL regularization prevents this collapse and maintains diversity."* (this is seed-to-seed *output* diversity of one trained model, not run-to-run training variance)
  - *"Figure 5 shows the effect of group size G using PickScore as the reward function. When the group size was reduced to G=12 and G=6, **training became unstable and eventually collapsed**, whereas G=24 remained stable throughout the process. We observe that smaller group sizes produce **inaccurate advantage estimates, increasing variance and leading to training collapse**, a phenomenon also reported in [71, 72]."*
  — all [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)
- The paper's full statement of GRPO hyperparameters (Appendix): *"Except for β, GRPO hyperparameters are fixed across tasks. We use a sampling timestep T=10 and an evaluation timestep T=40. Other settings include a group size G=24, an noise level a=0.7 and an image resolution of 512. The KL ratio β is set to 0.04 for GenEval and Text Rendering, and 0.01 for Pickscore. We use Lora with α=64 and r=32."* — note this matches the SD3 configs exactly (and says nothing about learning rate, clip range, or seeds). — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)
- Noise-level ablation wording, which reports a hard failure mode rather than a variance band: *"A small a (e.g., 0.1) limits exploration and slows reward improvement. Increasing a (up to 0.7) boosts exploration and speeds up reward gains. Beyond this point (e.g., from 0.7 to 1.0), further increases provide no additional benefit, as exploration is already sufficient. **We also observe that injecting too much noise by further increasing a degrades image quality, resulting in zero reward and failed training.**"* — [arXiv:2505.05470v5](https://arxiv.org/html/2505.05470v5)
- Training length is not fixed and not recorded: `config.num_epochs = 100000` with `# A large num_epochs is intentionally set here. Training will be manually stopped once sufficient`, and the author's stated stopping criterion in issue #110 is *"我一般是看实际效果"* (my translation: *"I generally just look at the actual visual results."*) — [config/grpo.py](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py), [issue #110](https://github.com/yifan123/flow_grpo/issues/110)

### Inferences
- Every published FlowGRPO number — SD3 and FLUX alike — is a **single run at seed 42, manually stopped on visual inspection**. There is no basis in the primary sources for estimating run-to-run spread, which matters directly for anyone trying to attribute a reproduction gap to seed variance versus configuration.
- The paper's one quantitative stability finding (G=24 stable, G=12/G=6 collapse) is itself single-run per setting.

### Gaps
- **OpenReview reviews: not retrieved.** The paper is a NeurIPS 2025 main-track poster ([neurips.cc/virtual/2025/poster/116065](https://neurips.cc/virtual/2025/loc/san-diego/poster/116065), [proceedings page](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a10c46572628d58cb44fb705f25cbbf-Abstract-Conference.html)), but a web search for its OpenReview forum returned no direct forum URL, and an `api2.openreview.net/notes/search` query for the title returned no notes. I therefore have **no reviewer or author-response material on variance or ablations** — this remains an open lead for anyone with the forum ID.

---

## Q6. GitHub issues reporting instability, divergence, or reproduction difficulty — FLUX specifically

### Takeaway
Yes, several, and the FLUX-specific ones cluster on two things: numerical precision (bf16 → NaN log-probs) and full-trajectory SDE producing noisy images / non-rising rewards at higher step counts. There is also an open, unresolved reproduction failure — but it is on the **SD3** fast-nocfg recipe, not FLUX.

### Cited Findings
- **#141 "FLUX-dev更高步数的训练"** (closed) — FLUX reward fails to rise and images show bright specks when training at 8 steps with every-step SDE; resolved only by switching to high-noise-only (window) training. Full quotes in Q2. — [issue #141](https://github.com/yifan123/flow_grpo/issues/141)
- **#177 "print("log_prob is nan")"** (closed) — NaN log-probs on a FLUX-family model. Author's triage: *"1. In the file `flux_kontext_pipeline_with_logprob.py` at line 213, is any of the inputs NaN? 2. If not, please check whether the model is using `bf16` precision."* Reporter: *"As expected, it's a precision issue. Thank you, expert!"* Follow-ups from two other users: *"i am also facing the same issue. So, when training flux-kontext, using **bf16** is preferred?"* and *"I encountered the same problem while training Qwen-Image... `all_log_probs: [tensor([nan], device='cuda:6'), ...]`"*. — [issue #177](https://github.com/yifan123/flow_grpo/issues/177)
- **#113 "Problem with Enabling KL Loss (beta > 0) in FLUX OCR Training: Shape Mismatch for prev_sample_mean and prev_sample_mean_ref"** (closed) — the KL path was broken on FLUX; commit `c86e18e` is titled "fix kl in flux and flux kontext". Relatedly, in **#110**, a user trying FLUX with `beta=0.04` reports a crash: *"尝试使用上图config（beta=0.04）以后训练报错如下: ... 如果beta设置为0则没有这个问题，reward快速上升，因此怀疑是beta设置太大了，这个参数是只能根据经验调整吗"* (my translation: *"After trying the config above (beta=0.04) training throws the error below... If beta is set to 0 there is no such problem and the reward rises quickly, so I suspect beta was set too large — can this parameter only be tuned empirically?"*). This is consistent with every shipped FLUX config using `beta = 0`. — [issue #113](https://github.com/yifan123/flow_grpo/issues/113), [issue #110](https://github.com/yifan123/flow_grpo/issues/110)
- **#204 "Over Optimization Issues of GRPO-GUARD on Flux-dev for Text Generation"** (open) — reward hacking on FLUX-dev text rendering. Author's mitigations: *"`grpo_guard` can improve training stability. A more direct approach to eliminating reward hacking is to directly adopt the MSE of velocity as the KL loss; alternatively, EMA can be used. With these two techniques in place, I barely observe any reward hacking at all."* and concretely: *"Change `kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)` to `kl_loss = ((v_pred - v_pred_ref) ** 2).mean()`"*, plus *"I have verified the two tricks mentioned above on more generalized prompts and larger-scale models, and they are proven to be effective."* The reporter pushes back on the paper's low-reward-hacking claim: *"your original OCR dataset is relatively simple, which is likely why you didn't observe reward hacking. In my more demanding case, this issue indeed exists."* — [issue #204](https://github.com/yifan123/flow_grpo/issues/204)
- **#215 "Training Issue on FLUX-kontext"** (open) and **#231 "question about 'sde' and 'cps' sampling when noise_level=0"** (open) — further unresolved FLUX-family / SDE-branch threads. — [issue list](https://github.com/yifan123/flow_grpo/issues)
- **#230 "Cannot reproduce GenEval 0.95 result with fewer GPUs — plateaus at 0.9 with reward hacking"** (open, SD3 not FLUX) — *"I'm trying to reproduce the `geneval_sd3_fast_nocfg` result (GenEval Score reaching 0.95) using 4 GPUs instead of the 32 GPUs. My training plateaus at ~0.9 and does not improve further, despite matching all optimization dynamics. The generated images also show signs of reward hacking — the model appears to game the GenEval evaluation pipeline rather than genuinely improving compositional generation."* The reporter documents matching total samples/epoch (1152), group_number (48), optimizer steps/epoch (2), effective batch/step (576), `learning_rate 3e-4`, `clip_range 1e-5`, `beta 0`, `noise_level 0.8`, `sde_type cps`, `sde_window_size 3`, `same_latent False`, `mixed_precision fp16` — only `train_batch_size` (9→12), `num_batches_per_epoch` (4→24) and `gradient_accumulation_steps` (2→12) differ. **As of this research the only reply is from another user** (*"how do you know that this config train_sd3_fast.py is for the reported result in the paper?"*); the maintainer has not answered. — [issue #230](https://github.com/yifan123/flow_grpo/issues/230)
- **#172 "Help replicating on low-VRAM GPUs (40GB/24GB) - reward fails to increase"** (closed) and **#151 "Is the training normal?"** (closed) — further reward-fails-to-rise reports. — [issue list](https://github.com/yifan123/flow_grpo/issues)
- A batch-size-sensitivity caveat that bears on reproducibility, README FAQ: *"When implementing a new model, please check whether using different batch sizes leads to slight differences in the output. **SD3 has this issue**, which is why I ensure that the batch size for training is the same as that used for data collection."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)
- An open P0-labelled issue flags the batching/accumulation logic itself: **#73 "Gradient accumulation and batch size handling in training loop"** (open, labels `enhancement, P0`). — [issue list](https://github.com/yifan123/flow_grpo/issues)

### Inferences
- The FLUX failure reports are concentrated in exactly the places the FLUX regime deviates from SD3: bf16 (NaNs), KL (shape bug → `beta=0` everywhere), and every-step SDE at higher step counts (noisy images, flat rewards). That is strong convergent evidence that the FLUX hyperparameter regime is a set of *workarounds for FLUX-specific numerical fragility*, not an independently optimised recipe.
- Issue #230 is the cleanest reproduction datapoint available, and it indicates that even with every documented hyperparameter matched and throughput held constant, a 4-GPU re-run of the *SD3* flagship recipe lands at ~0.90 rather than 0.95 with visible reward hacking — i.e. the published numbers carry an unquantified sensitivity to sharding/micro-batching that no error bar covers. Unanswered by the maintainers so far.

### Gaps
- No issue explicitly reports *divergence to NaN loss* or *seed-dependent collapse* on FLUX with the shipped `geneval_flux_fast`/`pickscore_flux_fast` configs; the FLUX instability reports are about `train_flux.py` (non-fast), FLUX-Kontext, or custom step counts.
- I did not read the GRPO-Guard paper (arXiv:2510.22319) itself, which likely quantifies the ratio bias per step; it is cited by the README as the source of the mean-below-1 / step-varying-variance observations.

---

## Q7. How many denoising steps receive gradient, and is high-noise-only intentional

### Takeaway
Explicitly intentional, and stated in the README FAQ: *"Training only on high-noise steps yields better results in this case."* Gradient reaches 9 of 10 steps in the SD3 flagship recipe, 5 of 6 in non-fast FLUX, and **3 of 6 — always the three highest-noise steps — in `geneval_flux_fast`/`pickscore_flux_fast`**, because the window-start randomisation is degenerate for FLUX's step count.

### Cited Findings
- **Non-fast scripts** gate trained steps with `timestep_fraction`: `num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)` appears in `scripts/train_sd3.py` (L345) and `scripts/train_flux.py` (L337). With `timestep_fraction = 0.99`: SD3 → `int(10*0.99) = 9` of 10; FLUX → `int(6*0.99) = 5` of 6. — [train_sd3.py](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3.py), [train_flux.py](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_flux.py)
- `base.py` comment on the knob: *"the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates."* — [config/base.py](https://github.com/yifan123/flow_grpo/blob/main/config/base.py)
- **`timestep_fraction` does not exist in the fast scripts**: `grep -rn "timestep_fraction" scripts/train_flux_fast.py scripts/train_sd3_fast.py` returns nothing. In the fast path the trained steps are exactly the window, and the `*_fast` FLUX configs do not set `timestep_fraction` at all (so it stays at the unused default 1.0). — repo grep
- **The window excludes the last step by design even when disabled**: `sde_window = (0, len(timesteps)-1)` with the comment *"最后一步接近图片，高斯分布很尖，概率很大，容易精度溢出，不参与训练"* (my translation: *"the last step is close to the image, the Gaussian is very sharp, the probability is very large, precision overflow happens easily — it does not take part in training"*). This is the same motivation as `timestep_fraction = 0.99` in the non-fast path: drop the final, lowest-noise step. — [flux_pipeline_with_logprob_fast.py](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/flux_pipeline_with_logprob_fast.py)
- **Intent, stated:** README FAQ — *"Note that log-probability errors tend to be smaller at high-noise steps and larger at low-noise steps. **Training only on high-noise steps yields better results in this case.**"* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)
- **Intent, corroborated in-thread and endorsed by the author:** issue #141, `Jayce-Ping`'s recommendation to *"只在前一半的时间步内随机选取一步采用sde"* ("randomly pick one step within only the first half of the timesteps to use SDE"), which the author called *"非常准确"* ("very accurate"). The FLUX fast configs encode exactly that with `sde_window_range = (0, config.sample.num_steps//2)`. — [issue #141](https://github.com/yifan123/flow_grpo/issues/141)
- **Related independent work** doing the same thing is cited approvingly in the README: *"Use the window mechanism from Flow-GRPO-Fast or [MixGRPO](https://www.arxiv.org/abs/2507.21802) — only train on partial steps."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)
- The README also unifies the window for Qwen-Image: *"In the implementation of Qwen-Image, we have unified Flow-GRPO and Flow-GRPO-Fast. You can control the size of the SDE window with `config.sample.sde_window_size`, and adjust the position of the window with `config.sample.sde_window_range`."* — [README.md](https://github.com/yifan123/flow_grpo/blob/main/README.md)

### Inferences
- **Flagged discrepancy (code vs. README):** README describes Flow-GRPO-Fast as training *"only one or two denoising step"*, but every shipped window config uses `sde_window_size = 3`, so three steps receive gradient.
- **Flagged discrepancy (config intent vs. config effect), FLUX only:** with `sde_window_range = (0, 3)` and `sde_window_size = 3`, `random.randint(0, 3-3)` is identically 0. The FLUX fast recipe therefore trains a **fixed** window of steps {0,1,2} every rollout, with zero randomisation of window position — unlike `geneval_sd3_fast_nocfg` (`num_steps=10`, range `(0,5)`) which genuinely samples `start ∈ {0,1,2}`. If the intent was a randomly placed high-noise window, the FLUX configs do not achieve it. I found no acknowledgement of this anywhere in the repo, issues, or paper.
- Taken together: in the FLUX fast regime, **gradient only ever reaches the top half of the noise schedule, at a fixed location, with the bottom half run as pure ODE**. That is the single largest structural difference from the SD3 flagship recipe, which back-propagates through 9 of 10 steps spanning the whole schedule.

### Gaps
- No ablation anywhere (paper or repo) isolating window *position* or window *size* on FLUX — the W&B links are single curves, and there is no table comparing e.g. `sde_window_size` 1/2/3 on FLUX.
- The paper's "Denoising Reduction" ablations (Figures 7a and 9) are about *how many inference steps are used during rollout* (10 vs 40), not about which steps receive gradient; they are SD3-only.
