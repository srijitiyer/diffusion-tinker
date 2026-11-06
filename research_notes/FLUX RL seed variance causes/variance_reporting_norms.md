# Methodological norms around seeds, error bars and variance reporting in diffusion-model RL

Scope note for the report writer: I verified seed/variance reporting by downloading the arXiv HTML full text of each paper (abstract + body + appendices + references) and doing exact string searches for `seed`, `error bar`, `standard deviation`, `standard error`, `variance`, `confidence interval`, `averaged over`, `trials`, `3 seeds`, `5 seeds`. Counts below are literal match counts in the full text, so "0 occurrences of 'error bar'" is a checkable fact, not an impression. Where a paper does not state a training-seed count, I record it as **not stated**.

## Survey of the major diffusion-RL papers: seeds per configuration and error bars

### Takeaway
Across the ten canonical diffusion/flow-matching RL and reward-finetuning papers I checked, **not one states a training-seed count, and the string "error bar" appears zero times in all ten full texts**. Every uncertainty figure any of them reports is computed over *evaluation samples* (or over human raters), never over independent training runs. Single-run reporting is not merely the norm — it is, in this sample, universal.

### Cited findings

Literal full-text keyword counts (arXiv HTML, retrieved 2026-10-27):

| Paper | arXiv | "seed" | "error bar" | "standard deviation" | "confidence interval" | Training seeds per config | Uncertainty reported? |
|---|---|---|---|---|---|---|---|
| DDPO (*Training Diffusion Models with RL*) | [2305.13301](https://arxiv.org/html/2305.13301) | 2 | 0 | 1 | 0 | **not stated** | Only "mean and one standard error over 50 samples" for one prompt; reward curves have no bands |
| DPOK | [2305.16381](https://arxiv.org/html/2305.16381) | 7 | 0 | 1 | 0 | **not stated** | Scores "averaged over 50 samples"; one std dev is over **eight human raters** |
| DRaFT | [2309.17400](https://arxiv.org/html/2309.17400) | 1 | 0 | 0 | 0 | **not stated** | None found |
| AlignProp | 2310.03739 | — | — | — | — | **not stated** | arXiv HTML not available (6 KB stub); not verified |
| ReFL / ImageReward | [2304.05977](https://arxiv.org/html/2304.05977) | 1 | 0 | 0 | 0 | **not stated** | None found |
| Diffusion-DPO | [2311.12908](https://arxiv.org/html/2311.12908) | 1 | 0 | 0 | 0 | **not stated** | None found (the single "seed" hit is about sampling seeds for qualitative figures) |
| Flow-GRPO | [2505.05470](https://arxiv.org/html/2505.05470) | 3 | 0 | 0 | 0 | **not stated** | None in tables; see "evaluation noise" section for what they *do* average over |
| DanceGRPO | [2505.07818](https://arxiv.org/html/2505.07818) | 3 | 0 | 0 | 0 | **not stated** | None found |
| SRPO | [2509.06942](https://arxiv.org/html/2509.06942) | 2 | 0 | 0 | 0 | **not stated** | None found |
| MixGRPO | [2507.21802](https://arxiv.org/html/2507.21802) | 0 | 0 | 1 (a math symbol, σ_t) | 0 | **not stated** | None found |
| DDRL (*Data-regularized RL for Diffusion Models at Scale*) | [2512.04332](https://arxiv.org/abs/2512.04332) | 2 | 0 | 2 | 0 | **not stated** | None; both "standard deviation" hits are about the GRPO advantage normalizer and about reward-hacking diagnostics |

- Crucially, in every one of these papers the word "seed" is used almost exclusively to mean the **sampling seed held fixed for qualitative side-by-side figures**, not a training-run seed. DDPO: "Each row shows a progression of samples for the same prompt and random seed over the course of training." — [DDPO, arXiv:2305.13301](https://arxiv.org/html/2305.13301)
- DPOK, same usage, four separate times: "Images in the same column are generated with the same random seed." — [DPOK, arXiv:2305.16381](https://arxiv.org/html/2305.16381)
- DRaFT: "Comparison of generations (using the same random seed) from DRaFT models fine-tuned using the LAION Aesthetic Classifier, Human Preference Score v2, and PickScore reward functions." — [DRaFT, arXiv:2309.17400](https://arxiv.org/html/2309.17400)
- SRPO: "Qualitative Comparison on FLUX, DanceGRPO and SRPO with same seed." — [SRPO, arXiv:2509.06942](https://arxiv.org/html/2509.06942)
- The only uncertainty statement in DDPO is over evaluation samples, not runs: "We only report results averaged over 50 samples for a single prompt, since universal guidance is very slow" and, for the comparison table, "scores are averaged over 50 samples for each prompt." — [DDPO, arXiv:2305.13301](https://arxiv.org/html/2305.13301)
- DPOK's only ± quantity is over human annotators, not runs: "The results show the mean and standard deviation averaged over eight independent human raters." — [DPOK, arXiv:2305.16381](https://arxiv.org/html/2305.16381)
- DDRL (Dec 2025), despite "over a million GPU hours of experiments and ten thousand double-blind human evaluations," reports no seed count and no run-to-run interval; its uncertainty-adjacent language is about reward hacking diagnostics: "if the diffusion loss of a post-trained checkpoint increases by more than 10%, its outputs are mostly hacked; a steep increase in rewards or a sharp decrease in standard deviation are also potential signals." — [DDRL, arXiv:2512.04332](https://arxiv.org/html/2512.04332) (submitted 3 Dec 2025, v3 24 Dec 2025, per [arXiv abstract page](https://arxiv.org/abs/2512.04332))
- DDRL explicitly characterises DanceGRPO's regularization as *seed fixing*: baselines anchor "to p_ref by either leveraging the reverse KL divergence (FlowGRPO) or by adopting heuristics such as a fixed initial random seed (DanceGRPO)." — [DDRL, arXiv:2512.04332](https://arxiv.org/html/2512.04332)

### Inferences
- Fixing the initial noise/seed is used in this literature as a *training stabiliser and anti-reward-hacking heuristic* (DanceGRPO per DDRL's description; Flow-GRPO's codebase exposes `config.sample.same_latent`, "Whether to use the same noise for the same prompt", see [flow_grpo issue #227 config dump](https://github.com/yifan123/flow_grpo/issues/227)). This means the field has a *methodological incentive to suppress* exactly the stochasticity that multi-seed reporting would measure, which further explains the absence of seed studies.
- Because none of these papers reports a seed count, a reader cannot distinguish "best of several runs" from "the only run" for any headline number in the diffusion-RL literature. The reporting convention does not permit that distinction.

### Gaps
- AlignProp (arXiv:2310.03739) has no arXiv HTML rendering, so I could not run the same literal keyword check on its full text; treat its row as unverified rather than as "no error bars."
- I could not retrieve OpenReview review threads (see below), so I cannot say whether any of these papers reported seed counts privately in rebuttal.

## Is single-seed reporting the norm? Explicit statements

### Takeaway
Yes — and I found no paper in the set that even *discusses* training-seed count. The closest thing to an explicit statement about randomness control comes from the Flow-GRPO authors in a GitHub thread, and it is about evaluation sampling, not training seeds.

### Cited findings
- Flow-GRPO's first author, explaining the Table 2 protocol: "the test prompts in Table 2 are from DrawBench. **To reduce randomness, we repeated each of the 200 prompts 5 times and reported the average reward over all 1000 samples.**" — [yifan123, flow_grpo issue #39, 2025-07-21](https://github.com/yifan123/flow_grpo/issues/39)
- The Flow-GRPO OCR/GenEval eval set is likewise a large fixed prompt set, not repeated training runs: the released config comments note "the test set has a total of 1018" prompts and sets `test_batch_size = 16` so that `8*16*n` lands near 1018. — [flow_grpo issue #227 config](https://github.com/yifan123/flow_grpo/issues/227)
- Flow-GRPO's own text frames randomness as an *input* to be engineered (exploration), never as an output to be measured: "RL depends on exploration… reduced randomness greatly lowers training efficiency. Deterministic sampling, with no randomness beyond the initial seed, is especially problematic." — [Flow-GRPO, arXiv:2505.05470](https://arxiv.org/html/2505.05470)
- Flow-GRPO on diversity collapse without KL: "Outputs converge to a single style, with different seeds producing nearly identical results." Again, "seeds" = sampling seeds. — [Flow-GRPO, arXiv:2505.05470](https://arxiv.org/html/2505.05470)
- Contrast with the *diffusion-policy-for-control* branch of the same broad field, which inherits D4RL/classic-RL norms and does report seeds: a 2026 survey records "DIVO results are reported as mean ± standard error over the last 10 evaluations and 5 random seeds." — [RL-driven Diffusion Models survey, arXiv:2610.21539](https://arxiv.org/html/2610.21539)

### Inferences
- There is a clear norm split *within* diffusion RL: offline-RL / robotics-control diffusion papers report 5 seeds ± standard error (inherited from the Henderson-era conventions of that community), while **text-to-image / text-to-video diffusion RL reports none**. The T2I branch's variance-reduction effort is spent entirely on the evaluation side (many prompts × repeats), not the training side.
- The practical implication for interpreting two FLUX seeds at OCR 0.558 and 0.651: no paper in this literature publishes a number you could compare that spread against, because no paper publishes a between-seed interval at all.

## Reviewer and reproduction concerns about variance / reproducibility

### Takeaway
OpenReview review text was not retrievable for me (bot wall + API 400s), but the method repositories contain a substantial, dated record of failed and divergent reproductions — including a still-open issue stating the official Flow-GRPO SD3 OCR config produces a flat reward curve, and a thread where the reward only started increasing after removing DeepSpeed.

### Cited findings
- **Open, unresolved: official OCR config does not reproduce.** "Can't reproduce sd3 OCR results… I'm trying to run the base setup provided for sd3 with OCR reward on 8xH200… But **the reward is pretty much stuck in place, both during training and eval**." — [kfirgoldberg, flow_grpo issue #227, opened 2026-02-05, still open](https://github.com/yifan123/flow_grpo/issues/227)
- **Hardware/parallelism sensitivity.** "While the training runs successfully without OOM errors and I've adjusted the configuration to match the original global batch size (1152), **I am not observing any stable increase in the PickScore reward.** … A colleague mentioned that he also struggled to reproduce the results on a different GPU setup and suggested that **training might only be stable on the exact 80GB GPU configuration.**" — [jaylee2000, flow_grpo issue #172, 2025-10-30](https://github.com/yifan123/flow_grpo/issues/172)
- Same thread, resolution and warning: "With DeepSpeed off, I can still run the OCR experiment on 24GB GPUs, and reward seems to be increasing consistently. My guess is that DeepSpeed was the culprit." … "**To anyone who comes across this issue later: I highly discourage using Deepspeed, as it seems to destabilize training.**" — [jaylee2000, flow_grpo issue #172, 2025-11-07 and 2025-11-13](https://github.com/yifan123/flow_grpo/issues/172)
- A second user in that thread reports reward *declining* on FLUX-Kontext and LoRA weights producing blue noise at inference ("奖励曲线起伏着下跌… 产生的图像全是蓝色噪声"). — [XiangqianHuang, flow_grpo issue #172, 2025-11-03](https://github.com/yifan123/flow_grpo/issues/172)
- **Author acknowledges reward-decrease failures on new models.** User: "I have tested the flow grpo on other models and found that **in some cases, the reward scores even decrease. Is this caused by the sensitivity to hyperparameters?**" Author: "I'm also experimenting with other models myself… Initially, after migrating, I noticed the reward kept dropping. After a few days of debugging, I found it was due to a minor bug introduced during the migration process." — [flow_grpo issue #23, 2025-06-02](https://github.com/yifan123/flow_grpo/issues/23)
- **Author's hyperparameter-rigidity advice** (i.e. the result is not robust to config perturbation): "If you reduce the number of GPUs, please keep the hyperparameters unchanged. **Lowering the batch size or group size may cause the reward to stop increasing.**" — [yifan123, flow_grpo issue #172, 2025-11-11](https://github.com/yifan123/flow_grpo/issues/172)
- **Numerical non-determinism acknowledged in the codebase design:** per the search snippet of the repo discussion, "different batch sizes can lead to slight differences in the output, and SD3 has this issue, which is why the batch size for training is ensured to be the same as that used for data collection" (surfaced via search of [the flow_grpo issue tracker](https://github.com/yifan123/flow_grpo/issues); I did not open the originating issue, so treat the exact wording as second-hand).
- **Evaluation-metric reproduction gaps on released checkpoints** (not training variance, but relevant to how tight these numbers are): users evaluating the *official released* Flow-GRPO checkpoints found UnifiedReward 0.2–0.3 lower than the paper, reproducibly. Table posted: paper 0.92 / 5.32 / 4.06 / 0.95 / 22.44 / 3.42 vs. reproduction 0.9082 / 5.3349 / 4.0475 / −1.6331 / 22.4371 / 3.1636 (OCR / aesthetic / DeQA / ImageReward / PickScore / UnifiedReward). The ImageReward gap was later traced to a code bug; the UnifiedReward gap was never resolved. — [flow_grpo issue #39, 2025-07-16 onward](https://github.com/yifan123/flow_grpo/issues/39)
- Still unresolved eight months later: "I reproduced the SD3.5M evaluation with 1000 prompts… Aes = 5.39 (5.39), DeQA = 4.08 (4.07), ImgRwd = 0.84 (0.87), Pickscore = 22.34 (22.34), UniRwd = 3.04 (3.33). Most metrics are consistent with the paper, but ImageReward is slightly lower and **UnifiedReward is noticeably lower than reported.**" — [ustc-yaojinchen, flow_grpo issue #39, 2026-03-23](https://github.com/yifan123/flow_grpo/issues/39)
- Author's own framing of what counts as reproduction — relative, not absolute: "I believe **the most important aspect is the relative change of the model after GRPO compared to before**, which, as your results show, is almost consistent with the paper." — [yifan123, flow_grpo issue #39, 2025-07-21](https://github.com/yifan123/flow_grpo/issues/39)
- **Reward hacking at fixed config on a different base model** (a config-transfer failure, documented in Chinese): migrating GRPO-GUARD to FLUX-dev for OCR, "所有参数均严格参照论文及官方代码设置，未做任何修改" (all parameters strictly per the paper and official code, unmodified), yet "随着优化步数增加，出现了栅格效应或图像质量严重退化的问题" (as optimization steps increase, grid artifacts or severe image-quality degradation appear), while "在SD3.5-M上复现实验时，未出现栅格效应及图像质量退化问题" (no such problem when reproducing on SD3.5-M). Notably: "评估曲线（eval）与奖励曲线（reward）均呈正常上升趋势，但通过视觉核验发现，图像质量从240个优化步数开始出现明显退化" — eval and reward curves both rose normally while image quality visibly collapsed from step 240. — [flow_grpo issue #204, 2025-12-12](https://github.com/yifan123/flow_grpo/issues/204)

### Inferences
- The reproduction record is consistent with a method whose outcome is strongly path-dependent on the optimizer/sharding stack, the batch and group sizes, and the base model — i.e. exactly the regime in which run-to-run variance is large. The *existence* of a still-open "reward is stuck flat on the official config" issue is strong evidence that the published OCR number is not a tight, repeatable point estimate.
- Flow-GRPO's headline OCR gain for SD3.5-M is 0.59 → 0.92 ([Flow-GRPO abstract](https://arxiv.org/abs/2505.05470)). A two-seed FLUX spread of 0.558 vs 0.651 (Δ = 0.093) is ≈28% of that entire headline 0.33 improvement, which would be large enough to flip many of the smaller ablation-table deltas in this literature if those deltas were also single-seed.

### Gaps
- I could not read OpenReview review/rebuttal text. `openreview.net/forum?id=…` served a browser-verification wall to WebFetch, and `api2.openreview.net/notes/search` returned HTTP 400 for my query form. So I have **no verified reviewer quotes** about seeds or variance for DDPO (ICLR 2024), Flow-GRPO (NeurIPS 2025), DRaFT or AlignProp. A search snippet claimed Flow-GRPO reviewers raised single-seed concerns, but I could not confirm it against the primary thread and I am therefore **not** reporting it as a finding. This is a real hole that would be worth filling with a browser session.
- I found no independent third-party blog post or technical report doing a clean multi-seed reproduction of any of these methods.

## Any quantitative between-seed standard deviation for a diffusion RL method?

### Takeaway
**I found none.** Not a single number. I could not locate any published between-training-seed standard deviation, variance, or confidence interval for any text-to-image or text-to-video diffusion-RL method on any fixed benchmark (OCR, GenEval, PickScore, HPS, ImageReward, aesthetic). This is the single most striking gap in the literature, and it is itself the finding.

### Cited findings
- The nearest published ± quantities in the T2I diffusion-RL canon are all over evaluation samples or raters, not runs: DDPO's "mean and one standard error over 50 samples" and DPOK's "standard deviation averaged over eight independent human raters." — [DDPO, arXiv:2305.13301](https://arxiv.org/html/2305.13301); [DPOK, arXiv:2305.16381](https://arxiv.org/html/2305.16381)
- The nearest multi-seed ± in the *adjacent* diffusion-RL-for-control literature is 5 seeds with standard error (DIVO, as catalogued in a survey) — but that is offline RL on D4RL-style benchmarks, not image generation. — [arXiv:2610.21539](https://arxiv.org/html/2610.21539)

### Inferences
- With only two runs, the sample standard deviation implied by 0.558 and 0.651 is |Δ|/√2 ≈ 0.066 OCR points (mean 0.6045). A two-sample std has ~1 degree of freedom, so the 95% interval on that std is extremely wide (roughly a factor of 0.45× to 16× by the χ²₁ distribution) — the right conclusion from two seeds is "σ is plausibly on the order of 0.03–0.15," not a precise figure. This arithmetic is mine, derived from the two numbers supplied; it is not from any source.
- Because the field publishes no comparison value, two seeds differing by 0.093 cannot be labelled anomalous *or* normal against community practice. It can only be compared against the magnitude of the effects the field reports, where it is large (see inference above).

### Gaps
- No quantitative between-seed std exists in the sources I could reach. If the report needs one, it will have to be generated, not cited.

## Evaluation noise versus training noise

### Takeaway
The literature has effectively *only* worked on the evaluation-noise side and has no published decomposition. Flow-GRPO's protocol — 200 DrawBench prompts × 5 repeats = 1000 samples, and a 1018-prompt OCR/GenEval test set — is designed to shrink evaluation noise to the point where it is small; nothing analogous exists for training noise.

### Cited findings
- Explicit evaluation-noise control: "To reduce randomness, we repeated each of the 200 prompts 5 times and reported the average reward over all 1000 samples." — [yifan123, flow_grpo issue #39, 2025-07-21](https://github.com/yifan123/flow_grpo/issues/39)
- The OCR/GenEval test set is 1018 prompts, per the released config's own comment. — [flow_grpo issue #227](https://github.com/yifan123/flow_grpo/issues/227)
- Training-side stochasticity is large enough in this setting that it can flip the *qualitative* outcome, not just shift a number: identical official hyperparameters gave a flat reward curve on 8×H200 ([issue #227](https://github.com/yifan123/flow_grpo/issues/227)) and a non-increasing PickScore reward that only recovered after removing DeepSpeed ([issue #172](https://github.com/yifan123/flow_grpo/issues/172)).
- Training-side instability is also acknowledged in-paper, indirectly: Flow-GRPO reports that "smaller group sizes produce inaccurate advantage estimates, increasing variance and leading to training collapse" (per search snippet of the paper's ablation discussion; I confirmed the paper contains group-size ablations but did not isolate this exact sentence, so treat the wording as approximate). — [Flow-GRPO, arXiv:2505.05470](https://arxiv.org/html/2505.05470)
- A worked example that eval-curve noise is *not* the dominant problem — the eval curve can look clean while the model is degrading: "评估曲线（eval）与奖励曲线（reward）均呈正常上升趋势，但…图像质量从240个优化步数开始出现明显退化" (both eval and reward curves rose normally, but image quality clearly degraded from step 240). — [flow_grpo issue #204, 2025-12-12](https://github.com/yifan123/flow_grpo/issues/204)

### Inferences
- With ~1000 evaluation samples and a binary-ish per-sample OCR score near p ≈ 0.6, the binomial standard error of a single evaluation is about √(0.6·0.4/1000) ≈ 0.0155, and on 1018 prompts sampled once it is similar. That is roughly a factor of 4 smaller than the implied between-seed σ ≈ 0.066 above, i.e. **evaluation sampling noise cannot plausibly account for a 0.093 gap** on an eval of that size; the residual must come from the training trajectory (or from checkpoint-selection noise). This is my calculation from the stated eval-set size, not a published decomposition.
- The community's practice of *fixing* the sampling seed for comparisons (DDPO, DPOK, DRaFT, SRPO qualitative figures; DanceGRPO's fixed initial noise as regularizer) further conflates the two noise sources: holding the sampling seed fixed removes evaluation noise from the *picture* while leaving training noise entirely unmeasured.

### Gaps
- No paper or report I found separates training-trajectory variance from evaluation-sampling variance for a diffusion RL method. There is no published variance decomposition to cite.

## Papers on reproducibility / statistical rigour specific to generative-model RL

### Takeaway
**I found no "Deep RL that Matters" analogue for generative-model / text-to-image RL.** There is no paper I could locate that studies seed counts, statistical power, or reproducibility specifically for diffusion-model RL. This appears to be a genuine absence in the literature, not a search failure on my part — but see Gaps.

### Cited findings
- The nearest thing in the generative-RL space is in LLM RLVR rather than diffusion: *A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility* ([arXiv:2504.07086](https://arxiv.org/pdf/2504.07086)), which audits RL-for-reasoning results for seed/eval sensitivity. I surfaced it in search but did not read its full text, so I cannot quote its seed-count recommendation; flagging it as the closest known analogue for the writer to chase.
- Within diffusion RL proper, the published surveys do not raise seed counts as a methodological issue: the [Alignment and Safety of Diffusion Models via RL and Reward Modeling survey, arXiv:2505.17352](https://arxiv.org/pdf/2505.17352) and the [RL-driven Diffusion Models survey, arXiv:2610.21539](https://arxiv.org/html/2610.21539) — the latter's only seed-count mention in the portion I inspected is the DIVO offline-RL row (5 seeds), with no variance-reporting recommendation for the T2I branch.

### Gaps
- I cannot prove a negative. My conclusion "no diffusion-RL reproducibility paper exists" is based on four distinct search phrasings plus keyword checks of two surveys, not an exhaustive sweep. If one exists it is obscure.
- I did not read *A Sober Look at Progress in Language Model Reasoning* and so cannot report its recommendations; it is the single highest-value follow-up.

## What the classic deep-RL variance literature recommends (and how it applies)

### Takeaway
Classic deep RL settled this question a decade ago with concrete guidance: Henderson et al. ran **5 seeds** per configuration and reported mean ± standard error, and showed that two *disjoint sets of 5 seeds* of the *same* algorithm and hyperparameters can produce statistically distinguishable distributions. Agarwal et al. showed that 3–10 runs is the prevalent-but-inadequate regime, that 50–100 runs would be needed for defensible median comparisons on Atari 100k, and that stratified-bootstrap percentile CIs work down to N=10. **Two seeds is below the floor of every recommendation in this literature.**

### Cited findings — Henderson et al., *Deep Reinforcement Learning that Matters* (arXiv:1709.06560, submitted 19 Sep 2017, v3 30 Jan 2019)
- Their own protocol: "To ensure fairness we **run five experiment trials for each evaluation, each with a different preset random seed** (all experiments use the same set of random seeds)." — [arXiv:1709.06560](https://ar5iv.labs.arxiv.org/html/1709.06560)
- All results tables are "Final average ± standard error across 5 trials of returns across the last 100 trajectories after 2M training samples." — [arXiv:1709.06560](https://ar5iv.labs.arxiv.org/html/1709.06560)
- The central warning, which is exactly the situation of two divergent seeds: "[one] can draw from the same distribution at all, just by averaging different runs with the same hyperparameters, but different random seeds" — and on prescribing a number: "**While there can be no specific number of trials specified as a recommendation**, it is possible that power analysis methods can be used to give a general idea to this extent as we will discuss later. However, more investigation is needed to answer this open problem." — [arXiv:1709.06560](https://ar5iv.labs.arxiv.org/html/1709.06560)
- Their prescribed procedure when intervals are wide: bootstrap CIs plus bootstrap power analysis. "In cases where confidence bounds are exceedingly large, **it may be necessary to run more trials (i.e. increase the sample size)**. … Another method to determine if the sample size must be increased is bootstrap power analysis… If we use our sample and give it some uniform lift (for example, scaling uniformly by 1.25), we can run many bootstrap simulations and determine what percentage of the simulations result in statistically significant [differences]." — [arXiv:1709.06560](https://ar5iv.labs.arxiv.org/html/1709.06560)
- They observed the expected pattern that instability inflates the needed N: "in more unstable settings, the bootstrap power percentage leans towards insignificant results in the lift experiment. Conversely, in stable trials (e.g. TRPO on Hopper-v1) with a small sample size, the lift experiment shows that no more trials are needed." — [arXiv:1709.06560](https://ar5iv.labs.arxiv.org/html/1709.06560)

### Cited findings — Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (arXiv:2108.13264v4, 5 Jan 2022; NeurIPS 2021)
- On prevailing practice: "Beginning with DQN on the ALE, **5 or less runs are common in the field**." — [arXiv:2108.13264v4](https://arxiv.org/html/2108.13264v4)
- "While evaluating more runs per task has been prescribed to reduce uncertainty and obtain reliable estimates, **3-10 runs are prevalent in deep RL** as it is often computationally prohibitive to evaluate more runs. For example, 5 runs each on 50+ Atari 2600 games in ALE using standard protocol requires more than 1000 GPU training days." — [arXiv:2108.13264v4](https://arxiv.org/html/2108.13264v4)
- On how many runs median comparisons actually need: "…we find that this number is closer to **50–100 runs in Atari 100k** – far too many to be computationally feasible for most research projects." — [arXiv:2108.13264v4](https://arxiv.org/html/2108.13264v4)
- Their practical floor: "We find that **percentile CIs provide good interval estimates for as few as N = 10 runs** for both median and IQM scores." — [arXiv:2108.13264v4](https://arxiv.org/html/2108.13264v4)
- The core prescription: "Since any performance estimate based on a finite number of runs is **a random variable**, we argue that it should be treated as such. Specifically, we argue for reporting aggregate performance measures using **interval estimates via stratified bootstrap confidence intervals, as opposed to point estimates**." — [arXiv:2108.13264v4](https://arxiv.org/html/2108.13264v4)
- Their critique of current tables reads as a direct description of diffusion-RL tables: "Current Evaluation Protocol: Point estimates • Ignore statistical uncertainty • Hinder results reproducibility … Tables with mean scores per task • Overwhelming beyond a few tasks • **Standard deviations often omitted** • Incomplete picture for multimodal and heavy-tailed distributions." Their recommendations: stratified bootstrap CIs, performance profiles, and interquartile mean (IQM). — [arXiv:2108.13264v4](https://arxiv.org/html/2108.13264v4)
- They explicitly reject fixed-seed reporting as an escape hatch: fixed seeds do not "address the question of whether an algorithm would perform well under similar conditions but with **different seeds**. Furthermore, fixed seeds might benefit certain algorithms more than others." — [arXiv:2108.13264v4](https://arxiv.org/html/2108.13264v4)
- They advise against dichotomous significance testing: "we emphasize using statistical thinking but **avoid statistical significance tests** (e.g., p-value < 0.05) because of their dichotomous nature… such as 1) lack of statistically significant results does not demonstrate the absence of effect." — [arXiv:2108.13264v4](https://arxiv.org/html/2108.13264v4)

### Cited findings — Colas et al., *How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments* (arXiv:1806.08295, 21 Jun 2018, v2 5 Jul 2018)
- From the abstract: "we explain how the number of random seeds relates to the probabilities of statistical errors. For both the t-test and the bootstrap confidence interval test, we recall **theoretical guidelines to determine the number of random seeds one should use** to provide a statistically significant [comparison]." — [arXiv:1806.08295](https://arxiv.org/abs/1806.08295)
- Agarwal et al. cite this paper as the canonical reference for the seed-count question. — [arXiv:2108.13264v4 reference 20](https://arxiv.org/html/2108.13264v4)

### Inferences
- The gap is stark and worth stating plainly in the report: the *classic* deep-RL literature's **minimum** acceptable practice (5 seeds with intervals, Henderson 2017) and its **recommended** practice (N≥10 with stratified bootstrap CIs and IQM, Agarwal 2021) both sit above what *any* diffusion-RL paper I checked does, which is 1 run with no interval. Diffusion RL has not adopted the conventions that classic RL adopted after its own reproducibility reckoning.
- Applying Henderson's framing directly to the case at hand: two seeds of an identical FLUX config terminating at 0.558 and 0.651 is the diffusion-RL instance of their central result — that same-hyperparameter, different-seed runs can land in distributions you would mistake for different algorithms. Under their own standard, two seeds is not enough to characterise a config at all; under Agarwal's, the honest output is an interval, and a two-run interval is uninformatively wide.
- Conversely, nothing in community *practice* makes a 0.093 two-seed gap exceptional, because the community has no practice of looking. The correct framing is: the gap is unremarkable relative to the field's (absent) measurement, and alarming relative to the field's reported effect sizes.

### Gaps
- I could not retrieve Colas et al.'s full text (PDF download timed out), so I do not have its specific tabulated seed counts for given effect sizes (the paper is known to tabulate required N against standardised effect size, but I will not quote numbers I did not verify). Fetching arXiv:1806.08295's PDF or ar5iv HTML is a quick follow-up that would yield concrete N-vs-effect-size guidance.
