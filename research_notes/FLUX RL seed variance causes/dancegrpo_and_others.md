# DanceGRPO and Published RL / Policy-Gradient Post-Training on FLUX.1-dev

Scope note: all findings below concern **FLUX.1-dev** unless explicitly flagged otherwise. No source found in this sweep reports RL post-training on **FLUX.1-schnell** (the timestep-distilled variant); that is recorded as a gap.

---

## Q1. DanceGRPO: which models does it train, and what exactly does it report for FLUX.1-dev?

### Takeaway
DanceGRPO (arXiv:2505.07818, ByteDance Seed + HKU, submitted 12 May 2025, final v3 28 Aug 2025) trains four backbones — Stable Diffusion, FLUX, HunyuanVideo, SkyReels-I2V — across text-to-image, text-to-video and image-to-video with five reward models. For FLUX.1-dev it reports HPS-v2.1 going from 0.304 to 0.372 (HPS-only reward) and GenEval from 0.659 to 0.687 in the dual-reward setting, with a visible reward-alignment trade-off: optimizing HPS alone *drops* CLIP Score (0.405 → 0.376) and GenEval (0.659 → 0.561).

### Cited Findings
- Title/authors/date: "DanceGRPO: Unleashing GRPO on Visual Generation", Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, Ping Luo. Submitted 12 May 2025; final version 28 Aug 2025 — [arXiv:2505.07818](https://arxiv.org/abs/2505.07818)
- Coverage claim: "one unified RL algorithm across two generative paradigms (diffusion models and rectified flows), three tasks (text-to-image, text-to-video, image-to-video), four foundation models (Stable Diffusion, HunyuanVideo, FLUX, SkyReel-I2V), and five reward models" — [DanceGRPO project page / paper](https://dancegrpo.github.io/)
- Headline: outperforms baselines "by up to 181%" on HPS-v2.1, CLIP Score, VideoAlign and GenEval — [arXiv:2505.07818](https://arxiv.org/abs/2505.07818). Note: the 181% figure is a cross-model maximum and is **not** a FLUX number.
- FLUX variant used is explicitly FLUX.1-dev, described as "a flow-based text-to-image generation model that advances the state-of-the-art across multiple benchmarks, leveraging a more complex architecture than Stable Diffusion" — [arXiv:2505.07818v1 HTML](https://arxiv.org/html/2505.07818v1)
- FLUX results (Table 3) — [arXiv:2505.07818v1 HTML](https://arxiv.org/html/2505.07818v1):

  | Setting | HPS-v2.1 | CLIP Score | Pick-a-Pic | GenEval |
  |---|---|---|---|---|
  | FLUX baseline | 0.304 | 0.405 | 0.224 | 0.659 |
  | FLUX + HPS-v2.1 reward | 0.372 | 0.376 | 0.230 | 0.561 |
  | FLUX + HPS-v2.1 & CLIP | 0.343 | 0.427 | 0.228 | 0.687 |

- Reward models used across the paper: five, covering "image/video aesthetics, text-image alignment, video motion quality, and binary feedback" — [arXiv:2505.07818](https://arxiv.org/abs/2505.07818)
- Baseline comparison table (reported as DDPO / ReFL / DPO vs. DanceGRPO) gives: baseline HPS-v2.1 0.239 → Ours 0.365, DDPO 0.297, ReFL 0.357, DPO 0.241; CLIP Score baseline 0.363 → Ours 0.421, DDPO 0.381, ReFL 0.418, DPO 0.367 — [arXiv:2505.07818v3 HTML](https://arxiv.org/html/2505.07818v3)

### Inferences
- The 0.239 baseline in the DDPO/ReFL/DPO comparison table is almost certainly **Stable Diffusion v1.4**, not FLUX, because the paper's own FLUX baseline is 0.304 in Table 3. One extraction of the v3 HTML described this table as being "on FLUX", which conflicts with the Table 3 numbers. **Treat the DDPO/ReFL/DPO head-to-head as an SD-family result until the PDF table caption is verified.** This is a conflict worth flagging in the report.
- FLUX + HPS-only is a clear reward-hacking signature: +0.068 HPS bought at the cost of −0.029 CLIP Score and −0.098 GenEval. The dual-reward run recovers alignment but gives up roughly half the HPS gain.

### Gaps
- Aesthetic Score and ImageReward are not reported for FLUX in DanceGRPO's own tables (other papers later report ImageReward for a DanceGRPO-on-FLUX reproduction; see Q4).
- The exact table caption / target model for the DDPO/ReFL/DPO comparison could not be confirmed from HTML extraction.

---

## Q2. DanceGRPO's FLUX hyperparameters

### Takeaway
DanceGRPO on FLUX.1-dev: 25 denoising steps, CFG **disabled** during sampling, timestep-selection ratio τ = 0.6 (i.e. ~60% of the 25 steps receive gradient, ~15 steps), injected noise level ε_t = 0.3, 32 prompts × 12 images per prompt per iteration (group size 12), 4 gradient updates per iteration, lr 1e-5 AdamW, grad-clip 1.0, clip range ε = 1e-4, >10,000 prompts, on 32× H800.

### Cited Findings
- Sampling steps: "we use 50 steps for Stable Diffusion, and 25 steps for FLUX and HunyuanVideo" — [arXiv:2505.07818v3 HTML](https://arxiv.org/html/2505.07818v3)
- Guidance: the paper states it is "disabling CFG during the sampling phase for models with high sample fidelity, such as HunyuanVideo and FLUX" — [arXiv:2505.07818v3 HTML](https://arxiv.org/html/2505.07818v3). No numeric CFG/guidance-scale value for FLUX was found in the HTML.
- Training table (Table 6): learning rate 1e-5; optimizer AdamW; gradient clip norm 1.0; prompts per iteration 32; images per prompt 12; gradient updates per iteration 4; clip range ε = 1e-4; noise level ε_t = 0.3; timestep selection τ = 0.6; hardware "32 H800 GPUs for flow-based text-to-image models" — [arXiv:2505.07818v1 HTML](https://arxiv.org/html/2505.07818v1)
- Prompt set: the authors "always use more than 10,000 prompts to optimize the models" — [arXiv:2505.07818v1 HTML](https://arxiv.org/html/2505.07818v1)
- Evaluation uses ODE samplers at 25 steps for FLUX (training rollouts use the SDE reformulation) — [arXiv:2505.07818v1 HTML](https://arxiv.org/html/2505.07818v1)
- Noise-level sensitivity: "Reducing ε_t leads to a significant performance degradation" and "noise level larger than 0.3 sometimes leads to noisy images after RLHF training" — [arXiv:2505.07818v1 HTML](https://arxiv.org/html/2505.07818v1)
- Independent reproduction of DanceGRPO-on-FLUX by the MixGRPO authors reports DanceGRPO's rollout NFE = 25 and gradient NFE = 14 per iteration, and 291.284 s/iteration — [MixGRPO, arXiv:2507.21802v2](https://arxiv.org/html/2507.21802v2). The gradient NFE of 14 is consistent with τ = 0.6 over 25 steps.
- MixGRPO's own FLUX config, for contrast: 25 sampling steps, SDE window w = 4, **only 4 timesteps optimized**, group size N = 12, lr 1e-5, AdamW wd 1e-4, grad-accum 3, batch size 1, 32 GPUs, max 300 iterations, bf16/fp32 mixed precision — [arXiv:2507.21802v2](https://arxiv.org/html/2507.21802v2)

### Inferences
- Trajectories per iteration = 32 prompts × 12 images = 384. At the ~300-iteration scale that reproductions use for FLUX, that is on the order of 10^5 sampled trajectories — a useful anchor for estimating what "one run" costs and therefore why nobody runs seeds.
- "CFG disabled" plus "25 steps" means FLUX rollouts are cheap relative to SD's 50-step CFG rollouts, so FLUX's effective per-trajectory signal comes from far fewer forward passes — one mechanical reason gradient estimates on FLUX are noisier per iteration.

### Gaps
- Total trajectory count / total training iterations for FLUX is **not stated** in DanceGRPO itself; the 300-iteration figure comes from MixGRPO's and BranchGRPO's reproductions, not from the DanceGRPO paper.
- No numeric guidance scale for FLUX (only "CFG disabled").
- The precise SDE↔ODE conversion constants for FLUX were not recoverable from the HTML.

---

## Q3. Does DanceGRPO report seeds, error bars, or variance? Does it discuss instability on FLUX?

### Takeaway
No. DanceGRPO reports single-run point estimates with no seeds, no error bars and no standard deviations anywhere, and its instability discussion is about **video** (HunyuanVideo/SkyReels) and about **DDPO on rectified-flow SDEs**, not about FLUX. Third parties later report that DanceGRPO's own τ = 0.6 FLUX configuration is the unstable one.

### Cited Findings
- No random seeds, error bars, standard deviations or repeated runs were found in the paper for FLUX or any other model — [arXiv:2505.07818v1 HTML](https://arxiv.org/html/2505.07818v1); [arXiv:2505.07818v3 HTML](https://arxiv.org/html/2505.07818v3)
- Instability statements that *are* present, verbatim: "inability to maintain stable optimization when scaling to large and diverse prompt sets"; "training exclusively on the conditional objective leads to divergent optimization trajectories"; "**DDPO always diverges when applied to rectified flow SDEs**"; "reducing gradient oscillation"; "assigning different noise vectors to samples with the same prompts always leads to reward hacking phenomena in video generation, including training instability" — [arXiv:2505.07818v3 HTML](https://arxiv.org/html/2505.07818v3)
- Shared initialization noise within a prompt group is the paper's named stability device: "Shared initialization noise is used for samples originating from the same textual prompt, preventing reward hacking phenomena that can destabilize video generation training" — [DanceGRPO project page](https://dancegrpo.github.io/)
- Secondary summary of the paper's own stability story: group-wise normalization "stabilizes training... by ensuring nonzero variance in group statistics"; "subsampling timesteps reduces compute and variance" — [EmergentMind topic page on DanceGRPO](https://www.emergentmind.com/topics/dancegrpo) (aggregator, not primary)
- **Third-party contradiction of the stability claim, on FLUX specifically**: BranchGRPO reports "DanceGRPO with reduced timestep fractions (tf=0.6) suffers from large reward fluctuations and unstable convergence", visualized in its Figure 1, while the full-timestep (tf=1.0) variant "converges more smoothly but at high cost" — [BranchGRPO, arXiv:2509.06040v1](https://arxiv.org/html/2509.06040v1)
- A further third-party characterization: "DanceGRPO exhibits pronounced instability, whereas Branch-GRPO performs branching at intermediate steps and propagates dense rewards backward for more efficient and stable optimization" — [BranchGRPO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/233d16f17f809981763db2f01b7f9603-Paper-Conference.pdf)
- A search-surfaced claim that in DanceGRPO's FLUX experiments "the text-video alignment dimension demonstrated significant instability, prompting its exclusion from final analysis" — surfaced in search over [BranchGRPO](https://arxiv.org/pdf/2509.06040) / [DanceGRPO](https://arxiv.org/pdf/2505.07818). **Flagged as low confidence**: "text-video alignment" is a video metric and makes little sense for FLUX; I could not confirm this sentence in either primary PDF.
- The DanceGRPO GitHub repo's issue tracker contains issues about reward instability and divergence for Qwen, HunyuanVideo and Wan 2.1, but **no FLUX-specific training-instability issue** was found; the only FLUX-related issue is a feature request, #91 "support flux kontext" (akk-123, 17 Oct 2025) — [github.com/XueZeyue/DanceGRPO/issues](https://github.com/XueZeyue/DanceGRPO/issues)

### Inferences
- The τ = 0.6 setting that DanceGRPO publishes as its FLUX default is precisely the setting a later paper identifies as unstable. That is the single sharpest "author claim vs. third-party reproduction" conflict in this evidence base, and it is specific to FLUX.
- DanceGRPO's stability argument rests on *within-group* noise sharing, which controls variance *across samples in a group* but says nothing about variance *across runs*. A method can therefore show smooth reward curves and still be seed-fragile — exactly the gap the seed-variance question is about.

### Gaps
- No DanceGRPO reward curve for FLUX with an uncertainty band could be located.
- Whether DanceGRPO's FLUX runs were ever repeated at all is unknowable from the paper.

---

## Q4. Other methods applied to FLUX.1-dev, their FLUX numbers, and their stability statements

### Takeaway
At least nine published methods train or evaluate on FLUX.1-dev: DanceGRPO, SRPO, MixGRPO (+MixGRPO-Flash), BranchGRPO, TempFlow-GRPO, GRPO-Guard, λ-Controlled GRPO, Flow-GRPO (via third-party ports), and reward-backprop methods (ReFL / DRaFT / AlignProp / DRTune / RSA-FT). Almost every one of them is motivated by an instability or over-optimization failure of the method before it, and essentially none of them report seeds or error bars.

### Cited Findings — SRPO (Tencent Hunyuan)
- "Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference", Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, Yansong Tang; 8 Sep 2025; trains FLUX.1-dev — [arXiv:2509.06942v1](https://arxiv.org/html/2509.06942v1)
- Table 1 numbers: FLUX baseline Aesthetic 5.867 / PickScore 22.671 / ImageReward 1.115 / HPSv2.1 0.289 / human realism 8.2%; SRPO Aesthetic 6.194 / PickScore 23.040 / ImageReward 1.118 / HPSv2.1 0.289 / human realism 38.9% (Aesthetic-quality excellent rate 40.5%) — [arXiv:2509.06942v1](https://arxiv.org/html/2509.06942v1)
- **DanceGRPO on FLUX scored 5.3% on SRPO's human realism eval** (vs. FLUX baseline 8.2%), i.e. worse than the untuned model on that axis; SRPO criticizes DanceGRPO for "undesirable artifacts, for example, excessive glossiness and pronounced edge highlights" — [arXiv:2509.06942v1](https://arxiv.org/html/2509.06942v1)
- Hyperparameters: 10 minutes of training on 32× H20; 25 sampling steps during training, 50 at inference; **CFG scale 3.5**; HPDv2 prompts; claims "75× greater training efficiency than DanceGRPO" — [arXiv:2509.06942v1](https://arxiv.org/html/2509.06942v1)
- Named failure mode: methods that "restrict optimization to only a few diffusion steps" are "more susceptible to reward hacking"; reward-model biases are specific and exploitable — "HPSv2 develops a preference for reddish tones, PickScore for purple images and ImageReward for overexposed regions" — [arXiv:2509.06942v1](https://arxiv.org/html/2509.06942v1)
- SRPO's FLUX checkpoint shipped publicly as a 12B flow transformer fine-tune — [Tencent-Hunyuan/SRPO on GitHub](https://github.com/Tencent-Hunyuan/SRPO); mirrored as a hosted model — [layer.ai FLUX.1 SRPO [dev]](https://layer.ai/models/bfl-flux-1-dev-srpo)

### Cited Findings — MixGRPO (Peking University + Tencent Hunyuan)
- Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, Zhao Zhong; v2 dated 29 Sep 2025; base model FLUX.1-dev — [arXiv:2507.21802v2](https://arxiv.org/html/2507.21802v2)
- FLUX results: baseline HPS-v2.1 0.313 / PickScore 0.227 / ImageReward 1.088 / UnifiedReward 3.370; DanceGRPO 0.356 / 0.233 / 1.436 / 3.397 (291.3 s/iter); MixGRPO **0.367 / 0.237 / 1.629 / 3.418** (150.8 s/iter); MixGRPO-Flash 0.358 / 0.236 / 1.528 / 3.407 (112.4 s/iter) — [arXiv:2507.21802v2](https://arxiv.org/html/2507.21802v2)
- Method: SDE sampling inside a sliding time window, ODE outside it, window shifting along the denoising direction during training — [arXiv:2507.21802](https://arxiv.org/pdf/2507.21802)
- **No error bars, variance metrics or seed reporting**; no analysis of training instability, reward collapse or convergence behaviour — [arXiv:2507.21802v2](https://arxiv.org/html/2507.21802v2)

### Cited Findings — BranchGRPO (ICLR 2026)
- Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, Shanghang Zhang; 7 Sep 2025; base model FLUX.1-Dev — [arXiv:2509.06040v1](https://arxiv.org/html/2509.06040v1)
- Table 1 on FLUX: baseline 0.313 / 0.227 / 1.112; DanceGRPO (tf=1.0) 0.360 / 0.229 / 1.189; DanceGRPO (tf=0.6) 0.353 / 0.228 / 1.219; MixGRPO (20,5) 0.359 / 0.228 / 1.211; BranchGRPO 0.363 / 0.229 / 1.233; BranchGRPO-WidPru 0.364 / 0.230 / 1.300; BranchGRPO-DepPru **0.369 / 0.231 / 1.319** (HPS-v2.1 / PickScore / ImageReward) — [arXiv:2509.06040v1](https://arxiv.org/html/2509.06040v1)
- Hyperparameters: 300 optimizer steps, batch 2/GPU, 16× H20, AdamW lr 1e-5 wd 1e-4, bf16, grad-accum 12, tree depth 20, branching factor 2 (16 leaves) — [arXiv:2509.06040v1](https://arxiv.org/html/2509.06040v1)
- Motivation: "Existing methods still face training instability due to sparse rewards" — [arXiv:2509.06040](https://arxiv.org/pdf/2509.06040)
- **No error bars, confidence intervals, seeds or statistical testing** — [arXiv:2509.06040v1](https://arxiv.org/html/2509.06040v1)

### Cited Findings — TempFlow-GRPO (Tencent WeChat Vision + Zhejiang University, ICLR 2026)
- Xiaoxuan He, Siming Fu, Yuke Zhao et al.; v2 dated 21 Sep 2025. Primary experiments on **SD3.5-M**, with FLUX.1-dev as additional validation at 1024×1024 — [arXiv:2508.04324v2](https://arxiv.org/html/2508.04324v2)
- FLUX.1-dev result: matches Flow-GRPO's 300-step performance in 80 steps, a **3.75× convergence speedup**, with HPSv3 as reward — [arXiv:2508.04324v2](https://arxiv.org/html/2508.04324v2); [ICLR 2026 version](https://proceedings.iclr.cc/paper_files/paper/2026/file/d75f561eaaf2cb754bc8d7e36d8af362-Paper-Conference.pdf)
- SD3.5-M GenEval: Flow-GRPO 0.88 → TempFlow-GRPO 0.97 at 3800 steps; SD3.5-M at higher resolution shows a 4.5× speedup (≈100 vs 450 steps) — [arXiv:2508.04324v2](https://arxiv.org/html/2508.04324v2)
- Head-to-head vs DanceGRPO on HPDv2: TempFlow-GRPO 38.5 at 280 iterations vs DanceGRPO 37.2 at 300 iterations (+1.3%) — [arXiv:2508.04324v2](https://arxiv.org/html/2508.04324v2)
- Hyperparameters: 10 sampling steps in training / 50 at evaluation; group size 24 formed as **4 seeds × 6 branches**; 48 total groups; KL β = 0.001 (PickScore) and 0.004 (GenEval) — [arXiv:2508.04324v2](https://arxiv.org/html/2508.04324v2)
- Variance finding: "Reward variance is highest in initial steps", so "early-stage interventions are most impactful for exploration"; the paper attacks the "temporal uniformity assumption" where "sparse terminal rewards with uniform credit assignment fail to capture the varying criticality of decisions across generation timesteps" — [arXiv:2508.04324v2](https://arxiv.org/html/2508.04324v2)
- Related statement surfaced in search over the 2026 literature: "The standard deviation of reward varies dramatically between timesteps, reaching a peak during early structural decisions and approaching zero during final refinements" — [TempFlow-GRPO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/d75f561eaaf2cb754bc8d7e36d8af362-Paper-Conference.pdf)

### Cited Findings — GRPO-Guard (Kuaishou/KwaiVGI et al.)
- Jing Wang, Jiajun Liang, Jie Liu, Henglin Liu, Gongye Liu, Jun Zheng, Wanyuan Pang, Ao Ma, Zhenyu Xie, Xintao Wang, Meng Wang, Pengfei Wan, Xiaodan Liang; submitted 25 Oct 2025, revised 30 Oct 2025; evaluated on **SD3.5M and Flux.1-dev** — [arXiv:2510.22319](https://arxiv.org/abs/2510.22319)
- Abstract, verbatim and directly on point for FLUX instability: "we observe a systematic shift in the importance-ratio distribution—its mean falls below 1 and its **variance differs substantially across timesteps**. This left-shifted and inconsistent distribution prevents positive-advantage samples from entering the clipped region, causing the mechanism to fail in constraining overconfident positive updates. As a result, the policy model inevitably enters an implicit over-optimization stage—while the proxy reward continues to increase, essential metrics such as image quality and text-prompt alignment deteriorate sharply, ultimately making the learned policy impractical for real-world use." — [arXiv:2510.22319](https://arxiv.org/abs/2510.22319)
- Project page — [jingw193.github.io/GRPO-Guard](https://jingw193.github.io/GRPO-Guard/)

### Cited Findings — λ-Controlled GRPO (the only source with CIs and multi-seed)
- Yufeng Wang (Stony Brook), Parivesh Priye (Georgia Tech), Meeshawn Marathe (Rivian), Ramit Pahwa (Rivian); 21 Sep 2026; **SD3.5-Medium primary, FLUX.1-dev as secondary generalization test** — [arXiv:2609.22041v2](https://arxiv.org/html/2609.22041v2)
- Instability described: "policy update changes systematically across denoising steps, with importance ratios drifting below one, becoming increasingly dispersed, clipping at different rates, and leaving fewer usable samples late in training", governed by a per-step scalar "path variance" λ_k — [arXiv:2609.22041v2](https://arxiv.org/html/2609.22041v2)
- SD3.5 hard-OCR held-out gains over a RatioNorm baseline: +0.0199 OCR reward, +1.44 pp exact-match, +3.54 pp substring-match — [arXiv:2609.22041v2](https://arxiv.org/html/2609.22041v2)
- Tiny-SD3 audit: late-step λ̂₇ reduced 0.00542 → 0.0001; log-ratio drift −0.00269 → −0.000041 — [arXiv:2609.22041v2](https://arxiv.org/html/2609.22041v2)
- **Seed protocol**: Tiny-SD3 experiments average over **3 independent runs**; SD3.5 hard-OCR and PickScore are single-GPU pilots (not multi-seed); and "the broader **FLUX campaign runs single-seed** on 8 GPUs" — [arXiv:2609.22041v2](https://arxiv.org/html/2609.22041v2)
- FLUX results are explicitly weak/inconclusive: only "directional support for cross-backbone generalization"; on PickScore λ-Controlled GRPO **ties** untuned Flow-GRPO at −0.0001, 95% CI [−0.0013, +0.0011], while beating RatioNorm by +0.0046; hard-OCR improvements "unresolved" with overlapping confidence intervals; GenEval shows no significant difference across methods — [arXiv:2609.22041v2](https://arxiv.org/html/2609.22041v2)

### Cited Findings — Flow-GRPO and reward-backprop baselines
- Flow-GRPO (NeurIPS 2025, yifan123) is the reference SDE-reformulation GRPO for flow models; its headline experiments are on SD3.5-M, with FLUX usage mainly via third-party ports — [arXiv:2505.05470](https://arxiv.org/html/2505.05470); [github.com/yifan123/flow_grpo](https://github.com/yifan123/flow_grpo)
- Flow-GRPO's own group-size ablation: "When group size was reduced to G=12 and G=6, training became unstable and eventually collapsed, whereas G=24 remained stable. Smaller group sizes produce inaccurate advantage estimates, increasing variance and leading to training collapse." — [github.com/yifan123/flow_grpo](https://github.com/yifan123/flow_grpo) / [arXiv:2505.05470](https://arxiv.org/html/2505.05470). **Directly relevant**: DanceGRPO and MixGRPO both run FLUX at **G = 12**, i.e. inside Flow-GRPO's documented collapse regime.
- Flow-GRPO reports that on FLUX.1-dev "the reward curve rises steadily throughout training without noticeable reward hacking", and that omitting KL regularization "caused a collapse in visual diversity" — [github.com/yifan123/flow_grpo](https://github.com/yifan123/flow_grpo)
- Flow-GRPO on DDPO: "DDPO's reward increases more slowly than Flow-GRPO's and eventually collapses in later stages" — [github.com/yifan123/flow_grpo](https://github.com/yifan123/flow_grpo). Consistent with DanceGRPO's stronger claim that "DDPO always diverges when applied to rectified flow SDEs" — [arXiv:2505.07818v3](https://arxiv.org/html/2505.07818v3)
- ReFL / DRaFT-K / AlignProp have all been integrated and tested on Flux.1-dev, and ReFL, DRaFT and DanceGRPO have been compared on FLUX.1-dev across Aesthetic predictor 2.5, PickScore, ImageReward, GenEval and human assessment — [RSA-FT, arXiv:2603.21175](https://arxiv.org/html/2603.21175v1) / [CVPR 2026 version](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Reward_Sharpness-Aware_Fine-Tuning_for_Diffusion_Models_CVPR_2026_paper.pdf)
- RSA-FT reports consistent gains over DRTune and ReFL on Flux.1-dev, framed as showing the approach "remains effective on recent large-scale architectures" — [arXiv:2603.21175](https://arxiv.org/pdf/2603.21175)
- Other 2025–2026 flow-RL entries that appear in the same lineage and use FLUX or SD3.5: DenseGRPO ([arXiv:2601.20218](https://arxiv.org/pdf/2601.20218)), chunk-level policy optimization ([arXiv:2510.21583](https://arxiv.org/html/2510.21583v2)), V-GRPO ([arXiv:2604.23380](https://arxiv.org/pdf/2604.23380)), GDRO ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_GDRO_Group-level_Reward_Post-training_Suitable_for_Diffusion_Models_CVPR_2026_paper.pdf)), Bayesian Prior-Guided Optimization ([arXiv:2511.18919](https://arxiv.org/pdf/2511.18919)), Diffusion-SDPO ([arXiv:2511.03317](https://arxiv.org/pdf/2511.03317)). I did not read these in full; they are listed as leads, not as evidence.

### Inferences
- The **FLUX.1-dev baseline HPS-v2.1 is reported inconsistently across papers**: 0.304 (DanceGRPO), 0.313 (MixGRPO), 0.313 (BranchGRPO), 0.289 (SRPO, as HPSv2.1). Same model, same metric, spread of ~0.024 — larger than several papers' claimed improvements over each other (e.g. BranchGRPO-DepPru 0.369 vs MixGRPO 0.359). This is independent evidence that FLUX evaluation itself carries variance comparable to the effect sizes being published.
- SRPO's human eval scoring DanceGRPO-on-FLUX at 5.3% realism *below* the 8.2% untuned FLUX baseline, while DanceGRPO reports +0.068 HPS on the same model, is a concrete case of a FLUX RL run that improves the proxy reward and degrades the thing the reward was standing in for.
- G = 12 on FLUX (DanceGRPO, MixGRPO) sitting inside Flow-GRPO's documented G = 12 collapse regime is the most actionable mechanistic hypothesis in this evidence base for why FLUX RL runs would be seed-fragile.

### Gaps
- DPOK was not found applied to FLUX in any source in this sweep.
- Diffusion-DPO / Flow-DPO on FLUX: only DanceGRPO's "DPO" baseline column was found, and its target model is ambiguous (see Q1 inference). No dedicated Flow-DPO-on-FLUX paper was located.
- No FLUX.1-**schnell** RL result was found at all. Given schnell is timestep-distilled (few-step), the 25-step rollout / τ-fraction machinery in every method above does not transfer, but no source states this explicitly.

---

## Q5. Is there a recurring reported pattern that FLUX is harder, noisier, or less responsive to RL than SD3/SD3.5?

### Takeaway
There is a strong *implicit* pattern — papers that headline SD3.5-M results consistently relegate FLUX to a secondary, weaker, or inconclusive position, and FLUX is where the "instability" and "over-optimization" papers point — but I found **no paper that states in so many words "FLUX is harder than SD3.5 for RL."** The closest thing to a direct statement is λ-Controlled GRPO's FLUX results being explicitly inconclusive while its SD3.5 results are significant.

### Cited Findings
- λ-Controlled GRPO: SD3.5 is the primary backbone with significant held-out gains; FLUX.1-dev is "secondary generalization tests" giving only "directional support", with PickScore tied to untuned Flow-GRPO (−0.0001, 95% CI [−0.0013, +0.0011]), hard-OCR "unresolved" with overlapping CIs, and GenEval showing no significant difference — [arXiv:2609.22041v2](https://arxiv.org/html/2609.22041v2)
- TempFlow-GRPO: primary experiments on SD3.5-M; FLUX.1-dev only "additional validation", and the FLUX claim is a *speedup* (3.75× convergence) rather than a higher final score — [arXiv:2508.04324v2](https://arxiv.org/html/2508.04324v2)
- Flow-GRPO: headline on SD3.5-M; FLUX appears as a stability sanity check — [arXiv:2505.05470](https://arxiv.org/html/2505.05470)
- GRPO-Guard tests both SD3.5M and Flux.1-dev and locates the failure in a mechanism — importance-ratio drift below 1 with "variance [that] differs substantially across timesteps" — that is generic to flow matching rather than FLUX-specific — [arXiv:2510.22319](https://arxiv.org/abs/2510.22319)
- The August 2025 survey "Integrating Reinforcement Learning with Visual Generative Models" (Yuanzhi Liang, Yijie Fang, Rui Li, Ziqi Ni, Ruijie Su, Chi Zhang, Xuelong Li; Institute of AI, China Telecom; 14 Aug 2025) **does not** discuss training instability, variance, reward hacking or seed sensitivity, and makes **no** FLUX-vs-SD3/SD3.5 difficulty comparison; DanceGRPO is the only FLUX-capable method it names — [arXiv:2508.10316v1](https://arxiv.org/html/2508.10316v1)
- Other surveys in the space (not read in full, listed as leads): "Alignment and Safety of Diffusion Models via Reinforcement Learning and Reward Modeling" ([arXiv:2505.17352](https://arxiv.org/pdf/2505.17352)); "Alignment of Diffusion Models: Fundamentals, Challenges, and Future" ([arXiv:2409.07253](https://arxiv.org/pdf/2409.07253))

### Inferences
- The pattern is real but it is a pattern in *where authors put FLUX in their papers*, not in anything they assert. For a report this should be stated as a structural observation about the literature, with the λ-Controlled GRPO CIs as the one quantitative instance.
- A plausible mechanical reason the literature avoids FLUX as primary backbone: FLUX.1-dev is 12B and guidance-distilled, so CFG is disabled during rollout (DanceGRPO), which removes a variance-reduction/conditioning lever that SD3.5-M runs retain.

### Gaps
- I did not find OpenReview review text for DanceGRPO, TempFlow-GRPO or BranchGRPO. Searches returned only aggregator pages and unrelated meta-research on peer review. OpenReview-specific reviewer complaints about FLUX variance remain unverified.

---

## Q6. Do any papers report multiple seeds on FLUX and show the spread?

### Takeaway
No. I found **zero** papers reporting multi-seed FLUX.1-dev RL results with a spread. The one paper that runs multiple seeds at all runs them on a *toy* backbone (Tiny-SD3, 3 runs) and explicitly states its FLUX campaign is single-seed.

### Cited Findings
- λ-Controlled GRPO: "Tiny-SD3 experiments average over 3 independent runs; SD3.5 hard-OCR and PickScore use single-GPU pilot protocols (not multi-seed). The broader **FLUX campaign runs single-seed** on 8 GPUs." — [arXiv:2609.22041v2](https://arxiv.org/html/2609.22041v2)
- DanceGRPO: no seeds, error bars, standard deviations or repeated runs found — [arXiv:2505.07818v1](https://arxiv.org/html/2505.07818v1); [v3](https://arxiv.org/html/2505.07818v3)
- MixGRPO: "no error bars, variance metrics, or seed reporting" — [arXiv:2507.21802v2](https://arxiv.org/html/2507.21802v2)
- BranchGRPO: "no error bars, confidence intervals, seeds reported, or statistical testing" — [arXiv:2509.06040v1](https://arxiv.org/html/2509.06040v1)
- The only place "seeds" appears as a *design* quantity rather than a replication quantity: TempFlow-GRPO's group size of 24 is constructed as "4 seeds × 6 branches" — i.e. seeds are initial-noise draws within one run, not independent training runs — [arXiv:2508.04324v2](https://arxiv.org/html/2508.04324v2). DanceGRPO likewise uses seed *sharing* within a prompt group as a variance-control device — [dancegrpo.github.io](https://dancegrpo.github.io/)

### Inferences
- The field's use of the word "seed" in this literature almost always means "initial latent noise within a rollout group", not "independent training run". Any report on FLUX RL seed variance should make this terminological distinction explicitly, because it is the main reason the question looks answered in the papers when it is not.
- Given 32× H800 or 16–32× H20 for ~300 iterations per run, a 3–5 seed FLUX study is a 5-figure GPU-hour commitment, which plausibly explains the universal absence of replication rather than any claim that variance is low.

### Gaps
- No published FLUX.1-dev reward curve with a shaded seed band was located anywhere.

---

## Q7. Any reports of FLUX RL runs where some seeds improve the reward and others degrade it?

### Takeaway
I found **no report of seed-level bidirectional outcomes on FLUX** (some seeds up, some down). What I did find is the adjacent and well-documented phenomenon: FLUX RL runs where the proxy reward rises monotonically while real quality degrades, and FLUX runs that collapse as a function of *hyperparameters and numerical precision* rather than seed — including a dated third-party reproduction thread with specific fixes.

### Cited Findings
- **Third-party reproduction report, FLUX-dev, dated**: issue #204 "Over Optimization Issues of GRPO-GUARD on Flux-dev for Text Generation", opened by **CIawevy on 12 Dec 2025** on the Flow-GRPO repo. Reported: "grid effects or severe degradation of image quality emerged" as optimization steps increased, with quality declining noticeably **after 240 steps**; reward hacking where the model produced "large-sized, same-font text, while neglecting typography and aesthetic features"; and that raising `num_steps` from 6 to 28 and adjusting noise levels **failed to fix it** — [flow_grpo issue #204](https://github.com/yifan123/flow_grpo/issues/204)
- Repo owner **yifan123 (12 Dec 2025)** recommended "directly adopting the MSE of velocity as the KL loss" plus EMA to improve training stability — [flow_grpo issue #204](https://github.com/yifan123/flow_grpo/issues/204)
- GRPO-Guard author **jingw193 (22 Dec 2025)** diagnosed a numerical cause: "limited decimal precision of BF16 introduces rounding errors" causing importance-ratio deviation, fixed by moving training precision from **BF16 to FP32** — [flow_grpo issue #204](https://github.com/yifan123/flow_grpo/issues/204)
- **CIawevy (21 Dec 2025)** reported that successful FLUX training required reducing `beta` by 100× to **4e-4** and setting `config.train.cfg = False` — [flow_grpo issue #204](https://github.com/yifan123/flow_grpo/issues/204)
- Divergent-direction outcome at the *method* level on FLUX: SRPO's human eval puts DanceGRPO-on-FLUX at 5.3% realism vs. 8.2% for untuned FLUX — a reward-optimized FLUX that is worse than the starting point on the target attribute — [arXiv:2509.06942v1](https://arxiv.org/html/2509.06942v1)
- Internal-inconsistency outcome within DanceGRPO's own FLUX table: HPS-only optimization raises HPS-v2.1 0.304 → 0.372 while dropping GenEval 0.659 → 0.561 — [arXiv:2505.07818v1](https://arxiv.org/html/2505.07818v1)
- Variance-reduction ablations reported in the 2026 flow-RL literature: "Without variance reduction methods, the naive baseline suffers from severe training instability, and removing either group-shared timestep-noise pairs or stratified timestep sampling similarly destabilizes training"; and "A naive implementation within GRPO suffers from training instability due to high-variance gradients, where noise in surrogate magnitudes overwhelms the reward signal" — surfaced in search across [V-GRPO, arXiv:2604.23380](https://arxiv.org/pdf/2604.23380) and [TempFlow-GRPO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/d75f561eaaf2cb754bc8d7e36d8af362-Paper-Conference.pdf). **Attribution between these two papers is not firm** — I did not open V-GRPO's full text, so treat the exact source of each sentence as unconfirmed.
- The one statement I found about seed-dependent divergence is explicitly about *language models*, not FLUX: "Seed-dependent instability of language-model fine-tuning is well documented, with runs that diverge partway through training and vanishing gradients as one driver" — surfaced in search, likely from [λ-Controlled GRPO, arXiv:2609.22041](https://arxiv.org/pdf/2609.22041); do not attribute to FLUX.

### Inferences
- The BF16→FP32 finding in issue #204 is the most concrete candidate mechanism for apparent seed variance on FLUX: BF16 rounding error in the importance ratio is a *stochastic, run-dependent* perturbation that would present exactly as "some seeds improve, some degrade" while actually being a numerics problem. This dovetails with GRPO-Guard's published finding that the ratio distribution is left-shifted with timestep-dependent variance.
- Combining issue #204 (beta needed 100× reduction; cfg must be off; FP32 needed) with Flow-GRPO's G=12 collapse ablation and DanceGRPO's ε_t sensitivity ("larger than 0.3 sometimes leads to noisy images"), FLUX RL appears to sit near several cliff edges at once. A configuration sitting on a cliff edge is precisely the regime in which seed variance becomes bimodal, even if no paper has measured it.
- Because the degradation in issue #204 appears "after 240 steps" and the standard FLUX protocol is ~300 steps, published FLUX results are being read off right at the onset of over-optimization. Run-to-run jitter in *when* that onset occurs would flip a run from improvement to degradation at a fixed step count.

### Gaps
- No source reports a seed-indexed set of FLUX RL outcomes. The question "do some seeds improve and others degrade" is, on this evidence, **unanswered in the published literature** — and that absence is itself the finding.
- Hugging Face blog posts, model cards and forum/Discord discussions on RL fine-tuning FLUX were not productively surfaced by these searches beyond the SRPO checkpoint pages; community-level seed-variance anecdotes remain unexplored.
- OpenReview review text (which the assignment expected to raise variance/reproducibility concerns) could not be retrieved for any of these papers.
