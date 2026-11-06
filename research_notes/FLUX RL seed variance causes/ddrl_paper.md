# DDRL (arXiv:2512.04332) — what it establishes about base models, numbers, recipe, and variance

**Paper**: "Data-regularized Reinforcement Learning for Diffusion Models at Scale", Haotian Ye, Kaiwen Zheng, Jiashu Xu, Puheng Li, Huayu Chen, Jiaqi Han, Sheng Liu, Qinsheng Zhang, Hanzi Mao, Zekun Hao, Prithvijit Chattopadhyay, Dinghao Yang, Liang Feng, Maosheng Liao, Junjie Bai, Ming-Yu Liu, James Zou, Stefano Ermon. Stanford / NVIDIA / Tsinghua. cs.LG. ([abs](https://arxiv.org/abs/2512.04332))

**Retrievability**: fully retrievable. I read the complete LaTeXML HTML of **v1 (3 Dec 2025)**, **v2 (6 Dec 2025)** and **v3 (24 Dec 2025)**. A full textual diff shows the three versions are **substantively identical** — the only changes are the footnote link text ("Code available soon" → "Website" → "Code here"), author-block line breaks, and the arXiv page chrome. **No table numbers, hyperparameters or model names changed between versions.** All section/table references below are to v2/v3 (identical numbering). ([v1](https://arxiv.org/html/2512.04332v1), [v2](https://arxiv.org/html/2512.04332v2), [v3](https://arxiv.org/html/2512.04332v3))

Secondary primary-ish sources: official code at [nvidia-cosmos/cosmos-rl `examples/ddrl.md`](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md) (released 24 Dec 2025) and the [NVIDIA project page](https://research.nvidia.com/labs/cosmos-lab/ddrl/).

---

## Q1: Which base models does DDRL evaluate on? Does it run FLUX.1-dev?

### Takeaway
**DDRL never runs FLUX.1-dev — the string "FLUX"/"Flux"/"flux" does not appear anywhere in any of the three arXiv versions.** Its models are NVIDIA **Cosmos2.5 2B and 14B** (video, the paper's main experiments) and **Stable Diffusion 3.5-Medium at 512x512** (a single text-to-image experiment, §4.4 / Table 4). SD3 (non-3.5) is also never used.

### Cited Findings
- Video base models: "For video generation, we use Cosmos2.5 [3, 2] 2B and 14B as the base models due to their capability in generating physically reasonable scenarios." — §4.1 Models ([v2](https://arxiv.org/html/2512.04332v2))
- The video checkpoints are not actually the public Cosmos2.5 release: "this model is different from the Cosmos2.5 models released in [3], which is the checkpoint after they post-train the merged checkpoint with DDRL" — i.e. the initialization is an internal pre-trained → six-category fine-tuned → merged checkpoint. §B.1 Model (appendix). ([v2](https://arxiv.org/html/2512.04332v2))
- Both video models are "dense transformer-based diffusion models", in the latent space of the **Wan-2.1 VAE**, 93 frames / 24 latent frames at 16 FPS, resolution within (1280, 720)–(720, 1280). §B.1 (appendix).
- Text encoder for video is **Cosmos-Reason1**. §B.1 (appendix).
- The only image model: "we apply DDRL to text-to-image (T2I) tasks based on SD3.5-Medium [11], with settings aligned with FlowGRPO." — §4.4 (main paper); appendix adds the resolution: "based on 512x512 resolution SD3.5-Medium [11]" — §B.3 (appendix).
- Searching all three HTML versions for FLUX returns **zero hits**; "SD3" appears 3 times (all SD3.5-Medium). Verified by direct text search of the downloaded HTML.
- The official code release only ships Cosmos-Predict2.5 configs (`cosmos-predict2-5-2b-720-reason-embedding-ddrl.toml`, `...-14b-...toml`) and Cosmos-Predict2.5-2B/14B post-trained checkpoints. No FLUX or SD support is exposed. — [examples/ddrl.md](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md)
- The project page independently lists only "Cosmos2.5 (2B and 14B)" and "Stable Diffusion 3.5-medium". — [NVIDIA project page](https://research.nvidia.com/labs/cosmos-lab/ddrl/)

### Inferences
- The SD3.5-Medium experiment is explicitly framed as a *secondary* demonstration — it lives in §4.4 "DDRL with synthetic data", whose purpose is to show DDRL works when no real anchor data exists, not as a headline image benchmark. The headline claims ("over a million GPU hours", "ten thousand double-blind human evaluations") are all video/Cosmos.
- Any claim that DDRL validates a method on FLUX.1-dev, or that it compares FLUX.1-dev against SD3/3.5, is unsupported by this paper.

### Gaps
- The paper offers **no discussion whatsoever** of FLUX, of architectural differences between FLUX-family and SD3.5-family flow models, or of whether DDRL transfers across image backbones. There is exactly one image backbone, one resolution, and one reward task.

---

## Q2: Main results tables, reproduced numerically

### Takeaway
There are two results tables. **Table 4 (main paper, §4.4)** is the SD3.5-Medium OCR table the question refers to; it has four metric columns — OCR, ClipScore, PickScore, ImageReward — and **no Aesthetic column** (Aesthetic appears in this paper only as one of the five VBench *video* sub-scores). On OCR, DanceGRPO (0.846) ≈ FlowGRPO (0.845) > DDRL (0.823) > base (0.566); on all three quality metrics DDRL ties or beats the base model while both baselines degrade.

### Cited Findings — Table 4 (main paper, §4.4): SD3.5-Medium, OCR reward
Caption: "Human voting results, OCR reward, and three OOD rewards after post-training SD3.5-Medium with different methods. We similarly mark the OCR reward if it is achieved by sacrificing human preference, e.g., generations are over-simplified or cartoonish compared with the base model." ([v2 §4.4](https://arxiv.org/html/2512.04332v2))

| Method | Δ-Vote ↑ (%) | OCR ↑ | ClipScore ↑ | PickScore ↑ | ImageReward ↑ |
|---|---|---|---|---|---|
| w/o RL | – | 0.566 | 0.321 | 0.865 | 1.13 |
| DanceGRPO | −53.0 | **0.846** (marked as hacked) | 0.309 | 0.837 | 0.80 |
| FlowGRPO | −22.1 | 0.845 (marked as hacked) | 0.313 | 0.855 | 1.03 |
| DDRL | 0 (reference) | 0.823 | 0.320 | **0.865** | **1.14** |

Who wins what (reading the table directly):
- **OCR (the optimised reward)**: DanceGRPO 0.846 > FlowGRPO 0.845 > DDRL 0.823 > base 0.566. DDRL is **third**, 0.023 below DanceGRPO. The paper flags both baselines' OCR numbers as hacking.
- **ClipScore**: base 0.321 > DDRL 0.320 > FlowGRPO 0.313 > DanceGRPO 0.309. DDRL does **not** beat the base model; it is 0.001 below it.
- **PickScore**: DDRL 0.865 = base 0.865 > FlowGRPO 0.855 > DanceGRPO 0.837.
- **ImageReward**: DDRL 1.14 > base 1.13 > FlowGRPO 1.03 > DanceGRPO 0.80.
- **Δ-Vote** (DDRL as reference, so DDRL = 0 by construction): DanceGRPO −53.0, FlowGRPO −22.1; the base model's Δ-Vote is **not reported** ("–") for the image setting.
- Text claim attached: "DDRL exhibits both improved OCR accuracy and over 20% human preference due to its fidelity to the base model's style and realism, even when we purely use synthetic data for regularization. In contrast, baseline methods tend to simplify images by using large, centralized captions." §4.4.
- Important selection caveat from the appendix: "The models are trained from around 300 iterations, and **we choose DDRL and baseline checkpoints with similar final OCR accuracy for evaluation.**" §B.3 (appendix). So Table 4 is an OCR-matched comparison, not an equal-compute comparison.

### Cited Findings — Table 1 (main paper, §4.2): video, Cosmos2.5, the paper's headline table
Caption: "DDRL aligns model with human preference better without severe reward hacking… Δ-Vote denotes the difference in human preference from pairwise video comparisons, calculated as the win percentage of the corresponding method minus the win percentage of DDRL. We mark a reward if its generated videos are less preferred by humans but the reward value is higher in this setting, which generally indicates reward hacking."

Cosmos2.5-**2B**:

| Method | VideoAlign T2V Δ-Vote | VideoAlign T2V Reward | VideoAlign I2V Δ-Vote | VideoAlign I2V Reward | VBench T2V Δ-Vote | VBench T2V Reward | VBench I2V Δ-Vote | VBench I2V Reward |
|---|---|---|---|---|---|---|---|---|
| w/o RL | −22.9 | 0.408 | −4.2 | 0.079 | −8.9 | 0.830 | −15.7 | 0.810 |
| DanceGRPO | −10.5 | 0.715 | −6.4 | 0.254 | −6.9 | 0.849 | −8.0 | 0.822 |
| FlowGRPO | −6.7 | 0.408 | −13.0 | 0.069 | −4.2 | 0.830 | −9.6 | 0.807 |
| DDRL | 0 | 0.604 | 0 | 0.177 | 0 | 0.842 | 0 | 0.819 |

Cosmos2.5-**14B**:

| Method | VideoAlign T2V Δ-Vote | VideoAlign T2V Reward | VideoAlign I2V Δ-Vote | VideoAlign I2V Reward | VBench T2V Δ-Vote | VBench T2V Reward | VBench I2V Δ-Vote | VBench I2V Reward |
|---|---|---|---|---|---|---|---|---|
| w/o RL | −8.4 | 0.359 | −12.5 | 0.058 | −4.5 | 0.820 | −4.5 | 0.813 |
| DanceGRPO | −4.9 | 0.494 | −5.9 | 0.160 | −3.1 | 0.834 | −4.2 | 0.818 |
| FlowGRPO | −6.7 | 0.476 | −11.6 | 0.128 | −7.3 | 0.829 | −5.4 | 0.808 |
| DDRL | 0 | 0.555 | 0 | 0.134 | 0 | 0.835 | 0 | 0.819 |

- Every Δ-Vote is negative, i.e. DDRL wins every pairwise human comparison in every one of the 8 settings. DanceGRPO beats DDRL on the raw VideoAlign reward in 3 of 4 settings (2B T2V 0.715 vs 0.604; 2B I2V 0.254 vs 0.177; 14B I2V 0.160 vs 0.134) and on VBench T2V for 2B (0.849 vs 0.842) — all flagged as hacking.
- FlowGRPO essentially fails to move rewards at 2B: VideoAlign T2V 0.408 = base 0.408; VBench T2V 0.830 = base 0.830; I2V reward 0.069 < base 0.079.

### Cited Findings — Table 3 (main paper): iteration + timestep-schedule ablation (VideoAlign, 2B)
| 𝒯 schedule | Iteration | NFEs per c | T2V | I2V |
|---|---|---|---|---|
| DDRL | 128 | 2N\|𝒯\| | 0.604 | 0.177 |
| DDRL | 256 | 2N\|𝒯\| | 0.627 | 0.180 |
| Reduced | 256 | N(1+\|𝒯\|) | 0.636 | 0.143 |

- "DDRL can continuously improve rewards without hacking even when training iterations are doubled. Additional human voting confirms that this extended training does not degrade user preference." §4.2 Ablations. (The confirming vote numbers are **not** given.)
- "computing the diffusion loss only once per step (by randomly sampling t) is as effective as optimizing the loss at every timestep" — though note I2V drops 0.180 → 0.143 in the "Reduced" row, which the text does not address.

### Cited Findings — Table 6 (appendix C): per-sub-score breakdown, video
Includes per-score means **and a STD column**. 2B T2V example: w/o RL text-alignment 1.69 / motion −0.45 / visual −0.01, mean 0.408, STD 0.520; DanceGRPO 1.50 / 0.16 / 0.49, mean 0.715, STD 0.683; FlowGRPO 1.67 / −0.43 / −0.02, mean 0.408, STD 0.556; DDRL 1.76 / −0.19 / 0.24, mean 0.604, STD 0.594. VBench 2B T2V: base 0.94/0.95/0.99/0.54/0.73 → 0.830 (STD 0.027); DanceGRPO 0.97/0.97/0.99/0.57/0.75 → 0.849 (0.022); FlowGRPO 0.95/0.95/0.99/0.54/0.72 → 0.830 (0.028); DDRL 0.96/0.96/0.99/0.56/0.73 → 0.842 (0.024). Full 14B and I2V rows are in Table 6.
- **The STD column is the spread of reward scores across the 1044 test prompts, not across training runs or seeds.** It appears beside "Mean" for the 3 VideoAlign and 5 VBench sub-scores.
- Table 6 contains an obvious typo: the 2B / I2V / w/o RL VBench Mean is printed as **4.050** with STD 0.160 where every neighbouring value is ~0.81 (Table 1 reports 0.810 for that cell). Flagging as a data-entry error in the appendix.

### Inferences
- On the image task DDRL's story is explicitly *not* "we win the reward": it loses OCR to both baselines and wins on vote + quality metrics. The 0.023 OCR gap plus the OCR-matched-checkpoint selection means the image experiment is designed as a quality-at-fixed-reward comparison.
- ClipScore 0.320 vs base 0.321 and PickScore 0.865 vs base 0.865 are differences at or below plausible evaluation noise, but the paper provides nothing with which to judge that (see Q4).

### Gaps
- **No Aesthetic-score column exists in Table 4.** If a comparison expects DDRL to report an Aesthetic number on the OCR task (as FlowGRPO's own paper does), that number is simply not in this paper.
- Table 4 reports no GenEval, no other FlowGRPO task (counting, colour, position, etc.) — only OCR.
- The base model's image Δ-Vote is not reported, so "over 20% human preference" in §4.4 cannot be decomposed into DDRL-vs-base.
- No number of image human-voting comparisons is given (the ~9600 figure is for video only).

---

## Q3: Exact training recipe

### Takeaway
The video recipe is given in detail (Tables 2/5 + §4.1 + §B.1). The **image/SD3.5 recipe is given in one appendix paragraph (§B.3)** and is "largely follows FlowGRPO": 48 groups per epoch, group size 24, LoRA α=64 r=32, lr 1e-4, diffusion-loss weight 0.001, 10-step SDE rollout, 40-step ODE eval, ~300 iterations. Several items you asked about (prompt-set size, which denoising steps get gradient on images, image guidance scale during rollout) are **not stated**.

### Cited Findings — image / SD3.5-Medium (§4.4 main + §B.3 appendix)
- Model: SD3.5-Medium at **512x512**. §B.3.
- Reward: OCR text-rendering accuracy, "the same text accuracy calculation strategy as FlowGRPO based on edit distance"; chosen because it is "the most vulnerable to reward hacking among FlowGRPO's settings". §B.3.
- Anchor data: **purely synthetic**, sampled from the base model with "40 ODE steps and its default CFG scale of 4.5". §B.3. Visualized in Figure 8. "Despite they are not accurate in text rendering, they provide abundant information of background details to regularize the RL process."
- "adopting the same number of groups per epoch (**48**), group size (**24**), LoRA configuration (**α=64, r=32**), and its training and evaluation prompts [as FlowGRPO]". §B.3. (= 48 prompts x 24 samples = 1,152 rollouts per epoch.)
- Learning rate: "We decrease the learning rate from **3e-4 to 1e-4** as we find it more stable for DDRL." §B.3.
- "The relative weight of the diffusion loss is set to **0.001**." §B.3.
- "For synthetic image data, forward noising and diffusion loss computation are performed exactly on the corresponding sampling timesteps." §B.3.
- Sampler: "We employ a **10-step SDE sampler for FlowGRPO training and a 40-step ODE sampler for evaluation**." §B.3.
- Baseline construction: "The DanceGRPO baseline is implemented by modifying FlowGRPO to **remove the reverse KL regularization and use the same initial noise** for data collection of the same prompt." §B.3.
- Duration + checkpoint selection: "The models are trained from around **300 iterations**, and we choose DDRL and baseline checkpoints with similar final OCR accuracy for evaluation." §B.3.

### Cited Findings — video / Cosmos2.5 (§4.1 main + §B.1 appendix)
- Table 2 (main paper, "Hyperparameters used in different methods"): DanceGRPO T=20, |𝒯|=12, β=0, no p_ref; FlowGRPO T=10, |𝒯|=10, β∈[0.001,0.1], uses p_ref; **DDRL T=20, |𝒯|=10, β=0.01, no p_ref**.
- Which steps get gradient: "we find that optimizing half of the steps is sufficient, and DDRL simply optimizes once every two steps" (§3.3 Timesteps & Guidance). The released config confirms `train_on = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]` out of 20 sample steps. — [examples/ddrl.md](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md)
- Guidance during training: "DDRL does not adopt CFG, and drops the condition c when calculating the diffusion loss with **20% probability**." §3.3. Released config: `guidance: float = 0.0  # We do not enable CFG during DDRL`.
- Table 5 (appendix B.1): batch size (number of unique conditions c) = **16** for both 2B and 14B; rollout size **N = 8**; GPUs per data instance 2 (2B) / 8 (14B); total GPUs **256** (2B) / **1024** (14B); optimizer **AdamW (β1=0.9, β2=0.99)**; learning rate **1e-5 (2B) / 3e-6 (14B)**.
- Iterations: main results use **128 iterations** ("we train all methods only for 128 iterations and use the exponential moving average (EMA) model weights to generate test videos"); the ablation goes to 256; "We always post-train the model for within a thousand iterations, consuming **no more than 8,000 unique data points**." §4.2, §B.1.
- Gradient accumulation: "For all methods, we accumulate the gradient over all t∈𝒯 and conduct **one full parameter update at the end of each iteration**." §4.1.
- Data: 1:1 mixture of T2V and I2V for rollout, drawn from the high-quality SFT dataset (human dynamics, physics, robotics, autonomous driving, smart spaces); "The training dataset is i.i.d. to the evaluation PAI-Bench dataset." §4.1, §B.1.
- Evaluation: PAI-Bench, **1044 prompts** across five categories (human, industry, robotics, driving, others); videos generated with **35 denoising steps** and **guidance strength 7 (2B) / 3 (14B)**, with CFG, separately for T2V and I2V. §4.1, §B.1 Evaluation details.
- Infrastructure: FSDP2 + context parallelism; rewards computed asynchronously on a **512-GPU** remote reward server; total compute "approximately one million GPU hours" on H100s. §4.1, §B.2.
- SFT-RL integration (§4.3, §B.1): conventional arm = SFT the vanilla pretrained model for **20,000 iterations on 512 GPUs**, AdamW (β1=0.9, β2=0.999), lr **3e-5**, then DDRL; direct arm = DDRL straight from the pretrained checkpoint. Result: "the direct DDRL approach achieves comparably high rewards and low diffusion loss (**0.119 v.s. 0.121**) to the SFT-then-RL pipeline."
- Hyperparameter-search budget: "**Over 15% of the computation is spent on selecting the best hyperparameters for baseline methods**… This includes the number of training iterations and the coefficient β." §4.2.

### Inferences
- Total trajectories per video arm is **not stated** but follows from Table 5: 16 conditions x N=8 = 128 rollouts/iteration, so 128 iterations ≈ 16,384 sampled trajectories per arm (256 iterations ≈ 32,768). Marked as my arithmetic, not a paper-reported figure.
- Likewise for images: 48 groups x 24 = 1,152 rollouts per epoch, but the paper never states how many iterations make an epoch, so ~300 iterations cannot be converted to a trajectory count.

### Gaps
- **Image rollout guidance scale is not stated.** §3.3 says DDRL does not use CFG; §B.3 says settings are "aligned with FlowGRPO" and quotes CFG 4.5 only for *generating the synthetic anchor data*. Whether the DDRL/baseline image rollouts used CFG 4.5 is left ambiguous.
- **Which image denoising steps receive gradient is not stated.** The "every other step, |𝒯|=10 of T=20" rule is given for video (T=20); the image rollout is a 10-step SDE sampler, and no |𝒯| for images is reported.
- **The size of the FlowGRPO OCR training/eval prompt sets is not stated** — only "its training and evaluation prompts".
- No epoch count for the image runs, no batch size in images beyond groups x group size, no optimizer/betas/weight-decay/warmup for the image runs, no LoRA target modules.
- No per-arm GPU-hour breakdown for the image experiment.

---

## Q4: Seeds, error bars, variance

### Takeaway
**The paper reports single runs throughout. There is no mention of multiple random seeds per configuration, no error bars, no confidence intervals, no standard deviations across runs, and no repeated-run variance of any kind.** The only "STD" in the paper is the spread of reward scores across test prompts (Table 6) and the within-rollout-group std used to normalise GRPO advantages. "Seed" appears exactly twice in the whole paper, neither time about experimental replication.

### Cited Findings
- Exhaustive search of the full text for seed/variance vocabulary yields only these uses:
  1. **"Existing algorithms anchor p_θ to p_ref by either leveraging the reverse KL divergence (FlowGRPO) or by adopting heuristics such as a fixed initial random seed (DanceGRPO)."** §3.1 — a description of DanceGRPO's *method*, not of experimental replication.
  2. "Seedance 1.0" — a reference title ([12]).
  - "std" in "A(x_0^n, c) ≜ (r(x_0^n,c) − mean)/std, where mean, std are within the group" — §2 Optimization, the GRPO advantage normaliser.
  - "We compute Z using the average reward within the rollout group, and additionally divide the standard deviation." §3.3 — again the advantage normaliser.
  - "a steep increase in rewards or a sharp decrease in standard deviation are also potential signals [of hacking]" §4.2 — proposed hacking *detector*, over generated samples.
  - Table 6 (appendix C) "STD" columns — per-prompt spread of the reward over the 1044 PAI-Bench test samples.
  - "each t≥2 term is a KL between Gaussians with the same covariance" — theory, Appendix A.
- The image appendix explicitly describes the baseline as the one that uses the *same* initial noise: "use the same initial noise for data collection of the same prompt" §B.3 — again a method detail (DanceGRPO's variance-reduction heuristic), not a seed-replication protocol.
- Where the paper does invest in statistical reliability, it is in **human voting, not in seeds**: "we conduct large-scale double-blind human voting across fifteen researchers… In total, we gather ~**9600** comparison data as the ground-truth" (§4.1); "In total, **15 people** participated in the voting, where **half of the voters are authors of this paper and the other half did not know this paper at all**" (§B.1 Evaluation details). No inter-rater agreement statistic, no vote confidence interval, and no sample size per pairwise cell are reported.
- Abstract-level framing: "With over a million GPU hours of experiments and ten thousand double-blind human evaluations…" — the claimed evidential scale is GPU hours and human votes, not repeated runs. ([abs](https://arxiv.org/abs/2512.04332))
- The released code exposes a seed knob (`use_same_seed: bool = False  # Whether to use same initial seed within a rollout group`) which is about intra-group rollout noise, not run-level replication. — [examples/ddrl.md](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md)

### Inferences
- Every number in Tables 1, 3, 4 and 6 should be read as a **single training run per (method, model, reward) cell**. With 8 video cells x 4 methods at 256–1024 GPUs each plus ~1M GPU hours already consumed, multi-seed replication was almost certainly out of budget — but the paper never says this, so it is an inference about why, not a reported limitation.
- Consequently the paper provides no basis for judging whether e.g. DDRL's ClipScore 0.320 vs base 0.321, or DDRL's VBench I2V 0.819 vs 0.818, are distinguishable from run-to-run noise. Differences of this size in Table 1/Table 4 are reported without any uncertainty quantification.

### Gaps
- **No statement anywhere acknowledging single-seed reporting or seed sensitivity as a limitation.** The paper has no dedicated Limitations section; §5 is "Related Works & Conclusions".
- No information on whether the human-voting Δ-Vote percentages carry sampling error (e.g. how many of the ~9600 comparisons fall in each of the 8 video cells x 3 baseline comparisons).

---

## Q5: data_beta / anchor-strength hyperparameter — is there a sweep?

### Takeaway
**No sweep of DDRL's own anchor strength is reported, and no sensitivity analysis of it exists in the paper.** The sweep that *is* reported (β ∈ [0.001, 0.1]) was run for *FlowGRPO's* reverse-KL coefficient; DDRL simply inherited the winning value. The anchor strength is set to β=0.01 for video and a diffusion-loss weight of 0.001 for images, with no justification for the latter and no ablation of either.

### Cited Findings
- "we also carefully sweep the regularization parameter **β ∈ [0.001, 0.1] for FlowGRPO, and select β = 0.01**, consistent with their paper. While this value does not prevent reward hacking, it mitigates the most severe artifacts, despite stifling legitimate reward growth. **For DDRL, we simply use the same β as well.**" §4.2 (main paper).
- Table 2 (main paper): DDRL β = **0.01** (single value, no range), versus FlowGRPO's swept range [0.001, 0.1].
- Image setting: "The relative weight of the diffusion loss is set to **0.001**." §B.3 (appendix) — stated flatly, with no sweep, no justification, and no note that it differs 10x from the video β.
- In the paper's formulation β is the RL temperature in the advantage A^n = (r^n − mean)/(β·std), which equivalently sets the diffusion-loss weight relative to the reward term (Algorithm 1, line 6, Eq. 9).
- The released code names this exact knob: "`data_beta: float = 0.01  # Coefficient of diffusion loss`", distinct from "`kl_beta: float = 0.0  # Coefficient of reverse KL`" (DDRL sets reverse KL to zero). Default matches the paper's video β=0.01. — [examples/ddrl.md](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md)
- The paper's two ablations (Table 3) vary **training iterations (128 vs 256)** and the **timestep schedule for the diffusion loss (full 𝒯 vs "Reduced" / one random t)**. Neither varies β / data_beta.
- The only β-variation experiment in the paper is on the *baseline*: "We empirically demonstrate our analysis by post-training with a significantly large coefficient **β = 0.1**. As shown in Fig. 3, while the KL divergence stabilizes since the middle of the training, the generated videos exhibit noise textures after longer training" §3.1 — i.e. large β is shown to fail *for reverse-KL regularisation*, not for DDRL's data anchor.

### Inferences
- Because β was tuned for FlowGRPO and then copied, DDRL's reported numbers are not at a DDRL-tuned anchor strength. The paper presents this as a fairness/simplicity argument ("we simply use the same β"), but it also means the paper establishes nothing about how DDRL behaves as the anchor strength varies.
- The 10x discrepancy between the video anchor weight (0.01) and the image anchor weight (0.001) is unexplained and is the closest thing to evidence that the anchor strength needs per-setting tuning — but the paper draws no such conclusion.

### Gaps
- **No DDRL β/data_beta sweep, no sensitivity curve, no reported failure mode at too-small or too-large anchor weight.** The question "how sensitive is DDRL to anchor strength?" is unanswered by the paper.
- No explanation for the video (0.01) vs image (0.001) difference.

---

## Q6: Reward hacking — does the unregularised baseline degrade on quality while gaining on reward?

### Takeaway
Yes, and this is the paper's central empirical claim, demonstrated in both modalities. On SD3.5-Medium (Table 4) DanceGRPO gains +0.280 OCR over the base model while losing on all three unoptimised quality metrics and losing 53 points of human preference to DDRL. On video (Fig. 4 / Table 6) DanceGRPO raises the *average* VideoAlign reward while the text-alignment sub-score **falls 16% (T2V) and 28% (I2V)**.

### Cited Findings
- Image, explicit reward-vs-quality trade: DanceGRPO OCR 0.566 → 0.846 (+0.280) while ClipScore 0.321 → 0.309, PickScore 0.865 → 0.837, ImageReward 1.13 → 0.80, Δ-Vote −53.0. FlowGRPO OCR → 0.845 with ClipScore 0.313, PickScore 0.855, ImageReward 1.03, Δ-Vote −22.1. Table 4.
- Both baselines' OCR values are typographically **marked** in Table 4 as hacking: "We similarly mark the OCR reward if it is achieved by sacrificing human preference, e.g., generations are over-simplified or cartoonish compared with the base model." Table 4 caption.
- Qualitative mechanism on images: "baseline methods tend to **simplify images by using large, centralized captions**" (§4.4); "existing methods hack the text rendering rewards by generating **over-simplified images that lose original realism**" (§3.1); Figure 6 caption: "With similarly high OCR accuracy, DDRL additionally maintains fidelity to the base model in terms of style and realism."
- Video, per-sub-score hacking: "**DanceGRPO shows a 16% (T2V) and 28% (I2V) decrease in the TA [text alignment] score, despite its average reward is the highest.** This aligns with the human voting result where its generated videos are not preferred. DDRL is the only algorithm that brings 'Pareto improvement'." Figure 4 caption. Table 6 numbers: 2B T2V text alignment 1.69 (base) → 1.50 (DanceGRPO) → 1.76 (DDRL); 2B I2V 1.57 → 1.46 → 1.61.
- "DanceGRPO achieves its high average reward at the cost of sacrificing one score, with text alignment increase being negative after post-training. This trade-off is a signal of reward hacking, indicating that one property is being compromised to exploit the reward. DDRL, by regularizing directly with the data-anchored diffusion loss, achieves a **Pareto improvement across all scores**." §4.2 Reward hacking analysis.
- Qualitative mechanism on video: "DanceGRPO optimizes rewards by generating **over-colorized, over-stylized videos that are unrealistic, ignoring the prompt requirements**. On the contrary, FlowGRPO generates **blurry, jittering videos that are visually unacceptable**." §3.1 (Fig. 1).
- The KL-is-not-enough result: "All methods are trained from the same base model using the same dataset for **256 iterations**. While DanceGRPO and FlowGRPO achieve higher rewards, humans consistently prefer videos generated by the base model." Figure 2 caption.
- Diagnostic heuristics offered (not validated): "if the **diffusion loss of a post-trained checkpoint increases by more than 10%**, its outputs are mostly hacked; a steep increase in rewards or a sharp decrease in standard deviation are also potential signals." §4.2. Immediately preceded by: "A reliable way to automatically detect reward hacking is highly beneficial. **Although we have yet to figure it out**, several metrics could be indicative."
- Causal thesis: "we identify **on-policy regularization as the primary cause of reward hacking** in diffusion RL" (§1); "the on-policy sampling, imperfect reference, and non-verifiable rewards jointly make the existing regularization incapable of supporting effective post-training algorithms for diffusion models" (§3.1).
- On the non-verifiability of vision rewards: "most rewards are non-verifiable, i.e., they are an approximation of the underlying true rewards… **are only accurate in a neighborhood of the real data manifold**. In other words, a higher reward does not necessarily imply a better sample." §2.

### Inferences
- The paper's framing is that reward-metric gains on an unregularised arm should be treated as *prima facie* suspect unless accompanied by held-out quality metrics or human votes. The ClipScore/PickScore/ImageReward columns in Table 4 are explicitly labelled "three **OOD** rewards" — i.e. metrics not being optimised, used as hacking detectors.

### Gaps
- The 10%-diffusion-loss-increase detector is presented as a hunch, not measured: no table of diffusion loss vs hacking verdict is given.
- No statistical test accompanies any of the degradation claims (see Q4).

---

## Q7: Training instability, divergence, and model-dependence

### Takeaway
The paper reports several stability interventions — reduced learning rates, dropped loss weighting, EMA evaluation, early stopping at 128 iterations — and reports that the method's measured advantage **shrinks at 14B because of the lr reduction**, which is the paper's only explicit model-scale dependence. It reports no divergence or NaN events, and no cross-architecture (e.g. FLUX-vs-SD) dependence of any kind.

### Cited Findings
- Image lr reduction for DDRL specifically: "We decrease the learning rate from 3e-4 to 1e-4 as **we find it more stable for DDRL**." §B.3 (appendix).
- 14B lr reduction for everyone: "Lastly, for **14B post-training, this is insufficient to prevent hacking, so we further reduce the learning rate to 3e-6 for all methods**." §4.2.
- Explicit scale-dependence of the result: "the advantage in **14B experiments is smaller due to the use of a smaller learning rate to mitigate baselines' hacking behaviors**, and the performance gap will enlarge as we continue post-training." §4.2. (The claim that the gap would enlarge is an assertion; no experiment supporting it is reported.)
- Early stopping + EMA as hacking mitigation: "both DanceGRPO and FlowGRPO (regardless of β ranging across two magnitudes) hack severely and become worse than the base model after training for 256 iterations. To reduce the phenomenon, **we train all methods only for 128 iterations and use the exponential moving average (EMA) model weights** to generate test videos." §4.2.
- Objective simplifications made for stability: "similar to standard diffusion training, **we omit the coefficient w_t for the sake of stability**. For reward maximization, we omit the monotonic transformation λ(·) such that the training reward scales are consistent and comparable between DDRL and baselines." §3.3. (The project page corroborates: "The practical implementation omits the monotonic transformation λ for training stability" — [project page](https://research.nvidia.com/labs/cosmos-lab/ddrl/).)
- A distinct instability of the *baseline* regularizer: even with β=0.1, "while the KL divergence stabilizes since the middle of the training, the generated videos exhibit **noise textures** after longer training, implying that controlling the reverse KL divergence is insufficient to regularize the policy." §3.1, Fig. 3.
- Hyperparameter fragility of baselines: "Notice that the phenomenon persists even when we carefully tune their hyperparameters such as regularization weight β, training iterations, or learning rate." §3.1. And >15% of total compute went to baseline hyperparameter selection (§4.2).
- Efficiency/memory claim: "DDRL needs **neither a reference model nor an old model** during training, thus saving memory by a large margin"; "the number of network evaluations (NFEs) per data of DDRL is similar to DanceGRPO (w/o KL divergence), and **a half compared to FlowGRPO**." §3.3.
- Conclusion's self-characterisation: "an empirically simple, **stable**, and highly scalable algorithm" §5.
- The paper's own residual caveat about generality: the SFT-RL integration result is hedged — "this result demonstrates a strong potential to simplify and unify the diffusion post-training paradigm, and **we encourage future research to examine this possibility more closely**." §4.3.

### Inferences
- The combination of (a) lr reduced for stability on images, (b) lr reduced again at 14B, (c) EMA + 128-iteration early stop, and (d) OCR-matched checkpoint selection means the headline comparisons are run in a deliberately conservative regime. That regime is chosen to keep the *baselines* presentable, which the paper states, but it also means DDRL is never shown at an aggressive learning rate.
- The paper's only model-dependence evidence is 2B vs 14B within one architecture family. It gives no evidence bearing on whether DDRL's behaviour transfers across different flow-matching backbones.

### Gaps
- **No reported divergence, loss spikes, NaNs, or collapsed runs** for any method — the paper does not discuss failure cases at that level.
- No discussion of sensitivity to the 20% condition-dropout rate, to the EMA decay, or to the choice of 128 iterations beyond "it mitigates baseline hacking".
- No cross-architecture robustness study. One video family (Cosmos2.5, 2 sizes) + one image model (SD3.5-Medium, one resolution, one reward) is the entire empirical surface.

---

## Cross-cutting notes for the report writer

- **The single most load-bearing fact for a FLUX-vs-SD3.5 question: DDRL contains no FLUX.1-dev experiment.** Its image evidence is one SD3.5-Medium / 512x512 / OCR-reward run per method. Any argument that DDRL speaks to FLUX.1-dev behaviour is extrapolation beyond the paper.
- **Versions**: v1 (3 Dec 2025), v2 (6 Dec 2025), v3 (24 Dec 2025) are substantively identical; a full text diff found only link-text and page-chrome changes. There is no updated or extended experimental section in later versions.
- **Main paper vs appendix split**: Tables 1, 2, 3, 4 and Figures 1–6 are main paper. Appendix A is the proof of Theorem 3.1; Appendix B.1/B.2/B.3 are video recipe, reward-server infrastructure, and the *entire* image recipe respectively (Table 5 = video hyperparameters, Figure 8 = synthetic anchor images); Appendix C is Table 6, the per-sub-score video breakdown. **Everything specific about the SD3.5 training setup lives in a single appendix paragraph (§B.3).**
- **Code**: released 24 Dec 2025 in nvidia-cosmos/cosmos-rl (`examples/ddrl.md`), Cosmos-Predict2.5 only; the anchor-strength knob is `data_beta` (default 0.01), reverse KL is `kl_beta` (set to 0.0 for DDRL), and `train_on = [0,2,...,18]` of 20 sample steps. No image/SD3.5 reproduction code is published there. — [examples/ddrl.md](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md)
- **Appendix typo to avoid propagating**: Table 6, Cosmos2.5-2B / I2V / w/o RL, VBench Mean is printed as 4.050 (STD 0.160); Table 1 gives 0.810 for the same cell.
