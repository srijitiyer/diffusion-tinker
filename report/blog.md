# Bringing RL post-training to diffusion models

*Draft - Srijit Iyer*

Language models have a comfortable post-training story. You pretrain, then you reach for TRL or OpenRLHF or verl, write a reward, and run PPO or DPO with a few lines of code. The infrastructure is solved well enough that nobody thinks about it.

Diffusion models don't have that. If you want to RL fine-tune Stable Diffusion to make more aesthetic images, or render readable text, or follow a style, you end up pulling research code from four different papers, each with its own training loop, its own assumptions, and its own quirks. The gap between "this method exists in a paper" and "I can run it on my model" is large, and it's mostly plumbing.

diffusion-tinker closes that gap. It's a TRL-style library for RL post-training of diffusion models, built on HuggingFace `diffusers`. You give it a model, a reward, and a list of prompts; it runs one of six algorithms behind a single `Trainer`/`Config` API:

```python
from diffusion_tinker import FlowGRPOTrainer, FlowGRPOConfig

trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs="aesthetic",
    train_prompts=[...],
    config=FlowGRPOConfig(num_epochs=30),
)
trainer.train()
```

The point, the same one Thinking Machines makes about Tinker for LLMs, is that you shouldn't have to understand the SDE log-probability derivation or the LoRA-on-a-frozen-transformer dance to fine-tune a diffusion model. You pick an algorithm and a reward; the library handles the rest.

It supports six algorithms (FlowGRPO, DDRL, DRaFT, DiffusionDPO, DDPO, and SFT) on Stable Diffusion 3.5 and FLUX.1, with built-in rewards for aesthetics, CLIP score, OCR accuracy, and human preference, plus a clean interface for your own.

## The part I want to write about

The interesting part of this project wasn't writing the trainers. It was finding out, right before I wanted to put it forward for release, that they didn't actually work, and that the test suite was happily telling me they did.

The library had passing unit tests and clean smoke runs. Every trainer would load a model, generate an image, compute a reward, and finish without an error. By the usual bar, it was ready. So I went looking one level down: do the gradients actually reach the weights, and are the images the model produces in the range the rewards expect?

They were not.

**DRaFT had never trained.** DRaFT works by backpropagating a reward straight through the last few denoising steps. Its loop runs a no-grad warmup over the early steps, then a gradient-tracked loop over the last few, all inside one `torch.autocast` block. PyTorch's autocast caches the half-precision cast of each weight the first time it sees it. Because the first time it saw the LoRA weights was during the no-grad warmup, it cached *detached* copies, and the gradient loop quietly reused them. The result: `loss.backward()` ran, but every LoRA gradient came back `None`, and the optimizer stepped on nothing. The reward sat flat for thirty epochs because the model was a spectator. One flag, `cache_enabled=False`, fixes it. Afterward the aesthetic reward climbed from 5.9 to 9.6.

**Every image was about half black.** The SD3 and FLUX VAEs decode to pixels in $[-1, 1]$, and the standard thing to do is rescale to $[0, 1]$ before clamping. The code skipped the rescale and clamped directly, so every pixel that should have been in the lower half of the range was crushed to zero. A quick check at runtime: 55% of the pixels in every generated image were below zero. The same bug ran in reverse on the encode side, feeding $[0,1]$ images to a VAE that wanted $[-1,1]$, so three of the trainers were learning to denoise toward mis-scaled targets. Two one-line fixes, touching essentially every image the library produces.

There were more, smaller ones: a precision mismatch between sampling and replay that biased the PPO importance ratio to 0.97 when it should be 1.0; three trainers calling a PEFT method that doesn't exist on diffusers models; an off-by-one in a gradient accumulation normalizer; a missing stop-gradient. None of them showed up in a test that only asked whether the code ran.

That's the lesson I'd want someone to take from this. For RL on generative models, "it ran and produced a number" is almost no evidence. The number can be computed on a broken image by a model that isn't learning, and everything will look fine. What you actually have to check is the machinery underneath: that gradients reach the parameters you're training, that the tensors are in the ranges your losses assume, that an importance ratio between a policy and itself comes out to one. I ended up verifying each of those directly, and that's the only reason I trust the numbers now.

## Where it stands

On the fixed code, five of the six algorithms reproduce cleanly on SD3.5. FlowGRPO with the OCR reward is the headline: it reaches a perfect 1.00 evaluation accuracy on rendering readable text, above the 0.823 and 0.845 the original papers report, and above the library's own earlier 0.95 figure, which had been measured on the half-black images. DRaFT climbs smoothly on aesthetics, DiffusionDPO learns a preference within twenty steps, SFT and DDRL both improve and stay stable.

Two honest caveats. DDPO's mechanism is correct now, but PPO is sample-hungry and I can't show a reward gain at the batch size a single 24GB card allows, so I'm not claiming one. And FLUX runs end-to-end on its native model, but only modestly: FLUX.1-schnell is a distilled four-step model that was never built for the stochastic sampling these methods need, so it's a poor RL target. The non-distilled FLUX.1-dev is the obvious next run.

The code, the per-run numbers, and the full list of fixes are at [github.com/srijitiyer/diffusion-tinker](https://github.com/srijitiyer/diffusion-tinker).
