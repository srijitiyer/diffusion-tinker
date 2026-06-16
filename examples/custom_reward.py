"""Custom reward function with reward_kwargs and training callbacks."""

from diffusion_tinker import FlowGRPOTrainer, FlowGRPOConfig, TrainerCallback, RewardContext


def style_reward(ctx: RewardContext):
    """Score images against custom user-defined criteria.

    ctx.metadata contains everything passed via reward_kwargs.
    ctx.latents gives access to the final denoised latent tensors.
    ctx.epoch gives the current training step (for curriculum schedules).
    """
    target = ctx.metadata.get("target_keyword", "cat")
    base_score = ctx.metadata.get("base_score", 0.5)

    scores = []
    for prompt in ctx.prompts:
        score = 1.0 if target.lower() in prompt.lower() else base_score
        scores.append(score)

    return scores


class LoggingCallback(TrainerCallback):
    """Logs training progress at each hook."""

    def on_train_begin(self, trainer):
        print(f"Training started with {len(trainer.train_prompts)} prompts")

    def on_epoch_end(self, trainer, epoch, metrics):
        if epoch % 5 == 0:
            print(f"[callback] Epoch {epoch}: {metrics}")

    def on_evaluate(self, trainer, epoch, mean_reward):
        print(f"[callback] Eval at epoch {epoch}: reward={mean_reward:.3f}")

    def on_save(self, trainer, epoch, path):
        print(f"[callback] Checkpoint saved: {path}")


trainer = FlowGRPOTrainer(
    model="stabilityai/stable-diffusion-3.5-medium",
    reward_funcs=style_reward,
    reward_kwargs={
        "target_keyword": "cat",
        "base_score": 0.3,
    },
    callbacks=[LoggingCallback()],
    train_prompts=[
        "a photograph of a cat on a windowsill",
        "a painting of a dog in a garden",
        "a sketch of a cat playing with yarn",
        "a portrait of a bird on a branch",
    ],
    config=FlowGRPOConfig(num_epochs=20),
)
trainer.train()
