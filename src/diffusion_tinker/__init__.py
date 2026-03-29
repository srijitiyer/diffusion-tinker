from diffusion_tinker._version import __version__


def __getattr__(name):
    if name == "DDRLConfig":
        from diffusion_tinker.trainers.ddrl_config import DDRLConfig

        return DDRLConfig
    if name == "DDRLTrainer":
        from diffusion_tinker.trainers.ddrl_trainer import DDRLTrainer

        return DDRLTrainer
    raise AttributeError(f"module 'diffusion_tinker' has no attribute {name!r}")


__all__ = ["DDRLTrainer", "DDRLConfig", "__version__"]
