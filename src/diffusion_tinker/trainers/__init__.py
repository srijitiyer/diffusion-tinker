_LAZY_IMPORTS = {
    "DDRLConfig": "diffusion_tinker.trainers.ddrl_config",
    "DDRLTrainer": "diffusion_tinker.trainers.ddrl_trainer",
    "FlowGRPOConfig": "diffusion_tinker.trainers.flowgrpo_config",
    "FlowGRPOTrainer": "diffusion_tinker.trainers.flowgrpo_trainer",
    "DiffusionDPOConfig": "diffusion_tinker.trainers.diffusion_dpo_config",
    "DiffusionDPOTrainer": "diffusion_tinker.trainers.diffusion_dpo_trainer",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'diffusion_tinker.trainers' has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
