from .register_models import MODELS
from utils import Config
import torch.nn as nn


def get_generator(config: Config) -> nn.Module:
    g_cfg = config.G_model
    g_factory = MODELS.get(g_cfg.name)
    if g_factory is None:
        raise ValueError(f"Unknown Generator: {g_cfg.name}")
    print(f"Generator: {g_cfg.name}\nGenerator Parameters: {g_cfg.parameters}")
    return g_factory(**g_cfg.parameters)

def get_discriminator(config: Config) -> nn.Module:
    d_cfg = config.D_model
    d_factory = MODELS.get(d_cfg.name)
    if d_factory is None:
        raise ValueError(f"Unknown Discriminator: {d_cfg.name}")
    print(f"Discriminator: {d_cfg.name}\nDiscriminator parameters: {d_cfg.parameters}")
    return d_factory(**d_cfg.parameters)