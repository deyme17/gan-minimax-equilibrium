from .register_optimizers import OPTIMIZERS
from torch.optim.optimizer import ParamsT
import torch.optim as opt
from utils import Config, RegistryConfig


def _get_optimizer(opt_cfg: RegistryConfig, params: ParamsT) -> opt.Optimizer:
    opt_factory = OPTIMIZERS.get(opt_cfg.name)
    if opt_factory is None:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")
    return opt_factory(params, **opt_cfg.parameters)


def get_G_optimizer(config: Config, params: ParamsT) -> opt.Optimizer: 
    optim = _get_optimizer(config.G_optimizer, params)
    print(f"Generator optimizer: {config.G_optimizer.name}")
    return optim

def get_D_optimizer(config: Config, params: ParamsT) -> opt.Optimizer:
    optim = _get_optimizer(config.D_optimizer, params)
    print(f"Discriminator optimizer: {config.D_optimizer.name}")
    return optim