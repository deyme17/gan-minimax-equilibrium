from .register_losses import LOSSES
from .base_gan_loss import BaseGANLoss
from utils import Config


def get_loss(config: Config) -> BaseGANLoss:
    loss_cfg = config.loss
    loss_factory = LOSSES.get(loss_cfg.name)
    if loss_factory is None:
        raise ValueError(f"Unknown loss function: {loss_cfg.name}")
    print(f"Loss function: {loss_cfg.name}")
    return loss_factory(**loss_cfg.parameters)