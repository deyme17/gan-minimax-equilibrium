from .base_gan_loss import BaseGANLoss
import torch.nn as nn
import torch


class HingeLoss(BaseGANLoss):
    """Hinge loss, aiming to maximize the margin between the scores of real and generated data."""
    def __init__(self) -> None:
        self.bce = nn.BCEWithLogitsLoss()

    def D_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_loss = torch.mean(torch.maximum(0, 1 - real))
        fake_loss = torch.mean(torch.maximum(0, 1 + fake))
        return real_loss + fake_loss
    
    def G_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return -fake.mean()