from .base_gan_loss import BaseGANLoss
import torch.nn as nn
import torch


class BCEGANLoss(BaseGANLoss):
    """Binary cross entropy loss with non-saturating generator's loss."""
    def __init__(self, label_smoothing: float = 0):
        self.real_label = 1 - label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def D_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_targets = torch.full_like(real, self.real_label)
        fake_targets = torch.zeros_like(fake)
        real_loss = self.bce(real, real_targets)
        fake_loss = self.bce(fake, fake_targets)
        return real_loss + fake_loss

    def G_loss(self, fake: torch.Tensor) -> torch.Tensor:
        targets = torch.ones_like(fake)
        return self.bce(fake, targets)