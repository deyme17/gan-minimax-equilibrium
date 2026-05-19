from .base_gan_loss import BaseGANLoss
import torch


class HingeLoss(BaseGANLoss):
    """Hinge loss, aiming to maximize the margin between the scores of real and generated data."""
    def __init__(self, label_smoothing: float = 0):
        self.real_label = 1 - label_smoothing
    
    def D_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_loss = torch.mean(torch.clamp(self.real_label - real, min=0))
        fake_loss = torch.mean(torch.clamp(1 + fake, min=0))
        return real_loss + fake_loss
    
    def G_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return -fake.mean()