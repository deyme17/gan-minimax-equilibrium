from .base_gan_loss import BaseGANLoss
import torch

class WassersteinLoss(BaseGANLoss):
    """Wasserstein (Earth Mover's distance) loss."""
    def D_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return fake.mean() - real.mean()
    
    def G_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return -fake.mean()