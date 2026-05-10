from abc import ABC, abstractmethod
import torch


class GANLoss(ABC):
    """Base class for GAN loss implementations."""
    @abstractmethod
    def D_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Calclulate discriminator loss between real and fake (generated) images."""
        pass
    
    @abstractmethod
    def G_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Calclulate generator loss on a fake image."""
        pass