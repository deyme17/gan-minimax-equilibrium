from .config_cls import Config
from pathlib import Path
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Optimizer
import torch



def set_seed(seed: int):
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def save_checkpoint(G: nn.Module, D: nn.Module, 
                    G_optim: Optimizer, D_optim: Optimizer, 
                    G_losses: list[float], D_losses: list[float], 
                    epoch: int, save_path: Path|str) -> None:
    """Save checkpoint dict with epoch, G/D model, optimizer, losses."""
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "G_optim": G_optim.state_dict(),
        "D_optim": D_optim.state_dict(),
        "G_losses": G_losses,
        "D_losses": D_losses,
    }, Path(save_path))



def load_checkpoint(path: Path|str, G: nn.Module, D: nn.Module,
                    map_location: str = None) -> tuple[nn.Module, nn.Module, dict]:
    """Load checkpoint and return (G, D, whole_checkpoint)."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=map_location or device)
    if "G" not in checkpoint or "D" not in checkpoint:
        raise KeyError(f"'G' or 'D key is not found in checkpoint: {path}")
    
    G.to(device)
    D.to(device)
    try:
        G.load_state_dict(checkpoint["G"], strict=True)
        D.load_state_dict(checkpoint["D"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load state_dict for {G.__name__} or {D.__name__}: {e}"
        )
    G.eval()
    D.eval()

    return G, D, checkpoint



def visualize_progress(G_losses: list[float], 
                       D_losses: list[float], 
                       title: str = "Losses",
                       save: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(G_losses, label='Generator Loss')
    ax.plot(D_losses, label='Discriminator Loss')
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(f"{title}.png")
        plt.close(fig)
    else:
        plt.show()



def gradient_penalty(critic: nn.Module, 
                     real: torch.Tensor, 
                     fake: torch.Tensor, 
                     device: str = "cuda") -> torch.Tensor:
    B = real.size(0)
    real = real.detach()
    fake = fake.detach()

    eps = torch.rand(B, 1, 1, 1, device=device)
    interpolated = eps * real + (1 - eps) * fake
    interpolated.requires_grad_(True)
    mixed_scores = critic(interpolated)

    grad = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad = grad.reshape(B, -1)                    # (B, C*H*W)
    grad_norm = grad.norm(2, dim=1)               # (B,)
    grad_penalty = ((grad_norm - 1) ** 2).mean()  # scalar
    return grad_penalty



def grad_norm(module: nn.Module) -> float:
    """Total L2 gradient norm across all parameters."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5



def clip_rate(norms: list[float], max_norm: float) -> float:
    if not norms:
        return 0.0
    return sum(1 for n in norms if n > max_norm) / len(norms)



def gan_init_weights(module: nn.Module):
    """Classic GAN parameter initialization from DCGAN paper."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)