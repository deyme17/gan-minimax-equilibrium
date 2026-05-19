import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import yaml
import wandb
from pathlib import Path

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from utils import (Config,
    save_checkpoint, load_checkpoint, set_seed,
    visualize_progress, gradient_penalty,
    grad_norm, clip_rate, gan_init_weights,
    add_instance_noise
)

from models import get_discriminator, get_generator
from optimizers import get_D_optimizer, get_G_optimizer
from dataset import get_dataloader
from transforms import get_transform
from losses import BaseGANLoss, get_loss



def train(G: nn.Module, 
          D: nn.Module, 
          D_optim: optim.Optimizer, 
          G_optim: optim.Optimizer,
          criterion: BaseGANLoss, 
          train_loader: DataLoader, 
          config: Config,
          curr_epoch: int = 0, 
          G_losses: list[float] = None,
          D_losses: list[float] = None,
          device: str = "cpu",
          experiment_tag: str = "experiment",
          use_wandb: bool = False) -> tuple[list[float], list[float]]:
    """
    GAN training loop. Return: (G_losses, D_losses).
    """
    if torch.cuda.is_available():
        print("[INFO] CUDA is used for training.")
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARNING] CUDA is not available.")
        
    if G_losses is None: G_losses = []
    if D_losses is None: D_losses = []

    D.train()
    G.train()

    for epoch in range(curr_epoch, config.epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        d_real_scores = []
        d_fake_scores = []
        g_norms = []
        d_norms = []

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch_idx, real_imgs in enumerate(pbar):
            ### Train Discriminator ###
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            
            D_optim.zero_grad(set_to_none=True)
            
            # real
            real_imgs = add_instance_noise(real_imgs)
            real_pred = D(real_imgs).view(-1)
            # fake
            noise = torch.randn(b_size, config.noise_dim, device=device)
            fake_imgs = G(noise).detach()
            fake_imgs = add_instance_noise(fake_imgs)
            fake_pred = D(fake_imgs).view(-1)
            
            D_loss = criterion.D_loss(real_pred, fake_pred)

            # apply gradient penalty
            if config.grad_penalty_lambda is not None:
                grad_penalty = gradient_penalty(D, real_imgs, fake_imgs, device)
                D_loss += config.grad_penalty_lambda * grad_penalty

            D_loss.backward()
            d_norms.append(grad_norm(D))
            
            # clip grad norm
            if config.D_max_norm is not None:
                clip_grad_norm_(D.parameters(), config.D_max_norm)
                
            D_optim.step()

            # apply weight clipping
            if config.weight_clip is not None:
                c = config.weight_clip
                for p in D.parameters():
                    p.data.clamp_(-c, c)

            epoch_d_loss += D_loss.item()
            d_real_scores.append(real_pred.mean().item())
            d_fake_scores.append(fake_pred.mean().item())
            
            ### Train Generator ###
            if (batch_idx + 1) % config.n_d == 0 or (batch_idx + 1) == len(train_loader):
                for _ in range(config.n_g):
                    noise = torch.randn(b_size, config.noise_dim, device=device)
                    
                    G_optim.zero_grad(set_to_none=True)
                    
                    fake_imgs = G(noise)
                    fake_pred = D(fake_imgs).view(-1)
                    
                    G_loss = criterion.G_loss(fake_pred)                    
                    G_loss.backward()
                    g_norms.append(grad_norm(G))

                    # clip grad norm                    
                    if config.G_max_norm is not None:
                        clip_grad_norm_(G.parameters(), config.G_max_norm)
                        
                    G_optim.step()
                    epoch_g_loss += G_loss.item()

            pbar.set_postfix({
                "D": f"{epoch_d_loss/(batch_idx+1):.3f}",
                "G": f"{epoch_g_loss/max(len(g_norms),1):.3f}",
                "Dx": f"{sum(d_real_scores)/len(d_real_scores):.2f}",
                "Dgz": f"{sum(d_fake_scores)/len(d_fake_scores):.2f}",
                "avg_gN": f"{sum(g_norms)/len(g_norms):.1f}" if g_norms else "-",
                "avg_dN": f"{sum(d_norms)/len(d_norms):.1f}",
            })
 
        # metrics
        n_d_steps = max(len(d_real_scores), 1)
        n_g_steps = max(len(g_norms), 1)
 
        avg_g_loss = epoch_g_loss / n_g_steps
        avg_d_loss = epoch_d_loss / n_d_steps
        mean_real = sum(d_real_scores) / n_d_steps
        mean_fake = sum(d_fake_scores) / n_d_steps
        balance = mean_real - mean_fake
        mean_g_norm = sum(g_norms) / n_g_steps
        mean_d_norm = sum(d_norms) / n_d_steps
 
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
 
        # checkpoint
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] G Loss: {G_losses[-1]:.3f} | D Loss: {D_losses[-1]:.3f}")
            save_checkpoint(
                G=G, D=D, G_optim=G_optim, D_optim=D_optim,
                G_losses=G_losses, D_losses=D_losses, epoch=epoch,
                save_path=config.checkpoints_dir / f"{experiment_tag}_checkpoint.pt"
            )
 
        # wandb
        if use_wandb:
            log = {
                "epoch": epoch + 1,
                "loss/G": avg_g_loss,
                "loss/D": avg_d_loss,
                "D/real_mean": mean_real,
                "D/fake_mean": mean_fake,
                "D/balance": balance,
                "grad/G_norm_mean": mean_g_norm,
                "grad/G_norm_max": max(g_norms) if g_norms else 0.0,
                "grad/D_norm_mean": mean_d_norm,
                "grad/D_norm_max": max(d_norms) if d_norms else 0.0,
            }
            if config.G_max_norm:
                log["grad/G_clip_rate"] = clip_rate(g_norms, config.G_max_norm)
            if config.D_max_norm:
                log["grad/D_clip_rate"] = clip_rate(d_norms, config.D_max_norm)
            if config.loss.name == "Wasserstein":
                log["D/W_distance"] = balance
            wandb.log(log)
 
    return G_losses, D_losses


# main section
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generative Adversarial Model training pipeline.")
    parser.add_argument("--config", type=str, required=True, default=None, help="Path to an experiment global config (.yml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--tag", type=str, default="experiment", help="Current train loop launch tag (used for saving)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="gan-minimax-equilibrium", dest="wandb_project")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}") 
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}") 

    # config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    # wandb init
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.tag,
            config=config_dict,
        )

    # data
    transforms = get_transform(config.input_dim)
    dataloader = get_dataloader(config, transforms)

    # model
    G = get_generator(config).to(device)
    D = get_discriminator(config).to(device)
    G.apply(gan_init_weights)
    D.apply(gan_init_weights)

    # optimizer
    G_optimizer = get_G_optimizer(config, G.parameters())
    D_optimizer = get_D_optimizer(config, D.parameters())

    # loss
    criterion = get_loss(config)

    # load checkpoint
    curr_epoch = 0
    D_losses = []
    G_losses = []
    if args.checkpoint is not None:
        G, D, state_dict = load_checkpoint(
            path=checkpoint_path, 
            G=G, D=D,
            map_location=device,
        )
        G_optimizer.load_state_dict(state_dict["G_optim"])
        D_optimizer.load_state_dict(state_dict["D_optim"])

        curr_epoch = state_dict["epoch"] + 1
        D_losses = state_dict["D_losses"]
        G_losses = state_dict["G_losses"]
        print(f"Resumed '{checkpoint_path}' at epoch {curr_epoch}.")

    # train
    G_losses, D_losses = train(
        G=G, D=D,
        G_optim=G_optimizer,
        D_optim=D_optimizer,
        criterion=criterion,
        train_loader=dataloader,
        config=config,
        curr_epoch=curr_epoch,
        D_losses=D_losses,
        G_losses=G_losses,
        device=device,
        experiment_tag=args.tag,
        use_wandb=args.wandb,
    )
    if args.wandb:
        wandb.finish()
    
    visualize_progress(G_losses, D_losses, title=args.tag, save=True)