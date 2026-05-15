import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import yaml
from pathlib import Path

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from utils import (Config, 
    save_checkpoint, load_checkpoint, 
    set_seed, visualize_progress
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
          G_losses: list[float] = [],
          D_losses: list[float] = [],
          device: str = "cpu",
          experiment_tag: str = "experiment") -> tuple[float, float]:
    """
    GAN training loop. Return: (G_losses, D_losses).
    """
    if torch.cuda.is_available():
        print("[INFO] CUDA is used for training.")
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARNING] CUDA is not available.")

    D.train()
    G.train()

    for epoch in range(curr_epoch, config.epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        ### train Discriminator ###
        pbar = tqdm(train_loader, desc=f"Train D: Epoch {epoch+1}/{config.epochs}", leave=False)
        for real_imgs in pbar:
            real_imgs = real_imgs.to(device)

            D_optim.zero_grad(set_to_none=True)

            # all real batches
            real_pred = D(real_imgs).view(-1)
            # all fake batches
            noise = torch.randn(real_imgs.shape[0], config.noise_dim, device=device)
            with torch.no_grad():
                fake_imgs = G(noise)
            fake_pred = D(fake_imgs.detach()).view(-1)

            D_loss = criterion.D_loss(real_pred, fake_pred)
            D_loss.backward()

            # grad clipping
            if config.max_norm is not None:
                clip_grad_norm_(D.parameters(), config.max_norm)

            D_optim.step()

            epoch_d_loss += D_loss.item()

        ### train Generator ###
            for _ in range(config.n_g):
                G_optim.zero_grad(set_to_none=True)

                noise = torch.randn(config.batch_size, config.noise_dim, device=device)
                fake_imgs = G(noise)
                fake_pred = D(fake_imgs).view(-1)

                G_loss = criterion.G_loss(fake_pred)
                G_loss.backward()

                # grad clipping
                if config.max_norm is not None:
                    clip_grad_norm_(G.parameters(), config.max_norm)       

                G_optim.step()

                epoch_g_loss += G_loss.item()

        # save loss
        G_losses.append(epoch_g_loss / (len(train_loader) * config.n_g))
        D_losses.append(epoch_d_loss / len(train_loader))

        # checkpoint
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] G Loss: {G_losses[-1]:.3f} | D Loss: {D_losses[-1]:.3f}")
            save_checkpoint(
                G=G, D=D, G_optim=G_optim, D_optim=D_optim,
                G_losses=G_losses, D_losses=D_losses, epoch=epoch,
                save_path=config.checkpoints_dir / f"{experiment_tag}_checkpoint.pt"
            )

    return G_losses, D_losses


# main section
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generative Adversarial Model training pipeline.")
    parser.add_argument("--config", type=str, required=True, default=None, help="Path to an experiment global config (.yml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--tag", type=str, default="experiment", help="Current train loop launch tag (used for saving)")
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

    # data
    transforms = get_transform(config.input_dim)
    dataloader = get_dataloader(config, transforms)

    # model
    G = get_generator(config).to(device)
    D = get_discriminator(config).to(device)

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
    )
    visualize_progress(G_losses, D_losses, save=True)