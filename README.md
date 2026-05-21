# GAN Minimax Equilibrium

> Experimental study of game-theoretic properties of Generative Adversarial Networks on the Animal Faces dataset.

---

## Overview

This project trains and compares multiple GAN variants to study the conditions under which Nash equilibrium is reached in the minimax game between a generator and discriminator. Eight configurations across three loss families are explored: **BCE (Vanilla GAN)**, **Wasserstein GAN** (weight clipping -> spectral norm -> gradient penalty), and **Hinge Loss GAN** (baseline -> SN -> GP).

The full theoretical analysis is in `docs/report.typ`.

---

## Results Summary

| Config | Loss | Regularization | Gradient stability | Notes |
|---|---|---|---|---|
| `test_vanilla_gan_baseline` | BCE | — | ❌ explosion | mode collapse |
| `test_vanilla_gan` | BCE | dropout, TTUR | ⚠️ stable but weak | D too weak to teach G |
| `test_wgan_wc` | Wasserstein | weight clip | ❌ catastrophic | capacity degradation |
| `test_wgan_sn` | Wasserstein | spectral norm | ⚠️ partial | global Lipschitz not guaranteed |
| `test_wgan_gp` | Wasserstein | gradient penalty | ✅ **best** | avg G norm ≈ 1.7, D norm ≈ 5.4 |
| `test_hinge_gan` | Hinge | — | ❌ unstable | D works, G can't learn stably |
| `test_hinge_gan_sn` | Hinge | spectral norm | ⚠️ moderate | D slightly stronger than G |
| `test_hinge_gan_gp` | Hinge | gradient penalty | ⚠️ good | close to WGAN-GP, more sensitive |

**Best config:** `test_wgan_gp.yaml` — WGAN with gradient penalty, PatchGAN critic, Layer Normalization, λ=10.

---

## Project Structure

```
gan-minimax-equilibrium/
├── configs/                  # YAML experiment configs
├── data/                     # Animal Faces dataset (not tracked)
├── checkpoints/              # Saved model checkpoints (not tracked)
├── docs/
│   ├── report.typ            # Full coursework report (Typst)
│   ├── eval_plots/           # G/D score distributions, gradient norms, mode collapse plots
│   ├── generated_imgs/       # Sample outputs per experiment
│   ├── training_plots/       # Loss curves per experiment
│   └── resources/            # Diagrams used in the report
├── models/
│   ├── generator.py          # BaselineGenerator, AdvancedGenerator, WGANAdvancedGenerator
│   ├── discriminator.py      # Discriminator, WGANCritic
│   ├── model_factory.py
│   ├── register_models.py
│   └── test_models.py        # pytest unit tests
├── losses/
│   ├── bce_loss.py           # Non-saturating BCE
│   ├── wasserstein_loss.py   # Earth Mover's Distance
│   └── hinge_loss.py         # Margin-based hinge
├── dataset/                  # Animal Faces dataloader
├── optimizers/               # AdamW, Adam, RMSprop factory
├── transforms/               # Resize, flip, normalize to [-1, 1]
├── utils/
│   ├── config_cls.py         # Dataclass config with from_dict
│   ├── registry.py           # Generic Registry[T]
│   └── helpers.py            # gradient_penalty, grad_norm, clip_rate, etc.
├── train.py                  # Main training loop
└── evaluation.ipynb          # Analysis and visualization
```

---

## Architectures

**Generators**
- `BaselineGenerator` — transposed convolutions (DCGAN-style)
- `AdvancedGenerator` — Upsample + Conv residual blocks with BatchNorm
- `WGANAdvancedGenerator` — same but with GroupNorm (BatchNorm violates Lipschitz)

**Discriminators**
- `Discriminator` — supports `patch`/global pooling, `batch`/`instance`/`none` norm, optional spectral norm
- `WGANCritic` — same but `layer`/`instance`/`none` norm only (no BatchNorm)

All models are registered via `Registry` and selected from YAML config.

---

## Setup

```bash
git clone https://github.com/deyme17/gan-minimax-equilibrium
cd gan-minimax-equilibrium
pip install -r requirements.txt
```

Download [Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) and place the `cat` subset under `data/`.

---

## Training

```bash
python train.py --config configs/test_wgan_gp.yaml --tag wgan_gp
```

With WandB logging:
```bash
python train.py --config configs/test_wgan_gp.yaml --tag wgan_gp --wandb --wandb-project gan-minimax-equilibrium
```

Resume from checkpoint:
```bash
python train.py --config configs/test_wgan_gp.yaml --checkpoint checkpoints/wgan_gp_checkpoint.pt --tag wgan_gp
```

**Key CLI args**

| Arg | Description |
|---|---|
| `--config` | Path to YAML config (required) |
| `--checkpoint` | Resume from checkpoint |
| `--tag` | Name used for checkpoint and plot files |
| `--wandb` | Enable WandB logging |
| `--wandb-project` | WandB project name |

---

## Config Reference

```yaml
noise_dim: 128
epochs: 50
n_g: 1          # generator updates per D step
n_d: 8          # discriminator updates per G step
grad_penalty_lambda: 10   # null to disable
weight_clip: null         # e.g. 0.02 for WGAN-WC
instance_noise: false     # adaptive noise on inputs
G_max_norm: 5   # gradient clipping for G (null to disable)
D_max_norm: 5   # gradient clipping for D

G_model:
  name: "WG"    # baseline_G | G | WG
  parameters: {n_z: 128, image_size: 512, ...}

D_model:
  name: "C"     # D | C
  parameters: {sn: false, norm: "layer", patch: true, ...}

loss:
  name: "Wasserstein"   # BCE | Wasserstein | Hinge
  parameters: {}

G_optimizer:
  name: "Adam"
  parameters: {lr: 0.0001, betas: [0., 0.9]}
```

---

## Dependencies

```bash
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4

torch>=2.2 --index-url https://download.pytorch.org/whl/cu126
torchvision>=0.17 --index-url https://download.pytorch.org/whl/cu126
wandb>=0.27.0

tqdm>=4.66
imageio>=2.34
Pillow>=10.2
pyyaml>=6.0.3

matplotlib>=3.8
seaborn>=0.13
```