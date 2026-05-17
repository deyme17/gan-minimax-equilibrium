import random as rnd
from collections import Counter

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from transforms import get_transform
from utils import Config
from dataset import get_dataloader


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    [-1, 1] -> [0, 1]
    """
    return (x * 0.5 + 0.5).clamp(0, 1)

def test_dataset() -> None:
    cfg = Config()
    dl = get_dataloader(cfg, get_transform())

    print(f"Dataset size: {len(dl.dataset)}")
    print(f"Batches: {len(dl)}")
    print(f"Batch size: {dl.batch_size}")

    shape_counter = Counter()

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    all_images = []

    for imgs in tqdm(dl, desc='Checking dataloader'):
        B, C, H, W = imgs.shape

        shape_counter[(C, H, W)] += B

        # stats
        pixel_sum += imgs.sum().item()
        pixel_sq_sum += (imgs ** 2).sum().item()
        total_pixels += imgs.numel()

        # random images
        for img in imgs:
            if len(all_images) < 100:
                all_images.append(img.cpu())
            else:
                idx = rnd.randint(0, len(all_images) - 1)
                all_images[idx] = img.cpu()

    mean = pixel_sum / total_pixels
    std = ((pixel_sq_sum / total_pixels) - mean ** 2) ** 0.5

    print('\n=== SHAPES ===')
    for shape, count in shape_counter.items():
        print(f'{shape}: {count}')

    print('\n=== STATS ===')
    print(f'Mean: {mean:.4f}')
    print(f'Std:  {std:.4f}')

    # min/max
    sample = torch.stack(all_images)

    print(f'Min: {sample.min().item():.4f}')
    print(f'Max: {sample.max().item():.4f}')

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    chosen = rnd.sample(all_images, 9)

    for ax, img in zip(axes.flatten(), chosen):
        img = denormalize(img)

        # CHW -> HWC
        img = img.permute(1, 2, 0)

        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_dataset()