import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
from PIL import Image


class PussyDataset(Dataset):
    def __init__(self, data_root: Path, transforms=None):
        self.data_root = data_root
        self.transforms = transforms
        self.images = sorted(
            [img for img in self.data_root.iterdir() 
            if img.suffix.lower() in [".jpg", "jpeg", ".png"]]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor|Image.Image:
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image