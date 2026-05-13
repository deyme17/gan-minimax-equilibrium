from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RegistryConfig:
    name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Config:
    # paths
    data_dir: Path = Path("data").resolve()
    checkpoints_dir: Path = Path("checkpoints").resolve()

    # dataloader
    batch_size: int = 16

    input_dim: int = 512
    noise_dim: int = 50
    input_ch: int = 3

    n_workers: int = 4
    persistent_workers: bool = n_workers > 0
    prefetch_factor: int = 2 if n_workers > 0 else 0

    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True

    # training
    seed: int = 17

    g_add_iter: int = 5
    epochs: int = 320
    early_stop: int = 15
    max_norm: float = 1.0

    # registries
    D_model: RegistryConfig = field(default_factory=RegistryConfig)
    G_model: RegistryConfig = field(default_factory=RegistryConfig)

    D_optimizer: RegistryConfig = field(default_factory=RegistryConfig)
    G_optimizer: RegistryConfig = field(default_factory=RegistryConfig)

    loss: RegistryConfig = field(default_factory=RegistryConfig)

    def __post_init__(self) -> None:
        """
        Derived/default logic after initialization.
        """
        self.data_dir = Path(self.data_dir).resolve()
        self.checkpoints_dir = Path(self.checkpoints_dir).resolve()
        self.persistent_workers = self.n_workers > 0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Config:
        """
        Create Config from loaded YAML dictionary.
        """
        cfg = cfg.copy()
        # nested dataclasses
        cfg["model"] = RegistryConfig(**cfg.get("model", {}))
        cfg["optimizer"] = RegistryConfig(**cfg.get("optimizer", {}))
        cfg["loss"] = RegistryConfig(**cfg.get("loss", {}))
        cfg["lr_scheduler"] = RegistryConfig(**cfg.get("lr_scheduler", {}))
        return cls(**cfg)