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
    noise_dim: int = 128
    input_ch: int = 3

    n_workers: int = 4
    persistent_workers: bool = False
    prefetch_factor: int|None = None

    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True

    # training
    seed: int = 17

    n_g: int = 2
    n_d: int = 1
    epochs: int = 320
    grad_penalty_lambda: float|None = None
    G_max_norm: float|None = None
    D_max_norm: float|None = None

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
        self.prefetch_factor = 2 if self.n_workers > 0 else None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "Config":
        cfg = cfg.copy()
        cfg["D_model"] = RegistryConfig(**cfg.get("D_model", {}))
        cfg["G_model"] = RegistryConfig(**cfg.get("G_model", {}))
        cfg["D_optimizer"] = RegistryConfig(**cfg.get("D_optimizer", {}))
        cfg["G_optimizer"] = RegistryConfig(**cfg.get("G_optimizer", {}))
        cfg["loss"] = RegistryConfig(**cfg.get("loss", {}))
        return cls(**cfg)