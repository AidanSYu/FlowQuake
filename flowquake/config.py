"""YAML-backed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    catalog_path: str = "reference/Datasets/ComCat/ComCat_catalog.csv"
    mcut: float = 2.5
    aux_start: str = "1971-01-01"
    train_start: str = "1981-01-01"
    val_start: str = "1998-01-01"
    test_start: str = "2007-01-01"
    test_end: str = "2020-01-17"


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 64
    n_heads: int = 8
    expand: int = 2
    chunk: int = 64
    flow_hidden: int = 256
    flow_layers: int = 3
    loss_weights: tuple = (1.0, 1.0, 0.5)


@dataclass
class TrainConfig:
    window: int = 2048
    burn_in: int = 256
    batch_size: int = 8
    steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 500
    grad_clip: float = 1.0
    seed: int = 1553
    val_every: int = 1000
    val_events: int = 4096        # subsample of val targets per check
    val_ode_steps: int = 32
    patience: int = 8             # val checks without improvement
    out_dir: str = "runs/comcat25"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = Config()
        for section in ("data", "model", "train"):
            sub = d.get(section, {})
            obj = getattr(cfg, section)
            for k, v in sub.items():
                if not hasattr(obj, k):
                    raise KeyError(f"Unknown config key: {section}.{k}")
                setattr(obj, k, tuple(v) if isinstance(getattr(obj, k), tuple) else v)
        return cfg

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(
                {"data": asdict(self.data), "model": asdict(self.model),
                 "train": asdict(self.train)},
                f, sort_keys=False,
            )
