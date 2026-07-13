"""Catalog loading + benchmark splits for EarthquakeNPP.

Mirrors the official benchmark contract (Datasets/README.md and
Experiments/guide_for_new_models.ipynb) so that our per-event TLL/SLL is directly
comparable to the published ETAS numbers.

Two evaluation protocols are supported by the rest of the package:

  * "whole-catalog" (default, this module's `CatalogTensors`): every event from
    `auxiliary_start` onward is fed causally through the encoder, and the temporal /
    spatial log-density is scored on every event whose *time* falls in the test window.
    This gives each test event its full real history, exactly as ETAS (a Hawkes process
    over the whole catalog) sees it — a fairer and stronger comparison than the
    32-event sliding window used by the reference dummy.

  * "windowed" (`make_sliding_windows`): reproduces the reference
    `SlidingWindowWrapper` (lookback=32) so the GRU dummy can be reproduced exactly for
    the encoder-lever ablation.

In both cases the metric is the mean over test events of
    TLL = log f(Δt_i | history_i)          (temporal log-density, units: 1/day)
    SLL = log p(x_i, y_i | history_i)       (spatial log-density,  units: 1/km^2)
matching the benchmark's definition.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------------------
# Benchmark configuration (authoritative table: Datasets/README.md "Dataset Partitioning")
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    csv: str  # path relative to the Datasets/ root
    mc: float  # magnitude of completeness (lower magnitude cutoff)
    auxiliary_start: str  # history burn-in begins
    train_start: str
    val_start: str
    test_start: str
    test_end: str


BENCHMARK: dict[str, DatasetConfig] = {
    "ComCat_25": DatasetConfig("ComCat/ComCat_catalog.csv", 2.5,
                               "1971-01-01", "1981-01-01", "1998-01-01", "2007-01-01", "2020-01-17"),
    "SCEDC_20": DatasetConfig("SCEDC/SCEDC_catalog.csv", 2.0,
                              "1981-01-01", "1985-01-01", "2005-01-01", "2014-01-01", "2020-01-01"),
    "SCEDC_25": DatasetConfig("SCEDC/SCEDC_catalog.csv", 2.5,
                              "1981-01-01", "1985-01-01", "2005-01-01", "2014-01-01", "2020-01-01"),
    "SCEDC_30": DatasetConfig("SCEDC/SCEDC_catalog.csv", 3.0,
                              "1981-01-01", "1985-01-01", "2005-01-01", "2014-01-01", "2020-01-01"),
    "SanJac_10": DatasetConfig("QTM/SanJac_catalog.csv", 1.0,
                               "2008-01-01", "2009-01-01", "2014-01-01", "2016-01-01", "2018-01-01"),
    "SaltonSea_10": DatasetConfig("QTM/SaltonSea_catalog.csv", 1.0,
                                  "2008-01-01", "2009-01-01", "2014-01-01", "2016-01-01", "2018-01-01"),
    "WHITE_06": DatasetConfig("WHITE/WHITE_catalog.csv", 0.6,
                              "2008-01-01", "2009-01-01", "2014-01-01", "2017-01-01", "2021-01-01"),
    "ETAS_25": DatasetConfig("ETAS/ETAS_California_catalog.csv", 1.0,
                             "1971-01-01", "1981-01-01", "1998-01-01", "2007-01-01", "2020-01-17"),
    "ETAS_incomplete_25": DatasetConfig("ETAS/ETAS_California_incomplete_catalog.csv", 1.0,
                                        "1971-01-01", "1981-01-01", "1998-01-01", "2007-01-01", "2020-01-17"),
    "Japan_Deprecated": DatasetConfig("Japan_Deprecated/Japan_catalog.csv", 2.5,
                                      "1990-01-01", "1992-01-01", "2007-01-01", "2011-01-01", "2020-01-01"),
}

SECONDS_PER_DAY = 86400.0


def data_root() -> Path:
    """Locate the Datasets/ directory of the cloned reference benchmark."""
    env = os.environ.get("EQNPP_DATA_ROOT")
    if env:
        return Path(env)
    # src/eqnpp/data.py -> project = parents[2]
    project = Path(__file__).resolve().parents[2]
    return project / "reference" / "Datasets"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_catalog(name: str, root: Optional[Path] = None) -> pd.DataFrame:
    """Load a benchmark catalog as a tidy DataFrame.

    Returns columns: time (datetime64), x, y (km, azimuthal-equidistant), magnitude.
    Filtered to magnitude >= Mc and the [auxiliary_start, test_end) window, sorted by time.
    """
    if name not in BENCHMARK:
        raise KeyError(f"Unknown dataset {name!r}. Options: {list(BENCHMARK)}")
    cfg = BENCHMARK[name]
    root = root or data_root()
    path = root / cfg.csv
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {path}\n"
            f"Clone the benchmark data into {root} (see README), or set EQNPP_DATA_ROOT."
        )
    df = pd.read_csv(path, parse_dates=["time"])
    needed = {"time", "x", "y", "magnitude"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}; has {list(df.columns)}")
    df = df[["time", "x", "y", "magnitude"]].copy()
    df = df[df["magnitude"] >= cfg.mc]
    aux = pd.Timestamp(cfg.auxiliary_start)
    end = pd.Timestamp(cfg.test_end)
    df = df[(df["time"] >= aux) & (df["time"] < end)]
    df = df.sort_values("time").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------------
# Whole-catalog tensors
# --------------------------------------------------------------------------------------


@dataclass
class Standardizer:
    """Affine standardizer for spatial coordinates, fit on the training window.

    Keeps the constant log-Jacobian so a density learned in standardized space can be
    reported in physical units (1/km^2): log p_phys(x) = log p_std(z) - log(sx*sy).
    """
    mean: torch.Tensor  # [2]
    std: torch.Tensor   # [2]

    def encode(self, xy: torch.Tensor) -> torch.Tensor:
        return (xy - self.mean) / self.std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    @property
    def log_det(self) -> torch.Tensor:
        # d z / d x = 1/std  ->  log|det dz/dx| = -sum log std
        return -torch.log(self.std).sum()


@dataclass
class CatalogTensors:
    """Whole-catalog tensors with per-event split labels.

    All tensors are 1-D over events (length N), ordered by time.
      t_days  : absolute time in days since auxiliary_start
      dt      : inter-event time to the *previous* event (days); dt[0] = 0
      x, y    : km coordinates (azimuthal equidistant)
      mag     : magnitude
      split   : 0=auxiliary, 1=train, 2=val, 3=test  (by the *target* event's time)

    A "transition" predicts event i from history (events < i). We only score / train on
    transitions whose target event i lies in the requested split, and require i>=1 so a
    history exists.
    """
    name: str
    cfg: DatasetConfig
    t_days: torch.Tensor
    dt: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    mag: torch.Tensor
    split: torch.Tensor  # long [N]
    standardizer: Standardizer

    @property
    def n(self) -> int:
        return self.t_days.shape[0]

    def features(self) -> torch.Tensor:
        """Per-event encoder input features [N, 4]:
        [log(dt+1e-3), x_std, y_std, mag-mc].  These describe the event that *just
        happened*; the encoder state after consuming event i conditions event i+1.
        """
        xy = torch.stack([self.x, self.y], dim=-1)
        xy_std = self.standardizer.encode(xy)
        log_dt = torch.log(self.dt.clamp_min(0.0) + 1e-3)
        mag0 = (self.mag - self.cfg.mc)
        return torch.stack([log_dt, xy_std[:, 0], xy_std[:, 1], mag0], dim=-1)

    def split_indices(self, split: int) -> torch.Tensor:
        """Indices i (target events) in `split` with i>=1 (a history exists)."""
        idx = torch.nonzero(self.split == split, as_tuple=False).squeeze(-1)
        return idx[idx >= 1]


SPLIT_AUX, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2, 3


def build_catalog_tensors(name: str, device: str | torch.device = "cpu",
                          root: Optional[Path] = None) -> CatalogTensors:
    cfg = BENCHMARK[name]
    df = load_catalog(name, root=root)
    aux = pd.Timestamp(cfg.auxiliary_start)
    t_days = (df["time"] - aux).dt.total_seconds().to_numpy() / SECONDS_PER_DAY
    x = df["x"].to_numpy().astype(np.float64)
    y = df["y"].to_numpy().astype(np.float64)
    mag = df["magnitude"].to_numpy().astype(np.float64)

    dt = np.diff(t_days, prepend=t_days[0])  # dt[0]=0
    dt = np.clip(dt, 0.0, None)

    # split label by target-event time
    times = df["time"].to_numpy()
    bounds = [pd.Timestamp(cfg.train_start), pd.Timestamp(cfg.val_start),
              pd.Timestamp(cfg.test_start), pd.Timestamp(cfg.test_end)]
    split = np.zeros(len(df), dtype=np.int64)  # aux
    split[times >= np.datetime64(bounds[0])] = SPLIT_TRAIN
    split[times >= np.datetime64(bounds[1])] = SPLIT_VAL
    split[times >= np.datetime64(bounds[2])] = SPLIT_TEST
    # times >= test_end already filtered out by load_catalog

    # standardizer fit on TRAIN events only
    train_mask = split == SPLIT_TRAIN
    if train_mask.sum() == 0:
        raise ValueError(f"No training events for {name}; check split dates.")
    xy_train = np.stack([x[train_mask], y[train_mask]], axis=-1)
    mean = torch.tensor(xy_train.mean(0), dtype=torch.float32)
    std = torch.tensor(xy_train.std(0) + 1e-6, dtype=torch.float32)
    standardizer = Standardizer(mean=mean.to(device), std=std.to(device))

    def t(a, dtype=torch.float32):
        return torch.tensor(a, dtype=dtype, device=device)

    return CatalogTensors(
        name=name, cfg=cfg,
        t_days=t(t_days), dt=t(dt), x=t(x), y=t(y), mag=t(mag),
        split=t(split, dtype=torch.long), standardizer=standardizer,
    )


# --------------------------------------------------------------------------------------
# Windowed loader (reproduces reference SlidingWindowWrapper for the encoder ablation)
# --------------------------------------------------------------------------------------


def make_sliding_windows(name: str, split: str, lookback: int = 32,
                         root: Optional[Path] = None):
    """Reproduce the reference per-split sliding-window dataset.

    Returns dict of tensors:
      hist_feat   [M, lookback, 4]  encoder inputs (log dt, x_std, y_std, mag-mc)
      target_dt   [M]               Δt (days) from last history event to target
      target_xy   [M, 2]            absolute (x, y) km of target event
      target_mag  [M]
    Times are reset to the split's first event (matching the reference), so this is the
    *bounded-history* protocol used by the dummy GRU.
    """
    cfg = BENCHMARK[name]
    df = load_catalog(name, root=root)
    times = df["time"].to_numpy()
    if split == "train":
        lo, hi = cfg.train_start, cfg.val_start
    elif split == "val":
        lo, hi = cfg.val_start, cfg.test_start
    elif split == "test":
        lo, hi = cfg.test_start, cfg.test_end
    else:
        raise ValueError(split)
    m = (times >= np.datetime64(pd.Timestamp(lo))) & (times < np.datetime64(pd.Timestamp(hi)))
    sub = df[m].reset_index(drop=True)
    if len(sub) <= lookback:
        raise ValueError(f"split {split} of {name} too short ({len(sub)}) for lookback {lookback}")
    t0 = sub["time"].iloc[0]
    t_days = (sub["time"] - t0).dt.total_seconds().to_numpy() / SECONDS_PER_DAY
    x = sub["x"].to_numpy(); y = sub["y"].to_numpy(); mag = sub["magnitude"].to_numpy()

    # standardize spatial with this split's stats only for *encoder inputs*
    mean = np.stack([x, y], -1).mean(0); std = np.stack([x, y], -1).std(0) + 1e-6
    dt = np.diff(t_days, prepend=t_days[0])

    hist_feat, target_dt, target_xy, target_mag = [], [], [], []
    for i in range(lookback, len(sub)):
        h = slice(i - lookback, i)
        log_dt = np.log(np.clip(dt[h], 0, None) + 1e-3)
        xs = (x[h] - mean[0]) / std[0]
        ys = (y[h] - mean[1]) / std[1]
        m0 = mag[h] - cfg.mc
        hist_feat.append(np.stack([log_dt, xs, ys, m0], -1))
        target_dt.append(t_days[i] - t_days[i - 1])
        target_xy.append([x[i], y[i]])
        target_mag.append(mag[i])

    to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return {
        "hist_feat": to(hist_feat),
        "target_dt": to(target_dt),
        "target_xy": to(target_xy),
        "target_mag": to(target_mag),
        "mc": cfg.mc,
    }
