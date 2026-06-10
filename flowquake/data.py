"""EarthquakeNPP catalog loading, benchmark splits, and training crops.

Conventions match the EarthquakeNPP harness exactly:
  - time in days, x/y in km (catalog columns), magnitude >= Mcut
  - splits by date: train [aux_start, val_start), val [val_start, test_start),
    test [test_start, test_end)
  - per-event scores: tll in log(1/day), sll in log(1/km^2)

Token i carries (log tau_i, x_i, y_i, m_i) where tau_i is the gap *preceding*
event i. The encoder state after event i conditions the prediction of event
i+1, so the loss/eval mask at position i selects targets whose event i+1
falls in the desired window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SECONDS_PER_DAY = 86400.0
TAU_FLOOR_DAYS = 1e-7  # ~9 ms; catalog's smallest nonzero gap is ~5e-8 d


@dataclass
class CatalogTensors:
    """Whole-catalog arrays plus split masks and normalization stats."""

    t_days: np.ndarray        # (E,) float64, days since first event
    feats: torch.Tensor       # (E, 4) float32 normalized tokens
    raw: torch.Tensor         # (E, 4) float32 [log_tau, x, y, mag] unnormalized
    target_train: np.ndarray  # (E,) bool: event is a train-period target
    target_val: np.ndarray
    target_test: np.ndarray
    stats: dict               # normalization stats (train-period only)
    times: pd.DatetimeIndex

    @property
    def n_events(self) -> int:
        return len(self.t_days)


def load_catalog(
    path: str,
    mcut: float,
    aux_start: str,
    train_start: str,
    val_start: str,
    test_start: str,
    test_end: str,
) -> CatalogTensors:
    df = pd.read_csv(path, parse_dates=["time"])
    df = df[df["magnitude"] >= mcut].sort_values("time").reset_index(drop=True)
    df = df[(df["time"] >= aux_start) & (df["time"] < test_end)].reset_index(drop=True)

    times = pd.DatetimeIndex(df["time"])
    t_days = (times - times[0]).total_seconds().to_numpy() / SECONDS_PER_DAY
    tau = np.diff(t_days, prepend=np.nan)
    tau[0] = np.nan  # event 0 has no preceding gap
    tau = np.clip(tau, TAU_FLOOR_DAYS, None)
    log_tau = np.log(tau)
    log_tau[0] = np.nanmedian(log_tau)  # harmless filler; event 0 is never a target

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    m = df["magnitude"].to_numpy(float)

    target_train = np.asarray((times >= train_start) & (times < val_start))
    target_val = np.asarray((times >= val_start) & (times < test_start))
    target_test = np.asarray((times >= test_start) & (times < test_end))

    # Normalization from the training period only (events before val_start).
    fit = np.asarray(times < val_start)
    stats = {"mcut": float(mcut)}
    for name, arr in [("log_tau", log_tau), ("x", x), ("y", y), ("mag", m)]:
        stats[f"{name}_mean"] = float(arr[fit].mean())
        stats[f"{name}_std"] = float(arr[fit].std() + 1e-8)

    raw = np.stack([log_tau, x, y, m], axis=1)
    feats = np.stack(
        [
            (log_tau - stats["log_tau_mean"]) / stats["log_tau_std"],
            (x - stats["x_mean"]) / stats["x_std"],
            (y - stats["y_mean"]) / stats["y_std"],
            (m - stats["mag_mean"]) / stats["mag_std"],
        ],
        axis=1,
    )

    return CatalogTensors(
        t_days=t_days,
        feats=torch.from_numpy(feats).float(),
        raw=torch.from_numpy(raw).float(),
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        stats=stats,
        times=times,
    )


class CropDataset(Dataset):
    """Random contiguous crops of the catalog for chunked training.

    Each item: tokens (W, 4), plus a loss mask over positions i in the crop
    selecting those whose *next* event (i+1, global indexing) is a training
    target and which lie past the burn-in prefix.
    """

    def __init__(
        self,
        cat: CatalogTensors,
        window: int = 2048,
        burn_in: int = 256,
        n_crops: int = 1024,
        seed: int = 0,
    ):
        self.cat = cat
        self.window = window
        self.burn_in = burn_in
        # next_is_train_target[i] == event i+1 is a train target
        nxt = np.zeros(cat.n_events, dtype=bool)
        nxt[:-1] = cat.target_train[1:]
        self.next_target = nxt
        # Latest useful crop start: crop must contain at least one target.
        valid = np.flatnonzero(nxt)
        self.lo = 0
        self.hi = int(valid.max()) - burn_in  # ensure targets can appear past burn-in
        self.n_crops = n_crops
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_crops

    def __getitem__(self, idx: int):
        a = int(self.rng.integers(self.lo, max(self.hi, 1)))
        b = min(a + self.window, self.cat.n_events)
        a = max(0, b - self.window)
        sl = slice(a, b)
        tokens = self.cat.feats[sl]
        # targets at position i (local) refer to global event a+i+1
        mask = torch.from_numpy(self.next_target[sl].copy())
        mask[: self.burn_in] = False
        mask[-1] = False  # last position has no in-crop successor
        target = self.cat.feats[a + 1 : b + 1]
        if target.shape[0] < tokens.shape[0]:  # crop touching catalog end
            pad = tokens.shape[0] - target.shape[0]
            target = torch.cat([target, torch.zeros(pad, 4)], dim=0)
            mask[-(pad + 1):] = False
        return tokens, target, mask


def full_sequence_batch(cat: CatalogTensors, which: str):
    """Whole-catalog tokens and target mask for exact evaluation.

    Returns (tokens (1, E, 4), target (1, E, 4), mask (1, E)) where mask[i]
    selects positions whose next event is a {val,test} target.
    """
    tgt = {"train": cat.target_train, "val": cat.target_val, "test": cat.target_test}[which]
    nxt = np.zeros(cat.n_events, dtype=bool)
    nxt[:-1] = tgt[1:]
    tokens = cat.feats.unsqueeze(0)
    target = torch.cat([cat.feats[1:], torch.zeros(1, 4)], dim=0).unsqueeze(0)
    mask = torch.from_numpy(nxt).unsqueeze(0)
    return tokens, target, mask
