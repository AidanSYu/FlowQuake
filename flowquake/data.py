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
RECENCY_LAGS = (1, 2, 4, 8, 16, 32, 64)  # Hawkes-style order statistics
# Per-event token width: [log_tau, x, y, m] + per-lag [log dt, dx, dy, mag].
TOKEN_DIM = 4 + 4 * len(RECENCY_LAGS)
LAST_K = 32  # kernel-mixture components: the last K event locations


def _lagged(arr: np.ndarray, k: int) -> np.ndarray:
    """arr shifted by k with clamp-to-first (early rows are aux-era only)."""
    if k == 0:
        return arr
    if k >= len(arr):
        return np.full_like(arr, arr[0])
    return np.concatenate([np.full(k, arr[0]), arr[:-k]])


def recency_matrix(t_days: np.ndarray, x: np.ndarray, y: np.ndarray,
                   m: np.ndarray) -> np.ndarray:
    """ETAS-kernel raw material per event i and lag k in RECENCY_LAGS:
    log(t_i - t_{i-k}) (Omori), x_i - x_{i-k}, y_i - y_{i-k} (where recent
    activity is, relative), and m_{i-k} (productivity). Heads see the next
    event mostly lands near one of the recent events; with displacements in
    the conditioning, the offset flow can route mass to any of them."""
    cols = []
    for k in RECENCY_LAGS:
        dt = np.clip(t_days - _lagged(t_days, k), TAU_FLOOR_DAYS, None)
        cols += [np.log(dt), x - _lagged(x, k), y - _lagged(y, k), _lagged(m, k)]
    return np.stack(cols, axis=1)


@dataclass
class CatalogTensors:
    """Whole-catalog arrays plus split masks and normalization stats."""

    t_days: np.ndarray        # (E,) float64, days since first event
    feats: torch.Tensor       # (E, TOKEN_DIM) float32 normalized tokens
    raw: torch.Tensor         # (E, 4) float32 [log_tau, x, y, mag] unnormalized
    lastk: torch.Tensor       # (E, LAST_K, 4) float32 raw [x, y, log_dt, m] of
                              # events i, i-1, .., i-K+1 (mixture components)
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

    rec = recency_matrix(t_days, x, y, m)
    stats["rec_mean"] = rec[fit].mean(axis=0).tolist()
    stats["rec_std"] = (rec[fit].std(axis=0) + 1e-8).tolist()
    rec_n = (rec - np.asarray(stats["rec_mean"])) / np.asarray(stats["rec_std"])

    # Mixture components: raw [x, y, log_dt, m] of events i, i-1, .., i-K+1.
    lastk = np.empty((len(t_days), LAST_K, 4), dtype=np.float32)
    for j in range(LAST_K):
        lastk[:, j, 0] = _lagged(x, j)
        lastk[:, j, 1] = _lagged(y, j)
        lastk[:, j, 2] = np.log(np.clip(t_days - _lagged(t_days, j), TAU_FLOOR_DAYS, None))
        lastk[:, j, 3] = _lagged(m, j)

    # Catalog bounding-box area (km^2) for the uniform background component.
    stats["bg_area"] = float((x.max() - x.min() + 20.0) * (y.max() - y.min() + 20.0))
    stats["bg_xmin"], stats["bg_ymin"] = float(x.min() - 10), float(y.min() - 10)
    stats["bg_xmax"], stats["bg_ymax"] = float(x.max() + 10), float(y.max() + 10)

    raw = np.stack([log_tau, x, y, m], axis=1)
    feats = np.concatenate(
        [
            np.stack(
                [
                    (log_tau - stats["log_tau_mean"]) / stats["log_tau_std"],
                    (x - stats["x_mean"]) / stats["x_std"],
                    (y - stats["y_mean"]) / stats["y_std"],
                    (m - stats["mag_mean"]) / stats["mag_std"],
                ],
                axis=1,
            ),
            rec_n,
        ],
        axis=1,
    )

    return CatalogTensors(
        t_days=t_days,
        feats=torch.from_numpy(feats).float(),
        raw=torch.from_numpy(raw).float(),
        lastk=torch.from_numpy(lastk),
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
        target = self.cat.feats[a + 1 : b + 1, :4]
        raw_next = self.cat.raw[a + 1 : b + 1, 1:4]  # (x_km, y_km, mag)
        if target.shape[0] < tokens.shape[0]:  # crop touching catalog end
            pad = tokens.shape[0] - target.shape[0]
            target = torch.cat([target, torch.zeros(pad, 4)], dim=0)
            raw_next = torch.cat([raw_next, torch.zeros(pad, 3)], dim=0)
            mask[-(pad + 1):] = False
        return tokens, target, mask, self.cat.lastk[sl], raw_next


def full_sequence_batch(cat: CatalogTensors, which: str):
    """Whole-catalog tokens and target mask for exact evaluation.

    Returns (tokens (1, E, 4), target (1, E, 4), mask (1, E)) where mask[i]
    selects positions whose next event is a {val,test} target.
    """
    tgt = {"train": cat.target_train, "val": cat.target_val, "test": cat.target_test}[which]
    nxt = np.zeros(cat.n_events, dtype=bool)
    nxt[:-1] = tgt[1:]
    tokens = cat.feats.unsqueeze(0)
    target = torch.cat([cat.feats[1:, :4], torch.zeros(1, 4)], dim=0).unsqueeze(0)
    raw_next = torch.cat([cat.raw[1:, 1:4], torch.zeros(1, 3)], dim=0).unsqueeze(0)
    mask = torch.from_numpy(nxt).unsqueeze(0)
    return tokens, target, mask, cat.lastk.unsqueeze(0), raw_next
