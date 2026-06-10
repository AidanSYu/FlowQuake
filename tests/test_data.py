from pathlib import Path

import numpy as np
import pytest
import torch

from flowquake.data import CropDataset, full_sequence_batch, load_catalog

CATALOG = Path(__file__).resolve().parents[1] / "reference/Datasets/ComCat/ComCat_catalog.csv"

pytestmark = pytest.mark.skipif(not CATALOG.exists(), reason="ComCat catalog not present")


@pytest.fixture(scope="module")
def cat():
    return load_catalog(
        str(CATALOG), 2.5, "1971-01-01", "1981-01-01",
        "1998-01-01", "2007-01-01", "2020-01-17",
    )


def test_split_counts_match_benchmark(cat):
    # grounded against the EarthquakeNPP harness (create_dataset.py splits)
    assert cat.n_events == 92263
    assert cat.target_val.sum() == 14741
    assert cat.target_test.sum() == 21889


def test_gaps_positive_and_finite(cat):
    log_tau = cat.raw[:, 0].numpy()
    assert np.isfinite(log_tau).all()
    t = cat.t_days
    assert (np.diff(t) >= 0).all()


def test_normalization_is_train_only(cat):
    # stats computed on pre-1998 events: roughly zero-mean/unit-std there
    fit = ~(cat.target_val | cat.target_test)
    f = cat.feats[torch.from_numpy(fit)]
    assert abs(f[:, 0].mean().item()) < 0.05
    assert abs(f[:, 0].std().item() - 1.0) < 0.05


def test_crop_alignment(cat):
    ds = CropDataset(cat, window=512, burn_in=64, n_crops=4, seed=1)
    tokens, target, mask = ds[0]
    assert tokens.shape == (512, 4) and target.shape == (512, 4)
    assert not mask[:64].any()
    assert not mask[-1]
    # target at position i is token at position i+1 (same crop, shifted)
    i = int(mask.nonzero()[0])
    assert torch.allclose(target[i], tokens[i + 1])


def test_full_sequence_mask_shift(cat):
    tokens, target, mask = full_sequence_batch(cat, "test")
    assert mask.sum().item() == 21889
    idx = mask[0].nonzero(as_tuple=True)[0]
    # position i predicts event i+1, which must be a test target
    assert cat.target_test[(idx + 1).numpy()].all()
    assert torch.allclose(target[0, idx], cat.feats[idx + 1])
