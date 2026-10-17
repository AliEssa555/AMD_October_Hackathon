"""Severity metric tests (spec 7.7 item 9, Phase 1 half).

The torch/numpy agreement check lives in ``test_severity_torch`` and is skipped until
``models/severity_torch.py`` exists in Phase 3.
"""

import numpy as np
import pytest

from yolo_der.data.severity import (
    assign_bins,
    atmospheric_light,
    dark_channel,
    luma,
    s_dark_from_mu,
    transmission,
)

TAU = 0.3
DCP = {"patch": 15, "omega": 0.95, "top_fraction": 0.001}


def _solid(value: float, size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.float32)


def test_luma_weights_sum_to_one():
    """A grey image's luma equals its grey level, so the BT.601 weights must sum to 1."""
    assert luma(_solid(0.42))[0, 0] == pytest.approx(0.42, abs=1e-6)


def test_all_black_image_is_maximally_dark():
    mu_y = float(luma(_solid(0.0)).mean())
    assert s_dark_from_mu(mu_y, TAU) == pytest.approx(1.0)


def test_mid_grey_image_is_not_dark():
    """mu_Y = 0.5 > tau, so Eq. 10 clamps s_dark to 0 rather than going negative."""
    mu_y = float(luma(_solid(0.5)).mean())
    assert s_dark_from_mu(mu_y, TAU) == pytest.approx(0.0)


def test_s_dark_is_linear_between_zero_and_tau():
    assert s_dark_from_mu(0.15, TAU) == pytest.approx(0.5)
    assert s_dark_from_mu(TAU, TAU) == pytest.approx(0.0)


def test_dark_channel_is_a_min_filter():
    """A single bright pixel in a dark field must not survive the patch minimum."""
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[32, 32] = 1.0
    assert dark_channel(img, DCP["patch"]).max() == pytest.approx(0.0)

    # Conversely, one dark pixel spreads over the patch.
    img = np.ones((64, 64, 3), dtype=np.float32)
    img[32, 32] = 0.0
    dc = dark_channel(img, DCP["patch"])
    assert dc[32, 32] == pytest.approx(0.0)
    assert dc[32 + DCP["patch"] // 2, 32] == pytest.approx(0.0)
    assert dc[0, 0] == pytest.approx(1.0)


def test_atmospheric_light_never_zero():
    """A fully black image must not produce A = 0; the I/A division would blow up."""
    black = _solid(0.0)
    a = atmospheric_light(black, dark_channel(black, DCP["patch"]), DCP["top_fraction"])
    assert np.all(a > 0)


def test_synthetic_haze_recovers_transmission():
    """Hazy image I = J*t + A*(1-t) with t = 0.3 should give s_fog ~ 0.7 (spec 7.7 item 9).

    J is random texture, A is white. The DCP estimate is approximate, hence the 0.1 band.
    """
    rng = np.random.default_rng(0)
    j = rng.random((256, 256, 3), dtype=np.float32)
    t_true, a = 0.3, 1.0
    hazy = (j * t_true + a * (1 - t_true)).astype(np.float32)

    t = transmission(hazy, **DCP)
    s_fog = 1.0 - float(t.mean())
    assert s_fog == pytest.approx(1 - t_true, abs=0.1)


def test_clear_image_has_low_fog_score():
    rng = np.random.default_rng(1)
    clear = rng.random((256, 256, 3), dtype=np.float32)
    assert 1.0 - float(transmission(clear, **DCP).mean()) < 0.3


def test_transmission_is_clipped_to_the_documented_range():
    rng = np.random.default_rng(2)
    t = transmission(rng.random((128, 128, 3), dtype=np.float32), **DCP)
    assert t.min() >= 0.1 - 1e-6
    assert t.max() <= 1.0 + 1e-6


def test_bins_come_from_test_split_quantiles_and_q1_is_darkest():
    """Cut points are the test split's quartiles, applied to every split (spec 6.3)."""
    mu = np.array([0.02, 0.05, 0.10, 0.20, 0.40, 0.90])
    splits = np.array(["test"] * 4 + ["train"] * 2)
    bins, cuts = assign_bins(mu, splits)

    assert cuts == pytest.approx(list(np.quantile(mu[:4], [0.25, 0.5, 0.75])))
    assert bins[0] == "q1", "darkest image lands in q1"
    assert bins[-1] == "q4", "brightest image lands in q4"
    # Monotone: brighter image never lands in a darker bin.
    order = {name: i for i, name in enumerate(["q1", "q2", "q3", "q4"])}
    ranks = [order[b] for b in bins]
    assert ranks == sorted(ranks)


def test_bins_fall_back_to_all_images_when_test_split_is_tiny():
    mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    splits = np.array(["train"] * 5)
    _, cuts = assign_bins(mu, splits)
    assert cuts == pytest.approx(list(np.quantile(mu, [0.25, 0.5, 0.75])))
