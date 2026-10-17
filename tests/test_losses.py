"""Routing regulariser tests (spec 7.7 item 10, E1)."""

import math

import pytest
import torch
from torch import nn

from yolo_der.losses import (
    collect_routing,
    load_balance,
    normalised_routing_entropy,
    regularisers,
    routing_entropy,
)
from yolo_der.models.der import DER


def uniform(b: int, n: int) -> torch.Tensor:
    return torch.full((b, n), 1.0 / n)


def one_hot(b: int, n: int) -> torch.Tensor:
    p = torch.zeros(b, n)
    p[:, 0] = 1.0
    return p


@pytest.mark.parametrize("n", [2, 4, 5])
def test_uniform_entropy_equals_log_n(n):
    assert routing_entropy(uniform(8, n)).item() == pytest.approx(math.log(n), abs=1e-5)


@pytest.mark.parametrize("n", [2, 4, 5])
def test_normalised_entropy_is_one_for_uniform_and_zero_for_decisive(n):
    assert normalised_routing_entropy(uniform(8, n)).item() == pytest.approx(1.0, abs=1e-5)
    assert normalised_routing_entropy(one_hot(8, n)).item() == pytest.approx(0.0, abs=1e-5)


def test_entropy_of_a_saturated_distribution_is_finite():
    """log(0) would be -inf; the epsilon is what keeps training alive (see test_der)."""
    assert torch.isfinite(routing_entropy(one_hot(4, 5)))


@pytest.mark.parametrize("n", [2, 4, 5])
def test_load_balance_is_zero_for_uniform(n):
    assert load_balance(uniform(16, n)).item() == pytest.approx(0.0, abs=1e-5)


def test_load_balance_penalises_collapse_onto_one_path():
    """Every image choosing path 0 is the failure mode this term exists to prevent."""
    collapsed = load_balance(one_hot(16, 4))
    assert collapsed.item() == pytest.approx(math.log(4), abs=1e-4)
    assert collapsed > load_balance(uniform(16, 4))


def test_load_balance_ignores_per_image_confidence():
    """Decisive per image but balanced across the batch must score 0: that is the goal.

    This is what makes the two terms complementary rather than contradictory -- entropy
    sharpens each image's choice, load balance only asks that the choices differ.
    """
    p = torch.eye(4).repeat(4, 1)  # 16 images, each one-hot, 4 per path
    assert load_balance(p).item() == pytest.approx(0.0, abs=1e-5)
    assert routing_entropy(p).item() == pytest.approx(0.0, abs=1e-5)


def test_collect_routing_finds_only_opted_in_modules():
    model = nn.Sequential(
        DER(32, collect_for_loss=True),
        DER(32, collect_for_loss=False),
    )
    model(torch.randn(2, 32, 8, 8))
    found = collect_routing(model)
    assert len(found) == 1
    assert found[0].shape == (2, 4)


def test_regularisers_are_exactly_zero_when_disabled():
    model = nn.Sequential(DER(32, collect_for_loss=True))
    model(torch.randn(2, 32, 8, 8))
    total, info = regularisers(model, entropy_weight=0.0, load_balance_weight=0.0)
    assert total.item() == 0.0
    assert info == {}


def test_regularisers_produce_a_gradient_that_reaches_the_router():
    model = nn.Sequential(DER(32, collect_for_loss=True))
    model(torch.randn(4, 32, 8, 8))
    total, info = regularisers(model, entropy_weight=0.01, load_balance_weight=0.01)

    assert set(info) == {"routing_entropy", "routing_load_balance", "routing_reg_total"}
    # At init routing is uniform: max entropy, zero imbalance.
    assert info["routing_entropy"] == pytest.approx(math.log(4), abs=1e-4)
    assert info["routing_load_balance"] == pytest.approx(0.0, abs=1e-5)

    total.backward()
    der = model[0]
    assert der.router[2].weight.grad is not None
    assert torch.isfinite(der.router[2].weight.grad).all()


def test_regularisers_are_averaged_over_insertion_points():
    """Adding insertion points must not silently scale the regulariser up."""
    one = nn.Sequential(DER(32, collect_for_loss=True))
    three = nn.Sequential(*[DER(32, collect_for_loss=True) for _ in range(3)])
    one(torch.randn(2, 32, 8, 8))
    for m in three:
        m(torch.randn(2, 32, 8, 8))

    a, _ = regularisers(one, entropy_weight=0.01)
    b, _ = regularisers(three, entropy_weight=0.01)
    assert a.item() == pytest.approx(b.item(), abs=1e-5)


def test_regularisers_handle_a_model_with_no_routing_modules():
    model = nn.Sequential(nn.Conv2d(3, 3, 1))
    total, info = regularisers(model, entropy_weight=0.01, load_balance_weight=0.01)
    assert total.item() == 0.0
    assert info == {}
