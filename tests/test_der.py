"""DER unit tests (spec 7.7 items 1-6, applied to the router and its paths)."""

import pytest
import torch

from yolo_der.losses import routing_entropy
from yolo_der.models.der import DER

CHANNELS = [32, 64, 256]


@pytest.mark.parametrize("c", CHANNELS)
def test_shape_is_preserved_for_odd_spatial_dims(c):
    """Odd, non-square H/W catch padding mistakes that 32x32 would hide."""
    der = DER(c)
    x = torch.randn(2, c, 13, 17)
    assert der(x).shape == x.shape


def test_identity_at_initialisation():
    """lambda_der starts at 0, so injecting into pretrained weights is a no-op at step 0."""
    der = DER(64).eval()
    x = torch.randn(4, 64, 16, 16)
    assert torch.equal(der(x), x)


def test_lambda_starts_at_zero():
    assert DER(32).lambda_der.detach().item() == 0.0


def test_routing_starts_uniform():
    """Zero-init on the router's second Linear makes Eq. 2 exactly uniform at init."""
    der = DER(64, num_paths=4).eval()
    weights = der.route(torch.randn(8, 64, 9, 9))
    assert weights.shape == (8, 4)
    assert torch.allclose(weights, torch.full_like(weights, 0.25), atol=1e-6)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_routing_weights_sum_to_one(n):
    der = DER(32, num_paths=n)
    with torch.no_grad():
        for p in der.router.parameters():
            p.normal_(0, 5)  # force a non-degenerate distribution
    weights = der.route(torch.randn(6, 32, 11, 11))
    assert weights.shape == (6, n)
    assert torch.allclose(weights.sum(dim=1), torch.ones(6), atol=1e-6)
    assert (weights >= 0).all()


def test_temperature_sharpens_routing():
    """T < 1 concentrates the distribution; T > 1 flattens it."""
    x = torch.randn(4, 32, 8, 8)
    sharp, flat = DER(32, temperature=0.1), DER(32, temperature=10.0)
    with torch.no_grad():
        for mod in (sharp, flat):
            torch.manual_seed(0)
            for p in mod.router.parameters():
                p.normal_(0, 1)
    # routing_entropy, not a hand-rolled version: at T=0.1 the softmax saturates to
    # exact zeros and log(0) would give NaN. That epsilon is why the helper exists.
    assert routing_entropy(sharp.route(x)) < routing_entropy(flat.route(x))


def test_gradients_reach_every_parameter():
    der = DER(32)
    with torch.no_grad():
        der.lambda_der.fill_(0.1)  # lambda = 0 would legitimately zero the path grads
    der(torch.randn(2, 32, 12, 12)).sum().backward()
    for name, p in der.named_parameters():
        assert p.grad is not None, f"{name} got no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad is not finite"


def test_outputs_stay_finite_with_large_random_weights():
    der = DER(64, num_paths=4)
    with torch.no_grad():
        for p in der.parameters():
            p.normal_(0, 10)
        der.lambda_der.fill_(1.0)
    out = der(torch.randn(2, 64, 16, 16) * 50)
    assert torch.isfinite(out).all()


def test_se_path_is_neutral_at_initialisation():
    """Path 3's delta must be exactly 0 at init (the 2*sigmoid choice, see module docs)."""
    der = DER(32, num_paths=4).eval()
    se_path = der.paths[2]
    x = torch.randn(3, 32, 8, 8)
    assert torch.allclose(se_path(x), torch.zeros_like(x), atol=1e-6)


def test_amp_keeps_dtype_and_stays_finite():
    """Module maths runs in fp32 internally but must return the caller's dtype (spec 7)."""
    der = DER(64)
    with torch.no_grad():
        der.lambda_der.fill_(0.5)
    x = torch.randn(2, 64, 16, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = der(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_recording_is_off_by_default_and_does_not_change_output():
    der = DER(32).eval()
    with torch.no_grad():
        der.lambda_der.fill_(0.3)
    x = torch.randn(2, 32, 8, 8)

    assert der.record is False
    plain = der(x)
    assert der.last == {}

    der.record = True
    recorded = der(x)
    assert torch.equal(plain, recorded), "recording must not alter the output"
    assert der.last["routing"].shape == (2, 4)
    assert not der.last["routing"].requires_grad, "recorded tensors must be detached"


def test_routing_for_loss_only_when_requested():
    x = torch.randn(2, 32, 8, 8)

    quiet = DER(32)
    quiet(x)
    assert quiet.routing_for_loss is None

    loud = DER(32, collect_for_loss=True)
    loud(x)
    assert loud.routing_for_loss is not None
    assert loud.routing_for_loss.requires_grad, "the loss needs a live graph"


def test_routing_for_loss_is_cleared_each_forward():
    """A stale tensor would pin a freed autograd graph across steps."""
    der = DER(32, collect_for_loss=True)
    x = torch.randn(2, 32, 8, 8)
    first = der(x)
    first.sum().backward()
    stale = der.routing_for_loss
    der(x)
    assert der.routing_for_loss is not stale


def test_conditioning_changes_routing():
    """E3: with cond_dim=1 the router sees s_dark and must actually use it."""
    der = DER(32, cond_dim=1)
    with torch.no_grad():
        for p in der.router.parameters():
            p.normal_(0, 1)
    x = torch.randn(4, 32, 8, 8)
    dark = der.route(x, torch.ones(4, 1))
    bright = der.route(x, torch.zeros(4, 1))
    assert not torch.allclose(dark, bright)


def test_conditioning_required_when_declared():
    with pytest.raises(ValueError, match="cond_dim"):
        DER(32, cond_dim=1).route(torch.randn(2, 32, 8, 8))


def test_router_only_builds_no_paths_and_refuses_forward():
    der = DER(32, num_paths=5, router_only=True)
    assert len(der.paths) == 0
    assert der.route(torch.randn(2, 32, 8, 8)).shape == (2, 5)
    with pytest.raises(RuntimeError, match="router_only"):
        der(torch.randn(2, 32, 8, 8))


@pytest.mark.parametrize("n", [0, 1, 6])
def test_invalid_num_paths_rejected(n):
    with pytest.raises(ValueError, match="num_paths"):
        DER(32, num_paths=n)


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError, match="temperature"):
        DER(32, temperature=0.0)


def test_parameter_cost_is_small():
    """Depthwise-separable paths keep DER cheap; a C^2*k^2 path would blow this up."""
    c = 128
    params = sum(p.numel() for p in DER(c).parameters())
    assert params < 4 * c * c, f"DER({c}) has {params} params, expected well under {4 * c * c}"
