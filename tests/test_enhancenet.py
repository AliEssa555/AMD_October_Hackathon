"""EnhanceNet unit tests (spec 7.7 items 1-6, including the branch parameter bounds)."""

import pytest
import torch

from yolo_der.models.enhancenet import NUM_BRANCHES, EnhanceNet

CHANNELS = [32, 64, 256]


@pytest.mark.parametrize("c", CHANNELS)
def test_shape_is_preserved_for_odd_spatial_dims(c):
    enh = EnhanceNet(c)
    x = torch.randn(2, c, 13, 17)
    assert enh(x).shape == x.shape


def test_identity_at_initialisation():
    enh = EnhanceNet(64).eval()
    x = torch.randn(4, 64, 16, 16)
    assert torch.equal(enh(x), x)


def test_lambda_starts_at_zero():
    assert EnhanceNet(32).lambda_enh.detach().item() == 0.0


def test_gate_starts_uniform():
    """Zero-init on the gate's second layer makes Eq. 8 exactly 0.2 per branch."""
    enh = EnhanceNet(64).eval()
    beta = enh.gate(torch.randn(8, 64, 9, 9))
    assert beta.shape == (8, NUM_BRANCHES)
    assert torch.allclose(beta, torch.full_like(beta, 1 / NUM_BRANCHES), atol=1e-6)


def test_gate_weights_sum_to_one():
    enh = EnhanceNet(32)
    with torch.no_grad():
        for p in (*enh.gate_fc1.parameters(), *enh.gate_fc2.parameters()):
            p.normal_(0, 5)
    beta = enh.gate(torch.randn(6, 32, 11, 11))
    assert torch.allclose(beta.sum(dim=1), torch.ones(6), atol=1e-6)
    assert (beta >= 0).all()


def test_tone_branch_is_a_no_op_at_gamma_one():
    """Gamma = 1 must make Eq. 3 the identity: sigmoid then logit cancel exactly.

    This is what makes the sigmoid/logit construction a legitimate tone *remap* rather
    than a distortion -- if it failed, every variant would start from a damaged feature map.
    """
    enh = EnhanceNet(16).eval()
    with torch.no_grad():
        enh.lambda_enh.fill_(1.0)
        # Put all gate mass on the tone branch (index 0) so out = x + F_tone.
        enh.gate_fc2.bias.copy_(torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0]))
    x = torch.randn(3, 16, 12, 12)
    # At init gamma = 1, so F_tone == x and the output is 2x (identity branch excluded by
    # the gate). Compare against that rather than against x.
    assert torch.allclose(enh(x), 2 * x, atol=1e-3)


def test_predicted_parameters_stay_inside_their_bounds():
    """Spec 7.7 item 4: with std-10 random heads every bounded parameter holds."""
    enh = EnhanceNet(64)
    with torch.no_grad():
        for p in enh.parameters():
            p.normal_(0, 10)
        enh.lambda_enh.fill_(1.0)
    enh.record = True
    out = enh(torch.randn(8, 64, 16, 16) * 20)

    assert torch.isfinite(out).all(), "bounded parameters must keep the output finite"
    gamma, alpha = enh.last["gamma"], enh.last["alpha"]
    kappa, scale = enh.last["kappa"], enh.last["scale"]
    # These are per-image means of per-channel values, so they lie inside the bounds.
    # The intervals are open in exact arithmetic but CLOSED in float32: with std-10
    # weights tanh and sigmoid saturate to exactly +/-1 and 0/1, so gamma reaches 1/3
    # and 3, kappa reaches 0 and 2, scale reaches 0.5 and 1.5. All harmless -- the
    # point of the bounds is that the branch formulas stay finite (asserted above),
    # and the endpoints are benign: kappa = 0 makes the detail branch the identity.
    assert (gamma >= 1 / 3).all() and (gamma <= 3).all()
    assert (alpha > 0).all()
    assert (kappa >= 0).all() and (kappa <= 2).all()
    assert (scale >= 0.5).all() and (scale <= 1.5).all()


def test_gradients_reach_every_parameter():
    enh = EnhanceNet(32)
    with torch.no_grad():
        enh.lambda_enh.fill_(0.1)
    enh(torch.randn(2, 32, 12, 12)).sum().backward()
    for name, p in enh.named_parameters():
        assert p.grad is not None, f"{name} got no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad is not finite"


def test_blur_kernel_is_a_buffer_without_gradient():
    """The detail branch's Gaussian is fixed (spec 7.7 item 3)."""
    enh = EnhanceNet(32)
    assert "blur_kernel" in dict(enh.named_buffers())
    assert "blur_kernel" not in dict(enh.named_parameters())
    assert not enh.blur_kernel.requires_grad
    assert torch.allclose(enh.blur_kernel.sum(), torch.tensor(1.0), atol=1e-6)


def test_detail_branch_sharpens_a_blurred_input():
    """Sanity check on Eq. 5: unsharp masking must raise local contrast."""
    enh = EnhanceNet(8).eval()
    with torch.no_grad():
        enh.lambda_enh.fill_(1.0)
        enh.gate_fc2.bias.copy_(torch.tensor([0.0, 0.0, 20.0, 0.0, 0.0]))  # detail only
    x = torch.zeros(1, 8, 32, 32)
    x[:, :, 12:20, 12:20] = 1.0  # a soft square
    out = enh(x)
    # Edge overshoot is the signature of unsharp masking.
    assert out.max() > x.max()


def test_amp_keeps_dtype_and_stays_finite():
    enh = EnhanceNet(64)
    with torch.no_grad():
        enh.lambda_enh.fill_(0.5)
    x = torch.randn(2, 64, 16, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = enh(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_constant_channel_does_not_divide_by_zero():
    """Sigma = 0 for a constant channel; the eps must keep z-scores finite."""
    enh = EnhanceNet(16)
    with torch.no_grad():
        enh.lambda_enh.fill_(1.0)
    out = enh(torch.ones(2, 16, 8, 8) * 3.0)
    assert torch.isfinite(out).all()


def test_recording_does_not_change_the_output():
    enh = EnhanceNet(32).eval()
    with torch.no_grad():
        enh.lambda_enh.fill_(0.3)
    x = torch.randn(2, 32, 8, 8)
    plain = enh(x)
    enh.record = True
    assert torch.equal(plain, enh(x))
    assert enh.last["gate"].shape == (2, NUM_BRANCHES)
    assert not enh.last["gate"].requires_grad


def test_external_gate_is_used_verbatim():
    """der_routes_enh: DER's router supplies beta, so the built-in gate must not exist."""
    enh = EnhanceNet(32, external_gate=True).eval()
    assert not hasattr(enh, "gate_fc1")
    with torch.no_grad():
        enh.lambda_enh.fill_(1.0)
    x = torch.randn(2, 32, 8, 8)

    identity_only = torch.zeros(2, NUM_BRANCHES)
    identity_only[:, 4] = 1.0  # all mass on the identity branch (Eq. 7)
    assert torch.allclose(enh(x, gate=identity_only), 2 * x, atol=1e-5)


def test_external_gate_contract_is_enforced():
    strict = EnhanceNet(32, external_gate=True)
    with pytest.raises(ValueError, match="requires gate"):
        strict(torch.randn(2, 32, 8, 8))
    with pytest.raises(ValueError, match="columns"):
        strict(torch.randn(2, 32, 8, 8), gate=torch.ones(2, 3))

    owns_gate = EnhanceNet(32)
    with pytest.raises(ValueError, match="owns its gate"):
        owns_gate(torch.randn(2, 32, 8, 8), gate=torch.ones(2, NUM_BRANCHES))


def test_conditioning_changes_the_gate():
    enh = EnhanceNet(32, cond_dim=1)
    with torch.no_grad():
        for p in (*enh.gate_fc1.parameters(), *enh.gate_fc2.parameters()):
            p.normal_(0, 1)
    x = torch.randn(4, 32, 8, 8)
    assert not torch.allclose(enh.gate(x, torch.ones(4, 1)), enh.gate(x, torch.zeros(4, 1)))


def test_parameter_cost_is_under_the_paper_claim_per_module():
    """Paper claims < 10% extra params overall; per module it should be far smaller."""
    c = 128
    params = sum(p.numel() for p in EnhanceNet(c).parameters())
    assert params < 4 * c * c, f"EnhanceNet({c}) has {params} params"
