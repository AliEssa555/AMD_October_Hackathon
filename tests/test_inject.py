"""Injection tests (spec 7.7 items 7-8): parity, metadata, load order, parameter cost.

These are the tests that catch gotcha 3 -- a silent "Transferred 300/355 items" means the
modules quietly started from scratch and the run is wasted GPU time.
"""

import pytest
import torch

from yolo_der.config import ModulesConfig
from yolo_der.models.der import DER
from yolo_der.models.enhancenet import EnhanceNet
from yolo_der.models.inject import (
    EnhancedLayer,
    SeverityProbe,
    added_parameter_report,
    find_channels,
    has_injected_weights,
    inject,
)

pytest.importorskip("ultralytics")

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

IMGSZ = 448
INSERT_AT = (4, 6, 9)


def build(nc: int = 12) -> DetectionModel:
    model = DetectionModel("yolov8n.yaml", nc=nc, verbose=False)
    model.eval()
    return model


def test_find_channels_matches_the_verified_backbone_table():
    """Spec 7.4's table, measured rather than trusted (hard rule 0.6)."""
    assert find_channels(build(), INSERT_AT, imgsz=IMGSZ) == {4: 64, 6: 128, 9: 256}


def test_insertion_points_are_all_skip_connection_sources():
    """4, 6 and 9 feed the neck, which is why EnhancedLayer must preserve `i`."""
    model = build()
    assert set(INSERT_AT) <= set(model.save)


def test_injection_is_identity_at_lambda_zero():
    """Spec 7.7 item 7: injected model's output is unchanged at initialisation."""
    torch.manual_seed(0)
    model = build()
    x = torch.zeros(1, 3, IMGSZ, IMGSZ).uniform_(0, 1)
    with torch.no_grad():
        before = model(x)

    inject(model, ModulesConfig(der=True, enhancenet=True), imgsz=IMGSZ)
    with torch.no_grad():
        after = model(x)

    assert torch.allclose(before[0], after[0], atol=1e-6)


def test_injection_preserves_layer_count_and_routing_metadata():
    model = build()
    original = {i: (model.model[i].f, model.model[i].i) for i in INSERT_AT}
    n_before = len(model.model)

    inject(model, ModulesConfig(der=True, enhancenet=True), imgsz=IMGSZ)

    assert len(model.model) == n_before, "indices must not shift"
    for i in INSERT_AT:
        layer = model.model[i]
        assert isinstance(layer, EnhancedLayer)
        assert (layer.f, layer.i) == original[i]
        assert layer.type.startswith("yolo_der.")
        assert layer.np > 0


def test_baseline_config_injects_nothing():
    model = build()
    inject(model, ModulesConfig(der=False, enhancenet=False), imgsz=IMGSZ)
    assert not any(isinstance(m, EnhancedLayer) for m in model.model)


@pytest.mark.parametrize(
    ("cfg", "want_der", "want_enh"),
    [
        (ModulesConfig(der=True), True, False),
        (ModulesConfig(enhancenet=True), False, True),
        (ModulesConfig(der=True, enhancenet=True), True, True),
    ],
)
def test_each_ablation_row_injects_the_right_modules(cfg, want_der, want_enh):
    model = build()
    inject(model, cfg, imgsz=IMGSZ)
    has_der = any(isinstance(m, DER) for m in model.modules())
    has_enh = any(isinstance(m, EnhanceNet) for m in model.modules())
    assert (has_der, has_enh) == (want_der, want_enh)
    with torch.no_grad():
        model(torch.zeros(1, 3, IMGSZ, IMGSZ))  # must still run end to end


def test_der_routes_enh_variant_runs_and_shares_the_gate():
    cfg = ModulesConfig(der=True, enhancenet=True, combine="der_routes_enh", der_paths=5)
    model = build()
    inject(model, cfg, imgsz=IMGSZ)

    der = next(m for m in model.modules() if isinstance(m, DER))
    enh = next(m for m in model.modules() if isinstance(m, EnhanceNet))
    assert der.router_only and len(der.paths) == 0
    assert enh.external_gate

    with torch.no_grad():
        model(torch.zeros(1, 3, IMGSZ, IMGSZ))


def test_severity_conditioning_installs_a_probe_and_feeds_the_routers():
    cfg = ModulesConfig(der=True, enhancenet=True, severity_cond=True)
    model = build()
    inject(model, cfg, imgsz=IMGSZ)

    assert isinstance(model.model[0], SeverityProbe)
    assert model.severity_context is not None

    # A black batch is maximally dark (s_dark = 1); a mid-grey batch is not dark at all.
    with torch.no_grad():
        model(torch.zeros(2, 3, IMGSZ, IMGSZ))
        dark = model.severity_context.cond.clone()
        model(torch.full((2, 3, IMGSZ, IMGSZ), 0.5))
        bright = model.severity_context.cond.clone()

    assert torch.allclose(dark, torch.ones_like(dark), atol=1e-5)
    assert torch.allclose(bright, torch.zeros_like(bright), atol=1e-5)


def test_added_parameters_are_under_the_papers_ten_percent_claim():
    """Spec Phase 3 acceptance: print and assert, don't take the claim on trust."""
    model = build()
    inject(model, ModulesConfig(der=True, enhancenet=True), imgsz=IMGSZ)
    report = added_parameter_report(model)
    print(
        f"\nbase={report['params_base']:,.0f}  added={report['params_added']:,.0f}  "
        f"(+{report['params_added_pct']:.2f}%)"
    )
    assert report["params_added"] > 0
    assert report["params_added_pct"] < 10.0, report


def test_has_injected_weights_distinguishes_the_two_load_orders():
    """Gotcha 3: wrapping renames model.4.cv1... to model.4.layer.cv1..."""
    plain = build()
    assert not has_injected_weights(plain.state_dict())

    injected = build()
    inject(injected, ModulesConfig(der=True, enhancenet=True), imgsz=IMGSZ)
    state = injected.state_dict()
    assert has_injected_weights(state)
    assert any(k.startswith("model.4.layer.") for k in state)


def test_pretrained_weights_transfer_fully_when_loaded_before_injection():
    """Fresh-run order: build -> load pretrained -> inject. Every key must transfer."""
    source = build()
    target = build()

    missing, unexpected = target.load_state_dict(source.state_dict(), strict=True)[:2] or (
        [],
        [],
    )
    assert not missing and not unexpected

    inject(target, ModulesConfig(der=True, enhancenet=True), imgsz=IMGSZ)
    # The wrapped layer still holds the loaded weights, just under a renamed key.
    assert torch.equal(
        target.model[4].layer.cv1.conv.weight,
        source.model[4].cv1.conv.weight,
    )


def test_our_checkpoint_loads_fully_when_injected_first():
    """Resume order: build -> inject -> load. 'Transferred N/N', no silent partial load."""
    cfg = ModulesConfig(der=True, enhancenet=True)
    saved = build()
    inject(saved, cfg, imgsz=IMGSZ)
    with torch.no_grad():  # make the module weights distinctive
        for m in saved.modules():
            if isinstance(m, (DER, EnhanceNet)):
                for p in m.parameters():
                    p.normal_(0, 0.5)

    fresh = build()
    inject(fresh, cfg, imgsz=IMGSZ)
    result = fresh.load_state_dict(saved.state_dict(), strict=True)
    assert not result.missing_keys, result.missing_keys
    assert not result.unexpected_keys, result.unexpected_keys

    for (ka, va), (kb, vb) in zip(
        sorted(fresh.state_dict().items()), sorted(saved.state_dict().items()), strict=True
    ):
        assert ka == kb
        assert torch.equal(va, vb), ka


def test_loading_our_checkpoint_without_injecting_first_fails_loudly():
    """The failure mode gotcha 3 warns about must not be silent."""
    cfg = ModulesConfig(der=True, enhancenet=True)
    saved = build()
    inject(saved, cfg, imgsz=IMGSZ)

    with pytest.raises(RuntimeError):
        build().load_state_dict(saved.state_dict(), strict=True)


def test_insert_at_out_of_range_is_rejected():
    with pytest.raises(IndexError, match="insert_at"):
        find_channels(build(), (99,), imgsz=IMGSZ)
