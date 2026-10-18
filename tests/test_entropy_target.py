"""The E1 regulariser must be attributable to one module (spec 10.2).

The first entropy sweep pushed every routing distribution at once, and EnhanceNet's
branch gate collapsed alongside DER's router -- normalised entropy 0.80 -> 0.47 -> 0.005
as the weight went 0.1 -> 1.0 -> 5.0. The -0.15 mAP50 that came with it therefore could
not be blamed on decisive *routing*: it may have been the loss of a gate that was working.
`entropy_target` separates them, and these tests pin that it really does.
"""

import pytest
import torch

from yolo_der.config import ModulesConfig
from yolo_der.losses import collect_routing, regularisers

pytest.importorskip("ultralytics")

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from yolo_der.models.inject import inject  # noqa: E402

IMGSZ = 448


def injected(**kw):
    """An injected model that has just run a forward, so routing_for_loss is populated."""
    model = DetectionModel("yolov8n.yaml", nc=12, verbose=False)
    cfg = ModulesConfig(der=True, enhancenet=True, entropy_reg=1.0, load_balance=1.0, **kw)
    inject(model, cfg, imgsz=IMGSZ)
    model.train()
    model(torch.zeros(1, 3, IMGSZ, IMGSZ).uniform_(0, 1))
    return model


def test_all_collects_both_module_kinds():
    model = injected()
    assert len(collect_routing(model, "all")) == len(collect_routing(model, "der")) + len(
        collect_routing(model, "enhancenet")
    )


def test_der_target_collects_only_routers():
    """The point of the whole change: push the router without touching the gate."""
    model = injected()
    der_only = collect_routing(model, "der")
    assert der_only, "no DER routing weights collected"
    # DER was built with 4 paths, EnhanceNet's gate has 5 branches.
    assert all(w.shape[-1] == 4 for w in der_only), [tuple(w.shape) for w in der_only]


def test_enhancenet_target_collects_only_gates():
    model = injected()
    enh_only = collect_routing(model, "enhancenet")
    assert enh_only
    assert all(w.shape[-1] == 5 for w in enh_only), [tuple(w.shape) for w in enh_only]


def test_targets_partition_the_modules():
    """Nothing counted twice, nothing dropped."""
    model = injected()
    ids = lambda ws: {id(w) for w in ws}  # noqa: E731
    der, enh, allw = (ids(collect_routing(model, t)) for t in ("der", "enhancenet", "all"))
    assert der & enh == set(), "a module was collected by both targets"
    assert der | enh == allw


def test_der_target_gives_a_different_loss_than_all():
    """If the filter did nothing, the regulariser value would be unchanged."""
    model = injected()
    total_all, _ = regularisers(model, entropy_weight=1.0, load_balance_weight=1.0, target="all")
    total_der, _ = regularisers(model, entropy_weight=1.0, load_balance_weight=1.0, target="der")
    assert not torch.isclose(total_all, total_der), (
        f"targeting changed nothing: all={total_all.item()} der={total_der.item()}"
    )


def test_der_only_entropy_matches_hand_computation():
    """A 4-path uniform router has entropy log(4); the DER-only term must be that."""
    import math

    model = injected()
    _, parts = regularisers(model, entropy_weight=1.0, load_balance_weight=0.0, target="der")
    # At init the router is uniform, so entropy sits at log(N) for N=4.
    assert parts["routing_entropy"] == pytest.approx(math.log(4), abs=0.02), parts


def test_config_accepts_and_defaults_the_target():
    assert ModulesConfig().entropy_target == "all"
    assert ModulesConfig(entropy_target="der").entropy_target == "der"


def test_model_forwards_the_target_from_config():
    """The path config -> model attribute -> regularisers must actually connect."""
    from yolo_der.models.yolo_der_model import YoloDERModel

    m = YoloDERModel(
        cfg="yolov8n.yaml",
        nc=12,
        der_cfg=ModulesConfig(der=True, enhancenet=True, entropy_reg=1.0, entropy_target="der"),
        imgsz=IMGSZ,
        verbose=False,
    )
    assert m._entropy_target == "der"


def test_unknown_target_raises_rather_than_pushing_everything():
    """A typo must not silently run a different experiment from the one configured.

    The first version of this returned [] for an unknown target, which the test caught as
    still collecting all six modules -- a fall-through. Silently pushing every module is
    precisely the failure `entropy_target` exists to prevent, so it raises.
    """
    model = injected()
    with pytest.raises(ValueError, match="entropy_target must be one of"):
        collect_routing(model, "typo")
