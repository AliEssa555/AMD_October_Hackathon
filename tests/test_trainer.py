"""Trainer and CLI tests (spec 9.2).

Every test here pins a bug that actually bit during the first end-to-end run, which is the
only reason to trust them: they are regressions, not guesses.
"""

import pytest
import torch

pytest.importorskip("ultralytics")

from ultralytics.cfg import get_cfg  # noqa: E402

from yolo_der.config import ModulesConfig  # noqa: E402
from yolo_der.models.inject import inject  # noqa: E402
from yolo_der.models.yolo_der_model import YoloDERModel  # noqa: E402
from yolo_der.trainer import NO_DECAY_SUFFIXES, YoloDERTrainer  # noqa: E402

IMGSZ = 448
NC = 12


def build_model(*, injected: bool, cfg: ModulesConfig | None = None) -> YoloDERModel:
    model = YoloDERModel(cfg="yolov8n.yaml", nc=NC, der_cfg=None, imgsz=IMGSZ, verbose=False)
    if injected:
        inject(model, cfg or ModulesConfig(der=True, enhancenet=True), imgsz=IMGSZ)
    return model


# --------------------------------------------------------------- weight transfer


def test_class_head_prefix_points_at_the_detect_head():
    """nc-dependent keys live under this prefix; everything else must transfer."""
    model = build_model(injected=False)
    assert YoloDERTrainer._class_head_prefix(model) == "model.22.cv3."


def test_nc_mismatch_in_the_class_head_is_accepted():
    """yolov8n.pt is COCO (nc=80) and ExDark is nc=12, so cv3 legitimately cannot load.

    The first smoke run failed here: the guard counted these 36 keys as a partial transfer
    and aborted a perfectly correct load.
    """
    model = build_model(injected=False)
    incoming = {k: v.clone() for k, v in model.state_dict().items()}
    head = YoloDERTrainer._class_head_prefix(model)
    # Simulate an 80-class head: wrong shape, right names.
    for key in [k for k in incoming if k.startswith(head)]:
        incoming[key] = torch.zeros(80, *incoming[key].shape[1:])

    model.load = lambda weights: None  # bypass Ultralytics' own loader
    YoloDERTrainer._load_and_verify(model, incoming)  # must not raise


def test_keys_absent_by_name_are_fatal():
    """This is gotcha 3: injection renames model.4.cv1 -> model.4.layer.cv1."""
    model = build_model(injected=True)
    plain = build_model(injected=False).state_dict()

    model.load = lambda weights: None
    with pytest.raises(RuntimeError, match="absent by name"):
        YoloDERTrainer._load_and_verify(model, plain)


def test_shape_mismatch_outside_the_class_head_is_fatal():
    """A backbone shape change is never legitimate, unlike the nc-dependent head."""
    model = build_model(injected=False)
    incoming = {k: v.clone() for k, v in model.state_dict().items()}
    incoming["model.0.conv.weight"] = torch.zeros(7, 3, 3, 3)

    model.load = lambda weights: None
    with pytest.raises(RuntimeError, match="outside the class head"):
        YoloDERTrainer._load_and_verify(model, incoming)


def test_an_exact_state_dict_loads_without_complaint():
    model = build_model(injected=True)
    model.load = lambda weights: None
    YoloDERTrainer._load_and_verify(model, dict(model.state_dict()))


# ------------------------------------------------------------------ loss names


class _LossNamesHarness(YoloDERTrainer):
    """Exercise the loss-name logic without constructing a real trainer."""

    def __init__(self, model):  # noqa: D107 - test double
        self.model = model
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")


def test_ent_loss_is_added_only_when_a_regulariser_is_on():
    """der_enh has both weights at 0, so it must NOT gain a column."""
    plain = build_model(injected=True, cfg=ModulesConfig(der=True, enhancenet=True))
    harness = _LossNamesHarness(plain)
    harness._extend_loss_names()
    assert harness.loss_names == ("box_loss", "cls_loss", "dfl_loss")

    reg = build_model(
        injected=True,
        cfg=ModulesConfig(der=True, enhancenet=True, entropy_reg=0.01, load_balance=0.01),
    )
    reg._entropy_weight, reg._load_balance_weight = 0.01, 0.01
    harness = _LossNamesHarness(reg)
    harness._extend_loss_names()
    assert harness.loss_names == ("box_loss", "cls_loss", "dfl_loss", "ent_loss")


def test_extending_loss_names_is_idempotent():
    """get_validator and set_model_attributes both call it; one column, not two."""
    reg = build_model(
        injected=True,
        cfg=ModulesConfig(der=True, enhancenet=True, entropy_reg=0.01),
    )
    reg._entropy_weight, reg._load_balance_weight = 0.01, 0.0
    harness = _LossNamesHarness(reg)
    harness._extend_loss_names()
    harness._extend_loss_names()
    assert harness.loss_names.count("ent_loss") == 1


def test_get_validator_is_overridden_because_it_resets_loss_names():
    """Ultralytics' DetectionTrainer.get_validator reassigns loss_names unconditionally.

    _setup_train calls it *after* set_model_attributes, so without this override the extra
    column vanishes and every metric after it shifts one position.
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    assert "get_validator" in vars(YoloDERTrainer), "override removed -- ent_loss will vanish"
    assert "loss_names" in DetectionTrainer.get_validator.__code__.co_names


# ------------------------------------------------------------------ config path


def test_der_cfg_is_not_a_valid_yolo_argument():
    """Documents why der_cfg is an instance attribute, not an args entry.

    Putting it on trainer.args kills the run when DetectionValidator re-validates a copy
    of those args through get_cfg.
    """
    with pytest.raises(SyntaxError, match="not a valid YOLO argument"):
        get_cfg(overrides={"der_cfg": "anything", "model": "yolov8n.yaml"})


def test_der_cfg_defaults_to_none_on_the_class():
    assert YoloDERTrainer.der_cfg is None


def test_no_decay_suffixes_match_the_rezero_parameter_names():
    """If a module renames its scale, the optimiser fix silently stops applying."""
    model = build_model(injected=True)
    names = [n for n, _ in model.named_parameters() if n.endswith(NO_DECAY_SUFFIXES)]
    # 3 insertion points x (lambda_der + lambda_enh)
    assert len(names) == 6, names
