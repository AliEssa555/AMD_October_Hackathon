"""Checkpoint round-trip (spec 7.7 item 8, gotchas 3-4).

Ultralytics checkpoints are **pickled whole models**, not just state dicts, so loading one
re-imports ``yolo_der``. If the package is not importable, or a class moved, the checkpoint
is unreadable. These tests pin the two things that matter: the pickle survives a round
trip, and every key transfers.
"""

import pytest
import torch

from yolo_der.config import ModulesConfig
from yolo_der.models.der import DER
from yolo_der.models.enhancenet import EnhanceNet
from yolo_der.models.inject import inject

pytest.importorskip("ultralytics")

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

IMGSZ = 448
CFG = ModulesConfig(der=True, enhancenet=True)


def build_injected() -> DetectionModel:
    model = DetectionModel("yolov8n.yaml", nc=12, verbose=False)
    inject(model, CFG, imgsz=IMGSZ)
    model.eval()
    return model


def randomise_modules(model: DetectionModel) -> None:
    """Give our modules distinctive weights so a partial load is detectable."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (DER, EnhanceNet)):
                for p in m.parameters():
                    p.normal_(0, 0.5)


def test_pickled_whole_model_round_trips(tmp_path):
    """This is the format Ultralytics actually writes to last.pt / best.pt."""
    model = build_injected()
    randomise_modules(model)
    with torch.no_grad():
        model.model[4].enhance.der.lambda_der.fill_(0.42)

    path = tmp_path / "last.pt"
    torch.save({"model": model}, path)
    reloaded = torch.load(path, weights_only=False)["model"]

    assert isinstance(reloaded, DetectionModel)
    assert reloaded.model[4].enhance.der.lambda_der.item() == pytest.approx(0.42)

    x = torch.zeros(1, 3, IMGSZ, IMGSZ).uniform_(0, 1)
    with torch.no_grad():
        assert torch.allclose(model(x)[0], reloaded(x)[0], atol=1e-6)


def test_every_key_transfers_between_two_injected_models(tmp_path):
    """The 'Transferred N/N' assertion from spec 7.4, done on the state dict."""
    saved = build_injected()
    randomise_modules(saved)
    path = tmp_path / "ckpt.pt"
    torch.save(saved.state_dict(), path)

    fresh = build_injected()
    state = torch.load(path, weights_only=True)
    result = fresh.load_state_dict(state, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    transferred = sum(1 for k, v in fresh.state_dict().items() if torch.equal(v, state[k]))
    assert transferred == len(state), f"Transferred {transferred}/{len(state)}"


def test_buffers_survive_the_round_trip():
    """The detail branch's fixed Gaussian is a buffer and must travel with the weights."""
    model = build_injected()
    state = model.state_dict()
    blur_keys = [k for k in state if k.endswith("blur_kernel")]
    assert len(blur_keys) == len(CFG.insert_at)

    fresh = build_injected()
    fresh.load_state_dict(state, strict=True)
    for k in blur_keys:
        assert torch.equal(fresh.state_dict()[k], state[k])


def test_forward_is_reproducible_after_reload(tmp_path):
    """Guards against a module whose behaviour depends on un-saved state."""
    model = build_injected()
    randomise_modules(model)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, DER):
                m.lambda_der.fill_(0.2)
            elif isinstance(m, EnhanceNet):
                m.lambda_enh.fill_(0.2)

    x = torch.zeros(2, 3, IMGSZ, IMGSZ).uniform_(0, 1)
    with torch.no_grad():
        expected = model(x)[0]

    path = tmp_path / "m.pt"
    torch.save({"model": model}, path)
    reloaded = torch.load(path, weights_only=False)["model"].eval()
    with torch.no_grad():
        assert torch.allclose(reloaded(x)[0], expected, atol=1e-6)
