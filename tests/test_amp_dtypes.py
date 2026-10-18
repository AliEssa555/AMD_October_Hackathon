"""Mixed-precision dtype contracts for DER and EnhanceNet (spec gotcha 13).

Every test here exists because of one failure: all six variant runs on the MI300X died
within 45 seconds with::

    RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half
      der.py in route ->  logits = self.router(z)

Ultralytics validates with ``model.half()`` (``args.half`` defaults to True on CUDA and
ROCm), so our parameters became Half while :func:`fp32_math` kept casting activations to
Float. No earlier test could have caught it: AMP never engages on CPU, so every local
smoke run exercised the fp32-everywhere path only.

The reproduction turns out not to need a GPU at all -- ``model.half()`` on CPU is enough,
which is what makes these tests worth having rather than a note in the docs.
"""

import pytest
import torch

from yolo_der.config import ModulesConfig
from yolo_der.models.common import apply_in, parameter_dtype
from yolo_der.models.der import DER
from yolo_der.models.enhancenet import EnhanceNet

pytest.importorskip("ultralytics")

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from yolo_der.models.inject import inject  # noqa: E402

IMGSZ = 448
CHANNELS = 64


def sample(channels: int = CHANNELS, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(2, channels, 28, 28, dtype=dtype).uniform_(0, 1)


# ------------------------------------------------------------------ helpers ----


def test_parameter_dtype_reports_the_modules_own_dtype():
    m = DER(channels=CHANNELS, num_paths=4)
    assert parameter_dtype(m) is torch.float32
    assert parameter_dtype(m.half()) is torch.float16


def test_parameter_dtype_falls_back_to_buffers_then_float32():
    """A parameter-free module must not break the helper."""

    class OnlyBuffer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.zeros(2, dtype=torch.float16))

    assert parameter_dtype(OnlyBuffer()) is torch.float16
    assert parameter_dtype(torch.nn.ReLU()) is torch.float32


def test_apply_in_returns_fp32_whatever_the_parameters_are():
    """The maths around the call stays fp32; only the parameter boundary moves."""
    linear = torch.nn.Linear(8, 4)
    x = torch.zeros(2, 8).uniform_(0, 1)
    assert apply_in(linear, x).dtype is torch.float32
    assert apply_in(linear.half(), x).dtype is torch.float32


def test_apply_in_is_a_no_op_numerically_in_fp32():
    linear = torch.nn.Linear(8, 4)
    x = torch.zeros(2, 8).uniform_(0, 1)
    assert torch.allclose(apply_in(linear, x), linear(x), atol=1e-6)


# -------------------------------------------------------- the halved modules ----


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("DER", lambda: DER(channels=CHANNELS, num_paths=4)),
        ("EnhanceNet", lambda: EnhanceNet(channels=CHANNELS)),
        ("DER+cond", lambda: DER(channels=CHANNELS, num_paths=4, cond_dim=2)),
        ("EnhanceNet+cond", lambda: EnhanceNet(channels=CHANNELS, cond_dim=2)),
    ],
)
def test_halved_module_runs(name, build):
    """This is the exact failure: Ultralytics halves the model, then validates."""
    module = build().eval().half()
    cond = torch.zeros(2, 2).uniform_(0, 1) if "cond" in name else None
    with torch.no_grad():
        out = (
            module(sample(dtype=torch.float16), cond)
            if cond is not None
            else module(sample(dtype=torch.float16))
        )
    assert out.dtype is torch.float16, "the module must hand back the dtype it was given"
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_der_routing_weights_are_always_fp32(dtype):
    """The entropy and load-balance losses (E1) consume these, so precision matters."""
    der = DER(channels=CHANNELS, num_paths=4).eval().to(dtype)
    with torch.no_grad():
        weights = der.route(sample(dtype=dtype))
    assert weights.dtype is torch.float32
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_enhancenet_gate_is_always_fp32_and_normalised(dtype):
    enh = EnhanceNet(channels=CHANNELS).eval().to(dtype)
    with torch.no_grad():
        beta = enh.gate(sample(dtype=dtype))
    assert beta.dtype is torch.float32
    assert beta.shape == (2, 5)
    assert torch.allclose(beta.sum(dim=1), torch.ones(2), atol=1e-5)


def test_identity_at_init_survives_halving():
    """ReZero means lambda=0, so a freshly built module must pass its input through.

    If halving broke this, every variant would start from a disturbed backbone rather
    than from pretrained YOLOv8n (spec 7.4).
    """
    x = sample(dtype=torch.float16)
    for module in (DER(channels=CHANNELS, num_paths=4), EnhanceNet(channels=CHANNELS)):
        with torch.no_grad():
            out = module.eval().half()(x)
        assert torch.allclose(out, x, atol=1e-3), f"{type(module).__name__} is not identity"


# ------------------------------------------------- the real Ultralytics path ----


@pytest.mark.parametrize("combine", ["sequential", "der_routes_enh"])
def test_injected_detection_model_validates_when_halved(combine):
    """End to end, the way Ultralytics actually does it on a GPU.

    `BaseValidator.__call__` runs `model.half() if self.args.half else model.float()`,
    and `args.half` is True by default on CUDA/ROCm -- so this is not a synthetic case,
    it is the validation pass of every run.
    """
    model = DetectionModel("yolov8n.yaml", nc=12, verbose=False)
    # der_routes_enh needs der_paths=5: DER's router *is* EnhanceNet's branch gate there,
    # so the path count has to equal the five branches (config.py enforces it).
    paths = 5 if combine == "der_routes_enh" else 4
    inject(
        model,
        ModulesConfig(der=True, enhancenet=True, combine=combine, der_paths=paths),
        imgsz=IMGSZ,
    )
    model.eval().half()

    x = torch.zeros(1, 3, IMGSZ, IMGSZ, dtype=torch.float16).uniform_(0, 1)
    with torch.no_grad():
        out = model(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out
    assert torch.isfinite(pred).all()


def test_injected_detection_model_trains_under_autocast():
    """The other half of AMP: autocast on, parameters still fp32.

    bfloat16 on CPU stands in for fp16 on the MI300X; what is being checked is that the
    fp32 regions interoperate with an autocast context at all, in both directions.
    """
    model = DetectionModel("yolov8n.yaml", nc=12, verbose=False)
    inject(model, ModulesConfig(der=True, enhancenet=True), imgsz=IMGSZ)
    model.train()

    x = torch.zeros(1, 3, IMGSZ, IMGSZ).uniform_(0, 1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out
    assert torch.isfinite(pred.float()).all()
    assert all(p.dtype is torch.float32 for p in model.parameters()), (
        "autocast must not have changed the parameters themselves"
    )
