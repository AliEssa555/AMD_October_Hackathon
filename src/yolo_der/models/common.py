"""Shared building blocks for DER and EnhanceNet.

Not in the spec's file list: it exists so the initialisation rules of spec Section 7 are
written **once** and cannot drift between the two modules. Those rules are:

* lightweight MLP hidden width ``d = max(C // 8, 16)``;
* Xavier-uniform on hidden ``Linear`` layers (paper Sec. 4.1);
* **zero-init the last ``Linear`` of every parameter head and gate**, so every predicted
  parameter starts at its neutral value and every gate starts uniform;
* module maths runs in fp32 even under AMP -- ``pow``, ``logit`` and division by ``sigma``
  are unstable in fp16 (spec Section 7, gotcha 8).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import nn


def hidden_width(channels: int) -> int:
    """Return the lightweight-MLP hidden width ``d = max(C // 8, 16)`` (spec Section 7)."""
    return max(channels // 8, 16)


def gap(x: torch.Tensor) -> torch.Tensor:
    """Global average pooling: ``(B, C, H, W) -> (B, C)``.

    Paper Eq. 1's ``Z``. The claim is that this summary carries brightness, contrast and
    haze cues; whether it really does is what extension E2 measures.
    """
    return x.mean(dim=(2, 3))


def two_layer_head(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """``Linear -> ReLU -> Linear`` with the spec's initialisation.

    The second ``Linear`` is zero-initialised, so the head's output starts at exactly zero
    and whatever bounded function consumes it starts at its neutral value.
    """
    head = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )
    nn.init.xavier_uniform_(head[0].weight)
    nn.init.zeros_(head[0].bias)
    nn.init.zeros_(head[2].weight)
    nn.init.zeros_(head[2].bias)
    return head


@contextmanager
def fp32_math(x: torch.Tensor) -> Iterator[torch.Tensor]:
    """Run the enclosed maths in fp32 with autocast off, yielding ``x`` as fp32.

    Usage::

        with fp32_math(x) as xf:
            out = ...            # all fp32
        return out.to(x.dtype)   # caller restores the incoming dtype

    ``device_type`` is taken from the tensor so the same code works on CPU, CUDA and ROCm
    (HIP is exposed through the CUDA API, so ``x.device.type`` is ``"cuda"`` on MI300X).
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        yield x.float()


def gaussian_kernel2d(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Return a normalised 2-D Gaussian kernel of shape ``(size, size)``.

    Used by EnhanceNet's detail branch (paper Eq. 5) as a fixed, non-learnable blur. It is
    registered as a buffer, so it has no gradient and travels with the state dict.
    """
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def depthwise_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a fixed 2-D kernel to every channel independently, with reflect padding.

    Reflect padding keeps the blur from darkening the borders, which a zero pad would do
    and which the detail branch would then "sharpen" into an edge artefact.
    """
    channels = x.shape[1]
    k = kernel.to(dtype=x.dtype, device=x.device)
    weight = k.expand(channels, 1, *k.shape)
    pad = k.shape[-1] // 2
    x = nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    return nn.functional.conv2d(x, weight, groups=channels)


class RecordingMixin:
    """Opt-in capture of routing/gate weights for the E1/E2 analyses (spec Section 7).

    ``record`` is off during training. When on, the forward stores **detached** tensors on
    ``self.last``. Recording must never change the output and no code path may branch on a
    tensor's *value*, so the module stays traceable for ONNX export (gotcha 19).
    """

    def __init__(self) -> None:
        self.record: bool = False
        self.last: dict[str, torch.Tensor] = {}
        # Set (not detached) when an extension loss needs the weights; cleared each forward
        # so a stale tensor never pins a freed autograd graph.
        self.routing_for_loss: torch.Tensor | None = None

    def _reset_capture(self) -> None:
        self.routing_for_loss = None
        if self.record:
            self.last = {}

    def _capture(self, **tensors: torch.Tensor) -> None:
        if self.record:
            self.last.update({k: v.detach() for k, v in tensors.items()})


def set_recording(module: nn.Module, enabled: bool) -> int:
    """Switch ``record`` on or off for every recording submodule; return how many."""
    n = 0
    for m in module.modules():
        if isinstance(m, RecordingMixin):
            m.record = enabled
            if not enabled:
                m.last = {}
            n += 1
    return n


def iter_recording(module: nn.Module) -> Iterator[tuple[str, RecordingMixin]]:
    """Yield ``(qualified_name, submodule)`` for every recording submodule."""
    for name, m in module.named_modules():
        if isinstance(m, RecordingMixin):
            yield name, m


def count_added_parameters(model: nn.Module, kinds: tuple[type, ...]) -> tuple[int, int]:
    """Return ``(total_params, params_belonging_to kinds)``.

    The paper claims the modules add < 10% of the detector's parameters; this is how we
    check that claim rather than restating it (spec Phase 3 acceptance).
    """
    total = sum(p.numel() for p in model.parameters())
    added = 0
    for m in model.modules():
        if isinstance(m, kinds):
            added += sum(p.numel() for p in m.parameters())
    return total, added


__all__ = [
    "RecordingMixin",
    "count_added_parameters",
    "depthwise_blur",
    "fp32_math",
    "gap",
    "gaussian_kernel2d",
    "hidden_width",
    "iter_recording",
    "set_recording",
    "two_layer_head",
]

# Kept for callers that want a typed alias of the bounded-parameter helpers.
BoundedFn = Callable[[torch.Tensor], torch.Tensor]
