"""Differentiable per-batch ``s_dark`` for severity conditioning (E3, spec 7.5).

Extension E3 asks whether telling the routers *how dark the image is* helps, instead of
making them infer it from pooled features. The scalar has to be computed **inside the
model** so it survives ONNX export and needs no extra input tensor at serving time.

This must agree with the NumPy implementation in ``yolo_der.data.severity`` to within
1e-5 (spec 7.7 item 9) -- the analysis joins per-image routing weights against the CSV
those functions produced, so a mismatch would silently corrupt every E2/E3 correlation.

Channel order: Ultralytics' ``Format._format_img`` flips BGR to RGB whenever
``random.uniform(0, 1) > self.bgr``, and ``bgr`` defaults to 0.0, so batches are **RGB**.
``YoloDERTrainer`` asserts ``bgr == 0`` for exactly this reason -- with the BGR
augmentation on, the luma weights below would be applied to the wrong channels.
"""

from __future__ import annotations

import torch

# BT.601 luma weights for R, G, B -- identical to data/severity.py's BT601 constant.
_LUMA = (0.299, 0.587, 0.114)
TAU = 0.3  # paper Eq. 10 darkness threshold


def luma(x: torch.Tensor) -> torch.Tensor:
    """Return the luma channel of an RGB batch.

    Args:
        x: ``(B, 3, H, W)`` with values in ``[0, 1]``.

    Returns:
        ``(B, 1, H, W)`` luma.
    """
    if x.shape[1] != 3:
        raise ValueError(f"expected 3 channels (RGB), got shape {tuple(x.shape)}")
    weights = torch.tensor(_LUMA, dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def mean_luma(x: torch.Tensor) -> torch.Tensor:
    """Return ``mu_Y``, the per-image mean luma, of shape ``(B, 1)``."""
    return luma(x).mean(dim=(2, 3))


def s_dark(x: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """Paper Eq. 10: ``s_dark = max(0, (tau - mu_Y) / tau)``, shape ``(B, 1)``.

    ``relu`` provides the ``max(0, .)`` clamp, which also keeps the function
    differentiable almost everywhere -- the gradient is simply zero for images already
    brighter than ``tau``, which is the behaviour we want: those images carry no darkness
    signal to learn from.
    """
    return torch.relu((tau - mean_luma(x)) / tau)
