"""DER -- Dynamic Enhancement Routing (paper Eq. 1-2, Fig. 2). Hand-written (spec 7.1).

The paper's maths, line by line::

    Z      = GAP(F)                          # (B, C)                       Eq. 1
    logits = W2 @ ReLU(W1 @ Z + b1) + b2     # (B, N)                       Eq. 1
    o_hat  = softmax(logits / T)             # routing weights, T = 1       Eq. 2
    F_der  = sum_n o_hat[n] * P_n(F)         # SOFT routing: every path runs
    out    = F + lambda_der * F_der          # lambda_der learnable, init 0

**The paper defines the router but never defines the N "enhancement paths".** Our
interpretation -- documented here and in the README as ours, not the paper's -- is N = 4
lightweight residual paths, each returning a *delta* to be added to F:

* **0** ``0`` -- the "do nothing" option, so the router can abstain on already-good
  features instead of being forced to enhance them.
* **1** DW 3x3 -> BN -> SiLU -> PW 1x1 -- local refinement.
* **2** DW 3x3 dilation 2 -> BN -> SiLU -> PW 1x1 -- wider context.
* **3** ``F * s - F`` with ``s = 2*sigmoid(MLP(GAP(F)))`` -- global channel recalibration.
* **4** DW 5x5 -> BN -> SiLU -> PW 1x1 -- only built when ``num_paths=5``.

Two deliberate choices worth knowing about:

* Path 3 is written in spec 7.1 as ``F * sigmoid(MLP(GAP(F))) - F``. Under the zero-init
  rule that gives ``sigmoid(0) = 0.5``, i.e. a delta of ``-0.5 * F`` -- not neutral. We use
  ``2 * sigmoid(...)`` instead, so ``s = 1`` and the delta is exactly **0** at
  initialisation, which is the point of the identity-at-init rule in spec Section 7. The
  cost is that the path may amplify a channel as well as suppress it.
* Path 4 exists only so ``num_paths=5`` is self-consistent. The ``der_routes_enh`` variant
  that needs 5 weights uses DER as a *router only* (``router_only=True``) and builds no
  paths at all, so path 4 is normally never instantiated.

The Fig. 2 extras (an SE block, a conv, an FC and BatchNorm before routing) are **not**
implemented in v1, per spec 7.1.
"""

from __future__ import annotations

import torch
from torch import nn

from yolo_der.models.common import (
    RecordingMixin,
    fp32_math,
    gap,
    hidden_width,
    two_layer_head,
)


class _ConvPath(nn.Module):
    """Depthwise-separable path: DW k x k (optionally dilated) -> BN -> SiLU -> PW 1x1.

    Depthwise-separable keeps the cost near ``C*k^2 + C^2`` instead of ``C^2*k^2``; the
    paper's "< 10% extra parameters" claim only survives if every path is cheap.
    """

    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2  # 'same' spatial size, for odd kernels
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return this path's delta for ``x``."""
        return self.pw(self.act(self.bn(self.dw(x))))


class _SEPath(nn.Module):
    """Channel recalibration: ``F * s - F`` with ``s = 2*sigmoid(MLP(GAP(F)))`` in (0, 2).

    ``s`` is exactly 1 at initialisation (zero-init head), so the delta starts at 0.
    Squeeze-and-excitation re-weights whole channels, making this the only one of the four
    paths that acts globally rather than locally.
    """

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.head = two_layer_head(channels, hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return this path's delta for ``x``."""
        s = 2.0 * torch.sigmoid(self.head(gap(x)))  # (B, C), neutral at 1
        return x * s[:, :, None, None] - x


class DER(nn.Module, RecordingMixin):
    """Dynamic Enhancement Routing (paper Eq. 1-2).

    Args:
        channels: ``C`` of the feature map this module wraps.
        num_paths: ``N``, how many enhancement paths the router mixes (2 to 5).
        hidden: router hidden width; defaults to ``max(C // 8, 16)``.
        temperature: softmax temperature ``T`` in Eq. 2. ``T < 1`` sharpens routing,
            ``T > 1`` flattens it. Left at 1 for every reported run.
        cond_dim: width of a conditioning vector concatenated to ``Z`` before ``W1``.
            Extension E3 passes the measured ``s_dark``, so ``cond_dim=1``.
        router_only: build the router but no paths, for the ``der_routes_enh`` variant
            where DER's weights drive EnhanceNet's branch gate.
        collect_for_loss: keep the (non-detached) routing weights on
            ``self.routing_for_loss`` so the E1 regularisers can consume them. Off by
            default: holding the tensor also holds its slice of the autograd graph.

    Shape:
        ``(B, C, H, W) -> (B, C, H, W)``. The output is **exactly** ``x`` at initialisation,
        because ``lambda_der`` starts at 0 (ReZero-style).
    """

    def __init__(
        self,
        channels: int,
        num_paths: int = 4,
        hidden: int | None = None,
        temperature: float = 1.0,
        cond_dim: int = 0,
        *,
        router_only: bool = False,
        collect_for_loss: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        RecordingMixin.__init__(self)

        if not 2 <= num_paths <= 5:
            raise ValueError(f"num_paths must be in 2..5 (paths 0-4 are defined), got {num_paths}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.channels = channels
        self.num_paths = num_paths
        self.temperature = float(temperature)
        self.cond_dim = int(cond_dim)
        self.router_only = bool(router_only)
        self.collect_for_loss = bool(collect_for_loss)

        d = hidden_width(channels) if hidden is None else hidden
        # Eq. 1: W1 @ [Z ; cond] + b1 -> ReLU -> W2 @ . + b2. The second Linear is
        # zero-initialised, so the logits start at 0 and Eq. 2 starts perfectly uniform.
        self.router = two_layer_head(channels + self.cond_dim, d, num_paths)

        # Path 0 is the zero path and owns no parameters, so it is not a submodule.
        # self.paths holds paths 1..N-1, hence the start=1 offset in forward().
        if router_only:
            self.paths = nn.ModuleList()
        else:
            built: list[nn.Module] = [_ConvPath(channels, kernel=3, dilation=1)]
            if num_paths > 2:
                built.append(_ConvPath(channels, kernel=3, dilation=2))
            if num_paths > 3:
                built.append(_SEPath(channels, d))
            if num_paths > 4:
                built.append(_ConvPath(channels, kernel=5, dilation=1))
            self.paths = nn.ModuleList(built)

        # ReZero: the module is a no-op until training decides otherwise. This is what
        # lets us inject into pretrained YOLOv8n without disturbing it at step 0.
        self.lambda_der = nn.Parameter(torch.zeros(1))

    def extra_repr(self) -> str:
        """Readable summary for ``print(model)``."""
        return (
            f"channels={self.channels}, num_paths={self.num_paths}, T={self.temperature}, "
            f"cond_dim={self.cond_dim}, router_only={self.router_only}"
        )

    def route(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Paper Eq. 1-2: return routing weights ``o_hat`` of shape ``(B, N)``.

        Separated from :meth:`forward` so the ``der_routes_enh`` variant can use DER purely
        as a router for EnhanceNet's branch gate. Always evaluated in fp32.
        """
        with fp32_math(x) as xf:
            z = gap(xf)  # Eq. 1: Z = GAP(F)
            if self.cond_dim:
                if cond is None:
                    raise ValueError(f"DER was built with cond_dim={self.cond_dim} but cond=None")
                z = torch.cat([z, cond.to(z.dtype).reshape(z.shape[0], -1)], dim=1)
            logits = self.router(z)  # Eq. 1
            return torch.softmax(logits / self.temperature, dim=1)  # Eq. 2

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Soft-route ``x`` through every path and add the mixed delta back."""
        self._reset_capture()
        if self.router_only:
            raise RuntimeError(
                "DER(router_only=True) builds no paths; call .route(x) and pass the weights "
                "to EnhanceNet(external_gate=True)."
            )

        weights = self.route(x, cond)  # (B, N), fp32
        with fp32_math(x) as xf:
            # Soft routing (Eq. 2 -> F_der): every path runs, outputs are mixed by o_hat.
            # Path 0's contribution is identically zero, so it is skipped structurally --
            # that is not a branch on a tensor value, so the graph stays ONNX-traceable.
            delta = torch.zeros_like(xf)
            for n, path in enumerate(self.paths, start=1):
                delta = delta + weights[:, n, None, None, None] * path(xf)
            out = xf + self.lambda_der.float() * delta  # ReZero residual

        self._capture(routing=weights)
        if self.collect_for_loss:
            self.routing_for_loss = weights
        return out.to(x.dtype)
