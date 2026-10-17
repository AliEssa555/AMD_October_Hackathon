"""EnhanceNet -- five differentiable enhancement branches (paper Eq. 3-9, spec 7.2).

A shared trunk reads the pooled feature summary once, then five heads predict the
parameters of five enhancement branches. A CBAM-inspired gate blends the branch outputs,
and the blend is added back through a learnable ``lambda_enh`` that starts at 0::

    h      = ReLU(W_t @ GAP(F) + b_t)              # shared trunk, width d
    beta   = softmax(W_g2 @ ReLU(W_g1 @ GAP(F)))   # branch gate, 5 weights      Eq. 8
    F_enh  = sum_i beta[i] * F_i                   # fusion                      Eq. 9
    out    = F + lambda_enh * F_enh                # lambda_enh learnable, init 0

``mu_c`` and ``sigma_c`` are the per-sample, per-channel spatial mean and standard
deviation of ``F``; ``eps = 1e-5``.

The five branches, with the bounded parameter each head predicts (every one neutral at
zero-init, so the predicted parameters start at their canonical values):

* **Tone** (Eq. 3) -- ``gamma = 3^tanh(a) in [1/3, 3]``, neutral 1.
* **Contrast** (Eq. 4) -- ``alpha = softplus(a)/ln2`` (neutral 1) and
  ``b = 0.1*tanh(a)*sigma`` (neutral 0).
* **Detail** (Eq. 5) -- ``kappa = 2*sigmoid(a) in (0, 2)``, neutral 1.
* **White balance** (Eq. 6) -- ``s = 1 + 0.5*tanh(a) in [0.5, 1.5]``, neutral 1.
* **Identity** (Eq. 7) -- no parameters; this is the branch that lets the gate opt out.

Note that "neutral" means the *parameter* starts at its canonical value, not that every
branch starts as the identity. The detail branch at ``kappa = 1`` really does sharpen --
sharpening is its job. Overall neutrality at initialisation comes from ``lambda_enh = 0``
and from the identity branch, not from the individual parameters.

Why the sigmoid/logit dance in the tone branch: ``x^gamma`` is only meaningful for inputs
in ``(0, 1)``, but feature maps are unbounded. So we squash each channel's distribution
into ``(0, 1)`` with a sigmoid of its z-score, apply the gamma curve there, and map back
with the logit. The result is a pure *tone remap of each channel's own distribution* --
monotonic, mean/scale-preserving at ``gamma = 1``, and free of the clipping that a naive
``clamp(F, 0, 1) ** gamma`` would cause. The ``clamp`` before the logit only keeps the
inverse finite; it bites beyond roughly +/-9 standard deviations.

Eq. 9 is implemented as written even though ``F_enh`` already contains ``F`` through the
identity branch, so ``out`` is ``F*(1 + lambda*beta_id) + ...``. That redundancy is the
paper's, and reproducing it faithfully matters more here than tidying it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from yolo_der.models.common import (
    RecordingMixin,
    depthwise_blur,
    fp32_math,
    gap,
    gaussian_kernel2d,
    hidden_width,
)

EPS = 1e-5
NUM_BRANCHES = 5
# Index order is fixed and is what the analysis plots and parquet files key on.
BRANCH_NAMES = ("tone", "contrast", "detail", "white_balance", "identity")


def _zero_head(in_dim: int, out_dim: int) -> nn.Linear:
    """A zero-initialised ``Linear``: its output starts at exactly 0 (spec Section 7)."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class EnhanceNet(nn.Module, RecordingMixin):
    """Five-branch feature enhancement with a learned soft gate (paper Eq. 3-9).

    Args:
        channels: ``C`` of the feature map this module wraps.
        hidden: trunk and gate hidden width; defaults to ``max(C // 8, 16)``.
        cond_dim: width of a conditioning vector concatenated to ``GAP(F)`` before the
            trunk and the gate. Extension E3 passes ``s_dark``, so ``cond_dim=1``.
        external_gate: skip the built-in gate and take ``beta`` from the caller. Used by
            the ``der_routes_enh`` variant, where DER's router supplies the branch weights.
        collect_for_loss: keep the (non-detached) gate weights on ``self.routing_for_loss``
            for the E1 regularisers.

    Shape:
        ``(B, C, H, W) -> (B, C, H, W)``, exactly ``x`` at initialisation because
        ``lambda_enh`` starts at 0.
    """

    def __init__(
        self,
        channels: int,
        hidden: int | None = None,
        cond_dim: int = 0,
        external_gate: bool = False,
        *,
        collect_for_loss: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        RecordingMixin.__init__(self)

        self.channels = channels
        self.cond_dim = int(cond_dim)
        self.external_gate = bool(external_gate)
        self.collect_for_loss = bool(collect_for_loss)

        d = hidden_width(channels) if hidden is None else hidden
        in_dim = channels + self.cond_dim

        # Shared trunk: h = ReLU(W_t @ GAP(F) + b_t). Xavier per paper Sec. 4.1.
        self.trunk = nn.Linear(in_dim, d)
        nn.init.xavier_uniform_(self.trunk.weight)
        nn.init.zeros_(self.trunk.bias)
        self.trunk_act = nn.ReLU(inplace=True)

        # Parameter heads, all zero-initialised -> every bounded parameter starts neutral.
        self.head_gamma = _zero_head(d, channels)  # Eq. 3, tone
        self.head_alpha = _zero_head(d, channels)  # Eq. 4, contrast gain
        self.head_bias = _zero_head(d, channels)  # Eq. 4, contrast offset
        self.head_kappa = _zero_head(d, 1)  # Eq. 5, detail strength (scalar)
        self.head_scale = _zero_head(d, channels)  # Eq. 6, white balance

        # Gate (Eq. 8), CBAM-inspired. The paper writes a generic sigma for the inner
        # non-linearity; we use ReLU and say so. Zero-init on the second layer means the
        # logits start at 0, so beta starts uniform at exactly 0.2 per branch.
        if not external_gate:
            self.gate_fc1 = nn.Linear(in_dim, d)
            nn.init.xavier_uniform_(self.gate_fc1.weight)
            nn.init.zeros_(self.gate_fc1.bias)
            self.gate_fc2 = _zero_head(d, NUM_BRANCHES)

        # Fixed 5x5 Gaussian for the detail branch: a buffer, so no gradient and it
        # travels with the state dict.
        self.register_buffer("blur_kernel", gaussian_kernel2d(size=5, sigma=1.0))

        # ReZero residual scale (Eq. 9).
        self.lambda_enh = nn.Parameter(torch.zeros(1))

    def extra_repr(self) -> str:
        """Readable summary for ``print(model)``."""
        return (
            f"channels={self.channels}, cond_dim={self.cond_dim}, "
            f"external_gate={self.external_gate}"
        )

    def _summary(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """Return ``[GAP(F) ; cond]``, the input shared by the trunk and the gate."""
        z = gap(x)
        if self.cond_dim:
            if cond is None:
                raise ValueError(f"EnhanceNet was built with cond_dim={self.cond_dim}, cond=None")
            z = torch.cat([z, cond.to(z.dtype).reshape(z.shape[0], -1)], dim=1)
        return z

    def gate(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Paper Eq. 8: branch weights ``beta`` of shape ``(B, 5)``, summing to 1."""
        if self.external_gate:
            raise RuntimeError("built with external_gate=True; pass gate=... to forward()")
        with fp32_math(x) as xf:
            z = self._summary(xf, cond)
            return torch.softmax(self.gate_fc2(torch.relu(self.gate_fc1(z))), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blend the five branches and add the result back through ``lambda_enh``."""
        self._reset_capture()

        if self.external_gate:
            if gate is None:
                raise ValueError("external_gate=True requires gate=(B, 5)")
            if gate.shape[-1] != NUM_BRANCHES:
                raise ValueError(f"gate must have {NUM_BRANCHES} columns, got {gate.shape}")
        elif gate is not None:
            raise ValueError("gate=... passed but this module owns its gate")

        with fp32_math(x) as xf:
            beta = gate.to(xf.dtype) if self.external_gate else self.gate(x, cond)  # (B, 5), Eq. 8

            # Per-sample, per-channel spatial statistics; keepdims for broadcasting.
            mu = xf.mean(dim=(2, 3), keepdim=True)
            sigma = xf.std(dim=(2, 3), keepdim=True, unbiased=False)
            z_score = (xf - mu) / (sigma + EPS)

            h = self.trunk_act(self.trunk(self._summary(xf, cond)))

            def per_channel(head: nn.Linear) -> torch.Tensor:
                return head(h)[:, :, None, None]  # (B, C) -> (B, C, 1, 1)

            # --- Eq. 3: tone --------------------------------------------------------
            # gamma in [1/3, 3] via 3^tanh(a); 1 at init.
            gamma = torch.pow(3.0, torch.tanh(per_channel(self.head_gamma)))
            normalised = torch.sigmoid(z_score)
            curved = torch.clamp(torch.pow(normalised, gamma), 1e-4, 1 - 1e-4)
            # logit(p) written as log(p) - log1p(-p): numerically steadier, and the form
            # the ONNX exporter handles without a dedicated logit op (gotcha 19).
            f_tone = mu + (sigma + EPS) * (torch.log(curved) - torch.log1p(-curved))

            # --- Eq. 4: contrast ----------------------------------------------------
            # alpha = softplus(a)/ln2 is > 0 and equals 1 at init; b is a small offset
            # scaled by sigma so it stays commensurate with the channel's own spread.
            alpha = nn.functional.softplus(per_channel(self.head_alpha)) / math.log(2.0)
            bias = 0.1 * torch.tanh(per_channel(self.head_bias)) * sigma
            f_contrast = alpha * z_score + mu + bias

            # --- Eq. 5: detail ------------------------------------------------------
            # Unsharp mask with a fixed Gaussian. kappa is one scalar per sample.
            kappa = 2.0 * torch.sigmoid(self.head_kappa(h))[:, :, None, None]
            blurred = depthwise_blur(xf, self.blur_kernel)
            f_detail = xf + kappa * (xf - blurred)

            # --- Eq. 6: white balance ----------------------------------------------
            scale = 1.0 + 0.5 * torch.tanh(per_channel(self.head_scale))  # [0.5, 1.5]
            f_wb = scale * xf

            # --- Eq. 7: identity ----------------------------------------------------
            f_identity = xf

            # --- Eq. 9: fusion ------------------------------------------------------
            branches = (f_tone, f_contrast, f_detail, f_wb, f_identity)
            f_enh = torch.zeros_like(xf)
            for i, branch in enumerate(branches):
                # beta[:, i, None, None, None] is (B, 1, 1, 1): broadcasts over C, H, W.
                f_enh = f_enh + beta[:, i, None, None, None] * branch

            out = xf + self.lambda_enh.float() * f_enh

        self._capture(
            gate=beta,
            gamma=gamma.mean(dim=(1, 2, 3)),
            alpha=alpha.mean(dim=(1, 2, 3)),
            kappa=kappa.reshape(kappa.shape[0], -1).mean(dim=1),
            scale=scale.mean(dim=(1, 2, 3)),
        )
        if self.collect_for_loss:
            self.routing_for_loss = beta
        return out.to(x.dtype)
