"""Wrap Ultralytics backbone layers with our modules (spec 7.4).

Verified against the installed Ultralytics 8.3.253 (hard rule 0.6). ``BaseModel._predict_once``
iterates ``self.model`` and relies on three attributes of each layer::

    if m.f != -1: x = y[m.f] if isinstance(m.f, int) else [...]   # routing
    x = m(x)
    y.append(x if m.i in self.save else None)                     # skip connections

so :class:`EnhancedLayer` must forward ``f``, ``i`` (and ``type``/``np``, used by
``model.info()``) from the layer it wraps. Replacement is **in place** at the same index,
so indices never shift and the neck's ``save`` list keeps working. For YOLOv8n at 448 px
the verified backbone is:

=====  =======  ==============  ========
Index  Layer    Out channels    Stride
=====  =======  ==============  ========
2      C2f      32              4
4      C2f      64              8  (P3)
6      C2f      128             16 (P4)
8      C2f      256             32
9      SPPF     256             32 (P5)
=====  =======  ==============  ========

Default ``insert_at = [4, 6, 9]`` -- P3, P4 and the last backbone block, which is as close
as YOLOv8 gets to the paper's "early C3k2, first A2C2F, after C2PSA". All three are in
``model.save`` ([4, 6, 9, 12, 15, 18, 21]), i.e. their outputs feed neck skip connections,
which is exactly why ``i`` must be preserved.

**State-dict keys change.** Wrapping renames ``model.4.cv1.conv.weight`` to
``model.4.layer.cv1.conv.weight``, so load order matters (spec 7.4, gotcha 3):

* fresh run from ``yolov8n.pt``: build -> ``model.load(pretrained)`` -> :func:`inject`;
* resume or load one of our checkpoints: build -> :func:`inject` -> ``model.load(ckpt)``.

:func:`has_injected_weights` is what lets ``get_model`` decide which order to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from yolo_der.models.der import DER
from yolo_der.models.enhancenet import NUM_BRANCHES, EnhanceNet
from yolo_der.models.severity_torch import s_dark

if TYPE_CHECKING:  # pragma: no cover
    from yolo_der.config import ModulesConfig

# Marker that appears in a state-dict key only once a model has been injected.
INJECTED_KEY_MARKER = ".layer."


@dataclass
class SeverityContext:
    """Carries the per-batch ``s_dark`` from the probe to every routing module.

    Shared by reference, written once per forward by :class:`SeverityProbe` and read by
    each :class:`EnhanceBlock`. A shared mutable holder is used rather than threading an
    extra argument through ``_predict_once``, because that method belongs to Ultralytics
    and we do not want to fork it.
    """

    cond: torch.Tensor | None = None


class SeverityProbe(nn.Module):
    """Wraps layer 0 to measure ``s_dark`` from the raw input batch (E3, spec 7.5).

    Sits at index 0 so it sees the network's actual input -- RGB in ``[0, 1]``. Computing
    it here rather than in the dataloader means the scalar is part of the traced graph and
    survives ONNX export.
    """

    def __init__(self, layer: nn.Module, context: SeverityContext) -> None:
        super().__init__()
        self.layer = layer
        self.context = context
        self.f, self.i = layer.f, layer.i
        self.type = "yolo_der.SeverityProbe"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record ``s_dark`` for this batch, then run the wrapped layer unchanged."""
        self.context.cond = s_dark(x.float()).to(x.dtype)  # (B, 1)
        return self.layer(x)


class EnhanceBlock(nn.Module):
    """Applies DER and/or EnhanceNet to one feature map, per ``modules.combine``.

    Two readings of the paper's Figures 1-3 are supported (spec 7.3):

    * ``sequential`` (default, and what the main ablation reports)::

          out = EnhanceNet(DER(F))

    * ``der_routes_enh`` (reported as a *variant*, not as "the paper's method"): DER is a
      router only, and its 5 weights become EnhanceNet's branch gate::

          beta = DER.route(F);  out = EnhanceNet(F, gate=beta)
    """

    def __init__(
        self,
        channels: int,
        cfg: ModulesConfig,
        context: SeverityContext | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.combine = cfg.combine
        self.context = context
        collect = cfg.entropy_reg > 0 or cfg.load_balance > 0
        cond_dim = cfg.cond_dim

        routes_enh = cfg.combine == "der_routes_enh"
        self.der = (
            DER(
                channels,
                num_paths=NUM_BRANCHES if routes_enh else cfg.der_paths,
                cond_dim=cond_dim,
                router_only=routes_enh,
                collect_for_loss=collect,
            )
            if cfg.der
            else None
        )
        self.enh = (
            EnhanceNet(
                channels,
                cond_dim=cond_dim,
                external_gate=routes_enh,
                collect_for_loss=collect and not routes_enh,
            )
            if cfg.enhancenet
            else None
        )
        if self.der is None and self.enh is None:
            raise ValueError("EnhanceBlock needs at least one of der / enhancenet enabled")

    def _cond(self) -> torch.Tensor | None:
        """Return the shared ``s_dark`` vector, if severity conditioning is on."""
        if self.context is None:
            return None
        if self.context.cond is None:
            raise RuntimeError(
                "severity_cond is on but no s_dark was recorded: is SeverityProbe installed "
                "at layer 0 and did the forward start there?"
            )
        return self.context.cond

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance ``x``, preserving its shape and dtype."""
        cond = self._cond()
        if self.combine == "der_routes_enh":
            assert self.der is not None and self.enh is not None  # enforced by ModulesConfig
            return self.enh(x, cond=cond, gate=self.der.route(x, cond))
        if self.der is not None:
            x = self.der(x, cond)
        if self.enh is not None:
            x = self.enh(x, cond)
        return x


class EnhancedLayer(nn.Module):
    """Wraps an Ultralytics layer: ``y = enhance(layer(x))``.

    Keeps Ultralytics' routing metadata (``f``, ``i``, ``type``, ``np``) so
    ``_predict_once`` and ``model.info()`` behave exactly as before.
    """

    def __init__(self, layer: nn.Module, enhance: nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.enhance = enhance
        self.f, self.i = layer.f, layer.i
        self.type = f"yolo_der.{type(enhance).__name__}"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped layer, then our enhancement."""
        return self.enhance(self.layer(x))


def find_channels(
    model: nn.Module, indices: tuple[int, ...] | list[int], imgsz: int = 448
) -> dict[int, int]:
    """Return ``{layer_index: out_channels}`` by running one dummy forward with hooks.

    Measured, never hard-coded: the channel table above is documentation, and a different
    model scale (s/m/l) or a changed yaml would silently invalidate it (spec 7.4).
    """
    wanted = set(indices)
    found: dict[int, int] = {}
    handles = []

    def make_hook(idx: int) -> Any:
        def hook(_m: nn.Module, _inp: Any, out: Any) -> None:
            if torch.is_tensor(out):
                found[idx] = out.shape[1]

        return hook

    layers = model.model
    for idx in sorted(wanted):
        if idx >= len(layers):
            raise IndexError(f"insert_at={idx} but the model has {len(layers)} layers")
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"no tensor output captured for layer(s) {sorted(missing)}")
    return found


def has_injected_weights(state: dict[str, Any]) -> bool:
    """True if ``state`` already contains :class:`EnhancedLayer`-renamed keys.

    ``get_model`` uses this to pick the load order (spec 7.4): our own checkpoints must be
    loaded *after* injection, pretrained YOLOv8n weights *before* it.
    """
    return any(INJECTED_KEY_MARKER in k for k in state)


def inject(model: nn.Module, cfg: ModulesConfig, imgsz: int = 448) -> nn.Module:
    """Replace ``model.model[i]`` with an :class:`EnhancedLayer` for each ``i`` in ``insert_at``.

    Mutates and returns ``model``. A no-op when neither module is enabled, so the baseline
    variant runs through exactly the same code path.
    """
    if not cfg.enabled:
        return model

    channels = find_channels(model, cfg.insert_at, imgsz=imgsz)

    context: SeverityContext | None = None
    if cfg.severity_cond:
        context = SeverityContext()
        # Index 0 sees the raw input batch; wrap it so s_dark is measured there.
        model.model[0] = SeverityProbe(model.model[0], context)

    for idx in cfg.insert_at:
        block = EnhanceBlock(channels[idx], cfg, context)
        model.model[idx] = EnhancedLayer(model.model[idx], block)

    # Keep the context reachable from the model, so analysis code and the loss can find it.
    model.severity_context = context
    return model


def added_parameter_report(model: nn.Module) -> dict[str, float]:
    """Return a params breakdown: total, added by our modules, and the added percentage.

    The paper claims the modules add < 10% of the detector's parameters. Phase 3's
    acceptance check prints this rather than taking the claim on trust.
    """
    total = sum(p.numel() for p in model.parameters())
    added = 0
    for m in model.modules():
        if isinstance(m, (DER, EnhanceNet)):
            added += sum(p.numel() for p in m.parameters())
    base = total - added
    return {
        "params_total": float(total),
        "params_base": float(base),
        "params_added": float(added),
        "params_added_pct": 100.0 * added / base if base else 0.0,
    }
