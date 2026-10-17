"""``YoloDERTrainer(DetectionTrainer)``: model build, loss names, optimiser groups (spec 9.2).

Verified against Ultralytics 8.3.253 (hard rule 0.6):

* ``DetectionTrainer.get_model(self, cfg=None, weights=None, verbose=True)``.
* ``BaseTrainer.build_optimizer`` sorts parameters into three groups --
  ``g[2]`` anything whose name contains ``"bias"`` (no decay), ``g[1]`` norm-layer weights
  and ``logit_scale`` (no decay), ``g[0]`` everything else (**with** decay) -- and then
  constructs the optimiser on ``g[2]`` before ``add_param_group``-ing ``g[0]`` and ``g[1]``.
  So the final group order is ``[biases, decayed weights, norm weights]``.
* ``_do_train``'s warm-up gives ``param_groups[0]`` the special ``warmup_bias_lr``.

That grouping is a problem for us: ``lambda_der`` and ``lambda_enh`` contain neither
``"bias"`` nor a norm layer, so they land in the **decayed** group, and weight decay would
drag the ReZero scales back toward 0 -- fighting the exact mechanism that lets the modules
learn to contribute. :meth:`YoloDERTrainer.build_optimizer` moves them into the no-decay
norm-weight group (not group 0, which would also hand them the higher bias warm-up rate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import optim
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK

from yolo_der.models.inject import added_parameter_report, has_injected_weights, inject
from yolo_der.models.yolo_der_model import YoloDERModel

if TYPE_CHECKING:  # pragma: no cover
    from yolo_der.config import ModulesConfig

# Parameters that must never be weight-decayed, matched by state-dict suffix.
NO_DECAY_SUFFIXES = ("lambda_der", "lambda_enh")


class YoloDERTrainer(DetectionTrainer):
    """Ultralytics detection trainer that builds and trains a :class:`YoloDERModel`.

    The ``modules`` config travels on ``self.args.der_cfg``, set by ``train.py`` before the
    trainer is constructed, because Ultralytics owns the constructor signature.
    """

    @property
    def der_cfg(self) -> ModulesConfig | None:
        """The resolved ``modules`` config, or ``None`` for the baseline."""
        return getattr(self.args, "der_cfg", None)

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> YoloDERModel:
        """Build the model, honouring the load-order rule of spec 7.4 / gotcha 3.

        * **Fresh run from ``yolov8n.pt``** -- build, ``load(pretrained)``, *then* inject.
          Pretrained keys are named ``model.4.cv1...``; injecting first would rename them
          to ``model.4.layer.cv1...`` and the transfer would silently half-fail.
        * **Resume / our own checkpoint** -- build, inject, *then* load, because those
          weights already carry the renamed keys.

        :func:`has_injected_weights` decides which case we are in, so neither path depends
        on a flag the caller has to remember to set.
        """
        der_cfg = self.der_cfg
        imgsz = int(self.args.imgsz)

        model = YoloDERModel(
            cfg=cfg or "yolov8n.yaml",
            nc=self.data["nc"],
            der_cfg=None,  # injection is handled explicitly below, in the right order
            imgsz=imgsz,
            verbose=verbose and RANK == -1,
            ch=self.data.get("channels", 3),
        )
        model.der_cfg = der_cfg

        incoming = self._state_dict_of(weights)
        already_injected = incoming is not None and has_injected_weights(incoming)

        if already_injected:
            self._inject(model, der_cfg, imgsz)
            self._load_and_verify(model, weights)
        else:
            if weights is not None:
                self._load_and_verify(model, weights)
            self._inject(model, der_cfg, imgsz)

        if der_cfg is not None and der_cfg.enabled:
            report = added_parameter_report(model)
            LOGGER.info(
                f"yolo_der: base={report['params_base']:,.0f} "
                f"added={report['params_added']:,.0f} (+{report['params_added_pct']:.2f}%) "
                f"| paper claims < 10%"
            )
        return model

    @staticmethod
    def _state_dict_of(weights: Any) -> dict | None:
        """Return a state dict for ``weights``, whether it is a model or a dict."""
        if weights is None:
            return None
        if isinstance(weights, dict):
            return weights.get("model", weights)
        if hasattr(weights, "state_dict"):
            return weights.state_dict()
        return None

    def _inject(self, model: YoloDERModel, der_cfg: ModulesConfig | None, imgsz: int) -> None:
        """Insert the modules and wire up the extra-loss weights."""
        if der_cfg is None or not der_cfg.enabled:
            return
        inject(model, der_cfg, imgsz=imgsz)
        model._entropy_weight = float(der_cfg.entropy_reg)
        model._load_balance_weight = float(der_cfg.load_balance)

    @staticmethod
    def _load_and_verify(model: YoloDERModel, weights: Any) -> None:
        """Load weights and fail loudly on a partial transfer (gotcha 3)."""
        before = sum(1 for _ in model.state_dict())
        model.load(weights)
        # Ultralytics' .load() prints "Transferred X/Y items". It does not raise on a
        # partial transfer, so a silent 300/355 would mean the modules started from
        # scratch and the GPU hours are wasted. Re-check the overlap ourselves.
        incoming = YoloDERTrainer._state_dict_of(weights) or {}
        own = model.state_dict()
        matched = sum(1 for k, v in incoming.items() if k in own and own[k].shape == v.shape)
        if incoming and matched < len(incoming):
            raise RuntimeError(
                f"partial weight transfer: {matched}/{len(incoming)} incoming keys matched "
                f"({before} keys in the model). Check the inject/load order (spec 7.4)."
            )

    def set_model_attributes(self) -> None:
        """Extend ``loss_names`` when an E1 regulariser adds a loss column."""
        super().set_model_attributes()
        model = self.model
        if getattr(model, "has_extra_loss", False) and "ent_loss" not in self.loss_names:
            # YoloDERModel.loss appends one element to both halves of Ultralytics'
            # (loss, loss_items) tuple; label_loss_items names columns from loss_names.
            self.loss_names = (*self.loss_names, "ent_loss")

    def build_optimizer(
        self,
        model: torch.nn.Module,
        name: str = "auto",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ) -> optim.Optimizer:
        """Build Ultralytics' optimiser, then exempt our ReZero scales from weight decay."""
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)

        targets = {id(p) for n, p in model.named_parameters() if n.endswith(NO_DECAY_SUFFIXES)}
        if not targets:
            return optimizer

        # Pull them out of every decayed group.
        moved: list[torch.nn.Parameter] = []
        for group in optimizer.param_groups:
            if group.get("weight_decay", 0.0) == 0.0:
                continue
            keep = []
            for p in group["params"]:
                (moved if id(p) in targets else keep).append(p)
            group["params"] = keep

        if not moved:
            return optimizer

        # Prefer an existing zero-decay group that is NOT group 0: group 0 is the bias
        # group and _do_train gives it the higher warmup_bias_lr during warm-up.
        destination = next(
            (
                g
                for i, g in enumerate(optimizer.param_groups)
                if i != 0 and g.get("weight_decay", 0.0) == 0.0
            ),
            None,
        )
        if destination is None:
            optimizer.add_param_group({"params": moved, "weight_decay": 0.0})
        else:
            destination["params"].extend(moved)

        LOGGER.info(
            f"yolo_der: {len(moved)} ReZero scale(s) exempted from weight decay "
            f"({', '.join(NO_DECAY_SUFFIXES)})"
        )
        for i, g in enumerate(optimizer.param_groups):
            LOGGER.info(
                f"yolo_der: param_group[{i}] n={len(g['params'])} "
                f"weight_decay={g.get('weight_decay', 0.0)}"
            )
        return optimizer
