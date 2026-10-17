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

    ``train.py`` assigns :attr:`der_cfg` on the instance after construction and before
    ``train()``, since Ultralytics owns the constructor signature.

    It must **not** be stashed on ``self.args``: ``DetectionValidator`` is built with a
    copy of ``self.args``, and its ``__init__`` passes that copy back through ``get_cfg``,
    which runs ``check_dict_alignment`` and aborts on any key that is not a documented YOLO
    argument -- so a ``der_cfg`` attribute there kills the run at validator-construction
    time with "'der_cfg' is not a valid YOLO argument". A plain instance attribute with a
    class-level default sidesteps the config system entirely.
    """

    #: Resolved ``modules`` config, or ``None`` for the baseline.
    der_cfg: ModulesConfig | None = None

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
    def _class_head_prefix(model: YoloDERModel) -> str:
        """State-dict prefix of the Detect head's classification branch.

        Its channel width is a function of ``nc``, so these are the only keys that may
        legitimately fail to transfer from a differently-classed checkpoint.
        """
        head = model.model[-1]
        return f"model.{getattr(head, 'i', len(model.model) - 1)}.cv3."

    @staticmethod
    def _load_and_verify(model: YoloDERModel, weights: Any) -> None:
        """Load weights and fail loudly on a *meaningful* partial transfer (gotcha 3).

        Ultralytics' ``.load()`` prints "Transferred X/Y items" and never raises, so a
        silent shortfall would mean the modules trained from scratch and the GPU hours were
        wasted. But not every shortfall is a bug, and the distinction matters:

        * **Absent by name** -- the real failure. Injection renames ``model.4.cv1...`` to
          ``model.4.layer.cv1...``, so loading in the wrong order makes the backbone keys
          vanish. Always fatal.
        * **Present but shape-mismatched** -- expected exactly once. ``yolov8n.pt`` is
          COCO-pretrained with ``nc=80`` and ExDark has 12, so the Detect head's
          classification branch cannot transfer. Verified against Ultralytics 8.3.253:
          loading ``yolov8n.pt`` into an ``nc=12`` model leaves 0 keys absent by name and
          36 shape-mismatched, all under ``model.22.cv3.``. Fatal only outside that branch.
        """
        model.load(weights)
        incoming = YoloDERTrainer._state_dict_of(weights) or {}
        if not incoming:
            return

        own = model.state_dict()
        head = YoloDERTrainer._class_head_prefix(model)
        absent = [k for k in incoming if k not in own]
        mismatched = [k for k in incoming if k in own and own[k].shape != incoming[k].shape]
        unexpected = [k for k in mismatched if not k.startswith(head)]

        if absent or unexpected:
            detail = []
            if absent:
                detail.append(f"{len(absent)} absent by name (e.g. {absent[:3]})")
            if unexpected:
                detail.append(
                    f"{len(unexpected)} shape-mismatched outside the class head "
                    f"(e.g. {unexpected[:3]})"
                )
            raise RuntimeError(
                "partial weight transfer: "
                + "; ".join(detail)
                + f". {len(incoming)} incoming keys, {len(own)} in the model. "
                "Check the inject/load order (spec 7.4, gotcha 3)."
            )

        transferred = len(incoming) - len(mismatched)
        LOGGER.info(
            f"yolo_der: transferred {transferred}/{len(incoming)} pretrained keys; "
            f"{len(mismatched)} skipped in the nc-dependent class head ({head}*) as expected"
        )

    def _extend_loss_names(self) -> None:
        """Append ``ent_loss`` when an E1 regulariser adds a loss column.

        ``YoloDERModel.loss`` appends one element to both halves of Ultralytics'
        ``(loss, loss_items)`` tuple, and ``label_loss_items`` names the columns from
        ``loss_names``; a mismatch silently mislabels the progress table and results.csv.
        """
        model = self.model
        if getattr(model, "has_extra_loss", False) and "ent_loss" not in tuple(self.loss_names):
            self.loss_names = (*self.loss_names, "ent_loss")

    def set_model_attributes(self) -> None:
        """Extend ``loss_names`` as early as possible."""
        super().set_model_attributes()
        self._extend_loss_names()

    def get_validator(self) -> Any:
        """Build the validator, then re-extend ``loss_names``.

        This override is the one that actually matters. ``DetectionTrainer.get_validator``
        assigns ``self.loss_names = "box_loss", "cls_loss", "dfl_loss"`` unconditionally,
        and ``_setup_train`` calls it *after* ``set_model_attributes`` -- so extending the
        names there alone is silently undone, and the extra column ends up unnamed while
        the metrics shift one position left. Verified against Ultralytics 8.3.253:
        ``detect/train.py:165`` versus ``engine/trainer.py:268`` and ``:332``.
        """
        validator = super().get_validator()
        self._extend_loss_names()
        return validator

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
