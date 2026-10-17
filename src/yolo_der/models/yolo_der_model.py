"""``YoloDERModel(DetectionModel)``: injection timing, severity probe, extra losses (spec 7.5).

Verified against Ultralytics 8.3.253 (hard rule 0.6):

* ``BaseModel.loss(batch, preds=None)`` returns ``self.criterion(preds, batch)``, and
  ``v8DetectionLoss.__call__`` ends with ``return loss * batch_size, loss.detach()`` where
  ``loss`` is the 3-vector ``(box, cls, dfl)``.
* ``BaseTrainer._do_train`` then does ``loss, self.loss_items = self.model(batch)`` followed
  by ``self.loss = loss.sum()``.

So appending a fourth element to each half of that tuple is exactly the right way to add a
loss term: the sum picks it up for backward, and ``loss_items`` gains a column that
``label_loss_items`` names from ``trainer.loss_names`` (which the trainer extends with
``"ent_loss"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from ultralytics.nn.tasks import DetectionModel

from yolo_der import losses
from yolo_der.models.inject import inject

if TYPE_CHECKING:  # pragma: no cover
    from yolo_der.config import ModulesConfig


class YoloDERModel(DetectionModel):
    """YOLOv8 detection model with DER / EnhanceNet inserted in the backbone.

    Args:
        cfg: model yaml (``"yolov8n.yaml"``) or a parsed dict, as for ``DetectionModel``.
        nc: number of classes.
        der_cfg: the resolved ``modules`` config. ``None`` leaves the model untouched,
            which is the baseline variant.
        imgsz: side length of the dummy forward used to measure channel counts.
        verbose: passed through to ``DetectionModel``.

    Injection is **not** done here when loading our own checkpoint -- see
    ``YoloDERTrainer.get_model`` and spec 7.4 for why the order matters. This class injects
    at construction, which is the "build then inject then load our weights" path.
    """

    def __init__(
        self,
        cfg: str | dict = "yolov8n.yaml",
        nc: int | None = None,
        der_cfg: ModulesConfig | None = None,
        imgsz: int = 448,
        verbose: bool = True,
        ch: int = 3,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.der_cfg = der_cfg
        self.severity_context = None
        self._entropy_weight = 0.0
        self._load_balance_weight = 0.0
        if der_cfg is not None and der_cfg.enabled:
            inject(self, der_cfg, imgsz=imgsz)
            self._entropy_weight = float(der_cfg.entropy_reg)
            self._load_balance_weight = float(der_cfg.load_balance)

    @property
    def has_extra_loss(self) -> bool:
        """True if an E1 regulariser is active, so the trainer must extend ``loss_names``."""
        return self._entropy_weight > 0.0 or self._load_balance_weight > 0.0

    def loss(self, batch: dict, preds: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Detection loss plus the routing regularisers (E1, paper Sec. 3.4 gap).

        ``L_enh`` from paper Eq. 12-13 is deliberately absent: it is a Smooth-L1 against
        "high-quality reference features", and ExDark has no clean/degraded image pairs, so
        the term is not computable. ``lambda_enh = 0`` and the deviation is documented
        rather than faked with an invented reference (spec 2.4).
        """
        loss, loss_items = super().loss(batch, preds)
        if not self.has_extra_loss:
            return loss, loss_items

        extra, _ = losses.regularisers(
            self,
            entropy_weight=self._entropy_weight,
            load_balance_weight=self._load_balance_weight,
        )
        # Scale the extra term the same way Ultralytics scales its own (by batch size), so
        # the relative weighting does not drift with batch size.
        batch_size = batch["img"].shape[0]
        scaled = (extra * batch_size).reshape(1).to(loss.dtype)
        return (
            torch.cat([loss, scaled]),
            torch.cat([loss_items, extra.detach().reshape(1).to(loss_items.dtype)]),
        )
