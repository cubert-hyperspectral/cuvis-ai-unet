"""Segmentation loss nodes: soft Dice and cross-entropy.

Both consume the ``DynUNet`` node's ``logits`` output (BHWC, last axis =
classes) plus an integer ``targets`` mask, and emit a scalar ``loss``. They
subclass a small local loss base so the plugin depends only on
``cuvis-ai-core`` (not the wider node library): category ``LOSS`` and execution
restricted to train/val/test (never inference).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_unet.functional import soft_dice_loss, to_bchw_targets

_LOSS_STAGES = {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}

_LOGITS_SPEC = PortSpec(
    dtype=torch.float32,
    shape=(-1, -1, -1, -1),
    description="Per-pixel class logits [B, H, W, num_classes]",
)
_TARGETS_SPEC = PortSpec(
    dtype=torch.int32,
    shape=(-1, -1, -1),
    description="Integer class-index mask [B, H, W] (cast to int64 internally; trailing [.,1] ok)",
)
_LOSS_OUT_SPEC = PortSpec(dtype=torch.float32, shape=(), description="Scalar loss value")


class _SegmentationLoss(Node):
    """Base for segmentation loss nodes: category LOSS, train/val/test only."""

    _category = NodeCategory.LOSS
    _tags = frozenset(
        {NodeTag.SEGMENTATION, NodeTag.TRAINING, NodeTag.DIFFERENTIABLE, NodeTag.TORCH}
    )

    def __init__(self, **kwargs: Any) -> None:
        assert "execution_stages" not in kwargs, "loss nodes fix their own execution stages"
        super().__init__(execution_stages=_LOSS_STAGES, **kwargs)


class DiceLoss(_SegmentationLoss):
    """Soft Dice loss over the DynUNet logits.

    Parameters
    ----------
    weight
        Scalar multiplier applied to the loss (for combining several losses).
    ignore_index
        Target value to exclude from the loss, or ``None`` to use every pixel.
    include_background
        Whether class 0 contributes to the per-class Dice mean (multiclass only).
    eps
        Smoothing constant for the Dice ratio.
    """

    INPUT_SPECS = {"logits": _LOGITS_SPEC, "targets": _TARGETS_SPEC}
    OUTPUT_SPECS = {"loss": _LOSS_OUT_SPEC}

    def __init__(
        self,
        weight: float = 1.0,
        ignore_index: int | None = None,
        include_background: bool = True,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.weight = float(weight)
        self.ignore_index = ignore_index
        self.include_background = bool(include_background)
        self.eps = float(eps)
        super().__init__(
            weight=self.weight,
            ignore_index=ignore_index,
            include_background=self.include_background,
            eps=self.eps,
            **kwargs,
        )

    def forward(self, logits: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute the weighted soft Dice loss from BHWC logits and a mask."""
        logits_bchw, targets_bhw = to_bchw_targets(logits, targets)
        loss = soft_dice_loss(
            logits_bchw,
            targets_bhw,
            ignore_index=self.ignore_index,
            include_background=self.include_background,
            eps=self.eps,
        )
        return {"loss": self.weight * loss}


class CrossEntropyLoss(_SegmentationLoss):
    """Pixel-wise cross-entropy over the DynUNet logits (multiclass, ``K >= 2``).

    Parameters
    ----------
    weight
        Scalar multiplier applied to the loss.
    class_weights
        Optional per-class rescaling passed to ``F.cross_entropy`` (helps with
        the strong class imbalance of small foreign objects).
    ignore_index
        Target value to exclude from the loss (default -100, PyTorch's default).
    """

    INPUT_SPECS = {"logits": _LOGITS_SPEC, "targets": _TARGETS_SPEC}
    OUTPUT_SPECS = {"loss": _LOSS_OUT_SPEC}

    def __init__(
        self,
        weight: float = 1.0,
        class_weights: list[float] | None = None,
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        self.weight = float(weight)
        self.class_weights = class_weights
        self.ignore_index = int(ignore_index)
        super().__init__(
            weight=self.weight,
            class_weights=class_weights,
            ignore_index=self.ignore_index,
            **kwargs,
        )
        if class_weights is not None:
            self.register_buffer("_class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute the weighted cross-entropy from BHWC logits and a mask."""
        logits_bchw, targets_bhw = to_bchw_targets(logits, targets)
        buf = getattr(self, "_class_weights", None)
        weights = buf.to(logits_bchw.dtype) if buf is not None else None
        loss = F.cross_entropy(
            logits_bchw, targets_bhw, weight=weights, ignore_index=self.ignore_index
        )
        return {"loss": self.weight * loss}


class OHEMCrossEntropyLoss(_SegmentationLoss):
    """Cross-entropy with online hard-example mining (OHEM) on the background.

    Keeps **all** foreground pixels (class index ``> 0``) plus the hardest
    background pixels — the top ``k = clamp(ratio * n_fg, min_kept, n_bg)`` by
    per-pixel loss — and averages cross-entropy over that kept set only. This
    forces the model to learn the normal-background pixels it currently over-fires
    on, instead of letting the overwhelming easy-background majority average the
    signal away. When a batch has no foreground the ``min_kept`` floor still mines
    the hardest ``min_kept`` background pixels; if nothing can be kept the loss
    falls back to the plain per-pixel mean.

    Parameters
    ----------
    weight
        Scalar multiplier applied to the loss (for combining several losses).
    ratio
        Number of hard background pixels to keep per foreground pixel.
    min_kept
        Lower bound on the number of background pixels kept, so batches with few
        or no foreground pixels still contribute a meaningful gradient.
    """

    INPUT_SPECS = {"logits": _LOGITS_SPEC, "targets": _TARGETS_SPEC}
    OUTPUT_SPECS = {"loss": _LOSS_OUT_SPEC}

    def __init__(
        self,
        weight: float = 1.0,
        ratio: float = 3.0,
        min_kept: int = 4096,
        **kwargs: Any,
    ) -> None:
        self.weight = float(weight)
        self.ratio = float(ratio)
        self.min_kept = int(min_kept)
        super().__init__(
            weight=self.weight,
            ratio=self.ratio,
            min_kept=self.min_kept,
            **kwargs,
        )

    def forward(self, logits: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute the weighted OHEM cross-entropy from BHWC logits and a mask."""
        logits_bchw, targets_bhw = to_bchw_targets(logits, targets)
        per_pixel = F.cross_entropy(logits_bchw, targets_bhw, reduction="none")
        fg = targets_bhw > 0
        n_fg = int(fg.sum())
        bg_losses = per_pixel[~fg]
        k = min(max(int(self.ratio * max(n_fg, 1)), self.min_kept), int(bg_losses.numel()))
        parts = []
        if n_fg > 0:
            parts.append(per_pixel[fg])
        if k > 0:
            parts.append(torch.topk(bg_losses, k).values)
        loss = torch.cat(parts).mean() if parts else per_pixel.mean()
        return {"loss": self.weight * loss}
