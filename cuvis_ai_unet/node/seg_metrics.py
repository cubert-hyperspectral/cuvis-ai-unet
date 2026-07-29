"""Foreground segmentation metrics as a cuvis-ai METRIC node.

The builtin metric catalog only has ``AnomalyDetectionMetrics`` (binary anomaly); there is
no multiclass/foreground segmentation IoU/Dice metric node, so this fills that gap. It runs
at VAL/TEST only and emits foreground-class IoU, Dice and pixel accuracy the same way
``examples/lentils/_engine.py:evaluate`` computes them (fg = argmax >= 1), so a TensorBoard
``val`` curve is directly comparable to the reported champion numbers.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

_VAL_TEST = {ExecutionStage.VAL, ExecutionStage.TEST}


class SegMetrics(Node):
    """Foreground IoU / Dice / pixel-accuracy over per-pixel class logits."""

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.SEGMENTATION, NodeTag.EVALUATION})

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel class logits [B, H, W, num_classes]",
        ),
        "targets": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Integer per-pixel class targets [B, H, W]",
        ),
    }
    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="list[Metric]")}

    def __init__(self, execution_stages: set[ExecutionStage] | None = None, **kwargs: Any) -> None:
        name, execution_stages = self.consume_base_kwargs(kwargs, execution_stages or _VAL_TEST)
        super().__init__(name=name, execution_stages=execution_stages, **kwargs)
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated per-frame scores (call before a fresh eval pass)."""
        self._ious: list[float] = []
        self._dices: list[float] = []
        self._correct_px = 0.0
        self._total_px = 0.0

    @torch.no_grad()
    def forward(
        self, logits: Tensor, targets: Tensor, context: Context | None = None, **_: Any
    ) -> dict[str, Any]:
        """Foreground (class >= 1) IoU / Dice averaged over object frames + pixel accuracy.

        Returns the per-batch metrics (consumed by the TensorBoard sink; the trainer pools them
        per epoch). Also accumulates the per-frame scores so ``compute()`` can report the aggregate
        over a whole eval pass; the accumulators reset at ``batch_idx == 0`` so each val/test pass
        starts fresh.
        """
        batch_idx = getattr(context, "batch_idx", 0)
        if batch_idx == 0:
            self.reset()
        pred_fg = logits.argmax(dim=-1) >= 1  # [B, H, W]
        tgt_fg = targets >= 1
        ious, dices = [], []
        for b in range(pred_fg.shape[0]):
            p, t = pred_fg[b], tgt_fg[b]
            inter = float((p & t).sum())
            psum, gsum = float(p.sum()), float(t.sum())
            if gsum == 0:  # normal frame: no foreground to score (matches evaluate.py)
                continue
            union = psum + gsum - inter
            ious.append(inter / union if union else 0.0)
            dices.append(2 * inter / (psum + gsum) if (psum + gsum) else 0.0)
        # Accumulate for compute() (aggregate over the pass), keep per-batch values for TensorBoard.
        self._ious.extend(ious)
        self._dices.extend(dices)
        self._correct_px += float((pred_fg == tgt_fg).sum())
        self._total_px += float(pred_fg.numel())
        pixel_acc = float((pred_fg == tgt_fg).float().mean())
        stage = getattr(context, "stage", ExecutionStage.VAL)
        epoch = getattr(context, "epoch", 0)

        def _m(n: str, v: float) -> Metric:
            return Metric(name=n, value=float(v), stage=stage, epoch=epoch, batch_idx=batch_idx)

        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        return {
            "metrics": [
                _m("fg_iou", mean(ious)),
                _m("fg_dice", mean(dices)),
                _m("pixel_acc", pixel_acc),
            ]
        }

    def compute(self) -> dict[str, float]:
        """Aggregate fg-IoU / fg-Dice (mean over object frames) + pixel accuracy over the pass.

        Read this after a ``Predictor`` test pass (the gold-standard node-metric idiom); ``nan``
        for fg-IoU/Dice when the pass contained no object frames.
        """
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
        return {
            "fg_iou": mean(self._ious),
            "fg_dice": mean(self._dices),
            "pixel_acc": self._correct_px / self._total_px if self._total_px else float("nan"),
        }


__all__ = ["SegMetrics"]
