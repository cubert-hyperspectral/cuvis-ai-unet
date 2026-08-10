"""Anomaly-scoring adapter for segmentation logits.

`SegmentationAnomalyScore` turns a segmentation head's per-pixel class logits into
the inputs an anomaly-detection metric (e.g. an image/pixel AUROC node) expects:

* ``scores`` — a per-pixel foreground-probability map (softmax over classes, summed
  over the non-background classes ``>= 1``; for the binary case this is class 1).
* ``anomaly_score`` — a per-image scalar: the mean of the top ``top_frac`` fraction
  of score pixels (default 0.1 %). Averaging the most-anomalous pixels rather than
  taking a single max makes the image score robust to lone hot pixels while still
  responding to small foreground regions.
* ``targets_bool`` — a boolean foreground mask, emitted only when an integer
  ``targets`` mask is connected (so the same node is usable at inference, where no
  mask exists).

It is a pure, deterministic transform of the logits; it does not itself compute a
metric. Default execution stages are VAL/TEST/INFERENCE — the scoring is needed for
evaluation metrics and for inference-time thresholding, not during training.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

_SCORE_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST, ExecutionStage.INFERENCE}


class SegmentationAnomalyScore(Node):
    """Adapt segmentation logits to anomaly-detection score maps + a per-image score.

    Parameters
    ----------
    top_frac
        Fraction of highest-scoring pixels averaged into the per-image
        ``anomaly_score`` (default ``0.001`` = top 0.1 %). At least one pixel is
        always kept.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.SEGMENTATION, NodeTag.TORCH})

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel class logits [B, H, W, num_classes]",
        ),
        "targets": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Optional integer class mask [B, H, W]; when connected, "
            "emits targets_bool for a paired AUROC node",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Foreground-probability map [B, H, W, 1]",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per-image score: mean of the top-`top_frac` score pixels [B]",
        ),
        "targets_bool": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Boolean foreground mask [B, H, W, 1] (only when targets connected)",
            optional=True,
        ),
    }

    def __init__(self, top_frac: float = 0.001, **kwargs: Any) -> None:
        self.top_frac = float(top_frac)
        if not 0.0 < self.top_frac <= 1.0:
            raise ValueError(
                f"SegmentationAnomalyScore: top_frac must be in (0, 1], got {self.top_frac}."
            )
        assert "execution_stages" not in kwargs, (
            "SegmentationAnomalyScore fixes its own execution stages"
        )
        super().__init__(execution_stages=_SCORE_STAGES, top_frac=self.top_frac, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        logits: Tensor,
        targets: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        probs = torch.softmax(logits.float(), dim=-1)
        # Foreground probability = sum over non-background classes (binary: class 1).
        scores = probs[..., 1:].sum(dim=-1, keepdim=True)
        batch = scores.shape[0]
        flat = scores.reshape(batch, -1)
        # Mean of the top `top_frac` fraction of pixels, via the ascending-sort
        # quantile floor `k = clamp(int((1 - top_frac) * N), 0, N - 1)` and the mean
        # of `flat[k:]`. This is the integer-floor convention shared with
        # `cuvis_ai_rfdetr.functional.top_frac_mean`, so a segmentation and a
        # detection head scored side by side use the identical image-score rule
        # (e.g. exactly the top 400 pixels of a 987x405 map at top_frac=0.001).
        n = flat.shape[1]
        k = min(max(int((1.0 - self.top_frac) * n), 0), n - 1)
        sorted_scores, _ = torch.sort(flat, dim=1)
        anomaly_score = sorted_scores[:, k:].mean(dim=1)
        out: dict[str, Tensor] = {
            "scores": scores.contiguous(),
            "anomaly_score": anomaly_score.contiguous(),
        }
        if targets is not None:
            out["targets_bool"] = (targets >= 1).unsqueeze(-1).contiguous()
        return out
