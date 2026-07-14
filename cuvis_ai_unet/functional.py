"""Pure-PyTorch loss functions used by the segmentation loss nodes.

Kept free of any cuvis-ai imports so it is unit-testable on its own and reused
by the node wrappers in ``node/losses.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def to_bchw_targets(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Convert BHWC logits to NCHW and squeeze a trailing target channel.

    ``logits`` ``[B, H, W, K]`` -> ``[B, K, H, W]``; ``targets`` ``[B, H, W, 1]``
    or ``[B, H, W]`` -> long ``[B, H, W]``.
    """
    logits = logits.permute(0, 3, 1, 2)
    if targets.dim() == 4 and targets.shape[-1] == 1:
        targets = targets[..., 0]
    return logits, targets.long()


def soft_dice_loss(
    logits_bchw: Tensor,
    targets_bhw: Tensor,
    *,
    ignore_index: int | None = None,
    include_background: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Soft Dice loss for binary (``K == 1``) or multiclass (``K > 1``) logits.

    ``logits_bchw`` is ``[B, K, H, W]`` and ``targets_bhw`` is an integer
    ``[B, H, W]`` label map. ``K == 1`` uses a sigmoid/binary formulation;
    ``K > 1`` uses softmax over one-hot targets. Pixels equal to ``ignore_index``
    are excluded from both the prediction and the target; when
    ``include_background`` is False, class 0 is dropped from the per-class mean
    (multiclass only). Returns ``1 - mean(dice)``.
    """
    num_classes = logits_bchw.shape[1]
    if num_classes == 1:
        probs = torch.sigmoid(logits_bchw)
        onehot = (targets_bhw == 1).unsqueeze(1).to(probs.dtype)
        cls = slice(0, 1)
    else:
        probs = torch.softmax(logits_bchw, dim=1)
        safe = targets_bhw
        if ignore_index is not None:
            safe = targets_bhw.masked_fill(targets_bhw == ignore_index, 0)
        onehot = F.one_hot(safe.clamp(0, num_classes - 1), num_classes)
        onehot = onehot.permute(0, 3, 1, 2).to(probs.dtype)
        cls = slice(0 if include_background else 1, num_classes)

    if ignore_index is not None:
        valid = (targets_bhw != ignore_index).unsqueeze(1).to(probs.dtype)
        probs = probs * valid
        onehot = onehot * valid

    p = probs[:, cls]
    g = onehot[:, cls]
    dims = (0, 2, 3)  # sum over batch and spatial dims -> one Dice per class
    inter = (p * g).sum(dims)
    denom = p.sum(dims) + g.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()
