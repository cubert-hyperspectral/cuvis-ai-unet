"""Unit tests for the pure loss functions (no cuvis-ai imports needed)."""

import pytest
import torch

from cuvis_ai_unet.functional import soft_dice_loss, to_bchw_targets

pytestmark = pytest.mark.unit


def _perfect_logits(targets: torch.Tensor, k: int) -> torch.Tensor:
    """Logits that put all mass on the true class of ``targets``."""
    b, h, w = targets.shape
    logits = torch.full((b, k, h, w), -10.0)
    logits.scatter_(1, targets.unsqueeze(1), 10.0)
    return logits


def test_to_bchw_targets_shapes() -> None:
    logits = torch.randn(2, 8, 6, 4)  # BHWC
    for tgt in (torch.randint(0, 4, (2, 8, 6, 1)), torch.randint(0, 4, (2, 8, 6))):
        lb, tb = to_bchw_targets(logits, tgt)
        assert lb.shape == (2, 4, 8, 6)
        assert tb.shape == (2, 8, 6)
        assert tb.dtype == torch.int64


def test_dice_perfect_and_wrong() -> None:
    tgt = torch.randint(0, 4, (2, 10, 8))
    assert soft_dice_loss(_perfect_logits(tgt, 4), tgt).item() < 1e-3
    wrong = _perfect_logits((tgt + 1) % 4, 4)
    assert soft_dice_loss(wrong, tgt).item() > 0.99


def test_dice_gradient_flows() -> None:
    tgt = torch.randint(0, 3, (2, 10, 8))
    logits = torch.randn(2, 3, 10, 8, requires_grad=True)
    soft_dice_loss(logits, tgt).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_dice_ignore_index_excludes_pixels() -> None:
    tgt = torch.randint(0, 4, (2, 10, 8))
    logits = _perfect_logits(tgt, 4)
    # Corrupt predictions only in the region we will mark ignored.
    logits[:, :, :5, :] = torch.randn(2, 4, 5, 8) * 5
    tgt_ig = tgt.clone()
    tgt_ig[:, :5, :] = 255
    # With ignore, the corrupted (ignored) region must not inflate the loss.
    assert soft_dice_loss(logits, tgt_ig, ignore_index=255).item() < 1e-3


def test_dice_binary_k1() -> None:
    logits = torch.randn(2, 1, 10, 8, requires_grad=True)
    tgt = (torch.rand(2, 10, 8) > 0.5).long()
    loss = soft_dice_loss(logits, tgt)
    loss.backward()
    assert 0.0 <= loss.item() <= 1.5
    assert logits.grad is not None


def test_dice_exclude_background_runs() -> None:
    tgt = torch.randint(0, 4, (2, 10, 8))
    loss = soft_dice_loss(_perfect_logits(tgt, 4), tgt, include_background=False)
    assert loss.item() < 1e-3
