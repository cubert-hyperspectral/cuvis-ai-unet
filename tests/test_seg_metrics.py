"""SegMetrics contract: fg-IoU / fg-Dice / pixel accuracy over per-pixel class logits.

Pure-tensor tests (no pipeline, no data): foreground scores average over object frames
only, pixel accuracy counts every frame, accumulators reset at ``batch_idx == 0``, and
``compute()`` aggregates a whole eval pass (NaN when the pass had no object frames).
"""

from __future__ import annotations

import math

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory
from cuvis_ai_schemas.execution import Context, Metric

from cuvis_ai_unet.node.seg_metrics import SegMetrics

H = W = 8


def _logits_from_mask(mask: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """One-hot logits [B, H, W, K] whose argmax reproduces ``mask`` exactly."""
    return torch.nn.functional.one_hot(mask.long(), num_classes).float() * 10.0


def _mask_with_block(rows: slice, cols: slice) -> torch.Tensor:
    """[1, H, W] int32 mask with a foreground block at ``[rows, cols]``."""
    m = torch.zeros(1, H, W, dtype=torch.int32)
    m[0, rows, cols] = 1
    return m


@pytest.mark.unit
def test_node_contract() -> None:
    node = SegMetrics(name="SegMetrics")
    assert node._category == NodeCategory.METRIC
    assert set(node.execution_stages) == {ExecutionStage.VAL, ExecutionStage.TEST}
    assert set(node.INPUT_SPECS) == {"logits", "targets"}
    assert set(node.OUTPUT_SPECS) == {"metrics"}


@pytest.mark.unit
def test_perfect_prediction_scores_one() -> None:
    node = SegMetrics(name="SegMetrics")
    targets = _mask_with_block(slice(2, 4), slice(2, 4))
    out = node.forward(_logits_from_mask(targets), targets, context=Context(stage=ExecutionStage.VAL))

    metrics = out["metrics"]
    assert all(isinstance(m, Metric) for m in metrics)
    by_name = {m.name: m.value for m in metrics}
    assert by_name["fg_iou"] == pytest.approx(1.0)
    assert by_name["fg_dice"] == pytest.approx(1.0)
    assert by_name["pixel_acc"] == pytest.approx(1.0)
    assert all(m.stage == ExecutionStage.VAL for m in metrics)


@pytest.mark.unit
def test_partial_overlap_iou_dice() -> None:
    # gt block 2x2 (4 px), prediction shifted to overlap 2 px: inter=2, union=6 -> IoU=1/3, Dice=1/2.
    node = SegMetrics(name="SegMetrics")
    targets = _mask_with_block(slice(2, 4), slice(2, 4))
    pred = _mask_with_block(slice(2, 4), slice(3, 5))
    out = node.forward(_logits_from_mask(pred), targets, context=Context(stage=ExecutionStage.VAL))

    by_name = {m.name: m.value for m in out["metrics"]}
    assert by_name["fg_iou"] == pytest.approx(1 / 3)
    assert by_name["fg_dice"] == pytest.approx(0.5)
    # 4 px false-negative-ish + 4 px predicted, 2 overlap -> 4 wrong pixels out of H*W
    assert by_name["pixel_acc"] == pytest.approx(1 - 4 / (H * W))


@pytest.mark.unit
def test_normal_frames_skip_fg_scores_but_count_pixels() -> None:
    # Batch of two frames: one all-background (perfectly predicted), one object frame with
    # partial overlap. fg-IoU/Dice average over the object frame only; pixel accuracy pools both.
    node = SegMetrics(name="SegMetrics")
    targets = torch.cat([_mask_with_block(slice(0, 0), slice(0, 0)),
                         _mask_with_block(slice(2, 4), slice(2, 4))])
    pred = torch.cat([_mask_with_block(slice(0, 0), slice(0, 0)),
                      _mask_with_block(slice(2, 4), slice(3, 5))])
    out = node.forward(_logits_from_mask(pred), targets, context=Context(stage=ExecutionStage.VAL))

    by_name = {m.name: m.value for m in out["metrics"]}
    assert by_name["fg_iou"] == pytest.approx(1 / 3)  # not diluted by the normal frame
    assert by_name["fg_dice"] == pytest.approx(0.5)
    assert by_name["pixel_acc"] == pytest.approx(1 - 4 / (2 * H * W))


@pytest.mark.unit
def test_compute_aggregates_pass_and_resets_at_batch_zero() -> None:
    node = SegMetrics(name="SegMetrics")
    targets = _mask_with_block(slice(2, 4), slice(2, 4))
    perfect = _logits_from_mask(targets)
    shifted = _logits_from_mask(_mask_with_block(slice(2, 4), slice(3, 5)))

    # One eval pass: batch 0 perfect, batch 1 partial -> aggregate = mean over both frames.
    node.forward(perfect, targets, context=Context(stage=ExecutionStage.VAL, batch_idx=0))
    node.forward(shifted, targets, context=Context(stage=ExecutionStage.VAL, batch_idx=1))
    agg = node.compute()
    assert agg["fg_iou"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert agg["fg_dice"] == pytest.approx((1.0 + 0.5) / 2)

    # A new pass starting at batch_idx == 0 drops the previous accumulators.
    node.forward(shifted, targets, context=Context(stage=ExecutionStage.VAL, batch_idx=0))
    agg = node.compute()
    assert agg["fg_iou"] == pytest.approx(1 / 3)
    assert agg["fg_dice"] == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_nan_without_object_frames() -> None:
    node = SegMetrics(name="SegMetrics")
    targets = torch.zeros(1, H, W, dtype=torch.int32)
    node.forward(_logits_from_mask(targets), targets, context=Context(stage=ExecutionStage.VAL))
    agg = node.compute()
    assert math.isnan(agg["fg_iou"]) and math.isnan(agg["fg_dice"])
    assert agg["pixel_acc"] == pytest.approx(1.0)
